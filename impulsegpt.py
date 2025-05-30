import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab: int = 50257
    d_model: int = 768
    ctx_len: int = 1024
    n_heads: int = 12
    n_layers: int = 12
    batchFirst: bool = True
    dtype: torch.dtype = torch.float32

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Precompute sinusoidal embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("sinusoidal", torch.einsum("i,j->ij", torch.arange(max_seq_len).float(), inv_freq))
        self.register_buffer("sin", torch.sin(self.sinusoidal))
        self.register_buffer("cos", torch.cos(self.sinusoidal))

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (length, batch, d_model).

        Returns:
            A tensor of shape (length, batch, d_model) with rotary positional embeddings applied.
        """
        length, batch, d_model = x.shape
        assert d_model == self.d_model, "Input d_model must match initialized d_model"

        # Apply rotary embeddings
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd dimensions
        x_rotated = torch.cat([x1 * self.cos[:length, None, :] - x2 * self.sin[:length, None, :],
                               x1 * self.sin[:length, None, :] + x2 * self.cos[:length, None, :]], dim=-1)
        return x_rotated
    

class Layer(nn.Module):
    def __init__(self, d_model, n_heads, ctx_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len = ctx_len)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        #self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, i):
        # rope only applies to q and k, not v

        seq_len = x.shape[1]
        q = self.wq(x)
        q = self.rope(q)
        k = self.wk(x)
        k = self.rope(k)
        v = self.wv(x)

        # Assume the input is of shape (batch, length, d_model)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim)
        # the Q, K, V tensors are now of shape (batch, length, n_heads, head_dim)


        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # now they are of shape (batch, n_heads, length, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn is of shape (batch, n_heads, length, length)
        # apply the mask to the attention scores before softmax
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # debugging print
        # if i == 0:
        #     print(f"Printing the attention map of layer {i+1}")
        #     plt.imshow(attn[0, 1].cpu().detach().numpy())
        #     print(attn[0, 0, 1])

        
        # the operations so far can be done with einsum in a much more succinct way i suppose
        
        out = attn @ v
        # print(f"Shape of out in layer {i+1} before permute: {out.shape}")
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.d_model)
        # out = self.wo(out)
        out = out + x # residual connection bypassing the attention
        out = F.layer_norm(out, out.shape[2:])
        
        out2 = self.ff(out)
        out2 = out2 + out # residual connection bypassing the feedforward
        out2 = F.layer_norm(out2, out2.shape[2:])
        return out2
        
class ImpulseGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.batchFirst = config.batchFirst
        self.vocab = config.vocab
        self.d_model = config.d_model
        self.ctx_len = config.ctx_len
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.embedding = nn.Embedding(self.vocab, self.d_model)
        self.layers = nn.ModuleList([Layer(self.d_model, self.n_heads, self.ctx_len) for _ in range(self.n_layers)])
        self.fc = nn.Linear(self.d_model, self.vocab)
 
    def forward(self, x):
        x = self.embedding(x)

        # the default input shape is expected to be (batch, length, d_model)
        # permute from (length, batch, d_model) to (batch, length, d_model) if batch_first is False
        if not self.batchFirst:
            x = x.permute(1, 0, 2)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, i)
        x = self.fc(x[:,-1,:])
        return x
    
    def generate(self, x, max_length):
        # Ensure the model is in evaluation mode
        self.eval()

        # Initialize the output tensor with the input
        output = x

        # Generate tokens one by one
        for _ in range(max_length):
            logits = self.forward(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

        return output