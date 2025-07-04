import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class Config:
    vocab: int = 50257
    d_model: int = 768
    ctx_len: int = 1024
    n_heads: int = 12
    n_layers: int = 12
    attn_drop: float = 0.1
    res_drop: float = 0.1
    batchFirst: bool = True
    dtype: torch.dtype = torch.float32
    

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024):
        """
        Rotary Positional Embedding.
        Args:
            dim: Dimension of the model (must be even)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        
        # Create position indices
        position = torch.arange(max_seq_len).unsqueeze(1)  # [seq_len, 1]
        
        # Create dimension indices
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )  # [dim/2]
        
        # Compute rotation matrices
        pos_emb = position * div_term  # [seq_len, dim/2]
        
        # Cache sin and cos values
        self.register_buffer('cos', torch.cos(pos_emb))  # [seq_len, dim/2]
        self.register_buffer('sin', torch.sin(pos_emb))  # [seq_len, dim/2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Tensor of shape (batch, seq_len, dim) with rotary embeddings applied
        """
        batch, seq_len, dim = x.shape
        x_reshape = x.view(batch, seq_len, -1, 2)  # [batch, seq_len, dim/2, 2]
        
        # Get pairs of consecutive features
        x1 = x_reshape[..., 0]  # [batch, seq_len, dim/2]
        x2 = x_reshape[..., 1]  # [batch, seq_len, dim/2]
        
        # Apply rotation using broadcasting
        cos = self.cos[:seq_len, :]  # [seq_len, dim/2]
        sin = self.sin[:seq_len, :]  # [seq_len, dim/2]
        
        # Rotate the features
        x1_rot = x1 * cos - x2 * sin  # [batch, seq_len, dim/2]
        x2_rot = x1 * sin + x2 * cos  # [batch, seq_len, dim/2]
        
        # Stack and reshape back
        output = torch.stack([x1_rot, x2_rot], dim=-1)  # [batch, seq_len, dim/2, 2]
        return output.view(batch, seq_len, dim)  # [batch, seq_len, dim]

class Layer(nn.Module):
    def __init__(self, d_model, n_heads, ctx_len, attn_dropout, res_dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = RoPE(dim=d_model, max_seq_len=ctx_len)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

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
        attn = self.attn_dropout(attn)
        # debugging print
        # if i == 0:
        #     print(f"Printing the attention map of layer {i+1}")
        #     plt.imshow(attn[0, 1].cpu().detach().numpy())
        #     print(attn[0, 0, 1])

        
        # the operations so far can be done with einsum in a much more succinct way i suppose
        
        out = attn @ v
        # print(f"Shape of out in layer {i+1} before permute: {out.shape}")
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.d_model)
        out = self.wo(out)
        out = out + self.res_dropout(x) # residual connection bypassing the attention
        out = F.layer_norm(out, out.shape[2:])
        
        out2 = self.ff(out)
        out2 = out2 + self.res_dropout(out) # residual connection bypassing the feedforward
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
        self.attn_drop = config.attn_drop
        self.res_drop = config.res_drop
        self.embedding = nn.Embedding(self.vocab, self.d_model)
        self.layers = nn.ModuleList([Layer(self.d_model, 
                                           self.n_heads, 
                                           self.ctx_len, 
                                           self.attn_drop, 
                                           self.res_drop) for _ in range(self.n_layers)])
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
    
    def generate(self, x, max_length, top_k=50, temp=1):
        # Ensure the model is in evaluation mode
        self.eval()

        # Initialize the output tensor with the input
        output = x

        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(output) / temp
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:,[-1]]] = -float('Inf')
                #print(logits)
                probs = F.softmax(logits, dim=-1)
                #print(f"The probs is: {probs}")
                next_token = torch.multinomial(probs, num_samples=1)
                #print(f"Token should be: {next_token}")
                output = torch.cat([output, next_token], dim=1)

        return output