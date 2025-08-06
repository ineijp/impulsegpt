# Impulse GPT
A personal PyTorch implementation of a tiny sized generative language model based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) with some changes. 

It has 2 implementations:

- In `impulegpt.py` is a re-implementation from scratch of scaled dot product attention and rotary positional encoding, without using existing modules like `torch.nn.MHA`. 

- In `impulsegpt_spda.py` is an implementation with a little more teeth, it uses the the PyTorch scaled dot product attention to utilize kernels like Flash Attention or Memory Efficient Attention. It also supports Grouped Query Attention.

For tokenizer it was trained with BPE tokenizer from [GPT-Neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m).


The model is trained with [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 1 epoch through over 2 millions rows of data.

The `trainer.ipynb` is a helper notebook to train the with mixed precision and test the model.

The `generate.ipynb` loads the model for inference.

With [uv](https://github.com/astral-sh/uv) installed, run `uv sync` in repo directory to install dependencies.


The trained weight is initialized with:

```python
import impulsegpt_sdpa
config = impulsegpt_sdpa.Config()
config.ctx_len = 128
config.n_layers = 12
config.d_model = 768
config.n_heads = 12
config.n_kv_heads = 4
config.vocab = 50000
config.gpa = True

```


## Generation examples

```
Once upon a time, there was a little girl named Mia. Mia liked to cut things. One day, she was out for a walk with her mom.
Mia saw a big tree. She wanted to cut it. She asked her mom for help. "Mom, can you cut the tree?" she asked. Her mom smiled and nodded.
Mia started to cut. It was a hard work. She did not give up. Her mom helped her. They did it together. Mia was very happy.
```

## TODO
- ~~A tokenizer other than a character-wise tokenizer is needed.~~
- ~~Grouped Query Attention for better performance/memory efficiency.~~