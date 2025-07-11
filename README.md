# Impulse GPT
A personal PyTorch implementation of a tiny sized generative language model based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) with some changes. Is has a re-implementation of scaled dot product attention and rotary positional encoding, without using existing modules like `torch.nn.MHA`. For tokenizer it was trained with BPE tokenizer from [Bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese).

The model is trained with [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Training is still in progress.

## Installation
*WIP*

## Usage
*WIP*
Initialize the model by first initialize a config object and pass it to impulsegpt.
```python
import impulsegpt

config = impulsegpt.config()
config.d_model = 768
config.n_heads = 12
config.n_layers = 12

model = impulsegpt.impulsegpt(config)
```

## TODO
~~A tokenizer other than a character-wise tokenizer is needed.~~
Grouped Query Attention for better performance/memory efficiency.