# Impulse GPT
A personal project of a toy sized generative model based on transformer architectures.

Still work in progess.

## Installation
WIP

## Usage

Initialize the model by first initialize a config object and pass it to impulsegpt.
```python
import impulsegpt

config = impulsegpt.config()
config.d_model = 512
config.n_heads = 8
config.n_layers = 12

model = impulsegpt.impulsegpt(config)
```

## TODO
A tokenizer other than a character-wise tokenizer is needed.