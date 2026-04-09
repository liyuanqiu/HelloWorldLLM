# 🤖 Hello World LLM

A character-level neural language model built **from scratch** in Python.
PyTorch is used only as a GPU tensor library — all training logic
(forward pass, backpropagation, SGD, generation) is implemented manually.

Trained on the full text of 《西游记》(Journey to the West).

## Quick Start

```bash
cd src
uv sync
uv run python main.py
```

Requires an NVIDIA GPU with CUDA support.

## What is this?

A minimal but complete implementation of a neural language model — the
same fundamental concept behind GPT, Claude, and other large language models.

It demonstrates all the core building blocks:

| Concept | What it does | Where in code |
|---------|-------------|---------------|
| **Tokenization** | Converts text to numbers | `encode()` / `decode()` |
| **Embeddings** | Maps each character to a learned vector | `tok_emb`, `pos_emb` |
| **Forward Pass** | Predicts next character probabilities | `forward()` |
| **Backpropagation** | Computes gradients for learning | `backward()` |
| **SGD Optimizer** | Updates weights to reduce loss | `sgd_update()` |
| **Text Generation** | Produces new text character-by-character | `generate()` |

## Architecture

```
Input: "悟空道：你这泼猴，竟敢"  (16 characters)
          │
          ▼
   ┌──────────────┐
   │  Token Embed  │  Each char → 32-dim vector
   │  + Pos Embed  │  Each position → 32-dim vector
   └──────┬───────┘
          │  Flatten to 512-dim vector
          ▼
   ┌──────────────┐
   │ Hidden Layer  │  512 → 128 neurons, tanh activation
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Output Layer  │  128 → 4427 logits (one per unique character)
   └──────┬───────┘
          │  Softmax
          ▼
   P(next char) = [0.001, ..., 0.05, ..., 0.12, ...]
                   '一'        '如'       '在'
```

## How it relates to real LLMs

| This Model | Real LLMs (GPT-4, Claude) |
|-----------|---------------------------|
| ~780K parameters | Billions of parameters |
| Character-level tokens | Sub-word tokens (BPE) |
| 1 hidden layer | 96+ transformer layers |
| No attention | Multi-head self-attention |
| 16-char context | 128K+ token context |
| SGD optimizer | AdamW optimizer |
| Single GPU (RTX 4090) | GPU cluster training |

The key missing piece is **self-attention** — the mechanism that allows the model
to dynamically focus on relevant parts of the input. Adding attention is what
turns a simple neural network into a Transformer.

## Project Structure

```
├── src/
│   ├── main.py          # Model: forward, backward, training, generation
│   ├── pyproject.toml   # Python project config (managed by uv)
│   └── uv.lock          # Locked dependencies
├── data/
│   └── xiyouji.txt      # 西游记 full text (training corpus)
└── CLAUDE.md             # AI coding assistant instructions
```

## License

MIT
