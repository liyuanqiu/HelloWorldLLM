# 🤖 Hello World LLM

A character-level neural language model built **entirely from scratch** in TypeScript.
No ML frameworks — just math.

## Quick Start

```bash
npm install
npm start
```

## What is this?

This is a minimal but complete implementation of a neural language model — the
same fundamental concept behind GPT, Claude, and other large language models.

It demonstrates all the core building blocks:

| Concept | What it does | Where in code |
|---------|-------------|---------------|
| **Tokenization** | Converts text to numbers | `encode()` / `decode()` |
| **Embeddings** | Maps each character to a learned vector | `tokEmb`, `posEmb` |
| **Forward Pass** | Predicts next character probabilities | `forward()` |
| **Backpropagation** | Computes gradients for learning | `backward()` |
| **SGD Optimizer** | Updates weights to reduce loss | `sgdUpdate()` |
| **Text Generation** | Produces new text character-by-character | `generate()` |

## Architecture

```
Input: "To be, or not to"  (16 characters)
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
   │ Output Layer  │  128 → vocab_size logits
   └──────┬───────┘
          │  Softmax
          ▼
   P(next char) = [0.01, 0.02, ..., 0.15, ...]
                   'a'    'b'        'n'
```

## How it relates to real LLMs

This model is a simplified version of what GPT and other LLMs do:

| This Model | Real LLMs (GPT-4, Claude) |
|-----------|---------------------------|
| ~44K parameters | Billions of parameters |
| Character-level tokens | Sub-word tokens (BPE) |
| 1 hidden layer | 96+ transformer layers |
| No attention | Multi-head self-attention |
| 16-char context | 128K+ token context |
| SGD optimizer | AdamW optimizer |
| CPU training | GPU cluster training |

The key missing piece is **self-attention** — the mechanism that allows the model
to dynamically focus on relevant parts of the input. Adding attention is what
turns a simple neural network into a Transformer.

## 🎮 GPU Acceleration (RTX 4090)

At this scale (~44K params), CPU is sufficient. To leverage your RTX 4090 for
larger models, the matrix multiplications in `forward()` and `backward()` are
the operations to offload. Options:

1. **WebGPU** — Write WGSL compute shaders for matmul (modern, cross-platform)
2. **gpu.js** — `npm install gpu.js` — translates JS to GPU compute shaders
3. **CUDA via Node-API** — Write C++ CUDA kernels, call from Node.js (maximum performance)

## Tutorial

- [English Tutorial](docs/tutorial.md) — Learn how every component works, line by line
- [中文教程](docs/tutorial.cn.md) — 逐行拆解每个组件的工作原理

## License

MIT
