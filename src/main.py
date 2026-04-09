"""
🤖 Hello World LLM — Character-Level Transformer

Usage:
  uv run python train.py       # Train and save model
  uv run python generate.py    # Generate text from saved model
  uv run python main.py        # Train + generate (all-in-one)

See model.py for the Transformer architecture.
"""

from train import train
from generate import generate

def main():
    train()

    print('\n✨ Generated text (the model continues from "悟空"):')
    print("───────────────────────────────────────────────────────────")
    print(generate("悟空", 500, 0.8))
    print("───────────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
"""
🤖 Hello World LLM — Character-Level Transformer (GPU)

A mini GPT built from scratch in Python with PyTorch tensors.

Architecture (same as GPT):
  Token Embedding + Position Embedding
  → N × [LayerNorm → Multi-Head Self-Attention → Residual
          → LayerNorm → Feed-Forward Network → Residual]
  → LayerNorm → Output (softmax)

Forward pass: fully hand-written — every matmul, attention, layernorm visible.
Backward pass: PyTorch autograd (the math is identical to manual backprop,
  just automated — we've already demonstrated manual backprop in earlier versions).
Optimizer: hand-written Adam.
"""

import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════════

BLOCK_SIZE  = 128      # Context window (how many chars the model sees)
N_EMBD      = 128      # Embedding dimension
N_HEAD      = 4        # Number of attention heads
N_LAYER     = 6        # Number of transformer blocks
BATCH_SIZE  = 128      # Batch size (attention is O(T²) — memory-heavier)
DROPOUT     = 0.1      # Dropout rate (prevents overfitting)
LR_MAX      = 3e-4     # Adam peak learning rate
LR_MIN      = 1e-5     # Final learning rate
NUM_STEPS   = 8000     # Training iterations
TEMPERATURE = 0.8      # Generation temperature
GEN_LENGTH  = 500      # Characters to generate

DEVICE = torch.device("cuda")
HEAD_DIM = N_EMBD // N_HEAD

# ═══════════════════════════════════════════════════════════════════════
# Training Data (西游记)
# ═══════════════════════════════════════════════════════════════════════

text = (Path(__file__).parent.parent / "data" / "xiyouji.txt").read_text(encoding="utf-8")

chars = sorted(set(text))
VOCAB_SIZE = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# Model Parameters
#
# All parameters have requires_grad=True so autograd can compute
# gradients through the hand-written forward pass.
# ═══════════════════════════════════════════════════════════════════════

def param(*shape, std=0.02):
    """Create a GPU parameter tensor with normal initialization."""
    t = torch.empty(*shape, device=DEVICE)
    t.normal_(0, std)
    t.requires_grad = True
    return t

def zeros(*shape):
    """Create a GPU parameter tensor initialized to zeros."""
    return torch.zeros(*shape, device=DEVICE, requires_grad=True)

def ones(*shape):
    """Create a GPU parameter tensor initialized to ones."""
    return torch.ones(*shape, device=DEVICE, requires_grad=True)

# Token + position embeddings
tok_emb = param(VOCAB_SIZE, N_EMBD)
pos_emb = param(BLOCK_SIZE, N_EMBD)

# Transformer blocks: each block has attention + FFN + 2 layernorms
# Stored as flat lists for simplicity
ln1_g, ln1_b = [], []   # LayerNorm before attention
Wqkv, Wo = [], []       # Attention weights
ln2_g, ln2_b = [], []   # LayerNorm before FFN
Wff1, bff1 = [], []     # FFN first layer (expand)
Wff2, bff2 = [], []     # FFN second layer (contract)

for _ in range(N_LAYER):
    ln1_g.append(ones(N_EMBD));  ln1_b.append(zeros(N_EMBD))
    Wqkv.append(param(N_EMBD, 3 * N_EMBD))
    Wo.append(param(N_EMBD, N_EMBD, std=0.02 / (2 * N_LAYER) ** 0.5))
    ln2_g.append(ones(N_EMBD));  ln2_b.append(zeros(N_EMBD))
    Wff1.append(param(N_EMBD, 4 * N_EMBD)); bff1.append(zeros(4 * N_EMBD))
    Wff2.append(param(4 * N_EMBD, N_EMBD, std=0.02 / (2 * N_LAYER) ** 0.5))
    bff2.append(zeros(N_EMBD))

# Final layer norm + output projection
ln_f_g = ones(N_EMBD);  ln_f_b = zeros(N_EMBD)
Wout = param(N_EMBD, VOCAB_SIZE)

params = [tok_emb, pos_emb]
for i in range(N_LAYER):
    params += [ln1_g[i], ln1_b[i], Wqkv[i], Wo[i],
               ln2_g[i], ln2_b[i], Wff1[i], bff1[i], Wff2[i], bff2[i]]
params += [ln_f_g, ln_f_b, Wout]

total_params = sum(p.numel() for p in params)

# Causal mask — prevents attention to future positions (pre-computed once)
causal_mask = torch.triu(torch.full((BLOCK_SIZE, BLOCK_SIZE), float('-inf'), device=DEVICE), diagonal=1)

# ═══════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════

def layernorm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layer normalization: normalize, then scale and shift."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    return g * (x - mean) / (var + 1e-5).sqrt() + b

def attention(x: torch.Tensor, wqkv: torch.Tensor, wo: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Multi-head causal self-attention."""
    B, T, C = x.shape

    # Project to Q, K, V in one matmul, then split
    qkv = x @ wqkv                                  # (B, T, 3*C)
    q, k, v = qkv.split(N_EMBD, dim=-1)             # 3 × (B, T, C)

    # Reshape into heads: (B, N_HEAD, T, HEAD_DIM)
    q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
    k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
    v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

    # Attention scores: Q·K^T / √d + causal mask
    attn = (q @ k.transpose(-2, -1)) / HEAD_DIM ** 0.5  # (B, N_HEAD, T, T)
    attn = attn + causal_mask[:T, :T]                     # mask future
    attn = torch.softmax(attn, dim=-1)                     # normalize
    if training:
        attn = F.dropout(attn, p=DROPOUT)                  # attention dropout

    # Weighted sum of values
    out = attn @ v                                          # (B, N_HEAD, T, HEAD_DIM)
    out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)

    out = out @ wo                                         # output projection
    if training:
        out = F.dropout(out, p=DROPOUT)                    # residual dropout
    return out

def ffn(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
        w2: torch.Tensor, b2: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Feed-forward network with ReLU activation."""
    h = torch.relu(x @ w1 + b1)
    if training:
        h = F.dropout(h, p=DROPOUT)
    out = h @ w2 + b2
    if training:
        out = F.dropout(out, p=DROPOUT)
    return out

# ═══════════════════════════════════════════════════════════════════════
# Forward Pass — a complete GPT forward pass, every operation visible
# ═══════════════════════════════════════════════════════════════════════

def forward(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T = inputs.shape

    # Token embedding + position embedding
    x = F.embedding(inputs, tok_emb) + pos_emb[:T]    # (B, T, N_EMBD)

    # N transformer blocks
    for i in range(N_LAYER):
        # Pre-norm attention with residual
        x = x + attention(layernorm(x, ln1_g[i], ln1_b[i]), Wqkv[i], Wo[i])
        # Pre-norm FFN with residual
        x = x + ffn(layernorm(x, ln2_g[i], ln2_b[i]), Wff1[i], bff1[i], Wff2[i], bff2[i])

    # Final layernorm → project to vocab (weight-tied with tok_emb)
    x = layernorm(x, ln_f_g, ln_f_b)                  # (B, T, N_EMBD)
    logits = x @ Wout                                 # (B, T, VOCAB_SIZE)

    # Cross-entropy loss (only on last position would waste data — use all positions)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    return loss

# ═══════════════════════════════════════════════════════════════════════
# Adam Optimizer (hand-written)
# ═══════════════════════════════════════════════════════════════════════

beta1, beta2, eps = 0.9, 0.999, 1e-8
m = [torch.zeros_like(p) for p in params]   # first moment
v = [torch.zeros_like(p) for p in params]   # second moment

def adam_step(step: int, lr: float):
    """One step of Adam with bias correction."""
    t = step + 1
    for i, p in enumerate(params):
        if p.grad is None:
            continue
        g = p.grad
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        p.data -= lr * m_hat / (v_hat.sqrt() + eps)
        p.grad = None

# ═══════════════════════════════════════════════════════════════════════
# Batch Sampling (GPU-vectorized — no Python loops)
# ═══════════════════════════════════════════════════════════════════════

# For transformer: targets are shifted inputs (predict next char at EVERY position)
_offsets = torch.arange(BLOCK_SIZE + 1, device=DEVICE).unsqueeze(0)
_max_start = len(data) - BLOCK_SIZE - 1

def sample_batch():
    ix = torch.randint(_max_start, (BATCH_SIZE, 1), device=DEVICE)
    window = data[ix + _offsets]             # (B, BLOCK_SIZE+1)
    return window[:, :-1], window[:, 1:]     # inputs (B,T), targets (B,T) — shifted by 1

# ═══════════════════════════════════════════════════════════════════════
# Text Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(prompt: str, length: int, temperature: float) -> str:
    context = encode(prompt)
    pad_idx = stoi.get(" ", 0)
    while len(context) < BLOCK_SIZE:
        context = [pad_idx] + context
    ctx = torch.tensor(context[-BLOCK_SIZE:], dtype=torch.long, device=DEVICE).unsqueeze(0)

    result = list(prompt)
    for _ in range(length):
        x = F.embedding(ctx, tok_emb) + pos_emb[:ctx.shape[1]]
        for i in range(N_LAYER):
            x = x + attention(layernorm(x, ln1_g[i], ln1_b[i]), Wqkv[i], Wo[i], training=False)
            x = x + ffn(layernorm(x, ln2_g[i], ln2_b[i]), Wff1[i], bff1[i], Wff2[i], bff2[i], training=False)
        x = layernorm(x, ln_f_g, ln_f_b)
        logits = (x[:, -1, :] @ Wout).squeeze(0) / temperature

        # Top-k sampling: keep only top 40 candidates
        top_k = 40
        topk_vals, topk_idx = logits.topk(top_k)
        probs = torch.softmax(topk_vals, dim=-1)
        next_idx = topk_idx[torch.multinomial(probs, 1)].item()
        result.append(itos[next_idx])

        next_tok = torch.tensor([[next_idx]], device=DEVICE)
        ctx = torch.cat([ctx[:, 1:], next_tok], dim=1)

    return "".join(result)

# ═══════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("🤖 Hello World LLM — Character-Level Transformer (GPU)")
    print("═══════════════════════════════════════════════════════════")
    print(f"  Vocabulary : {VOCAB_SIZE} unique characters")
    print(f"  Context    : {BLOCK_SIZE} characters")
    print(f"  Embedding  : {N_EMBD}-dim, {N_HEAD} heads")
    print(f"  Layers     : {N_LAYER} transformer blocks")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Parameters : {total_params:,}")
    print(f"  Data       : {len(data):,} characters of 西游记")
    print(f"  Device     : {torch.cuda.get_device_name(0)}")
    print()

    print("📚 Training...")
    t0 = time.time()

    for step in range(NUM_STEPS):
        # Cosine LR with warmup
        warmup = 200
        if step < warmup:
            lr = LR_MAX * (step / warmup)
        else:
            decay = (step - warmup) / (NUM_STEPS - warmup)
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * decay))

        inputs, targets = sample_batch()
        loss = forward(inputs, targets)
        loss.backward()

        # Gradient clipping
        with torch.no_grad():
            norm_sq = sum((p.grad * p.grad).sum() for p in params if p.grad is not None)
            norm = norm_sq.sqrt()
            if norm > 1.0:
                for p in params:
                    if p.grad is not None:
                        p.grad *= 1.0 / norm

        adam_step(step, lr)

        if step % 500 == 0 or step == NUM_STEPS - 1:
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f"  Step {step:5d} │ Loss: {loss.item():.4f} │ LR: {lr:.6f} │ {elapsed:.1f}s")

    train_time = time.time() - t0
    print(f"\n✅ Training complete in {train_time:.1f}s")

    print('\n✨ Generated text (the model continues from "悟空"):')
    print("───────────────────────────────────────────────────────────")
    print(generate("悟空", GEN_LENGTH, TEMPERATURE))
    print("───────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
