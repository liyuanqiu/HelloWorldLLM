"""
Model definition and shared utilities for the character-level Transformer.

This module contains:
  - Hyperparameters
  - Tokenizer (encode/decode)
  - Model parameter creation and loading
  - Building blocks (layernorm, attention, ffn)
  - Forward pass
  - Save/load checkpoint
"""

import torch
import torch.nn.functional as F
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════════

BLOCK_SIZE  = 128
N_EMBD      = 128
N_HEAD      = 4
N_LAYER     = 6
DROPOUT     = 0.1

DEVICE = torch.device("cuda")
HEAD_DIM = N_EMBD // N_HEAD

# ═══════════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════════

DATA_PATH = Path(__file__).parent.parent / "data" / "xiyouji.txt"
MODEL_DIR = Path(__file__).parent.parent / "model"

text = DATA_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
VOCAB_SIZE = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# Pre-encoded training data on GPU
data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)

# ═══════════════════════════════════════════════════════════════════════
# Model Parameters
# ═══════════════════════════════════════════════════════════════════════

def _param(*shape, std=0.02):
    t = torch.empty(*shape, device=DEVICE)
    t.normal_(0, std)
    t.requires_grad = True
    return t

def _zeros(*shape):
    return torch.zeros(*shape, device=DEVICE, requires_grad=True)

def _ones(*shape):
    return torch.ones(*shape, device=DEVICE, requires_grad=True)

# Parameter name → tensor mapping (ordered dict style using lists)
PARAM_NAMES: list[str] = []
PARAM_MAP: dict[str, torch.Tensor] = {}

def _register(name: str, tensor: torch.Tensor):
    PARAM_NAMES.append(name)
    PARAM_MAP[name] = tensor

def create_params():
    """Initialize all model parameters from scratch."""
    PARAM_NAMES.clear()
    PARAM_MAP.clear()

    _register("tok_emb", _param(VOCAB_SIZE, N_EMBD))
    _register("pos_emb", _param(BLOCK_SIZE, N_EMBD))

    for i in range(N_LAYER):
        _register(f"ln1_g.{i}", _ones(N_EMBD))
        _register(f"ln1_b.{i}", _zeros(N_EMBD))
        _register(f"Wqkv.{i}", _param(N_EMBD, 3 * N_EMBD))
        _register(f"Wo.{i}", _param(N_EMBD, N_EMBD, std=0.02 / (2 * N_LAYER) ** 0.5))
        _register(f"ln2_g.{i}", _ones(N_EMBD))
        _register(f"ln2_b.{i}", _zeros(N_EMBD))
        _register(f"Wff1.{i}", _param(N_EMBD, 4 * N_EMBD))
        _register(f"bff1.{i}", _zeros(4 * N_EMBD))
        _register(f"Wff2.{i}", _param(4 * N_EMBD, N_EMBD, std=0.02 / (2 * N_LAYER) ** 0.5))
        _register(f"bff2.{i}", _zeros(N_EMBD))

    _register("ln_f_g", _ones(N_EMBD))
    _register("ln_f_b", _zeros(N_EMBD))
    _register("Wout", _param(N_EMBD, VOCAB_SIZE))

def params() -> list[torch.Tensor]:
    return [PARAM_MAP[n] for n in PARAM_NAMES]

def p(name: str) -> torch.Tensor:
    return PARAM_MAP[name]

# ═══════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(path: Path | None = None):
    path = path or MODEL_DIR / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: PARAM_MAP[name].detach().cpu() for name in PARAM_NAMES}
    state["__meta__"] = dict(
        vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_embd=N_EMBD,
        n_head=N_HEAD, n_layer=N_LAYER, chars="".join(chars),
    )
    torch.save(state, path)
    print(f"💾 Model saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")

def load_checkpoint(path: Path | None = None):
    path = path or MODEL_DIR / "checkpoint.pt"
    state = torch.load(path, map_location=DEVICE, weights_only=True)

    PARAM_NAMES.clear()
    PARAM_MAP.clear()
    for name, tensor in state.items():
        if name == "__meta__":
            continue
        t = tensor.to(DEVICE)
        t.requires_grad = False  # no grad needed for inference
        PARAM_NAMES.append(name)
        PARAM_MAP[name] = t

    print(f"📂 Model loaded from {path}")

# ═══════════════════════════════════════════════════════════════════════
# Causal mask
# ═══════════════════════════════════════════════════════════════════════

causal_mask = torch.triu(torch.full((BLOCK_SIZE, BLOCK_SIZE), float('-inf'), device=DEVICE), diagonal=1)

# ═══════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════

def layernorm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    return g * (x - mean) / (var + 1e-5).sqrt() + b

def attention(x: torch.Tensor, wqkv: torch.Tensor, wo: torch.Tensor, training: bool) -> torch.Tensor:
    B, T, C = x.shape
    qkv = x @ wqkv
    q, k, v = qkv.split(N_EMBD, dim=-1)
    q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
    k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
    v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

    attn = (q @ k.transpose(-2, -1)) / HEAD_DIM ** 0.5
    attn = attn + causal_mask[:T, :T]
    attn = torch.softmax(attn, dim=-1)
    if training:
        attn = F.dropout(attn, p=DROPOUT)

    out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
    out = out @ wo
    if training:
        out = F.dropout(out, p=DROPOUT)
    return out

def ffn(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
        w2: torch.Tensor, b2: torch.Tensor, training: bool) -> torch.Tensor:
    h = torch.relu(x @ w1 + b1)
    if training:
        h = F.dropout(h, p=DROPOUT)
    out = h @ w2 + b2
    if training:
        out = F.dropout(out, p=DROPOUT)
    return out

# ═══════════════════════════════════════════════════════════════════════
# Forward Pass
# ═══════════════════════════════════════════════════════════════════════

def transformer_forward(inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Run the transformer and return logits (B, T, VOCAB_SIZE)."""
    B, T = inputs.shape
    x = F.embedding(inputs, p("tok_emb")) + p("pos_emb")[:T]

    for i in range(N_LAYER):
        x = x + attention(layernorm(x, p(f"ln1_g.{i}"), p(f"ln1_b.{i}")),
                          p(f"Wqkv.{i}"), p(f"Wo.{i}"), training)
        x = x + ffn(layernorm(x, p(f"ln2_g.{i}"), p(f"ln2_b.{i}")),
                     p(f"Wff1.{i}"), p(f"bff1.{i}"), p(f"Wff2.{i}"), p(f"bff2.{i}"), training)

    x = layernorm(x, p("ln_f_g"), p("ln_f_b"))
    return x @ p("Wout")
