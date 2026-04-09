"""Train the character-level Transformer on 西游记."""

import math
import time
import torch

from model import (
    DEVICE, BLOCK_SIZE, VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER,
    text, encode, data, create_params, params,
    transformer_forward, save_checkpoint,
)

# ═══════════════════════════════════════════════════════════════════════
# Training Hyperparameters
# ═══════════════════════════════════════════════════════════════════════

BATCH_SIZE = 128
LR_MAX     = 3e-4
LR_MIN     = 1e-5
NUM_STEPS  = 8000

# ═══════════════════════════════════════════════════════════════════════
# Batch Sampling (GPU-vectorized)
# ═══════════════════════════════════════════════════════════════════════

_offsets = torch.arange(BLOCK_SIZE + 1, device=DEVICE).unsqueeze(0)
_max_start = len(data) - BLOCK_SIZE - 1

def sample_batch():
    ix = torch.randint(_max_start, (BATCH_SIZE, 1), device=DEVICE)
    window = data[ix + _offsets]
    return window[:, :-1], window[:, 1:]

# ═══════════════════════════════════════════════════════════════════════
# Adam Optimizer (hand-written)
# ═══════════════════════════════════════════════════════════════════════

beta1, beta2, eps = 0.9, 0.999, 1e-8

def train():
    create_params()
    all_params = params()
    total = sum(p.numel() for p in all_params)

    m = [torch.zeros_like(p) for p in all_params]
    v = [torch.zeros_like(p) for p in all_params]

    def adam_step(step: int, lr: float):
        t = step + 1
        for i, p in enumerate(all_params):
            if p.grad is None:
                continue
            g = p.grad
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            p.data -= lr * m_hat / (v_hat.sqrt() + eps)
            p.grad = None

    print()
    print("🤖 Hello World LLM — Character-Level Transformer (GPU)")
    print("═══════════════════════════════════════════════════════════")
    print(f"  Vocabulary : {VOCAB_SIZE} unique characters")
    print(f"  Context    : {BLOCK_SIZE} characters")
    print(f"  Embedding  : {N_EMBD}-dim, {N_HEAD} heads")
    print(f"  Layers     : {N_LAYER} transformer blocks")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Parameters : {total:,}")
    print(f"  Data       : {len(data):,} characters of 西游记")
    print(f"  Device     : {torch.cuda.get_device_name(0)}")
    print()

    print("📚 Training...")
    t0 = time.time()

    for step in range(NUM_STEPS):
        warmup = 200
        if step < warmup:
            lr = LR_MAX * (step / warmup)
        else:
            decay = (step - warmup) / (NUM_STEPS - warmup)
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * decay))

        inputs, targets = sample_batch()
        logits = transformer_forward(inputs, training=True)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        loss.backward()

        with torch.no_grad():
            norm_sq = sum((p.grad * p.grad).sum() for p in all_params if p.grad is not None)
            norm = norm_sq.sqrt()
            if norm > 1.0:
                for p in all_params:
                    if p.grad is not None:
                        p.grad *= 1.0 / norm

        adam_step(step, lr)

        if step % 500 == 0 or step == NUM_STEPS - 1:
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f"  Step {step:5d} │ Loss: {loss.item():.4f} │ LR: {lr:.6f} │ {elapsed:.1f}s")

    train_time = time.time() - t0
    print(f"\n✅ Training complete in {train_time:.1f}s")

    save_checkpoint()


if __name__ == "__main__":
    train()
