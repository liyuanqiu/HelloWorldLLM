"""Generate text from a trained character-level Transformer."""

import sys
import torch

from model import (
    DEVICE, BLOCK_SIZE, VOCAB_SIZE,
    encode, itos, stoi,
    load_checkpoint, transformer_forward, p,
)

# ═══════════════════════════════════════════════════════════════════════
# Text Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(prompt: str = "悟空", length: int = 500, temperature: float = 0.8, top_k: int = 40) -> str:
    context = encode(prompt)
    pad_idx = stoi.get(" ", 0)
    while len(context) < BLOCK_SIZE:
        context = [pad_idx] + context
    ctx = torch.tensor(context[-BLOCK_SIZE:], dtype=torch.long, device=DEVICE).unsqueeze(0)

    result = list(prompt)
    for _ in range(length):
        logits = transformer_forward(ctx, training=False)
        logits = logits[:, -1, :].squeeze(0) / temperature

        topk_vals, topk_idx = logits.topk(top_k)
        probs = torch.softmax(topk_vals, dim=-1)
        next_idx = topk_idx[torch.multinomial(probs, 1)].item()
        result.append(itos[next_idx])

        next_tok = torch.tensor([[next_idx]], device=DEVICE)
        ctx = torch.cat([ctx[:, 1:], next_tok], dim=1)

    return "".join(result)


def main():
    load_checkpoint()

    prompt = sys.argv[1] if len(sys.argv) > 1 else "悟空"
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    print(f"\n✨ Generating {length} chars from \"{prompt}\" (temperature={temperature}):")
    print("───────────────────────────────────────────────────────────")
    print(generate(prompt, length, temperature))
    print("───────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
