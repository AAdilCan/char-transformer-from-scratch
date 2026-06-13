"""Generate text from a trained checkpoint.

Loads a checkpoint (model weights + tokenizer + config), encodes an optional
prompt, and autoregressively samples a continuation with temperature/top-k
control. With no prompt the model is seeded with a single newline, which is how
the corpus separates speakers, so it tends to start a fresh line of dialogue.

Examples:
    python scripts/generate.py --checkpoint checkpoints/best.pt --max-new-tokens 500
    python scripts/generate.py --checkpoint checkpoints/best.pt \
        --prompt "ROMEO:" --temperature 0.8 --top-k 40 --num-samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from char_transformer.checkpoint import load_checkpoint  # noqa: E402
from char_transformer.utils import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--prompt", default="", help="conditioning text")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="chars to sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="softmax temperature")
    parser.add_argument("--top-k", type=int, default=None, help="restrict to top-k logits")
    parser.add_argument("--num-samples", type=int, default=1, help="how many continuations")
    parser.add_argument("--seed", type=int, default=1337, help="rng seed")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    return parser.parse_args()


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Encode the prompt to a (1, T) context, defaulting to a newline seed."""
    if prompt == "":
        seed_char = "\n" if "\n" in tokenizer.stoi else tokenizer.chars[0]
        ids = [tokenizer.stoi[seed_char]]
    else:
        ids = tokenizer.encode(prompt)
    return torch.tensor([ids], dtype=torch.long, device=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer, _config, _payload = load_checkpoint(args.checkpoint, map_location=device)
    context = encode_prompt(tokenizer, args.prompt, device)

    for i in range(args.num_samples):
        out = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        if args.num_samples > 1:
            print(f"\n===== sample {i + 1}/{args.num_samples} =====")
        print(text)


if __name__ == "__main__":
    main()
