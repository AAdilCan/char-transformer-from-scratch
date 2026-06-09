"""Corpus loading, train/val splitting, and batch sampling.

The whole corpus is encoded once into a 1-D ``torch.long`` tensor. Training
batches are drawn by sampling random start offsets and slicing fixed-length
windows, which is the standard char-LM setup: cheap, and every step sees a
fresh random crop of the text. The validation split is a contiguous tail of
the corpus so it is never seen during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import CharTokenizer


def load_corpus(path: str | Path) -> str:
    """Read the raw text corpus from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path}; run scripts/download_data.py first"
        )
    return path.read_text(encoding="utf-8")


def train_val_split(
    data: torch.Tensor, val_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an encoded corpus into a leading train part and trailing val part.

    The split is contiguous (not shuffled) so validation text is held out as a
    block the model never trains on, mirroring how the model will be used to
    continue unseen prose.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(len(data) * val_fraction)
    if n_val == 0 or n_val == len(data):
        raise ValueError(
            f"val_fraction {val_fraction} yields an empty split for corpus of "
            f"length {len(data)}"
        )
    split = len(data) - n_val
    return data[:split], data[split:]


@dataclass
class Dataset:
    """An encoded corpus split into train/val tensors plus its tokenizer."""

    tokenizer: CharTokenizer
    train_data: torch.Tensor
    val_data: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def split(self, name: str) -> torch.Tensor:
        """Return the ``"train"`` or ``"val"`` tensor by name."""
        if name == "train":
            return self.train_data
        if name == "val":
            return self.val_data
        raise ValueError(f"unknown split {name!r}; expected 'train' or 'val'")


def build_dataset(path: str | Path, val_fraction: float = 0.1) -> Dataset:
    """Load the corpus, fit a tokenizer, encode, and split into train/val."""
    text = load_corpus(path)
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(encoded, val_fraction)
    return Dataset(tokenizer=tokenizer, train_data=train_data, val_data=val_data)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) windows from an encoded corpus.

    For each of ``batch_size`` rows, a random start offset ``i`` is drawn and
    the window ``data[i : i + block_size]`` becomes the input ``x`` while
    ``data[i + 1 : i + 1 + block_size]`` becomes the target ``y`` (the input
    shifted right by one, i.e. next-character prediction at every position).

    Args:
        data: 1-D ``long`` tensor of token ids.
        block_size: Context length of each window.
        batch_size: Number of windows in the batch.
        device: Device to place the returned tensors on.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A tuple ``(x, y)`` of shape ``(batch_size, block_size)``.
    """
    if data.dim() != 1:
        raise ValueError(f"expected a 1-D corpus tensor, got shape {tuple(data.shape)}")
    max_start = len(data) - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"corpus of length {len(data)} is too short for block_size {block_size}"
        )
    ix = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    # Non-blocking is a no-op on CPU but helps when pinned memory is used on GPU.
    return x.to(device), y.to(device)
