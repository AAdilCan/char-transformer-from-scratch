"""Device resolution and seeding helpers shared by training and generation."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    ``"auto"`` picks CUDA when available and falls back to CPU. ``"cpu"`` and
    ``"cuda"`` are honored as given; requesting CUDA when it is unavailable is
    an error rather than a silent downgrade, so a misconfigured run fails loud.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if spec not in {"cpu", "cuda"}:
        raise ValueError(f"unknown device spec {spec!r}; expected auto|cpu|cuda")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
