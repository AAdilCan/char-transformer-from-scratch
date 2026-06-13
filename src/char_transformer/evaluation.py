"""Evaluation metrics for the language model.

The model is trained with mean per-token cross-entropy (in nats), so the natural
companion metric is perplexity = exp(loss): the effective number of equally
likely characters the model is choosing among at each step. A model that has
learned nothing sits near the vocabulary size (65 for Shakespeare); a trained
character LM should land well below that.
"""

from __future__ import annotations

import math

import torch

from .data import Dataset
from .model import GPT
from .trainer import estimate_loss


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss (in nats) to perplexity."""
    if loss < 0:
        raise ValueError(f"loss must be >= 0, got {loss}")
    return math.exp(loss)


def evaluate(
    model: GPT,
    dataset: Dataset,
    *,
    block_size: int,
    batch_size: int = 64,
    eval_iters: int = 200,
    device: torch.device | str = "cpu",
    seed: int = 1337,
) -> dict[str, float]:
    """Estimate train/val loss and perplexity for ``model`` on ``dataset``.

    Uses the same averaged-over-random-batches estimator as training so the
    numbers are directly comparable to the values logged during the run.
    """
    device = torch.device(device) if isinstance(device, str) else device
    losses = estimate_loss(
        model,
        dataset,
        block_size=block_size,
        batch_size=batch_size,
        eval_iters=eval_iters,
        device=device,
        generator=torch.Generator().manual_seed(seed),
    )
    return {
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "train_perplexity": perplexity(losses["train"]),
        "val_perplexity": perplexity(losses["val"]),
    }
