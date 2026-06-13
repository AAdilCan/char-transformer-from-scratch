"""Plot training curves from a recorded :class:`TrainHistory`.

Matplotlib is forced onto the non-interactive ``Agg`` backend so the plots
render to PNG files in headless runs (CI, a server) without needing a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_curves(history: dict[str, list[float]], out_path: str | Path) -> Path:
    """Plot train and validation loss against training step.

    Args:
        history: Dict with ``steps``, ``train_loss``, and ``val_loss`` lists, as
            produced by :meth:`TrainHistory.to_dict`.
        out_path: Destination PNG path.

    Returns:
        The path written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps = history["steps"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, history["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(steps, history["val_loss"], label="val", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_lr_schedule(history: dict[str, list[float]], out_path: str | Path) -> Path:
    """Plot the learning rate against training step."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["steps"], history["learning_rate"], color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule (warmup + cosine decay)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
