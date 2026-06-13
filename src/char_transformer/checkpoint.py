"""Checkpoint save/load.

A checkpoint bundles everything needed to resume training or to generate text
later: the model weights, the full config (so the architecture can be rebuilt
exactly), the tokenizer vocabulary (so encode/decode matches what was trained),
the optimizer state, the step counter, and the loss history. Keeping the config
and tokenizer inside the checkpoint means generation never has to guess
hyperparameters or re-derive a vocabulary from a corpus that might have changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .model import GPT
from .tokenizer import CharTokenizer


def _config_to_payload(cfg: Config) -> dict[str, Any]:
    return cfg.to_dict()


def _config_from_payload(payload: dict[str, Any]) -> Config:
    return Config(
        model=ModelConfig(**payload["model"]),
        train=TrainConfig(**payload["train"]),
        data=DataConfig(**payload["data"]),
        out_dir=payload.get("out_dir", "checkpoints"),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    config: Config,
    tokenizer: CharTokenizer,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    history: dict[str, list[float]] | None = None,
    best_val_loss: float | None = None,
) -> Path:
    """Write a checkpoint to ``path``.

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        model: The model whose ``state_dict`` to save.
        config: Full run config, stored as nested dicts for portability.
        tokenizer: Vocabulary used to encode the training corpus.
        step: Number of optimization steps completed.
        optimizer: Optional optimizer state for resuming training.
        history: Optional dict of loss lists for plotting.
        best_val_loss: Optional best validation loss seen so far.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": _config_to_payload(config),
        "vocab": tokenizer.chars,
        "step": step,
        "history": history or {},
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[GPT, CharTokenizer, Config, dict[str, Any]]:
    """Load a checkpoint and rebuild the model and tokenizer.

    Returns:
        A tuple ``(model, tokenizer, config, payload)``. ``model`` is in eval
        mode with weights loaded; ``payload`` is the raw dict so callers can
        also recover ``step``, ``history``, or ``optimizer_state`` if needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found at {path}")
    # weights_only=False because the payload holds the config dict, not just
    # tensors; the file is produced by this project and trusted.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    config = _config_from_payload(payload["config"])
    tokenizer = CharTokenizer(payload["vocab"])
    model = GPT(config.model)
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    model.eval()
    return model, tokenizer, config, payload
