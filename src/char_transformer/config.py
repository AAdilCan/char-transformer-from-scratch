"""Typed configuration for the model, data, and training loop.

Configs are plain dataclasses so they can be constructed in code, loaded from a
YAML file, and serialized back out alongside a checkpoint for reproducibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the decoder-only Transformer."""

    vocab_size: int = 0  # filled in from the tokenizer at build time
    block_size: int = 128  # max context length in characters
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


@dataclass
class TrainConfig:
    """Optimization and training-loop hyperparameters."""

    batch_size: int = 64
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1  # final LR as a fraction of peak LR
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class DataConfig:
    """Where the corpus lives and how to split it."""

    data_path: str = "data/input.txt"
    val_fraction: float = 0.1


@dataclass
class Config:
    """Top-level config bundling model, training, and data settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    out_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a config from YAML, applying defaults for any missing fields."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            data=DataConfig(**raw.get("data", {})),
            out_dir=raw.get("out_dir", "checkpoints"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten the config into nested plain dicts (for JSON/YAML dumps)."""
        return dataclasses.asdict(self)
