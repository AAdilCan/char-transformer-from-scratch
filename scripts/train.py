"""Train the character-level Transformer from a YAML config.

Loads the config, builds the dataset (which fits the tokenizer and sets the
model's vocab_size), constructs the model, and runs the training loop. Best and
final checkpoints are written under ``out_dir``. A few CLI flags override the
most commonly tweaked hyperparameters without editing the YAML.

Examples:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --max-steps 1000 --device cpu --out-dir checkpoints/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from char_transformer.config import Config  # noqa: E402
from char_transformer.data import build_dataset  # noqa: E402
from char_transformer.logging_utils import get_logger  # noqa: E402
from char_transformer.model import GPT  # noqa: E402
from char_transformer.trainer import Trainer  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the char-level Transformer")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config")
    parser.add_argument("--data-path", default=None, help="override corpus path")
    parser.add_argument("--out-dir", default=None, help="override checkpoint dir")
    parser.add_argument("--max-steps", type=int, default=None, help="override step count")
    parser.add_argument("--warmup-steps", type=int, default=None, help="override warmup steps")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--device", default=None, help="auto|cpu|cuda")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Load the YAML config and apply any CLI overrides."""
    cfg = Config.from_yaml(args.config)
    if args.data_path is not None:
        cfg.data.data_path = args.data_path
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = args.warmup_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    dataset = build_dataset(cfg.data.data_path, val_fraction=cfg.data.val_fraction)
    cfg.model.vocab_size = dataset.vocab_size
    logger.info(
        "corpus %d train / %d val chars | vocab %d",
        len(dataset.train_data),
        len(dataset.val_data),
        dataset.vocab_size,
    )

    model = GPT(cfg.model)
    trainer = Trainer(model, dataset, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
