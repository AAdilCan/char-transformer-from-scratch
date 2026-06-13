"""Evaluate a checkpoint and plot its training curves.

Reports train/val loss and perplexity (re-estimated on fresh random batches),
and renders the loss curve and LR schedule that were recorded during training
into ``reports/``. The history lives inside the checkpoint, so no separate log
file is needed.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from char_transformer.checkpoint import load_checkpoint  # noqa: E402
from char_transformer.data import build_dataset  # noqa: E402
from char_transformer.evaluation import evaluate  # noqa: E402
from char_transformer.logging_utils import get_logger  # noqa: E402
from char_transformer.plotting import plot_loss_curves, plot_lr_schedule  # noqa: E402
from char_transformer.utils import resolve_device  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and plot curves")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--data-path", default=None, help="corpus path (defaults to config)")
    parser.add_argument("--reports-dir", default="reports", help="where to write plots/metrics")
    parser.add_argument("--eval-iters", type=int, default=200, help="batches per loss estimate")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    reports = Path(args.reports_dir)

    model, _tokenizer, config, payload = load_checkpoint(args.checkpoint, map_location=device)
    data_path = args.data_path or config.data.data_path
    dataset = build_dataset(data_path, val_fraction=config.data.val_fraction)

    metrics = evaluate(
        model,
        dataset,
        block_size=config.model.block_size,
        batch_size=config.train.batch_size,
        eval_iters=args.eval_iters,
        device=device,
        seed=config.train.seed,
    )
    metrics["step"] = payload.get("step")
    logger.info(
        "step %s | val loss %.4f | val ppl %.2f | train loss %.4f | train ppl %.2f",
        metrics["step"],
        metrics["val_loss"],
        metrics["val_perplexity"],
        metrics["train_loss"],
        metrics["train_perplexity"],
    )

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history = payload.get("history") or {}
    if history.get("steps"):
        plot_loss_curves(history, reports / "loss_curve.png")
        plot_lr_schedule(history, reports / "lr_schedule.png")
        logger.info("wrote loss_curve.png and lr_schedule.png to %s", reports)
    else:
        logger.warning("checkpoint has no recorded history; skipping plots")


if __name__ == "__main__":
    main()
