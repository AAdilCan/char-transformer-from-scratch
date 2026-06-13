"""Training loop for the character-level Transformer.

The loop is the standard char-LM recipe: sample a random batch of windows,
forward/backward, clip gradients, step AdamW under a cosine LR schedule with
linear warmup, and periodically estimate train/val loss on held-out batches.
The best-by-validation checkpoint is kept so a late-training overfit does not
cost us the best model.

Loss "estimation" averages over several random batches with the model in eval
mode (dropout off). A single batch is too noisy to compare across eval points,
so averaging a handful gives a stable curve worth plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import Config
from .data import Dataset, get_batch
from .logging_utils import get_logger
from .lr_schedule import cosine_lr
from .model import GPT
from .checkpoint import save_checkpoint
from .utils import resolve_device, set_seed

logger = get_logger(__name__)


@dataclass
class TrainHistory:
    """Loss values recorded at each evaluation point during training."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    def record(self, step: int, train: float, val: float, lr: float) -> None:
        self.steps.append(step)
        self.train_loss.append(train)
        self.val_loss.append(val)
        self.learning_rate.append(lr)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "steps": [float(s) for s in self.steps],
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
        }


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset: Dataset,
    *,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Estimate mean train and val loss over ``eval_iters`` random batches.

    The model is switched to eval mode (disabling dropout) for the duration and
    restored to its prior mode afterwards, so calling this mid-training does not
    silently leave dropout off.
    """
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        data = dataset.split(split)
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(
                data, block_size, batch_size, device=device, generator=generator
            )
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


class Trainer:
    """Drives optimization of a :class:`GPT` model on a :class:`Dataset`."""

    def __init__(self, model: GPT, dataset: Dataset, config: Config) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = resolve_device(config.train.device)
        self.model.to(self.device)

        # AdamW with decoupled weight decay applied only to 2-D weight matrices;
        # biases, embeddings, and LayerNorm gains are left undecayed, which is
        # the standard GPT setup and avoids shrinking parameters that should not
        # be regularized toward zero.
        self.optimizer = self._build_optimizer()

        self.history = TrainHistory()
        self.best_val_loss = float("inf")
        # A dedicated generator keeps batch sampling reproducible and decoupled
        # from any global RNG use elsewhere.
        self._gen = torch.Generator().manual_seed(config.train.seed)

    def _build_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.config.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=self.config.train.learning_rate, betas=(0.9, 0.99)
        )

    def _set_lr(self, step: int) -> float:
        tc = self.config.train
        lr = cosine_lr(
            step,
            peak_lr=tc.learning_rate,
            warmup_steps=tc.warmup_steps,
            max_steps=tc.max_steps,
            min_lr_ratio=tc.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _evaluate(self) -> dict[str, float]:
        tc = self.config.train
        return estimate_loss(
            self.model,
            self.dataset,
            block_size=self.config.model.block_size,
            batch_size=tc.batch_size,
            eval_iters=tc.eval_iters,
            device=self.device,
            generator=torch.Generator().manual_seed(tc.seed),
        )

    def train(self) -> TrainHistory:
        """Run the full training loop and return the recorded loss history."""
        set_seed(self.config.train.seed)
        tc = self.config.train
        mc = self.config.model
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "training %s params on %s for %d steps",
            f"{self.model.num_parameters():,}",
            self.device,
            tc.max_steps,
        )
        self.model.train()
        train_data = self.dataset.split("train")

        for step in range(tc.max_steps):
            lr = self._set_lr(step)

            if step % tc.eval_interval == 0 or step == tc.max_steps - 1:
                losses = self._evaluate()
                self.history.record(step, losses["train"], losses["val"], lr)
                logger.info(
                    "step %5d | train %.4f | val %.4f | lr %.2e",
                    step,
                    losses["train"],
                    losses["val"],
                    lr,
                )
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self._save("best.pt", step)

            x, y = get_batch(
                train_data,
                mc.block_size,
                tc.batch_size,
                device=self.device,
                generator=self._gen,
            )
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.optimizer.step()

        self._save("final.pt", tc.max_steps)
        logger.info(
            "done | best val loss %.4f | checkpoints in %s",
            self.best_val_loss,
            out_dir,
        )
        return self.history

    def _save(self, name: str, step: int) -> None:
        save_checkpoint(
            Path(self.config.out_dir) / name,
            model=self.model,
            config=self.config,
            tokenizer=self.dataset.tokenizer,
            step=step,
            optimizer=self.optimizer,
            history=self.history.to_dict(),
            best_val_loss=self.best_val_loss,
        )
