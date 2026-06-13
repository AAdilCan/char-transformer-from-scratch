"""Tests for the training loop, loss estimation, and history tracking."""

from __future__ import annotations

from pathlib import Path

import torch

from char_transformer.config import Config, DataConfig, ModelConfig, TrainConfig
from char_transformer.data import Dataset, train_val_split
from char_transformer.model import GPT
from char_transformer.tokenizer import CharTokenizer
from char_transformer.trainer import TrainHistory, Trainer, estimate_loss


def _toy_dataset() -> Dataset:
    # A short, highly repetitive corpus so a tiny model can actually drive the
    # loss down within a handful of steps.
    text = "hello world. " * 400
    tok = CharTokenizer.from_text(text)
    encoded = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = train_val_split(encoded, 0.1)
    return Dataset(tokenizer=tok, train_data=train, val_data=val)


def _toy_config(vocab_size: int, out_dir: Path) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
        ),
        train=TrainConfig(
            batch_size=16,
            max_steps=60,
            eval_interval=20,
            eval_iters=10,
            learning_rate=1e-3,
            warmup_steps=10,
            device="cpu",
            seed=0,
        ),
        data=DataConfig(),
        out_dir=str(out_dir),
    )


def test_train_history_record_appends_aligned_lists() -> None:
    h = TrainHistory()
    h.record(0, 2.0, 2.1, 1e-4)
    h.record(10, 1.5, 1.6, 3e-4)
    assert h.steps == [0, 10]
    assert h.train_loss == [2.0, 1.5]
    assert len(h.val_loss) == len(h.learning_rate) == 2


def test_estimate_loss_returns_both_splits() -> None:
    ds = _toy_dataset()
    model = GPT(ModelConfig(vocab_size=ds.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    out = estimate_loss(
        model, ds, block_size=16, batch_size=8, eval_iters=5, device=torch.device("cpu")
    )
    assert set(out) == {"train", "val"}
    assert out["train"] > 0 and out["val"] > 0


def test_estimate_loss_restores_training_mode() -> None:
    ds = _toy_dataset()
    model = GPT(ModelConfig(vocab_size=ds.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    model.train()
    estimate_loss(model, ds, block_size=16, batch_size=8, eval_iters=2, device=torch.device("cpu"))
    assert model.training is True


def test_optimizer_splits_decay_and_no_decay_groups(tmp_path: Path) -> None:
    ds = _toy_dataset()
    cfg = _toy_config(ds.vocab_size, tmp_path)
    trainer = Trainer(GPT(cfg.model), ds, cfg)
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == cfg.train.weight_decay
    assert groups[1]["weight_decay"] == 0.0
    # No-decay group holds 1-D params (biases, norms, embeddings only via dim<2).
    assert all(p.dim() < 2 for p in groups[1]["params"])


def test_training_reduces_loss_and_writes_checkpoints(tmp_path: Path) -> None:
    ds = _toy_dataset()
    cfg = _toy_config(ds.vocab_size, tmp_path)
    trainer = Trainer(GPT(cfg.model), ds, cfg)
    history = trainer.train()

    assert history.train_loss[-1] < history.train_loss[0]
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "final.pt").exists()
    assert trainer.best_val_loss < float("inf")
