"""Tests for perplexity and checkpoint evaluation metrics."""

from __future__ import annotations

import math

import pytest
import torch

from char_transformer.config import ModelConfig
from char_transformer.data import Dataset, train_val_split
from char_transformer.evaluation import evaluate, perplexity
from char_transformer.model import GPT
from char_transformer.tokenizer import CharTokenizer


def test_perplexity_is_exp_of_loss() -> None:
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(65)) == pytest.approx(65.0)


def test_perplexity_rejects_negative_loss() -> None:
    with pytest.raises(ValueError):
        perplexity(-0.1)


def test_untrained_perplexity_near_vocab_size() -> None:
    text = "abcdefgh " * 300
    tok = CharTokenizer.from_text(text)
    encoded = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = train_val_split(encoded, 0.1)
    ds = Dataset(tokenizer=tok, train_data=train, val_data=val)

    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=ds.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    metrics = evaluate(model, ds, block_size=16, batch_size=8, eval_iters=10, device="cpu")

    assert set(metrics) == {"train_loss", "val_loss", "train_perplexity", "val_perplexity"}
    # A fresh model has not learned anything, so perplexity should sit near the
    # vocabulary size (uniform-guess baseline), within a generous band.
    assert metrics["val_perplexity"] < 2 * ds.vocab_size
    assert metrics["val_perplexity"] == pytest.approx(math.exp(metrics["val_loss"]), rel=1e-6)
