"""Tests for the GPT model: shapes, causality, loss, and generation."""

from __future__ import annotations

import math

import pytest
import torch

from char_transformer.config import ModelConfig
from char_transformer.model import GPT


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0
    )


@pytest.fixture
def model(cfg: ModelConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(cfg)


def test_forward_logits_shape(model: GPT, cfg: ModelConfig) -> None:
    idx = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(idx)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is None


def test_forward_returns_scalar_loss_with_targets(model: GPT, cfg: ModelConfig) -> None:
    idx = torch.randint(0, cfg.vocab_size, (3, 16))
    targets = torch.randint(0, cfg.vocab_size, (3, 16))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0


def test_init_loss_near_uniform(cfg: ModelConfig) -> None:
    # A well-initialized LM should start near -ln(1/vocab) = ln(vocab).
    torch.manual_seed(1)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (32, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (32, cfg.block_size))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


def test_attention_is_causal(model: GPT, cfg: ModelConfig) -> None:
    # Mutating the last token must not change logits at earlier positions.
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    x_mut = x.clone()
    x_mut[0, -1] = (x_mut[0, -1] + 1) % cfg.vocab_size
    logits, _ = model(x)
    logits_mut, _ = model(x_mut)
    assert torch.allclose(logits[0, :-1], logits_mut[0, :-1], atol=1e-5)


def test_rejects_sequence_longer_than_block_size(model: GPT, cfg: ModelConfig) -> None:
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_generate_extends_context_by_max_new_tokens(model: GPT, cfg: ModelConfig) -> None:
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model.generate(idx, max_new_tokens=12, temperature=0.8, top_k=10)
    assert out.shape == (2, 17)
    assert torch.equal(out[:, :5], idx)  # original context preserved


def test_generate_stays_in_vocab_beyond_block_size(model: GPT, cfg: ModelConfig) -> None:
    # Generating past block_size exercises the context cropping path.
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=cfg.block_size + 5)
    assert out.min().item() >= 0
    assert out.max().item() < cfg.vocab_size


def test_generate_rejects_nonpositive_temperature(model: GPT, cfg: ModelConfig) -> None:
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError, match="temperature"):
        model.generate(idx, max_new_tokens=1, temperature=0.0)


def test_num_parameters_counts_trainable(cfg: ModelConfig) -> None:
    model = GPT(cfg)
    total = model.num_parameters()
    assert total > 0
    assert total == sum(p.numel() for p in model.parameters())


def test_invalid_vocab_size_rejected() -> None:
    bad = ModelConfig(vocab_size=0, block_size=16, n_layer=1, n_head=2, n_embd=16)
    with pytest.raises(ValueError, match="vocab_size"):
        GPT(bad)
