"""Tests for checkpoint save/load roundtripping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from char_transformer.checkpoint import load_checkpoint, save_checkpoint
from char_transformer.config import Config, ModelConfig
from char_transformer.model import GPT
from char_transformer.tokenizer import CharTokenizer


@pytest.fixture
def setup() -> tuple[GPT, Config, CharTokenizer]:
    cfg = Config(model=ModelConfig(vocab_size=12, block_size=16, n_layer=2, n_head=2, n_embd=32))
    torch.manual_seed(0)
    model = GPT(cfg.model)
    tokenizer = CharTokenizer(list("abcdefghijkl"))
    return model, cfg, tokenizer


def test_roundtrip_preserves_weights(tmp_path: Path, setup) -> None:
    model, cfg, tok = setup
    path = save_checkpoint(tmp_path / "ckpt.pt", model=model, config=cfg, tokenizer=tok, step=42)

    loaded, loaded_tok, loaded_cfg, payload = load_checkpoint(path)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)
    assert payload["step"] == 42


def test_roundtrip_preserves_config_and_vocab(tmp_path: Path, setup) -> None:
    model, cfg, tok = setup
    path = save_checkpoint(tmp_path / "ckpt.pt", model=model, config=cfg, tokenizer=tok, step=0)

    _, loaded_tok, loaded_cfg, _ = load_checkpoint(path)
    assert loaded_cfg.model.n_layer == cfg.model.n_layer
    assert loaded_cfg.model.block_size == cfg.model.block_size
    assert loaded_tok.chars == tok.chars
    assert loaded_tok.vocab_size == tok.vocab_size


def test_loaded_model_produces_identical_logits(tmp_path: Path, setup) -> None:
    model, cfg, tok = setup
    model.eval()
    path = save_checkpoint(tmp_path / "ckpt.pt", model=model, config=cfg, tokenizer=tok, step=0)
    loaded, _, _, _ = load_checkpoint(path)

    idx = torch.randint(0, cfg.model.vocab_size, (2, 8))
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = loaded(idx)
    assert torch.allclose(a, b, atol=1e-6)


def test_optimizer_state_is_saved_when_provided(tmp_path: Path, setup) -> None:
    model, cfg, tok = setup
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", model=model, config=cfg, tokenizer=tok, step=1, optimizer=opt
    )
    _, _, _, payload = load_checkpoint(path)
    assert "optimizer_state" in payload


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
