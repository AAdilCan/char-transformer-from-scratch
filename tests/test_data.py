"""Tests for corpus splitting and batch sampling."""

from __future__ import annotations

import torch

import pytest

from char_transformer.data import (
    Dataset,
    build_dataset,
    get_batch,
    load_corpus,
    train_val_split,
)
from char_transformer.tokenizer import CharTokenizer


def _toy_corpus(tmp_path, text: str = "abcdefgh" * 50):
    path = tmp_path / "input.txt"
    path.write_text(text, encoding="utf-8")
    return path, text


def test_load_corpus_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope.txt")


def test_train_val_split_sizes_and_disjointness() -> None:
    data = torch.arange(100)
    train, val = train_val_split(data, val_fraction=0.1)
    assert len(val) == 10
    assert len(train) == 90
    assert len(train) + len(val) == len(data)
    # Contiguous split: concatenation reconstructs the original corpus.
    assert torch.equal(torch.cat([train, val]), data)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_train_val_split_rejects_bad_fraction(frac: float) -> None:
    with pytest.raises(ValueError):
        train_val_split(torch.arange(100), val_fraction=frac)


def test_build_dataset_end_to_end(tmp_path) -> None:
    path, text = _toy_corpus(tmp_path)
    ds = build_dataset(path, val_fraction=0.2)
    assert isinstance(ds, Dataset)
    assert ds.vocab_size == len(set(text))
    # Decoding the concatenated splits recovers the original corpus exactly.
    recovered = ds.tokenizer.decode(torch.cat([ds.train_data, ds.val_data]).tolist())
    assert recovered == text


def test_dataset_split_lookup(tmp_path) -> None:
    path, _ = _toy_corpus(tmp_path)
    ds = build_dataset(path)
    assert torch.equal(ds.split("train"), ds.train_data)
    assert torch.equal(ds.split("val"), ds.val_data)
    with pytest.raises(ValueError):
        ds.split("test")


def test_get_batch_shapes_and_dtype() -> None:
    data = torch.arange(500)
    x, y = get_batch(data, block_size=16, batch_size=8)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.int64


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    data = torch.arange(500)
    x, y = get_batch(data, block_size=16, batch_size=4)
    # y[t] is the next token after x[t]; with arange data that means y == x + 1.
    assert torch.equal(y, x + 1)


def test_get_batch_is_reproducible_with_generator() -> None:
    data = torch.arange(500)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1, y1 = get_batch(data, block_size=8, batch_size=4, generator=g1)
    x2, y2 = get_batch(data, block_size=8, batch_size=4, generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_get_batch_rejects_non_1d() -> None:
    with pytest.raises(ValueError):
        get_batch(torch.zeros(4, 4), block_size=2, batch_size=2)


def test_get_batch_rejects_too_short_corpus() -> None:
    with pytest.raises(ValueError):
        get_batch(torch.arange(10), block_size=64, batch_size=2)
