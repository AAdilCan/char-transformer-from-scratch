"""Tests for the character tokenizer."""

from __future__ import annotations

import pytest

from char_transformer.tokenizer import CharTokenizer

SAMPLE = "hello, world!\nthe quick brown fox."


def test_from_text_builds_sorted_unique_vocab() -> None:
    tok = CharTokenizer.from_text(SAMPLE)
    assert tok.chars == sorted(set(SAMPLE))
    assert tok.vocab_size == len(set(SAMPLE))
    assert len(tok) == tok.vocab_size


def test_roundtrip_is_identity() -> None:
    tok = CharTokenizer.from_text(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_ids_are_contiguous_and_consistent() -> None:
    tok = CharTokenizer.from_text(SAMPLE)
    ids = tok.encode(SAMPLE)
    assert set(ids) <= set(range(tok.vocab_size))
    # stoi and itos must be exact inverses.
    for ch, i in tok.stoi.items():
        assert tok.itos[i] == ch


def test_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_duplicate_chars_rejected() -> None:
    with pytest.raises(ValueError):
        CharTokenizer(["a", "b", "a"])


def test_encode_unknown_char_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("z")


def test_decode_out_of_range_id_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.decode([99])


def test_save_and_load_roundtrip(tmp_path) -> None:
    tok = CharTokenizer.from_text(SAMPLE)
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tok.chars
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)
