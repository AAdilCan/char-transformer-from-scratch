"""Tests for device resolution and seeding helpers."""

from __future__ import annotations

import pytest
import torch

from char_transformer.utils import resolve_device, set_seed


def test_resolve_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_available_device() -> None:
    dev = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_resolve_unknown_spec_raises() -> None:
    with pytest.raises(ValueError, match="unknown device spec"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present")
def test_resolve_cuda_without_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_set_seed_makes_sampling_reproducible() -> None:
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
