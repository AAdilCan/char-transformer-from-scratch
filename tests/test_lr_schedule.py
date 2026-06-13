"""Tests for the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import pytest

from char_transformer.lr_schedule import cosine_lr


def test_warmup_ramps_linearly_to_peak() -> None:
    peak = 1e-3
    # Last warmup step (step == warmup_steps - 1) should hit the peak exactly.
    assert cosine_lr(9, peak_lr=peak, warmup_steps=10, max_steps=100) == pytest.approx(peak)
    # Halfway through warmup is ~half the peak.
    assert cosine_lr(4, peak_lr=peak, warmup_steps=10, max_steps=100) == pytest.approx(peak * 0.5)


def test_warmup_is_monotonic_increasing() -> None:
    vals = [cosine_lr(s, peak_lr=1.0, warmup_steps=20, max_steps=200) for s in range(20)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_decay_reaches_min_lr_at_end() -> None:
    peak, ratio = 1e-3, 0.1
    # The floor is hit exactly at step == max_steps; the final in-range step
    # sits just above it.
    assert cosine_lr(100, peak_lr=peak, warmup_steps=10, max_steps=100, min_lr_ratio=ratio) == (
        pytest.approx(peak * ratio)
    )
    last = cosine_lr(99, peak_lr=peak, warmup_steps=10, max_steps=100, min_lr_ratio=ratio)
    assert peak * ratio < last < peak * ratio * 1.05


def test_decay_is_monotonic_decreasing() -> None:
    vals = [cosine_lr(s, peak_lr=1.0, warmup_steps=10, max_steps=100) for s in range(10, 100)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_midpoint_of_cosine_is_halfway_between_peak_and_floor() -> None:
    peak, ratio = 1.0, 0.1
    min_lr = peak * ratio
    # Halfway through the decay span, cos(pi/2)=0 gives the midpoint.
    mid = cosine_lr(55, peak_lr=peak, warmup_steps=10, max_steps=100, min_lr_ratio=ratio)
    assert mid == pytest.approx(min_lr + 0.5 * (peak - min_lr), rel=1e-6)


def test_holds_floor_past_max_steps() -> None:
    peak, ratio = 1.0, 0.05
    assert cosine_lr(500, peak_lr=peak, warmup_steps=10, max_steps=100, min_lr_ratio=ratio) == (
        pytest.approx(peak * ratio)
    )


def test_no_warmup_starts_decaying_immediately() -> None:
    assert cosine_lr(0, peak_lr=1.0, warmup_steps=0, max_steps=100) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"peak_lr": 0.0, "warmup_steps": 10, "max_steps": 100},
        {"peak_lr": 1.0, "warmup_steps": 10, "max_steps": 100, "min_lr_ratio": 1.5},
        {"peak_lr": 1.0, "warmup_steps": -1, "max_steps": 100},
        {"peak_lr": 1.0, "warmup_steps": 10, "max_steps": 0},
        {"peak_lr": 1.0, "warmup_steps": 200, "max_steps": 100},
    ],
)
def test_invalid_arguments_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        cosine_lr(0, **kwargs)


def test_lr_stays_within_bounds_across_full_schedule() -> None:
    # Warmup ramps up from near zero (so LR can briefly sit below the floor),
    # but it must never exceed the peak and is always positive.
    peak, ratio, warmup = 3e-4, 0.1, 50
    for step in range(0, 1000):
        lr = cosine_lr(step, peak_lr=peak, warmup_steps=warmup, max_steps=1000, min_lr_ratio=ratio)
        assert 0.0 < lr <= peak + 1e-12
        assert not math.isnan(lr)
        # Past warmup, the cosine arm never drops below the floor.
        if step >= warmup:
            assert lr >= peak * ratio - 1e-12
