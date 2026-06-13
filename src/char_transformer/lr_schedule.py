"""Learning-rate schedule: linear warmup followed by cosine decay.

The schedule ramps the learning rate linearly from 0 to the peak over the first
``warmup_steps`` iterations, then decays it along a cosine curve down to a floor
of ``min_lr_ratio * peak``. Warmup keeps the first updates small while the
Adam moment estimates are still noisy; cosine decay gives a smooth, well-tested
anneal that ends at a small-but-nonzero rate so late training keeps refining.

The function is pure (step in, LR out) so it can be unit-tested and plotted in
isolation from the optimizer.
"""

from __future__ import annotations

import math


def cosine_lr(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Return the learning rate for a given training ``step`` (0-indexed).

    Args:
        step: Current optimization step, starting at 0.
        peak_lr: Maximum learning rate, reached at the end of warmup.
        warmup_steps: Number of steps spent linearly ramping up to ``peak_lr``.
        max_steps: Total number of training steps; decay completes here.
        min_lr_ratio: Floor learning rate as a fraction of ``peak_lr``.

    Returns:
        The learning rate to use at ``step``.
    """
    if peak_lr <= 0:
        raise ValueError(f"peak_lr must be > 0, got {peak_lr}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if warmup_steps < 0 or max_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and max_steps must be > 0")
    if warmup_steps > max_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) cannot exceed max_steps ({max_steps})"
        )

    min_lr = peak_lr * min_lr_ratio

    # Linear warmup. Step 0 gets a tiny nonzero LR rather than exactly 0 so the
    # very first update still moves; the ramp uses (step + 1) / warmup_steps.
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Past the end of schedule: hold at the floor.
    if step >= max_steps:
        return min_lr

    # Cosine decay from peak_lr down to min_lr over the post-warmup span.
    decay_steps = max_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)
