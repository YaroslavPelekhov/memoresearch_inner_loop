"""Dependency-free helpers for gradient diagnostic telemetry."""

from __future__ import annotations

import math


def gradient_norm_metrics(total_norm: float, threshold: float) -> dict[str, float]:
    """Build stable scalar telemetry from the norm returned by clipping."""

    is_finite = math.isfinite(total_norm)
    if is_finite:
        clip_coefficient = min(1.0, threshold / (total_norm + 1.0e-6))
        post_clip_norm = total_norm * clip_coefficient
    else:
        clip_coefficient = float("nan")
        post_clip_norm = float("nan")
    return {
        "l2_norm/grad/pre_clip_global": total_norm,
        "l2_norm/grad/post_clip_global": post_clip_norm,
        "gradient_clipping/clip_coefficient": clip_coefficient,
        "gradient_clipping/was_applied": float(is_finite and total_norm > threshold),
        "gradient_clipping/nonfinite": float(not is_finite),
    }
