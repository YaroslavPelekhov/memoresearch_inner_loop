from __future__ import annotations

import math

import pytest

from autoresearch.gradient_metrics import gradient_norm_metrics


def test_gradient_norm_metrics_reports_clipping() -> None:
    metrics = gradient_norm_metrics(total_norm=4.0, threshold=1.0)

    assert metrics["l2_norm/grad/pre_clip_global"] == 4.0
    assert metrics["l2_norm/grad/post_clip_global"] == pytest.approx(1.0)
    assert metrics["gradient_clipping/clip_coefficient"] == pytest.approx(0.25)
    assert metrics["gradient_clipping/was_applied"] == 1.0
    assert metrics["gradient_clipping/nonfinite"] == 0.0


def test_gradient_norm_metrics_reports_nonfinite_norm() -> None:
    metrics = gradient_norm_metrics(total_norm=float("nan"), threshold=1.0)

    assert math.isnan(metrics["l2_norm/grad/pre_clip_global"])
    assert math.isnan(metrics["l2_norm/grad/post_clip_global"])
    assert math.isnan(metrics["gradient_clipping/clip_coefficient"])
    assert metrics["gradient_clipping/was_applied"] == 0.0
    assert metrics["gradient_clipping/nonfinite"] == 1.0
