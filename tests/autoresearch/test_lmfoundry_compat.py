"""Unit tests for LLM Foundry compatibility helpers."""

import pytest

from autoresearch.lmfoundry_compat import _streaming_prefix_candidates


@pytest.mark.parametrize(
    ("base", "expected_first", "expected_last"),
    [
        (700_000, 700_000, 799_999),
        (800_000, 800_000, 899_999),
        (999_999, 999_999, 999_999),
    ],
)
def test_streaming_prefix_candidates_stay_in_private_range(
    base: int,
    expected_first: int,
    expected_last: int,
) -> None:
    candidates = _streaming_prefix_candidates(base)
    assert next(candidates) == expected_first
    assert list(candidates)[-1] == expected_last


@pytest.mark.parametrize("base", [-1, 1_000_000])
def test_streaming_prefix_candidates_reject_invalid_base(base: int) -> None:
    with pytest.raises(ValueError):
        next(_streaming_prefix_candidates(base))
