from __future__ import annotations

import json
from typing import Any, Iterable

STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"
STRUCTURED_BENCHMARK_FEEDBACK_MARKER = "[gigaevo] structured benchmark feedback:"
LEGACY_PROGRAMBENCH_STRUCTURED_FEEDBACK_MARKER = (
    "[programbench] structured failure feedback:"
)

DEFAULT_STRUCTURED_FEEDBACK_MARKERS = (
    STRUCTURED_FEEDBACK_MARKER,
    STRUCTURED_BENCHMARK_FEEDBACK_MARKER,
    LEGACY_PROGRAMBENCH_STRUCTURED_FEEDBACK_MARKER,
)


def extract_structured_feedback(
    text: str, markers: Iterable[str] | None = None
) -> dict[str, Any] | None:
    """Extract the last JSON object emitted after a structured-feedback marker."""

    marker_list = (
        DEFAULT_STRUCTURED_FEEDBACK_MARKERS if markers is None else tuple(markers)
    )
    marker_hits = []
    for marker in marker_list:
        index = text.rfind(marker)
        if index >= 0:
            marker_hits.append((index, marker))
    if not marker_hits:
        return None

    for index, marker in sorted(marker_hits, reverse=True):
        json_start = text.find("{", index + len(marker))
        if json_start < 0:
            continue
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
