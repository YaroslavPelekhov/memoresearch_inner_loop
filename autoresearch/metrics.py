"""Extract the fixed autoresearch objective from LLM Foundry logs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

CORE_TAG_SUFFIXES = (
    "metrics_gauntlet/core",
    "metrics/metrics_gauntlet/core",
)

_CONSOLE_CORE_PATTERN = re.compile(
    r"^\s*(?:Train|Eval)\s+(?P<tag>\S*metrics_gauntlet/core):\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def _event_accumulator(event_file: Path) -> Any:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - training environment dependency
        raise RuntimeError(
            "TensorBoard is required to parse LLM Foundry evaluation metrics"
        ) from exc
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def _read_console_core_metric(root: Path) -> tuple[float, str, Path]:
    """Read Composer's final CORE scalar when TensorBoard cannot be parsed."""

    log_files = sorted(
        root.rglob("train.stderr.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for log_file in log_files:
        found: list[tuple[float, str]] = []
        with log_file.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = _CONSOLE_CORE_PATTERN.match(line)
                if match is None:
                    continue
                tag = match.group("tag")
                if any(tag.endswith(suffix) for suffix in CORE_TAG_SUFFIXES):
                    found.append((float(match.group("value")), tag))
        if found:
            value, tag = found[-1]
            return value, tag, log_file
    raise KeyError(
        f"No LLM Foundry CORE scalar in Composer console logs under {root}; "
        f"expected a tag ending in {CORE_TAG_SUFFIXES}"
    )


def read_core_metric(log_root: str | Path) -> tuple[float, str, Path]:
    """Return the latest raw equal-weighted CORE metric and its exact tag.

    TensorBoard retains full scalar precision and is preferred. Composer's
    completed console output is a deliberate fallback so an optional reader
    dependency cannot invalidate an otherwise successful multi-hour run.
    """

    root = Path(log_root)
    event_files = sorted(
        root.rglob("events.out.tfevents.*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not event_files:
        return _read_console_core_metric(root)

    for event_file in event_files:
        try:
            accumulator = _event_accumulator(event_file)
        except (OSError, RuntimeError, ValueError):
            continue
        found: list[tuple[int, float, str]] = []
        scalar_tags = accumulator.Tags().get("scalars", [])
        for tag in scalar_tags:
            if not any(tag.endswith(suffix) for suffix in CORE_TAG_SUFFIXES):
                continue
            events = accumulator.Scalars(tag)
            if events:
                latest = events[-1]
                found.append((int(latest.step), float(latest.value), tag))
        if found:
            _, value, tag = max(found, key=lambda item: item[0])
            return value, tag, event_file
    return _read_console_core_metric(root)
