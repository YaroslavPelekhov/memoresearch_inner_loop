"""Crash-safe, auditable trajectory storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from autoresearch.multifidelity.models import RungObservation


def load_observations(path: Path) -> list[RungObservation]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("trajectory state must be a JSON list")
    return [RungObservation.model_validate(item) for item in value]


def save_observations(path: Path, observations: list[RungObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        [item.model_dump(mode="json") for item in observations],
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
