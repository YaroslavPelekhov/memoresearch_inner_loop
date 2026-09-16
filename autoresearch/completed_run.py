"""Build and verify immutable receipts for already completed baseline runs."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from autoresearch.benchmark import (
    _completed_evaluation_after_nonzero_exit,
    _dataset_capacity,
    _evaluation_provenance,
    _nonfinite_training_gradient,
    _nonfinite_training_loss,
    _parameter_count,
)
from autoresearch.metrics import read_core_metric

SCHEMA_VERSION = 1
KIND = "completed_baseline_benchmark"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_metrics(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "is_valid" in value:
            return value
    raise RuntimeError(f"No adapter metrics found in {path}")


def _record(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Receipt file is outside the completed run: {path}"
        ) from exc
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def build_receipt(
    run_dir: Path,
    *,
    adapter_stdout: Path,
    calibrated_commit: str,
    operational_commit: str,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    effective_config = run_dir / "effective-config.yaml"
    train_stdout = run_dir / "train.stdout.log"
    train_stderr = run_dir / "train.stderr.log"
    capacity = _dataset_capacity(effective_config)
    provenance = _evaluation_provenance(effective_config)
    if _nonfinite_training_gradient(train_stderr) is not None:
        raise RuntimeError("Completed baseline has a non-finite gradient")
    if _nonfinite_training_loss(train_stderr) is not None:
        raise RuntimeError("Completed baseline has a non-finite loss")
    completion = _completed_evaluation_after_nonzero_exit(
        train_stderr,
        expected_batches=int(capacity["batches"]),
        evaluation_provenance=provenance,
    )
    if completion is None:
        raise RuntimeError("Completed baseline lacks full teardown-safe evidence")
    core_value, core_tag, event_file = read_core_metric(run_dir)
    n_params = _parameter_count(train_stdout)
    if not math.isfinite(core_value) or n_params is None:
        raise RuntimeError("Completed baseline metric or parameter count is invalid")
    previous = _last_metrics(adapter_stdout)
    metrics = {
        "fitness": core_value,
        "is_valid": 1.0,
        "llmfoundry_core_equal_raw": core_value,
        "n_params": float(n_params),
        "wall_time_seconds": float(previous.get("wall_time_seconds", 0.0)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "calibrated_commit": calibrated_commit,
        "operational_commit": operational_commit,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "completion_evidence": completion,
        "core_tag": core_tag,
        "files": {
            "effective_config": _record(effective_config, run_dir),
            "train_stdout": _record(train_stdout, run_dir),
            "train_stderr": _record(train_stderr, run_dir),
            "event_file": _record(event_file, run_dir),
        },
        "original_adapter_metrics": previous,
    }


def verify_receipt(
    receipt_path: Path,
    *,
    calibrated_commit: str,
    operational_commit: str,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("Completed baseline receipt is not an object")
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "calibrated_commit": calibrated_commit,
        "operational_commit": operational_commit,
    }
    for key, expected in expected_header.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"Completed baseline receipt field {key} is invalid: "
                f"{receipt.get(key)!r} != {expected!r}"
            )
    run_dir = Path(str(receipt.get("run_dir", ""))).resolve()
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Completed baseline receipt has no file records")
    for name, raw_record in files.items():
        if not isinstance(raw_record, dict):
            raise RuntimeError(f"Completed baseline file record {name} is invalid")
        path = (run_dir / str(raw_record.get("path", ""))).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"Completed baseline file {name} escapes its run"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != raw_record.get("bytes")
            or file_sha256(path) != raw_record.get("sha256")
        ):
            raise RuntimeError(f"Completed baseline file {name} changed")
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Completed baseline receipt has no metrics")
    required = ("fitness", "is_valid", "llmfoundry_core_equal_raw", "n_params")
    if (
        any(
            type(metrics.get(key)) not in {int, float}
            or not math.isfinite(float(metrics[key]))
            for key in required
        )
        or float(metrics["is_valid"]) <= 0
    ):
        raise RuntimeError("Completed baseline receipt metrics are invalid")
    return receipt
