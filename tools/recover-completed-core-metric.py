#!/usr/bin/env python3
"""Recover a completed run rejected only because its CORE parser was missing."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.benchmark import (
    STRUCTURED_FEEDBACK_MARKER,
    _nonfinite_training_gradient,
    _nonfinite_training_loss,
    _parameter_count,
)
from autoresearch.metrics import read_core_metric


def _last_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "is_valid" in value:
            return value
    return None


def _last_feedback(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    prefix = STRUCTURED_FEEDBACK_MARKER + " "
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def recover(run_dir: Path) -> dict[str, float]:
    run_dir = run_dir.resolve()
    stdout_path = run_dir / "adapter.stdout.log"
    stderr_path = run_dir / "adapter.stderr.log"
    train_stdout_path = run_dir / "train.stdout.log"
    train_stderr_path = run_dir / "train.stderr.log"
    contract_path = run_dir / "calibration-contract.json"
    receipt_path = run_dir / "metric-recovery.json"

    previous = _last_metrics(stdout_path)
    if previous is None:
        raise RuntimeError("Adapter emitted no previous metrics")
    if float(previous.get("is_valid", 0.0)) > 0.0:
        return {key: float(value) for key, value in previous.items()}
    feedback = _last_feedback(stderr_path)
    if feedback is None or feedback.get("status") != "metric_missing":
        raise RuntimeError("Run was not rejected solely as metric_missing")
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_batches = int(contract["batches"])
    train_log = train_stderr_path.read_text(encoding="utf-8", errors="replace")
    completion_markers = (
        f"[batch={expected_batches}/{expected_batches}]",
        f"Train time/batch: {expected_batches}",
        " Done.",
    )
    missing = [marker for marker in completion_markers if marker not in train_log]
    if missing:
        raise RuntimeError(f"Training completion markers are missing: {missing}")
    if _nonfinite_training_gradient(train_stderr_path) is not None:
        raise RuntimeError("Cannot recover a run with a non-finite gradient")
    if _nonfinite_training_loss(train_stderr_path) is not None:
        raise RuntimeError("Cannot recover a run with a non-finite loss")

    core_value, core_tag, metric_path = read_core_metric(run_dir)
    if not math.isfinite(core_value):
        raise RuntimeError(f"Recovered CORE metric is not finite: {core_value}")
    n_params = _parameter_count(train_stdout_path)
    metrics = {
        "fitness": core_value,
        "is_valid": 1.0,
        "llmfoundry_core_equal_raw": core_value,
        "wall_time_seconds": float(previous.get("wall_time_seconds", 0.0)),
        **({"n_params": float(n_params)} if n_params is not None else {}),
    }
    receipt = {
        "status": "complete",
        "recovered": True,
        "reason": feedback,
        "core_tag": core_tag,
        "metric_path": str(metric_path),
        "metric_path_sha256": _sha256(metric_path),
        "calibration_contract": contract,
        "calibration_contract_sha256": _sha256(contract_path),
        "train_stdout_sha256": _sha256(train_stdout_path),
        "train_stderr_sha256": _sha256(train_stderr_path),
        "metrics": metrics,
    }
    _write_json_atomic(receipt_path, receipt)
    recovered_feedback = {
        "status": "complete",
        "recovered": True,
        "core_tag": core_tag,
        "event_file": str(metric_path),
        "recovery_receipt": str(receipt_path),
        "execution_contract": feedback.get("execution_contract"),
    }
    with stderr_path.open("a", encoding="utf-8") as stream:
        stream.write(
            STRUCTURED_FEEDBACK_MARKER
            + " "
            + json.dumps(recovered_feedback, sort_keys=True)
            + "\n"
        )
    with stdout_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
