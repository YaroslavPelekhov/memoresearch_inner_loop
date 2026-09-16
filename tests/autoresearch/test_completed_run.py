from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.completed_run import (
    KIND,
    SCHEMA_VERSION,
    file_sha256,
    verify_receipt,
)


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _receipt(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    files = {}
    for name in ("effective_config", "train_stdout", "train_stderr", "event_file"):
        path = run_dir / name
        path.write_text(f"immutable {name}\n", encoding="utf-8")
        files[name] = _record(path, run_dir)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "calibrated_commit": "calibrated",
        "operational_commit": "operational",
        "run_dir": str(run_dir),
        "metrics": {
            "fitness": 0.35,
            "is_valid": 1.0,
            "llmfoundry_core_equal_raw": 0.35,
            "n_params": 143_710_848.0,
        },
        "files": files,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_verify_receipt_accepts_exact_files_and_commits(tmp_path: Path) -> None:
    path = _receipt(tmp_path)

    receipt = verify_receipt(
        path,
        calibrated_commit="calibrated",
        operational_commit="operational",
    )

    assert receipt["metrics"]["fitness"] == 0.35


def test_verify_receipt_rejects_changed_completed_run(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    changed = Path(receipt["run_dir"]) / "train_stderr"
    changed.write_text("changed after receipt\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="train_stderr changed"):
        verify_receipt(
            path,
            calibrated_commit="calibrated",
            operational_commit="operational",
        )


def test_verify_receipt_rejects_a_different_operational_commit(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)

    with pytest.raises(RuntimeError, match="operational_commit"):
        verify_receipt(
            path,
            calibrated_commit="calibrated",
            operational_commit="candidate",
        )
