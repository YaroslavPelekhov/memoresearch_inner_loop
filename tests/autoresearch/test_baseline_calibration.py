from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = runpy.run_path(str(ROOT / "tools/run-baseline-calibration.py"))


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_all_calibration_horizons_use_25_percent_warmup() -> None:
    assert CALIBRATION["DIAGNOSTIC_BATCHES"] == 3_300
    assert CALIBRATION["DIAGNOSTIC_WARMUP"] == 825
    assert CALIBRATION["SCREEN_BATCHES"] == 30_518
    assert CALIBRATION["SCREEN_WARMUP"] == 7_630
    assert CALIBRATION["CONFIRM_BATCHES"] == 87_715
    assert CALIBRATION["CONFIRM_WARMUP"] == 21_929


def test_full_screen_retains_every_finite_upper_diagnostic_lr() -> None:
    assert CALIBRATION["_screen_lrs"]([1.625e-3, 1.75e-3]) == (
        1.25e-3,
        1.5e-3,
        1.625e-3,
        1.75e-3,
    )


def test_failure_triggered_full_screen_lrs_descend_conservatively() -> None:
    assert CALIBRATION["FALLBACK_FULL_LR_GROUPS"] == (
        (1.0e-3, 0.75e-3),
        (0.5e-3,),
    )


def test_calibration_rechecks_frozen_source_before_each_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    tracked = source / "tracked.txt"
    tracked.write_text("fixed\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixed source",
    )
    commit = _git(source, "rev-parse", "HEAD")
    (tmp_path / "calibration-suite.json").write_text(
        '{"commit":"' + commit + '"}\n', encoding="utf-8"
    )

    assert CALIBRATION["_assert_fixed_source"](source, tmp_path) == commit

    tracked.write_text("contaminated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="became dirty"):
        CALIBRATION["_assert_fixed_source"](source, tmp_path)


def test_calibration_summary_is_published_atomically(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    CALIBRATION["_write_json_atomic"](summary_path, {"selected_lr": 0.0015})

    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "selected_lr": 0.0015
    }
    assert not (tmp_path / "summary.json.part").exists()
