from __future__ import annotations

from datetime import datetime
import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "monitor-dclm-autoresearch.py"
SPEC = importlib.util.spec_from_file_location("monitor_dclm_autoresearch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_benchmark_elapsed_seconds_matches_exact_run_dir(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    other = tmp_path / "run-other"
    monkeypatch.setattr(
        MONITOR,
        "_command",
        lambda *_args: (
            f"900 python -m autoresearch.benchmark --run-dir {other}\n"
            f"123 python -m autoresearch.benchmark --run-dir={run_dir}"
        ),
    )

    assert MONITOR._benchmark_elapsed_seconds(run_dir) == 123


def test_snapshot_reports_timeout_risk(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    candidate = run_dir / "training" / "candidates" / "abc123"
    candidate.mkdir(parents=True)
    (candidate / "train.stderr.log").write_text(
        """[batch=20/30518]:
 Train loss/train/total: 9.5000
 Train time/remaining_estimate: 2.0000
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(MONITOR, "_benchmark_elapsed_seconds", lambda _path: 100)
    monkeypatch.setattr(MONITOR, "_command", lambda *_args: "")

    value = MONITOR.snapshot(
        run_dir,
        gpu_index=0,
        benchmark_timeout_seconds=1_000,
    )

    observed = value["candidates"][0]
    assert observed["benchmark_elapsed_seconds"] == 100
    assert observed["benchmark_timeout_remaining_seconds"] == 900
    assert observed["eta_exceeds_benchmark_timeout"] is True


def test_mutation_backend_reports_latest_auth_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    older = run_dir / "system" / "repo_mutation" / "logs" / "older" / "agent"
    latest = run_dir / "system" / "repo_mutation" / "logs" / "latest" / "agent"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    older_stderr = older / "stderr.txt"
    latest_stderr = latest / "stderr.txt"
    older_stderr.write_text("successful mutation\n", encoding="utf-8")
    latest_stderr.write_text(
        "ERROR: Your refresh token was already used. Please log out.\n",
        encoding="utf-8",
    )
    os.utime(older_stderr, (1, 1))
    os.utime(latest_stderr, (2, 2))

    value = MONITOR._mutation_backend_snapshot(run_dir)

    assert value["status"] == "auth_failed"
    assert value["latest_attempt"] == str(latest_stderr)
    assert value["failure_reason"] == (
        "Codex authentication expired; log out and sign in again"
    )


def test_mutation_backend_is_unknown_without_attempts(tmp_path: Path) -> None:
    assert MONITOR._mutation_backend_snapshot(tmp_path) == {
        "status": "unknown",
        "latest_attempt": None,
    }


def test_timeout_projection_reports_latest_safe_clearance() -> None:
    observed_at = datetime.fromisoformat("2026-08-28T12:00:00+00:00")

    value = MONITOR._timeout_projection(
        {
            "batch": 1_000,
            "total_batches": 31_000,
            "throughput_batches_per_second": 0.25,
        },
        timeout_remaining_seconds=50_000,
        reference_throughput=2.5,
        observed_at=observed_at,
    )

    assert value == {
        "completed_reference_throughput_batches_per_second": 2.5,
        "minimum_required_throughput_batches_per_second": 0.6,
        "current_throughput_completion_seconds": 120_000,
        "current_throughput_meets_timeout": False,
        "current_throughput_projected_finish_at": "2026-08-29T21:20:00+00:00",
        "reference_completion_seconds": 12_000,
        "contention_clearance_headroom_seconds": 42_222,
        "contention_clear_by": "2026-08-28T23:43:42.222222+00:00",
    }


def test_timeout_projection_reports_when_reference_cannot_finish() -> None:
    observed_at = datetime.fromisoformat("2026-08-28T12:00:00+00:00")

    value = MONITOR._timeout_projection(
        {
            "batch": 1_000,
            "total_batches": 31_000,
            "throughput_batches_per_second": 0.25,
        },
        timeout_remaining_seconds=10_000,
        reference_throughput=2.5,
        observed_at=observed_at,
    )

    assert value["contention_clearance_headroom_seconds"] == 0
    assert value["contention_clear_by"] == observed_at.isoformat()
