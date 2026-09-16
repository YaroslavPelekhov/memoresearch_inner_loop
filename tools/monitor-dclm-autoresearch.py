#!/usr/bin/env python3
"""Persist lightweight health snapshots for a long-running DCLM experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 14 * 60 * 60
_CODEX_AUTH_ERROR_PATTERNS = (
    "refresh token was already used",
    "provided authentication token is expired",
    "http 401",
)


def _command(*arguments: str) -> str:
    return subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    return value if isinstance(value, str) else value[0]


def _candidate_snapshot(path: Path) -> dict[str, Any]:
    log_path = path / "train.stderr.log"
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    batch_matches = re.findall(r"\[batch=(\d+)/(\d+)\]", text)
    batch, total_batches = (
        (int(batch_matches[-1][0]), int(batch_matches[-1][1]))
        if batch_matches
        else (None, None)
    )
    completed = "| INFO |  Done." in text and "| DEBUG |  Engine closed." in text
    failed = not completed and bool(
        re.search(
            r"(?:train\.py FAILED|ChildFailedError|CUDA out of memory|"
            r"(?:NotImplementedError|RuntimeError|ValueError):)",
            text,
            re.I,
        )
    )
    failure_matches = re.findall(
        r"(?:\[rank\d+\]:\s*)?"
        r"((?:NotImplementedError|RuntimeError|ValueError):[^\n]+)",
        text,
    )
    status = "completed" if completed else "failed" if failed else "running"
    return {
        "commit": path.name,
        "status": status,
        "batch": batch,
        "total_batches": total_batches,
        "loss": (
            float(value)
            if (
                value := _last_match(
                    r"Train loss/train/total:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "gradient_nonfinite": (
            float(value)
            if (
                value := _last_match(
                    r"Train gradient_clipping/nonfinite:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "gradient_pre_clip": (
            float(value)
            if (
                value := _last_match(
                    r"Train l2_norm/grad/pre_clip_global:\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "gradient_post_clip": (
            float(value)
            if (
                value := _last_match(
                    r"Train l2_norm/grad/post_clip_global:\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "throughput_batches_per_second": (
            float(value)
            if (
                value := _last_match(
                    r"Train throughput/batches_per_sec:\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "remaining_estimate_hours": (
            float(value)
            if (
                value := _last_match(
                    r"Train time/remaining_estimate:\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "core": (
            float(value)
            if (
                value := _last_match(
                    r"Train metrics_gauntlet/core:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                    text,
                )
            )
            is not None
            else None
        ),
        "completed": completed,
        "failed": failed,
        "failure_reason": failure_matches[-1] if failure_matches else None,
        "has_nan_loss": bool(
            re.search(
                r"Train loss/train/(?:total|loss_ce|z_loss):\s*[+-]?nan\b", text, re.I
            )
        ),
        "log_mtime": (
            datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().isoformat()
            if log_path.is_file()
            else None
        ),
    }


def _pending_reviews(run_dir: Path) -> list[dict[str, Any]]:
    review_root = run_dir / "system" / "review"
    pending: list[dict[str, Any]] = []
    if not review_root.is_dir():
        return pending
    for path in sorted(review_root.rglob("*.json")):
        if path.name == "active_idea.json":
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("status") != "pending":
            continue
        pending.append(
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "title": item.get("title")
                or item.get("original_proposal", {}).get("title"),
                "path": str(path),
            }
        )
    return pending


def _mutation_backend_snapshot(run_dir: Path) -> dict[str, Any]:
    """Report definitive Codex auth failures from the latest mutation attempt."""

    logs_root = run_dir / "system" / "repo_mutation" / "logs"
    stderr_paths = list(logs_root.glob("*/agent/stderr.txt"))
    if not stderr_paths:
        return {"status": "unknown", "latest_attempt": None}

    latest = max(stderr_paths, key=lambda path: path.stat().st_mtime)
    text = latest.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    auth_failed = any(pattern in lowered for pattern in _CODEX_AUTH_ERROR_PATTERNS)
    return {
        "status": "auth_failed" if auth_failed else "no_auth_error_observed",
        "latest_attempt": str(latest),
        "latest_attempt_at": datetime.fromtimestamp(latest.stat().st_mtime)
        .astimezone()
        .isoformat(),
        "failure_reason": (
            "Codex authentication expired; log out and sign in again"
            if auth_failed
            else None
        ),
    }


def _gpu_snapshot(index: int) -> dict[str, Any]:
    rows = _command(
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ).splitlines()
    selected = next(
        (row for row in rows if row.split(",", 1)[0].strip() == str(index)), None
    )
    if selected is None:
        return {"index": index, "available": False, "processes": []}
    raw_index, uuid, utilization, used, total, power = (
        value.strip() for value in selected.split(",")
    )
    process_rows = _command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ).splitlines()
    processes = []
    for row in process_rows:
        fields = [value.strip() for value in row.split(",", 3)]
        if len(fields) == 4 and fields[0] == uuid:
            processes.append(
                {
                    "pid": int(fields[1]),
                    "memory_mib": int(fields[2]),
                    "name": fields[3],
                }
            )
    return {
        "index": int(raw_index),
        "available": True,
        "uuid": uuid,
        "utilization_percent": int(utilization),
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "power_watts": float(power),
        "processes": processes,
    }


def _benchmark_elapsed_seconds(run_dir: Path) -> int | None:
    """Return elapsed wall time for the benchmark serving ``run_dir``."""

    rows = _command("ps", "-eo", "etimes=,args=").splitlines()
    expected = str(run_dir.resolve())
    for row in rows:
        fields = row.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        raw_elapsed, command = fields
        match = re.search(r"(?:^|\s)--run-dir(?:=|\s+)(\S+)", command)
        if match is None:
            continue
        try:
            command_run_dir = str(Path(match.group(1)).resolve())
            elapsed = int(raw_elapsed)
        except (OSError, ValueError):
            continue
        if command_run_dir == expected:
            return elapsed
    return None


def _timeout_projection(
    candidate: dict[str, Any],
    *,
    timeout_remaining_seconds: int,
    reference_throughput: float | None,
    observed_at: datetime,
) -> dict[str, Any]:
    """Project when GPU contention must clear for a candidate to finish.

    The projection assumes the candidate keeps its current throughput until
    contention clears, then matches the throughput of a completed candidate
    from the same run. It is deliberately operational metadata only: it does
    not affect candidate validity or fitness.
    """

    batch = candidate.get("batch")
    total_batches = candidate.get("total_batches")
    current_throughput = candidate.get("throughput_batches_per_second")
    if not (
        isinstance(batch, int)
        and isinstance(total_batches, int)
        and total_batches > batch
        and type(current_throughput) in {int, float}
        and current_throughput > 0
        and type(reference_throughput) in {int, float}
        and reference_throughput > 0
    ):
        return {}

    remaining_batches = total_batches - batch
    minimum_required = (
        remaining_batches / timeout_remaining_seconds
        if timeout_remaining_seconds > 0
        else None
    )
    current_completion_seconds = remaining_batches / current_throughput
    reference_completion_seconds = remaining_batches / reference_throughput
    result: dict[str, Any] = {
        "completed_reference_throughput_batches_per_second": round(
            reference_throughput, 4
        ),
        "minimum_required_throughput_batches_per_second": (
            round(minimum_required, 4) if minimum_required is not None else None
        ),
        "current_throughput_completion_seconds": round(current_completion_seconds),
        "current_throughput_meets_timeout": (
            current_completion_seconds <= timeout_remaining_seconds
        ),
        "current_throughput_projected_finish_at": datetime.fromtimestamp(
            observed_at.timestamp() + current_completion_seconds,
            tz=observed_at.tzinfo,
        ).isoformat(),
        "reference_completion_seconds": round(reference_completion_seconds),
        "contention_clearance_headroom_seconds": None,
        "contention_clear_by": None,
    }
    if timeout_remaining_seconds < reference_completion_seconds:
        result["contention_clearance_headroom_seconds"] = 0
        result["contention_clear_by"] = observed_at.isoformat()
        return result
    if current_throughput >= reference_throughput:
        return result

    # t + (remaining - current_rate * t) / reference_rate <= timeout_remaining
    headroom = (timeout_remaining_seconds - reference_completion_seconds) / (
        1 - current_throughput / reference_throughput
    )
    headroom = max(0.0, min(float(timeout_remaining_seconds), headroom))
    result["contention_clearance_headroom_seconds"] = round(headroom)
    result["contention_clear_by"] = datetime.fromtimestamp(
        observed_at.timestamp() + headroom,
        tz=observed_at.tzinfo,
    ).isoformat()
    return result


def snapshot(
    run_dir: Path,
    gpu_index: int,
    benchmark_timeout_seconds: int = DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    candidates_root = run_dir / "training" / "candidates"
    candidate_paths = (
        [path for path in sorted(candidates_root.iterdir()) if path.is_dir()]
        if candidates_root.is_dir()
        else []
    )
    candidates = [_candidate_snapshot(path) for path in candidate_paths]
    reference_throughput = max(
        (
            float(candidate["throughput_batches_per_second"])
            for candidate in candidates
            if candidate["completed"]
            and type(candidate.get("throughput_batches_per_second")) in {int, float}
            and candidate["throughput_batches_per_second"] > 0
        ),
        default=None,
    )
    observed_at = datetime.now().astimezone()
    for path, candidate in zip(candidate_paths, candidates):
        if candidate["status"] != "running":
            continue
        elapsed = _benchmark_elapsed_seconds(path)
        if elapsed is None:
            continue
        timeout_remaining = max(0, benchmark_timeout_seconds - elapsed)
        eta_hours = candidate["remaining_estimate_hours"]
        candidate.update(
            {
                "benchmark_elapsed_seconds": elapsed,
                "benchmark_timeout_seconds": benchmark_timeout_seconds,
                "benchmark_timeout_remaining_seconds": timeout_remaining,
                "eta_exceeds_benchmark_timeout": (
                    eta_hours is not None and eta_hours * 60 * 60 > timeout_remaining
                ),
            }
        )
        candidate.update(
            _timeout_projection(
                candidate,
                timeout_remaining_seconds=timeout_remaining,
                reference_throughput=reference_throughput,
                observed_at=observed_at,
            )
        )
    sessions = _command("tmux", "list-sessions", "-F", "#{session_name}").splitlines()
    contract_path = run_dir / "handoff-contract.json"
    contract = (
        json.loads(contract_path.read_text(encoding="utf-8"))
        if contract_path.is_file()
        else {}
    )
    deadline_epoch = contract.get("deadline_epoch")
    return {
        "observed_at": observed_at.isoformat(),
        "run_dir": str(run_dir),
        "deadline_epoch": deadline_epoch,
        "deadline_remaining_seconds": (
            max(0, int(deadline_epoch) - int(observed_at.timestamp()))
            if isinstance(deadline_epoch, int)
            else None
        ),
        "handoff_tmux_alive": "calibrated-autoresearch-handoff" in sessions,
        "lease_tmux_alive": f"gpu{gpu_index}-lease" in sessions,
        "gpu": _gpu_snapshot(gpu_index),
        "candidates": candidates,
        "pending_reviews": _pending_reviews(run_dir),
        "mutation_backend": _mutation_backend_snapshot(run_dir),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument(
        "--benchmark-timeout-seconds",
        type=int,
        default=DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.benchmark_timeout_seconds <= 0:
        parser.error("--benchmark-timeout-seconds must be positive")

    run_dir = args.run_dir.resolve()
    monitor_dir = run_dir / "monitor"
    previous_digest: str | None = None
    reported_date: str | None = None
    while True:
        try:
            value = snapshot(
                run_dir,
                args.gpu_index,
                benchmark_timeout_seconds=args.benchmark_timeout_seconds,
            )
            _write_json_atomic(monitor_dir / "latest.json", value)
            state = dict(value)
            state.pop("observed_at", None)
            digest = hashlib.sha256(
                json.dumps(state, sort_keys=True).encode()
            ).hexdigest()
            today = datetime.now().astimezone().date().isoformat()
            if digest != previous_digest:
                print(
                    json.dumps({"event": "state_change", **value}, sort_keys=True),
                    flush=True,
                )
                previous_digest = digest
            if today != reported_date:
                print(
                    json.dumps({"event": "daily_report", **value}, sort_keys=True),
                    flush=True,
                )
                reported_date = today
            # Keep the daily artifact current throughout the day while emitting
            # only one daily_report event per monitor process and calendar day.
            _write_json_atomic(monitor_dir / "daily" / f"{today}.json", value)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "monitor_error",
                        "observed_at": datetime.now().astimezone().isoformat(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
