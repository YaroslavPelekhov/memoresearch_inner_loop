#!/usr/bin/env python3
"""Monitor and restart a detached supervised idea campaign when necessary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def _tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _gpu() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        utilization, used, total = (
            int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
        )
    except (ValueError, IndexError):
        return {"available": False}
    return {
        "available": True,
        "utilization_percent": utilization,
        "memory_used_mib": used,
        "memory_total_mib": total,
    }


def _state(root: Path) -> dict[str, object]:
    path = root / "campaign-state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _latest_progress(root: Path) -> float:
    ignored = {"watchdog.jsonl", "supervisor.lock"}
    mtimes = [
        path.stat().st_mtime
        for path in root.rglob("*")
        if path.is_file() and path.name not in ignored
    ]
    return max(mtimes, default=0.0)


def _launch(
    session: str, cwd: Path, command: list[str], environment: dict[str, str]
) -> None:
    shell_command = shlex.join(
        ["env", *(f"{key}={value}" for key, value in environment.items()), *command]
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(cwd), shell_command],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=3600)
    args = parser.parse_args()
    root = args.campaign_root.expanduser().resolve()
    specification = json.loads(args.command_json.read_text(encoding="utf-8"))
    command = [str(value) for value in specification["command"]]
    environment = {
        str(key): str(value)
        for key, value in specification.get("environment", {}).items()
    }
    cwd = Path(specification["cwd"])
    target_rounds = int(specification["target_rounds"])
    log_path = root / "watchdog.jsonl"

    while True:
        state = _state(root)
        completed = int(state.get("completed_rounds", 0))
        alive = _tmux_alive(args.session)
        gpu = _gpu()
        event = "observed"
        if completed >= target_rounds:
            event = "target_complete"
        elif not alive:
            _launch(args.session, cwd, command, environment)
            alive = True
            event = "restarted_dead_supervisor"
        elif (
            gpu.get("utilization_percent") == 0
            and time.time() - _latest_progress(root) > args.stale_seconds
        ):
            subprocess.run(["tmux", "kill-session", "-t", args.session], check=False)
            _launch(args.session, cwd, command, environment)
            event = "restarted_stale_supervisor"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "epoch": time.time(),
                        "event": event,
                        "completed_rounds": completed,
                        "campaign_status": state.get("status"),
                        "supervisor_alive": alive,
                        "gpu": gpu,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if completed >= target_rounds:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
