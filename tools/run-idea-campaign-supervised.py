#!/usr/bin/env python3
"""Run one idea round at a time until a campaign reaches a target count."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _state(root: Path) -> dict[str, object]:
    path = root / "campaign-state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _record(root: Path, **event: object) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": time.time(), **event}, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--target-rounds", type=int, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--evolution-generations", type=int, default=4)
    parser.add_argument("--retry-limit", type=int, default=3)
    args = parser.parse_args()
    root = args.campaign_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.target_rounds < 1:
        parser.error("--target-rounds must be positive")

    supervisor_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--campaign-root",
        str(root),
        "--target-rounds",
        str(args.target_rounds),
        "--model",
        args.model,
        "--evolution-generations",
        str(args.evolution_generations),
        "--retry-limit",
        str(args.retry_limit),
    ]
    (root / "launch-spec.json").write_text(
        json.dumps(
            {
                "cwd": str(Path.cwd()),
                "command": supervisor_command,
                "environment": {
                    name: os.environ[name]
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "AUTORESEARCH_PYTHON",
                        "DCLM_MODEL_PATH",
                        "DCLM_MDS_PATH",
                        "DCLM_TRAIN_SPLIT",
                        "DCLM_SCREEN_MDS_PATH",
                        "DCLM_SCREEN_VALIDATION_SPLIT",
                        "DCLM_CORE_PATH",
                        "DCLM_EVAL_CACHE_PATH",
                        "AUTORESEARCH_LR",
                        "AUTORESEARCH_EXPERIMENT_TIMEOUT_SECONDS",
                        "AUTORESEARCH_BENCHMARK_TIMEOUT",
                        "AUTORESEARCH_STAGE_TIMEOUT",
                        "AUTORESEARCH_DAG_TIMEOUT",
                        "CODEX_MODEL",
                        "CODEX_SERVICE_TIER",
                    )
                    if name in os.environ
                },
                "target_rounds": args.target_rounds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with (root / "supervisor.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another campaign supervisor already holds the lock", file=sys.stderr)
            return 2

        consecutive_failures = 0
        while int(_state(root).get("completed_rounds", 0)) < args.target_rounds:
            before = int(_state(root).get("completed_rounds", 0))
            command = [
                sys.executable,
                "-m",
                "autoresearch.ideas.campaign",
                "--campaign-root",
                str(root),
                "--rounds",
                "1",
                "--filter-mode",
                "codex",
                "--model",
                args.model,
                "--evolution-generations",
                str(args.evolution_generations),
            ]
            _record(root, event="round_start", completed_rounds=before, command=command)
            result = subprocess.run(command, check=False)
            after = int(_state(root).get("completed_rounds", 0))
            _record(
                root,
                event="round_exit",
                returncode=result.returncode,
                completed_rounds=after,
            )
            if result.returncode == 0 and after > before:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures > args.retry_limit:
                _record(root, event="retry_limit_reached")
                return 1
            time.sleep(30)

        _record(root, event="target_complete", completed_rounds=args.target_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
