#!/usr/bin/env python3
"""Stop a diagnostic trainer after a logged batch milestone."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import time

import psutil

BATCH_PATTERN = re.compile(rb"\[batch=(\d+)/\d+\]")


def latest_batch(path: Path, tail_bytes: int = 4 * 1024 * 1024) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - tail_bytes))
        matches = BATCH_PATTERN.findall(handle.read())
    return int(matches[-1]) if matches else None


def matching_torchrun(required_text: list[str]) -> psutil.Process | None:
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if "torchrun" in command and all(text in command for text in required_text):
            return process
    return None


def trainer_child(torchrun: psutil.Process) -> psutil.Process | None:
    try:
        children = torchrun.children(recursive=False)
    except psutil.NoSuchProcess:
        return None
    for child in children:
        try:
            if "train.py" in " ".join(child.cmdline()):
                return child
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stop-after-batch", type=int, required=True)
    parser.add_argument("--match", action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.stop_after_batch < 1:
        parser.error("--stop-after-batch must be positive")

    while True:
        batch = latest_batch(args.log)
        torchrun = matching_torchrun(args.match)
        if torchrun is None:
            print(
                f"{datetime.now().astimezone().isoformat()} trainer already "
                f"exited at batch={batch}"
            )
            return 0
        if batch is not None and batch >= args.stop_after_batch:
            trainer = trainer_child(torchrun)
            if trainer is None:
                time.sleep(args.poll_seconds)
                continue
            print(
                f"{datetime.now().astimezone().isoformat()} stopping trainer "
                f"pid={trainer.pid} at batch={batch}",
                flush=True,
            )
            trainer.terminate()
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
