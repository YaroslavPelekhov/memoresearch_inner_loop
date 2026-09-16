#!/usr/bin/env python3
"""Create a hash-bound receipt for a baseline that finished before teardown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.completed_run import build_receipt


def _write_json_atomic(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--adapter-stdout", type=Path, required=True)
    parser.add_argument("--calibrated-commit", required=True)
    parser.add_argument("--operational-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        args.run_dir,
        adapter_stdout=args.adapter_stdout,
        calibrated_commit=args.calibrated_commit,
        operational_commit=args.operational_commit,
    )
    _write_json_atomic(args.output, receipt)
    print(json.dumps(receipt["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
