#!/usr/bin/env python3
"""Run a fail-closed executable smoke test for an autoresearch candidate.

The normal benchmark intentionally exits successfully after reporting an invalid
experiment so GigaEvo can ingest its diagnostics.  A mutation smoke gate needs
the opposite contract: it must exit non-zero unless the structured result says
``is_valid == 1``.  A content-keyed receipt lets the coding agent and harness run
the same command without executing the expensive smoke test twice.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _content_fingerprint(root: Path, paths: list[str]) -> str:
    digest = sha256()
    for relative in sorted(paths):
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/codex-smoke"))
    parser.add_argument(
        "--check-file",
        action="append",
        default=[],
        help="Candidate file included in the reusable smoke receipt fingerprint.",
    )
    return parser


def main() -> int:
    args, benchmark_args = _parser().parse_known_args()
    if args.batches < 1:
        raise SystemExit("--batches must be positive")
    root = Path.cwd().resolve()
    checked_files = args.check_file or ["autoresearch/model/gdn.py"]
    run_dir = args.run_dir.resolve()
    receipt_path = run_dir / "smoke-receipt.json"
    fingerprint = _content_fingerprint(root, checked_files)

    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            receipt = {}
        if (
            receipt.get("content_fingerprint") == fingerprint
            and receipt.get("batches") == args.batches
            and receipt.get("is_valid") == 1.0
        ):
            print(json.dumps(receipt["metrics"], sort_keys=True))
            print(
                "Executable smoke test: PASS (verified cached receipt)", file=sys.stderr
            )
            return 0

    command = [
        sys.executable,
        "-m",
        "autoresearch.benchmark",
        "--smoke-batches",
        str(args.batches),
        "--run-dir",
        str(run_dir),
        *benchmark_args,
    ]
    process = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    print(process.stdout, end="")
    print(process.stderr, end="", file=sys.stderr)
    metrics = _last_json_object(process.stdout)
    valid = float(metrics.get("is_valid", 0.0)) >= 0.5
    if process.returncode != 0 or not valid:
        print(
            "Executable smoke test: FAIL "
            f"(benchmark_exit={process.returncode}, is_valid={metrics.get('is_valid')})",
            file=sys.stderr,
        )
        return 1

    receipt = {
        "batches": args.batches,
        "checked_files": checked_files,
        "content_fingerprint": fingerprint,
        "is_valid": 1.0,
        "metrics": metrics,
    }
    _write_json(receipt_path, receipt)
    print("Executable smoke test: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
