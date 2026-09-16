#!/usr/bin/env python3
"""Reuse one verified seed result; benchmark every descendant normally."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.completed_run import file_sha256, verify_receipt

STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    head = _git_head()
    baseline_commit = os.environ.get("AUTORESEARCH_REUSED_BASELINE_COMMIT")
    if baseline_commit and head == baseline_commit:
        receipt_path = Path(os.environ["AUTORESEARCH_BASELINE_RECEIPT"]).resolve()
        expected_receipt_sha256 = os.environ["AUTORESEARCH_BASELINE_RECEIPT_SHA256"]
        actual_receipt_sha256 = file_sha256(receipt_path)
        if actual_receipt_sha256 != expected_receipt_sha256:
            raise RuntimeError("Completed baseline receipt changed after handoff")
        receipt = verify_receipt(
            receipt_path,
            calibrated_commit=os.environ["AUTORESEARCH_CALIBRATED_COMMIT"],
            operational_commit=head,
        )
        feedback = {
            "status": "complete",
            "reused_completed_baseline": True,
            "receipt": str(receipt_path),
            "receipt_sha256": actual_receipt_sha256,
            "completion_evidence": receipt["completion_evidence"],
            "core_tag": receipt["core_tag"],
        }
        print(
            STRUCTURED_FEEDBACK_MARKER + " " + json.dumps(feedback, sort_keys=True),
            file=sys.stderr,
        )
        print(json.dumps(receipt["metrics"], sort_keys=True))
        return 0
    command = [sys.executable, "-m", "autoresearch.benchmark", *sys.argv[1:]]
    os.execvpe(command[0], command, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
