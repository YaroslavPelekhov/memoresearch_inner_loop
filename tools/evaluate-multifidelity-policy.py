#!/usr/bin/env python3
"""Evaluate a frozen policy on an untouched locked-test outcome file."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.evaluation import (
    PolicyOutcome,
    evaluate_locked_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.95)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=29)
    parser.add_argument("--minimum-groups", type=int, default=20)
    parser.add_argument("--minimum-winner-groups", type=int, default=20)
    args = parser.parse_args()

    records: list[PolicyOutcome] = []
    with args.outcomes.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split") != "locked_test":
                raise ValueError(
                    f"line {line_number} is not from the locked_test split"
                )
            records.append(
                PolicyOutcome(
                    run_id=str(value["run_id"]),
                    group_id=str(value["group_id"]),
                    eventual_winner=value["eventual_winner"],
                    survived=value["survived"],
                    full_compute=float(value["full_compute"]),
                    cascade_compute=float(value["cascade_compute"]),
                )
            )
    if any(
        not isinstance(record.eventual_winner, bool)
        or not isinstance(record.survived, bool)
        for record in records
    ):
        raise ValueError("eventual_winner and survived must be booleans")

    result = evaluate_locked_policy(
        records,
        recall_floor=args.recall_floor,
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        minimum_groups=args.minimum_groups,
        minimum_winner_groups=args.minimum_winner_groups,
    )
    payload = {
        "version": 1,
        "locked_test_outcomes": str(args.outcomes.resolve()),
        "locked_test_sha256": sha256(args.outcomes.read_bytes()).hexdigest(),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
