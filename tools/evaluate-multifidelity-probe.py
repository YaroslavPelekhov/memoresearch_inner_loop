#!/usr/bin/env python3
"""Test incremental LR-to-zero probe signal on frozen locked-test predictions."""

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
    ProbePredictionOutcome,
    evaluate_probe_incremental_signal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--minimum-groups", type=int, default=20)
    args = parser.parse_args()

    records: list[ProbePredictionOutcome] = []
    with args.predictions.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split") != "locked_test":
                raise ValueError(
                    f"line {line_number} is not from the locked_test split"
                )
            if int(value["budget_batches"]) != args.budget:
                continue
            records.append(
                ProbePredictionOutcome(
                    run_id=str(value["run_id"]),
                    group_id=str(value["group_id"]),
                    final_fitness=float(value["final_fitness"]),
                    trajectory_prediction=float(value["trajectory_prediction"]),
                    probe_prediction=float(value["probe_prediction"]),
                )
            )

    result = evaluate_probe_incremental_signal(
        records,
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        minimum_groups=args.minimum_groups,
    )
    payload = {
        "version": 1,
        "locked_test_predictions": str(args.predictions.resolve()),
        "locked_test_sha256": sha256(args.predictions.read_bytes()).hexdigest(),
        "budget_batches": args.budget,
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
