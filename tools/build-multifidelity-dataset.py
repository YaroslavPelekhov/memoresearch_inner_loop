#!/usr/bin/env python3
"""Build a raw rung dataset from complete observe-only trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.dataset import infer_group_id
from autoresearch.multifidelity.models import RungObservation
from autoresearch.multifidelity.state import load_observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-budget", type=int, default=4096)
    args = parser.parse_args()

    complete: list[tuple[Path, list[RungObservation], float]] = []
    for path in sorted(args.root.rglob("multifidelity-trajectory.json")):
        observations = load_observations(path)
        if not observations or observations[-1].budget_batches != args.final_budget:
            continue
        final_fitness = observations[-1].features.main_fitness
        if math.isfinite(final_fitness):
            complete.append((path, observations, final_fitness))
    if not complete:
        parser.error("no complete trajectories found")

    records: list[dict[str, object]] = []
    for path, observations, final_fitness in complete:
        group_id = infer_group_id(args.root, path)
        for observation in observations[:-1]:
            records.append(
                {
                    "run_id": str(path.parent),
                    "group_id": group_id,
                    "trajectory_path": str(path),
                    "commit": observation.commit,
                    "budget_batches": observation.budget_batches,
                    **observation.features.model_dump(mode="json"),
                    "final_fitness": final_fitness,
                    "plan_sha256": observation.metadata.get("plan_sha256"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "complete_trajectories": len(complete),
                "groups": len({record["group_id"] for record in records}),
                "rung_records": len(records),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
