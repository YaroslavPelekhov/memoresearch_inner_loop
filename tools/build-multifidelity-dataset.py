#!/usr/bin/env python3
"""Build rung-level eventual-winner labels from complete trajectory files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.models import RungObservation
from autoresearch.multifidelity.state import load_observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-budget", type=int, default=4096)
    parser.add_argument("--winner-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.winner_fraction <= 1.0:
        parser.error("--winner-fraction must lie in (0, 1]")

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

    winner_count = max(1, math.ceil(len(complete) * args.winner_fraction))
    ranked = sorted(
        enumerate(complete),
        key=lambda item: (-item[1][2], str(item[1][0])),
    )
    winner_indexes = {index for index, _ in ranked[:winner_count]}
    records: list[dict[str, object]] = []
    for index, (path, observations, final_fitness) in enumerate(complete):
        for observation in observations[:-1]:
            records.append(
                {
                    "run_id": str(path.parent),
                    "trajectory_path": str(path),
                    "commit": observation.commit,
                    "budget_batches": observation.budget_batches,
                    **observation.features.model_dump(mode="json"),
                    "final_fitness": final_fitness,
                    "eventual_winner": index in winner_indexes,
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
                "eventual_winners": winner_count,
                "rung_records": len(records),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
