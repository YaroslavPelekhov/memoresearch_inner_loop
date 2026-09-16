#!/usr/bin/env python3
"""Create frozen group-level statistical splits and winner labels."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.dataset import (
    SPLIT_NAMES,
    assign_group_splits,
    freeze_winner_threshold,
    label_grouped_records,
    split_summary,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            for field in ("run_id", "group_id", "final_fitness"):
                if field not in value:
                    raise ValueError(f"line {line_number} is missing {field}")
            records.append(value)
    if not records:
        raise ValueError("raw dataset is empty")
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winner-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--probability-calibration-fraction", type=float, default=0.15)
    parser.add_argument("--policy-selection-fraction", type=float, default=0.15)
    parser.add_argument("--locked-test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    records = _read_jsonl(args.records)
    fractions = {
        "train": args.train_fraction,
        "probability_calibration": args.probability_calibration_fraction,
        "policy_selection": args.policy_selection_fraction,
        "locked_test": args.locked_test_fraction,
    }
    assignments = assign_group_splits(
        [str(record["group_id"]) for record in records],
        fractions=fractions,
        seed=args.seed,
    )
    threshold = freeze_winner_threshold(
        records,
        assignments,
        winner_fraction=args.winner_fraction,
    )
    labeled = label_grouped_records(
        records,
        assignments,
        winner_threshold=threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split in SPLIT_NAMES:
        path = args.output_dir / f"{split}.jsonl"
        hashes[split] = _write_jsonl(
            path,
            [record for record in labeled if record["split"] == split],
        )
    manifest = {
        "version": 1,
        "source": str(args.records.resolve()),
        "source_sha256": sha256(args.records.read_bytes()).hexdigest(),
        "seed": args.seed,
        "fractions": fractions,
        "winner_definition": {
            "reference_split": "train",
            "winner_fraction": args.winner_fraction,
            "frozen_fitness_threshold": threshold,
        },
        "split_sha256": hashes,
        "summary": split_summary(labeled),
        "group_assignments": assignments,
    }
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
