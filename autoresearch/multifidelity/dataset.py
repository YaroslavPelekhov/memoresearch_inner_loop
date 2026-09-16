"""Leakage-resistant grouped dataset construction."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
from typing import Any

SPLIT_NAMES = (
    "train",
    "probability_calibration",
    "policy_selection",
    "locked_test",
)


def infer_group_id(root: Path, trajectory_path: Path) -> str:
    """Group descendants from one idea version/lineage together."""

    relative = trajectory_path.resolve().relative_to(root.resolve())
    parts = relative.parts
    if "implementations" in parts:
        boundary = parts.index("implementations")
        if boundary:
            return "/".join(parts[:boundary])
    return relative.parent.as_posix()


def assign_group_splits(
    group_ids: list[str],
    *,
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each complete lineage group to exactly one statistical split."""

    groups = sorted(set(group_ids))
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError(f"at least {len(SPLIT_NAMES)} groups are required")
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions must define exactly {SPLIT_NAMES}")
    if any(value <= 0.0 for value in fractions.values()) or not math.isclose(
        sum(fractions.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("split fractions must be positive and sum to one")

    exact = {name: len(groups) * fractions[name] for name in SPLIT_NAMES}
    counts = {name: max(1, math.floor(exact[name])) for name in SPLIT_NAMES}
    while sum(counts.values()) > len(groups):
        donor = max(
            (name for name in SPLIT_NAMES if counts[name] > 1),
            key=lambda name: (counts[name] - exact[name], counts[name]),
        )
        counts[donor] -= 1
    while sum(counts.values()) < len(groups):
        receiver = max(
            SPLIT_NAMES,
            key=lambda name: (exact[name] - counts[name], -counts[name]),
        )
        counts[receiver] += 1

    random.Random(seed).shuffle(groups)
    assignments: dict[str, str] = {}
    offset = 0
    for name in SPLIT_NAMES:
        for group_id in groups[offset : offset + counts[name]]:
            assignments[group_id] = name
        offset += counts[name]
    return assignments


def freeze_winner_threshold(
    records: list[dict[str, Any]],
    assignments: dict[str, str],
    *,
    winner_fraction: float,
) -> float:
    """Derive the winner threshold from unique train runs only."""

    if not 0.0 < winner_fraction <= 1.0:
        raise ValueError("winner_fraction must lie in (0, 1]")
    train_fitness_by_run: dict[str, float] = {}
    for record in records:
        group_id = str(record["group_id"])
        if assignments[group_id] != "train":
            continue
        run_id = str(record["run_id"])
        fitness = float(record["final_fitness"])
        previous = train_fitness_by_run.setdefault(run_id, fitness)
        if previous != fitness:
            raise ValueError(f"inconsistent final fitness for run {run_id}")
    if not train_fitness_by_run:
        raise ValueError("train split contains no complete runs")
    ranked = sorted(train_fitness_by_run.values(), reverse=True)
    winner_count = max(1, math.ceil(len(ranked) * winner_fraction))
    return ranked[winner_count - 1]


def label_grouped_records(
    records: list[dict[str, Any]],
    assignments: dict[str, str],
    *,
    winner_threshold: float,
) -> list[dict[str, Any]]:
    """Apply one frozen reference threshold to every non-train split."""

    labeled: list[dict[str, Any]] = []
    for record in records:
        group_id = str(record["group_id"])
        if group_id not in assignments:
            raise ValueError(f"group has no split assignment: {group_id}")
        labeled.append(
            {
                **record,
                "split": assignments[group_id],
                "winner_threshold": winner_threshold,
                "eventual_winner": float(record["final_fitness"]) >= winner_threshold,
            }
        )
    return labeled


def split_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        selected = [record for record in records if record["split"] == split]
        runs = {str(record["run_id"]) for record in selected}
        winners = {
            str(record["run_id"]) for record in selected if record["eventual_winner"]
        }
        groups = Counter(str(record["group_id"]) for record in selected)
        summary[split] = {
            "groups": len(groups),
            "runs": len(runs),
            "winners": len(winners),
            "rung_records": len(selected),
        }
    return summary
