"""Locked-test evaluation for a frozen multi-fidelity policy."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from autoresearch.multifidelity.statistics import (
    clopper_pearson_lower,
    paired_cluster_bootstrap,
)


@dataclass(frozen=True)
class PolicyOutcome:
    run_id: str
    group_id: str
    eventual_winner: bool
    survived: bool
    full_compute: float
    cascade_compute: float


@dataclass(frozen=True)
class ProbePredictionOutcome:
    run_id: str
    group_id: str
    final_fitness: float
    trajectory_prediction: float
    probe_prediction: float


def evaluate_probe_incremental_signal(
    records: list[ProbePredictionOutcome],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 17,
    minimum_groups: int = 20,
) -> dict[str, object]:
    """Paired one-sided test of incremental probe signal on locked runs."""

    if not records:
        raise ValueError("probe evaluation contains no outcomes")
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("probe evaluation contains duplicate run ids")
    if minimum_groups < 2:
        raise ValueError("minimum_groups must be at least two")
    differences = [
        (record.trajectory_prediction - record.final_fitness) ** 2
        - (record.probe_prediction - record.final_fitness) ** 2
        for record in records
    ]
    interval = paired_cluster_bootstrap(
        differences,
        [record.group_id for record in records],
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        interval="one_sided",
    )
    groups = len({record.group_id for record in records})
    enough_groups = groups >= minimum_groups
    return {
        "runs": len(records),
        "groups": groups,
        "endpoint": "paired_squared_error_improvement",
        "effect": interval.estimate,
        "one_sided_lower": interval.lower,
        "confidence": confidence,
        "minimum_groups": minimum_groups,
        "enough_groups": enough_groups,
        "incremental_signal_supported": enough_groups and interval.lower > 0.0,
    }


def _metrics(records: list[PolicyOutcome]) -> tuple[float, float]:
    winners = [record for record in records if record.eventual_winner]
    recall = (
        sum(record.survived for record in winners) / len(winners)
        if winners
        else float("nan")
    )
    full_compute = sum(record.full_compute for record in records)
    if full_compute <= 0.0:
        raise ValueError("full compute must be positive")
    cascade_compute = sum(record.cascade_compute for record in records)
    return recall, 1.0 - cascade_compute / full_compute


def evaluate_locked_policy(
    records: list[PolicyOutcome],
    *,
    recall_floor: float = 0.95,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 29,
    minimum_groups: int = 20,
    minimum_winner_groups: int = 20,
) -> dict[str, object]:
    """Evaluate one frozen policy without tuning on the locked-test outcomes."""

    if not 0.0 < recall_floor <= 1.0:
        raise ValueError("recall_floor must lie in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if minimum_groups < 2:
        raise ValueError("minimum_groups must be at least two")
    if minimum_winner_groups < 2:
        raise ValueError("minimum_winner_groups must be at least two")
    if not records:
        raise ValueError("locked test contains no outcomes")
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("locked test contains duplicate run ids")
    if any(
        record.full_compute <= 0.0 or record.cascade_compute < 0.0 for record in records
    ):
        raise ValueError("compute values are outside the admissible range")

    groups: dict[str, list[PolicyOutcome]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    if len(groups) < 2:
        raise ValueError("locked test requires at least two independent groups")
    winners = [record for record in records if record.eventual_winner]
    if not winners:
        raise ValueError("locked test contains no eventual winners")
    winner_groups = len({record.group_id for record in winners})
    survivors = sum(record.survived for record in winners)
    recall, compute_saving = _metrics(records)
    exact_lower = clopper_pearson_lower(survivors, len(winners), confidence)

    group_ids = sorted(groups)
    rng = random.Random(seed)
    recall_samples: list[float] = []
    compute_samples: list[float] = []
    for _ in range(resamples):
        sample: list[PolicyOutcome] = []
        for _ in group_ids:
            sample.extend(groups[rng.choice(group_ids)])
        sample_recall, sample_compute = _metrics(sample)
        if not np.isnan(sample_recall):
            recall_samples.append(sample_recall)
        compute_samples.append(sample_compute)
    if len(recall_samples) < resamples // 2:
        raise ValueError("too few bootstrap samples contain eventual winners")
    alpha = 1.0 - confidence
    cluster_recall_lower = float(np.quantile(recall_samples, alpha))
    compute_saving_lower = float(np.quantile(compute_samples, alpha))
    enough_groups = len(groups) >= minimum_groups
    enough_winner_groups = winner_groups >= minimum_winner_groups
    return {
        "runs": len(records),
        "groups": len(groups),
        "winners": len(winners),
        "winner_groups": winner_groups,
        "surviving_winners": survivors,
        "winner_recall": recall,
        "winner_recall_exact_lower": exact_lower,
        "winner_recall_cluster_bootstrap_lower": cluster_recall_lower,
        "compute_saving": compute_saving,
        "compute_saving_cluster_bootstrap_lower": compute_saving_lower,
        "confidence": confidence,
        "recall_floor": recall_floor,
        "minimum_groups": minimum_groups,
        "minimum_winner_groups": minimum_winner_groups,
        "enough_groups": enough_groups,
        "enough_winner_groups": enough_winner_groups,
        "certified": (
            enough_groups
            and enough_winner_groups
            and exact_lower >= recall_floor
            and cluster_recall_lower >= recall_floor
            and compute_saving_lower > 0.0
        ),
    }
