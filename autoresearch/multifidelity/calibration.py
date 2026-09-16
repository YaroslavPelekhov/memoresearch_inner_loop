"""Gate calibration under explicit lower-confidence-bound constraints."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.multifidelity.statistics import clopper_pearson_lower


@dataclass(frozen=True)
class LabeledProbability:
    probability: float
    eventual_winner: bool
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class CalibratedGate:
    threshold: float
    successes: int
    trials: int
    observed_rate: float
    lower_confidence_bound: float


def calibrate_kill_gate(
    records: list[LabeledProbability],
    *,
    recall_floor: float = 0.95,
    confidence: float = 0.95,
) -> CalibratedGate:
    """Choose the largest safe kill threshold by winner-recall lower bound."""

    winners = [record for record in records if record.eventual_winner]
    if not winners:
        raise ValueError("gate calibration requires eventual winners")
    scores = [
        record.upper if record.upper is not None else record.probability
        for record in records
    ]
    winner_scores = [
        record.upper if record.upper is not None else record.probability
        for record in winners
    ]
    candidates = sorted({0.0, *scores})
    safe: list[CalibratedGate] = []
    for threshold in candidates:
        retained = sum(score >= threshold for score in winner_scores)
        lower = clopper_pearson_lower(retained, len(winners), confidence)
        if lower >= recall_floor:
            safe.append(
                CalibratedGate(
                    threshold=threshold,
                    successes=retained,
                    trials=len(winners),
                    observed_rate=retained / len(winners),
                    lower_confidence_bound=lower,
                )
            )
    if not safe:
        lower = clopper_pearson_lower(len(winners), len(winners), confidence)
        raise ValueError(
            "insufficient winners to certify the recall floor: "
            f"best lower bound is {lower:.6f}"
        )
    return max(safe, key=lambda gate: gate.threshold)


def calibrate_promote_gate(
    records: list[LabeledProbability],
    *,
    precision_floor: float = 0.80,
    confidence: float = 0.95,
) -> CalibratedGate:
    """Choose the smallest promotion threshold with supported precision."""

    scores = [
        record.lower if record.lower is not None else record.probability
        for record in records
    ]
    candidates = sorted({0.0, 1.0, *scores})
    safe: list[CalibratedGate] = []
    for threshold in candidates:
        promoted = [
            record
            for record, score in zip(records, scores, strict=True)
            if score > threshold
        ]
        successes = sum(record.eventual_winner for record in promoted)
        lower = clopper_pearson_lower(successes, len(promoted), confidence)
        if promoted and lower >= precision_floor:
            safe.append(
                CalibratedGate(
                    threshold=threshold,
                    successes=successes,
                    trials=len(promoted),
                    observed_rate=successes / len(promoted),
                    lower_confidence_bound=lower,
                )
            )
    if not safe:
        raise ValueError(
            "no promotion threshold certifies the requested precision floor"
        )
    return min(safe, key=lambda gate: gate.threshold)
