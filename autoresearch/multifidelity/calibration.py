"""Gate calibration under explicit lower-confidence-bound constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist


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


def wilson_lower_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Wilson lower bound for a Bernoulli rate."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = rate + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    )
    return (center - radius) / denominator


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
        lower = wilson_lower_bound(retained, len(winners), confidence)
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
        lower = wilson_lower_bound(len(winners), len(winners), confidence)
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
        lower = wilson_lower_bound(successes, len(promoted), confidence)
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
