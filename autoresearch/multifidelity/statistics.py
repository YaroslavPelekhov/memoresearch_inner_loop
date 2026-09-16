"""Statistical primitives for locked multi-fidelity policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta CDF."""

    max_iterations = 300
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        coefficient = -(
            (a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        update = d * c
        result *= update
        if abs(update - 1.0) <= epsilon:
            return result
    raise ArithmeticError("regularized beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    scale = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _beta_continued_fraction(a, b, x) / a
    return 1.0 - scale * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_ppf(probability: float, a: float, b: float) -> float:
    """Invert the beta CDF by deterministic bisection."""

    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if _regularized_beta(midpoint, a, b) < probability:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= 1.0e-13:
            break
    return (lower + upper) / 2.0


def clopper_pearson_lower(
    successes: int, trials: int, confidence: float = 0.95
) -> float:
    """Exact one-sided lower confidence bound for a Bernoulli rate."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0 or successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    return _beta_ppf(alpha, successes, trials - successes + 1)


def clopper_pearson_upper(
    successes: int, trials: int, confidence: float = 0.95
) -> float:
    """Exact one-sided upper confidence bound for a Bernoulli rate."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return 1.0
    if successes == trials:
        return 1.0
    return _beta_ppf(confidence, successes + 1, trials - successes)


def paired_cluster_bootstrap(
    differences: list[float],
    group_ids: list[str],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 17,
    interval: Literal["two_sided", "one_sided"] = "two_sided",
) -> ConfidenceInterval:
    """Paired CI that resamples whole lineage/campaign groups."""

    if len(differences) != len(group_ids) or not differences:
        raise ValueError("differences and group_ids must have equal nonzero length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if interval not in {"two_sided", "one_sided"}:
        raise ValueError("interval must be two_sided or one_sided")
    grouped: dict[str, list[float]] = {}
    for difference, group_id in zip(differences, group_ids, strict=True):
        grouped.setdefault(group_id, []).append(float(difference))
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two groups")

    groups = sorted(grouped)
    rng = random.Random(seed)
    sampled_means: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for _ in groups:
            sample.extend(grouped[rng.choice(groups)])
        sampled_means.append(float(np.mean(sample)))
    alpha = 1.0 - confidence
    lower_quantile = alpha / 2.0 if interval == "two_sided" else alpha
    upper_quantile = 1.0 - lower_quantile
    return ConfidenceInterval(
        estimate=float(np.mean(differences)),
        lower=float(np.quantile(sampled_means, lower_quantile)),
        upper=float(np.quantile(sampled_means, upper_quantile)),
        confidence=confidence,
    )
