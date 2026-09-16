"""Small auditable probability model artifact used by the online cascade."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field

from autoresearch.multifidelity.models import (
    ProbabilityEstimate,
    ProbeFeatures,
    StrictModel,
)


class LinearRungModel(StrictModel):
    intercept: float
    coefficients: dict[str, float]
    interval_radius: float = Field(ge=0.0, le=1.0)


class ProbabilityModel(StrictModel):
    """Per-rung logistic models with held-out calibration intervals."""

    model_id: str
    calibrated: bool
    per_budget: dict[int, LinearRungModel]

    @classmethod
    def from_json(cls, path: Path) -> ProbabilityModel:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def predict(
        self, budget_batches: int, features: ProbeFeatures
    ) -> ProbabilityEstimate | None:
        rung_model = self.per_budget.get(budget_batches)
        if rung_model is None:
            return None
        values = features.model_dump()
        logit = rung_model.intercept
        for name, coefficient in rung_model.coefficients.items():
            value = values.get(name)
            if value is None:
                return None
            logit += coefficient * float(value)
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        return ProbabilityEstimate(
            probability=probability,
            lower=max(0.0, probability - rung_model.interval_radius),
            upper=min(1.0, probability + rung_model.interval_radius),
            model_id=self.model_id,
        )
