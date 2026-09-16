"""Small auditable probability model artifact used by the online cascade."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoresearch.multifidelity.models import (
    ProbabilityEstimate,
    ProbeFeatures,
    StrictModel,
)


class LinearPredictor(StrictModel):
    intercept: float
    coefficients: dict[str, float]
    feature_defaults: dict[str, float] = Field(default_factory=dict)

    def score(self, features: ProbeFeatures) -> float:
        values = features.model_dump()
        score = self.intercept
        for name, coefficient in self.coefficients.items():
            value = values.get(name)
            if value is None:
                value = self.feature_defaults.get(name)
            if value is None:
                raise ValueError(f"feature {name!r} is unavailable and has no default")
            score += coefficient * float(value)
        return score


class LinearRungModel(LinearPredictor):
    interval_radius: float = Field(ge=0.0, le=1.0)
    bootstrap_models: list[LinearPredictor] = Field(default_factory=list)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    calibration_method: str = "fixed_radius"


class ProbabilityModel(StrictModel):
    """Per-rung logistic models with held-out calibration intervals."""

    model_id: str
    calibrated: bool
    frozen: bool = False
    variant: Literal["trajectory_only", "probe_aware"] | None = None
    split_manifest_sha256: str | None = None
    train_split_sha256: str | None = None
    probability_calibration_split_sha256: str | None = None
    locked_test_split_sha256: str | None = None
    per_budget: dict[int, LinearRungModel]
    fitness_per_budget: dict[int, LinearPredictor] = Field(default_factory=dict)

    @model_validator(mode="after")
    def frozen_models_have_provenance(self) -> ProbabilityModel:
        if self.frozen and not all(
            (
                self.calibrated,
                self.variant,
                self.split_manifest_sha256,
                self.train_split_sha256,
                self.probability_calibration_split_sha256,
                self.locked_test_split_sha256,
            )
        ):
            raise ValueError("frozen models require complete split provenance")
        return self

    @classmethod
    def from_json(cls, path: Path) -> ProbabilityModel:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def predict(
        self, budget_batches: int, features: ProbeFeatures
    ) -> ProbabilityEstimate | None:
        rung_model = self.per_budget.get(budget_batches)
        if rung_model is None:
            return None
        try:
            logit = rung_model.score(features)
        except ValueError:
            return None
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        if rung_model.bootstrap_models:
            bootstrap_probabilities = [
                1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, model.score(features)))))
                for model in rung_model.bootstrap_models
            ]
            alpha = 1.0 - rung_model.confidence_level
            lower = min(
                probability,
                float(np.quantile(bootstrap_probabilities, alpha / 2.0)),
            )
            upper = max(
                probability,
                float(np.quantile(bootstrap_probabilities, 1.0 - alpha / 2.0)),
            )
        else:
            lower = max(0.0, probability - rung_model.interval_radius)
            upper = min(1.0, probability + rung_model.interval_radius)
        return ProbabilityEstimate(
            probability=probability,
            lower=lower,
            upper=upper,
            model_id=self.model_id,
        )

    def predict_fitness(
        self, budget_batches: int, features: ProbeFeatures
    ) -> float | None:
        model = self.fitness_per_budget.get(budget_batches)
        if model is None:
            return None
        try:
            return model.score(features)
        except ValueError:
            return None
