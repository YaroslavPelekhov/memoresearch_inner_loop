"""Typed contracts for multi-fidelity observations and decisions."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Decision(StrEnum):
    KILL = "kill"
    MORE_EVIDENCE = "more_evidence"
    PROMOTE = "promote"


class FidelityRung(StrictModel):
    """One checkpoint budget and its statistically calibrated decision gates."""

    budget_batches: int = Field(ge=1)
    probe_batches: int = Field(default=0, ge=0)
    kill_gate: float = Field(ge=0.0, le=1.0)
    promote_gate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def gates_do_not_overlap(self) -> FidelityRung:
        if self.kill_gate >= self.promote_gate:
            raise ValueError("kill_gate must be smaller than promote_gate")
        return self


class MultiFidelityPlan(StrictModel):
    """Preregistered cascade; active pruning requires a calibrated policy."""

    version: int = Field(default=1, ge=1)
    schedule_reference_batches: int = Field(ge=1)
    observe_only: bool = True
    calibrated: bool = False
    confidence_method: Literal["exact_clopper_pearson"] = "exact_clopper_pearson"
    winner_recall_floor: float = Field(default=0.95, gt=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    rungs: list[FidelityRung] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_cascade(self) -> MultiFidelityPlan:
        budgets = [rung.budget_batches for rung in self.rungs]
        if budgets != sorted(set(budgets)):
            raise ValueError("rung budgets must be unique and strictly increasing")
        if budgets[-1] != self.schedule_reference_batches:
            raise ValueError("the final rung must equal schedule_reference_batches")
        kill_gates = [rung.kill_gate for rung in self.rungs]
        promote_gates = [rung.promote_gate for rung in self.rungs]
        if kill_gates != sorted(kill_gates):
            raise ValueError("kill gates must be nondecreasing with budget")
        if promote_gates != sorted(promote_gates, reverse=True):
            raise ValueError("promote gates must be nonincreasing with budget")
        widths = [rung.promote_gate - rung.kill_gate for rung in self.rungs]
        if widths != sorted(widths, reverse=True):
            raise ValueError("MORE_EVIDENCE widths must shrink with budget")
        if not self.observe_only and not self.calibrated:
            raise ValueError("active pruning requires calibrated=true")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> MultiFidelityPlan:
        import yaml

        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class ProbabilityEstimate(StrictModel):
    """Calibrated eventual-winner probability and its uncertainty interval."""

    probability: float = Field(ge=0.0, le=1.0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    model_id: str

    @model_validator(mode="after")
    def interval_contains_probability(self) -> ProbabilityEstimate:
        if not self.lower <= self.probability <= self.upper:
            raise ValueError("probability must lie inside [lower, upper]")
        return self


class ProbeFeatures(StrictModel):
    """Features that distinguish current quality from reachable local quality."""

    main_fitness: float
    delta_fitness: float | None = None
    probe_fitness: float | None = None
    local_headroom: float | None = None
    probe_progress: float | None = None


class GateDecision(StrictModel):
    recommended: Decision
    executed: Decision
    reason: str
    kill_gate: float
    promote_gate: float
    observe_only: bool


class RungObservation(StrictModel):
    commit: str
    budget_batches: int
    main_metrics: dict[str, float]
    probe_metrics: dict[str, float] | None = None
    features: ProbeFeatures
    probability: ProbabilityEstimate | None = None
    decision: GateDecision
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    wall_time_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
