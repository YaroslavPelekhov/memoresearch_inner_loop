"""Typed records at the idea/evolution boundary.

Raw benchmark artifacts remain owned by GigaEvo. These records are the small,
human-readable index that connects ideas to implementations and takeaways.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchTask(StrictModel):
    id: str
    title: str
    description: str
    fixed_contract: list[str]
    mutable_files: list[str]
    canonical_ref: str
    proposal_count: int = Field(default=5, ge=2, le=20)
    parent_candidate_count: int = Field(default=8, ge=1, le=50)
    evolution_generations: int = Field(default=4, ge=1)
    screen_batches: int | None = Field(default=None, ge=1)
    confirmation_batches: int | None = Field(default=None, ge=1)
    screen_eval_batches: int = Field(default=32, ge=1)
    multifidelity_plan: str | None = None

    @model_validator(mode="after")
    def require_two_complete_horizons(self) -> ResearchTask:
        if (self.screen_batches is None) != (self.confirmation_batches is None):
            raise ValueError("screen and confirmation batches must be set together")
        if (
            self.screen_batches is not None
            and self.confirmation_batches is not None
            and self.confirmation_batches <= self.screen_batches
        ):
            raise ValueError("confirmation must be longer than the screen")
        return self


class Idea(StrictModel):
    id: str
    title: str
    hypothesis: str
    mechanism: str
    implementation_direction: str
    success_criteria: list[str]
    risks: list[str] = Field(default_factory=list)
    source_idea_ids: list[str] = Field(default_factory=list)
    takeaway_ids: list[str] = Field(default_factory=list)
    status: Literal["proposed", "approved", "rejected", "completed", "failed"] = (
        "proposed"
    )
    scientific_status: Literal[
        "not_evaluated", "preliminary", "supported", "refuted", "inconclusive"
    ] = "not_evaluated"


class ParentSelection(StrictModel):
    implementation_id: str
    commit: str
    reason: str


class IdeaVersion(StrictModel):
    idea_id: str
    version: int = Field(ge=1)
    implementation_prompt: str
    parent: ParentSelection
    seed_commit: str = ""
    mutable_files: list[str]
    immutable_base_commit: str
    evolution_generations: int = Field(ge=1)
    status: Literal["initializing", "evolving", "completed", "failed"] = "initializing"
    scientific_status: Literal[
        "not_evaluated", "preliminary", "supported", "refuted", "inconclusive"
    ] = "not_evaluated"


class Implementation(StrictModel):
    id: str
    idea_id: str | None = None
    idea_version: int | None = None
    commit: str
    parent_commit: str | None = None
    fitness: float | None = None
    screen_fitness: float | None = None
    confirmation_fitness: float | None = None
    screen_metrics: dict[str, float] = Field(default_factory=dict)
    confirmation_metrics: dict[str, float] = Field(default_factory=dict)
    screen_feedback: dict[str, Any] = Field(default_factory=dict)
    confirmation_feedback: dict[str, Any] = Field(default_factory=dict)
    is_valid: bool | None = None
    verdict: str = "unknown"
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class Takeaway(StrictModel):
    id: str
    idea_id: str
    idea_version: int
    kind: Literal[
        "supported_mechanism",
        "refuted_mechanism",
        "operational_feasibility",
        "implementation_lesson",
        "infrastructure_failure",
        "uncertainty",
    ]
    claim: str
    scope_and_conditions: str
    confidence: Literal["low", "medium", "high"]
    supporting_implementation_ids: list[str] = Field(default_factory=list)
    contradicting_implementation_ids: list[str] = Field(default_factory=list)
    recommended_next_action: str
    deduplication_key: str = ""

    @model_validator(mode="after")
    def require_evidence_citation(self) -> Takeaway:
        if not (
            self.supporting_implementation_ids or self.contradicting_implementation_ids
        ):
            raise ValueError("A takeaway must cite at least one implementation")
        return self


class CampaignState(StrictModel):
    campaign_id: str
    next_idea_number: int = 1
    completed_rounds: int = 0
    active_idea_id: str | None = None
    status: Literal["ready", "awaiting_human", "running", "failed", "completed"] = (
        "ready"
    )
