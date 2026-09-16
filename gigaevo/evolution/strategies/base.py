from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from gigaevo.programs.program import Program


class BatchAdmissionResult(BaseModel):
    """Outcome of admitting one generation of new programs.

    ``accepted_new_program_ids`` and ``rejected_new_program_ids`` describe only
    the programs supplied to :meth:`EvolutionStrategy.add_batch`. Together they
    must form a complete partition of that input; the engine validates this
    because the result itself does not know the input IDs. Accepted programs are
    the newcomers active in the archive after the complete batch decision;
    rejected programs are not active after that decision.

    ``evicted_incumbent_program_ids`` describes programs that were active before
    the batch and were removed to make room for the selected newcomers. It must
    not be used for a new program that was rejected from this batch.
    """

    accepted_new_program_ids: list[str] = Field(default_factory=list)
    rejected_new_program_ids: list[str] = Field(default_factory=list)
    evicted_incumbent_program_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_id_sets(self) -> BatchAdmissionResult:
        fields = (
            "accepted_new_program_ids",
            "rejected_new_program_ids",
            "evicted_incumbent_program_ids",
        )
        for field_name in fields:
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicate IDs")

        accepted = set(self.accepted_new_program_ids)
        rejected = set(self.rejected_new_program_ids)
        evicted = set(self.evicted_incumbent_program_ids)
        if accepted & rejected:
            raise ValueError("accepted and rejected new program IDs must be disjoint")
        if (accepted | rejected) & evicted:
            raise ValueError(
                "new program IDs and evicted incumbent IDs must be disjoint"
            )
        return self


class StrategyMetrics(BaseModel):
    """Generic metrics that any evolution strategy can provide."""

    total_programs: int = Field(
        default=0, ge=0, description="Total number of programs in the strategy"
    )

    active_populations: int = Field(
        default=0, ge=0, description="Number of active populations/islands"
    )

    strategy_specific_metrics: dict[str, Any] | None = Field(
        default=None, description="Strategy-specific metrics and statistics"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def programs_per_population(self) -> float:
        """Calculate average programs per population."""
        if self.active_populations == 0:
            return 0.0
        return self.total_programs / self.active_populations

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_programs(self) -> bool:
        """Check if strategy contains any programs."""
        return self.total_programs > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed fields."""
        result = {
            "total_programs": self.total_programs,
            "active_populations": self.active_populations,
            "programs_per_population": round(self.programs_per_population, 2),
            "has_programs": self.has_programs,
        }
        if self.strategy_specific_metrics:
            result.update(self.strategy_specific_metrics)
        return result


class EvolutionStrategy(ABC):
    """
    Abstract base class for evolution strategies.

    Defines the core interface that all evolution strategies must implement,
    along with optional capabilities for enhanced monitoring and control.
    """

    @abstractmethod
    async def add(self, program: Program) -> bool:
        """
        Add a program to the evolution strategy.

        Args:
            program: The program to add

        Returns:
            True if program was added/updated, False otherwise
        """
        ...

    async def add_batch(self, programs: list[Program]) -> BatchAdmissionResult:
        """Admit a generation of programs.

        Strategies that need a generation-wide decision can override this
        method. The default deliberately calls :meth:`add` sequentially, in
        input order, so existing strategies retain their admission behavior.
        A failure while adding one program rejects only that program and does
        not prevent the rest of the generation from being considered.

        Returns:
            A structured admission result. ``metadata["admission_errors"]``
            contains per-program exceptions raised by ``add``, when present.
        """
        incumbents_before = list(dict.fromkeys(await self.get_program_ids()))
        provisionally_accepted_ids: list[str] = []
        errors: dict[str, str] = {}

        for program in programs:
            try:
                if await self.add(program):
                    provisionally_accepted_ids.append(program.id)
            except Exception as exc:
                errors[program.id] = f"{type(exc).__name__}: {exc}"

        # Reconcile sequential decisions with the final active archive. A new
        # program may be accepted by one ``add`` call and then displaced by a
        # later program in the same generation.
        active_after = set(await self.get_program_ids())
        accepted_ids = [
            program.id for program in programs if program.id in active_after
        ]
        accepted_set = set(accepted_ids)
        rejected_ids = [
            program.id for program in programs if program.id not in accepted_set
        ]
        evicted_ids = [
            program_id
            for program_id in incumbents_before
            if program_id not in active_after
        ]

        metadata: dict[str, Any] = {}
        if errors:
            metadata["admission_errors"] = errors
        displaced_new_ids = [
            program_id
            for program_id in provisionally_accepted_ids
            if program_id not in accepted_set
        ]
        if displaced_new_ids:
            metadata["displaced_new_program_ids"] = displaced_new_ids

        return BatchAdmissionResult(
            accepted_new_program_ids=accepted_ids,
            rejected_new_program_ids=rejected_ids,
            evicted_incumbent_program_ids=evicted_ids,
            metadata=metadata,
        )

    @abstractmethod
    async def select_elites(self, total: int) -> list[Program]:
        """
        Select elite programs from the strategy.

        Args:
            total: Number of elites to select

        Returns:
            List of selected elite programs
        """
        ...

    @abstractmethod
    async def get_program_ids(self) -> list[str]:
        """
        Get all programs managed by this strategy.

        Returns:
            List of all Program objects in the strategy
        """
        ...

    async def remove_program_by_id(self, program_id: str) -> bool:
        """
        Remove a program from the strategy by ID.

        Args:
            program_id: ID of the program to remove

        Returns:
            True if program was removed, False if not found
        """
        raise NotImplementedError("Strategy does not support program removal")

    # Optional capabilities - strategies can override these for enhanced functionality

    async def get_metrics(self) -> StrategyMetrics | None:
        """
        Get strategy-specific metrics.

        Returns:
            StrategyMetrics object or None if not supported
        """
        return None

    async def cleanup(self) -> None:
        """
        Perform cleanup operations.

        Override this method if strategy supports cleanup operations.
        """

    async def pause(self) -> None:
        """
        Pause strategy operations.

        Override this method if strategy supports pause/resume.
        """

    async def resume(self) -> None:
        """
        Resume strategy operations.

        Override this method if strategy supports pause/resume.
        """

    async def restore_state(self) -> None:
        """Restore persisted counters from storage after a resume.

        Override in strategies that maintain in-memory counters which must
        survive a stop/restart cycle (e.g., generation count, migration timer).
        """

    async def reset_state(self) -> None:
        """Clear persisted strategy state before starting a fresh non-resume run.

        Override in strategies that maintain Redis side indexes outside the
        primary ProgramStorage namespace.
        """

    async def reindex_archive(self) -> None:
        """Re-evaluate archive placements using current program metrics.

        Clears the archive and re-inserts all programs, so that programs
        whose fitness changed (e.g., from external stats updates) compete
        fairly against each other. Default is a no-op for strategies where
        fitness is immutable.
        """
