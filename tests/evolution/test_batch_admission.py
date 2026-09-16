"""Focused tests for generation-wide strategy admission."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.strategies.base import (
    BatchAdmissionResult,
    EvolutionStrategy,
)
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _program() -> Program:
    return Program(code="def solve(): return 1", state=ProgramState.DONE)


class _SequentialReplacingStrategy(EvolutionStrategy):
    def __init__(self, incumbent_id: str):
        self.active_ids = [incumbent_id]
        self.calls: list[str] = []
        self.error_id: str | None = None

    async def add(self, program: Program) -> bool:
        self.calls.append(program.id)
        if program.id == self.error_id:
            raise RuntimeError("broken candidate")
        self.active_ids = [program.id]
        return True

    async def select_elites(self, total: int) -> list[Program]:
        return []

    async def get_program_ids(self) -> list[str]:
        return list(self.active_ids)


class _CuratingStrategy(EvolutionStrategy):
    def __init__(self, incumbent_id: str):
        self.active_ids = [incumbent_id]
        self.batch_calls: list[list[Program]] = []
        self.result: BatchAdmissionResult | None = None

    async def add(self, program: Program) -> bool:
        raise AssertionError("batch strategy must not use single-program add")

    async def add_batch(self, programs: list[Program]) -> BatchAdmissionResult:
        self.batch_calls.append(list(programs))
        assert self.result is not None
        self.active_ids = [
            *self.result.accepted_new_program_ids,
            *(
                program_id
                for program_id in self.active_ids
                if program_id not in set(self.result.evicted_incumbent_program_ids)
            ),
        ]
        return self.result

    async def select_elites(self, total: int) -> list[Program]:
        return []

    async def get_program_ids(self) -> list[str]:
        return list(self.active_ids)


def _engine(strategy: EvolutionStrategy) -> EvolutionEngine:
    storage = AsyncMock()
    storage.get_ids_by_status.return_value = []
    writer = MagicMock()
    writer.bind.return_value = writer
    tracker = MagicMock()
    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(),
        writer=writer,
        metrics_tracker=tracker,
    )
    engine.state = AsyncMock()
    return engine


async def test_default_batch_preserves_order_and_reports_final_archive() -> None:
    incumbent = _program()
    first = _program()
    survivor = _program()
    broken = _program()
    strategy = _SequentialReplacingStrategy(incumbent.id)
    strategy.error_id = broken.id

    result = await strategy.add_batch([first, survivor, broken])

    assert strategy.calls == [first.id, survivor.id, broken.id]
    assert result.accepted_new_program_ids == [survivor.id]
    assert result.rejected_new_program_ids == [first.id, broken.id]
    assert result.evicted_incumbent_program_ids == [incumbent.id]
    assert result.metadata["displaced_new_program_ids"] == [first.id]
    assert (
        "RuntimeError: broken candidate"
        in result.metadata["admission_errors"][broken.id]
    )


def test_batch_result_rejects_overlapping_decisions() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        BatchAdmissionResult(
            accepted_new_program_ids=["candidate"],
            rejected_new_program_ids=["candidate"],
        )


async def test_engine_filters_then_applies_one_atomic_batch_result() -> None:
    incumbent = _program()
    accepted = _program()
    rejected = _program()
    invalid = _program()
    strategy = _CuratingStrategy(incumbent.id)
    strategy.result = BatchAdmissionResult(
        accepted_new_program_ids=[accepted.id],
        rejected_new_program_ids=[rejected.id],
        evicted_incumbent_program_ids=[incumbent.id],
        metadata={"curation_round": 1},
    )
    engine = _engine(strategy)
    engine.config.program_acceptor = MagicMock()
    engine.config.program_acceptor.is_accepted.side_effect = lambda program: (
        program.id != invalid.id
    )
    engine.storage.get_ids_by_status.return_value = [
        incumbent.id,
        accepted.id,
        rejected.id,
        invalid.id,
    ]
    engine.storage.mget.return_value = [accepted, rejected, invalid]

    await engine._ingest_completed_programs()

    assert len(strategy.batch_calls) == 1
    assert [program.id for program in strategy.batch_calls[0]] == [
        accepted.id,
        rejected.id,
    ]
    engine.storage.batch_transition_by_ids.assert_awaited_once_with(
        [invalid.id, rejected.id, incumbent.id],
        ProgramState.DONE.value,
        ProgramState.DISCARDED.value,
    )
    assert accepted.state == ProgramState.DONE
    assert rejected.state == ProgramState.DISCARDED
    assert invalid.state == ProgramState.DISCARDED
    assert engine.metrics.added == 1
    assert engine.metrics.rejected_validation == 1
    assert engine.metrics.rejected_strategy == 1

    outcomes = {
        call.args[0].id: call.kwargs["outcome"]
        for call in engine.mutation_operator.on_program_ingested.await_args_list
    }
    assert outcomes == {
        accepted.id: MutationOutcome.ACCEPTED,
        rejected.id: MutationOutcome.REJECTED_STRATEGY,
        invalid.id: MutationOutcome.REJECTED_ACCEPTOR,
    }


async def test_malformed_batch_result_reconciles_from_active_archive() -> None:
    incumbent = _program()
    rejected = _program()
    accepted = _program()
    strategy = _CuratingStrategy(incumbent.id)
    # Missing ``accepted`` from the result makes it invalid, while the strategy
    # snapshot represents the partial/final write the engine must honor.
    strategy.result = BatchAdmissionResult(
        rejected_new_program_ids=[rejected.id],
        evicted_incumbent_program_ids=[incumbent.id],
    )
    engine = _engine(strategy)

    async def malformed_batch(programs: list[Program]) -> BatchAdmissionResult:
        strategy.active_ids = [accepted.id]
        assert [program.id for program in programs] == [rejected.id, accepted.id]
        assert strategy.result is not None
        return strategy.result

    strategy.add_batch = malformed_batch  # type: ignore[method-assign]

    result = await engine._admit_generation_batch(
        [rejected, accepted],
        incumbent_program_ids={incumbent.id},
    )

    assert result.accepted_new_program_ids == [accepted.id]
    assert result.rejected_new_program_ids == [rejected.id]
    assert result.evicted_incumbent_program_ids == [incumbent.id]
    assert result.metadata["reconciled_from_active_archive"] is True


async def test_first_resumed_step_curates_recovered_child_before_new_mutants() -> None:
    incumbent = _program()
    recovered = _program()
    new_mutant = _program()
    strategy = _CuratingStrategy(incumbent.id)
    engine = _engine(strategy)
    engine.config.program_acceptor = MagicMock()
    engine.config.program_acceptor.is_accepted.return_value = True
    engine.storage.snapshot = MagicMock()
    engine.storage.load_run_state.return_value = None
    engine.storage.get_ids_by_status.side_effect = [
        [incumbent.id, recovered.id],
        [incumbent.id, recovered.id, new_mutant.id],
    ]
    engine.storage.mget.side_effect = [[recovered], [new_mutant]]
    engine._await_idle = AsyncMock()  # type: ignore[method-assign]
    engine._select_elites_for_mutation = AsyncMock(  # type: ignore[method-assign]
        return_value=[incumbent]
    )
    engine._create_mutants = AsyncMock(  # type: ignore[method-assign]
        return_value=[new_mutant.id]
    )
    engine._refresh_archive_programs = AsyncMock(  # type: ignore[method-assign]
        return_value=0
    )
    engine._format_archive_best_summary = AsyncMock(  # type: ignore[method-assign]
        return_value=""
    )

    async def admit_all(programs: list[Program]) -> BatchAdmissionResult:
        strategy.batch_calls.append(list(programs))
        accepted_ids = [program.id for program in programs]
        strategy.active_ids = list(dict.fromkeys([*accepted_ids, *strategy.active_ids]))
        return BatchAdmissionResult(accepted_new_program_ids=accepted_ids)

    strategy.add_batch = admit_all  # type: ignore[method-assign]

    await engine.restore_state()
    await engine.step()

    assert [[program.id for program in batch] for batch in strategy.batch_calls] == [
        [recovered.id],
        [new_mutant.id],
    ]
    engine.storage.batch_move_status_sets.assert_not_awaited()
    assert engine._resume_ingest_pending is False
