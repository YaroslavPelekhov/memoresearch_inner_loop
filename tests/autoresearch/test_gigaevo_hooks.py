from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.gigaevo_hooks import (
    ReviewedIdeaPreStepHook,
    ReviewedMemoryPostStepHook,
    ReviewedMemoryProvider,
)
from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.programs.program import Program


class _Storage:
    def __init__(self, programs: list[Program]) -> None:
        self.programs = programs

    async def get_all(self, **_kwargs: object) -> list[Program]:
        return self.programs


class _Provider:
    async def select_cards(self, *_args: object, **_kwargs: object) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


class _StaleApprovedProvider:
    async def select_cards(self, *_args: object, **_kwargs: object) -> MemorySelection:
        return MemorySelection(cards=["stale card"], card_ids=["program-stale"])


class _LLMThatMustNotRun:
    async def ainvoke(self, _messages: object) -> object:
        raise AssertionError("invalid execution must not reach the ideator")


def _program(*, fitness: float, is_valid: float) -> Program:
    program = Program(code="{}")
    program.metrics = {"fitness": fitness, "is_valid": is_valid}
    return program


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fitness", "is_valid"),
    [(-1.0, 0.0), (float("nan"), 1.0)],
)
async def test_invalid_program_cannot_be_an_ideation_incumbent(
    tmp_path: Path, fitness: float, is_valid: float
) -> None:
    hook = ReviewedIdeaPreStepHook(
        storage=_Storage([_program(fitness=fitness, is_valid=is_valid)]),
        llm=_LLMThatMustNotRun(),
        memory_provider=_Provider(),
        task_description="test",
        review_dir=tmp_path,
        review_mode="auto",
    )

    with pytest.raises(RuntimeError, match="without a valid, finite"):
        await hook._propose(0, 1)

    assert not (tmp_path / "ideas").exists()


@pytest.mark.asyncio
async def test_auto_mode_rejects_and_does_not_publish_invalid_card(
    tmp_path: Path,
) -> None:
    invalid = _program(fitness=-1.0, is_valid=0.0)
    hook = ReviewedMemoryPostStepHook(
        storage=_Storage([invalid]),
        checkpoint_dir=tmp_path / "extracted",
        approved_checkpoint_dir=tmp_path / "approved",
        review_dir=tmp_path / "review",
        review_mode="auto",
        task_description="test",
    )

    await hook()

    review_path = tmp_path / "review" / "memory" / f"program-{invalid.id}.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["status"] == "rejected"
    approved = json.loads(
        (tmp_path / "approved" / "api_index.json").read_text(encoding="utf-8")
    )
    assert approved["memory_cards"] == {}


@pytest.mark.asyncio
async def test_auto_mode_publishes_valid_finite_card(tmp_path: Path) -> None:
    valid = _program(fitness=0.31, is_valid=1.0)
    hook = ReviewedMemoryPostStepHook(
        storage=_Storage([valid]),
        checkpoint_dir=tmp_path / "extracted",
        approved_checkpoint_dir=tmp_path / "approved",
        review_dir=tmp_path / "review",
        review_mode="auto",
        task_description="test",
    )

    await hook()

    review_path = tmp_path / "review" / "memory" / f"program-{valid.id}.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["status"] == "approved"
    approved = json.loads(
        (tmp_path / "approved" / "api_index.json").read_text(encoding="utf-8")
    )
    assert list(approved["memory_cards"]) == [f"program-{valid.id}"]


@pytest.mark.asyncio
async def test_auto_mode_does_not_expose_card_without_review_sidecar(
    tmp_path: Path,
) -> None:
    provider = ReviewedMemoryProvider(
        provider=_StaleApprovedProvider(),
        review_dir=tmp_path / "review",
        review_mode="auto",
    )

    selection = await provider.select_cards(
        _program(fitness=0.31, is_valid=1.0),
        task_description="test",
        metrics_description="fitness",
    )

    assert selection.cards == []
    assert selection.card_ids == []
