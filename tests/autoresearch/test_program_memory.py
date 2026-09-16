from __future__ import annotations

from autoresearch.program_memory import program_to_card
from gigaevo.programs.program import Program


def test_smoke_execution_is_labeled_as_smoke_not_scientific_baseline() -> None:
    program = Program(code="{}")
    program.metrics = {
        "fitness": 0.0,
        "is_valid": 1.0,
        "llmfoundry_core_equal_raw": 0.0,
    }
    program.metadata["repo_benchmark_feedback"] = {
        "structured_feedback": {"status": "smoke_complete"}
    }

    card = program_to_card(
        program,
        parent=None,
        primary_key="fitness",
        task_description="test",
    )

    assert card.verdict == "smoke"
    assert "was smoke" in card.description


def test_diagnostic_execution_is_not_a_scientific_baseline() -> None:
    program = Program(code="{}")
    program.metrics = {
        "fitness": 0.0,
        "is_valid": 1.0,
        "llmfoundry_core_equal_raw": 0.0,
    }
    program.metadata["repo_benchmark_feedback"] = {
        "structured_feedback": {"status": "diagnostic_complete"}
    }

    card = program_to_card(
        program,
        parent=None,
        primary_key="fitness",
        task_description="test",
    )

    assert card.verdict == "diagnostic"
    assert "was diagnostic" in card.description
