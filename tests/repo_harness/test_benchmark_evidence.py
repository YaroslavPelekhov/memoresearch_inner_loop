from __future__ import annotations

from math import isclose

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.repo_harness.benchmark_evidence import (
    DefaultBenchmarkEvidenceProvider,
    build_benchmark_evidence,
)


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Benchmark fitness",
                is_primary=True,
                higher_is_better=True,
            ),
            "cost": MetricSpec(
                description="Evaluation cost",
                higher_is_better=False,
                unit="USD",
            ),
            "is_valid": MetricSpec(
                description="Validity",
                higher_is_better=True,
            ),
        }
    )


def _program(
    *,
    fitness: float,
    cost: float = 1.0,
    structured: dict | None = None,
    feedback_extra: dict | None = None,
) -> Program:
    program = Program(code="{}")
    program.add_metrics({"fitness": fitness, "cost": cost, "is_valid": 1.0})
    feedback = {
        "returncode": 0,
        "duration_seconds": 12.5,
        "metrics": dict(program.metrics),
    }
    if structured is not None:
        feedback["structured_feedback"] = structured
    feedback.update(feedback_extra or {})
    program.metadata["repo_benchmark_feedback"] = feedback
    return program


def test_aggregate_only_evidence_preserves_metrics_resources_and_bounded_raw():
    program = _program(
        fitness=0.4,
        cost=0.25,
        structured={
            "benchmark": "continuous-optimizer",
            "summary": {
                "metrics": {"auxiliary": 3.0},
                "resource_metrics": {"gpu_seconds": 7.5, "backend": "cuda"},
                "details": "x" * 1000,
            },
        },
    )

    evidence = build_benchmark_evidence(
        program,
        metrics_context=_metrics_context(),
        max_raw_feedback_chars=180,
    )

    assert evidence["benchmark"] == "continuous-optimizer"
    assert evidence["aggregate_metrics"] == {
        "auxiliary": 3.0,
        "cost": 0.25,
        "fitness": 0.4,
        "is_valid": 1.0,
    }
    assert evidence["metric_metadata"]["cost"]["higher_is_better"] is False
    assert evidence["scope"]["completeness"] == "aggregate_only"
    assert evidence["unit_results"] == []
    assert evidence["resource_metrics"] == {
        "backend": "cuda",
        "cost": 0.25,
        "duration_seconds": 12.5,
        "gpu_seconds": 7.5,
    }
    assert evidence["raw_feedback"]["source"] == "structured_feedback"
    assert evidence["raw_feedback"]["truncated"] is True
    assert len(evidence["raw_feedback"]["content"]) <= 180


def test_known_universe_adds_only_real_unknown_ids_and_marks_partial_scope():
    program = _program(
        fitness=0.5,
        structured={
            "benchmark": "generic-cases",
            "scope": {
                "unit_type": "scenario",
                "universe_ids": ["a", "b", "c", "d"],
            },
            "cases": [
                {
                    "case_id": "a",
                    "passed": True,
                    "n_trials": 2,
                    "mean_cost_usd": 0.1,
                },
                {"case_id": "b", "reward": 0.25, "n_trials": 4},
            ],
            # This is diagnostic detail for case b, not a second unit table.
            "examples": [
                {"case_id": "b", "status": "error"},
                {"case_id": "d", "status": "error", "passed": False},
            ],
        },
    )

    evidence = build_benchmark_evidence(program)

    assert evidence["scope"] == {
        "unit_type": "scenario",
        "completeness": "partial",
        "universe_count": 4,
        "evaluated_count": 3,
        "unknown_count": 1,
        "repetitions": 7,
        "source": "cases",
    }
    assert evidence["unit_results"] == [
        {
            "unit_id": "a",
            "status": "success",
            "score": 1.0,
            "repetitions": 2,
            "observed": True,
            "metrics": {"mean_cost_usd": 0.1},
        },
        {
            "unit_id": "b",
            "status": "partial",
            "score": 0.25,
            "repetitions": 4,
            "observed": True,
            "metrics": {},
        },
        {
            "unit_id": "c",
            "status": "unknown",
            "score": None,
            "repetitions": 0,
            "observed": False,
            "metrics": {},
        },
        {
            "unit_id": "d",
            "status": "error",
            "score": 0.0,
            "repetitions": 1,
            "observed": True,
            "metrics": {},
        },
    ]


def test_trials_are_aggregated_without_fabricating_unknown_universe_members():
    program = _program(
        fitness=0.5,
        structured={
            "trials": [
                {"trial": "alpha__1", "reward": 1.0},
                {"trial": "alpha__2", "reward": 0.0},
                {"trial": "beta__1", "status": "skipped"},
            ]
        },
    )

    evidence = build_benchmark_evidence(program)

    assert evidence["scope"]["completeness"] == "unknown"
    assert evidence["scope"]["universe_count"] is None
    assert evidence["scope"]["unknown_count"] is None
    assert [unit["unit_id"] for unit in evidence["unit_results"]] == ["alpha", "beta"]
    assert evidence["unit_results"][0] == {
        "unit_id": "alpha",
        "status": "partial",
        "score": 0.5,
        "repetitions": 2,
        "observed": True,
        "metrics": {},
    }
    assert evidence["unit_results"][1]["observed"] is False
    assert evidence["unit_results"][1]["status"] == "unknown"


def test_selected_units_define_scoped_universe_without_benchmark_specific_names():
    program = _program(
        fitness=0.5,
        structured={
            "summary": {"selected_tasks": ["alpha", "beta", "gamma"]},
            "cases": [
                {"task_id": "alpha", "passed": True},
                {"task_id": "beta", "passed": False},
            ],
        },
    )

    evidence = build_benchmark_evidence(program)

    assert evidence["scope"]["completeness"] == "partial"
    assert evidence["scope"]["universe_count"] == 3
    assert evidence["scope"]["evaluated_count"] == 2
    assert evidence["scope"]["unknown_count"] == 1
    assert evidence["unit_results"][-1] == {
        "unit_id": "gamma",
        "status": "unknown",
        "score": None,
        "repetitions": 0,
        "observed": False,
        "metrics": {},
    }


def test_staged_subset_is_explicitly_projected_not_full_or_failed():
    program = _program(
        fitness=0.6,
        structured={
            "evaluation_completeness": "full",
            "cases": [{"case_id": "failed-parent-case", "passed": False}],
        },
        feedback_extra={
            "staged_validation": {
                "enabled": True,
                "promoted_to_full": False,
                "failed_cases": ["failed-parent-case"],
            }
        },
    )

    evidence = build_benchmark_evidence(program)

    assert evidence["scope"]["completeness"] == "projected"
    assert evidence["unit_results"][0]["observed"] is True


def test_parent_and_archive_comparisons_respect_directions_and_observed_units():
    parent = _program(
        fitness=0.5,
        cost=2.0,
        structured={
            "cases": [
                {"case_id": "new", "passed": False},
                {"case_id": "lost", "passed": True},
                {"case_id": "better", "reward": 0.2},
            ]
        },
    )
    archived = _program(
        fitness=0.6,
        cost=1.5,
        structured={
            "cases": [
                {"case_id": "new", "passed": False},
                {"case_id": "lost", "passed": False},
                {"case_id": "better", "reward": 0.5},
            ]
        },
    )
    candidate = _program(
        fitness=0.7,
        cost=1.0,
        structured={
            "cases": [
                {"case_id": "new", "passed": True},
                {"case_id": "lost", "passed": False},
                {"case_id": "better", "reward": 0.8},
                {
                    "case_id": "projected-win",
                    "passed": True,
                    "observed": False,
                },
            ]
        },
    )

    evidence = DefaultBenchmarkEvidenceProvider(
        metrics_context=_metrics_context()
    ).extract(candidate, parent=parent, archive=[archived, candidate, archived])

    comparison = evidence["comparison_with_parent"]
    assert isclose(comparison["metric_deltas"]["fitness"], 0.2)
    assert comparison["metric_deltas"]["cost"] == -1.0
    assert comparison["directional_metric_deltas"]["cost"] == 1.0
    assert comparison["improved_metrics"] == ["cost", "fitness"]
    assert comparison["improved_units"] == ["better", "new"]
    assert comparison["regressed_units"] == ["lost"]
    assert comparison["newly_successful_units"] == ["new"]
    assert comparison["lost_success_units"] == ["lost"]
    assert "projected-win" not in comparison["improved_units"]

    distinction = evidence["distinction_from_archive"]
    assert distinction["archive_program_count"] == 1
    assert distinction["uniquely_successful_units"] == ["new"]
    assert distinction["best_known_units"] == ["better", "new"]
    assert distinction["best_known_metrics"] == ["cost", "fitness"]
    assert distinction["comparison_basis"] == "observed_unit_results_only"
