from __future__ import annotations

import pytest

from autoresearch.multifidelity.calibration import (
    LabeledProbability,
    calibrate_kill_gate,
)
from autoresearch.multifidelity.dataset import (
    SPLIT_NAMES,
    assign_group_splits,
    freeze_winner_threshold,
    label_grouped_records,
)
from autoresearch.multifidelity.evaluation import (
    PolicyOutcome,
    ProbePredictionOutcome,
    evaluate_locked_policy,
    evaluate_probe_incremental_signal,
)
from autoresearch.multifidelity.features import build_probe_features
from autoresearch.multifidelity.models import (
    Decision,
    FidelityRung,
    MultiFidelityPlan,
    ProbabilityEstimate,
    RungObservation,
)
from autoresearch.multifidelity.policy import ShrinkingGatePolicy
from autoresearch.multifidelity.runner import _resume_promoted_run_at_final
from autoresearch.multifidelity.statistics import clopper_pearson_lower


def test_plan_requires_narrowing_gates_and_fixed_final_horizon() -> None:
    plan = MultiFidelityPlan(
        schedule_reference_batches=100,
        rungs=[
            FidelityRung(
                budget_batches=10,
                probe_batches=2,
                kill_gate=0.01,
                promote_gate=0.99,
            ),
            FidelityRung(
                budget_batches=100,
                kill_gate=0.20,
                promote_gate=0.80,
            ),
        ],
    )
    assert plan.observe_only is True

    with pytest.raises(ValueError, match="active pruning requires"):
        MultiFidelityPlan.model_validate(
            {
                **plan.model_dump(),
                "observe_only": False,
            }
        )


def test_policy_uses_interval_bounds_and_observe_only_override() -> None:
    rung = FidelityRung(
        budget_batches=10,
        kill_gate=0.10,
        promote_gate=0.80,
    )
    estimate = ProbabilityEstimate(
        probability=0.04,
        lower=0.02,
        upper=0.08,
        model_id="heldout-v1",
    )
    active = ShrinkingGatePolicy(observe_only=False).decide(estimate, rung)
    shadow = ShrinkingGatePolicy(observe_only=True).decide(estimate, rung)

    assert active.recommended == Decision.KILL
    assert active.executed == Decision.KILL
    assert shadow.recommended == Decision.KILL
    assert shadow.executed == Decision.MORE_EVIDENCE


def test_promotion_means_skip_directly_to_full_budget() -> None:
    rung = FidelityRung(
        budget_batches=10,
        kill_gate=0.10,
        promote_gate=0.80,
    )
    estimate = ProbabilityEstimate(
        probability=0.95,
        lower=0.90,
        upper=0.98,
        model_id="heldout-v1",
    )
    decision = ShrinkingGatePolicy(observe_only=False).decide(estimate, rung)
    observation = {
        "commit": "candidate",
        "budget_batches": 10,
        "main_metrics": {"fitness": 1.0},
        "features": {"main_fitness": 1.0},
        "probability": estimate.model_dump(),
        "decision": decision.model_dump(),
    }

    assert decision.executed == Decision.PROMOTE
    assert _resume_promoted_run_at_final(
        [RungObservation.model_validate(observation)],
        final_budget=100,
    )


def test_probe_features_match_preregistered_definitions() -> None:
    features = build_probe_features(
        main_fitness=0.40,
        previous_main_fitness=0.35,
        probe_fitness=0.55,
        previous_probe_fitness=0.43,
    )

    assert features.delta_fitness == pytest.approx(0.05)
    assert features.local_headroom == pytest.approx(0.15)
    assert features.probe_progress == pytest.approx(0.12)


def test_kill_gate_requires_recall_lower_bound() -> None:
    records = [LabeledProbability(0.20, True) for _ in range(100)]
    records.extend(LabeledProbability(0.05, False) for _ in range(100))

    gate = calibrate_kill_gate(records, recall_floor=0.95, confidence=0.95)

    assert gate.threshold == 0.20
    assert gate.observed_rate == 1.0
    assert gate.lower_confidence_bound >= 0.95
    assert clopper_pearson_lower(58, 58, 0.95) < 0.95
    assert clopper_pearson_lower(59, 59, 0.95) >= 0.95
    assert clopper_pearson_lower(59, 59, 0.95) == pytest.approx(0.05 ** (1 / 59))


def test_kill_gate_refuses_to_claim_significance_with_too_few_winners() -> None:
    records = [LabeledProbability(0.20, True) for _ in range(10)]

    with pytest.raises(ValueError, match="insufficient winners"):
        calibrate_kill_gate(records, recall_floor=0.95, confidence=0.95)


def test_grouped_split_freezes_winner_threshold_on_train_only() -> None:
    records = [
        {
            "run_id": f"run-{index}",
            "group_id": f"group-{index}",
            "final_fitness": float(index),
        }
        for index in range(12)
    ]
    assignments = assign_group_splits(
        [str(record["group_id"]) for record in records],
        fractions={
            "train": 0.50,
            "probability_calibration": 1 / 6,
            "policy_selection": 1 / 6,
            "locked_test": 1 / 6,
        },
        seed=7,
    )
    threshold = freeze_winner_threshold(
        records,
        assignments,
        winner_fraction=0.25,
    )
    labeled = label_grouped_records(
        records,
        assignments,
        winner_threshold=threshold,
    )

    train_values = sorted(
        (
            float(record["final_fitness"])
            for record in records
            if assignments[str(record["group_id"])] == "train"
        ),
        reverse=True,
    )
    assert threshold == train_values[1]
    assert {record["split"] for record in labeled} == set(SPLIT_NAMES)
    assert all(
        record["eventual_winner"] == (float(record["final_fitness"]) >= threshold)
        for record in labeled
    )


def test_locked_policy_requires_exact_and_cluster_robust_recall() -> None:
    records = [
        PolicyOutcome(
            run_id=f"run-{index}",
            group_id=f"group-{index}",
            eventual_winner=True,
            survived=True,
            full_compute=100.0,
            cascade_compute=50.0,
        )
        for index in range(59)
    ]

    result = evaluate_locked_policy(records, resamples=200, minimum_groups=20)

    assert result["winner_recall_exact_lower"] >= 0.95
    assert result["winner_recall_cluster_bootstrap_lower"] == 1.0
    assert result["compute_saving"] == 0.5
    assert result["certified"] is True


def test_probe_incremental_signal_uses_paired_grouped_error() -> None:
    records = [
        ProbePredictionOutcome(
            run_id=f"run-{index}",
            group_id=f"group-{index}",
            final_fitness=float(index),
            trajectory_prediction=float(index) + 1.0,
            probe_prediction=float(index) + 0.1,
        )
        for index in range(20)
    ]

    result = evaluate_probe_incremental_signal(
        records,
        resamples=200,
        minimum_groups=20,
    )

    assert result["effect"] == pytest.approx(0.99)
    assert result["one_sided_lower"] > 0.0
    assert result["incremental_signal_supported"] is True
