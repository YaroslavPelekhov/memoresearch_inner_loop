from __future__ import annotations

import pytest

from autoresearch.multifidelity.calibration import (
    LabeledProbability,
    calibrate_kill_gate,
    wilson_lower_bound,
)
from autoresearch.multifidelity.features import build_probe_features
from autoresearch.multifidelity.models import (
    Decision,
    FidelityRung,
    MultiFidelityPlan,
    ProbabilityEstimate,
)
from autoresearch.multifidelity.policy import ShrinkingGatePolicy


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
    assert wilson_lower_bound(10, 10, 0.95) < 0.95


def test_kill_gate_refuses_to_claim_significance_with_too_few_winners() -> None:
    records = [LabeledProbability(0.20, True) for _ in range(10)]

    with pytest.raises(ValueError, match="insufficient winners"):
        calibrate_kill_gate(records, recall_floor=0.95, confidence=0.95)
