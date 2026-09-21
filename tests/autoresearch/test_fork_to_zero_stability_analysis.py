from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_analysis() -> ModuleType:
    path = ROOT / "analysis/fork-to-zero-stability-v1/analyze.py"
    spec = importlib.util.spec_from_file_location("fork_zero_stability_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis()


def _observation(
    *,
    main: float = 0.15,
    control: float = 0.151,
    probe: float = 0.152,
    clip_fraction: float = 0.1,
    norm_p95: float = 2.0,
) -> dict[str, object]:
    return {
        "features": {
            "main_fitness": main,
            "control_fitness": control,
            "probe_fitness": probe,
        },
        "main_metrics": {
            "heldout_loss_final": 5.0,
            "gradient_clipping_fraction": clip_fraction,
            "gradient_norm_pre_clip_p95": norm_p95,
        },
        "metadata": {
            "control_is_valid": True,
            "probe_is_valid": True,
            "control_feedback": {"status": "screen_complete"},
            "probe_feedback": {"status": "screen_complete"},
        },
    }


def test_detector_alarms_use_frozen_margins() -> None:
    values = ANALYSIS.detector_values(
        _observation(control=0.1479, probe=0.1458, clip_fraction=0.6)
    )

    assert values["control_alarm"] is True
    assert values["fork_alarm"] is True
    assert values["gradient_alarm"] is True


def test_intrinsic_probe_failure_is_an_alarm() -> None:
    observation = _observation()
    observation["features"]["probe_fitness"] = None
    observation["metadata"]["probe_is_valid"] = False
    observation["metadata"]["probe_feedback"] = {
        "status": "nonfinite_training_gradient"
    }

    assert ANALYSIS.detector_values(observation)["fork_alarm"] is True


def test_outcome_censors_infrastructure_and_labels_nonfinite() -> None:
    checkpoint = _observation()
    outcome = ANALYSIS.classify_outcome(
        checkpoint,
        None,
        [{"status": "train_failed", "error_type": "cuda_oom"}],
    )
    assert outcome == (None, "censored_incomplete_main", None)

    outcome = ANALYSIS.classify_outcome(
        checkpoint,
        None,
        [
            {
                "status": "nonfinite_training_gradient",
                "nonfinite_gradient": {"batch": 1300},
            }
        ],
    )
    assert outcome == (True, "later_nonfinite_main", 1300)


def test_terminal_loss_ratio_is_frozen_at_ten_percent() -> None:
    checkpoint = _observation()
    stable = {"main_metrics": {"heldout_loss_final": 5.49}}
    collapsed = {"main_metrics": {"heldout_loss_final": 5.5}}

    assert ANALYSIS.classify_outcome(checkpoint, stable, [])[0] is False
    assert ANALYSIS.classify_outcome(checkpoint, collapsed, [])[0] is True


def _candidate(
    campaign: str, outcome: bool, fork: bool, control: bool
) -> dict[str, object]:
    return {
        "campaign": campaign,
        "gpu": "0" if "gpu0" in campaign else "1",
        "commit": f"{campaign}-{outcome}-{fork}-{control}",
        "eligible_primary": True,
        "outcome": outcome,
        "outcome_batch": 4096 if outcome else None,
        "fork_alarm": fork,
        "fork_score": float(fork),
        "fork_earliest_checkpoint": 512 if fork else None,
        "control_alarm": control,
        "control_score": float(control),
        "control_earliest_checkpoint": 512 if control else None,
        "gradient_alarm": False,
        "gradient_score": 0.0,
        "gradient_earliest_checkpoint": None,
    }


def test_evaluation_refuses_positive_claim_when_sample_is_too_small() -> None:
    rows = [
        _candidate("gpu0-pool1", True, True, False),
        _candidate("gpu1-pool1", False, False, False),
    ]

    result = ANALYSIS.evaluate(rows)

    assert result["sample_sufficiency_passed"] is False
    assert result["verdict"] == "inconclusive_insufficient_sample"
    assert result["fork_minus_control_sensitivity"]["estimate"] == pytest.approx(1.0)


def test_aggregate_excludes_repeated_canonical_commit() -> None:
    def row(commit: str, *, is_baseline: bool) -> dict[str, object]:
        return {
            "campaign": "fork-zero-stability-v1-gpu0-pool1",
            "gpu": "0",
            "commit": commit,
            "checkpoint": 512,
            "is_baseline": is_baseline,
            "outcome": False,
            "outcome_batch": None,
            "fork_alarm": False,
            "fork_score": 0.0,
            "control_alarm": False,
            "control_score": 0.0,
            "gradient_alarm": False,
            "gradient_score": 0.0,
        }

    rows = [
        row("canonical", is_baseline=True),
        row("canonical", is_baseline=False),
        row("candidate", is_baseline=False),
    ]

    candidates = ANALYSIS.aggregate_candidates(rows)

    assert [candidate["commit"] for candidate in candidates] == ["candidate"]


def test_evaluation_accepts_only_full_preregistered_campaign_matrix() -> None:
    rows = []
    for campaign in sorted(ANALYSIS.EXPECTED_CAMPAIGNS):
        rows.extend(
            [
                _candidate(campaign, True, True, False),
                _candidate(campaign, True, True, False),
                _candidate(campaign, False, False, False),
                _candidate(campaign, False, False, False),
                _candidate(campaign, False, False, False),
            ]
        )

    result = ANALYSIS.evaluate(rows)

    assert result["sample_sufficiency_passed"] is True
    assert result["verdict"] == "supported_as_instability_detector"
    assert result["fork_minus_control_sensitivity"]["ci95_lower"] == pytest.approx(1.0)


def test_collect_applies_plan_hash_and_terminal_label(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("version: 3\n", encoding="utf-8")
    plan_hash = sha256(plan.read_bytes()).hexdigest()
    campaign = tmp_path / "fork-zero-stability-v1-gpu0-pool1"
    training = campaign / "round-1/implementations/candidate/training"
    training.mkdir(parents=True)

    def rung(batch: int, loss: float) -> dict[str, object]:
        matched = batch in {512, 1024}
        return {
            "commit": "candidate-commit",
            "budget_batches": batch,
            "main_metrics": {"heldout_loss_final": loss},
            "features": {
                "main_fitness": 0.15,
                "control_fitness": 0.151 if matched else None,
                "probe_fitness": 0.152 if matched else None,
            },
            "metadata": {
                "plan_sha256": plan_hash,
                "control_is_valid": matched,
                "probe_is_valid": matched,
                "control_feedback": {"status": "screen_complete"},
                "probe_feedback": {"status": "screen_complete"},
            },
        }

    trajectory = [
        rung(256, 5.2),
        rung(512, 5.0),
        rung(1024, 4.8),
        rung(2048, 5.0),
        rung(4096, 5.6),
    ]
    (training / "multifidelity-trajectory.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )

    rows, provenance = ANALYSIS.collect([campaign], plan_hash)

    assert provenance["plan_sha256"] == plan_hash
    assert len(rows) == 2
    assert rows[0]["outcome"] is True
    assert rows[1]["outcome"] is True
