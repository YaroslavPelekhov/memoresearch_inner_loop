#!/usr/bin/env python3
"""Replay frozen gates on complete locked-test trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.models import MultiFidelityPlan, ProbabilityEstimate
from autoresearch.multifidelity.policy import ShrinkingGatePolicy


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _outcome(records: list[dict[str, Any]], plan: MultiFidelityPlan) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: int(record["budget_batches"]))
    expected_budgets = [rung.budget_batches for rung in plan.rungs[:-1]]
    if [int(record["budget_batches"]) for record in ordered] != expected_budgets:
        raise ValueError(f"run {ordered[0]['run_id']} does not contain every rung")
    winner_values = {record["eventual_winner"] for record in ordered}
    group_values = {str(record["group_id"]) for record in ordered}
    if len(winner_values) != 1 or len(group_values) != 1:
        raise ValueError("run labels or group ids are inconsistent across rungs")

    policy = ShrinkingGatePolicy(observe_only=False)
    probe_compute = 0
    visited: list[int] = []
    terminal = "full_budget"
    survived = True
    cascade_compute = plan.schedule_reference_batches
    for record, rung in zip(ordered, plan.rungs[:-1], strict=True):
        visited.append(rung.budget_batches)
        probe_compute += rung.probe_batches
        estimate = ProbabilityEstimate(
            probability=float(record["probability"]),
            lower=float(record["probability_lower"]),
            upper=float(record["probability_upper"]),
            model_id=str(record["policy_model_variant"]),
        )
        decision = policy.decide(estimate, rung)
        if decision.executed.value == "kill":
            survived = False
            terminal = "kill"
            cascade_compute = rung.budget_batches
            break
        if decision.executed.value == "promote":
            terminal = "promote_to_full"
            cascade_compute = plan.schedule_reference_batches
            break
    cascade_compute += probe_compute
    return {
        "split": "locked_test",
        "run_id": str(ordered[0]["run_id"]),
        "group_id": next(iter(group_values)),
        "eventual_winner": next(iter(winner_values)),
        "survived": survived,
        "full_compute": plan.schedule_reference_batches,
        "cascade_compute": cascade_compute,
        "terminal_decision": terminal,
        "visited_budgets": visited,
        "probe_compute": probe_compute,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-freeze-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.policy_freeze_manifest.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("stage") != "policy_frozen":
        raise ValueError("simulation requires a policy-freeze manifest")
    plan_path = Path(str(freeze["calibrated_plan"]))
    if _sha256(plan_path) != freeze["calibrated_plan_sha256"]:
        raise ValueError("calibrated plan changed after policy freeze")
    plan = MultiFidelityPlan.from_yaml(plan_path)

    receipt_path = (
        args.prediction_receipt.resolve()
        if args.prediction_receipt
        else args.predictions.with_suffix(".receipt.json").resolve()
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("split") != "locked_test"
        or receipt.get("output_sha256") != _sha256(args.predictions)
        or receipt.get("freeze_manifest_sha256") != _sha256(freeze_path)
        or receipt.get("locked_test_opened") is not True
    ):
        raise ValueError("invalid locked-test prediction receipt")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                grouped[str(value["run_id"])].append(value)
    if not grouped:
        raise ValueError("locked-test predictions are empty")

    outcomes = [_outcome(records, plan) for records in grouped.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for outcome in sorted(outcomes, key=lambda value: value["run_id"]):
            stream.write(json.dumps(outcome, sort_keys=True) + "\n")
    output_receipt = {
        "version": 1,
        "policy_freeze_manifest": str(freeze_path),
        "policy_freeze_manifest_sha256": _sha256(freeze_path),
        "locked_predictions_sha256": _sha256(args.predictions),
        "outcomes": str(args.output.resolve()),
        "outcomes_sha256": _sha256(args.output),
        "runs": len(outcomes),
    }
    output_receipt_path = args.output.with_suffix(".receipt.json")
    output_receipt_path.write_text(
        json.dumps(output_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**output_receipt, "receipt": str(output_receipt_path)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
