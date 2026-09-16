#!/usr/bin/env python3
"""Freeze calibrated gates and model artifacts before opening locked_test."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.models import MultiFidelityPlan
from autoresearch.multifidelity.predictor import ProbabilityModel


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _validated_models(
    freeze_path: Path, freeze: dict[str, Any]
) -> dict[str, dict[str, str]]:
    validated: dict[str, dict[str, str]] = {}
    for variant in ("trajectory_only", "probe_aware"):
        reference = freeze["models"][variant]
        path = _resolve(freeze_path.parent, str(reference["path"])).resolve()
        if _sha256(path) != reference["sha256"]:
            raise ValueError(f"{variant} model hash differs from model freeze")
        model = ProbabilityModel.from_json(path)
        if not model.frozen or model.variant != variant:
            raise ValueError(f"invalid frozen {variant} model")
        if model.split_manifest_sha256 != freeze["split_manifest_sha256"]:
            raise ValueError(f"{variant} model belongs to another split manifest")
        validated[variant] = {
            "path": str(path),
            "sha256": str(reference["sha256"]),
            "model_id": model.model_id,
        }
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-freeze-manifest", type=Path, required=True)
    parser.add_argument("--calibrated-plan", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path, required=True)
    parser.add_argument("--policy-predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_freeze_path = args.model_freeze_manifest.resolve()
    model_freeze = json.loads(model_freeze_path.read_text(encoding="utf-8"))
    if (
        model_freeze.get("stage") != "models_frozen"
        or model_freeze.get("locked_test_opened") is not False
    ):
        raise ValueError("invalid model-freeze stage")
    models = _validated_models(model_freeze_path, model_freeze)

    plan = MultiFidelityPlan.from_yaml(args.calibrated_plan)
    if not plan.calibrated or not plan.observe_only:
        raise ValueError(
            "policy freeze requires calibrated gates still in observe-only mode"
        )
    evidence = json.loads(args.calibration_evidence.read_text(encoding="utf-8"))
    predictions_hash = _sha256(args.policy_predictions)
    if evidence.get("expected_split") != "policy_selection":
        raise ValueError("gate evidence was not produced on policy_selection")
    if evidence.get("records_sha256") != predictions_hash:
        raise ValueError("gate evidence does not match policy predictions")
    if evidence.get("calibrated_plan_sha256") != _sha256(args.calibrated_plan):
        raise ValueError("gate evidence does not match calibrated plan")

    receipt_path = (
        args.prediction_receipt.resolve()
        if args.prediction_receipt
        else args.policy_predictions.with_suffix(".receipt.json").resolve()
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("split") != "policy_selection"
        or receipt.get("output_sha256") != predictions_hash
        or receipt.get("freeze_manifest_sha256") != _sha256(model_freeze_path)
        or receipt.get("locked_test_opened") is not False
    ):
        raise ValueError("invalid policy-selection prediction receipt")

    frozen = {
        "version": 1,
        "stage": "policy_frozen",
        "locked_test_opened": False,
        "model_freeze_manifest": str(model_freeze_path),
        "model_freeze_manifest_sha256": _sha256(model_freeze_path),
        "split_manifest": model_freeze["split_manifest"],
        "split_manifest_sha256": model_freeze["split_manifest_sha256"],
        "split_sha256": model_freeze["split_sha256"],
        "models": models,
        "policy_model_variant": receipt["policy_model_variant"],
        "calibrated_plan": str(args.calibrated_plan.resolve()),
        "calibrated_plan_sha256": _sha256(args.calibrated_plan),
        "calibration_evidence": str(args.calibration_evidence.resolve()),
        "calibration_evidence_sha256": _sha256(args.calibration_evidence),
        "policy_predictions": str(args.policy_predictions.resolve()),
        "policy_predictions_sha256": predictions_hash,
        "prediction_receipt": str(receipt_path),
        "prediction_receipt_sha256": _sha256(receipt_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy_freeze_manifest": str(args.output),
                "sha256": _sha256(args.output),
                "locked_test_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
