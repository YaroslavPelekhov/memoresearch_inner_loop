#!/usr/bin/env python3
"""Generate paired model predictions under a frozen multi-fidelity protocol."""

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

from autoresearch.multifidelity.predictor import ProbabilityModel
from autoresearch.multifidelity.training import features_from_record


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve_model_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_models(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, ProbabilityModel]:
    models: dict[str, ProbabilityModel] = {}
    for variant in ("trajectory_only", "probe_aware"):
        reference = manifest["models"][variant]
        path = _resolve_model_path(manifest_path, str(reference["path"]))
        if _sha256(path) != reference["sha256"]:
            raise ValueError(f"{variant} model hash differs from freeze manifest")
        model = ProbabilityModel.from_json(path)
        if not model.frozen or model.variant != variant:
            raise ValueError(f"invalid frozen {variant} model")
        if model.split_manifest_sha256 != manifest["split_manifest_sha256"]:
            raise ValueError(f"{variant} model belongs to another split manifest")
        models[variant] = model
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("policy_selection", "locked_test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy-variant",
        choices=("trajectory_only", "probe_aware"),
    )
    parser.add_argument("--unlock-locked-test", action="store_true")
    args = parser.parse_args()

    manifest_path = args.freeze_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.split == "policy_selection":
        if manifest.get("stage") != "models_frozen":
            raise ValueError("policy selection requires a model-freeze manifest")
        policy_variant = args.policy_variant or "probe_aware"
    elif manifest.get("stage") != "policy_frozen" or not args.unlock_locked_test:
        raise ValueError(
            "locked_test requires a policy-freeze manifest and --unlock-locked-test"
        )
    else:
        policy_variant = str(manifest["policy_model_variant"])
        if args.policy_variant is not None and args.policy_variant != policy_variant:
            raise ValueError("locked_test policy variant differs from policy freeze")
    expected_hash = manifest["split_sha256"][args.split]
    if _sha256(args.records) != expected_hash:
        raise ValueError(f"{args.split} records hash differs from freeze manifest")
    models = _load_models(manifest_path, manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        args.records.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != args.split:
                raise ValueError(f"wrong split on line {line_number}")
            budget = int(record["budget_batches"])
            features = features_from_record(record)
            estimates = {
                variant: model.predict(budget, features)
                for variant, model in models.items()
            }
            fitness = {
                variant: model.predict_fitness(budget, features)
                for variant, model in models.items()
            }
            if any(value is None for value in estimates.values()) or any(
                value is None for value in fitness.values()
            ):
                raise ValueError(f"model cannot predict line {line_number}")
            selected = estimates[policy_variant]
            assert selected is not None
            output = {
                **record,
                "probability": selected.probability,
                "probability_lower": selected.lower,
                "probability_upper": selected.upper,
                "policy_model_variant": policy_variant,
            }
            for variant in ("trajectory_only", "probe_aware"):
                estimate = estimates[variant]
                assert estimate is not None
                output[f"{variant}_probability"] = estimate.probability
                output[f"{variant}_probability_lower"] = estimate.lower
                output[f"{variant}_probability_upper"] = estimate.upper
                output[f"{variant}_fitness_prediction"] = fitness[variant]
            # Compatibility names consumed by the locked probe evaluator.
            output["trajectory_prediction"] = fitness["trajectory_only"]
            output["probe_prediction"] = fitness["probe_aware"]
            destination.write(json.dumps(output, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError("prediction input is empty")

    receipt = {
        "version": 1,
        "split": args.split,
        "freeze_manifest": str(manifest_path),
        "freeze_manifest_sha256": _sha256(manifest_path),
        "input": str(args.records.resolve()),
        "input_sha256": expected_hash,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "records": count,
        "policy_model_variant": policy_variant,
        "locked_test_opened": args.split == "locked_test",
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**receipt, "receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
