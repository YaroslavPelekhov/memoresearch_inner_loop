#!/usr/bin/env python3
"""Fit and freeze trajectory-only and probe-aware per-rung models."""

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
from autoresearch.multifidelity.training import fit_probability_model


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("split") != expected_split:
                raise ValueError(
                    f"line {line_number} does not belong to {expected_split}"
                )
            records.append(value)
    return records


def _model_id(
    variant: str,
    manifest_hash: str,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> str:
    payload = json.dumps(
        {
            "variant": variant,
            "split_manifest_sha256": manifest_hash,
            "seed": seed,
            "bootstrap_resamples": bootstrap_resamples,
        },
        sort_keys=True,
    ).encode()
    return f"{variant}-{sha256(payload).hexdigest()[:16]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    manifest_path = (
        args.split_manifest.resolve()
        if args.split_manifest
        else split_dir / "split-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_hashes = manifest.get("split_sha256")
    if not isinstance(split_hashes, dict):
        raise ValueError("split manifest has no split_sha256 mapping")
    train_path = split_dir / "train.jsonl"
    calibration_path = split_dir / "probability_calibration.jsonl"
    for name, path in (
        ("train", train_path),
        ("probability_calibration", calibration_path),
    ):
        if _sha256(path) != split_hashes.get(name):
            raise ValueError(f"{name} split hash differs from its manifest")
    # Deliberately do not open policy_selection.jsonl or locked_test.jsonl here.
    train = _read_jsonl(train_path, expected_split="train")
    calibration = _read_jsonl(
        calibration_path,
        expected_split="probability_calibration",
    )
    plan = MultiFidelityPlan.from_yaml(args.plan)
    if not plan.observe_only:
        raise ValueError("model fitting requires the original observe-only plan")

    manifest_hash = _sha256(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_refs: dict[str, dict[str, str]] = {}
    for index, variant in enumerate(("trajectory_only", "probe_aware")):
        model = fit_probability_model(
            train,
            calibration,
            variant=variant,
            model_id=_model_id(
                variant,
                manifest_hash,
                seed=args.seed + index,
                bootstrap_resamples=args.bootstrap_resamples,
            ),
            split_manifest_sha256=manifest_hash,
            train_split_sha256=str(split_hashes["train"]),
            probability_calibration_split_sha256=str(
                split_hashes["probability_calibration"]
            ),
            locked_test_split_sha256=str(split_hashes["locked_test"]),
            bootstrap_resamples=args.bootstrap_resamples,
            confidence_level=args.confidence,
            seed=args.seed + index,
            l2=args.l2,
        )
        path = args.output_dir / f"{variant}-model.json"
        path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
        model_refs[variant] = {
            "path": path.name,
            "sha256": _sha256(path),
            "model_id": model.model_id,
        }

    freeze = {
        "version": 1,
        "stage": "models_frozen",
        "locked_test_opened": False,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest_hash,
        "split_sha256": split_hashes,
        "observe_only_plan": str(args.plan.resolve()),
        "observe_only_plan_sha256": _sha256(args.plan),
        "models": model_refs,
        "training": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence": args.confidence,
            "seed": args.seed,
            "l2": args.l2,
        },
    }
    freeze_path = args.output_dir / "model-freeze-manifest.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_freeze_manifest": str(freeze_path),
                "models": model_refs,
                "locked_test_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
