from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run(*arguments: object) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.splitlines()[-1])


def _write_split(path: Path, split: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(8):
            winner = index % 2 == 0
            final_fitness = 1.0 if winner else 0.0
            for budget in (10, 20):
                main_fitness = final_fitness * 0.6 + budget / 100.0
                record = {
                    "split": split,
                    "run_id": f"{split}-run-{index}",
                    "group_id": f"{split}-group-{index}",
                    "budget_batches": budget,
                    "main_fitness": main_fitness,
                    "delta_fitness": None if budget == 10 else 0.1,
                    "probe_fitness": main_fitness + (0.2 if winner else -0.1),
                    "local_headroom": 0.2 if winner else -0.1,
                    "probe_progress": None
                    if budget == 10
                    else (0.1 if winner else -0.1),
                    "final_fitness": final_fitness,
                    "eventual_winner": winner,
                    "winner_threshold": 0.5,
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")


def test_frozen_pipeline_keeps_locked_test_closed_until_policy_freeze(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_hashes: dict[str, str] = {}
    for split in (
        "train",
        "probability_calibration",
        "policy_selection",
        "locked_test",
    ):
        path = split_dir / f"{split}.jsonl"
        _write_split(path, split)
        split_hashes[split] = _sha256(path)
    split_manifest = split_dir / "split-manifest.json"
    split_manifest.write_text(
        json.dumps({"version": 1, "split_sha256": split_hashes}),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "schedule_reference_batches": 100,
                "observe_only": True,
                "calibrated": False,
                "confidence_method": "exact_clopper_pearson",
                "winner_recall_floor": 0.20,
                "confidence_level": 0.80,
                "rungs": [
                    {
                        "budget_batches": 10,
                        "probe_batches": 2,
                        "kill_gate": 0.05,
                        "promote_gate": 0.95,
                    },
                    {
                        "budget_batches": 20,
                        "probe_batches": 2,
                        "kill_gate": 0.10,
                        "promote_gate": 0.90,
                    },
                    {
                        "budget_batches": 100,
                        "probe_batches": 0,
                        "kill_gate": 0.40,
                        "promote_gate": 0.60,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model_dir = tmp_path / "models"
    fitted = _run(
        ROOT / "tools/fit-multifidelity-models.py",
        "--split-dir",
        split_dir,
        "--plan",
        plan_path,
        "--output-dir",
        model_dir,
        "--bootstrap-resamples",
        10,
    )
    model_freeze = Path(str(fitted["model_freeze_manifest"]))
    assert json.loads(model_freeze.read_text())["locked_test_opened"] is False

    policy_predictions = tmp_path / "policy-predictions.jsonl"
    _run(
        ROOT / "tools/predict-multifidelity-models.py",
        "--freeze-manifest",
        model_freeze,
        "--records",
        split_dir / "policy_selection.jsonl",
        "--split",
        "policy_selection",
        "--output",
        policy_predictions,
    )
    locked_attempt = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/predict-multifidelity-models.py"),
            "--freeze-manifest",
            str(model_freeze),
            "--records",
            str(split_dir / "locked_test.jsonl"),
            "--split",
            "locked_test",
            "--output",
            str(tmp_path / "premature.jsonl"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert locked_attempt.returncode != 0
    assert "policy-freeze manifest" in locked_attempt.stderr

    calibrated_plan = tmp_path / "calibrated-plan.yaml"
    _run(
        ROOT / "tools/calibrate-multifidelity-gates.py",
        "--plan",
        plan_path,
        "--records",
        policy_predictions,
        "--output",
        calibrated_plan,
        "--promotion-precision-floor",
        0.20,
    )
    policy_freeze = tmp_path / "policy-freeze.json"
    _run(
        ROOT / "tools/freeze-multifidelity-policy.py",
        "--model-freeze-manifest",
        model_freeze,
        "--calibrated-plan",
        calibrated_plan,
        "--calibration-evidence",
        calibrated_plan.with_suffix(".calibration.json"),
        "--policy-predictions",
        policy_predictions,
        "--output",
        policy_freeze,
    )

    locked_predictions = tmp_path / "locked-predictions.jsonl"
    _run(
        ROOT / "tools/predict-multifidelity-models.py",
        "--freeze-manifest",
        policy_freeze,
        "--records",
        split_dir / "locked_test.jsonl",
        "--split",
        "locked_test",
        "--unlock-locked-test",
        "--output",
        locked_predictions,
    )
    outcomes = tmp_path / "locked-outcomes.jsonl"
    _run(
        ROOT / "tools/simulate-multifidelity-policy.py",
        "--policy-freeze-manifest",
        policy_freeze,
        "--predictions",
        locked_predictions,
        "--output",
        outcomes,
    )
    policy_report = tmp_path / "policy-report.json"
    policy_result = _run(
        ROOT / "tools/evaluate-multifidelity-policy.py",
        "--outcomes",
        outcomes,
        "--output",
        policy_report,
        "--recall-floor",
        0.20,
        "--confidence",
        0.80,
        "--bootstrap-resamples",
        50,
        "--minimum-groups",
        2,
        "--minimum-winner-groups",
        2,
    )
    assert policy_result["winners"] == 4

    probe_report = tmp_path / "probe-report.json"
    probe_result = _run(
        ROOT / "tools/evaluate-multifidelity-probe.py",
        "--predictions",
        locked_predictions,
        "--output",
        probe_report,
        "--budget",
        10,
        "--confidence",
        0.80,
        "--bootstrap-resamples",
        50,
        "--minimum-groups",
        2,
    )
    assert probe_result["runs"] == 8
