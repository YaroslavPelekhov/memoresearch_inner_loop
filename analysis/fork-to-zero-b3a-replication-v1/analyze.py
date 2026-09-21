#!/usr/bin/env python3
"""Frozen analysis for the paired b3a210 replication and onset map."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

SEEDS = (31001, 31002, 31003, 31004, 31005, 31006)
CHECKPOINTS = (512, 768, 1024)
FINAL_BATCH = 4096
ALARM_MARGIN = -0.002
COLLAPSE_LOSS_RATIO = 1.10
INTRINSIC_STATUSES = {"nonfinite_training_gradient", "nonfinite_training_loss"}
MARKER = "[gigaevo] structured feedback:"


def _feedback(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if MARKER not in line:
            continue
        try:
            value = json.loads(line.split(MARKER, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _intrinsic(feedback: dict[str, Any] | None) -> bool:
    return bool(feedback and feedback.get("status") in INTRINSIC_STATUSES)


def alarm_values(observation: dict[str, Any]) -> tuple[bool | None, bool | None]:
    features = observation.get("features", {})
    metadata = observation.get("metadata", {})
    main = features.get("main_fitness")
    control = features.get("control_fitness")
    fork = features.get("probe_fitness")
    control_valid = bool(metadata.get("control_is_valid")) and isinstance(
        control, int | float
    )
    fork_valid = bool(metadata.get("probe_is_valid")) and isinstance(
        fork, int | float
    )

    if _intrinsic(metadata.get("control_feedback")):
        control_alarm: bool | None = True
    elif control_valid and isinstance(main, int | float):
        control_alarm = float(control) - float(main) <= ALARM_MARGIN
    else:
        control_alarm = None

    if _intrinsic(metadata.get("probe_feedback")):
        fork_alarm: bool | None = True
    elif fork_valid and control_valid:
        fork_alarm = float(fork) - float(control) <= ALARM_MARGIN
    else:
        fork_alarm = None
    return fork_alarm, control_alarm


def classify_run(
    trajectory: list[dict[str, Any]], main_failures: list[dict[str, Any]]
) -> dict[str, Any]:
    by_budget = {
        int(item["budget_batches"]): item
        for item in trajectory
        if isinstance(item, dict) and "budget_batches" in item
    }
    intrinsic_failures = [item for item in main_failures if _intrinsic(item)]
    final = by_budget.get(FINAL_BATCH)
    collapse = None
    collapse_reason = "censored_incomplete_main"
    collapse_batch = None
    if intrinsic_failures:
        collapse = True
        collapse_reason = "later_nonfinite_main"
        batches: list[int] = []
        for feedback in intrinsic_failures:
            for key in ("nonfinite_gradient", "nonfinite_loss"):
                detail = feedback.get(key)
                if isinstance(detail, dict) and isinstance(detail.get("batch"), int):
                    batches.append(int(detail["batch"]))
        collapse_batch = min(batches) if batches else None
    elif final is not None:
        final_loss = final.get("main_metrics", {}).get("heldout_loss_final")
        ratios = []
        if isinstance(final_loss, int | float):
            for checkpoint in CHECKPOINTS:
                item = by_budget.get(checkpoint)
                loss = (item or {}).get("main_metrics", {}).get("heldout_loss_final")
                if isinstance(loss, int | float) and float(loss) > 0:
                    ratios.append(float(final_loss) / float(loss))
        if ratios:
            collapse = max(ratios) >= COLLAPSE_LOSS_RATIO
            collapse_reason = "terminal_loss_ratio" if collapse else "stable_complete"
            collapse_batch = FINAL_BATCH
        else:
            collapse_reason = "censored_missing_loss"

    fork_alarms = []
    control_alarms = []
    eligible = []
    for checkpoint in CHECKPOINTS:
        item = by_budget.get(checkpoint)
        if item is None:
            continue
        fork_alarm, control_alarm = alarm_values(item)
        if fork_alarm is not None and control_alarm is not None:
            eligible.append(checkpoint)
        if fork_alarm is True:
            fork_alarms.append(checkpoint)
        if control_alarm is True:
            control_alarms.append(checkpoint)
    fork_earliest = min(fork_alarms, default=None)
    control_earliest = min(control_alarms, default=None)
    fork_leads_control = bool(
        collapse is True
        and fork_earliest is not None
        and (control_earliest is None or fork_earliest < control_earliest)
        and (collapse_batch is None or fork_earliest < collapse_batch)
    )
    return {
        "collapse": collapse,
        "collapse_batch": collapse_batch,
        "collapse_reason": collapse_reason,
        "control_earliest_checkpoint": control_earliest,
        "eligible_checkpoints": eligible,
        "fork_earliest_checkpoint": fork_earliest,
        "fork_leads_control": fork_leads_control,
        "last_successful_budget": max(by_budget, default=0),
    }


def _main_failures(training: Path) -> list[dict[str, Any]]:
    failures = []
    for path in sorted(training.glob("main/budget-*/wrapper.stderr.log")):
        value = _feedback(path)
        if value:
            failures.append(value)
    return failures


def collect(campaign_root: Path, plan_hash: str) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for variant in ("candidate", "parent"):
            training = campaign_root / "pairs" / f"seed-{seed}" / variant / "training"
            trajectory_path = training / "multifidelity-trajectory.json"
            row: dict[str, Any] = {
                "seed": seed,
                "variant": variant,
                "trajectory": str(trajectory_path),
            }
            if not trajectory_path.is_file():
                row.update(
                    {
                        "collapse": None,
                        "collapse_reason": "missing_trajectory",
                        "fork_leads_control": False,
                    }
                )
                rows.append(row)
                continue
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            observed_hashes = {
                item.get("metadata", {}).get("plan_sha256")
                for item in trajectory
                if isinstance(item, dict)
            }
            if observed_hashes - {plan_hash}:
                raise RuntimeError(
                    f"plan hash mismatch for seed {seed} {variant}: {observed_hashes}"
                )
            row.update(classify_run(trajectory, _main_failures(training)))
            rows.append(row)
    return rows


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [row for row in rows if row["variant"] == "candidate"]
    parent = [row for row in rows if row["variant"] == "parent"]
    candidate_collapses = sum(row.get("collapse") is True for row in candidate)
    parent_collapses = sum(row.get("collapse") is True for row in parent)
    fork_leads = sum(bool(row.get("fork_leads_control")) for row in candidate)
    complete_matrix = len(rows) == 12 and all(
        row.get("collapse") is not None for row in rows
    )
    mechanism = complete_matrix and candidate_collapses >= 4 and parent_collapses <= 1
    early_warning = complete_matrix and fork_leads >= 4
    if mechanism and early_warning:
        verdict = "full_support"
    elif mechanism:
        verdict = "mutation_replicated_fork_advantage_not_supported"
    elif not complete_matrix:
        verdict = "inconclusive_incomplete_matrix"
    else:
        verdict = "not_replicated"
    return {
        "candidate_collapses": candidate_collapses,
        "complete_matrix": complete_matrix,
        "fork_leads_control_candidate_runs": fork_leads,
        "fork_early_warning_supported": early_warning,
        "mutation_mechanism_replicated": mechanism,
        "parent_collapses": parent_collapses,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    output = (args.output_dir or campaign_root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_hash = sha256(args.plan.read_bytes()).hexdigest()
    rows = collect(campaign_root, plan_hash)
    summary = evaluate(rows)
    summary.update(
        {
            "campaign_root": str(campaign_root),
            "plan_sha256": plan_hash,
            "protocol": "fork-to-zero-b3a-replication-v1",
        }
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# b3a210 replication and onset-map result",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Candidate collapses: {summary['candidate_collapses']}/6",
        f"- Parent collapses: {summary['parent_collapses']}/6",
        "- Candidate runs where Fork strictly leads control: "
        f"{summary['fork_leads_control_candidate_runs']}/6",
        f"- Complete matrix: {summary['complete_matrix']}",
        "",
        "The frozen rules require at least 4/6 candidate collapses, at most 1/6 "
        "parent collapses, and a strict Fork lead in at least 4/6 candidate runs.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
