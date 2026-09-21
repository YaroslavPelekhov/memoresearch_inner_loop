#!/usr/bin/env python3
"""Frozen confirmatory analysis for Fork-to-zero instability detection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

SEED = 20260921
BOOTSTRAPS = 20_000
CHECKPOINTS = (512, 1024)
ALARM_MARGIN = -0.002
COLLAPSE_LOSS_RATIO = 1.10
GRADIENT_CLIP_FRACTION = 0.50
GRADIENT_NORM_P95 = 10.0
REQUIRED_CAMPAIGNS = 8
MIN_POSITIVES = 10
MIN_NEGATIVES = 20
EXPECTED_CAMPAIGNS = {
    f"fork-zero-stability-v1-gpu{gpu}-pool{pool}"
    for gpu in (0, 1)
    for pool in range(1, 5)
}
INTRINSIC_STATUSES = {
    "nonfinite_training_gradient",
    "nonfinite_training_loss",
}
INFRASTRUCTURE_ERRORS = {
    "cuda_oom",
    "checkpoint_failure",
    "dtype_mismatch",
    "kernel_compile_failure",
    "network_failure",
    "storage_exhausted",
    "timeout",
}
MARKER = "[gigaevo] structured feedback:"


def _last_feedback(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(
        path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if MARKER not in line:
            continue
        try:
            value = json.loads(line.split(MARKER, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _intrinsic_failure(feedback: dict[str, Any] | None) -> bool:
    return bool(feedback and feedback.get("status") in INTRINSIC_STATUSES)


def _failure_batch(feedback: dict[str, Any] | None) -> int | None:
    if not feedback:
        return None
    for key in ("nonfinite_gradient", "nonfinite_loss"):
        detail = feedback.get(key)
        if isinstance(detail, dict) and isinstance(detail.get("batch"), int):
            return int(detail["batch"])
    return None


def _fitness(observation: dict[str, Any]) -> float | None:
    value = observation.get("features", {}).get("main_fitness")
    return float(value) if isinstance(value, int | float) else None


def _loss(observation: dict[str, Any]) -> float | None:
    value = observation.get("main_metrics", {}).get("heldout_loss_final")
    return float(value) if isinstance(value, int | float) else None


def classify_outcome(
    checkpoint: dict[str, Any],
    final: dict[str, Any] | None,
    later_main_failures: list[dict[str, Any]],
) -> tuple[bool | None, str, int | None]:
    """Apply the preregistered collapse label without looking at branch values."""

    intrinsic = [item for item in later_main_failures if _intrinsic_failure(item)]
    if intrinsic:
        batches = [
            value for value in map(_failure_batch, intrinsic) if value is not None
        ]
        return True, "later_nonfinite_main", min(batches) if batches else None
    if final is None:
        return None, "censored_incomplete_main", None
    checkpoint_loss = _loss(checkpoint)
    final_loss = _loss(final)
    if checkpoint_loss is None or final_loss is None:
        return None, "censored_missing_loss", None
    collapsed = final_loss >= COLLAPSE_LOSS_RATIO * checkpoint_loss
    return collapsed, "terminal_loss_ratio" if collapsed else "stable_complete", 4096


def detector_values(observation: dict[str, Any]) -> dict[str, Any]:
    """Apply frozen alarms to one matched checkpoint."""

    features = observation.get("features", {})
    metrics = observation.get("main_metrics", {})
    metadata = observation.get("metadata", {})
    main = features.get("main_fitness")
    control = features.get("control_fitness")
    probe = features.get("probe_fitness")
    control_feedback = metadata.get("control_feedback")
    probe_feedback = metadata.get("probe_feedback")
    control_valid = bool(metadata.get("control_is_valid")) and isinstance(
        control, int | float
    )
    probe_valid = bool(metadata.get("probe_is_valid")) and isinstance(
        probe, int | float
    )

    control_alarm: bool | None
    control_score: float | None
    if _intrinsic_failure(control_feedback):
        control_alarm, control_score = True, 1.0e9
    elif control_valid and isinstance(main, int | float):
        control_delta = float(control) - float(main)
        control_alarm = control_delta <= ALARM_MARGIN
        control_score = -control_delta
    else:
        control_alarm, control_score = None, None

    fork_alarm: bool | None
    fork_score: float | None
    if _intrinsic_failure(probe_feedback):
        fork_alarm, fork_score = True, 1.0e9
    elif probe_valid and control_valid:
        fork_delta = float(probe) - float(control)
        fork_alarm = fork_delta <= ALARM_MARGIN
        fork_score = -fork_delta
    else:
        fork_alarm, fork_score = None, None

    clip_fraction = metrics.get("gradient_clipping_fraction")
    norm_p95 = metrics.get("gradient_norm_pre_clip_p95")
    gradient_available = isinstance(clip_fraction, int | float) or isinstance(
        norm_p95, int | float
    )
    gradient_score = None
    gradient_alarm = None
    if gradient_available:
        normalized = []
        if isinstance(clip_fraction, int | float):
            normalized.append(float(clip_fraction) / GRADIENT_CLIP_FRACTION)
        if isinstance(norm_p95, int | float):
            normalized.append(float(norm_p95) / GRADIENT_NORM_P95)
        gradient_score = max(normalized)
        gradient_alarm = gradient_score >= 1.0

    return {
        "fork_alarm": fork_alarm,
        "fork_score": fork_score,
        "control_alarm": control_alarm,
        "control_score": control_score,
        "gradient_alarm": gradient_alarm,
        "gradient_score": gradient_score,
    }


def _main_failures(training_root: Path, after_batch: int) -> list[dict[str, Any]]:
    result = []
    for path in sorted(training_root.glob("main/budget-*/wrapper.stderr.log")):
        match = re.search(r"budget-(\d+)", str(path.parent))
        if match is None or int(match.group(1)) <= after_batch:
            continue
        feedback = _last_feedback(path)
        if feedback:
            result.append(feedback)
    return result


def _gpu_from_campaign(name: str) -> str:
    match = re.search(r"gpu(\d+)", name)
    return match.group(1) if match else "unknown"


def collect(
    campaign_roots: list[Path], expected_plan_hash: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = Counter()
    source_digest = sha256()
    observed_plan_hashes: set[str] = set()
    failures: list[dict[str, Any]] = []
    for root in campaign_roots:
        campaign = root.name
        for path in sorted(root.rglob("multifidelity-trajectory.json")):
            raw = path.read_bytes()
            source_digest.update(campaign.encode())
            source_digest.update(str(path.relative_to(root)).encode())
            source_digest.update(raw)
            audit["trajectories"] += 1
            try:
                trajectory = json.loads(raw)
            except json.JSONDecodeError:
                audit["unreadable"] += 1
                continue
            if not isinstance(trajectory, list) or not trajectory:
                audit["empty"] += 1
                continue
            baseline = "baselines" in path.parts
            if baseline:
                audit["baselines"] += 1
            by_budget = {
                int(item["budget_batches"]): item
                for item in trajectory
                if isinstance(item, dict) and "budget_batches" in item
            }
            commit = str(trajectory[0].get("commit", "unknown"))
            final = by_budget.get(4096)
            for checkpoint_batch in CHECKPOINTS:
                checkpoint = by_budget.get(checkpoint_batch)
                if checkpoint is None:
                    audit[f"missing_checkpoint_{checkpoint_batch}"] += 1
                    continue
                plan_hash = checkpoint.get("metadata", {}).get("plan_sha256")
                if isinstance(plan_hash, str):
                    observed_plan_hashes.add(plan_hash)
                outcome, outcome_reason, outcome_batch = classify_outcome(
                    checkpoint,
                    final,
                    _main_failures(path.parent, checkpoint_batch),
                )
                detectors = detector_values(checkpoint)
                if outcome is None:
                    audit["censored_checkpoint_rows"] += 1
                if (
                    detectors["fork_alarm"] is None
                    or detectors["control_alarm"] is None
                ):
                    audit["unpaired_checkpoint_rows"] += 1
                rows.append(
                    {
                        "campaign": campaign,
                        "gpu": _gpu_from_campaign(campaign),
                        "commit": commit,
                        "run_id": str(path.parent),
                        "is_baseline": baseline,
                        "checkpoint": checkpoint_batch,
                        "outcome": outcome,
                        "outcome_reason": outcome_reason,
                        "outcome_batch": outcome_batch,
                        **detectors,
                    }
                )
            if final is None:
                later = _main_failures(path.parent, 0)
                failures.append(
                    {
                        "campaign": campaign,
                        "commit": commit,
                        "last_successful_budget": max(by_budget, default=0),
                        "statuses": ";".join(
                            str(item.get("status", "unknown")) for item in later
                        ),
                        "error_types": ";".join(
                            str(item.get("error_type", "")) for item in later
                        ),
                    }
                )
    if observed_plan_hashes != {expected_plan_hash}:
        raise RuntimeError(
            "campaign plan hash mismatch: "
            f"expected {expected_plan_hash}, observed {sorted(observed_plan_hashes)}"
        )
    return rows, {
        "source_sha256": source_digest.hexdigest(),
        "plan_sha256": expected_plan_hash,
        "audit": dict(audit),
        "incomplete_runs": failures,
    }


def aggregate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce checkpoints and within-campaign repeated commits conservatively."""

    # GigaEvo may evaluate the canonical commit again when it is included in the
    # seed refs.  The preregistration excludes canonical baselines, regardless
    # of which artifact directory contains the repeated trajectory.
    baseline_commits = {
        str(row["commit"]) for row in rows if bool(row["is_baseline"])
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["is_baseline"] or str(row["commit"]) in baseline_commits:
            continue
        grouped[(str(row["campaign"]), str(row["commit"]))].append(row)
    result = []
    for (campaign, commit), values in sorted(grouped.items()):
        known_outcomes = [
            bool(row["outcome"]) for row in values if row["outcome"] is not None
        ]
        outcome = any(known_outcomes) if known_outcomes else None
        paired = [
            row
            for row in values
            if row["outcome"] is not None
            and row["fork_alarm"] is not None
            and row["control_alarm"] is not None
        ]
        item: dict[str, Any] = {
            "campaign": campaign,
            "gpu": values[0]["gpu"],
            "commit": commit,
            "replicate_checkpoint_rows": len(values),
            "outcome": outcome,
            "eligible_primary": bool(paired) and outcome is not None,
        }
        for detector in ("fork", "control", "gradient"):
            detector_rows = paired if detector in {"fork", "control"} else values
            alarms = [
                row
                for row in detector_rows
                if row["outcome"] is not None and row[f"{detector}_alarm"] is True
            ]
            scores = [
                float(row[f"{detector}_score"])
                for row in detector_rows
                if row["outcome"] is not None and row[f"{detector}_score"] is not None
            ]
            item[f"{detector}_alarm"] = bool(alarms)
            item[f"{detector}_earliest_checkpoint"] = (
                min(int(row["checkpoint"]) for row in alarms) if alarms else None
            )
            item[f"{detector}_score"] = max(scores) if scores else None
        positive_rows = [row for row in values if row["outcome"] is True]
        item["outcome_batch"] = min(
            (
                int(row["outcome_batch"])
                for row in positive_rows
                if row["outcome_batch"] is not None
            ),
            default=None,
        )
        result.append(item)
    return result


def confusion(rows: list[dict[str, Any]], detector: str) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("eligible_primary") and row.get("outcome") is not None
    ]
    tp = sum(row["outcome"] is True and row[f"{detector}_alarm"] for row in eligible)
    fp = sum(row["outcome"] is False and row[f"{detector}_alarm"] for row in eligible)
    tn = sum(
        row["outcome"] is False and not row[f"{detector}_alarm"] for row in eligible
    )
    fn = sum(
        row["outcome"] is True and not row[f"{detector}_alarm"] for row in eligible
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "detector": detector,
        "n": len(eligible),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": ratio(tp, tp + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
    }


def _difference(rows: list[dict[str, Any]]) -> float | None:
    fork = confusion(rows, "fork")["sensitivity"]
    control = confusion(rows, "control")["sensitivity"]
    if fork is None or control is None:
        return None
    return float(fork - control)


def campaign_bootstrap(
    rows: list[dict[str, Any]], *, resamples: int = BOOTSTRAPS
) -> dict[str, Any]:
    campaigns = sorted({str(row["campaign"]) for row in rows})
    by_campaign = {
        campaign: [row for row in rows if row["campaign"] == campaign]
        for campaign in campaigns
    }
    estimate = _difference(rows)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for campaign in rng.choice(campaigns, size=len(campaigns), replace=True):
            sampled.extend(by_campaign[str(campaign)])
        value = _difference(sampled)
        if value is not None and math.isfinite(value):
            samples.append(value)
    return {
        "estimate": estimate,
        "ci95_lower": float(np.quantile(samples, 0.025)) if samples else None,
        "ci95_upper": float(np.quantile(samples, 0.975)) if samples else None,
        "resamples": resamples,
        "finite_resamples": len(samples),
    }


def roc_auc(rows: list[dict[str, Any]], detector: str) -> float | None:
    positives = [
        float(row[f"{detector}_score"])
        for row in rows
        if row.get("eligible_primary")
        and row.get("outcome") is True
        and row.get(f"{detector}_score") is not None
    ]
    negatives = [
        float(row[f"{detector}_score"])
        for row in rows
        if row.get("eligible_primary")
        and row.get("outcome") is False
        and row.get(f"{detector}_score") is not None
    ]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def average_precision(rows: list[dict[str, Any]], detector: str) -> float | None:
    scored = [
        row
        for row in rows
        if row.get("eligible_primary")
        and row.get("outcome") is not None
        and row.get(f"{detector}_score") is not None
    ]
    positives = sum(row["outcome"] is True for row in scored)
    if positives == 0:
        return None
    scored.sort(key=lambda row: float(row[f"{detector}_score"]), reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(scored, start=1):
        if row["outcome"] is True:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def lead_time(rows: list[dict[str, Any]], detector: str) -> dict[str, Any]:
    leads = []
    for row in rows:
        alarm_batch = row.get(f"{detector}_earliest_checkpoint")
        outcome_batch = row.get("outcome_batch")
        if (
            row.get("outcome") is True
            and alarm_batch is not None
            and outcome_batch is not None
        ):
            leads.append(int(outcome_batch) - int(alarm_batch))
    return {
        "detected_positive_cases": len(leads),
        "mean_lead_batches": float(np.mean(leads)) if leads else None,
        "median_lead_batches": float(np.median(leads)) if leads else None,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    keys: list[str] = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)


def evaluate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row["eligible_primary"]]
    campaigns = sorted({str(row["campaign"]) for row in eligible})
    positives = sum(row["outcome"] is True for row in eligible)
    negatives = sum(row["outcome"] is False for row in eligible)
    gpu_classes = {
        gpu: {
            bool(row["outcome"])
            for row in eligible
            if row["gpu"] == gpu and row["outcome"] is not None
        }
        for gpu in sorted({str(row["gpu"]) for row in eligible})
    }
    sufficient = (
        set(campaigns) == EXPECTED_CAMPAIGNS
        and positives >= MIN_POSITIVES
        and negatives >= MIN_NEGATIVES
        and len(gpu_classes) == 2
        and all(classes == {False, True} for classes in gpu_classes.values())
    )
    tables = {name: confusion(eligible, name) for name in ("fork", "control")}
    gradient_rows = [row for row in eligible if row.get("gradient_score") is not None]
    tables["gradient"] = confusion(gradient_rows, "gradient")
    bootstrap = campaign_bootstrap(eligible)
    gpu_differences = {
        gpu: _difference([row for row in eligible if row["gpu"] == gpu])
        for gpu in gpu_classes
    }
    fork = tables["fork"]
    supported = bool(
        sufficient
        and fork["specificity"] is not None
        and fork["specificity"] >= 0.90
        and bootstrap["estimate"] is not None
        and bootstrap["estimate"] >= 0.15
        and bootstrap["ci95_lower"] is not None
        and bootstrap["ci95_lower"] > 0.0
        and all(value is not None and value > 0.0 for value in gpu_differences.values())
    )
    if not sufficient:
        verdict = "inconclusive_insufficient_sample"
    elif supported:
        verdict = "supported_as_instability_detector"
    else:
        verdict = "not_supported_as_instability_detector"
    return {
        "verdict": verdict,
        "sample_sufficiency_passed": sufficient,
        "independent_campaigns": len(campaigns),
        "eligible_candidates": len(eligible),
        "positive_candidates": positives,
        "negative_candidates": negatives,
        "gpu_outcome_classes": {
            gpu: sorted(classes) for gpu, classes in gpu_classes.items()
        },
        "confusion": tables,
        "fork_minus_control_sensitivity": bootstrap,
        "gpu_sensitivity_differences": gpu_differences,
        "continuous_scores": {
            detector: {
                "roc_auc": roc_auc(eligible, detector),
                "average_precision": average_precision(eligible, detector),
            }
            for detector in ("fork", "control", "gradient")
        },
        "lead_time": {
            detector: lead_time(eligible, detector)
            for detector in ("fork", "control", "gradient")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = args.plan.resolve()
    expected_hash = sha256(plan.read_bytes()).hexdigest()
    checkpoint_rows, provenance = collect(
        [path.resolve() for path in args.campaign_root], expected_hash
    )
    candidates = aggregate_candidates(checkpoint_rows)
    evaluation = evaluate(candidates)
    summary = {
        "analysis_type": "preregistered_confirmatory",
        "protocol_constants": {
            "seed": SEED,
            "bootstrap_resamples": BOOTSTRAPS,
            "checkpoints": list(CHECKPOINTS),
            "alarm_margin": ALARM_MARGIN,
            "collapse_loss_ratio": COLLAPSE_LOSS_RATIO,
            "gradient_clip_fraction": GRADIENT_CLIP_FRACTION,
            "gradient_norm_p95": GRADIENT_NORM_P95,
            "required_campaigns": REQUIRED_CAMPAIGNS,
            "expected_campaign_names": sorted(EXPECTED_CAMPAIGNS),
            "minimum_positives": MIN_POSITIVES,
            "minimum_negatives": MIN_NEGATIVES,
        },
        "campaigns": [path.name for path in args.campaign_root],
        **provenance,
        **evaluation,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "checkpoint_rows.csv", checkpoint_rows)
    _write_csv(output / "candidate_detection.csv", candidates)
    _write_csv(output / "incomplete_runs.csv", provenance["incomplete_runs"])
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fork = evaluation["confusion"]["fork"]
    control = evaluation["confusion"]["control"]
    interval = evaluation["fork_minus_control_sensitivity"]
    lines = [
        "# Fork-to-zero instability detector confirmation",
        "",
        f"Verdict: **{evaluation['verdict']}**.",
        "",
        "## Sample",
        "",
        f"- Independent campaigns: {evaluation['independent_campaigns']} / {REQUIRED_CAMPAIGNS} required.",
        f"- Eligible candidates: {evaluation['eligible_candidates']}.",
        f"- Positive / negative cases: {evaluation['positive_candidates']} / {evaluation['negative_candidates']}.",
        f"- Sample sufficiency passed: {evaluation['sample_sufficiency_passed']}.",
        "",
        "## Primary comparison",
        "",
        f"- Fork sensitivity: {fork['sensitivity']}.",
        f"- Control sensitivity: {control['sensitivity']}.",
        f"- Fork specificity: {fork['specificity']}.",
        f"- Sensitivity difference: {interval['estimate']} "
        f"(campaign-bootstrap 95% CI {interval['ci95_lower']} to {interval['ci95_upper']}).",
        f"- GPU-stratified differences: {json.dumps(evaluation['gpu_sensitivity_differences'], sort_keys=True)}.",
        "",
        "## Interpretation",
        "",
        "The confirmatory claim is evaluated only by the frozen conjunction of "
        "sample sufficiency, specificity, effect size, campaign-bootstrap lower "
        "bound, and GPU-direction checks. Candidate ranking is outside scope.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), **evaluation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
