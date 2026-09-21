#!/usr/bin/env python3
"""Exploratory matched-control analysis for the Fork-to-zero mechanism test."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEED = 20260921
BOOTSTRAPS = 10_000
MAIN_FEATURES = ("main_256", "main_512", "main_1024")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ranked[end] == ranked[start]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return result


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = _average_ranks(np.asarray(list(left), dtype=float))
    y = _average_ranks(np.asarray(list(right), dtype=float))
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_interval(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def bootstrap_statistic(
    rows: list[dict[str, Any]],
    statistic,
    *,
    seed: int,
    resamples: int = BOOTSTRAPS,
) -> tuple[float, float, float]:
    estimate = float(statistic(rows))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(resamples):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        samples.append(float(statistic(sample)))
    lower, upper = _quantile_interval(samples)
    return estimate, lower, upper


def _fitness(observation: dict[str, Any]) -> float:
    return float(observation["features"]["main_fitness"])


def _lineage(root: Path, trajectory: Path) -> str:
    parts = trajectory.relative_to(root).parts
    if "implementations" not in parts:
        return f"{root.name}/" + "/".join(parts[:-1])
    boundary = parts.index("implementations")
    return f"{root.name}/" + "/".join(parts[:boundary])


def collect(campaign_roots: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclusions = Counter()
    incomplete_failures: list[dict[str, Any]] = []
    source_digest = sha256()
    plan_hashes: set[str] = set()
    for root in campaign_roots:
        for path in sorted(root.rglob("multifidelity-trajectory.json")):
            raw = path.read_bytes()
            source_digest.update(str(path.relative_to(root)).encode())
            source_digest.update(raw)
            try:
                trajectory = json.loads(raw)
            except json.JSONDecodeError:
                exclusions["unreadable"] += 1
                continue
            exclusions["all_trajectories"] += 1
            is_baseline = "baselines" in path.parts
            if is_baseline:
                exclusions["baselines"] += 1
            if not trajectory:
                exclusions["empty"] += 1
                continue
            complete = trajectory[-1].get("budget_batches") == 4096
            if not complete:
                exclusions["incomplete"] += 1
                for log in sorted(path.parent.rglob("wrapper.stderr.log")):
                    feedback: dict[str, Any] | None = None
                    for line in reversed(log.read_text(encoding="utf-8").splitlines()):
                        marker = "[gigaevo] structured feedback:"
                        if marker not in line:
                            continue
                        try:
                            value = json.loads(line.split(marker, 1)[1].strip())
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict):
                            feedback = value
                            break
                    if not feedback or feedback.get("status") in {
                        "screen_complete",
                        "multifidelity_complete",
                    }:
                        continue
                    incomplete_failures.append(
                        {
                            "campaign": root.name,
                            "commit": str(trajectory[0].get("commit")),
                            "last_budget": int(trajectory[-1]["budget_batches"]),
                            "stage": str(log.relative_to(path.parent)),
                            "status": feedback.get("status"),
                            "error_type": feedback.get("error_type"),
                            "nonfinite_batch": (
                                feedback.get("nonfinite_gradient") or {}
                            ).get("batch"),
                        }
                    )
                continue
            by_budget = {int(item["budget_batches"]): item for item in trajectory}
            if not all(budget in by_budget for budget in (256, 512, 1024, 4096)):
                exclusions["missing_required_rung"] += 1
                continue
            matched = by_budget[1024]
            metadata = matched.get("metadata", {})
            valid_pair = bool(metadata.get("control_is_valid")) and bool(
                metadata.get("probe_is_valid")
            )
            if not valid_pair:
                exclusions["invalid_matched_pair"] += 1
                continue
            features = matched["features"]
            control = features.get("control_fitness")
            probe = features.get("probe_fitness")
            if control is None or probe is None:
                exclusions["missing_matched_fitness"] += 1
                continue
            plan_hash = metadata.get("plan_sha256")
            if plan_hash:
                plan_hashes.add(str(plan_hash))
            row = {
                "run_id": str(path.parent),
                "campaign": root.name,
                "group_id": _lineage(root, path),
                "commit": str(trajectory[0]["commit"]),
                "is_baseline": is_baseline,
                "main_256": _fitness(by_budget[256]),
                "main_512": _fitness(by_budget[512]),
                "main_1024": _fitness(matched),
                "control_fitness": float(control),
                "probe_fitness": float(probe),
                "fork_effect": float(probe) - float(control),
                "final_fitness": _fitness(by_budget[4096]),
            }
            row["future_gain_from_control"] = row["final_fitness"] - row[
                "control_fitness"
            ]
            rows.append(row)
    if len(plan_hashes) != 1:
        raise RuntimeError(f"expected one plan hash, found {sorted(plan_hashes)}")
    return rows, {
        "source_sha256": source_digest.hexdigest(),
        "plan_sha256": next(iter(plan_hashes)),
        "exclusions": dict(exclusions),
        "incomplete_failure_audit": incomplete_failures,
    }


def aggregate_commits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["commit"]].append(row)
    numeric = (
        *MAIN_FEATURES,
        "control_fitness",
        "probe_fitness",
        "fork_effect",
        "final_fitness",
        "future_gain_from_control",
    )
    result = []
    for commit, repetitions in sorted(grouped.items()):
        campaigns = sorted({str(row["campaign"]) for row in repetitions})
        item: dict[str, Any] = {
            "commit": commit,
            "campaign": campaigns[0] if len(campaigns) == 1 else "cross_campaign",
            "campaigns": ";".join(campaigns),
            "group_ids": ";".join(sorted({str(row["group_id"]) for row in repetitions})),
            "replicates": len(repetitions),
        }
        for name in numeric:
            values = np.asarray([float(row[name]) for row in repetitions])
            item[name] = float(np.mean(values))
            item[f"{name}_replicate_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def correlation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, signal in enumerate(
        (*MAIN_FEATURES, "control_fitness", "probe_fitness", "fork_effect")
    ):
        estimate, lower, upper = bootstrap_statistic(
            rows,
            lambda sample, name=signal: spearman(
                [row[name] for row in sample],
                [row["final_fitness"] for row in sample],
            ),
            seed=SEED + index,
        )
        result.append(
            {
                "signal": signal,
                "n_unique_commits": len(rows),
                "spearman_final": estimate,
                "commit_bootstrap_ci95_lower": lower,
                "commit_bootstrap_ci95_upper": upper,
            }
        )
    difference, lower, upper = bootstrap_statistic(
        rows,
        lambda sample: spearman(
            [row["probe_fitness"] for row in sample],
            [row["final_fitness"] for row in sample],
        )
        - spearman(
            [row["control_fitness"] for row in sample],
            [row["final_fitness"] for row in sample],
        ),
        seed=SEED + 20,
    )
    result.append(
        {
            "signal": "probe_minus_control_correlation",
            "n_unique_commits": len(rows),
            "spearman_final": difference,
            "commit_bootstrap_ci95_lower": lower,
            "commit_bootstrap_ci95_upper": upper,
        }
    )
    return result


def _fit_ridge(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    l2: float,
) -> np.ndarray:
    x_train = np.asarray([[float(row[name]) for name in features] for row in train])
    x_test = np.asarray([[float(row[name]) for name in features] for row in test])
    target = np.asarray([float(row["final_fitness"]) for row in train])
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales < 1.0e-12] = 1.0
    x_train = (x_train - means) / scales
    x_test = (x_test - means) / scales
    design = np.column_stack((np.ones(len(x_train)), x_train))
    test_design = np.column_stack((np.ones(len(x_test)), x_test))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 1.0e-10
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return test_design @ weights


MODEL_FEATURES = {
    "trajectory": MAIN_FEATURES,
    "trajectory_plus_control": (*MAIN_FEATURES, "control_fitness"),
    "trajectory_plus_fork_to_zero": (*MAIN_FEATURES, "probe_fitness"),
    "trajectory_plus_both": (
        *MAIN_FEATURES,
        "control_fitness",
        "probe_fitness",
    ),
}


def cross_campaign_predictions(
    rows: list[dict[str, Any]], *, l2: float
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["campaign"] != "cross_campaign"]
    campaigns = sorted({str(row["campaign"]) for row in eligible})
    if len(campaigns) != 2:
        raise RuntimeError(f"expected two campaigns, found {campaigns}")
    predictions = []
    for held_out in campaigns:
        train = [row for row in eligible if row["campaign"] != held_out]
        test = [row for row in eligible if row["campaign"] == held_out]
        by_variant = {
            variant: _fit_ridge(train, test, features, l2=l2)
            for variant, features in MODEL_FEATURES.items()
        }
        for index, row in enumerate(test):
            item = {
                "commit": row["commit"],
                "campaign": held_out,
                "final_fitness": row["final_fitness"],
            }
            for variant, values in by_variant.items():
                item[variant] = float(values[index])
            predictions.append(item)
    return predictions


def model_table(
    predictions: list[dict[str, Any]], *, l2: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = []
    actual = np.asarray([float(row["final_fitness"]) for row in predictions])
    for variant in MODEL_FEATURES:
        predicted = np.asarray([float(row[variant]) for row in predictions])
        result.append(
            {
                "ridge_l2": l2,
                "variant": variant,
                "features": ";".join(MODEL_FEATURES[variant]),
                "n_commits": len(predictions),
                "mae": float(np.mean(np.abs(predicted - actual))),
                "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                "prediction_spearman": spearman(predicted, actual),
            }
        )
    squared_gain = []
    absolute_gain = []
    for row in predictions:
        target = float(row["final_fitness"])
        control_error = float(row["trajectory_plus_control"]) - target
        probe_error = float(row["trajectory_plus_fork_to_zero"]) - target
        squared_gain.append(control_error**2 - probe_error**2)
        absolute_gain.append(abs(control_error) - abs(probe_error))
    gain_rows = [
        {
            **row,
            "squared_error_gain": squared_gain[index],
            "absolute_error_gain": absolute_gain[index],
        }
        for index, row in enumerate(predictions)
    ]
    squared = bootstrap_statistic(
        gain_rows,
        lambda sample: float(np.mean([row["squared_error_gain"] for row in sample])),
        seed=SEED + 30,
    )
    absolute = bootstrap_statistic(
        gain_rows,
        lambda sample: float(np.mean([row["absolute_error_gain"] for row in sample])),
        seed=SEED + 31,
    )
    gain = {
        "squared_error_gain_control_minus_fork": squared[0],
        "squared_error_gain_ci95_lower": squared[1],
        "squared_error_gain_ci95_upper": squared[2],
        "absolute_error_gain_control_minus_fork": absolute[0],
        "absolute_error_gain_ci95_lower": absolute[1],
        "absolute_error_gain_ci95_upper": absolute[2],
        "campaign_squared_error_gain": {
            campaign: float(
                np.mean(
                    [
                        row["squared_error_gain"]
                        for row in gain_rows
                        if row["campaign"] == campaign
                    ]
                )
            )
            for campaign in sorted({str(row["campaign"]) for row in gain_rows})
        },
    }
    return result, gain


def _pairwise_accuracy(
    rows: list[dict[str, Any]], signal: str, *, reversal_only: bool = False
) -> float:
    scores = []
    for left, right in itertools.combinations(rows, 2):
        final_delta = float(left["final_fitness"]) - float(right["final_fitness"])
        if final_delta == 0.0:
            continue
        main_delta = float(left["main_1024"]) - float(right["main_1024"])
        if reversal_only and main_delta * final_delta >= 0.0:
            continue
        signal_delta = float(left[signal]) - float(right[signal])
        if signal_delta == 0.0:
            scores.append(0.5)
        else:
            scores.append(float(signal_delta * final_delta > 0.0))
    return float(np.mean(scores)) if scores else float("nan")


def selection_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    result = []
    for group_id, candidates in sorted(grouped.items()):
        if len(candidates) < 2:
            continue
        best = max(float(row["final_fitness"]) for row in candidates)
        item: dict[str, Any] = {
            "group_id": group_id,
            "campaign": candidates[0]["campaign"],
            "n_candidates": len(candidates),
        }
        for signal in ("main_1024", "control_fitness", "probe_fitness"):
            selected = max(candidates, key=lambda row: float(row[signal]))
            item[f"{signal}_regret"] = best - float(selected["final_fitness"])
            item[f"{signal}_top1_hit"] = float(
                math.isclose(float(selected["final_fitness"]), best)
            )
            item[f"{signal}_pairwise_accuracy"] = _pairwise_accuracy(
                candidates, signal
            )
            item[f"{signal}_reversal_accuracy"] = _pairwise_accuracy(
                candidates, signal, reversal_only=True
            )
        result.append(item)
    return result


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for signal in ("main_1024", "control_fitness", "probe_fitness"):
        for metric in ("regret", "top1_hit", "pairwise_accuracy", "reversal_accuracy"):
            key = f"{signal}_{metric}"
            values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
            summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    for metric, direction in (
        ("regret", "control_minus_probe"),
        ("top1_hit", "probe_minus_control"),
        ("pairwise_accuracy", "probe_minus_control"),
        ("reversal_accuracy", "probe_minus_control"),
    ):
        control = f"control_fitness_{metric}"
        probe = f"probe_fitness_{metric}"
        usable = [
            row
            for row in rows
            if math.isfinite(float(row[control])) and math.isfinite(float(row[probe]))
        ]
        sign = 1.0 if direction == "probe_minus_control" else -1.0
        estimate, lower, upper = bootstrap_statistic(
            usable,
            lambda sample, c=control, p=probe, s=sign: float(
                np.mean([s * (float(row[p]) - float(row[c])) for row in sample])
            ),
            seed=SEED + 40 + len(summary),
        )
        summary[f"{metric}_gain_{direction}"] = estimate
        summary[f"{metric}_gain_lineage_bootstrap_ci95"] = [lower, upper]
        summary[f"{metric}_campaign_means"] = {
            campaign: float(
                np.mean(
                    [
                        sign * (float(row[probe]) - float(row[control]))
                        for row in usable
                        if row["campaign"] == campaign
                    ]
                )
            )
            for campaign in sorted({str(row["campaign"]) for row in usable})
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_rows, provenance = collect([path.resolve() for path in args.campaign_root])
    experimental = [row for row in all_rows if not row["is_baseline"]]
    commits = aggregate_commits(experimental)
    correlations = correlation_table(commits)

    predictions = cross_campaign_predictions(commits, l2=1.0)
    models, model_gain = model_table(predictions, l2=1.0)
    extreme_fork_effect = min(commits, key=lambda row: float(row["fork_effect"]))
    commits_without_extreme = [
        row for row in commits if row["commit"] != extreme_fork_effect["commit"]
    ]
    robust_predictions = cross_campaign_predictions(commits_without_extreme, l2=1.0)
    robust_models, robust_gain = model_table(robust_predictions, l2=1.0)
    sensitivity = []
    for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
        current_predictions = cross_campaign_predictions(commits, l2=l2)
        current_models, current_gain = model_table(current_predictions, l2=l2)
        sensitivity.extend(current_models)
        sensitivity.append(
            {
                "ridge_l2": l2,
                "variant": "fork_vs_control_gain",
                "features": "paired comparison",
                "n_commits": len(current_predictions),
                "mae": current_gain["absolute_error_gain_control_minus_fork"],
                "rmse": current_gain["squared_error_gain_control_minus_fork"],
                "prediction_spearman": float("nan"),
            }
        )

    selections = selection_table(experimental)
    selection_stats = selection_summary(selections)
    effect_stats = {
        "mean": float(np.mean([row["fork_effect"] for row in commits])),
        "median": float(np.median([row["fork_effect"] for row in commits])),
        "sd": float(np.std([row["fork_effect"] for row in commits], ddof=1)),
        "positive_fraction": float(
            np.mean([float(row["fork_effect"] > 0.0) for row in commits])
        ),
        "probe_control_spearman": spearman(
            [row["probe_fitness"] for row in commits],
            [row["control_fitness"] for row in commits],
        ),
        "effect_vs_future_gain_spearman": spearman(
            [row["fork_effect"] for row in commits],
            [row["future_gain_from_control"] for row in commits],
        ),
    }

    repeated = [row for row in commits if int(row["replicates"]) > 1]
    repeatability = {
        "repeated_commits": len(repeated),
        "repeated_runs": sum(int(row["replicates"]) for row in repeated),
        "max_replicates": max((int(row["replicates"]) for row in repeated), default=1),
        "median_probe_replicate_sd": float(
            np.median([row["probe_fitness_replicate_sd"] for row in repeated])
        )
        if repeated
        else None,
        "median_control_replicate_sd": float(
            np.median([row["control_fitness_replicate_sd"] for row in repeated])
        )
        if repeated
        else None,
    }

    write_csv(output / "eligible_runs.csv", experimental)
    write_csv(output / "unique_commits.csv", commits)
    write_csv(output / "correlations.csv", correlations)
    write_csv(output / "cross_campaign_predictions.csv", predictions)
    write_csv(output / "model_comparison.csv", models)
    write_csv(output / "model_sensitivity.csv", sensitivity)
    write_csv(output / "selection_by_lineage.csv", selections)

    summary = {
        "analysis_type": "exploratory_post_collection",
        **provenance,
        "campaigns": [path.name for path in args.campaign_root],
        "eligible_runs_including_baselines": len(all_rows),
        "eligible_experimental_runs": len(experimental),
        "experimental_lineages": len({row["group_id"] for row in experimental}),
        "unique_experimental_commits": len(commits),
        "cross_campaign_model_commits": len(predictions),
        "independent_campaigns": 2,
        "correlations": correlations,
        "model_comparison": models,
        "fork_vs_control_model_gain": model_gain,
        "extreme_fork_effect_sensitivity": {
            "excluded_commit": extreme_fork_effect["commit"],
            "fork_effect": extreme_fork_effect["fork_effect"],
            "main_1024": extreme_fork_effect["main_1024"],
            "control_fitness": extreme_fork_effect["control_fitness"],
            "probe_fitness": extreme_fork_effect["probe_fitness"],
            "final_fitness": extreme_fork_effect["final_fitness"],
            "model_comparison_without_commit": robust_models,
            "fork_vs_control_gain_without_commit": robust_gain,
        },
        "selection": selection_stats,
        "fork_effect": effect_stats,
        "repeatability": repeatability,
        "inference_limit": (
            "The 12 idea lineages form two sequential evolutionary campaigns. "
            "Repeated parent commits connect lineages within each campaign, so only "
            "two campaign-level clusters are independent. Commit/lineage bootstrap "
            "intervals are exploratory, not confirmatory."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    corr = {row["signal"]: row for row in correlations}
    model = {row["variant"]: row for row in models}
    gain = model_gain
    selection = selection_stats
    campaign_gains = gain["campaign_squared_error_gain"]
    model_direction_consistent = all(value > 0.0 for value in campaign_gains.values())
    model_ci_positive = gain["squared_error_gain_ci95_lower"] > 0.0
    regret_campaigns = selection["regret_campaign_means"]
    regret_direction_consistent = all(value >= 0.0 for value in regret_campaigns.values())
    direct_ci = corr["probe_minus_control_correlation"]
    direct_positive = direct_ci["commit_bootstrap_ci95_lower"] > 0.0
    selection_positive = (
        selection["regret_gain_lineage_bootstrap_ci95"][0] > 0.0
        and regret_direction_consistent
    )
    robust_positive = (
        robust_gain["squared_error_gain_control_minus_fork"] > 0.0
        and robust_gain["absolute_error_gain_control_minus_fork"] > 0.0
    )
    if (
        model_direction_consistent
        and model_ci_positive
        and direct_positive
        and selection_positive
        and robust_positive
    ):
        verdict = "positive exploratory evidence"
    else:
        verdict = (
            "not supported for candidate ranking; possible instability "
            "stress-test signal"
        )
    summary["verdict"] = verdict
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fork-to-zero matched-control analysis",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "This analysis compares a 256-batch LR-to-zero fork at batch 1,024 "
        "against an equal-compute continuation of the original scheduler. All "
        "reported predictive targets are the untouched main-branch fitness at batch 4,096.",
        "",
        "## Dataset",
        "",
        f"- {len(experimental)} complete experimental runs with valid matched pairs.",
        f"- {len(commits)} unique experimental commits across "
        f"{summary['experimental_lineages']} idea lineages and two campaigns.",
        f"- Cross-campaign models use {len(predictions)} commits; commits observed in "
        "both campaigns are excluded from that comparison.",
        f"- Exclusions: {json.dumps(provenance['exclusions'], sort_keys=True)}.",
        "",
        "## Direct signal",
        "",
        f"- Equal-compute control vs final: rho={corr['control_fitness']['spearman_final']:.3f} "
        f"(exploratory 95% commit-bootstrap CI "
        f"{corr['control_fitness']['commit_bootstrap_ci95_lower']:.3f} to "
        f"{corr['control_fitness']['commit_bootstrap_ci95_upper']:.3f}).",
        f"- Fork-to-zero vs final: rho={corr['probe_fitness']['spearman_final']:.3f} "
        f"(CI {corr['probe_fitness']['commit_bootstrap_ci95_lower']:.3f} to "
        f"{corr['probe_fitness']['commit_bootstrap_ci95_upper']:.3f}).",
        f"- Difference in correlation (fork minus control): "
        f"{corr['probe_minus_control_correlation']['spearman_final']:.3f} "
        f"(CI {corr['probe_minus_control_correlation']['commit_bootstrap_ci95_lower']:.3f} "
        f"to {corr['probe_minus_control_correlation']['commit_bootstrap_ci95_upper']:.3f}).",
        f"- Fork effect Z-C vs final: rho={corr['fork_effect']['spearman_final']:.3f}.",
        f"- Fork and control signals correlate at rho={effect_stats['probe_control_spearman']:.3f}.",
        "",
        "## Cross-campaign prediction",
        "",
        f"- Trajectory + control: MAE={model['trajectory_plus_control']['mae']:.6f}, "
        f"RMSE={model['trajectory_plus_control']['rmse']:.6f}.",
        f"- Trajectory + Fork-to-zero: MAE={model['trajectory_plus_fork_to_zero']['mae']:.6f}, "
        f"RMSE={model['trajectory_plus_fork_to_zero']['rmse']:.6f}.",
        f"- Paired squared-error gain, control minus fork: "
        f"{gain['squared_error_gain_control_minus_fork']:.9f} "
        f"(exploratory 95% commit-bootstrap CI "
        f"{gain['squared_error_gain_ci95_lower']:.9f} to "
        f"{gain['squared_error_gain_ci95_upper']:.9f}).",
        f"- Campaign-specific squared-error gains: {json.dumps(campaign_gains, sort_keys=True)}.",
        f"- The apparent gain is dominated by commit `{extreme_fork_effect['commit']}`: "
        f"main@1024={extreme_fork_effect['main_1024']:.6f}, "
        f"control={extreme_fork_effect['control_fitness']:.6f}, "
        f"fork={extreme_fork_effect['probe_fitness']:.6f}, and "
        f"final={extreme_fork_effect['final_fitness']:.6f}.",
        f"- Without that commit, control MAE="
        f"{next(row for row in robust_models if row['variant'] == 'trajectory_plus_control')['mae']:.6f} "
        f"and Fork-to-zero MAE="
        f"{next(row for row in robust_models if row['variant'] == 'trajectory_plus_fork_to_zero')['mae']:.6f}; "
        f"the absolute-error gain becomes "
        f"{robust_gain['absolute_error_gain_control_minus_fork']:.9f}.",
        "",
        "## Candidate selection within idea lineages",
        "",
        f"- Mean top-1 regret, control: {selection['mean_control_fitness_regret']:.6f}.",
        f"- Mean top-1 regret, Fork-to-zero: {selection['mean_probe_fitness_regret']:.6f}.",
        f"- Regret gain, control minus fork: "
        f"{selection['regret_gain_control_minus_probe']:.6f} "
        f"(lineage-bootstrap CI "
        f"{selection['regret_gain_lineage_bootstrap_ci95'][0]:.6f} to "
        f"{selection['regret_gain_lineage_bootstrap_ci95'][1]:.6f}).",
        f"- Pairwise-ranking gain, fork minus control: "
        f"{selection['pairwise_accuracy_gain_probe_minus_control']:.3f}.",
        f"- Early-rank-reversal accuracy gain, fork minus control: "
        f"{selection['reversal_accuracy_gain_probe_minus_control']:.3f}.",
        f"- For reference, raw main@1024 has mean top-1 regret "
        f"{selection['mean_main_1024_regret']:.6f} and top-1 hit rate "
        f"{selection['mean_main_1024_top1_hit']:.3f}, versus Fork-to-zero "
        f"regret {selection['mean_probe_fitness_regret']:.6f} and hit rate "
        f"{selection['mean_probe_fitness_top1_hit']:.3f}.",
        "",
        "## Incomplete-run audit",
        "",
        *[
            f"- `{item['commit']}` at {item['stage']}: "
            f"status={item['status']}, error={item['error_type']}, "
            f"nonfinite_batch={item['nonfinite_batch']}."
            for item in provenance["incomplete_failure_audit"]
        ],
        "",
        "## Interpretation limits",
        "",
        "The comparison is post-collection exploratory. More importantly, repeated "
        "parent commits connect the six sequential lineages on each GPU. The dataset "
        "therefore contains only two independent campaign-level clusters, not twelve "
        "independent experiments. The next confirmatory run must freeze this analysis "
        "and use independently seeded candidate pools rather than a sequential chain.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": verdict, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
