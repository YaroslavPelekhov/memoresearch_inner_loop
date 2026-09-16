#!/usr/bin/env python3
"""Plot best-so-far repo-harness fitness against cumulative LLM spend."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _fitness_curve(
    nodes: list[dict[str, Any]], usage: list[dict[str, Any]]
) -> tuple[list[float], list[float], float]:
    usage_events = sorted(
        (
            _timestamp(str(record["created_at"])),
            float(record.get("estimated_cost_usd") or 0.0),
        )
        for record in usage
        if record.get("created_at")
    )
    reflection_times = {
        str(record["program_id"]): _timestamp(str(record["created_at"]))
        for record in usage
        if record.get("source") == "repo_reflection"
        and record.get("program_id")
        and record.get("created_at")
    }

    evaluations: list[tuple[datetime, float]] = []
    for node in nodes:
        fitness = float(node.get("metrics", {}).get("fitness", float("nan")))
        if not math.isfinite(fitness):
            continue
        evaluated_at = reflection_times.get(str(node.get("id")))
        if evaluated_at is None:
            # In reflection-enabled runs this is normally an invalid seed whose
            # LLM reflection was intentionally skipped. It has no attributable
            # LLM cost and cannot improve the valid best-so-far curve.
            if reflection_times:
                continue
            created_at = node.get("created_at")
            if not created_at:
                continue
            evaluated_at = _timestamp(str(created_at))
        evaluations.append((evaluated_at, fitness))
    evaluations.sort()

    costs: list[float] = []
    best_values: list[float] = []
    cumulative_cost = 0.0
    best = 0.0
    usage_index = 0
    for evaluated_at, fitness in evaluations:
        while (
            usage_index < len(usage_events)
            and usage_events[usage_index][0] <= evaluated_at
        ):
            cumulative_cost += usage_events[usage_index][1]
            usage_index += 1
        best = max(best, fitness)
        costs.append(max(cumulative_cost, 1e-6))
        best_values.append(best)

    total_cost = sum(cost for _, cost in usage_events)
    if costs:
        costs.append(max(total_cost, costs[-1]))
        best_values.append(best)
    return costs, best_values, total_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--label-decimals", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    data = json.loads((run_dir / "viz" / "data.json").read_text())
    usage = _load_jsonl(run_dir / "codex_usage.jsonl")
    nodes = list(data.get("nodes") or [])
    costs, best_values, total_cost = _fitness_curve(nodes, usage)
    if not costs:
        raise SystemExit(f"No plottable programs found in {run_dir}")

    best = max(best_values)
    model_names = sorted(
        {str(record.get("model")) for record in usage if record.get("model")}
    )
    model_label = "+".join(model_names) or "unknown-model"
    output = args.output or run_dir / "fitness_per_dollar.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.2, 7.0), dpi=100)
    ax.step(
        costs,
        best_values,
        where="post",
        color="#2ca02c",
        linewidth=2.8,
        label=(
            f"codex/{model_label}  "
            f"(${total_cost:.2f}, {len(nodes)} programs, best {best:.4f})"
        ),
    )
    labelled_fitness: float | None = None
    jump_points: list[tuple[float, float]] = []
    for cost, fitness in zip(costs, best_values, strict=True):
        rounded = round(fitness, args.label_decimals)
        if labelled_fitness is None:
            labelled_fitness = rounded
            continue
        if rounded > labelled_fitness:
            jump_points.append((cost, fitness))
            labelled_fitness = rounded
    ax.scatter(
        [cost for cost, _ in jump_points],
        [fitness for _, fitness in jump_points],
        color="#2ca02c",
        s=26,
        zorder=3,
    )
    for idx, (cost, fitness) in enumerate(jump_points):
        is_last = idx == len(jump_points) - 1
        close_next = (
            idx + 1 < len(jump_points)
            and jump_points[idx + 1][0] / cost < 1.5
            and jump_points[idx + 1][1] - fitness < 0.001
        )
        ax.annotate(
            f"{fitness:.{args.label_decimals}f}",
            xy=(cost, fitness),
            xytext=(-38 if close_next else 7, -18 if is_last else 7),
            textcoords="offset points",
            fontsize=10,
            color="#1f6f24",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
    ax.set_xscale("log")
    ax.set_xlabel("cumulative LLM spend, USD (log; full run cost)", fontsize=14)
    ax.set_ylabel("best-so-far fitness", fontsize=14)
    ax.set_title(
        args.title
        or f"heilbron archive-curator: fitness per dollar — codex/{model_label}",
        fontsize=18,
        pad=10,
    )
    ax.grid(True, which="major", alpha=0.28, linewidth=1.1)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="lower right", fontsize=13, framealpha=0.9)
    ax.set_ylim(bottom=min(-0.001, min(best_values) - 0.001))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
