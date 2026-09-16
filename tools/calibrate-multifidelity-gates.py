#!/usr/bin/env python3
"""Calibrate shrinking gates from out-of-sample eventual-winner predictions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.calibration import (
    LabeledProbability,
    calibrate_kill_gate,
    calibrate_promote_gate,
)
from autoresearch.multifidelity.models import MultiFidelityPlan


def _records(
    path: Path, *, expected_split: str
) -> tuple[dict[int, list[LabeledProbability]], set[str]]:
    grouped: dict[int, list[LabeledProbability]] = {}
    group_ids: set[str] = set()
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                budget = int(value["budget_batches"])
                probability = float(value["probability"])
                lower = float(value.get("probability_lower", probability))
                upper = float(value.get("probability_upper", probability))
                winner = value["eventual_winner"]
                split = str(value["split"])
                run_id = str(value["run_id"])
                group_id = str(value["group_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid record on line {line_number}") from exc
            if not isinstance(winner, bool):
                raise ValueError(
                    f"eventual_winner must be boolean on line {line_number}"
                )
            if not 0.0 <= lower <= probability <= upper <= 1.0:
                raise ValueError(f"probability out of range on line {line_number}")
            if split != expected_split:
                raise ValueError(
                    f"line {line_number} belongs to split {split!r}, "
                    f"expected {expected_split!r}"
                )
            key = (run_id, budget)
            if key in seen:
                raise ValueError(f"duplicate run/budget on line {line_number}: {key}")
            seen.add(key)
            group_ids.add(group_id)
            grouped.setdefault(budget, []).append(
                LabeledProbability(probability, winner, lower=lower, upper=upper)
            )
    return grouped, group_ids


def _conservative_monotone_gates(
    kill_caps: list[float], promote_floors: list[float]
) -> tuple[list[float], list[float]]:
    """Make gates monotone only by widening, never weakening constraints."""

    kills = [0.0] * len(kill_caps)
    promotes = [1.0] * len(promote_floors)
    suffix_kill = 1.0
    suffix_promote = 0.0
    for index in range(len(kill_caps) - 1, -1, -1):
        suffix_kill = min(suffix_kill, kill_caps[index])
        suffix_promote = max(suffix_promote, promote_floors[index])
        kills[index] = suffix_kill
        promotes[index] = suffix_promote
    return kills, promotes


def _widen_to_shrinking_widths(
    kills: list[float], promotes: list[float]
) -> tuple[list[float], list[float]]:
    """Enforce shrinking widths by making earlier decisions more conservative."""

    widened_kills = list(kills)
    widened_promotes = list(promotes)
    for index in range(len(kills) - 2, -1, -1):
        required = widened_promotes[index + 1] - widened_kills[index + 1]
        current = widened_promotes[index] - widened_kills[index]
        missing = max(0.0, required - current)
        lower_kill = min(widened_kills[index], missing)
        widened_kills[index] -= lower_kill
        missing -= lower_kill
        widened_promotes[index] = min(1.0, widened_promotes[index] + missing)
    return widened_kills, widened_promotes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", default="policy_selection")
    parser.add_argument("--promotion-precision-floor", type=float, default=0.80)
    args = parser.parse_args()

    plan = MultiFidelityPlan.from_yaml(args.plan)
    grouped, group_ids = _records(
        args.records,
        expected_split=args.expected_split,
    )
    calibration_rungs = plan.rungs[:-1]
    missing = [
        rung.budget_batches
        for rung in calibration_rungs
        if rung.budget_batches not in grouped
    ]
    if missing:
        parser.error(f"missing calibration records for budgets: {missing}")

    kill_results = [
        calibrate_kill_gate(
            grouped[rung.budget_batches],
            recall_floor=plan.winner_recall_floor,
            confidence=plan.confidence_level,
        )
        for rung in calibration_rungs
    ]
    promote_results = [
        calibrate_promote_gate(
            grouped[rung.budget_batches],
            precision_floor=args.promotion_precision_floor,
            confidence=plan.confidence_level,
        )
        for rung in calibration_rungs
    ]
    kills, promotes = _conservative_monotone_gates(
        [result.threshold for result in kill_results],
        [result.threshold for result in promote_results],
    )
    final_gate = plan.rungs[-1]
    kills = [min(value, final_gate.kill_gate) for value in kills]
    promotes = [max(value, final_gate.promote_gate) for value in promotes]
    all_kills, all_promotes = _widen_to_shrinking_widths(
        [*kills, final_gate.kill_gate],
        [*promotes, final_gate.promote_gate],
    )
    kills = all_kills[:-1]
    promotes = all_promotes[:-1]

    value = plan.model_dump(mode="json")
    value["calibrated"] = True
    value["observe_only"] = True
    for index, rung in enumerate(value["rungs"][:-1]):
        rung["kill_gate"] = kills[index]
        rung["promote_gate"] = promotes[index]
        if rung["kill_gate"] >= rung["promote_gate"]:
            parser.error(f"calibrated gates overlap at budget {rung['budget_batches']}")
    # Revalidate all ordering and shrinking-width invariants before publishing.
    calibrated = MultiFidelityPlan.model_validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(calibrated.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    evidence_path = args.output.with_suffix(".calibration.json")
    evidence_path.write_text(
        json.dumps(
            {
                "version": 1,
                "records": str(args.records.resolve()),
                "records_sha256": sha256(args.records.read_bytes()).hexdigest(),
                "calibrated_plan": str(args.output.resolve()),
                "calibrated_plan_sha256": sha256(args.output.read_bytes()).hexdigest(),
                "expected_split": args.expected_split,
                "groups": len(group_ids),
                "confidence_method": plan.confidence_method,
                "confidence_level": plan.confidence_level,
                "winner_recall_floor": plan.winner_recall_floor,
                "kill_gates": [result.__dict__ for result in kill_results],
                "promote_gates": [result.__dict__ for result in promote_results],
                "applied_gates": [
                    {
                        "budget_batches": rung.budget_batches,
                        "kill_gate": rung.kill_gate,
                        "promote_gate": rung.promote_gate,
                    }
                    for rung in calibrated.rungs
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calibration_evidence": str(evidence_path),
                "observe_only": True,
                "budgets": [rung.budget_batches for rung in calibrated.rungs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
