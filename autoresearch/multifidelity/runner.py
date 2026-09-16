"""Run one candidate through checkpoint rungs and shadow LR-to-zero probes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from autoresearch.multifidelity.features import build_probe_features, fitness_from_loss
from autoresearch.multifidelity.models import (
    Decision,
    MultiFidelityPlan,
    RungObservation,
)
from autoresearch.multifidelity.policy import ShrinkingGatePolicy
from autoresearch.multifidelity.predictor import ProbabilityModel
from autoresearch.multifidelity.state import load_observations, save_observations

STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"


def _resume_promoted_run_at_final(
    observations: list[RungObservation], *, final_budget: int
) -> bool:
    return bool(
        observations
        and observations[-1].budget_batches != final_budget
        and observations[-1].decision.executed == Decision.PROMOTE
    )


def _last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _structured_feedback(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if STRUCTURED_FEEDBACK_MARKER not in line:
            continue
        raw = line.split(STRUCTURED_FEEDBACK_MARKER, 1)[1].strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _valid(metrics: dict[str, Any]) -> bool:
    try:
        return float(metrics.get("is_valid", 0.0)) >= 0.5
    except (TypeError, ValueError):
        return False


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }


def _fitness(metrics: dict[str, Any]) -> float | None:
    try:
        loss = float(metrics["heldout_loss_final"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(loss) or loss < 0.0:
        return None
    return fitness_from_loss(loss)


def _checkpoint_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_checkpoint(folder: Path) -> Path | None:
    exact = sorted(folder.rglob("latest-rank0.pt"))
    if exact:
        return exact[-1].resolve()
    candidates = sorted(
        folder.rglob("*-rank0.pt"), key=lambda path: path.stat().st_mtime
    )
    return candidates[-1].resolve() if candidates else None


def _run_benchmark(
    *,
    label: str,
    run_dir: Path,
    benchmark_args: list[str],
    extra_args: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "autoresearch.benchmark",
        *benchmark_args,
        *extra_args,
        "--run-dir",
        str(run_dir),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wrapper.stdout.log").write_text(process.stdout, encoding="utf-8")
    (run_dir / "wrapper.stderr.log").write_text(process.stderr, encoding="utf-8")
    metrics = _last_json_object(process.stdout)
    feedback = _structured_feedback(process.stderr)
    if process.returncode != 0 and not feedback:
        feedback = {
            "status": "benchmark_wrapper_failed",
            "failure_stage": label,
            "returncode": process.returncode,
            "stderr_tail": process.stderr[-2000:],
        }
    return metrics, feedback


def _emit(metrics: dict[str, float], feedback: dict[str, Any]) -> None:
    print(json.dumps(metrics, sort_keys=True))
    print(
        STRUCTURED_FEEDBACK_MARKER + " " + json.dumps(feedback, sort_keys=True),
        file=sys.stderr,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multifidelity-plan",
        type=Path,
        default=Path("config/multifidelity/dclm-140m.yaml"),
    )
    parser.add_argument("--probability-model", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--screen-eval-batches", type=int, default=32)
    return parser


def main() -> int:
    args, benchmark_args = _parser().parse_known_args()
    plan = MultiFidelityPlan.from_yaml(args.multifidelity_plan)
    plan_hash = sha256(args.multifidelity_plan.read_bytes()).hexdigest()
    probability_model = (
        ProbabilityModel.from_json(args.probability_model)
        if args.probability_model is not None
        else None
    )
    if not plan.observe_only:
        if probability_model is None or not probability_model.calibrated:
            raise ValueError("active pruning requires a calibrated probability model")

    run_dir = args.run_dir.resolve()
    state_path = run_dir / "multifidelity-trajectory.json"
    observations = load_observations(state_path)
    if any(item.commit != args.commit for item in observations):
        raise ValueError("run directory contains a trajectory for another commit")
    if any(
        item.metadata.get("plan_sha256") not in {None, plan_hash}
        for item in observations
    ):
        raise ValueError("run directory contains a trajectory for another plan")
    if (
        observations
        and observations[-1].budget_batches == plan.rungs[-1].budget_batches
        and observations[-1].decision.executed == Decision.PROMOTE
    ):
        final_metrics = observations[-1].main_metrics
        _emit(final_metrics, {"status": "cached", "observations": len(observations)})
        return 0
    if observations and observations[-1].decision.executed == Decision.KILL:
        _emit(
            {"fitness": 0.0, "is_valid": 1.0},
            {
                "status": "cached_kill",
                "budget_batches": observations[-1].budget_batches,
                "trajectory": str(state_path),
            },
        )
        return 0

    completed_budgets = {item.budget_batches for item in observations}
    previous_checkpoint = (
        Path(observations[-1].checkpoint_path)
        if observations and observations[-1].checkpoint_path
        else None
    )
    if previous_checkpoint is not None:
        expected_hash = observations[-1].checkpoint_sha256
        if (
            expected_hash is None
            or not previous_checkpoint.is_file()
            or _checkpoint_sha256(previous_checkpoint) != expected_hash
        ):
            raise RuntimeError("saved main-branch checkpoint failed resume validation")
    previous_main_fitness = (
        observations[-1].features.main_fitness if observations else None
    )
    previous_probe_fitness = next(
        (
            item.features.probe_fitness
            for item in reversed(observations)
            if item.features.probe_fitness is not None
        ),
        None,
    )
    policy = ShrinkingGatePolicy(observe_only=plan.observe_only)
    final_index = len(plan.rungs) - 1
    promoted_to_final = _resume_promoted_run_at_final(
        observations,
        final_budget=plan.rungs[-1].budget_batches,
    )

    for index, rung in enumerate(plan.rungs):
        if rung.budget_batches in completed_budgets:
            continue
        if promoted_to_final and index != final_index:
            continue
        main_dir = run_dir / "main" / f"budget-{rung.budget_batches}"
        checkpoint_dir = main_dir / "checkpoints"
        main_extra = [
            "--screen-batches",
            str(rung.budget_batches),
            "--screen-eval-batches",
            str(args.screen_eval_batches),
            "--schedule-reference-batches",
            str(plan.schedule_reference_batches),
            "--save-folder",
            str(checkpoint_dir),
            "--save-interval-batches",
            str(rung.budget_batches),
        ]
        if previous_checkpoint is not None:
            main_extra.extend(["--load-path", str(previous_checkpoint)])
        main_metrics, main_feedback = _run_benchmark(
            label=f"main-{rung.budget_batches}",
            run_dir=main_dir,
            benchmark_args=benchmark_args,
            extra_args=main_extra,
        )
        main_fitness = _fitness(main_metrics) if _valid(main_metrics) else None
        checkpoint = _find_checkpoint(checkpoint_dir)
        if main_fitness is None or checkpoint is None:
            _emit(
                {"fitness": 0.0, "is_valid": 0.0},
                {
                    **main_feedback,
                    "status": "multifidelity_main_failed",
                    "budget_batches": rung.budget_batches,
                    "checkpoint_found": checkpoint is not None,
                },
            )
            return 0
        checkpoint_hash = _checkpoint_sha256(checkpoint)

        probe_metrics: dict[str, Any] | None = None
        probe_feedback: dict[str, Any] = {}
        probe_fitness: float | None = None
        if rung.probe_batches:
            probe_dir = run_dir / "probes" / f"budget-{rung.budget_batches}"
            probe_metrics, probe_feedback = _run_benchmark(
                label=f"probe-{rung.budget_batches}",
                run_dir=probe_dir,
                benchmark_args=benchmark_args,
                extra_args=[
                    "--screen-batches",
                    str(rung.budget_batches + rung.probe_batches),
                    "--screen-eval-batches",
                    str(args.screen_eval_batches),
                    "--schedule-reference-batches",
                    str(plan.schedule_reference_batches),
                    "--load-path",
                    str(checkpoint),
                    "--probe-from-batch",
                    str(rung.budget_batches),
                    "--probe-batches",
                    str(rung.probe_batches),
                ],
            )
            if _checkpoint_sha256(checkpoint) != checkpoint_hash:
                raise RuntimeError("probe modified the main-branch checkpoint")
            if _valid(probe_metrics):
                probe_fitness = _fitness(probe_metrics)

        features = build_probe_features(
            main_fitness=main_fitness,
            previous_main_fitness=previous_main_fitness,
            probe_fitness=probe_fitness,
            previous_probe_fitness=previous_probe_fitness,
        )
        estimate = (
            probability_model.predict(rung.budget_batches, features)
            if probability_model is not None
            else None
        )
        decision = policy.decide(
            estimate,
            rung,
            final_rung=index == final_index,
        )
        clean_main_metrics = _finite_metrics(main_metrics)
        clean_main_metrics["fitness"] = main_fitness
        clean_main_metrics["is_valid"] = 1.0
        observation = RungObservation(
            commit=args.commit,
            budget_batches=rung.budget_batches,
            main_metrics=clean_main_metrics,
            probe_metrics=(
                _finite_metrics(probe_metrics) if probe_metrics is not None else None
            ),
            features=features,
            probability=estimate,
            decision=decision,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=checkpoint_hash,
            wall_time_seconds=float(main_metrics.get("wall_time_seconds", 0.0)),
            metadata={
                "plan_sha256": plan_hash,
                "main_feedback": main_feedback,
                "probe_feedback": probe_feedback,
                "probe_is_valid": probe_fitness is not None,
                "probe_wall_time_seconds": (
                    float(probe_metrics.get("wall_time_seconds", 0.0))
                    if probe_metrics is not None
                    else 0.0
                ),
            },
        )
        observations.append(observation)
        save_observations(state_path, observations)
        previous_checkpoint = checkpoint
        previous_main_fitness = main_fitness
        if probe_fitness is not None:
            previous_probe_fitness = probe_fitness

        if decision.executed == Decision.KILL:
            _emit(
                {"fitness": 0.0, "is_valid": 1.0},
                {
                    "status": "killed_by_multifidelity_policy",
                    "budget_batches": rung.budget_batches,
                    "decision": decision.model_dump(mode="json"),
                    "trajectory": str(state_path),
                },
            )
            return 0
        if decision.executed == Decision.PROMOTE and index != final_index:
            promoted_to_final = True

    final = observations[-1]
    final_metrics = dict(final.main_metrics)
    final_loss = final_metrics.get("heldout_loss_final")
    if final_loss is not None:
        final_metrics["fitness"] = fitness_from_loss(final_loss)
    final_metrics["is_valid"] = 1.0
    _emit(
        final_metrics,
        {
            "status": "multifidelity_complete",
            "observe_only": plan.observe_only,
            "trajectory": str(state_path),
            "rungs_completed": len(observations),
            "budgets_completed": [item.budget_batches for item in observations],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
