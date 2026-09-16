"""Event-driven 1024 -> 4096 promotion for one fixed research idea."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from tensorboardX import SummaryWriter

from autoresearch.ideas.store import _write_json

STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"
LOSS_KEY = "heldout_loss_auc"


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


def _loss(metrics: dict[str, Any]) -> float | None:
    try:
        value = float(metrics[LOSS_KEY])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fitness_from_loss(loss: float) -> float:
    return 1.0 / (1.0 + loss)


def _scientific_metrics(
    screen_metrics: dict[str, Any], *, fitness: float, is_valid: bool
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in screen_metrics.items():
        if key == "llmfoundry_core_equal_raw":
            # Screen mode does not evaluate CORE; do not turn its placeholder
            # zero into an apparent scientific observation.
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            result[str(key)] = float(value)
    result["fitness"] = float(fitness)
    result["is_valid"] = 1.0 if is_valid else 0.0
    return result


def initialize_promotion_state(
    path: Path,
    *,
    idea_contract: dict[str, Any],
    confirmed_commit: str,
    screen_metrics: dict[str, float],
    confirmation_metrics: dict[str, float],
    screen_batches: int,
    confirmation_batches: int,
    screen_eval_batches: int,
    tensorboard_dir: Path,
) -> None:
    """Create the durable promotion state and the two TensorBoard charts."""

    screen_loss = _loss(screen_metrics)
    confirmation_loss = _loss(confirmation_metrics)
    if screen_loss is None or confirmation_loss is None:
        raise ValueError("Confirmed parent requires finite short and long heldout loss")
    baseline_metrics = _scientific_metrics(
        screen_metrics,
        fitness=_fitness_from_loss(confirmation_loss),
        is_valid=True,
    )
    baseline_feedback = {
        "status": "confirmed_incumbent",
        "promotion": "baseline",
        "screen_batches": screen_batches,
        "confirmation_batches": confirmation_batches,
    }
    state = {
        "version": 1,
        "idea": idea_contract,
        "screen_batches": screen_batches,
        "confirmation_batches": confirmation_batches,
        "screen_eval_batches": screen_eval_batches,
        "tensorboard_dir": str(tensorboard_dir.resolve()),
        "attempt": 0,
        "confirmed": {
            "commit": confirmed_commit,
            "screen_metrics": screen_metrics,
            "confirmation_metrics": confirmation_metrics,
        },
        "promotion_bar_loss": screen_loss,
        "last_failure": None,
        "results": {
            confirmed_commit: {
                "metrics": baseline_metrics,
                "feedback": baseline_feedback,
                "screen_metrics": screen_metrics,
                "confirmation_metrics": confirmation_metrics,
            }
        },
    }
    _write_json(path, state)

    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(tensorboard_dir)) as writer:
        writer.add_custom_scalars(
            {
                "Evolution": {
                    "Validation loss": [
                        "Multiline",
                        [
                            "Evolution/validation_loss/short_candidate",
                            "Evolution/validation_loss/promotion_bar",
                            "Evolution/validation_loss/long_confirmation",
                            "Evolution/validation_loss/confirmed_frontier",
                        ],
                    ],
                    "CORE": [
                        "Multiline",
                        [
                            "Evolution/core/candidate",
                            "Evolution/core/confirmed_frontier",
                        ],
                    ],
                }
            }
        )
        writer.add_text(
            "Evolution/fixed_idea_contract",
            "```json\n" + json.dumps(idea_contract, indent=2, sort_keys=True) + "\n```",
            global_step=0,
        )
        writer.add_scalar("Evolution/validation_loss/promotion_bar", screen_loss, 0)
        writer.add_scalar(
            "Evolution/validation_loss/confirmed_frontier", confirmation_loss, 0
        )


def _failure_note(
    *, commit: str, phase: str, feedback: dict[str, Any]
) -> dict[str, Any]:
    note: dict[str, Any] = {"commit": commit, "phase": phase}
    for key in ("status", "failure_stage", "error_type", "error"):
        value = feedback.get(key)
        if value not in (None, ""):
            note[key] = value
    tail = feedback.get("stderr_tail")
    if isinstance(tail, str) and tail.strip():
        note["stderr_tail"] = tail[-2000:]
    return note


def _write_evolution_scalars(
    state: dict[str, Any],
    *,
    step: int,
    screen_loss: float | None,
    long_loss: float | None,
) -> None:
    confirmed = state["confirmed"]
    confirmed_loss = _loss(confirmed["confirmation_metrics"])
    tensorboard_dir = Path(state["tensorboard_dir"])
    with SummaryWriter(str(tensorboard_dir)) as writer:
        if screen_loss is not None:
            writer.add_scalar(
                "Evolution/validation_loss/short_candidate", screen_loss, step
            )
        writer.add_scalar(
            "Evolution/validation_loss/promotion_bar",
            float(state["promotion_bar_loss"]),
            step,
        )
        if long_loss is not None:
            writer.add_scalar(
                "Evolution/validation_loss/long_confirmation", long_loss, step
            )
        if confirmed_loss is not None:
            writer.add_scalar(
                "Evolution/validation_loss/confirmed_frontier", confirmed_loss, step
            )


def _run_horizon(
    *,
    label: str,
    batches: int,
    eval_batches: int,
    run_dir: Path,
    benchmark_args: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon_dir = run_dir / label
    command = [
        sys.executable,
        "-m",
        "autoresearch.benchmark",
        *benchmark_args,
        "--screen-batches",
        str(batches),
        "--screen-eval-batches",
        str(eval_batches),
        "--run-dir",
        str(horizon_dir),
    ]
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{label}.stdout.log").write_text(process.stdout, encoding="utf-8")
    (run_dir / f"{label}.stderr.log").write_text(process.stderr, encoding="utf-8")
    metrics = _last_json_object(process.stdout)
    feedback = _structured_feedback(process.stderr)
    if process.returncode != 0 and not feedback:
        feedback = {
            "status": "benchmark_wrapper_failed",
            "failure_stage": label,
            "error_type": "nonzero_exit",
            "error": f"benchmark exited with {process.returncode}",
            "stderr_tail": process.stderr[-2000:],
        }
    return metrics, feedback


def _emit(metrics: dict[str, float], feedback: dict[str, Any]) -> None:
    print(json.dumps(metrics, sort_keys=True))
    print(
        STRUCTURED_FEEDBACK_MARKER + " " + json.dumps(feedback, sort_keys=True),
        file=sys.stderr,
    )


def _cached_result(state: dict[str, Any], commit: str) -> bool:
    cached = state.get("results", {}).get(commit)
    if not isinstance(cached, dict):
        return False
    metrics = cached.get("metrics")
    feedback = cached.get("feedback")
    if not isinstance(metrics, dict) or not isinstance(feedback, dict):
        return False
    _emit(metrics, feedback)
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-state", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args, benchmark_args = _parser().parse_known_args()
    state_path = args.promotion_state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    commit = args.commit

    if _cached_result(state, commit):
        return 0

    confirmed = state["confirmed"]
    state["attempt"] = int(state.get("attempt", 0)) + 1
    step = state["attempt"]
    screen_metrics, screen_feedback = _run_horizon(
        label=f"screen-{state['screen_batches']}",
        batches=int(state["screen_batches"]),
        eval_batches=int(state["screen_eval_batches"]),
        run_dir=args.run_dir,
        benchmark_args=benchmark_args,
    )
    screen_loss = _loss(screen_metrics) if _valid(screen_metrics) else None
    long_loss: float | None = None
    confirmation_metrics: dict[str, Any] | None = None

    if screen_loss is None:
        state["last_failure"] = _failure_note(
            commit=commit, phase="short_screen", feedback=screen_feedback
        )
        metrics = _scientific_metrics(screen_metrics, fitness=0.0, is_valid=False)
        feedback = {
            **screen_feedback,
            "promotion": "screen_failed_technically",
            "summary": (
                "The implementation did not complete the short experiment; no "
                "scientific conclusion was reached."
            ),
        }
    else:
        confirmed_screen_loss = _loss(confirmed["screen_metrics"])
        assert confirmed_screen_loss is not None
        promotion_bar = float(state["promotion_bar_loss"])
        eligible = screen_loss < confirmed_screen_loss and screen_loss < promotion_bar
        if not eligible:
            state["last_failure"] = None
            metrics = _scientific_metrics(screen_metrics, fitness=0.0, is_valid=True)
            feedback = {
                **screen_feedback,
                "promotion": "not_eligible",
                "screen_loss": screen_loss,
                "confirmed_screen_loss": confirmed_screen_loss,
                "promotion_bar_loss": promotion_bar,
            }
        else:
            confirmation_metrics, confirmation_feedback = _run_horizon(
                label=f"confirmation-{state['confirmation_batches']}",
                batches=int(state["confirmation_batches"]),
                eval_batches=int(state["screen_eval_batches"]),
                run_dir=args.run_dir,
                benchmark_args=benchmark_args,
            )
            long_loss = (
                _loss(confirmation_metrics) if _valid(confirmation_metrics) else None
            )
            if long_loss is None:
                state["last_failure"] = _failure_note(
                    commit=commit,
                    phase="long_confirmation",
                    feedback=confirmation_feedback,
                )
                metrics = _scientific_metrics(
                    screen_metrics, fitness=0.0, is_valid=False
                )
                feedback = {
                    **confirmation_feedback,
                    "promotion": "confirmation_failed_technically",
                    "screen_loss": screen_loss,
                    "summary": (
                        "The implementation improved the short screen but did not "
                        "complete confirmation; this is not evidence against the idea."
                    ),
                }
            else:
                state["last_failure"] = None
                state["promotion_bar_loss"] = screen_loss
                confirmed_long_loss = _loss(confirmed["confirmation_metrics"])
                assert confirmed_long_loss is not None
                if long_loss < confirmed_long_loss:
                    state["confirmed"] = {
                        "commit": commit,
                        "screen_metrics": screen_metrics,
                        "confirmation_metrics": confirmation_metrics,
                    }
                    metrics = _scientific_metrics(
                        screen_metrics,
                        fitness=_fitness_from_loss(long_loss),
                        is_valid=True,
                    )
                    feedback = {
                        **confirmation_feedback,
                        "promotion": "confirmed_win",
                        "screen_loss": screen_loss,
                        "confirmation_loss": long_loss,
                        "previous_confirmed_loss": confirmed_long_loss,
                    }
                else:
                    metrics = _scientific_metrics(
                        screen_metrics, fitness=0.0, is_valid=True
                    )
                    feedback = {
                        **confirmation_feedback,
                        "promotion": "short_progress_not_confirmed",
                        "screen_loss": screen_loss,
                        "confirmation_loss": long_loss,
                        "confirmed_loss": confirmed_long_loss,
                    }

    state["results"][commit] = {
        "metrics": metrics,
        "feedback": feedback,
        "screen_metrics": screen_metrics,
        "confirmation_metrics": confirmation_metrics,
    }
    _write_json(state_path, state)
    _write_evolution_scalars(
        state, step=step, screen_loss=screen_loss, long_loss=long_loss
    )
    _emit(metrics, feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
