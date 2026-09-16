from __future__ import annotations

import json
from pathlib import Path
import sys

from autoresearch.ideas import promotion


def _metrics(loss: float, *, valid: bool = True) -> dict[str, float]:
    return {
        "fitness": 1.0 / (1.0 + loss) if valid else 0.0,
        "is_valid": 1.0 if valid else 0.0,
        "heldout_loss_auc": loss,
        "heldout_loss_final": loss,
        "heldout_loss_tail_mean": loss,
        "heldout_evaluations": 4.0 if valid else 0.0,
    }


def _initialize(tmp_path: Path) -> Path:
    path = tmp_path / "promotion-state.json"
    promotion.initialize_promotion_state(
        path,
        idea_contract={"id": "idea-test", "mechanism": "fixed"},
        confirmed_commit="baseline",
        screen_metrics=_metrics(5.0),
        confirmation_metrics=_metrics(4.0),
        screen_batches=1024,
        confirmation_batches=4096,
        screen_eval_batches=32,
        tensorboard_dir=tmp_path / "tb",
    )
    return path


def test_short_record_promotes_only_after_long_improvement(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = _initialize(tmp_path)
    runs = iter([(_metrics(4.8), {}), (_metrics(3.8), {})])
    monkeypatch.setattr(promotion, "_run_horizon", lambda **_kwargs: next(runs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion",
            "--promotion-state",
            str(state_path),
            "--commit",
            "winner",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    assert promotion.main() == 0

    state = json.loads(state_path.read_text())
    assert state["confirmed"]["commit"] == "winner"
    assert state["promotion_bar_loss"] == 4.8
    result = state["results"]["winner"]
    assert result["feedback"]["promotion"] == "confirmed_win"
    assert result["metrics"]["fitness"] == 1.0 / 4.8


def test_unconfirmed_long_run_raises_the_short_promotion_bar(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = _initialize(tmp_path)
    runs = iter([(_metrics(4.8), {}), (_metrics(4.2), {})])
    monkeypatch.setattr(promotion, "_run_horizon", lambda **_kwargs: next(runs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion",
            "--promotion-state",
            str(state_path),
            "--commit",
            "screen-only",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    assert promotion.main() == 0

    state = json.loads(state_path.read_text())
    assert state["confirmed"]["commit"] == "baseline"
    assert state["promotion_bar_loss"] == 4.8
    assert state["results"]["screen-only"]["metrics"]["fitness"] == 0.0
    assert (
        state["results"]["screen-only"]["feedback"]["promotion"]
        == "short_progress_not_confirmed"
    )


def test_technical_failure_is_carried_without_rejecting_the_idea(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = _initialize(tmp_path)
    monkeypatch.setattr(
        promotion,
        "_run_horizon",
        lambda **_kwargs: (
            {"fitness": 0.0, "is_valid": 0.0},
            {
                "status": "train_failed",
                "failure_stage": "training",
                "error_type": "RuntimeError",
                "stderr_tail": "model failed",
            },
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion",
            "--promotion-state",
            str(state_path),
            "--commit",
            "broken",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    assert promotion.main() == 0

    state = json.loads(state_path.read_text())
    assert state["confirmed"]["commit"] == "baseline"
    assert state["last_failure"] == {
        "commit": "broken",
        "phase": "short_screen",
        "status": "train_failed",
        "failure_stage": "training",
        "error_type": "RuntimeError",
        "stderr_tail": "model failed",
    }
    assert "scientific conclusion" in state["results"]["broken"]["feedback"]["summary"]
