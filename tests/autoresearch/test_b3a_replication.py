from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = _load(
    ROOT / "tools/run-fork-to-zero-b3a-replication.py", "b3a_replication_driver"
)
ANALYSIS = _load(
    ROOT / "analysis/fork-to-zero-b3a-replication-v1/analyze.py",
    "b3a_replication_analysis",
)


def _observation(
    budget: int, *, main: float, control: float | None, fork: float | None, loss: float
) -> dict[str, object]:
    paired = control is not None and fork is not None
    return {
        "budget_batches": budget,
        "features": {
            "main_fitness": main,
            "control_fitness": control,
            "probe_fitness": fork,
        },
        "main_metrics": {"heldout_loss_final": loss},
        "metadata": {
            "control_is_valid": paired,
            "probe_is_valid": paired,
            "control_feedback": {"status": "screen_complete"},
            "probe_feedback": {"status": "screen_complete"},
        },
    }


def test_gpu_assignment_is_balanced_and_swapped() -> None:
    assignments = [DRIVER.gpu_assignment(index) for index in range(1, 7)]

    assert assignments[0] == {"candidate": 0, "parent": 1}
    assert assignments[1] == {"candidate": 1, "parent": 0}
    assert sum(item["candidate"] == 0 for item in assignments) == 3


def test_fork_must_strictly_precede_control() -> None:
    trajectory = [
        _observation(512, main=0.15, control=0.151, fork=0.148, loss=5.0),
        _observation(768, main=0.15, control=0.147, fork=0.144, loss=5.0),
        _observation(1024, main=0.14, control=0.137, fork=0.134, loss=5.2),
        _observation(4096, main=0.09, control=None, fork=None, loss=6.0),
    ]

    result = ANALYSIS.classify_run(trajectory, [])

    assert result["collapse"] is True
    assert result["fork_earliest_checkpoint"] == 512
    assert result["control_earliest_checkpoint"] == 768
    assert result["fork_leads_control"] is True


def test_simultaneous_alarm_is_not_a_fork_advantage() -> None:
    trajectory = [
        _observation(512, main=0.15, control=0.147, fork=0.144, loss=5.0),
        _observation(4096, main=0.09, control=None, fork=None, loss=6.0),
    ]

    result = ANALYSIS.classify_run(trajectory, [])

    assert result["fork_earliest_checkpoint"] == 512
    assert result["control_earliest_checkpoint"] == 512
    assert result["fork_leads_control"] is False


def test_frozen_success_rule_requires_both_effects() -> None:
    rows = []
    for seed in ANALYSIS.SEEDS:
        rows.append(
            {
                "seed": seed,
                "variant": "candidate",
                "collapse": seed <= 31004,
                "fork_leads_control": seed <= 31004,
            }
        )
        rows.append(
            {
                "seed": seed,
                "variant": "parent",
                "collapse": False,
                "fork_leads_control": False,
            }
        )

    result = ANALYSIS.evaluate(rows)

    assert result["complete_matrix"] is True
    assert result["mutation_mechanism_replicated"] is True
    assert result["fork_early_warning_supported"] is True
    assert result["verdict"] == "full_support"
