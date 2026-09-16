from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import yaml

from autoresearch.config_builder import build_effective_config

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "autoresearch/config/dclm-140m.yaml"
CANDIDATE = ROOT / "autoresearch/config/candidate.yaml"


def test_candidate_overlay_changes_only_decoder_plugin() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    effective = build_effective_config(BASE, CANDIDATE)

    base_overrides = base["model"].pop("config_overrides")
    effective_overrides = effective["model"].pop("config_overrides")
    assert effective == base

    plugin_keys = {
        "decoder_layer_module",
        "decoder_layer_type",
        "decoder_layer_kwargs",
    }
    for key in plugin_keys:
        base_overrides.pop(key, None)
        effective_overrides.pop(key, None)
    assert effective_overrides == base_overrides
    assert effective["parameter_count_bounds"] == [142_363_014, 145_239_034]
    assert effective["save_folder"] is None
    assert effective["algorithms"]["gradient_clipping"] == {
        "clipping_type": "norm",
        "clipping_threshold": 1.0,
        "diagnostics": {"enabled": True, "log_interval": 1},
    }
    assert effective["callbacks"]["optimizer_monitor"] == {
        "log_optimizer_metrics": True,
        "batch_log_interval": 10,
    }
    dataset = effective["train_loader"]["dataset"]
    assert "epoch_size" not in dataset
    assert (
        not {
            "proportion",
            "repeat",
            "choose",
        }
        & dataset["streams"]["dclm-baseline"].keys()
    )
    assert len(effective["icl_tasks"]) == 15
    assert {task["gauntlet_tags"][0] for task in effective["icl_tasks"]} == {"core"}


def test_candidate_overlay_rejects_training_change(tmp_path: Path) -> None:
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    candidate["model"]["config_overrides"]["hidden_size"] = 1024
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable keys: hidden_size"):
        build_effective_config(BASE, invalid)


def test_optimizer_lr_is_numeric_and_human_overridable(monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_LR", raising=False)
    assert OmegaConf.load(BASE).optimizer.lr == pytest.approx(0.75e-3)

    monkeypatch.setenv("AUTORESEARCH_LR", "1.25e-3")
    assert OmegaConf.load(BASE).optimizer.lr == pytest.approx(1.25e-3)


def test_wsd_schedule_keeps_final_learning_rate_headroom() -> None:
    scheduler = OmegaConf.to_container(OmegaConf.load(BASE).scheduler, resolve=True)
    assert scheduler == {
        "name": "stacked",
        "schedule_rows": [
            [0.25, 1.0, "linear"],
            [0.9, 1.0, "linear"],
            [1.0, 0.1, "cosine"],
        ],
    }
