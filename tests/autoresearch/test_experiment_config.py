from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_llm_foundry_experiment_serializes_gpu_work() -> None:
    config = yaml.safe_load(
        (ROOT / "config/experiment/llm_foundry_autoresearch.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["max_mutations_per_generation"] == 1
    assert config["dag_concurrency"] == 1
    assert config["max_concurrent_dags"] == 1


def test_fixed_idea_evolution_keeps_one_idea_for_all_generations() -> None:
    config = yaml.safe_load(
        (ROOT / "config/experiment/llm_foundry_idea_evolution.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["pre_step_hook"] is None
    assert config["max_mutations_per_generation"] == 1
    assert config["repo_harness"]["allowed_changed_files"] == [
        "autoresearch/model/gdn.py"
    ]
    assert "AUTORESEARCH_ACTIVE_IDEA_PATH" in config["repo_harness"]["active_idea_path"]

    command = config["repo_harness"]["mutation_backend"]["command"]
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--approve-for-me" not in command
