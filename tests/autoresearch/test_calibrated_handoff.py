from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = runpy.run_path(str(ROOT / "tools/launch-calibrated-autoresearch.py"))


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_handoff_rejects_invalid_confirmation(tmp_path: Path) -> None:
    (tmp_path / "calibration-suite.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps({"calibration_suite": {}, "confirmation": {"is_valid": 0.0}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="confirmation is not valid"):
        HANDOFF["_validate_summary"](tmp_path)


def test_handoff_accepts_exact_validated_suite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / "tracked.txt").write_text("fixed source\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixed source",
    )
    commit = _git(source, "rev-parse", "HEAD")
    mds_root = tmp_path / "mds"
    mds_root.mkdir()
    corpus_manifest = mds_root / "corpus-manifest.json"
    corpus_manifest.write_text('{"fixed":true}\n', encoding="utf-8")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    tasks = []
    for index in range(15):
        dataset = evaluation / f"task-{index:02d}.jsonl"
        dataset.write_text(f'{{"query":"q{index}"}}\n', encoding="utf-8")
        tasks.append(
            {
                "label": f"task-{index:02d}",
                "dataset_uri": str(dataset),
                "dataset_sha256": HANDOFF["_file_sha256"](dataset),
                "dataset_bytes": dataset.stat().st_size,
                "num_fewshot": [0],
                "fewshot_random_seed": 1234,
                "icl_task_type": "multiple_choice",
                "continuation_delimiter": None,
                "gauntlet_tags": ["core"],
            }
        )
    suite = {
        "commit": commit,
        "corpus_manifest": str(corpus_manifest),
        "corpus_manifest_sha256": HANDOFF["_file_sha256"](corpus_manifest),
        "evaluation_root": str(evaluation),
        "evaluation_sha256": HANDOFF["_tree_sha256"](evaluation),
        "physical_gpu_index": 1,
        "gpu_uuid": "GPU-test",
    }
    summary = {
        "calibration_suite": suite,
        "selected_lr": 0.0015,
        "confirmation": {
            "is_valid": 1.0,
            "llmfoundry_core_equal_raw": 0.31,
            "n_params": 143_710_848.0,
        },
        "confirmation_contract": {
            "commit": commit,
            "lr": 0.0015,
            "seed": 2048,
            "batches": 87_715,
            "warmup_batches": 21_929,
            "diagnostic": False,
            "physical_gpu_index": 1,
            "gpu_uuid": "GPU-test",
            "mds_path": str(mds_root),
        },
    }
    (tmp_path / "calibration-suite.json").write_text(
        json.dumps(suite), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    confirmation_run = tmp_path / "lr-0p0015-seed-2048-confirmation"
    confirmation_run.mkdir()
    effective_config = confirmation_run / "effective-config.yaml"
    effective_config.write_text("fixed: true\n", encoding="utf-8")
    event_file = confirmation_run / "events.out.tfevents.test"
    event_file.write_text("event\n", encoding="utf-8")
    evaluation_payload = {"gauntlet_weighting": "EQUAL", "tasks": tasks}
    evaluation_contract = {
        **evaluation_payload,
        "contract_sha256": sha256(
            json.dumps(
                evaluation_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    feedback = {
        "status": "complete",
        "n_params": 143_710_848,
        "effective_config": str(effective_config),
        "event_file": str(event_file),
        "core_tag": "metrics/metrics_gauntlet/core",
        "execution_contract": {
            "dataset": {
                "available_tokens": 4_333_377_536,
                "batches": 87_715,
                "sequence_length": 2_048,
                "global_batch_size": 16,
                "required_tokens": 87_715 * 16 * 2_048,
                "explicit_epoch_size": None,
                "weighted_streams": [],
                "uses_distinct_samples": True,
                "has_complete_provenance": True,
                "is_sufficient": True,
                "corpus_manifests": [
                    {
                        "path": str(corpus_manifest),
                        "sha256": HANDOFF["_file_sha256"](corpus_manifest),
                    }
                ],
            },
            "evaluation": evaluation_contract,
        },
    }
    (confirmation_run / "adapter.stderr.log").write_text(
        HANDOFF["STRUCTURED_FEEDBACK_MARKER"]
        + " "
        + json.dumps(feedback, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    assert HANDOFF["_validate_summary"](tmp_path) == summary

    feedback["execution_contract"]["dataset"]["corpus_manifests"][0]["sha256"] = (
        "0" * 64
    )
    (confirmation_run / "adapter.stderr.log").write_text(
        HANDOFF["STRUCTURED_FEEDBACK_MARKER"]
        + " "
        + json.dumps(feedback, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="calibrated corpus manifest"):
        HANDOFF["_validate_summary"](tmp_path)


def test_handoff_timeout_hierarchy_cannot_preempt_benchmark() -> None:
    benchmark = 14 * 60 * 60
    assert HANDOFF["BENCHMARK_TIMEOUT_SECONDS"] == benchmark
    assert HANDOFF["STAGE_TIMEOUT_SECONDS"] > benchmark
    assert HANDOFF["DAG_TIMEOUT_SECONDS"] > HANDOFF["STAGE_TIMEOUT_SECONDS"]


def test_handoff_uses_only_time_remaining_before_absolute_deadline() -> None:
    assert HANDOFF["_remaining_run_seconds"](1_000, now=900.25) == (1_000, 100)

    with pytest.raises(RuntimeError, match="deadline has already passed"):
        HANDOFF["_remaining_run_seconds"](1_000, now=1_000.0)


def test_handoff_loads_the_selected_gpu_lease_token(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    gpu0_token = "a" * 64
    gpu1_token = "b" * 64
    (runs / "gpu0-lease.token").write_text(gpu0_token + "\n", encoding="utf-8")
    (runs / "gpu1-lease.token").write_text(gpu1_token + "\n", encoding="utf-8")

    assert HANDOFF["_lease_token"](tmp_path, 0) == gpu0_token
    assert HANDOFF["_lease_token"](tmp_path, 1) == gpu1_token


def test_operational_source_allows_only_harness_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / "decoder.py").write_text("calibrated\n", encoding="utf-8")
    _git(source, "add", "decoder.py")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "calibrated",
    )
    calibrated_commit = _git(source, "rev-parse", "HEAD")
    benchmark = source / "autoresearch" / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text("operational fix\n", encoding="utf-8")
    _git(source, "add", "autoresearch/benchmark.py")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "operational",
    )

    source_commit, changed = HANDOFF["_validate_operational_source"](
        source, calibrated_commit=calibrated_commit
    )

    assert source_commit == _git(source, "rev-parse", "HEAD")
    assert changed == ["autoresearch/benchmark.py"]


def test_operational_source_rejects_scientific_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / "decoder.py").write_text("calibrated\n", encoding="utf-8")
    _git(source, "add", "decoder.py")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "calibrated",
    )
    calibrated_commit = _git(source, "rev-parse", "HEAD")
    (source / "decoder.py").write_text("changed architecture\n", encoding="utf-8")
    _git(source, "add", "decoder.py")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "scientific change",
    )

    with pytest.raises(RuntimeError, match="scientific or unapproved"):
        HANDOFF["_validate_operational_source"](
            source, calibrated_commit=calibrated_commit
        )
