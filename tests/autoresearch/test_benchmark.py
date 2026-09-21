from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from autoresearch.benchmark import (
    _apply_checkpoint_contract,
    _apply_convergence_probe_contract,
    _apply_runtime_overrides,
    _apply_schedule_reference,
    _apply_screen_contract,
    _build_command,
    _completed_evaluation_after_nonzero_exit,
    _configure_cuda_linker_path,
    _dataset_capacity,
    _evaluation_provenance,
    _heldout_loss_metrics,
    _nonfinite_training_gradient,
    _nonfinite_training_loss,
    _parameter_count,
    _steady_state_tokens_per_second,
    _summarize_gradient_samples,
    _training_overrides,
)
from autoresearch.mds_provenance import MDS_PROVENANCE_POLICY, mds_source_tree_sha256


def test_runtime_overrides_are_applied_and_recorded(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "global_train_batch_size: 16\n"
        "device_train_microbatch_size: 2\n"
        "device_eval_batch_size: 1\n"
        "train_loader:\n"
        "  num_workers: 1\n"
        "algorithms:\n"
        "  gradient_clipping:\n"
        "    diagnostics:\n"
        "      log_interval: 1\n"
        "callbacks:\n"
        "  optimizer_monitor:\n"
        "    log_optimizer_metrics: true\n",
        encoding="utf-8",
    )

    contract = _apply_runtime_overrides(
        config,
        train_microbatch_size=16,
        eval_batch_size=8,
        loader_workers=8,
        gradient_log_interval=20,
        disable_optimizer_metrics=True,
    )

    assert contract == {
        "global_train_batch_size": 16,
        "device_train_microbatch_size": 16,
        "gradient_accumulation_steps": 1,
        "device_eval_batch_size": 8,
        "train_loader_num_workers": 8,
        "gradient_diagnostics_log_interval": 20,
        "optimizer_monitor_enabled": False,
    }
    rendered = config.read_text(encoding="utf-8")
    assert "device_train_microbatch_size: 16" in rendered
    assert "optimizer_monitor" not in rendered


def test_runtime_override_rejects_nondivisible_microbatch(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "global_train_batch_size: 16\n"
        "device_train_microbatch_size: 2\n"
        "device_eval_batch_size: 1\n"
        "train_loader: {num_workers: 1}\n"
        "algorithms:\n"
        "  gradient_clipping:\n"
        "    diagnostics: {log_interval: 1}\n"
        "callbacks: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must divide"):
        _apply_runtime_overrides(
            config,
            train_microbatch_size=3,
            eval_batch_size=None,
            loader_workers=None,
            gradient_log_interval=None,
            disable_optimizer_metrics=False,
        )


def test_gradient_diagnostic_summary_uses_finite_nearest_rank_p95() -> None:
    summary = _summarize_gradient_samples(
        [float("nan"), *map(float, range(1, 21))],
        [0.0, 1.0, 1.0, float("nan")],
    )

    assert summary == {
        "gradient_norm_pre_clip_max": 20.0,
        "gradient_norm_pre_clip_p95": 19.0,
        "gradient_diagnostic_observations": 20.0,
        "gradient_clipping_fraction": pytest.approx(2 / 3),
    }


def test_gradient_diagnostic_summary_requires_finite_norms() -> None:
    assert _summarize_gradient_samples([float("nan")], [1.0]) == {}


def test_screen_contract_has_exactly_four_heldout_evaluations(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_duration: 30518ba\n"
        "eval_interval: 30518ba\n"
        "eval_before_train: skip\n"
        "eval_subset_num_batches: -1\n"
        "console_log_interval: 20ba\n"
        "train_loader:\n"
        "  dataset:\n"
        "    streams:\n"
        "      source: {local: /corpus, split: original}\n"
        "eval_loader:\n"
        "  name: text\n"
        "  dataset:\n"
        "    streams:\n"
        "      heldout: {local: /heldout, split: validation}\n"
        "eval_gauntlet: {weighting: EQUAL}\n"
        "icl_tasks: [{label: core}]\n",
        encoding="utf-8",
    )

    contract = _apply_screen_contract(config, batches=1024, eval_batches=32)

    assert contract == {
        "mode": "heldout_loss_screen",
        "training_batches": 1024,
        "evaluation_interval_batches": 256,
        "evaluation_subset_batches": 32,
        "expected_evaluations": 4,
        "training_root": "/heldout",
        "training_split": "train",
        "validation_split": "validation",
    }
    rendered = config.read_text(encoding="utf-8")
    assert "max_duration: 1024ba" in rendered
    assert "eval_interval: 256ba" in rendered
    assert "local: /heldout" in rendered
    assert "split: train" in rendered
    assert "icl_tasks" not in rendered
    assert "eval_gauntlet" not in rendered


def test_fixed_schedule_and_probe_decay_do_not_compress_main_lr(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_duration: 4096ba\n"
        "eval_interval: 1024ba\n"
        "console_log_interval: 20ba\n"
        "scheduler:\n"
        "  name: stacked\n"
        "  schedule_rows:\n"
        "    - [0.25, 1.0, linear]\n"
        "    - [0.90, 1.0, linear]\n"
        "    - [1.00, 0.10, cosine]\n",
        encoding="utf-8",
    )

    schedule = _apply_schedule_reference(config, reference_batches=4096)
    probe = _apply_convergence_probe_contract(config, fork_batch=256, probe_batches=96)

    assert schedule["absolute_schedule_rows"] == [
        [1024, 1.0, "linear"],
        [3686, 1.0, "linear"],
        [4096, 0.1, "cosine"],
    ]
    assert probe["fork_lr_multiplier"] == pytest.approx(0.25)
    assert probe["schedule_rows"] == [
        [256, 0.25, "linear"],
        [352, 0.0, "cosine"],
    ]
    rendered = config.read_text(encoding="utf-8")
    assert "max_duration: 352ba" in rendered
    assert "eval_interval: 352ba" in rendered


def test_probe_checkpoint_contract_is_read_only(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "load_path: null\n"
        "load_weights_only: true\n"
        "autoresume: true\n"
        "save_folder: old\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "main.pt"

    contract = _apply_checkpoint_contract(
        config,
        load_path=checkpoint,
        save_folder=None,
        save_interval_batches=None,
    )

    assert contract["full_state_resume"] is True
    assert contract["writes_checkpoint"] is False
    rendered = config.read_text(encoding="utf-8")
    assert "load_weights_only: false" in rendered
    assert "autoresume: false" in rendered
    assert "save_folder: null" in rendered


def test_heldout_loss_metrics_use_four_point_auc_and_two_point_tail(
    tmp_path: Path,
) -> None:
    log = tmp_path / "train.stderr.log"
    log.write_text(
        "Eval metrics/eval/LanguageCrossEntropy: 6.0\n"
        "Eval metrics/eval/LanguageCrossEntropy: 5.0\n"
        "Eval metrics/eval/LanguageCrossEntropy: 4.0\n"
        "Eval metrics/eval/LanguageCrossEntropy: 3.0\n",
        encoding="utf-8",
    )

    assert _heldout_loss_metrics(log) == {
        "heldout_loss_final": 3.0,
        "heldout_loss_tail_mean": 3.5,
        "heldout_loss_auc": 4.5,
        "heldout_evaluations": 4.0,
    }


def test_evaluation_provenance_hashes_tasks_and_gauntlet(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"query":"a"}\n', encoding="utf-8")
    second.write_text('{"query":"b"}\n', encoding="utf-8")
    config = tmp_path / "effective.yaml"
    config.write_text(
        "eval_gauntlet:\n"
        "  weighting: EQUAL\n"
        "icl_tasks:\n"
        "  - label: first\n"
        f"    dataset_uri: {first}\n"
        "    num_fewshot: 5\n"
        "    icl_task_type: multiple_choice\n"
        "    continuation_delimiter: ' '\n"
        "    gauntlet_tags: [core]\n"
        "  - label: second\n"
        f"    dataset_uri: {second}\n"
        "    num_fewshot: 0\n"
        "    icl_task_type: language_modeling\n",
        encoding="utf-8",
    )

    provenance = _evaluation_provenance(config)

    assert provenance["gauntlet_weighting"] == "EQUAL"
    assert (
        provenance["tasks"][0]["dataset_sha256"]
        == sha256(first.read_bytes()).hexdigest()
    )
    assert provenance["tasks"][1]["dataset_bytes"] == second.stat().st_size
    assert provenance["tasks"][0]["fewshot_random_seed"] == 1234
    assert (
        provenance["contract_sha256"]
        == sha256(
            json.dumps(
                {
                    "gauntlet_weighting": "EQUAL",
                    "tasks": provenance["tasks"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def test_evaluation_provenance_rejects_missing_dataset(tmp_path: Path) -> None:
    config = tmp_path / "effective.yaml"
    config.write_text(
        "icl_tasks:\n"
        "  - label: missing\n"
        f"    dataset_uri: {tmp_path / 'missing.jsonl'}\n",
        encoding="utf-8",
    )

    try:
        _evaluation_provenance(config)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing evaluation fixture was accepted")


def test_dataset_capacity_rejects_recycled_smoke_corpus(tmp_path: Path) -> None:
    split = tmp_path / "mds" / "dclm-baseline-1.0"
    split.mkdir(parents=True)
    (split / "index.json").write_text(
        json.dumps({"shards": [{"samples": 37_422}]}),
        encoding="utf-8",
    )
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_seq_len: 2048\n"
        "global_train_batch_size: 16\n"
        "max_duration: 30518ba\n"
        "train_loader:\n"
        "  dataset:\n"
        "    streams:\n"
        "      dclm:\n"
        f"        local: {tmp_path / 'mds'}\n"
        "        split: dclm-baseline-1.0\n",
        encoding="utf-8",
    )

    capacity = _dataset_capacity(config)

    assert capacity["available_tokens"] == 76_640_256
    assert capacity["required_tokens"] == 1_000_013_824
    assert capacity["is_sufficient"] is False
    assert capacity["has_complete_provenance"] is False
    assert capacity["uses_distinct_samples"] is True


def test_dataset_capacity_detects_explicit_resampling(tmp_path: Path) -> None:
    split = tmp_path / "mds" / "dclm-baseline-1.0"
    split.mkdir(parents=True)
    (split / "index.json").write_text(
        json.dumps({"shards": [{"samples": 1_500_000}]}),
        encoding="utf-8",
    )
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_seq_len: 2048\n"
        "global_train_batch_size: 16\n"
        "max_duration: 87715ba\n"
        "train_loader:\n"
        "  dataset:\n"
        "    epoch_size: 1062M\n"
        "    streams:\n"
        "      dclm:\n"
        f"        local: {tmp_path / 'mds'}\n"
        "        split: dclm-baseline-1.0\n"
        "        proportion: 100.0\n",
        encoding="utf-8",
    )

    capacity = _dataset_capacity(config)

    assert capacity["is_sufficient"] is True
    assert capacity["uses_distinct_samples"] is False
    assert capacity["explicit_epoch_size"] == "1062M"
    assert capacity["weighted_streams"] == ["dclm"]


def test_dataset_capacity_accepts_distinct_confirmation_corpus(
    tmp_path: Path,
) -> None:
    split = tmp_path / "mds" / "dclm-baseline-1.0"
    split.mkdir(parents=True)
    (split / "index.json").write_text(
        json.dumps({"shards": [{"samples": 1_500_000}]}),
        encoding="utf-8",
    )
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_seq_len: 2048\n"
        "global_train_batch_size: 16\n"
        "max_duration: 87715ba\n"
        "train_loader:\n"
        "  dataset:\n"
        "    streams:\n"
        "      dclm:\n"
        f"        local: {tmp_path / 'mds'}\n"
        "        split: dclm-baseline-1.0\n",
        encoding="utf-8",
    )

    capacity = _dataset_capacity(config)

    assert capacity["available_tokens"] == 3_072_000_000
    assert capacity["required_tokens"] == 2_874_245_120
    assert capacity["is_sufficient"] is True
    assert capacity["has_complete_provenance"] is False


def test_dataset_capacity_validates_corpus_manifest(tmp_path: Path) -> None:
    mds = tmp_path / "mds"
    split = mds / "dclm-baseline-1.0"
    split.mkdir(parents=True)
    shard_path = split / "shard.00000.mds.zstd"
    shard_path.write_bytes(b"pinned mds bytes")
    index_path = split / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 2,
                "shards": [
                    {
                        "format": "mds",
                        "compression": "zstd",
                        "column_names": ["dtype", "tokens"],
                        "column_encodings": ["int8", "bytes"],
                        "samples": 1_500_000,
                        "zip_data": {
                            "basename": shard_path.name,
                            "bytes": shard_path.stat().st_size,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mds_digest = mds_source_tree_sha256(split)
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    tokenizer_digest = sha256()
    tokenizer_digest.update(b"tokenizer.json\0")
    tokenizer_digest.update(
        sha256((tokenizer / "tokenizer.json").read_bytes()).hexdigest().encode()
    )
    tokenizer_digest.update(b"\n")
    (mds / "corpus-manifest.json").write_text(
        json.dumps(
            {
                "samples": 1_500_000,
                "tokens": 3_072_000_000,
                "sequence_length": 2048,
                "mds_index_sha256": sha256(index_path.read_bytes()).hexdigest(),
                "mds_tree_sha256": mds_digest,
                "mds_provenance_policy": MDS_PROVENANCE_POLICY,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "tokenizer": str(tokenizer),
                "tokenizer_sha256": tokenizer_digest.hexdigest(),
                "tokenizer_boundary": {
                    "policy": "no_implicit_special_tokens_plus_one_explicit_eos",
                    "implicit_empty_ids": [],
                    "explicit_eos_ids": [0],
                    "eos_token_id": 0,
                },
                "eos_text": "<|endoftext|>",
                "tokens_dtype": "uint16",
                "compression": "zstd",
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "effective.yaml"
    config.write_text(
        "max_seq_len: 2048\n"
        "global_train_batch_size: 16\n"
        "max_duration: 87715ba\n"
        "train_loader:\n"
        "  dataset:\n"
        "    streams:\n"
        "      dclm:\n"
        f"        local: {mds}\n"
        "        split: dclm-baseline-1.0\n",
        encoding="utf-8",
    )

    capacity = _dataset_capacity(config)

    assert capacity["has_complete_provenance"] is True
    assert capacity["corpus_manifests"] == [
        {
            "path": str(mds / "corpus-manifest.json"),
            "sha256": sha256((mds / "corpus-manifest.json").read_bytes()).hexdigest(),
            "source_manifest_sha256": sha256(source_manifest.read_bytes()).hexdigest(),
            "tokenizer_sha256": tokenizer_digest.hexdigest(),
            "mds_index_sha256": sha256(index_path.read_bytes()).hexdigest(),
            "mds_tree_sha256": mds_digest,
        }
    ]

    shard_path.write_bytes(b"x" * shard_path.stat().st_size)
    with pytest.raises(ValueError, match="MDS tree hash mismatch"):
        _dataset_capacity(config)


def test_nonfinite_training_loss_reports_first_batch(tmp_path: Path) -> None:
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "[batch=3140/30518]:\n"
        " Train loss/train/total: 4.3447\n"
        "[batch=3160/30518]:\n"
        " Train loss/train/total: nan\n",
        encoding="utf-8",
    )

    assert _nonfinite_training_loss(log_path) == {
        "batch": 3160,
        "metric": "total",
        "value": "nan",
    }


def test_nonfinite_training_loss_accepts_finite_log(tmp_path: Path) -> None:
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "[batch=20/100]:\n Train loss/train/total: 6.2\n",
        encoding="utf-8",
    )

    assert _nonfinite_training_loss(log_path) is None


def test_nonfinite_training_gradient_extracts_batch_and_parameters(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "ERROR [autoresearch] nonfinite_gradient batch=3151 "
        "parameters=model.layers.0.q_proj.weight,model.embed_tokens.weight\n",
        encoding="utf-8",
    )

    assert _nonfinite_training_gradient(log_path) == {
        "batch": 3151,
        "parameters": [
            "model.layers.0.q_proj.weight",
            "model.embed_tokens.weight",
        ],
    }


def test_completed_evaluation_accepts_teardown_only_failure(tmp_path: Path) -> None:
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "[Eval batch=10/10] Eval on first/0-shot data:\n"
        " Eval metrics/first/0-shot/InContextLearningLMAccuracy: 0.25\n"
        "[Eval batch=20/20] Eval on second/10-shot data:\n"
        " Eval metrics/second/10-shot/InContextLearningMultipleChoiceAccuracy: 0.5\n"
        "[batch=30518/30518]:\n"
        " Train metrics_gauntlet/core: 0.375\n"
        " Train time/batch: 30518\n"
        "08/24/2026 04:30:22 | INFO |  Done.\n"
        "08/24/2026 04:30:22 | DEBUG |  Engine closed.\n"
        "munmap_chunk(): invalid pointer\n",
        encoding="utf-8",
    )

    evidence = _completed_evaluation_after_nonzero_exit(
        log_path,
        expected_batches=30_518,
        evaluation_provenance={"tasks": [{"label": "first"}, {"label": "second"}]},
    )

    assert evidence == {
        "expected_batches": 30_518,
        "completed_evaluation_labels": ["first", "second"],
        "console_core": 0.375,
        "trainer_done": True,
        "engine_closed": True,
    }


@pytest.mark.parametrize(
    "missing_line",
    [
        "[batch=30518/30518]:",
        "Train time/batch: 30518",
        "Eval metrics/second/10-shot/InContextLearningMultipleChoiceAccuracy: 0.5",
        "Train metrics_gauntlet/core: 0.375",
        "| INFO |  Done.",
        "| DEBUG |  Engine closed.",
    ],
)
def test_completed_evaluation_rejects_missing_proof(
    tmp_path: Path, missing_line: str
) -> None:
    lines = [
        "[batch=30518/30518]:",
        "Train time/batch: 30518",
        "Eval metrics/first/0-shot/InContextLearningLMAccuracy: 0.25",
        "Eval metrics/second/10-shot/InContextLearningMultipleChoiceAccuracy: 0.5",
        "Train metrics_gauntlet/core: 0.375",
        "08/24/2026 04:30:22 | INFO |  Done.",
        "08/24/2026 04:30:22 | DEBUG |  Engine closed.",
    ]
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "\n".join(line for line in lines if missing_line not in line) + "\n",
        encoding="utf-8",
    )

    assert (
        _completed_evaluation_after_nonzero_exit(
            log_path,
            expected_batches=30_518,
            evaluation_provenance={"tasks": [{"label": "first"}, {"label": "second"}]},
        )
        is None
    )


def test_diagnostic_overrides_retain_tensorboard_logging() -> None:
    overrides = _training_overrides(smoke_batches=0, diagnostic_batches=3300)

    assert "debug.overrides.max_duration=3300ba" in overrides
    assert "debug.startup_speedups.disable_all_evaluators=true" in overrides
    assert "debug.overrides.save_folder=null" in overrides
    assert "debug.overrides.loggers={}" not in overrides


def test_smoke_overrides_disable_persistent_logging() -> None:
    overrides = _training_overrides(smoke_batches=12, diagnostic_batches=0)

    assert "debug.overrides.loggers={}" in overrides


def test_parameter_count_marker_uses_last_rank_zero_value(tmp_path: Path) -> None:
    log_path = tmp_path / "train.stdout.log"
    log_path.write_text(
        "noise\n[autoresearch] n_params=139000001\n[autoresearch] n_params=139000002\n",
        encoding="utf-8",
    )

    assert _parameter_count(log_path) == 139_000_002


def test_speed_monitor_marker_uses_last_warmed_value(tmp_path: Path) -> None:
    log_path = tmp_path / "train.stderr.log"
    log_path.write_text(
        "Train throughput/tokens_per_sec: 33000.5\n"
        "Train throughput/tokens_per_sec: 34018.3831\n",
        encoding="utf-8",
    )

    assert _steady_state_tokens_per_second(log_path) == 34018.3831


def test_speed_monitor_falls_back_to_last_tensorboard_value(tmp_path: Path) -> None:
    from tensorboardX import SummaryWriter

    log_path = tmp_path / "train.stderr.log"
    log_path.write_text("no console throughput\n", encoding="utf-8")
    writer = SummaryWriter(str(tmp_path / "tensorboard"))
    writer.add_scalar("throughput/tokens_per_sec", 81_234.5, 20)
    writer.add_scalar("throughput/tokens_per_sec", 82_345.5, 40)
    writer.close()

    assert _steady_state_tokens_per_second(log_path, tmp_path) == pytest.approx(
        82_345.5
    )


def test_build_command_uses_torchrun_and_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMFOUNDRY_TORCHRUN", "/runtime/torchrun")

    command = _build_command(
        tmp_path,
        tmp_path / "effective.yaml",
        4,
        ("debug.overrides.max_duration=12ba",),
    )

    assert command[:4] == [
        "/runtime/torchrun",
        "--standalone",
        "--nproc_per_node",
        "4",
    ]
    assert command[-1] == "debug.overrides.max_duration=12ba"


def test_cuda_stub_path_is_added_without_losing_existing_path(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "libcuda.so").touch()
    monkeypatch.setenv("AUTORESEARCH_CUDA_STUBS", str(tmp_path))
    environment = {"LIBRARY_PATH": "/existing"}

    _configure_cuda_linker_path(environment)

    assert environment["LIBRARY_PATH"] == f"{tmp_path}:/existing"
