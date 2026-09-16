#!/usr/bin/env python3
"""Launch mandatory-review autoresearch from a validated calibration suite."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.completed_run import file_sha256, verify_receipt

BENCHMARK_TIMEOUT_SECONDS = 14 * 60 * 60
STAGE_TIMEOUT_SECONDS = BENCHMARK_TIMEOUT_SECONDS + 10 * 60
DAG_TIMEOUT_SECONDS = BENCHMARK_TIMEOUT_SECONDS + 20 * 60
DEFAULT_AUTORESEARCH_SECONDS = 7 * 24 * 60 * 60
STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"
EXPECTED_CONFIRMATION_BATCHES = 87_715
EXPECTED_SEQUENCE_LENGTH = 2_048
EXPECTED_GLOBAL_BATCH_SIZE = 16
EXPECTED_PARAMETER_COUNT = 143_710_848
EXPECTED_CORE_TASKS = 15
OPERATIONAL_ONLY_PATHS = {
    "autoresearch/benchmark.py",
    "autoresearch/completed_run.py",
    "config/experiment/llm_foundry_autoresearch.yaml",
    "tests/autoresearch/test_benchmark.py",
    "tests/autoresearch/test_completed_run.py",
    "tools/build-completed-baseline-receipt.py",
    "tools/run-autoresearch-benchmark.py",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Directory is empty: {path}")
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(_file_sha256(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_operational_source(
    source: Path, *, calibrated_commit: str
) -> tuple[str, list[str]]:
    source = source.resolve()
    if _git_output(source, "status", "--porcelain"):
        raise RuntimeError("Operational autoresearch source worktree is dirty")
    source_commit = _git_output(source, "rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", calibrated_commit, source_commit],
        cwd=source,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            "Operational autoresearch source does not descend from calibration"
        )
    changed_output = _git_output(
        source, "diff", "--name-only", f"{calibrated_commit}..{source_commit}"
    )
    changed = [line for line in changed_output.splitlines() if line]
    disallowed = sorted(set(changed) - OPERATIONAL_ONLY_PATHS)
    if disallowed:
        raise RuntimeError(
            "Operational source changes scientific or unapproved files: "
            + ", ".join(disallowed)
        )
    return source_commit, changed


def _is_finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _slug(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _structured_feedback(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Confirmation feedback log does not exist: {path}")
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.startswith(STRUCTURED_FEEDBACK_MARKER):
            continue
        try:
            feedback = json.loads(line[len(STRUCTURED_FEEDBACK_MARKER) :].strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Confirmation structured feedback is invalid JSON"
            ) from exc
        if not isinstance(feedback, dict):
            raise RuntimeError("Confirmation structured feedback is not an object")
        return feedback
    raise RuntimeError("Confirmation structured feedback is missing")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_confirmation_feedback(
    calibration_root: Path,
    *,
    suite: dict[str, Any],
    selected_lr: float,
    confirmation: dict[str, Any],
) -> None:
    run_dir = calibration_root / f"lr-{_slug(selected_lr)}-seed-2048-confirmation"
    feedback = _structured_feedback(run_dir / "adapter.stderr.log")
    if feedback.get("status") != "complete":
        raise RuntimeError("Confirmation feedback does not report a complete run")
    if feedback.get("n_params") != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Confirmation feedback has the wrong parameter count")
    if confirmation.get("n_params") != float(EXPECTED_PARAMETER_COUNT):
        raise RuntimeError("Confirmation metrics have the wrong parameter count")

    effective_config = Path(str(feedback.get("effective_config", "")))
    if effective_config.resolve() != (run_dir / "effective-config.yaml").resolve():
        raise RuntimeError(
            "Confirmation feedback points to a different effective config"
        )
    if not effective_config.is_file():
        raise RuntimeError("Confirmation effective config is missing")
    event_file = Path(str(feedback.get("event_file", "")))
    if not event_file.is_file() or not _is_within(event_file, run_dir):
        raise RuntimeError("Confirmation CORE event file is missing or outside its run")
    core_tag = feedback.get("core_tag")
    if not isinstance(core_tag, str) or not core_tag.endswith("metrics_gauntlet/core"):
        raise RuntimeError("Confirmation feedback does not identify the raw CORE tag")

    execution = feedback.get("execution_contract")
    if not isinstance(execution, dict):
        raise RuntimeError("Confirmation feedback is missing execution_contract")
    dataset = execution.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError("Confirmation feedback is missing its dataset contract")
    required_tokens = (
        EXPECTED_CONFIRMATION_BATCHES
        * EXPECTED_GLOBAL_BATCH_SIZE
        * EXPECTED_SEQUENCE_LENGTH
    )
    expected_dataset = {
        "batches": EXPECTED_CONFIRMATION_BATCHES,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
        "required_tokens": required_tokens,
        "explicit_epoch_size": None,
        "weighted_streams": [],
        "uses_distinct_samples": True,
        "has_complete_provenance": True,
        "is_sufficient": True,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise RuntimeError(
                f"Confirmation dataset contract field {key} is invalid: "
                f"{dataset.get(key)!r} != {expected!r}"
            )
    available_tokens = dataset.get("available_tokens")
    if not isinstance(available_tokens, int) or available_tokens < required_tokens:
        raise RuntimeError(
            "Confirmation corpus does not contain enough physical tokens"
        )
    manifests = dataset.get("corpus_manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RuntimeError("Confirmation dataset provenance is ambiguous")
    manifest_record = manifests[0]
    corpus_manifest = Path(str(suite["corpus_manifest"]))
    if not isinstance(manifest_record, dict) or (
        Path(str(manifest_record.get("path", ""))).resolve()
        != corpus_manifest.resolve()
        or manifest_record.get("sha256") != suite["corpus_manifest_sha256"]
    ):
        raise RuntimeError("Confirmation did not bind the calibrated corpus manifest")

    evaluation = execution.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError("Confirmation feedback is missing its evaluation contract")
    tasks = evaluation.get("tasks")
    if (
        evaluation.get("gauntlet_weighting") != "EQUAL"
        or not isinstance(tasks, list)
        or len(tasks) != EXPECTED_CORE_TASKS
    ):
        raise RuntimeError(
            "Confirmation did not use the exact 15-task EQUAL CORE suite"
        )
    evaluation_root = Path(str(suite["evaluation_root"]))
    labels: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise RuntimeError("Confirmation evaluation task contract is invalid")
        dataset_path = Path(str(task.get("dataset_uri", "")))
        label = task.get("label")
        if (
            not isinstance(label, str)
            or label in labels
            or not dataset_path.is_file()
            or not _is_within(dataset_path, evaluation_root)
            or task.get("dataset_sha256") != _file_sha256(dataset_path)
            or task.get("dataset_bytes") != dataset_path.stat().st_size
            or task.get("fewshot_random_seed") != 1234
            or task.get("gauntlet_tags") != ["core"]
        ):
            raise RuntimeError("Confirmation evaluation task provenance is invalid")
        labels.add(label)
    evaluation_payload = {
        "gauntlet_weighting": evaluation["gauntlet_weighting"],
        "tasks": tasks,
    }
    contract_hash = sha256(
        json.dumps(evaluation_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if evaluation.get("contract_sha256") != contract_hash:
        raise RuntimeError("Confirmation evaluation contract hash is invalid")


def _remaining_run_seconds(
    deadline_epoch: int | None, *, now: float
) -> tuple[int, int]:
    """Resolve an absolute experiment deadline and its remaining wall time."""

    deadline = (
        int(deadline_epoch)
        if deadline_epoch is not None
        else int(math.ceil(now + DEFAULT_AUTORESEARCH_SECONDS))
    )
    remaining = int(math.ceil(deadline - now))
    if remaining <= 0:
        raise RuntimeError(f"Autoresearch deadline has already passed: {deadline}")
    return deadline, remaining


def _gpu_uuid(index: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_by_index = {
        int(raw_index.strip()): uuid.strip()
        for line in completed.stdout.splitlines()
        for raw_index, uuid in [line.split(",", 1)]
    }
    if index not in gpu_by_index:
        raise RuntimeError(f"Physical GPU index {index} does not exist")
    return gpu_by_index[index]


def _assert_gpu_idle(index: int, uuid: str) -> None:
    if _gpu_uuid(index) != uuid:
        raise RuntimeError(f"Physical GPU {index} UUID changed after calibration")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    occupants = [
        line
        for line in completed.stdout.splitlines()
        if line.split(",", 1)[0].strip() == uuid
    ]
    if occupants:
        raise RuntimeError(
            f"Physical GPU {index} is occupied before handoff: {occupants}"
        )


def _lease_token(repo: Path, gpu_index: int) -> str:
    token_file = repo / "runs" / f"gpu{gpu_index}-lease.token"
    token = token_file.read_text(encoding="utf-8").strip()
    if len(token) != 64 or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise RuntimeError(f"GPU lease token is invalid: {token_file}")
    return token


def _validate_summary(calibration_root: Path) -> dict[str, Any]:
    summary_path = calibration_root / "summary.json"
    suite_path = calibration_root / "calibration-suite.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if summary.get("calibration_suite") != suite:
        raise RuntimeError(
            "Calibration summary does not embed the exact suite contract"
        )
    confirmation = summary.get("confirmation")
    if not isinstance(confirmation, dict) or not (
        _is_finite_number(confirmation.get("is_valid"))
        and float(confirmation["is_valid"]) > 0
    ):
        raise RuntimeError("Calibration confirmation is not valid")
    confirmation_core = confirmation.get("llmfoundry_core_equal_raw")
    if not _is_finite_number(confirmation_core):
        raise RuntimeError("Calibration confirmation CORE score is not finite")
    selected_lr = summary.get("selected_lr")
    if not (_is_finite_number(selected_lr) and float(selected_lr) > 0):
        raise RuntimeError("Calibration selected_lr is not finite and positive")
    confirmation_contract = summary.get("confirmation_contract")
    if not isinstance(confirmation_contract, dict):
        raise RuntimeError("Calibration summary is missing confirmation_contract")
    expected_contract = {
        "commit": suite["commit"],
        "lr": selected_lr,
        "seed": 2048,
        "batches": 87_715,
        "warmup_batches": 21_929,
        "diagnostic": False,
        "physical_gpu_index": suite["physical_gpu_index"],
        "gpu_uuid": suite["gpu_uuid"],
    }
    for key, expected in expected_contract.items():
        if confirmation_contract.get(key) != expected:
            raise RuntimeError(
                f"Confirmation contract field {key} differs from suite: "
                f"{confirmation_contract.get(key)!r} != {expected!r}"
            )
    corpus_manifest = Path(str(suite["corpus_manifest"]))
    if Path(str(confirmation_contract.get("mds_path", ""))).resolve() != (
        corpus_manifest.parent.resolve()
    ):
        raise RuntimeError("Confirmation used a different MDS root from the suite")
    if _file_sha256(corpus_manifest) != suite["corpus_manifest_sha256"]:
        raise RuntimeError("Calibration corpus manifest changed after confirmation")
    evaluation_root = Path(str(suite["evaluation_root"]))
    if _tree_sha256(evaluation_root) != suite["evaluation_sha256"]:
        raise RuntimeError("Calibration evaluation data changed after confirmation")
    source = calibration_root / "source"
    if _git_output(source, "rev-parse", "HEAD") != suite["commit"]:
        raise RuntimeError("Calibration source worktree commit changed")
    if _git_output(source, "status", "--porcelain"):
        raise RuntimeError("Calibration source worktree is dirty")
    _validate_confirmation_feedback(
        calibration_root,
        suite=suite,
        selected_lr=float(selected_lr),
        confirmation=confirmation,
    )
    return summary


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=repo / "runs" / "baseline-calibration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "runs" / "calibrated-autoresearch-week1",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Clean operational-only source descended from the calibrated commit.",
    )
    parser.add_argument(
        "--baseline-receipt",
        type=Path,
        help="Hash-bound completed baseline result to reuse only at source HEAD.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Physical GPU to use; defaults to the calibrated GPU.",
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Clean Redis database allocated to this run.",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--deadline-epoch",
        type=int,
        default=(
            int(os.environ["AUTORESEARCH_DEADLINE_EPOCH"])
            if "AUTORESEARCH_DEADLINE_EPOCH" in os.environ
            else None
        ),
        help=(
            "Absolute Unix deadline for the complete one-week allocation. "
            "The autoresearch wall clock receives only the time remaining after "
            "calibration."
        ),
    )
    args = parser.parse_args()
    args.calibration_root = args.calibration_root.resolve()
    args.output_root = args.output_root.resolve()
    summary_path = args.calibration_root / "summary.json"
    while args.wait and not summary_path.is_file():
        time.sleep(args.poll_seconds)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Calibration summary does not exist: {summary_path}")
    if not 0 <= args.redis_db <= 15:
        raise ValueError("Redis database must be between 0 and 15")
    summary = _validate_summary(args.calibration_root)
    suite = summary["calibration_suite"]
    gpu_index = (
        args.gpu_index
        if args.gpu_index is not None
        else int(suite["physical_gpu_index"])
    )
    gpu_uuid = _gpu_uuid(gpu_index)
    _assert_gpu_idle(gpu_index, gpu_uuid)
    deadline_epoch, run_seconds = _remaining_run_seconds(
        args.deadline_epoch, now=time.time()
    )
    calibrated_source = (args.calibration_root / "source").resolve()
    source = (args.source or calibrated_source).resolve()
    source_commit, operational_diff = _validate_operational_source(
        source, calibrated_commit=str(suite["commit"])
    )
    receipt_path = args.baseline_receipt.resolve() if args.baseline_receipt else None
    receipt_sha256 = None
    if receipt_path is not None:
        verify_receipt(
            receipt_path,
            calibrated_commit=str(suite["commit"]),
            operational_commit=source_commit,
        )
        receipt_sha256 = file_sha256(receipt_path)
    contract = {
        "calibration_summary_sha256": _file_sha256(summary_path),
        "calibration_suite": suite,
        "selected_lr": summary["selected_lr"],
        "calibrated_source": str(calibrated_source),
        "source": str(source),
        "source_commit": source_commit,
        "operational_diff": operational_diff,
        "baseline_receipt": str(receipt_path) if receipt_path else None,
        "baseline_receipt_sha256": receipt_sha256,
        "execution_gpu_index": gpu_index,
        "execution_gpu_uuid": gpu_uuid,
        "redis_db": args.redis_db,
        "review_mode": "mandatory",
        "screen_batches": 30_518,
        "screen_warmup_batches": 7_630,
        "deadline_epoch": deadline_epoch,
        "run_seconds_at_handoff": run_seconds,
    }
    contract_path = args.output_root / "handoff-contract.json"
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if (
            not contract_path.is_file()
            or json.loads(contract_path.read_text(encoding="utf-8")) != contract
        ):
            raise RuntimeError(
                f"Refusing ambiguous existing handoff output: {args.output_root}"
            )
        raise RuntimeError(
            f"Calibrated autoresearch was already launched: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(contract_path, contract)
    data = repo.parent / "data"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "AUTORESEARCH_GPU_LEASE_TOKEN": _lease_token(repo, gpu_index),
        "AUTORESEARCH_DATA_ROOT": str(data),
        "DCLM_MODEL_PATH": str(data / "model-template-gpt-neox-140m"),
        "DCLM_MDS_PATH": str(Path(str(suite["corpus_manifest"])).parent),
        "DCLM_CORE_PATH": str(suite["evaluation_root"]),
        "DCLM_EVAL_CACHE_PATH": str(data / "eval-cache"),
        "AUTORESEARCH_OUTPUT_DIR": str(args.output_root),
        "AUTORESEARCH_RUN_ROOT": str(args.output_root / "training"),
        "AUTORESEARCH_MEMORY_DIR": str(args.output_root / "memory"),
        "AUTORESEARCH_RUN_NAME": (f"calibrated-{str(suite['commit'])[:12]}-week1"),
        "AUTORESEARCH_REVIEW_MODE": "mandatory",
        "AUTORESEARCH_LR": str(summary["selected_lr"]),
        "AUTORESEARCH_SEED": "2048",
        "AUTORESEARCH_MAX_DURATION": "30518ba",
        "AUTORESEARCH_WARMUP_DURATION": "7630ba",
        "AUTORESEARCH_EVAL_INTERVAL": "30518ba",
        "AUTORESEARCH_DEADLINE_EPOCH": str(deadline_epoch),
        "AUTORESEARCH_RUN_SECONDS": str(run_seconds),
        "AUTORESEARCH_EXPERIMENT_TIMEOUT_SECONDS": str(BENCHMARK_TIMEOUT_SECONDS),
        "AUTORESEARCH_BENCHMARK_TIMEOUT": str(BENCHMARK_TIMEOUT_SECONDS),
        "AUTORESEARCH_STAGE_TIMEOUT": str(STAGE_TIMEOUT_SECONDS),
        "AUTORESEARCH_DAG_TIMEOUT": str(DAG_TIMEOUT_SECONDS),
    }
    if receipt_path is not None:
        environment.update(
            {
                "AUTORESEARCH_REUSED_BASELINE_COMMIT": source_commit,
                "AUTORESEARCH_CALIBRATED_COMMIT": str(suite["commit"]),
                "AUTORESEARCH_BASELINE_RECEIPT": str(receipt_path),
                "AUTORESEARCH_BASELINE_RECEIPT_SHA256": str(receipt_sha256),
            }
        )
    command = [
        str(source / "tools" / "run-autoresearch"),
        f"redis.db={args.redis_db}",
    ]
    print(
        f"Launching calibrated mandatory-review autoresearch from {source} "
        f"at LR={summary['selected_lr']}",
        flush=True,
    )
    os.chdir(source)
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
