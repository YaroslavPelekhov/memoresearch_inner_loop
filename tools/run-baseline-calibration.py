#!/usr/bin/env python3
"""Run the preregistered baseline LR calibration sequentially on one GPU."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PHASE_A_LRS = (1.625e-3, 1.75e-3)
FULL_LRS = (1.25e-3, 1.5e-3)
# Failure-triggered extensions.  These groups are considered only when every
# setting in the preceding group is invalid, so the original preregistered
# comparison remains intact and lower-LR work is not started speculatively.
FALLBACK_FULL_LR_GROUPS = ((1.0e-3, 0.75e-3), (0.5e-3,))
DIAGNOSTIC_BATCHES = 3_300
DIAGNOSTIC_WARMUP = 825
SCREEN_BATCHES = 30_518
SCREEN_WARMUP = 7_630
CONFIRM_BATCHES = 87_715
CONFIRM_WARMUP = 21_929
SIGNIFICANT_CORE_DELTA = 0.001


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
    gpu_by_index: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        raw_index, uuid = (part.strip() for part in line.split(",", 1))
        gpu_by_index[int(raw_index)] = uuid
    if index not in gpu_by_index:
        raise RuntimeError(f"Physical GPU index {index} does not exist")
    return gpu_by_index[index]


def _assert_gpu_idle(index: int) -> str:
    uuid = _gpu_uuid(index)
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
    occupants: list[str] = []
    for line in completed.stdout.splitlines():
        process_uuid, raw_pid = (part.strip() for part in line.split(",", 1))
        if process_uuid != uuid:
            continue
        pid = int(raw_pid)
        command_path = Path(f"/proc/{pid}/cmdline")
        try:
            command = (
                command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except FileNotFoundError:
            continue
        occupants.append(f"pid={pid} command={command.strip()}")
    if occupants:
        raise RuntimeError(
            f"Physical GPU {index} ({uuid}) is not idle: " + "; ".join(occupants)
        )
    return uuid


def _slug(lr: float) -> str:
    return f"{lr:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _last_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "is_valid" in value:
            return value
    return None


def _contract_matches(path: Path, contract: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return existing == contract


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _assert_fixed_source(repo: Path, output_root: Path) -> str:
    suite = json.loads(
        (output_root / "calibration-suite.json").read_text(encoding="utf-8")
    )
    expected_commit = str(suite["commit"])
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Calibration source moved to {actual_commit}; expected {expected_commit}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Calibration source worktree became dirty")
    return actual_commit


def _run(
    *,
    repo: Path,
    output_root: Path,
    python: str,
    lr: float,
    seed: int,
    batches: int,
    warmup: int,
    diagnostic: bool,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon = (
        "diagnostic-3300"
        if diagnostic
        else ("confirmation" if batches == CONFIRM_BATCHES else "screen-1b")
    )
    run_dir = output_root / f"lr-{_slug(lr)}-seed-{seed}-{horizon}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_commit = _assert_fixed_source(repo, output_root)
    contract = {
        "commit": source_commit,
        "lr": lr,
        "seed": seed,
        "batches": batches,
        "warmup_batches": warmup,
        "diagnostic": diagnostic,
        "mds_path": os.environ["DCLM_MDS_PATH"],
    }
    gpu_index = int(os.environ["AUTORESEARCH_PHYSICAL_GPU_INDEX"])
    contract["physical_gpu_index"] = gpu_index
    contract["gpu_uuid"] = _gpu_uuid(gpu_index)
    contract_path = run_dir / "calibration-contract.json"
    stdout_path = run_dir / "adapter.stdout.log"
    stderr_path = run_dir / "adapter.stderr.log"
    previous = _last_json(stdout_path)
    if previous is not None and _contract_matches(contract_path, contract):
        return previous, contract
    _assert_gpu_idle(gpu_index)
    if any(run_dir.iterdir()) and not contract_path.exists():
        raise RuntimeError(f"Refusing ambiguous existing run directory: {run_dir}")
    _write_json_atomic(contract_path, contract)

    environment = {
        **os.environ,
        "AUTORESEARCH_LR": str(lr),
        "AUTORESEARCH_SEED": str(seed),
        "AUTORESEARCH_MAX_DURATION": f"{batches}ba",
        "AUTORESEARCH_WARMUP_DURATION": f"{warmup}ba",
        "AUTORESEARCH_EVAL_INTERVAL": f"{batches}ba",
        "RUN_NAME": run_dir.name,
    }
    command = [
        python,
        "-m",
        "autoresearch.benchmark",
        "--gpus",
        "1",
        "--timeout",
        str(timeout),
        "--run-dir",
        str(run_dir),
    ]
    if diagnostic:
        command.extend(["--diagnostic-batches", str(batches)])
    print(f"[calibration] starting {run_dir.name}", flush=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    result = _last_json(stdout_path)
    if result is None:
        raise RuntimeError(
            f"Calibration adapter produced no metrics for {run_dir}; "
            f"exit={completed.returncode}"
        )
    print(f"[calibration] completed {run_dir.name}: {result}", flush=True)
    return result, contract


def _is_valid(result: dict[str, Any]) -> bool:
    return float(result.get("is_valid", 0.0)) > 0.0


def _screen_lrs(valid_diagnostic_lrs: list[float]) -> tuple[float, ...]:
    """Retain every finite upper diagnostic setting for full CORE screening."""

    return tuple(dict.fromkeys((*FULL_LRS, *valid_diagnostic_lrs)))


def _fitness(result: dict[str, Any]) -> float:
    return float(result["llmfoundry_core_equal_raw"])


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Evaluation data directory does not exist: {path}")
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Evaluation data directory is empty: {path}")
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(file_path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_source(repo: Path, output_root: Path, data: Path, gpu_index: int) -> Path:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus_manifest = data / "dclm-mds-3b-v1" / "corpus-manifest.json"
    corpus_manifest_hash = sha256(corpus_manifest.read_bytes()).hexdigest()
    evaluation_root = data / "dclm-core"
    evaluation_hash = _tree_sha256(evaluation_root)
    gpu_uuid = _gpu_uuid(gpu_index)
    expected_suite = {
        "commit": commit,
        "corpus_manifest": str(corpus_manifest),
        "corpus_manifest_sha256": corpus_manifest_hash,
        "evaluation_root": str(evaluation_root),
        "evaluation_sha256": evaluation_hash,
        "physical_gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
    }
    suite_path = output_root / "calibration-suite.json"
    source = output_root / "source"
    if suite_path.is_file():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if not source.exists() and suite != expected_suite:
            raise RuntimeError("Calibration contract changed before source creation")
        expected_commit = str(suite["commit"])
        for key, expected in expected_suite.items():
            if key == "commit":
                continue
            if suite.get(key) != expected:
                raise RuntimeError(
                    f"Calibration suite field {key} changed: "
                    f"{suite.get(key)!r} != {expected!r}"
                )
    else:
        expected_commit = commit
        _write_json_atomic(suite_path, expected_suite)
    if not source.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(source), expected_commit],
            cwd=repo,
            check=True,
        )
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Calibration worktree is {actual_commit}, expected {expected_commit}"
        )
    return source


def main() -> int:
    main_repo = Path(__file__).resolve().parents[1]
    data = main_repo.parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=main_repo / "runs" / "baseline-calibration",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("AUTORESEARCH_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=int(os.environ.get("AUTORESEARCH_PHYSICAL_GPU_INDEX", "0")),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    repo = _prepare_source(main_repo, args.output_root, data, args.gpu_index)

    defaults = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "AUTORESEARCH_PHYSICAL_GPU_INDEX": str(args.gpu_index),
        "DCLM_MODEL_PATH": str(data / "model-template-gpt-neox-140m"),
        "DCLM_MDS_PATH": str(data / "dclm-mds-3b-v1"),
        "DCLM_CORE_PATH": str(data / "dclm-core"),
        "DCLM_EVAL_CACHE_PATH": str(data / "eval-cache"),
    }
    for key, value in defaults.items():
        os.environ[key] = value

    results: dict[str, dict[str, Any]] = {}
    valid_diagnostic_lrs: list[float] = []
    for lr in PHASE_A_LRS:
        result, _ = _run(
            repo=repo,
            output_root=args.output_root,
            python=args.python,
            lr=lr,
            seed=2048,
            batches=DIAGNOSTIC_BATCHES,
            warmup=DIAGNOSTIC_WARMUP,
            diagnostic=True,
            timeout=4 * 60 * 60,
        )
        results[f"diagnostic:{lr}:2048"] = result
        if _is_valid(result):
            valid_diagnostic_lrs.append(lr)

    seed_2048: list[tuple[float, dict[str, Any]]] = []
    screen_groups = (
        _screen_lrs(valid_diagnostic_lrs),
        *FALLBACK_FULL_LR_GROUPS,
    )
    for group_index, screen_lrs in enumerate(screen_groups):
        if group_index:
            print(
                "[calibration] every higher representative screen was invalid; "
                f"starting failure-triggered LR group {screen_lrs}",
                flush=True,
            )
        for lr in screen_lrs:
            result, _ = _run(
                repo=repo,
                output_root=args.output_root,
                python=args.python,
                lr=lr,
                seed=2048,
                batches=SCREEN_BATCHES,
                warmup=SCREEN_WARMUP,
                diagnostic=False,
                timeout=14 * 60 * 60,
            )
            results[f"screen:{lr}:2048"] = result
            if _is_valid(result):
                seed_2048.append((lr, result))
        if seed_2048:
            break
    if not seed_2048:
        raise RuntimeError(
            "Every representative and failure-triggered 1B baseline run was invalid"
        )

    seed_2048.sort(key=lambda item: _fitness(item[1]), reverse=True)
    leaders = seed_2048[:2]
    repeated: dict[float, list[float]] = {
        lr: [_fitness(result)] for lr, result in leaders
    }
    for lr, _ in leaders:
        result, _ = _run(
            repo=repo,
            output_root=args.output_root,
            python=args.python,
            lr=lr,
            seed=2049,
            batches=SCREEN_BATCHES,
            warmup=SCREEN_WARMUP,
            diagnostic=False,
            timeout=14 * 60 * 60,
        )
        results[f"screen:{lr}:2049"] = result
        if _is_valid(result):
            repeated[lr].append(_fitness(result))
        else:
            repeated.pop(lr, None)

    if not repeated:
        raise RuntimeError("Every leading setting failed its seed repeat")
    if len(repeated) == 1:
        selected_lr = next(iter(repeated))
        decision = "only one finite leading setting"
    else:
        ranked = sorted(
            repeated.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
            reverse=True,
        )
        best_lr, best_scores = ranked[0]
        runner_up_lr, runner_up_scores = ranked[1]
        paired_deltas = [
            left - right for left, right in zip(best_scores, runner_up_scores)
        ]
        if paired_deltas and all(
            delta >= SIGNIFICANT_CORE_DELTA for delta in paired_deltas
        ):
            selected_lr = best_lr
            decision = "CORE advantage exceeded the preregistered threshold per seed"
        else:
            selected_lr = min(best_lr, runner_up_lr)
            decision = "CORE ordering was within seed noise; selected lower stable LR"

    confirmation, confirmation_contract = _run(
        repo=repo,
        output_root=args.output_root,
        python=args.python,
        lr=selected_lr,
        seed=2048,
        batches=CONFIRM_BATCHES,
        warmup=CONFIRM_WARMUP,
        diagnostic=False,
        timeout=40 * 60 * 60,
    )
    results[f"confirmation:{selected_lr}:2048"] = confirmation
    summary = {
        "selected_lr": selected_lr,
        "decision": decision,
        "representative_core_by_lr": repeated,
        "confirmation": confirmation,
        "confirmation_contract": confirmation_contract,
        "calibration_suite": json.loads(
            (args.output_root / "calibration-suite.json").read_text(encoding="utf-8")
        ),
        "results": results,
    }
    _write_json_atomic(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if _is_valid(confirmation) else 1


if __name__ == "__main__":
    raise SystemExit(main())
