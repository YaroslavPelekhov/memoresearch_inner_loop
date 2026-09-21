#!/usr/bin/env python3
"""Run the frozen b3a210 candidate/parent replication as six paired waves."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any

CANDIDATE_COMMIT = "b3a210ded839250e22eb148718fc9d401ec90052"
PARENT_COMMIT = "0369584bf978a6850f96d29d3dc07fe00f32aaa9"
SEEDS = (31001, 31002, 31003, 31004, 31005, 31006)
MODEL_PATH = Path("autoresearch/model/gdn.py")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return process.stdout


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def gpu_assignment(pair_index: int) -> dict[str, int]:
    """Balance code variants across physical GPUs without breaking pairing."""

    if pair_index < 1:
        raise ValueError("pair_index must be positive")
    if pair_index % 2:
        return {"candidate": 0, "parent": 1}
    return {"candidate": 1, "parent": 0}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _commit_exists(repo: Path, commit: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True,
        capture_output=True,
    )


def _prepare_worktree(
    *, repo: Path, worktree: Path, harness_commit: str, source_commit: str
) -> str:
    marker = worktree / ".fork-to-zero-source.json"
    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                harness_commit,
            ],
            check=True,
        )
    observed_head = str(_run_git(worktree, "rev-parse", "HEAD")).strip()
    if observed_head != harness_commit:
        raise RuntimeError(
            f"worktree {worktree} is at {observed_head}, expected {harness_commit}"
        )
    source = _run_git(repo, "show", f"{source_commit}:{MODEL_PATH}", text=False)
    assert isinstance(source, bytes)
    model_path = worktree / MODEL_PATH
    model_path.write_bytes(source)
    source_hash = _sha256(source)
    expected_marker = {
        "harness_commit": harness_commit,
        "model_path": str(MODEL_PATH),
        "model_sha256": source_hash,
        "source_commit": source_commit,
    }
    if marker.exists():
        observed = json.loads(marker.read_text(encoding="utf-8"))
        if observed != expected_marker:
            raise RuntimeError(f"worktree provenance mismatch: {worktree}")
    else:
        _write_json(marker, expected_marker)
    return source_hash


def _command(
    *, python: Path, worktree: Path, plan: Path, source_commit: str, run_dir: Path
) -> list[str]:
    return [
        str(python),
        "-u",
        str(worktree / "tools/run-multifidelity-autoresearch-benchmark.py"),
        "--multifidelity-plan",
        str(plan),
        "--commit",
        source_commit,
        "--run-dir",
        str(run_dir),
        "--screen-eval-batches",
        "32",
        "--gpus",
        "1",
        "--train-microbatch-size",
        "16",
        "--eval-batch-size",
        "32",
        "--loader-workers",
        "8",
        "--gradient-log-interval",
        "20",
        "--disable-optimizer-metrics",
    ]


def _lane_environment(base: dict[str, str], *, gpu: int, seed: int) -> dict[str, str]:
    env = dict(base)
    workspace = Path(env["AUTORESEARCH_WORKSPACE_ROOT"])
    lane_root = workspace / "data-lanes" / f"gpu{gpu}"
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "AUTORESEARCH_SEED": str(seed),
            "AUTORESEARCH_STREAMING_PREFIX_BASE": str(700000 + gpu * 100000),
            "DCLM_MDS_PATH": str(lane_root / Path(env["DCLM_MDS_PATH"]).name),
            "DCLM_SCREEN_MDS_PATH": str(
                lane_root / Path(env["DCLM_SCREEN_MDS_PATH"]).name
            ),
            "DCLM_EVAL_CACHE_PATH": str(lane_root / "eval-cache"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    Path(env["DCLM_EVAL_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    return env


def _launch_job(
    *,
    python: Path,
    worktree: Path,
    plan: Path,
    source_commit: str,
    job_root: Path,
    gpu: int,
    seed: int,
) -> tuple[subprocess.Popen[str], Any, Any, list[str]]:
    run_dir = job_root / "training"
    job_root.mkdir(parents=True, exist_ok=True)
    command = _command(
        python=python,
        worktree=worktree,
        plan=plan,
        source_commit=source_commit,
        run_dir=run_dir,
    )
    stdout_handle = (job_root / "sequence.stdout.log").open("a", encoding="utf-8")
    stderr_handle = (job_root / "sequence.stderr.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=worktree,
        env=_lane_environment(os.environ, gpu=gpu, seed=seed),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return process, stdout_handle, stderr_handle, command


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=root
        / "config/multifidelity/dclm-140m-fork-to-zero-b3a-replication-v1.yaml",
    )
    parser.add_argument("--candidate-commit", default=CANDIDATE_COMMIT)
    parser.add_argument("--parent-commit", default=PARENT_COMMIT)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    campaign_root = args.campaign_root.resolve()
    plan = args.plan.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if seeds != SEEDS:
        raise ValueError(f"protocol requires seeds {SEEDS}, received {seeds}")
    for commit in (args.candidate_commit, args.parent_commit):
        _commit_exists(repo, commit)
    observed_parent = str(
        _run_git(repo, "rev-parse", f"{args.candidate_commit}^")
    ).strip()
    if observed_parent != args.parent_commit:
        raise RuntimeError(
            f"candidate parent is {observed_parent}, expected {args.parent_commit}"
        )
    changed = str(
        _run_git(
            repo,
            "diff",
            "--name-only",
            args.parent_commit,
            args.candidate_commit,
        )
    ).splitlines()
    if changed != [str(MODEL_PATH)]:
        raise RuntimeError(f"candidate must differ from parent only in {MODEL_PATH}: {changed}")
    if not plan.is_file():
        raise FileNotFoundError(plan)
    harness_commit = str(_run_git(repo, "rev-parse", "HEAD")).strip()
    python = Path(os.environ["AUTORESEARCH_PYTHON"]).resolve()
    if not python.is_file():
        raise FileNotFoundError(python)

    manifest: dict[str, Any] = {
        "campaign_root": str(campaign_root),
        "candidate_commit": args.candidate_commit,
        "created_at": _utc_now(),
        "harness_commit": harness_commit,
        "model_path": str(MODEL_PATH),
        "parent_commit": args.parent_commit,
        "plan": str(plan),
        "plan_sha256": _sha256(plan.read_bytes()),
        "protocol": "fork-to-zero-b3a-replication-v1",
        "seeds": list(seeds),
        "status": "validated" if args.dry_run else "preparing",
        "waves": [],
    }
    for pair_index, seed in enumerate(seeds, start=1):
        manifest["waves"].append(
            {
                "assignment": gpu_assignment(pair_index),
                "pair": pair_index,
                "seed": seed,
                "status": "pending",
            }
        )
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_keys = (
            "candidate_commit",
            "harness_commit",
            "parent_commit",
            "plan_sha256",
            "protocol",
            "seeds",
        )
        if any(previous.get(key) != manifest.get(key) for key in frozen_keys):
            raise RuntimeError("existing campaign manifest has different frozen inputs")

    worktrees = campaign_root / "worktrees"
    variant_specs = {
        "candidate": args.candidate_commit,
        "parent": args.parent_commit,
    }
    manifest["model_sha256"] = {}
    for variant, source_commit in variant_specs.items():
        manifest["model_sha256"][variant] = _prepare_worktree(
            repo=repo,
            worktree=worktrees / variant,
            harness_commit=harness_commit,
            source_commit=source_commit,
        )
    manifest["status"] = "running"
    manifest["started_at"] = _utc_now()
    _write_json(manifest_path, manifest)

    any_nonzero = False
    for wave in manifest["waves"]:
        pair_index = int(wave["pair"])
        seed = int(wave["seed"])
        wave["status"] = "running"
        wave["started_at"] = _utc_now()
        jobs: dict[str, tuple[subprocess.Popen[str], Any, Any, list[str]]] = {}
        for variant, source_commit in variant_specs.items():
            gpu = int(wave["assignment"][variant])
            job_root = (
                campaign_root / "pairs" / f"seed-{seed}" / variant
            )
            jobs[variant] = _launch_job(
                python=python,
                worktree=worktrees / variant,
                plan=plan,
                source_commit=source_commit,
                job_root=job_root,
                gpu=gpu,
                seed=seed,
            )
            wave.setdefault("jobs", {})[variant] = {
                "command": jobs[variant][3],
                "gpu": gpu,
                "run_dir": str(job_root / "training"),
                "source_commit": source_commit,
                "status": "running",
            }
        _write_json(manifest_path, manifest)
        for variant, (process, stdout_handle, stderr_handle, _) in jobs.items():
            returncode = process.wait()
            stdout_handle.close()
            stderr_handle.close()
            job = wave["jobs"][variant]
            job["returncode"] = returncode
            job["finished_at"] = _utc_now()
            job["status"] = "exited" if returncode == 0 else "wrapper_failed"
            any_nonzero = any_nonzero or returncode != 0
            _write_json(manifest_path, manifest)
        wave["finished_at"] = _utc_now()
        wave["status"] = (
            "finished"
            if all(job["returncode"] == 0 for job in wave["jobs"].values())
            else "finished_with_wrapper_failure"
        )
        _write_json(manifest_path, manifest)
        print(
            f"finished pair {pair_index}/{len(seeds)} seed={seed} "
            f"candidate_gpu={wave['assignment']['candidate']} "
            f"parent_gpu={wave['assignment']['parent']}",
            flush=True,
        )

    manifest["finished_at"] = _utc_now()
    manifest["status"] = "finished_with_wrapper_failure" if any_nonzero else "finished"
    _write_json(manifest_path, manifest)
    return 1 if any_nonzero else 0


if __name__ == "__main__":
    raise SystemExit(main())
