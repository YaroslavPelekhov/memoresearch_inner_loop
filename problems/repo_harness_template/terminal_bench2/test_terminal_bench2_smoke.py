#!/usr/bin/env python3
"""Run one Terminal-Bench 2 task outside GigaEvo.

As a script this performs a live one-task run. When collected by pytest it is
skipped unless RUN_TB2_LIVE=1 is set, so regular test runs do not unexpectedly
start Docker containers or LLM calls.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark.py"
WORKSPACE_ROOT = HERE.parents[3]
DEFAULT_CANDIDATE_REPO = (
    WORKSPACE_ROOT / "meta-harness" / "reference_examples" / "terminal_bench_2"
)


def supplied_cli_options(argv: list[str]) -> set[str]:
    options: set[str] = set()
    for arg in argv:
        if arg.startswith("--"):
            options.add(arg.split("=", 1)[0])
    return options


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    supplied_options = getattr(args, "supplied_options", set())
    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--candidate-repo",
        str(Path(args.candidate_repo).expanduser().resolve()),
        "--task",
        args.task,
        "--runs",
        str(args.runs),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
    ]
    if args.agent:
        cmd.extend(["--agent", args.agent])
    else:
        cmd.extend(["--agent-import-path", args.agent_import_path])
    if args.model and (not args.openrouter or "--model" in supplied_options):
        cmd.extend(["--model", args.model])
    if args.openrouter:
        cmd.append("--openrouter")
    if args.openrouter_model:
        cmd.extend(["--openrouter-model", args.openrouter_model])
    if args.openrouter_api_base:
        cmd.extend(["--openrouter-api-base", args.openrouter_api_base])
    if args.openrouter_api_key:
        cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
    if args.api_base and (not args.openrouter or "--api-base" in supplied_options):
        cmd.extend(["--api-base", args.api_base])
    if args.api_key and (not args.openrouter or "--api-key" in supplied_options):
        cmd.extend(["--api-key", args.api_key])
    if args.environment:
        cmd.extend(["--environment", args.environment])
    if args.runner:
        cmd.extend(["--runner", args.runner])
    cmd.extend(passthrough)
    return cmd


def run_smoke(args: argparse.Namespace, passthrough: list[str]) -> dict[str, float]:
    cmd = build_command(args, passthrough)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    try:
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"benchmark did not emit metrics JSON:\n{proc.stdout}"
        ) from exc
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if float(metrics.get("is_valid", 0.0)) < 1.0:
        raise SystemExit("benchmark did not produce usable TB2 trial results")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live one-task Terminal-Bench 2 smoke run outside GigaEvo."
    )
    parser.add_argument("--candidate-repo", default=str(DEFAULT_CANDIDATE_REPO))
    parser.add_argument(
        "--task",
        default=os.environ.get("TB2_SMOKE_TASK", "extract-elf"),
    )
    parser.add_argument(
        "--agent-import-path",
        default=os.environ.get(
            "TB2_AGENT_IMPORT_PATH",
            "agents.baseline_terminus2:AgentHarness",
        ),
    )
    parser.add_argument("--agent", default=os.environ.get("TB2_AGENT"))
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("TB2_RUNS", "1")),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("TB2_CONCURRENCY", "1")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("TB2_TIMEOUT", "7200")),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TB2_MODEL") or os.environ.get("HARBOR_MODEL"),
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        default=os.environ.get("TB2_OPENROUTER", "").strip().lower()
        in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--openrouter-model",
        default=os.environ.get("TB2_OPENROUTER_MODEL")
        or os.environ.get("OPENROUTER_MODEL"),
    )
    parser.add_argument(
        "--openrouter-api-base",
        default=os.environ.get("TB2_OPENROUTER_API_BASE")
        or os.environ.get("OPENROUTER_API_BASE"),
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("TB2_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY"),
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("TB2_OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TB2_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--environment", default=os.environ.get("TB2_ENVIRONMENT"))
    parser.add_argument("--runner", default=os.environ.get("TB2_RUNNER"))
    return parser


def test_terminal_bench2_one_task_live() -> None:
    if os.environ.get("RUN_TB2_LIVE") != "1":
        import pytest

        pytest.skip("set RUN_TB2_LIVE=1 to run the live TB2 smoke test")
    args = build_parser().parse_args([])
    metrics = run_smoke(args, [])
    assert {"fitness", "is_valid", "cost"} <= set(metrics)


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    args.supplied_options = supplied_cli_options(sys.argv[1:])
    run_smoke(args, passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
