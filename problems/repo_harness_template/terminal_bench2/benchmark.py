#!/usr/bin/env python3
"""Run a Terminal-Bench 2 benchmark for a repo-harness candidate.

This script is intended to be used as ``repo_harness.benchmark_command``.
GigaEvo runs benchmark commands from a temporary checkout of the candidate
agent repository, so this script defaults to treating the current working
directory as the candidate repo.

It invokes Harbor, reads the per-trial ``result.json`` files Harbor writes, and
prints only the metrics JSON that GigaEvo needs on stdout:

    {"fitness": 0.0, "is_valid": 1.0, "cost": 0.0}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


DEFAULT_DATASET = "terminal-bench@2.0"
DEFAULT_AGENT_IMPORT_PATH = "agents.baseline_terminus2:AgentHarness"
DEFAULT_SMOKE_TASK = "extract-elf"
DEFAULT_JOBS_DIR = Path(__file__).resolve().parent / "jobs"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/google/gemini-3.1-flash-lite-preview"
STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"
DOCKER_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
DOCKER_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*__[a-z0-9][a-z0-9_-]*$")
AGENT_CONFIG_PATHS = (
    "gigaevo_agent_config.json",
    ".gigaevo/agent_config.json",
)
MAX_FAILURE_TEXT_CHARS = 3000
INFRA_FAILURE_MARKERS = (
    "all predefined address pools have been fully subnetted",
    "cannot connect to the docker daemon",
    "docker compose command failed",
    "error response from daemon",
    "failed to create network",
    "no such network",
    "permission denied while trying to connect to the docker api",
)
LLM_CALL_FAILURE_EXCEPTION_TYPES = {
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "InternalServerError",
    "OpenAIError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServiceUnavailableError",
}
LLM_CALL_FAILURE_MARKERS = (
    "unknown error in llm interaction",
    "openaiexception",
    "openai.api",
    "api connection error",
    "api status error",
    "api timeout",
    "apierror",
    "apiconnectionerror",
    "apitimeouterror",
    "badrequesterror",
    "authenticationerror",
    "permissiondeniederror",
    "ratelimiterror",
    "internalservererror",
    "serviceunavailable",
    "clientconnectorerror",
    "cannot connect to host",
    "connect call failed",
    "connection refused",
    "connection error",
    "readtimeout",
    "server disconnected",
    "temporarily unavailable",
)

# Same 30-task cheaper bring-up subset used by the TB2 reference example.
HARD_TASKS = [
    "bn-fit-modify",
    "cancel-async-tasks",
    "circuit-fibsqrt",
    "configure-git-webserver",
    "dna-assembly",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "fix-code-vulnerability",
    "fix-ocaml-gc",
    "gpt2-codegolf",
    "install-windows-3.11",
    "llm-inference-batching-scheduler",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "model-extraction-relu-logits",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-rust-c",
    "protein-assembly",
    "regex-chess",
    "sam-cell-seg",
    "sparql-university",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "video-processing",
    "write-compressor",
]

# Deterministic 20-task bring-up set with the most even difficulty mix the
# local TB2 dataset permits: 4 easy tasks exist, so this uses all 4 easy tasks
# plus 8 medium and 8 hard tasks, interleaved to avoid long same-difficulty runs.
BALANCED_20_TASKS = [
    "fix-git",
    "extract-elf",
    "configure-git-webserver",
    "cobol-modernization",
    "log-summary-date-ranges",
    "cancel-async-tasks",
    "prove-plus-comm",
    "sqlite-db-truncate",
    "regex-chess",
    "overfull-hbox",
    "git-multibranch",
    "model-extraction-relu-logits",
    "code-from-image",
    "path-tracing",
    "mailman",
    "fix-code-vulnerability",
    "polyglot-c-py",
    "train-fasttext",
    "query-optimize",
    "llm-inference-batching-scheduler",
]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class TrialRecord:
    task: str
    reward: float
    cost_usd: float | None
    result_path: str | None


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def tail(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def split_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in value.replace(",", " ").split():
            part = part.strip()
            if part:
                out.append(part)
    return out


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def default_workspace_root() -> Path | None:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "terminal-bench-2").is_dir():
            return parent
    return None


def default_dataset_root() -> Path | None:
    env_path = os.environ.get("TB2_DATASET_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    workspace_root = default_workspace_root()
    if workspace_root is None:
        return None
    root = workspace_root / "terminal-bench-2"
    return root if root.is_dir() else None


def discover_local_tasks(dataset_root: Path | None) -> list[str]:
    if dataset_root is None or not dataset_root.is_dir():
        return []
    return sorted(path.parent.name for path in dataset_root.glob("*/task.toml"))


def selected_tasks(args: argparse.Namespace) -> list[str]:
    explicit = split_values(args.tasks) + split_values(args.task)
    if explicit:
        tasks = explicit
    elif args.task_set == "smoke":
        tasks = [args.smoke_task]
    elif args.task_set == "balanced20":
        tasks = list(BALANCED_20_TASKS)
    elif args.task_set == "hard":
        tasks = list(HARD_TASKS)
    else:
        tasks = []

    if args.max_tasks is not None:
        if not tasks:
            dataset_root = Path(args.dataset_root) if args.dataset_root else None
            tasks = discover_local_tasks(dataset_root)
            if not tasks:
                raise SystemExit(
                    "--max-tasks with --task-set full requires a local TB2 "
                    "dataset root. Set --dataset-root or TB2_DATASET_ROOT."
                )
        tasks = tasks[: args.max_tasks]

    return tasks


def parse_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def build_env(args: argparse.Namespace, candidate_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.is_absolute():
            env_path = candidate_repo / env_path
        for key, value in parse_dotenv(env_path).items():
            if args.dotenv_override or key not in env:
                env[key] = value

    if args.model:
        env["HARBOR_MODEL"] = args.model
    if args.api_base:
        env["OPENAI_API_BASE"] = args.api_base
        env["OPENAI_BASE_URL"] = args.api_base
    if args.api_key:
        env["OPENAI_API_KEY"] = args.api_key
        if args.openrouter:
            env["OPENROUTER_API_KEY"] = args.api_key
    return env


def apply_openrouter_defaults(
    args: argparse.Namespace, supplied_options: set[str]
) -> None:
    if not args.openrouter:
        return
    if "--model" not in supplied_options:
        args.model = args.openrouter_model
    if "--api-base" not in supplied_options:
        args.api_base = args.openrouter_api_base
    if "--api-key" not in supplied_options and args.openrouter_api_key:
        args.api_key = args.openrouter_api_key


def supplied_cli_options(argv: list[str]) -> set[str]:
    options: set[str] = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        options.add(arg.split("=", 1)[0])
    return options


def default_runner(candidate_repo: Path) -> list[str]:
    runner = os.environ.get("TB2_RUNNER")
    if runner:
        return shlex.split(runner)
    if shutil.which("uv") and (candidate_repo / "pyproject.toml").exists():
        return ["uv", "run"]
    return []


def build_harbor_command(
    args: argparse.Namespace,
    *,
    candidate_repo: Path,
    job_name: str,
    jobs_dir: Path,
    tasks: list[str],
) -> list[str]:
    runner = shlex.split(args.runner) if args.runner else default_runner(candidate_repo)
    cmd = [*runner, "harbor", "run"]

    if args.agent:
        cmd.extend(["--agent", args.agent])
    else:
        cmd.extend(["--agent-import-path", args.agent_import_path])

    cmd.extend(["-d", args.dataset])
    if args.model:
        cmd.extend(["-m", args.model])
    if args.environment:
        cmd.extend(["-e", args.environment])

    cmd.extend(
        [
            "-n",
            str(args.concurrency),
            "--n-attempts",
            str(args.runs),
            "--job-name",
            job_name,
            "--jobs-dir",
            str(jobs_dir),
        ]
    )

    for task in tasks:
        cmd.extend(["-i", task])
    for kwarg in build_agent_kwargs(args, candidate_repo=candidate_repo):
        cmd.extend(["--agent-kwarg", kwarg])
    for extra in args.harbor_arg or []:
        cmd.append(extra)
    return cmd


def build_agent_kwargs(
    args: argparse.Namespace, candidate_repo: Path | None = None
) -> list[str]:
    """Build Harbor ``--agent-kwarg`` values from benchmark-local options."""

    kwargs = list(args.agent_kwarg or [])
    explicit_keys = {
        item.split("=", 1)[0].strip()
        for item in kwargs
        if "=" in item and item.split("=", 1)[0].strip()
    }
    if not args.local_llama_defaults:
        return _merge_candidate_agent_kwargs(
            kwargs,
            candidate_repo=candidate_repo,
            protected_keys=explicit_keys,
        )

    model_info = {
        "max_input_tokens": args.local_context_tokens,
        "max_output_tokens": args.local_output_tokens,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }
    defaults = [
        ("model_info", json.dumps(model_info, separators=(",", ":"))),
        ("proactive_summarization_threshold", str(args.local_summarization_threshold)),
        ("max_turns", str(args.local_max_turns)),
        ("temperature", str(args.local_temperature)),
        (
            "llm_call_kwargs",
            json.dumps(
                {"max_tokens": args.local_response_tokens},
                separators=(",", ":"),
            ),
        ),
    ]

    existing = set(explicit_keys)
    for key, value in defaults:
        if key not in existing:
            kwargs.append(f"{key}={value}")
            existing.add(key)
    return _merge_candidate_agent_kwargs(
        kwargs,
        candidate_repo=candidate_repo,
        protected_keys=explicit_keys,
    )


def _merge_candidate_agent_kwargs(
    kwargs: list[str],
    *,
    candidate_repo: Path | None,
    protected_keys: set[str],
) -> list[str]:
    config = load_candidate_agent_config(candidate_repo)
    agent_kwargs = config.get("agent_kwargs")
    if not isinstance(agent_kwargs, dict):
        return kwargs

    merged: list[str] = []
    replacement_by_key = {
        str(key): _agent_kwarg_value(value)
        for key, value in agent_kwargs.items()
        if str(key) and str(key) not in protected_keys
    }
    seen: set[str] = set()
    for item in kwargs:
        if "=" not in item:
            merged.append(item)
            continue
        key = item.split("=", 1)[0].strip()
        if key in replacement_by_key:
            merged.append(f"{key}={replacement_by_key[key]}")
            seen.add(key)
        else:
            merged.append(item)
            seen.add(key)

    for key, value in replacement_by_key.items():
        if key not in seen:
            merged.append(f"{key}={value}")
    return merged


def _agent_kwarg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load_candidate_agent_config(candidate_repo: Path | None) -> dict[str, Any]:
    if candidate_repo is None:
        return {}
    for relative_path in AGENT_CONFIG_PATHS:
        path = candidate_repo / relative_path
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            eprint(f"[tb2] warning: could not parse agent config {path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            eprint(f"[tb2] warning: agent config {path} is not a JSON object")
            return {}
        payload = dict(payload)
        payload["_path"] = str(path)
        return payload
    return {}


def extract_agent_kwargs_from_command(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    iterator = iter(command)
    for token in iterator:
        if token != "--agent-kwarg":
            continue
        try:
            item = next(iterator)
        except StopIteration:
            break
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value
    return values


def agent_config_summary(candidate_repo: Path, command: list[str]) -> dict[str, Any]:
    config = load_candidate_agent_config(candidate_repo)
    return {
        "candidate_config_path": config.get("_path"),
        "candidate_agent_kwargs": config.get("agent_kwargs")
        if isinstance(config.get("agent_kwargs"), dict)
        else {},
        "effective_agent_kwargs": extract_agent_kwargs_from_command(command),
    }


def add_runtime_limit_hints(
    structured_feedback: dict[str, Any],
    *,
    failure_summaries: list[dict[str, Any]],
    agent_config: dict[str, Any],
) -> None:
    effective_kwargs = agent_config.get("effective_agent_kwargs")
    if not isinstance(effective_kwargs, dict):
        return
    max_turns = coerce_float(effective_kwargs.get("max_turns"))
    if max_turns is None or max_turns <= 0:
        return

    hit_limit_tasks = []
    for item in failure_summaries:
        agent_metrics = item.get("agent_metrics")
        if not isinstance(agent_metrics, dict):
            continue
        episodes = coerce_float(agent_metrics.get("n_episodes"))
        if episodes is None or episodes < max_turns:
            continue
        task = item.get("task")
        if task:
            hit_limit_tasks.append(str(task))

    if not hit_limit_tasks:
        return
    hints = structured_feedback.setdefault("hints", [])
    if not isinstance(hints, list):
        return
    tasks = ", ".join(sorted(set(hit_limit_tasks))[:8])
    hints.append(
        "One or more failed trials reached the configured max_turns="
        f"{int(max_turns)} limit ({tasks}). Consider increasing max_turns or "
        "making the harness reach decisive actions faster."
    )


def harbor_compose_project_name(name: str) -> str:
    """Approximate Harbor's Docker Compose project-name normalization."""

    normalized = re.sub(r"[^a-z0-9_-]+", "", name.lower())
    return normalized.strip("-_")


def known_terminal_bench_tasks(
    *, args: argparse.Namespace, selected: list[str]
) -> set[str]:
    tasks = set(selected)
    tasks.update(HARD_TASKS)
    tasks.update(BALANCED_20_TASKS)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    tasks.update(discover_local_tasks(dataset_root))
    return {harbor_compose_project_name(task) for task in tasks if task}


def is_harbor_compose_project(project: str, known_tasks: set[str]) -> bool:
    project = harbor_compose_project_name(project)
    if not DOCKER_PROJECT_RE.match(project):
        return False
    task = project.rsplit("__", 1)[0]
    return not known_tasks or task in known_tasks


def docker_run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["docker", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        eprint("[tb2] docker cleanup skipped: docker executable not found")
    except subprocess.TimeoutExpired:
        eprint(f"[tb2] docker cleanup command timed out: docker {' '.join(args)}")
    except OSError as exc:
        eprint(f"[tb2] docker cleanup command failed to start: {exc}")
    return None


def docker_lines(args: list[str], *, timeout: float) -> list[str]:
    result = docker_run(args, timeout=timeout)
    if result is None:
        return []
    if result.returncode != 0:
        message = tail((result.stderr or result.stdout or "").strip(), 1000)
        eprint(
            f"[tb2] docker cleanup command failed ({result.returncode}): "
            f"docker {' '.join(args)}\n{message}"
        )
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def docker_compose_projects_from_job_dir(job_dir: Path) -> set[str]:
    projects: set[str] = set()
    if not job_dir.is_dir():
        return projects

    for trial_dir in job_dir.iterdir():
        if trial_dir.is_dir() and "__" in trial_dir.name:
            project = harbor_compose_project_name(trial_dir.name)
            if project:
                projects.add(project)

    for path in job_dir.glob("**/exception.txt"):
        text = read_text(path, max_chars=MAX_FAILURE_TEXT_CHARS)
        for match in re.finditer(r"--project-name\s+([^\s]+)", text):
            project = harbor_compose_project_name(match.group(1))
            if project:
                projects.add(project)

    return projects


def docker_compose_projects_from_daemon(
    *, known_tasks: set[str], timeout: float
) -> set[str]:
    projects: set[str] = set()
    for project in docker_lines(
        [
            "ps",
            "-a",
            "--filter",
            f"label={DOCKER_COMPOSE_PROJECT_LABEL}",
            "--format",
            f"{{{{.Label \"{DOCKER_COMPOSE_PROJECT_LABEL}\"}}}}",
        ],
        timeout=timeout,
    ):
        project = harbor_compose_project_name(project)
        if is_harbor_compose_project(project, known_tasks):
            projects.add(project)

    for network_name in docker_lines(
        ["network", "ls", "--format", "{{.Name}}"],
        timeout=timeout,
    ):
        if not network_name.endswith("_default"):
            continue
        project = harbor_compose_project_name(network_name[: -len("_default")])
        if is_harbor_compose_project(project, known_tasks):
            projects.add(project)

    return projects


def cleanup_docker_projects(
    projects: set[str],
    *,
    phase: str,
    timeout: float,
    force_running: bool,
) -> None:
    if not projects:
        return

    removed_containers = 0
    removed_networks = 0
    removed_volumes = 0
    cleaned_projects = 0
    skipped_running: list[str] = []

    for project in sorted(projects):
        project = harbor_compose_project_name(project)
        if not project:
            continue
        label_filter = f"label={DOCKER_COMPOSE_PROJECT_LABEL}={project}"
        running = docker_lines(["ps", "-q", "--filter", label_filter], timeout=timeout)
        if running and not force_running:
            skipped_running.append(project)
            continue

        containers = docker_lines(
            ["ps", "-aq", "--filter", label_filter],
            timeout=timeout,
        )
        if containers:
            result = docker_run(["rm", "-f", *containers], timeout=timeout)
            if result is not None and result.returncode == 0:
                removed_containers += len(containers)
            elif result is not None:
                eprint(
                    f"[tb2] docker cleanup failed to remove containers for {project}: "
                    f"{tail(result.stderr or result.stdout or '', 1000)}"
                )

        volumes = docker_lines(
            ["volume", "ls", "-q", "--filter", label_filter],
            timeout=timeout,
        )
        if volumes:
            result = docker_run(["volume", "rm", "-f", *volumes], timeout=timeout)
            if result is not None and result.returncode == 0:
                removed_volumes += len(volumes)
            elif result is not None:
                eprint(
                    f"[tb2] docker cleanup failed to remove volumes for {project}: "
                    f"{tail(result.stderr or result.stdout or '', 1000)}"
                )

        network_names = docker_lines(
            ["network", "ls", "--format", "{{.Name}}"],
            timeout=timeout,
        )
        networks = [
            name
            for name in network_names
            if name.endswith("_default")
            if harbor_compose_project_name(name[: -len("_default")]) == project
        ]
        if networks:
            result = docker_run(["network", "rm", *networks], timeout=timeout)
            if result is not None and result.returncode == 0:
                removed_networks += len(networks)
            elif result is not None:
                eprint(
                    f"[tb2] docker cleanup failed to remove networks for {project}: "
                    f"{tail(result.stderr or result.stdout or '', 1000)}"
                )

        cleaned_projects += 1

    if skipped_running:
        eprint(
            f"[tb2] docker cleanup {phase}: skipped {len(skipped_running)} "
            "active Harbor project(s)"
        )
    if removed_containers or removed_networks or removed_volumes:
        eprint(
            f"[tb2] docker cleanup {phase}: removed {removed_containers} "
            f"container(s), {removed_networks} network(s), {removed_volumes} "
            f"volume(s) across {cleaned_projects} Harbor project(s)"
        )


def cleanup_stale_harbor_docker(
    *, args: argparse.Namespace, tasks: list[str], phase: str
) -> None:
    known_tasks = known_terminal_bench_tasks(args=args, selected=tasks)
    projects = docker_compose_projects_from_daemon(
        known_tasks=known_tasks,
        timeout=args.docker_cleanup_timeout,
    )
    cleanup_docker_projects(
        projects,
        phase=phase,
        timeout=args.docker_cleanup_timeout,
        force_running=False,
    )


def cleanup_job_harbor_docker(
    *, args: argparse.Namespace, job_dir: Path, phase: str
) -> None:
    projects = docker_compose_projects_from_job_dir(job_dir)
    cleanup_docker_projects(
        projects,
        phase=phase,
        timeout=args.docker_cleanup_timeout,
        force_running=True,
    )


def terminate_process_group(proc: subprocess.Popen[str], *, grace_seconds: float) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        eprint(f"[tb2] warning: failed to terminate Harbor process group: {exc}")
        proc.terminate()

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:
        eprint(f"[tb2] warning: failed to kill Harbor process group: {exc}")
        proc.kill()
    proc.wait()


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float | None,
    log_dir: Path,
) -> CommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "harbor_stdout.log"
    stderr_path = log_dir / "harbor_stderr.log"
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def pump(stream, path: Path, chunks: list[str], prefix: str) -> None:
        with path.open("w", encoding="utf-8", errors="replace") as fh:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                fh.write(line)
                fh.flush()
                eprint(f"{prefix}{line.rstrip()}")
        stream.close()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))

    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_thread = threading.Thread(
        target=pump,
        args=(proc.stdout, stdout_path, stdout_chunks, "[harbor] "),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=(proc.stderr, stderr_path, stderr_chunks, "[harbor:err] "),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(proc, grace_seconds=60)
        stderr_chunks.append(f"\nTimed out after {timeout}s")
        with stderr_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\nTimed out after {timeout}s")
    except BaseException:
        terminate_process_group(proc, grace_seconds=30)
        raise

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    return CommandResult(
        proc.returncode if proc.returncode is not None else 124,
        "".join(stdout_chunks),
        "".join(stderr_chunks),
        timed_out=timed_out,
    )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_text(path: Path, max_chars: int | None = None) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    if max_chars is not None:
        return tail(text, max_chars)
    return text


def artifact_descriptor(
    *,
    name: str,
    path: Path,
    kind: str,
    role: str = "failure",
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "kind": kind,
        "role": role,
        "description": description,
    }


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_trial_records(job_dir: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    if not job_dir.is_dir():
        return records

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            records.append(TrialRecord(task, 0.0, None, None))
            continue

        payload = read_json(result_path)
        if payload is None:
            records.append(TrialRecord(task, 0.0, None, str(result_path)))
            continue

        verifier_result = payload.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        reward = coerce_float(rewards.get("reward"))
        if reward is None:
            reward = coerce_float(verifier_result.get("reward"))
        if reward is None:
            reward = 0.0

        agent_result = payload.get("agent_result") or {}
        cost = coerce_float(agent_result.get("cost_usd"))
        records.append(TrialRecord(task, reward, cost, str(result_path)))

    return records


def unique_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def extract_context_errors(text: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for match in re.finditer(
        r"maximum context length is\s+(\d+)\s+tokens\.\s+"
        r"However, you requested\s+(\d+)\s+tokens",
        text,
        flags=re.IGNORECASE,
    ):
        max_tokens = int(match.group(1))
        requested_tokens = int(match.group(2))
        errors.append(
            {
                "type": "context_length_exceeded",
                "max_tokens": max_tokens,
                "requested_tokens": requested_tokens,
                "over_by_tokens": requested_tokens - max_tokens,
            }
        )
    if "could not broadcast input array" in text:
        errors.append(
            {
                "type": "llama_cpp_context_overflow",
                "message": (
                    "llama.cpp raised a broadcast-shape ValueError, which usually "
                    "means the request exceeded the server n_ctx window before it "
                    "could return a normal context-length error."
                ),
            }
        )
    return unique_dicts(errors)


def extract_litellm_model_warnings(text: str) -> list[str]:
    warnings = []
    for line in text.splitlines():
        if "Failed to retrieve model info" in line or "fallback context limit" in line:
            warnings.append(line.strip())
    return sorted(set(warnings))[:8]


def is_llm_call_failure(summary: dict[str, Any]) -> bool:
    """Return True for model/API-call failures, not task failures."""

    if summary.get("context_errors"):
        return True

    exception_type = str(summary.get("exception_type") or "")
    exception_type_name = exception_type.rsplit(".", 1)[-1]
    if exception_type_name in LLM_CALL_FAILURE_EXCEPTION_TYPES:
        return True

    text = "\n".join(
        str(summary.get(key) or "")
        for key in (
            "exception_type",
            "exception_message",
            "trial_log_tail",
            "traceback",
        )
    ).lower()
    return any(marker in text for marker in LLM_CALL_FAILURE_MARKERS)


def is_infrastructure_failure(summary: dict[str, Any]) -> bool:
    text = "\n".join(
        str(summary.get(key) or "")
        for key in (
            "exception_type",
            "exception_message",
            "trial_log_tail",
            "traceback",
        )
    ).lower()
    return any(marker in text for marker in INFRA_FAILURE_MARKERS)


def extract_episode_summary(trial_dir: Path) -> dict[str, Any]:
    agent_dir = trial_dir / "agent"
    episode_dirs = sorted(
        agent_dir.glob("episode-*"),
        key=lambda path: int(path.name.split("-", 1)[1])
        if path.name.split("-", 1)[1].isdigit()
        else 0,
    )
    response_texts = [
        read_text(path / "response.txt", max_chars=MAX_FAILURE_TEXT_CHARS)
        for path in episode_dirs
        if (path / "response.txt").exists()
    ]
    technical_difficulties = sum(
        "Technical difficulties" in text for text in response_texts
    )
    no_valid_json = sum("No valid JSON object found" in text for text in response_texts)
    return {
        "episode_count": len(episode_dirs),
        "response_count": len(response_texts),
        "technical_difficulties_responses": technical_difficulties,
        "no_valid_json_responses": no_valid_json,
        "last_response_excerpt": tail(response_texts[-1], 800)
        if response_texts
        else "",
        **(
            extract_trajectory_summary(agent_dir / "trajectory.json")
            if not response_texts
            else {}
        ),
    }


def extract_trajectory_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        return {}
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return {}

    agent_steps = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("source") == "agent"
    ]
    last_agent = agent_steps[-1] if agent_steps else {}
    tool_calls = last_agent.get("tool_calls") if isinstance(last_agent, dict) else None
    commands: list[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, dict) and isinstance(
                arguments.get("keystrokes"), str
            ):
                commands.append(arguments["keystrokes"])

    observation_excerpt = ""
    observation = last_agent.get("observation") if isinstance(last_agent, dict) else None
    if isinstance(observation, dict):
        results = observation.get("results")
        if isinstance(results, list) and results:
            content = results[-1].get("content") if isinstance(results[-1], dict) else ""
            observation_excerpt = tail(str(content or ""), 800)

    return {
        "trajectory_step_count": len(steps),
        "agent_step_count": len(agent_steps),
        "last_agent_message_excerpt": tail(str(last_agent.get("message") or ""), 800)
        if isinstance(last_agent, dict)
        else "",
        "last_commands": commands[-5:],
        "last_observation_excerpt": observation_excerpt,
    }


def trial_artifacts(trial_dir: Path, result_path: Path) -> list[dict[str, Any]]:
    candidates = [
        artifact_descriptor(
            name="result.json",
            path=result_path,
            kind="trial_result",
            description="Harbor per-trial result JSON.",
        ),
        artifact_descriptor(
            name="trial.log",
            path=trial_dir / "trial.log",
            kind="trial_log",
            description="Full per-trial Harbor log.",
        ),
        artifact_descriptor(
            name="exception.txt",
            path=trial_dir / "exception.txt",
            kind="exception",
            description="Exception traceback or message when present.",
        ),
        artifact_descriptor(
            name="trajectory.json",
            path=trial_dir / "agent" / "trajectory.json",
            kind="agent_trajectory",
            description="Agent trajectory with prompts, actions, observations, and metrics.",
        ),
        artifact_descriptor(
            name="terminal_pane.txt",
            path=trial_dir / "agent" / "terminus_2.pane",
            kind="terminal_pane",
            description="Final terminal pane capture.",
        ),
        artifact_descriptor(
            name="verifier_stdout.txt",
            path=trial_dir / "verifier" / "test-stdout.txt",
            kind="verifier_stdout",
            description="Verifier stdout explaining pass/fail details.",
        ),
        artifact_descriptor(
            name="verifier_ctrf.json",
            path=trial_dir / "verifier" / "ctrf.json",
            kind="verifier_result",
            description="Verifier structured result when available.",
        ),
    ]
    return [item for item in candidates if Path(str(item["path"])).exists()]


def build_mutation_hints(
    *,
    exception_type: str | None,
    context_errors: list[dict[str, Any]],
    litellm_model_warnings: list[str],
    combined_text: str,
    episode_summary: dict[str, Any],
) -> list[str]:
    hints: list[str] = []
    if context_errors:
        hints.extend(
            [
                "Reduce prompt/history growth so the agent stays below the local model context window.",
                "Summarize or discard old terminal observations before reaching 70-80% of the configured context.",
                "Register or alias the local LiteLLM model with its real context window so Harbor summarizes earlier.",
                "Increase llama.cpp --n_ctx for smoke tests if VRAM allows, but do not rely on large context as the only fix.",
            ]
        )
        if any(
            error.get("type") == "llama_cpp_context_overflow"
            for error in context_errors
        ):
            hints.append(
                "For local llama.cpp, rerun this benchmark with --local-llama-defaults "
                "or pass explicit Terminus-2 agent kwargs for model_info, max_tokens, "
                "and proactive_summarization_threshold."
            )
    if litellm_model_warnings:
        hints.append(
            "Avoid unknown LiteLLM model metadata for local aliases; use a known OpenAI-compatible alias or custom model cost/context map."
        )
    if exception_type == "AgentTimeoutError":
        hints.append(
            "Shorten the agent loop and add decisive fallback behavior after repeated LLM/context failures to avoid timing out."
        )
    if "Output length exceeded" in combined_text or "hit max_tokens limit" in combined_text:
        hints.append(
            "Constrain model responses to concise action JSON and lower unnecessary max output tokens."
        )
    if episode_summary.get("technical_difficulties_responses", 0) > 0:
        hints.append(
            "Replace repeated generic 'Technical difficulties' fallback turns with a state reset, compact summary, or immediate recovery action."
        )
    if "Parser warnings" in combined_text:
        hints.append(
            "Make the agent response format stricter so parser warnings do not waste turns."
        )
    if any(marker in combined_text.lower() for marker in INFRA_FAILURE_MARKERS):
        hints.append(
            "Host Docker/Harbor setup failed before agent execution; clean stale "
            "Terminal-Bench Docker resources instead of mutating the agent."
        )
    if not hints:
        hints.append(
            "Inspect trial.log, exception.txt, and agent/episode-* logs for the concrete failure before mutating the harness."
        )
    return hints


def extract_trial_failure_summary(trial_dir: Path) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    payload = read_json(result_path) or {}
    exception_info = payload.get("exception_info") or {}
    agent_result = payload.get("agent_result") or {}
    agent_metadata = agent_result.get("metadata") or {}
    verifier_result = payload.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}

    exception_text = read_text(trial_dir / "exception.txt")
    trial_log_tail = read_text(trial_dir / "trial.log", max_chars=MAX_FAILURE_TEXT_CHARS)
    traceback_text = str(exception_info.get("exception_traceback") or "")
    combined_text = "\n".join([exception_text, trial_log_tail, traceback_text])
    episode_summary = extract_episode_summary(trial_dir)
    context_errors = extract_context_errors(combined_text)
    litellm_model_warnings = extract_litellm_model_warnings(combined_text)
    exception_type = exception_info.get("exception_type")

    summary = {
        "trial": trial_dir.name,
        "task": payload.get("task_name") or trial_dir.name.rsplit("__", 1)[0],
        "reward": rewards.get("reward"),
        "result_path": str(result_path) if result_path.exists() else None,
        "exception_type": exception_type,
        "exception_message": exception_info.get("exception_message"),
        "context_errors": context_errors,
        "litellm_model_warnings": litellm_model_warnings,
        "agent_metrics": {
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "n_cache_tokens": agent_result.get("n_cache_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
            "n_episodes": agent_metadata.get("n_episodes"),
            "summarization_count": agent_metadata.get("summarization_count"),
            "api_request_count": len(agent_metadata.get("api_request_times_msec", [])),
        },
        "episode_summary": episode_summary,
        "trial_log_tail": trial_log_tail,
        "artifacts": trial_artifacts(trial_dir, result_path),
        "mutation_hints": build_mutation_hints(
            exception_type=exception_type,
            context_errors=context_errors,
            litellm_model_warnings=litellm_model_warnings,
            combined_text=combined_text,
            episode_summary=episode_summary,
        ),
    }
    summary["llm_call_failed"] = is_llm_call_failure(summary)
    summary["infrastructure_failed"] = is_infrastructure_failure(summary)
    return summary


def collect_failure_summaries(job_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    if not job_dir.is_dir():
        return summaries
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        result_path = trial_dir / "result.json"
        payload = read_json(result_path) or {}
        rewards = ((payload.get("verifier_result") or {}).get("rewards") or {})
        reward = coerce_float(rewards.get("reward"))
        has_exception = bool(payload.get("exception_info")) or (
            trial_dir / "exception.txt"
        ).exists()
        if has_exception or reward is None or reward <= 0:
            summaries.append(extract_trial_failure_summary(trial_dir))
    return summaries


def extract_top_level_result(job_dir: Path) -> dict[str, Any] | None:
    payload = read_json(job_dir / "result.json")
    if not payload:
        return None
    stats = payload.get("stats") or {}
    evals = stats.get("evals") or {}
    exception_stats: dict[str, int] = {}
    for eval_payload in evals.values():
        for exc_name, trials in (eval_payload.get("exception_stats") or {}).items():
            exception_stats[exc_name] = exception_stats.get(exc_name, 0) + len(trials)
    return {
        "n_total_trials": payload.get("n_total_trials"),
        "n_errors": stats.get("n_errors"),
        "exception_counts": exception_stats,
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
    }


def compute_metrics(
    records: list[TrialRecord],
    returncode: int,
    failure_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    if not records:
        return {"fitness": 0.0, "is_valid": 0.0, "cost": 0.0}

    successes = sum(1 for record in records if record.reward > 0)
    fitness = successes / len(records)
    costs = [record.cost_usd for record in records if record.cost_usd is not None]
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    has_llm_call_failure = any(
        bool(summary.get("llm_call_failed")) or is_llm_call_failure(summary)
        for summary in (failure_summaries or [])
    )
    has_infrastructure_failure = any(
        bool(summary.get("infrastructure_failed"))
        or is_infrastructure_failure(summary)
        for summary in (failure_summaries or [])
    )
    is_valid = (
        1.0
        if returncode in (0, 124)
        and not has_llm_call_failure
        and not has_infrastructure_failure
        else 0.0
    )
    return {
        "fitness": round(fitness, 6),
        "is_valid": is_valid,
        "cost": round(mean_cost, 6),
    }


def build_case_results(records: list[TrialRecord]) -> list[dict[str, Any]]:
    by_task: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)

    cases: list[dict[str, Any]] = []
    for task, task_records in sorted(by_task.items()):
        rewards = [record.reward for record in task_records]
        reward = sum(rewards) / len(rewards) if rewards else 0.0
        costs = [
            record.cost_usd
            for record in task_records
            if record.cost_usd is not None
        ]
        mean_cost = sum(costs) / len(costs) if costs else None
        cases.append(
            {
                "case_id": task,
                "task": task,
                "reward": round(reward, 6),
                "passed": reward > 0,
                "status": "passed" if reward > 0 else "failed",
                "n_trials": len(task_records),
                "mean_cost_usd": round(mean_cost, 6)
                if mean_cost is not None
                else None,
                "result_paths": [
                    record.result_path
                    for record in task_records
                    if record.result_path is not None
                ],
            }
        )
    return cases


def compact_failure_summary(item: dict[str, Any]) -> dict[str, Any]:
    llm_call_failed = bool(item.get("llm_call_failed")) or is_llm_call_failure(item)
    infrastructure_failed = bool(
        item.get("infrastructure_failed")
    ) or is_infrastructure_failure(item)
    return {
        "task": item.get("task"),
        "trial": item.get("trial"),
        "reward": item.get("reward"),
        "exception_type": item.get("exception_type"),
        "exception_message": item.get("exception_message"),
        "context_errors": item.get("context_errors"),
        "agent_metrics": item.get("agent_metrics"),
        "episode_summary": item.get("episode_summary"),
        "llm_call_failed": llm_call_failed,
        "infrastructure_failed": infrastructure_failed,
        "mutation_hints": item.get("mutation_hints"),
        "result_path": item.get("result_path"),
        "artifacts": item.get("artifacts"),
    }


def unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def build_structured_feedback(
    *,
    result: CommandResult,
    tasks: list[str],
    records: list[TrialRecord],
    metrics: dict[str, float],
    failure_summaries: list[dict[str, Any]],
    top_level_result: dict[str, Any] | None,
    task_set: str,
) -> dict[str, Any]:
    cases = build_case_results(records)
    failed_by_task: dict[str, int] = {}
    exception_counts: dict[str, int] = {}
    hints: list[Any] = []
    llm_call_failures = 0
    infrastructure_failures = 0
    for item in failure_summaries:
        task = str(item.get("task") or "unknown")
        failed_by_task[task] = failed_by_task.get(task, 0) + 1
        exc = item.get("exception_type")
        if exc:
            exc_name = str(exc)
            exception_counts[exc_name] = exception_counts.get(exc_name, 0) + 1
        if bool(item.get("llm_call_failed")) or is_llm_call_failure(item):
            llm_call_failures += 1
        if bool(item.get("infrastructure_failed")) or is_infrastructure_failure(item):
            infrastructure_failures += 1
        hints.extend(item.get("mutation_hints") or [])

    failure_clusters = [
        {"name": task, "count": count}
        for task, count in sorted(
            failed_by_task.items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]
    return {
        "schema_version": 1,
        "benchmark": "terminal_bench_2",
        "summary": {
            "task_set": task_set,
            "selected_tasks": tasks,
            "selected_cases": tasks,
            "n_trials": len(records),
            "failed_trials": len(failure_summaries),
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "metrics": metrics,
            "exception_counts": exception_counts,
            "llm_call_failures": llm_call_failures,
            "infrastructure_failures": infrastructure_failures,
        },
        "failure_clusters": failure_clusters,
        "cases": cases,
        "examples": [compact_failure_summary(item) for item in failure_summaries[:5]],
        "hints": unique_strings(hints),
        "raw": {
            "top_level_result": top_level_result,
        },
    }


def write_summary(
    *,
    args: argparse.Namespace,
    command: list[str],
    result: CommandResult,
    job_dir: Path,
    tasks: list[str],
    records: list[TrialRecord],
    metrics: dict[str, float],
    failure_summaries: list[dict[str, Any]] | None = None,
) -> None:
    per_task: dict[str, list[float]] = {}
    for record in records:
        per_task.setdefault(record.task, []).append(record.reward)
    if failure_summaries is None:
        failure_summaries = collect_failure_summaries(job_dir)
    top_level_result = extract_top_level_result(job_dir)
    structured_feedback = build_structured_feedback(
        result=result,
        tasks=tasks,
        records=records,
        metrics=metrics,
        failure_summaries=failure_summaries,
        top_level_result=top_level_result,
        task_set=args.task_set,
    )
    agent_config = agent_config_summary(Path(args.candidate_repo).resolve(), command)
    structured_feedback["agent_config"] = agent_config
    add_runtime_limit_hints(
        structured_feedback,
        failure_summaries=failure_summaries,
        agent_config=agent_config,
    )

    summary = {
        "metrics": metrics,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "job_dir": str(job_dir),
        "command": command,
        "task_set": args.task_set,
        "selected_tasks": tasks,
        "n_trials": len(records),
        "per_task_rewards": per_task,
        "top_level_result": top_level_result,
        "failure_summaries": failure_summaries,
        "structured_feedback": structured_feedback,
        "agent_config": agent_config,
        "stdout_tail": tail(result.stdout),
        "stderr_tail": tail(result.stderr),
    }

    if failure_summaries:
        eprint(STRUCTURED_FEEDBACK_MARKER)
        eprint(json.dumps(structured_feedback, indent=2, sort_keys=True))

    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else job_dir / "gigaevo_summary.json"
    )
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    except OSError as exc:
        eprint(f"[tb2] warning: failed to write summary {summary_path}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench 2 through Harbor and emit GigaEvo metrics JSON."
    )
    parser.add_argument(
        "--candidate-repo",
        default=os.environ.get("TB2_CANDIDATE_REPO", "."),
        help="Candidate agent repo to run Harbor from. Defaults to cwd.",
    )
    parser.add_argument(
        "--agent-import-path",
        default=os.environ.get("TB2_AGENT_IMPORT_PATH", DEFAULT_AGENT_IMPORT_PATH),
        help=(
            "Harbor custom agent import path. Ignored when --agent is set. "
            f"Default: {DEFAULT_AGENT_IMPORT_PATH}"
        ),
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("TB2_AGENT"),
        help="Use a Harbor-installed agent name instead of --agent-import-path.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TB2_DATASET", DEFAULT_DATASET),
        help=f"Harbor dataset id. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--task-set",
        choices=["full", "hard", "balanced20", "smoke"],
        default=os.environ.get("TB2_TASK_SET", "full"),
        help=(
            "Task preset. full passes no -i flags unless --max-tasks is used. "
            "balanced20 is a deterministic 4 easy / 8 medium / 8 hard mix."
        ),
    )
    parser.add_argument(
        "--tasks",
        action="append",
        default=None,
        help="Comma/space separated explicit task ids. May be repeated.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Single task id alias for --tasks. May be repeated.",
    )
    parser.add_argument(
        "--smoke-task",
        default=os.environ.get("TB2_SMOKE_TASK", DEFAULT_SMOKE_TASK),
        help=f"Task used by --task-set smoke. Default: {DEFAULT_SMOKE_TASK}",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=env_int("TB2_MAX_TASKS"),
        help="Truncate the selected task list. For full, requires local task files.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root() or ""),
        help="Local terminal-bench-2 task directory, used only for task discovery.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=env_int("TB2_RUNS", 1),
        help="Harbor --n-attempts value.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=env_int("TB2_CONCURRENCY", 1),
        help="Harbor --n-concurrent value.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TB2_MODEL") or os.environ.get("HARBOR_MODEL"),
        help="Harbor/LiteLLM model name, for example openai/qwen-local.",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        default=env_bool("TB2_OPENROUTER", True),
        help=(
            "Use OpenRouter instead of a local OpenAI-compatible model. Sets the "
            "OpenRouter API base/key and a default openrouter/... model unless "
            "--model, --api-base, or --api-key are supplied."
        ),
    )
    parser.add_argument(
        "--openrouter-model",
        default=env_first("TB2_OPENROUTER_MODEL", "OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL,
        help=(
            "Model used by --openrouter when --model is omitted. "
            f"Default: {DEFAULT_OPENROUTER_MODEL}"
        ),
    )
    parser.add_argument(
        "--openrouter-api-base",
        default=env_first("TB2_OPENROUTER_API_BASE", "OPENROUTER_API_BASE")
        or DEFAULT_OPENROUTER_API_BASE,
        help=(
            "API base used by --openrouter when --api-base is omitted. "
            f"Default: {DEFAULT_OPENROUTER_API_BASE}"
        ),
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=env_first(
            "TB2_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
            "TB2_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
        help=(
            "API key used by --openrouter when --api-key is omitted. Defaults to "
            "TB2_OPENROUTER_API_KEY, OPENROUTER_API_KEY, TB2_OPENAI_API_KEY, "
            "or OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("TB2_OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE"),
        help="OpenAI-compatible API base for local/proxy models.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TB2_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        help="OpenAI-compatible API key for local/proxy models.",
    )
    parser.add_argument(
        "--environment",
        default=os.environ.get("TB2_ENVIRONMENT"),
        help="Optional Harbor environment, e.g. runloop. Omit for local Docker default.",
    )
    parser.add_argument(
        "--runner",
        default=os.environ.get("TB2_RUNNER"),
        help="Command prefix before `harbor run`, e.g. `uv run`.",
    )
    parser.add_argument(
        "--jobs-dir",
        default=os.environ.get("TB2_JOBS_DIR", str(DEFAULT_JOBS_DIR)),
        help="Directory where Harbor writes job outputs.",
    )
    parser.add_argument(
        "--job-name",
        default=os.environ.get("TB2_JOB_NAME"),
        help="Harbor job name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=env_float("TB2_TIMEOUT", 7200.0),
        help="Outer timeout in seconds for Harbor.",
    )
    parser.add_argument(
        "--docker-cleanup",
        choices=["off", "after", "before-and-after"],
        default=os.environ.get("TB2_DOCKER_CLEANUP", "before-and-after"),
        help=(
            "Clean Harbor Docker Compose containers/networks/volumes owned by "
            "Terminal-Bench projects. before-and-after also removes stale stopped "
            "projects before launching Harbor."
        ),
    )
    parser.add_argument(
        "--docker-cleanup-timeout",
        type=float,
        default=env_float("TB2_DOCKER_CLEANUP_TIMEOUT", 30.0),
        help="Timeout in seconds for each Docker cleanup command.",
    )
    parser.add_argument(
        "--env-file",
        default=os.environ.get("TB2_ENV_FILE", ".env"),
        help="Optional .env file to load from the candidate repo.",
    )
    parser.add_argument(
        "--dotenv-override",
        action="store_true",
        help="Let --env-file values override the current process environment.",
    )
    parser.add_argument(
        "--harbor-arg",
        action="append",
        default=None,
        help="Additional single token to append to the Harbor command. Repeat as needed.",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=None,
        help=(
            "Additional Harbor --agent-kwarg value in key=value form. Repeat as needed. "
            "Useful for Terminus-2 kwargs such as max_turns or model_info."
        ),
    )
    parser.add_argument(
        "--local-llama-defaults",
        action="store_true",
        default=env_bool("TB2_LOCAL_LLAMA_DEFAULTS", False),
        help=(
            "Append conservative Terminus-2 kwargs for local llama.cpp/OpenAI-compatible "
            "servers so Harbor summarizes before the server n_ctx limit is reached."
        ),
    )
    parser.add_argument(
        "--local-context-tokens",
        type=int,
        default=env_int("TB2_LOCAL_CONTEXT_TOKENS", 8192),
        help="model_info.max_input_tokens used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--local-output-tokens",
        type=int,
        default=env_int("TB2_LOCAL_OUTPUT_TOKENS", 1024),
        help="model_info.max_output_tokens used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--local-response-tokens",
        type=int,
        default=env_int("TB2_LOCAL_RESPONSE_TOKENS", 1024),
        help="Per-call max_tokens used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--local-summarization-threshold",
        type=int,
        default=env_int("TB2_LOCAL_SUMMARIZATION_THRESHOLD", 3000),
        help="proactive_summarization_threshold used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--local-max-turns",
        type=int,
        default=env_int("TB2_LOCAL_MAX_TURNS", 8),
        help="max_turns used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--local-temperature",
        type=float,
        default=env_float("TB2_LOCAL_TEMPERATURE", 0.2),
        help="temperature used by --local-llama-defaults.",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("TB2_SUMMARY_PATH"),
        help="Optional path for a detailed JSON summary.",
    )
    parser.add_argument(
        "--parse-only-job-dir",
        default=None,
        help="Parse an existing Harbor job dir instead of launching Harbor.",
    )
    return parser


class SignalExit(SystemExit):
    pass


def install_termination_signal_handlers() -> None:
    def raise_signal_exit(signum: int, _frame: Any) -> None:
        raise SignalExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, raise_signal_exit)
        except (OSError, ValueError):
            continue


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    apply_openrouter_defaults(args, supplied_cli_options(argv))
    candidate_repo = Path(args.candidate_repo).expanduser().resolve()
    if not candidate_repo.is_dir():
        eprint(f"[tb2] candidate repo does not exist: {candidate_repo}")
        print(json.dumps({"fitness": 0.0, "is_valid": 0.0, "cost": 0.0}))
        return 0

    tasks = selected_tasks(args)
    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    job_name = args.job_name or f"tb2-{int(time.time())}-{os.getpid()}"
    job_dir = (
        Path(args.parse_only_job_dir).expanduser().resolve()
        if args.parse_only_job_dir
        else jobs_dir / job_name
    )

    command: list[str] = []
    result = CommandResult(0, "", "")
    if args.parse_only_job_dir:
        eprint(f"[tb2] parsing existing job dir: {job_dir}")
    else:
        install_termination_signal_handlers()
        if args.docker_cleanup == "before-and-after":
            cleanup_stale_harbor_docker(args=args, tasks=tasks, phase="before")
        command = build_harbor_command(
            args,
            candidate_repo=candidate_repo,
            job_name=job_name,
            jobs_dir=jobs_dir,
            tasks=tasks,
        )
        env = build_env(args, candidate_repo)
        eprint(f"[tb2] candidate repo: {candidate_repo}")
        eprint(f"[tb2] job dir: {job_dir}")
        eprint(f"[tb2] tasks: {', '.join(tasks) if tasks else 'full dataset'}")
        eprint(f"[tb2] command: {' '.join(shlex.quote(part) for part in command)}")
        try:
            result = run_command(
                command,
                cwd=candidate_repo,
                env=env,
                timeout=args.timeout,
                log_dir=job_dir,
            )
        finally:
            if args.docker_cleanup in {"after", "before-and-after"}:
                cleanup_job_harbor_docker(args=args, job_dir=job_dir, phase="after")
        if result.stdout:
            eprint("[tb2] harbor stdout tail:")
            eprint(tail(result.stdout))
        if result.stderr:
            eprint("[tb2] harbor stderr tail:")
            eprint(tail(result.stderr))

    records = parse_trial_records(job_dir)
    failure_summaries = collect_failure_summaries(job_dir)
    metrics = compute_metrics(
        records,
        result.returncode,
        failure_summaries=failure_summaries,
    )
    write_summary(
        args=args,
        command=command,
        result=result,
        job_dir=job_dir,
        tasks=tasks,
        records=records,
        metrics=metrics,
        failure_summaries=failure_summaries,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
