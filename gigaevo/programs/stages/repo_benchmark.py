from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import shlex
import shutil
import time
from typing import Any
import uuid

from loguru import logger

from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.repo_harness.artifacts import capture_evaluation_artifacts
from gigaevo.repo_harness.feedback import (
    DEFAULT_STRUCTURED_FEEDBACK_MARKERS,
    extract_structured_feedback,
)
from gigaevo.repo_harness.git_utils import add_worktree, ensure_git_repo, remove_worktree
from gigaevo.repo_harness.manifest import RepoCandidateManifest


@dataclass(frozen=True)
class _BenchmarkRun:
    label: str
    command: str | list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    metrics: dict[str, float]
    structured_feedback: dict[str, Any] | None
    evaluation_artifacts: dict[str, Any] | None


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _render_command(command: str | list[str], variables: dict[str, str]) -> str | list[str]:
    if isinstance(command, str):
        return command.format(**variables)
    return [str(part).format(**variables) for part in command]


def _build_benchmark_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if overrides is not None:
        env.update({str(k): str(v) for k, v in overrides.items()})
    # External benchmark commands do not need pytest runner metadata, and
    # inheriting it can confuse nested subprocess/test tooling.
    for key in tuple(env):
        if key.startswith("PYTEST_"):
            env.pop(key, None)
    return env


async def _terminate_benchmark_process(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> tuple[bytes, bytes]:
    """Give the benchmark wrapper a chance to clean up, then kill its group."""

    if proc.returncode is not None:
        return await proc.communicate()

    try:
        proc.terminate()
    except ProcessLookupError:
        return await proc.communicate()

    try:
        return await asyncio.wait_for(proc.communicate(), timeout=grace_seconds)
    except TimeoutError:
        pass

    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return await proc.communicate()


async def _run_benchmark_command(
    command: str | list[str],
    *,
    cwd: Path,
    timeout: float,
    shell: bool,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    if shell:
        cmd = command if isinstance(command, str) else " ".join(map(shlex.quote, command))
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    else:
        argv = shlex.split(command) if isinstance(command, str) else [str(x) for x in command]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        stdout_b, stderr_b = await _terminate_benchmark_process(
            proc,
            grace_seconds=90,
        )
        stderr_b += f"\nTimed out after {timeout}s".encode()
    except asyncio.CancelledError:
        await _terminate_benchmark_process(proc, grace_seconds=90)
        raise
    duration = time.monotonic() - started
    return (
        proc.returncode if proc.returncode is not None else 124,
        stdout_b.decode(errors="replace"),
        stderr_b.decode(errors="replace"),
        duration,
    )


def _extract_metrics_from_payload(payload: Any) -> dict[str, float]:
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]
    if not isinstance(payload, dict):
        raise ValueError("Benchmark metrics JSON must be an object or {metrics: object}")
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        else:
            metrics[str(key)] = float(value)
    return metrics


def _parse_stdout_json(stdout: str) -> dict[str, float]:
    stripped = stdout.strip()
    if not stripped:
        raise ValueError("Benchmark stdout is empty; cannot parse metrics JSON")
    try:
        return _extract_metrics_from_payload(json.loads(stripped))
    except json.JSONDecodeError:
        for line in reversed(stripped.splitlines()):
            try:
                return _extract_metrics_from_payload(json.loads(line))
            except json.JSONDecodeError:
                continue
    raise ValueError("Benchmark stdout did not contain a JSON metrics object")


@StageRegistry.register(
    description="Checkout a repo candidate commit and run an external benchmark"
)
class RepoBenchmarkStage(Stage):
    """Evaluate a Git-backed repo candidate and attach feedback for mutation."""

    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    def __init__(
        self,
        *,
        source_repo: str | Path,
        benchmark_command: str | list[str],
        storage: Any | None = None,
        worktree_root: str | Path = ".gigaevo/repo_eval/worktrees",
        metrics_path: str | Path | None = None,
        stdout_json: bool = True,
        shell: bool = False,
        benchmark_timeout: float | None = None,
        env: dict[str, str] | None = None,
        fallback_metrics: dict[str, float] | None = None,
        structured_feedback_path: str | Path | None = None,
        structured_feedback_markers: list[str] | tuple[str, ...] | None = None,
        artifact_root: str | Path | None = None,
        capture_artifacts: bool = True,
        max_artifact_files: int = 200,
        max_artifact_file_bytes: int = 50_000_000,
        max_artifact_total_bytes: int = 250_000_000,
        fail_on_nonzero: bool = False,
        keep_worktrees: bool = False,
        feedback_tail_chars: int = 12000,
        staged_validation_enabled: bool = False,
        staged_primary_metric: str = "fitness",
        staged_case_arg_names: list[str] | tuple[str, ...] | None = None,
        staged_case_joiner: str = ",",
        staged_case_failure_threshold: float = 0.999999,
        staged_min_improvement: float = 1.0e-9,
        staged_max_cases: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.source_repo = ensure_git_repo(source_repo)
        self.benchmark_command = benchmark_command
        self.storage = storage
        self.worktree_root = Path(worktree_root).expanduser().resolve()
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.stdout_json = stdout_json
        self.shell = shell
        self.benchmark_timeout = benchmark_timeout
        self.env = env
        self.fallback_metrics = fallback_metrics
        self.structured_feedback_path = (
            Path(structured_feedback_path) if structured_feedback_path else None
        )
        self.structured_feedback_markers = (
            DEFAULT_STRUCTURED_FEEDBACK_MARKERS
            if structured_feedback_markers is None
            else tuple(structured_feedback_markers)
        )
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve() if artifact_root else None
        )
        self.capture_artifacts = capture_artifacts
        self.max_artifact_files = max_artifact_files
        self.max_artifact_file_bytes = max_artifact_file_bytes
        self.max_artifact_total_bytes = max_artifact_total_bytes
        self.fail_on_nonzero = fail_on_nonzero
        self.keep_worktrees = keep_worktrees
        self.feedback_tail_chars = feedback_tail_chars
        self.staged_validation_enabled = staged_validation_enabled
        self.staged_primary_metric = staged_primary_metric
        self.staged_case_arg_names = tuple(
            staged_case_arg_names
            if staged_case_arg_names is not None
            else ("--tasks", "--task", "--cases", "--case")
        )
        self.staged_case_joiner = staged_case_joiner
        self.staged_case_failure_threshold = staged_case_failure_threshold
        self.staged_min_improvement = staged_min_improvement
        self.staged_max_cases = staged_max_cases

    async def compute(self, program: Program) -> StageIO:
        manifest = RepoCandidateManifest.from_program(
            program, default_repo_path=self.source_repo
        )
        repo = ensure_git_repo(manifest.repo_path)
        worktree = (
            self.worktree_root
            / f"{program.short_id}-{manifest.commit[:12]}-{uuid.uuid4().hex[:8]}"
        )
        if worktree.exists():
            shutil.rmtree(worktree)
        add_worktree(repo, worktree, manifest.commit, detach=True)

        try:
            manifest_path = worktree / ".gigaevo_candidate.json"
            manifest_path.write_text(manifest.to_program_code())
            variables = {
                "worktree": str(worktree),
                "manifest_path": str(manifest_path),
                "commit": manifest.commit,
                "entrypoint": manifest.entrypoint or "",
            }
            timeout = self.benchmark_timeout or self.timeout
            command_env = _build_benchmark_env(self.env)

            parent_feedback = await self._load_primary_parent_feedback(program)
            staged_plan = self._staged_case_plan(parent_feedback)
            staged_validation: dict[str, Any] | None = None
            run: _BenchmarkRun

            if staged_plan is not None:
                failed_cases = staged_plan["failed_cases"]
                preliminary = await self._execute_benchmark_run(
                    label="parent_failed_cases",
                    cases=failed_cases,
                    variables=variables,
                    worktree=worktree,
                    timeout=timeout,
                    env=command_env,
                    program=program,
                    manifest=manifest,
                )
                improved, comparison = self._staged_subset_improved(
                    parent_feedback=parent_feedback,
                    child_feedback=preliminary.structured_feedback,
                    child_metrics=preliminary.metrics,
                    cases=failed_cases,
                )
                staged_validation = {
                    "enabled": True,
                    "mode": "parent_failed_cases_first",
                    "parent_program_id": staged_plan.get("parent_program_id"),
                    "failed_cases": failed_cases,
                    "preliminary": self._run_summary(preliminary),
                    "comparison": comparison,
                    "promoted_to_full": improved,
                }
                if improved:
                    run = await self._execute_benchmark_run(
                        label="full",
                        cases=None,
                        variables=variables,
                        worktree=worktree,
                        timeout=timeout,
                        env=command_env,
                        program=program,
                        manifest=manifest,
                    )
                    staged_validation["full"] = self._run_summary(run)
                else:
                    run = preliminary
                    run = self._replace_run_metrics(
                        run,
                        self._project_metrics_after_subset(
                            parent_feedback=parent_feedback,
                            child_feedback=preliminary.structured_feedback,
                            child_metrics=preliminary.metrics,
                            cases=failed_cases,
                        ),
                    )
            else:
                run = await self._execute_benchmark_run(
                    label="full",
                    cases=None,
                    variables=variables,
                    worktree=worktree,
                    timeout=timeout,
                    env=command_env,
                    program=program,
                    manifest=manifest,
                )

            metrics = run.metrics
            feedback = self._build_feedback(
                manifest=manifest,
                worktree=worktree,
                run=run,
                staged_validation=staged_validation,
            )
            if run.evaluation_artifacts is not None:
                program.set_metadata(
                    "repo_evaluation_artifacts", run.evaluation_artifacts
                )
            program.set_metadata("repo_candidate", manifest.model_dump())
            program.set_metadata("repo_benchmark_feedback", feedback)
            program.add_metrics(metrics)
            logger.info(
                "[RepoBenchmarkStage] {} commit {} -> metrics {}",
                program.short_id,
                manifest.commit[:12],
                metrics,
            )
            return FloatDictContainer(data=metrics)
        finally:
            if not self.keep_worktrees and worktree.exists():
                try:
                    remove_worktree(repo, worktree)
                except Exception as exc:
                    logger.warning(
                        "[RepoBenchmarkStage] Failed to remove worktree {}: {}",
                        worktree,
                        exc,
                    )

    async def _execute_benchmark_run(
        self,
        *,
        label: str,
        cases: list[str] | None,
        variables: dict[str, str],
        worktree: Path,
        timeout: float,
        env: dict[str, str],
        program: Program,
        manifest: RepoCandidateManifest,
    ) -> _BenchmarkRun:
        command, constrained = self._render_benchmark_command(variables, cases)
        if cases is not None and not constrained:
            raise RuntimeError(
                "Staged validation requested a case subset, but benchmark_command "
                "does not expose a supported case filter. Add a selected_cases "
                "placeholder or configure staged_case_arg_names."
            )

        run_env = dict(env)
        run_env["GIGAEVO_BENCHMARK_PHASE"] = label
        if cases is not None:
            run_env["GIGAEVO_BENCHMARK_SELECTED_CASES"] = self.staged_case_joiner.join(
                cases
            )
            run_env["GIGAEVO_BENCHMARK_SELECTED_CASES_JSON"] = json.dumps(cases)

        code, stdout, stderr, duration = await _run_benchmark_command(
            command,
            cwd=worktree,
            timeout=timeout,
            shell=self.shell,
            env=run_env,
        )

        try:
            metrics = self._load_metrics(worktree=worktree, stdout=stdout)
        except Exception:
            if self.fallback_metrics is None:
                raise
            metrics = dict(self.fallback_metrics)
        if code != 0 and self.fail_on_nonzero:
            raise RuntimeError(
                f"Benchmark command failed with exit code {code}: {stderr[-1000:]}"
            )
        if not metrics:
            raise ValueError("Benchmark produced no metrics")

        structured_feedback = self._load_structured_feedback(
            worktree=worktree, stderr=stderr
        )
        evaluation_artifacts = self._capture_evaluation_artifacts(
            program=program,
            manifest=manifest,
            worktree=worktree,
            stdout=stdout,
            stderr=stderr,
            structured_feedback=structured_feedback,
            label=label,
        )
        return _BenchmarkRun(
            label=label,
            command=command,
            returncode=code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            metrics=metrics,
            structured_feedback=structured_feedback,
            evaluation_artifacts=evaluation_artifacts,
        )

    def _render_benchmark_command(
        self, variables: dict[str, str], cases: list[str] | None
    ) -> tuple[str | list[str], bool]:
        case_variables = self._case_variables(cases or [])
        command_variables = {**variables, **case_variables}
        has_case_placeholder = self._command_has_case_placeholder(
            self.benchmark_command
        )
        command = _render_command(self.benchmark_command, command_variables)
        if cases is None:
            return command, True
        command, replaced = self._replace_case_args(command, cases)
        return command, bool(has_case_placeholder or replaced)

    def _case_variables(self, cases: list[str]) -> dict[str, str]:
        joined = self.staged_case_joiner.join(cases)
        return {
            "selected_cases": joined,
            "selected_cases_csv": ",".join(cases),
            "selected_cases_json": json.dumps(cases),
            "selected_cases_args": " ".join(shlex.quote(case) for case in cases),
            "selected_case_count": str(len(cases)),
        }

    @staticmethod
    def _command_has_case_placeholder(command: str | list[str]) -> bool:
        text = command if isinstance(command, str) else "\n".join(map(str, command))
        return "{selected_cases" in text or "{selected_case_count" in text

    def _replace_case_args(
        self, command: str | list[str], cases: list[str]
    ) -> tuple[str | list[str], bool]:
        if not self.staged_case_arg_names:
            return command, False
        value = self.staged_case_joiner.join(cases)
        if isinstance(command, str):
            replaced = False
            rendered = command
            for flag in self.staged_case_arg_names:
                for pattern in (f"{flag}=", f"{flag} "):
                    index = rendered.find(pattern)
                    if index < 0:
                        continue
                    start = index + len(pattern)
                    end = rendered.find(" ", start)
                    if end < 0:
                        end = len(rendered)
                    separator = "=" if pattern.endswith("=") else " "
                    rendered = (
                        rendered[:index]
                        + flag
                        + separator
                        + shlex.quote(value)
                        + rendered[end:]
                    )
                    replaced = True
                    break
                if replaced:
                    break
            return rendered, replaced

        argv = [str(part) for part in command]
        replaced = False
        for index, part in enumerate(list(argv)):
            for flag in self.staged_case_arg_names:
                if part == flag:
                    if index + 1 < len(argv):
                        argv[index + 1] = value
                    else:
                        argv.append(value)
                    replaced = True
                    break
                if part.startswith(f"{flag}="):
                    argv[index] = f"{flag}={value}"
                    replaced = True
                    break
            if replaced:
                break
        return argv, replaced

    async def _load_primary_parent_feedback(
        self, program: Program
    ) -> dict[str, Any] | None:
        if not self.staged_validation_enabled or self.storage is None:
            return None
        parent_id = program.metadata.get("primary_parent_program_id")
        if not parent_id and program.lineage.parents:
            parent_id = program.lineage.parents[0]
        if not parent_id:
            return None
        try:
            parent = await self.storage.get(str(parent_id))
        except Exception as exc:
            logger.warning(
                "[RepoBenchmarkStage] Could not load parent {} for staged validation: {}",
                parent_id,
                exc,
            )
            return None
        if parent is None:
            return None
        feedback = parent.metadata.get("repo_benchmark_feedback")
        if not isinstance(feedback, dict):
            return None
        copied = dict(feedback)
        copied["_program_id"] = parent.id
        copied["_short_id"] = parent.short_id
        copied["_metrics"] = dict(parent.metrics)
        return copied

    def _staged_case_plan(
        self, parent_feedback: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not self.staged_validation_enabled or not parent_feedback:
            return None
        if not self._command_supports_case_filter():
            logger.warning(
                "[RepoBenchmarkStage] Staged validation enabled, but benchmark_command "
                "has no selected_cases placeholder or configured case flag; running full benchmark."
            )
            return None
        failed_cases = self._failed_case_ids(parent_feedback)
        if self.staged_max_cases is not None:
            failed_cases = failed_cases[: self.staged_max_cases]
        if not failed_cases:
            return None
        return {
            "parent_program_id": parent_feedback.get("_program_id"),
            "failed_cases": failed_cases,
        }

    def _command_supports_case_filter(self) -> bool:
        if self._command_has_case_placeholder(self.benchmark_command):
            return True
        text_parts = (
            [self.benchmark_command]
            if isinstance(self.benchmark_command, str)
            else [str(part) for part in self.benchmark_command]
        )
        for part in text_parts:
            for flag in self.staged_case_arg_names:
                if part == flag or part.startswith(f"{flag}="):
                    return True
        return False

    def _staged_subset_improved(
        self,
        *,
        parent_feedback: dict[str, Any],
        child_feedback: dict[str, Any] | None,
        child_metrics: dict[str, float],
        cases: list[str],
    ) -> tuple[bool, dict[str, Any]]:
        parent_scores = self._case_scores_with_defaults(parent_feedback)
        child_scores = self._case_scores_with_defaults(child_feedback)
        parent_avg = self._average_case_score(parent_scores, cases, default=0.0)
        child_avg = self._average_case_score(child_scores, cases, default=None)
        used_child_metric = False
        if child_avg is None:
            child_avg = child_metrics.get(self.staged_primary_metric, 0.0)
            used_child_metric = True
        delta = float(child_avg) - float(parent_avg)
        improved = delta > self.staged_min_improvement
        return improved, {
            "parent_subset_score": float(parent_avg),
            "child_subset_score": float(child_avg),
            "delta": delta,
            "min_improvement": self.staged_min_improvement,
            "used_child_metric_fallback": used_child_metric,
        }

    def _project_metrics_after_subset(
        self,
        *,
        parent_feedback: dict[str, Any],
        child_feedback: dict[str, Any] | None,
        child_metrics: dict[str, float],
        cases: list[str],
    ) -> dict[str, float]:
        projected = dict(child_metrics)
        parent_scores = self._case_scores_with_defaults(parent_feedback)
        child_scores = self._case_scores_with_defaults(child_feedback)
        if parent_scores and child_scores:
            combined = dict(parent_scores)
            for case in cases:
                if case in child_scores:
                    combined[case] = child_scores[case]
            if combined:
                projected[self.staged_primary_metric] = round(
                    sum(combined.values()) / len(combined),
                    6,
                )
                return projected

        parent_metrics = parent_feedback.get("metrics")
        if not isinstance(parent_metrics, dict):
            parent_metrics = parent_feedback.get("_metrics")
        if (
            isinstance(parent_metrics, dict)
            and self.staged_primary_metric in parent_metrics
        ):
            projected[self.staged_primary_metric] = float(
                parent_metrics[self.staged_primary_metric]
            )
        return projected

    @staticmethod
    def _replace_run_metrics(
        run: _BenchmarkRun, metrics: dict[str, float]
    ) -> _BenchmarkRun:
        return _BenchmarkRun(
            label=run.label,
            command=run.command,
            returncode=run.returncode,
            stdout=run.stdout,
            stderr=run.stderr,
            duration_seconds=run.duration_seconds,
            metrics=metrics,
            structured_feedback=run.structured_feedback,
            evaluation_artifacts=run.evaluation_artifacts,
        )

    def _build_feedback(
        self,
        *,
        manifest: RepoCandidateManifest,
        worktree: Path,
        run: _BenchmarkRun,
        staged_validation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        feedback: dict[str, Any] = {
            "commit": manifest.commit,
            "branch": manifest.branch,
            "entrypoint": manifest.entrypoint,
            "command": run.command,
            "returncode": run.returncode,
            "duration_seconds": run.duration_seconds,
            "metrics": run.metrics,
            "stdout_tail": _tail(run.stdout, self.feedback_tail_chars),
            "stderr_tail": _tail(run.stderr, self.feedback_tail_chars),
            "worktree": str(worktree),
            "benchmark_run_label": run.label,
        }
        if run.structured_feedback is not None:
            feedback["structured_feedback"] = run.structured_feedback
            feedback["structured_failure_feedback"] = run.structured_feedback
        if run.evaluation_artifacts is not None:
            feedback["evaluation_artifacts"] = run.evaluation_artifacts
        if staged_validation is not None:
            feedback["staged_validation"] = staged_validation
        return feedback

    @staticmethod
    def _run_summary(run: _BenchmarkRun) -> dict[str, Any]:
        return {
            "label": run.label,
            "command": run.command,
            "returncode": run.returncode,
            "duration_seconds": run.duration_seconds,
            "metrics": run.metrics,
            "structured_feedback_summary": (
                run.structured_feedback.get("summary")
                if isinstance(run.structured_feedback, dict)
                else None
            ),
            "artifact_manifest": (
                run.evaluation_artifacts.get("manifest_path")
                if isinstance(run.evaluation_artifacts, dict)
                else None
            ),
        }

    def _structured_feedback(
        self, feedback: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not isinstance(feedback, dict):
            return None
        structured = feedback.get("structured_feedback") or feedback.get(
            "structured_failure_feedback"
        )
        if isinstance(structured, dict):
            return structured
        if (
            "summary" in feedback
            or "cases" in feedback
            or "failure_clusters" in feedback
        ):
            return feedback
        return None

    def _case_scores_with_defaults(
        self, feedback: dict[str, Any] | None
    ) -> dict[str, float]:
        structured = self._structured_feedback(feedback)
        scores = self._case_scores(structured)
        selected_cases = self._selected_case_ids(feedback)
        failed_cases = set(self._failed_case_ids(feedback))
        if selected_cases and failed_cases:
            for case in selected_cases:
                scores.setdefault(
                    case,
                    0.0 if case in failed_cases else 1.0,
                )
        return scores

    def _case_scores(self, structured: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(structured, dict):
            return {}
        values: dict[str, list[float]] = {}
        for key in ("cases", "case_results", "trials", "examples"):
            items = structured.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                case_id = self._case_id(item)
                score = self._case_score(item)
                if case_id is None or score is None:
                    continue
                values.setdefault(case_id, []).append(score)
        return {
            case_id: sum(case_values) / len(case_values)
            for case_id, case_values in values.items()
            if case_values
        }

    def _selected_case_ids(self, feedback: dict[str, Any] | None) -> list[str]:
        structured = self._structured_feedback(feedback)
        candidates: list[Any] = []
        if isinstance(structured, dict):
            summary = structured.get("summary")
            if isinstance(summary, dict):
                candidates.extend(
                    [
                        summary.get("selected_cases"),
                        summary.get("selected_tasks"),
                        summary.get("cases"),
                        summary.get("tasks"),
                    ]
                )
            candidates.extend(
                [
                    structured.get("selected_cases"),
                    structured.get("selected_tasks"),
                    structured.get("tasks"),
                ]
            )
        if isinstance(feedback, dict):
            candidates.extend(
                [
                    feedback.get("selected_cases"),
                    feedback.get("selected_tasks"),
                ]
            )
        for candidate in candidates:
            values = self._string_list(candidate)
            if values:
                return values
        return []

    def _failed_case_ids(self, feedback: dict[str, Any] | None) -> list[str]:
        structured = self._structured_feedback(feedback)
        scores = self._case_scores_with_defaults_without_failures(feedback)
        failed = [
            case
            for case, score in scores.items()
            if score < self.staged_case_failure_threshold
        ]
        if failed:
            return self._unique_strings(failed)

        failed_candidates: list[str] = []
        if isinstance(structured, dict):
            clusters = structured.get("failure_clusters")
            if isinstance(clusters, list):
                for cluster in clusters:
                    if isinstance(cluster, dict):
                        name = cluster.get("name") or cluster.get("case_id")
                        if name is not None:
                            failed_candidates.append(str(name))
            examples = structured.get("examples")
            if isinstance(examples, list):
                for item in examples:
                    if isinstance(item, dict):
                        case_id = self._case_id(item)
                        if case_id is not None:
                            failed_candidates.append(case_id)
        return self._unique_strings(failed_candidates)

    def _case_scores_with_defaults_without_failures(
        self, feedback: dict[str, Any] | None
    ) -> dict[str, float]:
        structured = self._structured_feedback(feedback)
        return self._case_scores(structured)

    @staticmethod
    def _case_id(item: dict[str, Any]) -> str | None:
        for key in ("case_id", "task", "task_id", "name", "id"):
            value = item.get(key)
            if value is not None and str(value):
                return str(value)
        trial = item.get("trial")
        if trial is not None and str(trial):
            return str(trial).rsplit("__", 1)[0]
        return None

    @staticmethod
    def _case_score(item: dict[str, Any]) -> float | None:
        for key in ("score", "reward", "pass_rate", "success_rate"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        for key in ("passed", "success", "succeeded"):
            value = item.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
        status = str(item.get("status") or "").lower()
        if status in {"passed", "pass", "success", "succeeded"}:
            return 1.0
        if status in {"failed", "fail", "failure", "error"}:
            return 0.0
        return None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [
                part.strip()
                for part in value.replace(",", " ").split()
                if part.strip()
            ]
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        return []

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _average_case_score(
        scores: dict[str, float], cases: list[str], *, default: float | None
    ) -> float | None:
        values = [scores[case] for case in cases if case in scores]
        if not values:
            return default
        return sum(values) / len(values)

    def _load_metrics(self, *, worktree: Path, stdout: str) -> dict[str, float]:
        if self.metrics_path is not None:
            path = self.metrics_path
            if not path.is_absolute():
                path = worktree / path
            if path.exists():
                return _extract_metrics_from_payload(json.loads(path.read_text()))
        if self.stdout_json:
            return _parse_stdout_json(stdout)
        return {}

    def _load_structured_feedback(
        self, *, worktree: Path, stderr: str
    ) -> dict[str, Any] | None:
        file_feedback = self._load_structured_feedback_file(worktree)
        if file_feedback is not None:
            return file_feedback
        return extract_structured_feedback(stderr, self.structured_feedback_markers)

    def _load_structured_feedback_file(self, worktree: Path) -> dict[str, Any] | None:
        if self.structured_feedback_path is None:
            return None
        path = self.structured_feedback_path
        if not path.is_absolute():
            path = worktree / path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning(
                "[RepoBenchmarkStage] Could not parse structured feedback file {}: {}",
                path,
                exc,
            )
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "[RepoBenchmarkStage] Structured feedback file {} did not contain an object",
                path,
            )
            return None
        return payload

    def _capture_evaluation_artifacts(
        self,
        *,
        program: Program,
        manifest: RepoCandidateManifest,
        worktree: Path,
        stdout: str,
        stderr: str,
        structured_feedback: dict[str, Any] | None,
        label: str = "full",
    ) -> dict[str, Any] | None:
        if not self.capture_artifacts or self.artifact_root is None:
            return None
        safe_label = "".join(
            ch if ch.isalnum() or ch in "._-" else "-" for ch in label
        ).strip("-") or "run"
        artifact_dir = (
            self.artifact_root
            / f"{program.short_id}-{safe_label}-{manifest.commit[:12]}-{uuid.uuid4().hex[:8]}"
        )
        return capture_evaluation_artifacts(
            artifact_root=artifact_dir,
            structured_feedback=structured_feedback,
            base_dirs=[worktree],
            stdout=stdout,
            stderr=stderr,
            max_files=self.max_artifact_files,
            max_file_bytes=self.max_artifact_file_bytes,
            max_total_bytes=self.max_artifact_total_bytes,
        )
