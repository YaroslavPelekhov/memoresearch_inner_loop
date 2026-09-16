from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any
import uuid

from loguru import logger

from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.exceptions import MutationError
from gigaevo.llm.codex_usage import (
    append_usage_record,
    infer_usage_ledger_from_mutation_log_root,
    with_usage_context,
)
from gigaevo.programs.program import Program
from gigaevo.repo_harness.backends import CodingAgentBackend
from gigaevo.repo_harness.git_utils import (
    add_worktree,
    changed_files,
    commit_all,
    diff_stat,
    ensure_git_repo,
    remove_worktree,
    rev_parse,
    run_git,
    status_porcelain,
)
from gigaevo.repo_harness.manifest import RepoCandidateManifest

DEFAULT_MUTATION_PROMPT = """You are the mutation agent inside a GigaEvo loop.

You are editing an evolving candidate repository. The repository may be an
agent harness, a benchmark submission, a solver, a circuit implementation, or
another Git-backed artifact depending on the task. Read the repository, read
the selected-parent mutation brief, synthesize compatible ideas from the
parents, and leave the working tree ready to commit.

Mutation brief:
{brief_path}

Rules:
- When the brief contains a FIXED IDEA CONTRACT, that contract overrides the
  general mutation freedom below. You may repair or redesign its implementation,
  but you must not replace, omit, or drift away from its required mechanism.
- Treat the Task Description section in the brief as authoritative for task
  intent, constraints, allowed inputs, and required output format.
- Use benchmark feedback from the brief as the main source of truth for observed
  failures and remaining improvement opportunities.
- If benchmark feedback or reflection includes evaluation artifact paths, inspect
  the relevant logs, verifier output, trajectories, or result files before
  changing code when the failure mechanism is unclear.
- The worktree starts from the primary parent. Treat secondary parents as source
  material: inspect their scores, reflections, and diffs from the primary parent,
  then merge the best compatible mechanisms.
- You may make a larger architectural change when it tests a clear hypothesis
  about a recurring failure mode. Examples include redesigning the agent loop,
  adding planner/executor/verifier phases, introducing task-state tracking,
  changing recovery or summarization policy, replacing brittle parser behavior,
  or refactoring session handling into clearer components.
- The selected parents are evidence, not a boundary. You may discard parent
  mechanisms, introduce new files, move responsibilities between modules, or
  replace a subsystem when the resulting design is coherent and benchmark
  feedback justifies the experiment.
- Prefer one or two deliberate design experiments over many small knob changes.
  If the parent already has repeated local patches around the same failure
  cluster, consider a different architecture rather than another patch.
- Return the phenotype produced by the mutation's central experiment even when
  a cheap local evaluation scores below the primary parent. Do not restore the
  incumbent output, add an "accept only if better" score guard, or otherwise
  hide a valid regression. Candidate comparison and archive selection belong to
  the outer evolution loop. Remove or bypass inherited score-only guards on the
  execution path exercised by your experiment. Keep fallbacks only for crashes,
  invalid outputs, or violations of hard task constraints.
- Do not hardcode benchmark answers, task IDs, hidden tests, verifier outputs,
  generated scores, or evaluator quirks.
- Do not change the benchmark, evaluator, metric parser, or sandbox policy unless
  the brief explicitly allows it.
- If the task is an agent-harness task and the candidate repository contains
  runtime or agent configuration files, treat them as evolvable harness code.
  Tune them freely when they are part of a larger behavioral hypothesis, but do
  not rely only on config churn unless the evidence points to configuration.
- Keep changes reviewable, but do not avoid multi-file refactors when they make
  the experimental design easier to understand and test.
- Run the exact Required Executable Smoke Test from the brief before finishing.
  A pass requires the structured result to report is_valid == 1. If it fails,
  diagnose the logs, repair the implementation, and rerun it until it passes.
- In your final response, give a compact fidelity checklist mapping each fixed
  idea requirement to the code that implements it, followed by the smoke result.
"""


def _slug(text: str, *, max_len: int = 72) -> str:
    slug = re.sub(r"[^A-Za-z0-9._/-]+", "-", text).strip("-")
    return slug[:max_len] or "candidate"


def _json_safe(obj: Any, *, max_chars: int = 20000) -> str:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def _clip_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _compact_selection_metadata(metadata: Any) -> dict[str, Any]:
    """Keep selector judgment without repeating IDs or LLM accounting data."""

    if not isinstance(metadata, dict):
        return {}
    duplicated_or_operational = {
        "codex_usage",
        "created_at",
        "selected_parent_ids",
        "selected_parent_short_ids",
    }
    return {
        key: value
        for key, value in metadata.items()
        if key not in duplicated_or_operational
    }


def _compact_benchmark_feedback(feedback: Any, *, max_chars: int) -> Any:
    """Retain diagnostic evidence while removing canonical metric/artifact copies."""

    if not isinstance(feedback, dict) or max_chars <= 0:
        return None

    compact: dict[str, Any] = {
        key: feedback[key]
        for key in ("benchmark_run_label", "duration_seconds", "returncode")
        if key in feedback
    }
    structured = feedback.get("structured_feedback")
    if not isinstance(structured, dict):
        structured = feedback.get("structured_failure_feedback")

    if isinstance(structured, dict):
        # Parent summaries already contain the canonical metrics and actionable
        # artifact paths. The two structured-feedback keys are commonly exact
        # aliases, so expose one diagnostic copy only.
        diagnostics = {
            key: value
            for key, value in structured.items()
            if key not in {"aggregate_metrics", "artifacts", "metrics"}
        }
        if diagnostics:
            compact["structured_feedback"] = diagnostics
    else:
        diagnostics = {
            key: value
            for key, value in feedback.items()
            if key
            not in {
                "benchmark_run_label",
                "branch",
                "command",
                "commit",
                "duration_seconds",
                "entrypoint",
                "evaluation_artifacts",
                "metrics",
                "returncode",
                "stderr_tail",
                "stdout_tail",
                "structured_failure_feedback",
                "structured_feedback",
                "worktree",
            }
        }
        if diagnostics:
            compact["diagnostics"] = diagnostics

    stderr_tail = feedback.get("stderr_tail")
    if isinstance(stderr_tail, str) and stderr_tail.strip():
        compact["stderr_tail"] = _clip_text(stderr_tail, max_chars=min(4000, max_chars))

    # Structured feedback already explains successful benchmark stdout, which
    # generally repeats the same metric JSON. Keep stdout for unstructured
    # benchmarks, where it may be the only available evidence.
    stdout_tail = feedback.get("stdout_tail")
    if (
        not isinstance(structured, dict)
        and isinstance(stdout_tail, str)
        and stdout_tail.strip()
    ):
        compact["stdout_tail"] = _clip_text(stdout_tail, max_chars=min(4000, max_chars))

    if not compact:
        return None
    serialized = json.dumps(compact, indent=2, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return compact
    return {
        "truncated": True,
        "original_chars": len(serialized),
        "excerpt": _clip_text(serialized, max_chars=max_chars),
    }


def _compact_evaluation_artifacts(artifacts: Any) -> Any:
    """Keep paths the agent can inspect without storage bookkeeping copies."""

    if not isinstance(artifacts, dict):
        return None
    items = artifacts.get("items")
    if not isinstance(items, list):
        return None
    useful_keys = (
        "kind",
        "name",
        "role",
        "description",
        "captured",
        "captured_path",
        "source_path",
    )
    compact_items = [
        {key: item[key] for key in useful_keys if key in item}
        for item in items
        if isinstance(item, dict)
    ]
    return {"items": [item for item in compact_items if item]} or None


async def _run_command(
    command: str | list[str],
    *,
    cwd: Path,
    timeout: float,
    shell: bool = False,
) -> tuple[int, str, str]:
    import asyncio

    if shell:
        cmd = (
            command if isinstance(command, str) else " ".join(map(shlex.quote, command))
        )
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        argv = (
            shlex.split(command)
            if isinstance(command, str)
            else [str(x) for x in command]
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        out_b, err_b = await proc.communicate()
        err_b += f"\nTimed out after {timeout}s".encode()
    return (
        proc.returncode if proc.returncode is not None else 124,
        out_b.decode(errors="replace"),
        err_b.decode(errors="replace"),
    )


class RepoHarnessMutationOperator(MutationOperator):
    """Mutation operator that lets a coding-agent backend edit a whole Git repo.

    Parent and child candidates are Git commits. GigaEvo stores only a manifest
    in Program.code plus structured metadata for lineage, feedback, and logs.
    """

    def __init__(
        self,
        *,
        source_repo: str | Path,
        backend: CodingAgentBackend,
        worktree_root: str | Path = ".gigaevo/repo_mutation/worktrees",
        log_root: str | Path = ".gigaevo/repo_mutation/logs",
        usage_log_path: str | Path | None = None,
        entrypoint: str | None = None,
        branch_prefix: str = "gigaevo/candidate",
        prompt_template: str = DEFAULT_MUTATION_PROMPT,
        prompt_template_path: str | Path | None = None,
        problem_context: Any | None = None,
        timeout: float = 2400.0,
        smoke_commands: list[str | list[str]] | None = None,
        smoke_timeout: float = 300.0,
        smoke_shell: bool = False,
        allow_no_changes: bool = False,
        keep_worktrees: bool = True,
        max_feedback_chars: int = 30000,
        max_parent_diff_chars: int = 20000,
        max_parent_reflection_chars: int = 6000,
        allowed_changed_files: list[str] | None = None,
        active_idea_path: str | Path | None = None,
        recent_failure_path: str | Path | None = None,
        **ignored_kwargs: Any,
    ):
        if ignored_kwargs:
            logger.debug(
                "[RepoHarnessMutationOperator] Ignoring inherited mutation config keys: {}",
                sorted(ignored_kwargs),
            )
        self.source_repo = ensure_git_repo(source_repo)
        self.backend = backend
        self.worktree_root = Path(worktree_root).expanduser().resolve()
        self.log_root = Path(log_root).expanduser().resolve()
        self.usage_log_path = (
            Path(usage_log_path).expanduser().resolve()
            if usage_log_path
            else infer_usage_ledger_from_mutation_log_root(self.log_root)
        )
        self.entrypoint = entrypoint
        self.problem_task_description = self._load_problem_task_description(
            problem_context
        )
        self.branch_prefix = branch_prefix.rstrip("/")
        self.timeout = timeout
        self.smoke_commands = smoke_commands or []
        self.smoke_timeout = smoke_timeout
        self.smoke_shell = smoke_shell
        self.allow_no_changes = allow_no_changes
        self.keep_worktrees = keep_worktrees
        self.max_feedback_chars = max_feedback_chars
        self.max_parent_diff_chars = max_parent_diff_chars
        self.max_parent_reflection_chars = max_parent_reflection_chars
        self.allowed_changed_files = (
            frozenset(str(item) for item in allowed_changed_files)
            if allowed_changed_files is not None
            else None
        )
        self.active_idea_path = (
            Path(active_idea_path).expanduser().resolve()
            if active_idea_path is not None
            else None
        )
        self.recent_failure_path = (
            Path(recent_failure_path).expanduser().resolve()
            if recent_failure_path is not None
            else None
        )
        if prompt_template_path:
            self.prompt_template = Path(prompt_template_path).read_text()
        else:
            self.prompt_template = prompt_template

    def _load_problem_task_description(self, problem_context: Any | None) -> str | None:
        if problem_context is None:
            return None
        try:
            task_description = getattr(problem_context, "task_description", None)
        except Exception as exc:
            logger.warning(
                "[RepoHarnessMutationOperator] Could not load problem task description: {}",
                exc,
            )
            return None
        if not task_description:
            return None
        return str(task_description).strip()

    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        if not selected_parents:
            logger.warning("[RepoHarnessMutationOperator] No parents provided")
            return None

        selection_metadata = getattr(selected_parents, "selection_metadata", None)
        parent = selected_parents[0]
        parent_manifests = [self._manifest_for_parent(p) for p in selected_parents]
        parent_manifest = parent_manifests[0]

        repo = ensure_git_repo(parent_manifest.repo_path)
        parent_commit = rev_parse(repo, parent_manifest.commit)
        candidate_id = uuid.uuid4().hex[:12]
        branch = f"{self.branch_prefix}/{parent.short_id}-{candidate_id}"
        worktree = self.worktree_root / _slug(branch.replace("/", "-"))
        log_dir = self.log_root / candidate_id

        logger.info(
            "[RepoHarnessMutationOperator] Mutating parent set {} primary={} commit {} via {}",
            [p.short_id for p in selected_parents],
            parent.short_id,
            parent_commit[:12],
            getattr(self.backend, "name", type(self.backend).__name__),
        )

        if worktree.exists():
            shutil.rmtree(worktree)
        add_worktree(repo, worktree, parent_commit, branch=branch)

        usage_record: dict[str, Any] | None = None
        try:
            approved_idea = self._load_approved_idea()
            brief_path = self._write_brief(
                selected_parents=selected_parents,
                parent_manifests=parent_manifests,
                primary_parent=parent,
                primary_manifest=parent_manifest,
                worktree=worktree,
                log_dir=log_dir,
                selection_metadata=selection_metadata,
                memory_instructions=memory_instructions,
                approved_idea=approved_idea,
            )
            prompt = self.prompt_template.format(
                brief_path=str(brief_path),
                worktree=str(worktree),
                parent_commit=parent_commit,
                branch=branch,
            )
            run = await self.backend.run(
                prompt=prompt,
                cwd=worktree,
                log_dir=log_dir / "agent",
                timeout=self.timeout,
                variables={
                    "brief_path": str(brief_path),
                    "worktree": str(worktree),
                    "parent_commit": parent_commit,
                    "branch": branch,
                },
            )
            usage_record = with_usage_context(
                run.usage,
                source="mutation",
                generation=max(p.generation for p in selected_parents) + 1,
                candidate_id=candidate_id,
                primary_parent_program_id=parent.id,
                selected_parent_program_ids=[p.id for p in selected_parents],
            )
            if run.exit_code != 0:
                raise MutationError(
                    f"{run.backend_name} failed with exit code {run.exit_code}: "
                    f"{run.stderr[-1000:]}"
                )

            await self._run_smoke_commands(worktree=worktree, log_dir=log_dir)

            if not status_porcelain(worktree).strip():
                current = rev_parse(worktree)
                if current == parent_commit and not self.allow_no_changes:
                    logger.info(
                        "[RepoHarnessMutationOperator] Backend produced no changes"
                    )
                    return None
            child_commit = commit_all(
                worktree,
                message=f"gigaevo mutation {candidate_id} from {parent.short_id}",
                allow_empty=self.allow_no_changes,
            )
            if usage_record:
                usage_record["commit"] = child_commit
            files = changed_files(repo, parent_commit, child_commit)
            if not files and not self.allow_no_changes:
                logger.info("[RepoHarnessMutationOperator] Commit has no diff")
                return None
            if self.allowed_changed_files is not None:
                disallowed = sorted(set(files) - self.allowed_changed_files)
                if disallowed:
                    raise MutationError(
                        "Candidate changed files outside the allowlist: "
                        + ", ".join(disallowed)
                    )

            manifest = RepoCandidateManifest(
                repo_path=str(repo),
                commit=child_commit,
                parent_commit=parent_commit,
                branch=branch,
                entrypoint=parent_manifest.entrypoint or self.entrypoint,
                mutation_agent=run.backend_name,
                mutation_session_log=run.log_path,
                changed_files=files,
                extra={
                    "worktree": str(worktree),
                    "diff_stat": diff_stat(repo, parent_commit, child_commit),
                    "primary_parent_program_id": parent.id,
                    "selected_parent_program_ids": [p.id for p in selected_parents],
                    "selected_parent_commits": [
                        manifest.commit for manifest in parent_manifests
                    ],
                    "parent_selection": selection_metadata,
                    "codex_usage": usage_record,
                },
            )
            metadata: dict[str, Any] = {
                "repo_candidate": manifest.model_dump(),
                "git_commit": child_commit,
                "parent_commit": parent_commit,
                "parent_commits": [manifest.commit for manifest in parent_manifests],
                "primary_parent_program_id": parent.id,
                "selected_parent_program_ids": [p.id for p in selected_parents],
                "git_branch": branch,
                "repo_path": str(repo),
                "worktree_path": str(worktree),
                "changed_files": files,
                "mutation_agent": run.backend_name,
                "mutation_session_log": run.log_path,
                "mutation_duration_seconds": run.duration_seconds,
                "codex_usage": usage_record,
                "parent_selection": selection_metadata,
                MutationSpec.META_MODEL: run.backend_name,
                MutationSpec.META_OUTPUT: {
                    "archetype": (
                        "approved_research_idea"
                        if approved_idea is not None
                        else "repo_coding_agent_edit"
                    ),
                    "justification": (
                        approved_idea.get("hypothesis")
                        if approved_idea is not None
                        else "Repository-level edit generated from selected parent feedback."
                    ),
                    "changes": (
                        [
                            {
                                "description": approved_idea["proposed_change"],
                                "motivation": approved_idea["expected_mechanism"],
                            }
                        ]
                        if approved_idea is not None
                        else []
                    ),
                    "insights_used": (
                        approved_idea.get("memory_ids", [])
                        if approved_idea is not None
                        else []
                    ),
                    "approved_idea_id": (
                        approved_idea.get("id") if approved_idea is not None else None
                    ),
                    "changed_files": files,
                    "selected_parent_count": len(selected_parents),
                    "parent_selection": selection_metadata,
                },
            }
            return MutationSpec(
                code=manifest.to_program_code(),
                parents=selected_parents,
                name=f"Repo mutation: {run.backend_name} {child_commit[:12]}",
                metadata=metadata,
            )
        except Exception as exc:
            raise MutationError(f"Failed repo harness mutation: {exc}") from exc
        finally:
            append_usage_record(self.usage_log_path, usage_record)
            if not self.keep_worktrees and worktree.exists():
                try:
                    remove_worktree(repo, worktree)
                except Exception as exc:
                    logger.warning(
                        "[RepoHarnessMutationOperator] Failed to remove worktree {}: {}",
                        worktree,
                        exc,
                    )

    def _write_brief(
        self,
        *,
        selected_parents: list[Program],
        parent_manifests: list[RepoCandidateManifest],
        primary_parent: Program,
        primary_manifest: RepoCandidateManifest,
        worktree: Path,
        log_dir: Path,
        selection_metadata: dict[str, Any] | None,
        memory_instructions: str | None,
        approved_idea: dict[str, Any] | None,
    ) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        primary_repo = ensure_git_repo(primary_manifest.repo_path)
        primary_commit = rev_parse(primary_repo, primary_manifest.commit)
        parent_summaries = self._selected_parent_summaries(
            parents=selected_parents,
            manifests=parent_manifests,
            primary_repo=primary_repo,
            primary_commit=primary_commit,
        )
        compact_selection = _compact_selection_metadata(selection_metadata)
        feedback = {
            # Put unique benchmark evidence before parent diffs so bounding the
            # payload cannot discard it in favor of repeated provenance.
            "benchmark_feedback_for_primary_parent": _compact_benchmark_feedback(
                primary_parent.metadata.get("repo_benchmark_feedback"),
                max_chars=min(20000, max(4000, self.max_feedback_chars // 3)),
            ),
            "entrypoint": primary_manifest.entrypoint or self.entrypoint,
            "parents": parent_summaries,
            "memory_instructions": memory_instructions,
            "approved_research_idea": approved_idea,
        }
        feedback = {key: value for key, value in feedback.items() if value is not None}
        task_description = _clip_text(
            self.problem_task_description or "No task description provided.",
            max_chars=20000,
        )
        parent_set_lines = []
        for idx, (parent, manifest) in enumerate(
            zip(selected_parents, parent_manifests)
        ):
            role = "Primary worktree base" if idx == 0 else f"Secondary source {idx}"
            commit = rev_parse(ensure_git_repo(manifest.repo_path), manifest.commit)
            parent_set_lines.append(f"- {role}: `{parent.short_id}` at `{commit}`")
        parent_set_text = "\n".join(parent_set_lines)
        approved_idea_section = ""
        if approved_idea is not None:
            approved_idea_section = (
                "# FIXED IDEA CONTRACT — MUST NOT BE REPLACED\n\n"
                "Implement every requirement in this reviewed idea. You may repair "
                "or refine implementation details, but you must not substitute a "
                "different mechanism or research idea. This contract overrides any "
                "generic mutation freedom elsewhere in the prompt.\n\n"
                f"```json\n{_json_safe(approved_idea, max_chars=12000)}\n```\n\n"
            )
        smoke_section = self._smoke_test_section(worktree)
        recent_failure_section = self._recent_failure_section()
        brief = (
            "# GigaEvo Repo Mutation Brief\n\n"
            f"Repository worktree: `{worktree}`\n\n"
            f"{approved_idea_section}"
            f"{smoke_section}"
            f"{recent_failure_section}"
            "## Task Description\n\n"
            f"{task_description}\n\n"
            "## Selected Parent Set\n\n"
            f"{parent_set_text}\n"
            f"- Parent count: {len(selected_parents)}\n\n"
            "## Parent Selection\n\n"
            f"```json\n{_json_safe(compact_selection, max_chars=12000)}\n```\n\n"
            "## Parent Feedback\n\n"
            f"```json\n{_json_safe(feedback, max_chars=self.max_feedback_chars)}\n```\n\n"
        )
        path = log_dir / "mutation_brief.md"
        path.write_text(brief)
        return path

    def _smoke_test_section(self, worktree: Path) -> str:
        if not self.smoke_commands:
            return ""
        commands: list[str] = []
        for command in self.smoke_commands:
            if isinstance(command, str):
                rendered = command.replace("{worktree}", str(worktree))
            else:
                rendered = shlex.join(
                    str(part).replace("{worktree}", str(worktree)) for part in command
                )
            commands.append(f"```bash\n{rendered}\n```")
        return (
            "## Required Executable Smoke Test\n\n"
            "Run the following command after your final edit. A process exit code "
            "of zero and a structured `is_valid == 1` result are both required. "
            "If it fails, fix the implementation and rerun it before finishing.\n\n"
            + "\n\n".join(commands)
            + "\n\n"
        )

    def _recent_failure_section(self) -> str:
        if self.recent_failure_path is None or not self.recent_failure_path.is_file():
            return ""
        try:
            state = json.loads(self.recent_failure_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ""
        failure = state.get("last_failure") if isinstance(state, dict) else None
        if not isinstance(failure, dict):
            return ""
        return (
            "## Most Recent Incomplete Experiment\n\n"
            "This attempt failed technically before completing the required horizon. "
            "It is not scientific evidence against the fixed idea. The current "
            "worktree remains based on the last confirmed parent; inspect the failed "
            "commit if useful and reimplement the same idea safely.\n\n"
            f"```json\n{_json_safe(failure, max_chars=5000)}\n```\n\n"
        )

    def _load_approved_idea(self) -> dict[str, Any] | None:
        if self.active_idea_path is None:
            return None
        try:
            value = json.loads(self.active_idea_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MutationError(
                f"No reviewed active idea at {self.active_idea_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise MutationError(
                f"Active idea is invalid JSON: {self.active_idea_path}"
            ) from exc
        if not isinstance(value, dict):
            raise MutationError("Active idea must be a JSON object")
        if value.get("status") not in {"approved", "redacted"}:
            raise MutationError(
                f"Active idea is not approved: status={value.get('status')!r}"
            )
        for key in ("id", "hypothesis", "proposed_change", "expected_mechanism"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise MutationError(f"Active idea has invalid field {key!r}")
        return value

    def _manifest_for_parent(self, parent: Program) -> RepoCandidateManifest:
        try:
            return RepoCandidateManifest.from_program(
                parent, default_repo_path=self.source_repo
            )
        except ValueError:
            return RepoCandidateManifest.from_metadata(
                parent, default_repo_path=self.source_repo
            )

    def _selected_parent_summaries(
        self,
        *,
        parents: list[Program],
        manifests: list[RepoCandidateManifest],
        primary_repo: Path,
        primary_commit: str,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for idx, (parent, manifest) in enumerate(zip(parents, manifests)):
            repo = ensure_git_repo(manifest.repo_path)
            commit = rev_parse(repo, manifest.commit)
            repo_reflection = parent.metadata.get("repo_reflection")
            benchmark_feedback = parent.metadata.get("repo_benchmark_feedback")
            evaluation_artifacts = parent.metadata.get("repo_evaluation_artifacts")
            if evaluation_artifacts is None and isinstance(benchmark_feedback, dict):
                evaluation_artifacts = benchmark_feedback.get("evaluation_artifacts")
            own_parent_commit = manifest.parent_commit
            if isinstance(repo_reflection, dict) and repo_reflection.get(
                "parent_commit"
            ):
                own_parent_commit = str(repo_reflection["parent_commit"])

            summary: dict[str, Any] = {
                "role": "primary_base" if idx == 0 else "secondary_source",
                "short_id": parent.short_id,
                "metrics": parent.metrics,
                "lineage": {
                    "parents": [
                        program_id[:8] for program_id in parent.lineage.parents
                    ],
                    "generation": parent.lineage.generation,
                },
                "changed_files": manifest.changed_files
                or self._reflection_changed_files(repo_reflection),
                "repo_reflection": self._compact_reflection(repo_reflection),
                "evaluation_artifacts": _compact_evaluation_artifacts(
                    evaluation_artifacts
                ),
            }
            stage_errors = parent.format_errors(include_traceback=False)
            if stage_errors and stage_errors != "<No stage errors found>":
                summary["stage_errors"] = stage_errors
            if own_parent_commit:
                summary["own_improvement_diff"] = self._diff_context(
                    repo=repo,
                    base=own_parent_commit,
                    head=commit,
                    max_chars=self.max_parent_diff_chars,
                )
            if idx != 0:
                if repo == primary_repo:
                    # When the secondary's direct parent is the primary, these
                    # two diffs are byte-for-byte identical. Keep the relational
                    # label and drop the duplicate own-improvement copy.
                    if (
                        own_parent_commit
                        and rev_parse(repo, own_parent_commit) == primary_commit
                    ):
                        summary["diff_from_primary_parent"] = summary.pop(
                            "own_improvement_diff"
                        )
                    else:
                        summary["diff_from_primary_parent"] = self._diff_context(
                            repo=primary_repo,
                            base=primary_commit,
                            head=commit,
                            max_chars=self.max_parent_diff_chars,
                        )
                else:
                    summary["diff_from_primary_parent"] = {
                        "skipped": "parent lives in a different git repository",
                        "repo_path": str(repo),
                    }
            summaries.append(summary)
        return summaries

    def _diff_context(
        self, *, repo: Path, base: str, head: str, max_chars: int
    ) -> dict[str, Any]:
        rev_range = f"{base}..{head}"
        stat = run_git(repo, ["diff", "--stat", rev_range], check=False).stdout
        name_status = run_git(
            repo, ["diff", "--name-status", rev_range], check=False
        ).stdout
        diff = run_git(
            repo,
            ["diff", "--find-renames", "--find-copies", rev_range],
            check=False,
        ).stdout
        return {
            "base": base,
            "head": head,
            "diff_stat": stat,
            "name_status": name_status,
            "diff": self._clip(diff, max_chars),
            "diff_was_truncated": len(diff) > max_chars,
        }

    def _reflection_changed_files(self, reflection: Any) -> list[str]:
        if isinstance(reflection, dict) and isinstance(
            reflection.get("changed_files"), list
        ):
            return [str(path) for path in reflection["changed_files"]]
        return []

    def _compact_reflection(self, reflection: Any) -> Any:
        """Keep the interpretation and outcome, not its prompt/provenance copies."""

        if reflection is None:
            return None
        if not isinstance(reflection, dict):
            return {
                "summary": self._clip(str(reflection), self.max_parent_reflection_chars)
            }

        compact: dict[str, Any] = {}
        summary = reflection.get("reflection") or reflection.get("summary")
        if summary:
            compact["summary"] = self._clip(
                str(summary), self.max_parent_reflection_chars
            )
        for source_key, output_key, max_chars in (
            ("reflection_status", "status", 100),
            ("skip_reason", "skip_reason", 1000),
            ("llm_error", "error", 1000),
        ):
            value = reflection.get(source_key)
            if value not in (None, ""):
                compact[output_key] = self._clip(str(value), max_chars)
        for key in ("metrics_delta", "delta"):
            value = reflection.get(key)
            if isinstance(value, dict) and value:
                compact[key] = value
                break
        attempt = reflection.get("attempt_record")
        if isinstance(attempt, dict) and attempt.get("verdict"):
            compact["verdict"] = self._clip(str(attempt["verdict"]), 100)
        return compact or None

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...<truncated>"

    async def _run_smoke_commands(self, *, worktree: Path, log_dir: Path) -> None:
        if not self.smoke_commands:
            return
        smoke_dir = log_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for idx, command in enumerate(self.smoke_commands, start=1):
            rendered_command: str | list[str]
            if isinstance(command, str):
                rendered_command = command.replace("{worktree}", str(worktree))
            else:
                rendered_command = [
                    str(part).replace("{worktree}", str(worktree)) for part in command
                ]
            code, stdout, stderr = await _run_command(
                rendered_command,
                cwd=worktree,
                timeout=self.smoke_timeout,
                shell=self.smoke_shell,
            )
            prefix = smoke_dir / f"{idx:02d}"
            (prefix.with_suffix(".stdout.txt")).write_text(stdout)
            (prefix.with_suffix(".stderr.txt")).write_text(stderr)
            if code != 0:
                raise MutationError(
                    f"Smoke command {idx} failed with exit code {code}: "
                    f"{stderr[-1000:]}"
                )
