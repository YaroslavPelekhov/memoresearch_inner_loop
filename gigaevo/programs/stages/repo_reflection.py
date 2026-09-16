from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
import tiktoken

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.codex_usage import codex_usage_scope, response_usage
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.repo_harness.git_utils import ensure_git_repo, run_git
from gigaevo.repo_harness.manifest import RepoCandidateManifest

_DEFAULT_PROMPT_TOKEN_ENCODING = "o200k_base"
_PROMPT_BUDGET_CHARS_PER_TOKEN = 3
_FEEDBACK_METADATA_KEYS = (
    "benchmark_run_label",
    "duration_seconds",
    "returncode",
)
_STRUCTURED_FEEDBACK_KEYS = (
    "summary",
    "cases",
    "case_results",
    "trials",
    "examples",
    "failure_clusters",
)
_ARTIFACT_ITEM_KEYS = (
    "kind",
    "name",
    "role",
    "description",
    "bytes",
    "captured",
    "captured_path",
    "source_path",
)


class RepoReflectionInputs(StageIO):
    """Optional metrics edge used to order/cache repo reflection after benchmarking."""

    metrics: FloatDictContainer | None


def _clip(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    half = max(0, (max_chars - 80) // 2)
    return text[:half] + "\n...<truncated repo-reflection payload>...\n" + text[-half:]


def _json_safe(obj: Any, *, max_chars: int) -> str:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str, ensure_ascii=True)
    return _clip(text, max_chars)


def _bounded_json_value(value: Any, *, max_chars: int) -> Any:
    """Keep JSON structure when it fits and otherwise expose a bounded excerpt."""

    text = json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=True)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_chars": len(text),
        "excerpt": _clip(text, max_chars),
    }


def _compact_benchmark_feedback(feedback: Any, *, max_chars: int) -> Any:
    """Remove prompt-time copies while retaining benchmark decision evidence."""

    if not isinstance(feedback, dict) or max_chars <= 0:
        return None

    compact: dict[str, Any] = {
        key: feedback[key] for key in _FEEDBACK_METADATA_KEYS if key in feedback
    }
    structured = feedback.get("structured_feedback")
    if not isinstance(structured, dict):
        structured = feedback.get("structured_failure_feedback")
    if not isinstance(structured, dict) and any(
        key in feedback for key in _STRUCTURED_FEEDBACK_KEYS
    ):
        structured = {
            key: value
            for key, value in feedback.items()
            if key
            not in {
                "command",
                "metrics",
                "evaluation_artifacts",
                "stdout_tail",
                "stderr_tail",
                "worktree",
            }
        }

    if isinstance(structured, dict):
        # Metrics and artifacts have authoritative top-level prompt sections.
        # Keeping them here produced exact duplicates in real reflection calls.
        canonical = {
            key: value
            for key, value in structured.items()
            if key not in {"aggregate_metrics", "artifacts", "metrics"}
        }
        if canonical:
            # Reserve room for run status and stderr so bounding one large
            # diagnostic does not collapse the entire feedback object into an
            # opaque string excerpt.
            evidence_budget = max(1000, max_chars - min(2000, max_chars // 4))
            compact["structured_feedback"] = _bounded_json_value(
                canonical, max_chars=evidence_budget
            )
    else:
        diagnostics = {
            key: value
            for key, value in feedback.items()
            if key
            not in {
                *_FEEDBACK_METADATA_KEYS,
                "branch",
                "command",
                "commit",
                "entrypoint",
                "evaluation_artifacts",
                "metrics",
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
        compact["stderr_tail"] = _clip(stderr_tail, min(4000, max_chars))

    # Successful structured feedback already explains the metrics represented
    # by benchmark stdout. Preserve stdout only for unstructured benchmarks.
    stdout_tail = feedback.get("stdout_tail")
    if (
        not isinstance(structured, dict)
        and isinstance(stdout_tail, str)
        and stdout_tail.strip()
    ):
        compact["stdout_tail"] = _clip(stdout_tail, min(4000, max_chars))

    return _bounded_json_value(compact, max_chars=max_chars)


def _compact_evaluation_artifacts(artifacts: Any, *, max_chars: int) -> Any:
    """Keep actionable artifact references without storage limits and path copies."""

    if not isinstance(artifacts, dict) or max_chars <= 0:
        return None
    items = artifacts.get("items")
    compact_items: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            compact_item = {
                key: item[key] for key in _ARTIFACT_ITEM_KEYS if key in item
            }
            if compact_item:
                compact_items.append(compact_item)
    compact = {
        "items": compact_items,
        "manifest_path": artifacts.get("manifest_path"),
        "stats": artifacts.get("stats"),
    }
    compact = {key: value for key, value in compact.items() if value is not None}
    return _bounded_json_value(compact, max_chars=max_chars)


def _message_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(response)


def _name_status_files(name_status: str) -> list[str]:
    files: list[str] = []
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append(parts[-1])
    return files


def _metric_delta(
    parent: dict[str, float], child: dict[str, float]
) -> dict[str, float]:
    keys = set(parent) | set(child)
    return {
        key: float(child.get(key, 0.0)) - float(parent.get(key, 0.0)) for key in keys
    }


def _agent_trace_excerpt(
    artifacts: Any, *, max_chars: int, max_cases: int = 5
) -> dict[str, Any] | None:
    """Load a diverse, bounded sample of exact agent traces for the reflection LLM.

    Reflection is a plain model call and cannot open local artifact paths itself.
    Mutation coding agents still receive the complete captured directory and can
    inspect any trace; this helper embeds representative failures for reflection.
    """

    if max_chars <= 0 or not isinstance(artifacts, dict):
        return None
    items = artifacts.get("items")
    if not isinstance(items, list):
        return None
    trace_root: Path | None = None
    for item in items:
        if not isinstance(item, dict) or not item.get("captured"):
            continue
        if item.get("kind") != "agent_trajectories":
            continue
        captured_path = item.get("captured_path")
        if isinstance(captured_path, str):
            candidate = Path(captured_path).expanduser().resolve()
            if candidate.is_dir():
                trace_root = candidate
                break
    if trace_root is None:
        return None

    index_path = trace_root / "index.jsonl"
    if not index_path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "artifact_root": str(trace_root),
            "read_error": f"{type(exc).__name__}: {exc}",
        }

    failures = [row for row in rows if not bool(row.get("strict"))]
    selected: list[dict[str, Any]] = []
    seen_clusters: set[tuple[Any, ...]] = set()
    for row in failures:
        cluster = (
            row.get("hops"),
            row.get("status"),
            bool(row.get("error")),
            bool(row.get("judge_error")),
        )
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        selected.append(row)
        if len(selected) >= max_cases:
            break
    if len(selected) < max_cases:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in failures if id(row) not in selected_ids)
        selected = selected[:max_cases]

    per_case_chars = max(1000, max_chars // max(len(selected), 1))
    excerpts: list[dict[str, Any]] = []
    for row in selected:
        trace_file = row.get("trace_file")
        if not isinstance(trace_file, str):
            continue
        trace_path = (trace_root / trace_file).resolve()
        if trace_root not in trace_path.parents or not trace_path.is_file():
            continue
        try:
            trace_text = trace_path.read_text(encoding="utf-8")
        except OSError as exc:
            trace_text = f"<{type(exc).__name__}: {exc}>"
        excerpts.append(
            {
                "case_id": row.get("case_id"),
                "hops": row.get("hops"),
                "status": row.get("status"),
                "trace_path": str(trace_path),
                "exact_trace_excerpt": _clip(trace_text, per_case_chars),
            }
        )
    return {
        "artifact_root": str(trace_root),
        "selection": "diverse strict failures by hop/status/error stage",
        "sampled_cases": excerpts,
        "full_trace_count": len(rows),
    }


@StageRegistry.register(
    description="Summarize repo-level diffs, benchmark feedback, and lineage"
)
class RepoReflectionStage(Stage):
    """LLM-backed repo insight/lineage summary for Git-backed candidates.

    The repo harness stores a manifest in ``Program.code`` while the real
    artifact lives in Git. This stage translates that Git state into a compact
    mutation context: deterministic diff data plus an LLM interpretation of the
    parent->child transition.
    """

    InputsModel = RepoReflectionInputs
    OutputModel = StringContainer

    def __init__(
        self,
        *,
        llm: Any,
        storage: ProgramStorage,
        task_description: str,
        metrics_context: MetricsContext,
        max_diff_chars: int = 20000,
        max_feedback_chars: int = 8000,
        max_prompt_tokens: int = 150000,
        prompt_token_encoding: str = _DEFAULT_PROMPT_TOKEN_ENCODING,
        max_prompt_chars: int | None = None,
        max_trace_excerpt_chars: int = 18000,
        fail_open: bool = True,
        skip_llm_for_clear_regressions: bool = False,
        skip_llm_for_invalid: bool = False,
        regression_skip_tolerance: float = 1.0e-9,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.storage = storage
        self.task_description = task_description
        self.metrics_context = metrics_context
        self.max_diff_chars = max_diff_chars
        self.max_feedback_chars = max_feedback_chars
        if max_prompt_chars is not None:
            max_prompt_tokens = max(
                1000, int(max_prompt_chars) // _PROMPT_BUDGET_CHARS_PER_TOKEN
            )
        if max_prompt_tokens < 1000:
            raise ValueError(
                f"max_prompt_tokens must be at least 1000, got {max_prompt_tokens}"
            )
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.prompt_token_encoding = str(prompt_token_encoding).strip()
        try:
            self._prompt_tokenizer = tiktoken.get_encoding(self.prompt_token_encoding)
        except ValueError as exc:
            raise ValueError(
                f"unknown prompt_token_encoding {self.prompt_token_encoding!r}"
            ) from exc
        self.max_trace_excerpt_chars = max_trace_excerpt_chars
        self.fail_open = fail_open
        self.skip_llm_for_clear_regressions = skip_llm_for_clear_regressions
        self.skip_llm_for_invalid = skip_llm_for_invalid
        self.regression_skip_tolerance = abs(float(regression_skip_tolerance))

    async def compute(self, program: Program) -> StringContainer:
        manifest = RepoCandidateManifest.from_program(program)
        parent_program = await self._get_parent(program)
        parent_manifest = self._parent_manifest(parent_program)
        parent_commit = manifest.parent_commit or (
            parent_manifest.commit if parent_manifest else None
        )

        repo = ensure_git_repo(manifest.repo_path)
        diff_stat = ""
        name_status = ""
        diff = ""
        if parent_commit:
            rev_range = f"{parent_commit}..{manifest.commit}"
            diff_stat = run_git(repo, ["diff", "--stat", rev_range], check=False).stdout
            name_status = run_git(
                repo, ["diff", "--name-status", rev_range], check=False
            ).stdout
            diff = run_git(
                repo,
                ["diff", "--find-renames", "--find-copies", rev_range],
                check=False,
            ).stdout

        changed_files = manifest.changed_files or _name_status_files(name_status)
        child_metrics = self._current_metrics(program)
        parent_metrics = parent_program.metrics if parent_program else {}
        metrics_delta = (
            _metric_delta(parent_metrics, child_metrics)
            if parent_metrics and child_metrics
            else {}
        )
        benchmark_feedback = _compact_benchmark_feedback(
            program.metadata.get("repo_benchmark_feedback"),
            max_chars=self.max_feedback_chars,
        )
        evaluation_artifacts = program.metadata.get("repo_evaluation_artifacts")
        compact_artifacts = _compact_evaluation_artifacts(
            evaluation_artifacts,
            max_chars=min(12000, max(0, self.max_feedback_chars)),
        )
        parent_feedback = _compact_benchmark_feedback(
            (
                parent_program.metadata.get("repo_benchmark_feedback")
                if parent_program
                else None
            ),
            max_chars=self.max_feedback_chars,
        )
        parent_reflection = self._parent_reflection(parent_program)
        payload = {
            "task_description": self.task_description,
            "candidate": {
                "program_id": program.id,
                "short_id": program.short_id,
                "commit": manifest.commit,
                "parent_commit": parent_commit,
                "branch": manifest.branch,
                "entrypoint": manifest.entrypoint,
                "changed_files": changed_files,
                "mutation_agent": manifest.mutation_agent,
            },
            "parent": {
                "program_id": parent_program.id if parent_program else None,
                "short_id": parent_program.short_id if parent_program else None,
                "commit": parent_manifest.commit if parent_manifest else parent_commit,
            },
            "metrics": {
                "description": self._metrics_description(),
                "child": child_metrics,
                "parent": parent_metrics,
                "delta": metrics_delta,
            },
            "benchmark_evidence": benchmark_feedback,
            "evaluation_artifacts": compact_artifacts,
            "evaluation_trace_excerpt": _agent_trace_excerpt(
                evaluation_artifacts, max_chars=self.max_trace_excerpt_chars
            ),
            "parent_benchmark_evidence": parent_feedback,
            "parent_repo_reflection": parent_reflection,
            "git": {
                "diff_stat": diff_stat,
                "name_status": name_status,
                "diff": _clip(diff, self.max_diff_chars),
                "diff_was_truncated": len(diff) > self.max_diff_chars,
            },
        }

        skip_reason = self._llm_skip_reason(payload)
        llm_skipped = skip_reason is not None
        llm_error = None
        codex_usage = None
        prompt_token_count = None
        if llm_skipped:
            reflection = self._skipped_reflection(
                payload, skip_reason=skip_reason or "reflection skipped"
            )
            logger.info(
                "[RepoReflectionStage] Skipped LLM reflection for {}: {}",
                program.short_id,
                skip_reason,
            )
        else:
            try:
                prompt = self._build_prompt(payload)
                prompt_token_count = self._count_prompt_tokens(prompt)
                logger.debug(
                    "[RepoReflectionStage] Prompt for {} uses {}/{} tokens ({})",
                    program.short_id,
                    prompt_token_count,
                    self.max_prompt_tokens,
                    self.prompt_token_encoding,
                )
                with codex_usage_scope(
                    source="repo_reflection",
                    generation=program.generation,
                    program_id=program.id,
                    commit=manifest.commit,
                ):
                    reflection, codex_usage = await self._call_llm(prompt)
            except Exception as exc:
                if not self.fail_open:
                    raise
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[RepoReflectionStage] Reflection LLM failed for {}: {}",
                    program.short_id,
                    llm_error,
                )
                reflection = self._fallback_reflection(payload, llm_error=llm_error)

        reflection_status = (
            "skipped" if llm_skipped else "failed" if llm_error else "generated"
        )
        attempt_record = self._attempt_record(
            payload,
            reflection_status=reflection_status,
            skip_reason=skip_reason,
            llm_error=llm_error,
        )
        output = self._format_output(
            payload,
            reflection,
            llm_error=llm_error,
            reflection_status=reflection_status,
            skip_reason=skip_reason,
        )
        program.set_metadata(
            "repo_reflection",
            {
                "commit": manifest.commit,
                "parent_commit": parent_commit,
                "changed_files": changed_files,
                "diff_stat": diff_stat,
                "metrics_child": child_metrics,
                "metrics_parent": parent_metrics,
                "metrics_delta": metrics_delta,
                "llm_error": llm_error,
                "llm_skipped": llm_skipped,
                "skip_reason": skip_reason,
                "reflection_status": reflection_status,
                "attempt_record": attempt_record,
                "reflection": reflection,
                "codex_usage": codex_usage,
                "prompt_token_count": prompt_token_count,
                "prompt_token_budget": self.max_prompt_tokens,
                "prompt_token_encoding": self.prompt_token_encoding,
            },
        )
        return StringContainer(data=output)

    async def _get_parent(self, program: Program) -> Program | None:
        if not program.lineage.parents:
            return None
        return await self.storage.get(program.lineage.parents[0])

    def _parent_manifest(self, parent: Program | None) -> RepoCandidateManifest | None:
        if parent is None:
            return None
        try:
            return RepoCandidateManifest.from_program(parent)
        except ValueError:
            return None

    def _current_metrics(self, program: Program) -> dict[str, float]:
        params = self.params
        if params.metrics is not None:
            return dict(params.metrics.data)
        return dict(program.metrics)

    def _parent_reflection(self, parent: Program | None) -> dict[str, Any] | None:
        if parent is None:
            return None
        reflection = parent.metadata.get("repo_reflection")
        if not isinstance(reflection, dict):
            return None
        return {
            "commit": reflection.get("commit"),
            "parent_commit": reflection.get("parent_commit"),
            "changed_files": reflection.get("changed_files"),
            "reflection": _clip(str(reflection.get("reflection") or ""), 4000),
        }

    def _metrics_description(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, spec in self.metrics_context.specs.items():
            out[key] = {
                "description": spec.description,
                "higher_is_better": self.metrics_context.is_higher_better(key),
                "lower_bound": spec.lower_bound,
                "upper_bound": spec.upper_bound,
                "is_primary": spec.is_primary,
            }
        return out

    def _primary_key(self) -> str | None:
        try:
            return self.metrics_context.get_primary_key()
        except Exception:
            return None

    def _objective_delta(self, delta: dict[str, float]) -> tuple[str, float] | None:
        primary_key = self._primary_key()
        if not primary_key or primary_key not in delta:
            return None
        raw_delta = float(delta[primary_key])
        try:
            higher_is_better = self.metrics_context.is_higher_better(primary_key)
        except Exception:
            higher_is_better = True
        return primary_key, raw_delta if higher_is_better else -raw_delta

    def _llm_skip_reason(self, payload: dict[str, Any]) -> str | None:
        metrics = payload["metrics"]
        child_metrics = metrics.get("child") or {}
        if self.skip_llm_for_invalid:
            validity = child_metrics.get(VALIDITY_KEY, 1.0)
            try:
                if float(validity) <= 0.0:
                    return f"{VALIDITY_KEY}={validity}; deterministic reflection only"
            except (TypeError, ValueError):
                return f"{VALIDITY_KEY}={validity!r}; deterministic reflection only"

        if not self.skip_llm_for_clear_regressions:
            return None

        objective_delta = self._objective_delta(metrics.get("delta") or {})
        if objective_delta is None:
            return None
        primary_key, signed_delta = objective_delta
        if signed_delta < -self.regression_skip_tolerance:
            return (
                f"clear primary-metric regression on {primary_key}: "
                f"objective_delta={signed_delta:.6g}; deterministic reflection only"
            )
        return None

    def _verdict(self, payload: dict[str, Any]) -> str:
        child_metrics = payload["metrics"].get("child") or {}
        try:
            if float(child_metrics.get(VALIDITY_KEY, 1.0)) <= 0.0:
                return "invalid"
        except (TypeError, ValueError):
            return "invalid"

        objective_delta = self._objective_delta(payload["metrics"].get("delta") or {})
        if objective_delta is None:
            return "unknown"
        _, signed_delta = objective_delta
        if signed_delta > self.regression_skip_tolerance:
            return "improved"
        if signed_delta < -self.regression_skip_tolerance:
            return "regressed"
        return "neutral"

    def _benchmark_summary(self, feedback: Any) -> dict[str, Any]:
        if not isinstance(feedback, dict):
            return {}
        structured = feedback.get("structured_feedback") or feedback.get(
            "structured_failure_feedback"
        )
        if not isinstance(structured, dict):
            structured = feedback if isinstance(feedback.get("summary"), dict) else {}
        summary = structured.get("summary") if isinstance(structured, dict) else None
        failure_clusters = (
            structured.get("failure_clusters") if isinstance(structured, dict) else None
        )
        return {
            "returncode": feedback.get("returncode"),
            "duration_seconds": feedback.get("duration_seconds"),
            "summary": summary if isinstance(summary, dict) else None,
            "failure_cluster_count": len(failure_clusters)
            if isinstance(failure_clusters, list)
            else None,
        }

    def _attempt_record(
        self,
        payload: dict[str, Any],
        *,
        reflection_status: str,
        skip_reason: str | None,
        llm_error: str | None,
    ) -> dict[str, Any]:
        candidate = payload["candidate"]
        parent = payload["parent"]
        metrics = payload["metrics"]
        git = payload["git"]
        objective_delta = self._objective_delta(metrics.get("delta") or {})
        return {
            "schema_version": 1,
            "program_id": candidate["program_id"],
            "short_id": candidate["short_id"],
            "commit": candidate["commit"],
            "parent_program_id": parent["program_id"],
            "parent_short_id": parent["short_id"],
            "parent_commit": candidate["parent_commit"],
            "changed_files": candidate["changed_files"],
            "diff_stat": git["diff_stat"],
            "name_status": git["name_status"],
            "metrics_child": metrics["child"],
            "metrics_parent": metrics["parent"],
            "metrics_delta": metrics["delta"],
            "primary_key": objective_delta[0]
            if objective_delta
            else self._primary_key(),
            "objective_primary_delta": objective_delta[1] if objective_delta else None,
            "verdict": self._verdict(payload),
            "reflection_status": reflection_status,
            "skip_reason": skip_reason,
            "llm_error": llm_error,
            "benchmark": self._benchmark_summary(payload.get("benchmark_evidence")),
        }

    def _build_prompt(self, payload: dict[str, Any]) -> list[BaseMessage]:
        system = (
            "You analyze repository-level mutations in an evolutionary search over "
            "coding-agent harnesses. Produce compact, factual markdown for the next "
            "mutation agent. Do not invent benchmark facts. Use the diff, changed "
            "files, benchmark feedback, evaluation artifacts, and metric deltas as "
            "evidence. This is a self-contained summarization task: do not invoke "
            "tools, execute shell commands, inspect the filesystem, or search for "
            "additional context."
        )
        instruction_prefix = (
            "Return markdown with these sections:\n"
            "1. Change Summary - what changed, by file or subsystem.\n"
            "2. Lineage Assessment - whether the child improved or regressed vs parent.\n"
            "3. Likely Mechanism - why the change may have affected metrics.\n"
            "4. Risks - overfitting, brittleness, broken abstractions, or benchmark leakage.\n\n"
            "Keep it concise. Prefer evidence-backed bullets over speculation.\n"
            "Keep the analysis retrospective. Do not propose follow-up edits, mutation "
            "plans, next steps, or instructions for what the next agent should change.\n"
            "Use evaluation_trace_excerpt as exact pipeline evidence before diagnosing "
            "failures. It is a bounded sample; distinguish repeated mechanisms from "
            "case-specific details. Complete trace paths remain in evaluation_artifacts "
            "for tool-capable mutation agents.\n\n"
            "All available evidence is in Context JSON. Do not use tools or read files.\n\n"
            "Context JSON:\n"
        )
        prompt_payload = json.loads(json.dumps(payload, default=str))
        messages = self._prompt_messages(
            system=system,
            instruction_prefix=instruction_prefix,
            payload=prompt_payload,
        )
        if self._count_prompt_tokens(messages) <= self.max_prompt_tokens:
            return messages

        self._compress_prompt_payload(prompt_payload, emergency=False)
        messages = self._prompt_messages(
            system=system,
            instruction_prefix=instruction_prefix,
            payload=prompt_payload,
            compact_json=True,
        )
        if self._count_prompt_tokens(messages) <= self.max_prompt_tokens:
            return messages

        self._compress_prompt_payload(prompt_payload, emergency=True)
        messages = self._prompt_messages(
            system=system,
            instruction_prefix=instruction_prefix,
            payload=prompt_payload,
            compact_json=True,
        )
        token_count = self._count_prompt_tokens(messages)
        if token_count > self.max_prompt_tokens:
            raise ValueError(
                "repo reflection prompt exceeds max_prompt_tokens after structured "
                f"compression ({token_count} > {self.max_prompt_tokens} tokens)"
            )
        return messages

    @staticmethod
    def _prompt_messages(
        *,
        system: str,
        instruction_prefix: str,
        payload: dict[str, Any],
        compact_json: bool = False,
    ) -> list[BaseMessage]:
        context_json = json.dumps(
            payload,
            separators=(",", ":") if compact_json else None,
            indent=None if compact_json else 2,
            sort_keys=True,
            default=str,
            ensure_ascii=True,
        )
        return [
            SystemMessage(content=system),
            HumanMessage(content=instruction_prefix + context_json),
        ]

    def _count_prompt_tokens(self, messages: list[BaseMessage]) -> int:
        rendered = "\n\n".join(
            f"{str(getattr(message, 'type', type(message).__name__)).upper()}:\n"
            f"{message.content}"
            for message in messages
        )
        return len(self._prompt_tokenizer.encode_ordinary(rendered))

    def _compress_prompt_payload(
        self, payload: dict[str, Any], *, emergency: bool
    ) -> None:
        limits = (
            {
                "task_description": 8000,
                "benchmark_evidence": 3500,
                "parent_benchmark_evidence": 2500,
                "evaluation_artifacts": 2000,
                "evaluation_trace_excerpt": 4000,
                "parent_repo_reflection": 1200,
                "diff": 1200,
            }
            if emergency
            else {
                "task_description": 30000,
                "benchmark_evidence": 12000,
                "parent_benchmark_evidence": 8000,
                "evaluation_artifacts": 6000,
                "evaluation_trace_excerpt": 12000,
                "parent_repo_reflection": 3000,
                "diff": 8000,
            }
        )
        task_description = payload.get("task_description")
        if isinstance(task_description, str):
            payload["task_description"] = _clip(
                task_description, limits["task_description"]
            )

        for key in ("benchmark_evidence", "parent_benchmark_evidence"):
            value = payload.get(key)
            if value is not None:
                payload[key] = self._compress_feedback_evidence(
                    value, max_chars=limits[key]
                )

        for key in (
            "evaluation_artifacts",
            "evaluation_trace_excerpt",
            "parent_repo_reflection",
        ):
            value = payload.get(key)
            if value is not None:
                payload[key] = _bounded_json_value(value, max_chars=limits[key])

        git = payload.get("git")
        if isinstance(git, dict) and isinstance(git.get("diff"), str):
            git["diff"] = _clip(git["diff"], limits["diff"])
            git["diff_was_prompt_compressed"] = True

    @staticmethod
    def _compress_feedback_evidence(value: Any, *, max_chars: int) -> Any:
        if not isinstance(value, dict) or "structured_feedback" not in value:
            return _bounded_json_value(value, max_chars=max_chars)
        compact = dict(value)
        structured_budget = max(800, max_chars - min(1200, max_chars // 3))
        compact["structured_feedback"] = _bounded_json_value(
            compact["structured_feedback"], max_chars=structured_budget
        )
        for key in ("stderr_tail", "stdout_tail"):
            if isinstance(compact.get(key), str):
                compact[key] = _clip(compact[key], min(1000, max_chars))
        return compact

    async def _call_llm(
        self, prompt: list[BaseMessage]
    ) -> tuple[str, dict[str, Any] | None]:
        response = await self.llm.ainvoke(prompt)
        return _message_text(response).strip(), response_usage(response)

    def _fallback_reflection(self, payload: dict[str, Any], *, llm_error: str) -> str:
        candidate = payload["candidate"]
        metrics = payload["metrics"]
        git = payload["git"]
        return (
            "## Change Summary\n"
            f"- Changed files: {candidate['changed_files'] or 'none recorded'}\n"
            f"- Diffstat:\n```text\n{git['diff_stat'] or 'N/A'}\n```\n\n"
            "## Lineage Assessment\n"
            f"- Parent metrics: {metrics['parent'] or 'N/A'}\n"
            f"- Child metrics: {metrics['child'] or 'N/A'}\n"
            f"- Delta: {metrics['delta'] or 'N/A'}\n\n"
            "## Reflection Status\n"
            f"- LLM reflection failed open: {llm_error}\n"
        )

    def _skipped_reflection(self, payload: dict[str, Any], *, skip_reason: str) -> str:
        candidate = payload["candidate"]
        metrics = payload["metrics"]
        git = payload["git"]
        verdict = self._verdict(payload)
        objective_delta = self._objective_delta(metrics.get("delta") or {})
        objective_line = (
            f"- Objective primary delta: {objective_delta[1]:.6g} "
            f"on `{objective_delta[0]}`\n"
            if objective_delta
            else "- Objective primary delta: unavailable\n"
        )
        return (
            "## Change Summary\n"
            f"- Changed files: {candidate['changed_files'] or 'none recorded'}\n"
            f"- Diffstat:\n```text\n{git['diff_stat'] or 'N/A'}\n```\n\n"
            "## Lineage Assessment\n"
            f"- Verdict: `{verdict}`.\n"
            f"{objective_line}"
            f"- Parent metrics: {metrics['parent'] or 'N/A'}\n"
            f"- Child metrics: {metrics['child'] or 'N/A'}\n"
            f"- Delta: {metrics['delta'] or 'N/A'}\n\n"
            "## Likely Mechanism\n"
            "- Expensive LLM reflection was skipped because this candidate met a "
            f"deterministic skip rule: {skip_reason}.\n"
            "- Treat this as an evaluated attempt record, not as an LLM-diagnosed "
            "mechanism.\n\n"
            "## Risks\n"
            "- Do not infer a successful reusable mechanism from this candidate "
            "without inspecting its artifacts manually.\n"
        )

    def _format_output(
        self,
        payload: dict[str, Any],
        reflection: str,
        *,
        llm_error: str | None,
        reflection_status: str,
        skip_reason: str | None,
    ) -> str:
        candidate = payload["candidate"]
        parent = payload["parent"]
        metrics = payload["metrics"]
        git = payload["git"]
        feedback = payload.get("benchmark_evidence")
        artifacts = payload.get("evaluation_artifacts")
        parent_feedback = payload.get("parent_benchmark_evidence")
        parent_reflection = payload.get("parent_repo_reflection")
        error_block = f"## Reflection Error\n{llm_error}\n\n" if llm_error else ""
        skip_block = (
            "## Reflection Skip\n"
            f"- Status: {reflection_status}\n"
            f"- Reason: {skip_reason}\n\n"
            if skip_reason
            else ""
        )
        return (
            "# Repo Reflection Context\n\n"
            "## Candidate\n"
            f"- Program: {candidate['short_id']}\n"
            f"- Commit: {candidate['commit']}\n"
            f"- Parent program: {parent['short_id'] or 'none'}\n"
            f"- Parent commit: {candidate['parent_commit'] or 'none'}\n"
            f"- Changed files: {candidate['changed_files'] or 'none recorded'}\n\n"
            "## LLM Repo Insight And Lineage\n"
            f"{reflection.strip() or 'N/A'}\n\n"
            f"{error_block}"
            f"{skip_block}"
            "## Metrics\n"
            f"```json\n{_json_safe(metrics, max_chars=6000)}\n```\n\n"
            "## Git Diff Stat\n"
            f"```text\n{git['diff_stat'] or 'N/A'}\n```\n\n"
            "## Name Status\n"
            f"```text\n{git['name_status'] or 'N/A'}\n```\n\n"
            "## Benchmark Evidence\n"
            f"```json\n{_json_safe(feedback, max_chars=self.max_feedback_chars)}\n```\n\n"
            "## Evaluation Artifacts\n"
            f"```json\n{_json_safe(artifacts, max_chars=12000)}\n```\n\n"
            "## Parent Benchmark Evidence\n"
            f"```json\n{_json_safe(parent_feedback, max_chars=self.max_feedback_chars)}\n```\n\n"
            "## Parent Repo Reflection\n"
            f"```json\n{_json_safe(parent_reflection, max_chars=6000)}\n```\n\n"
            "## Raw Diff Excerpt\n"
            f"```diff\n{git['diff'] or 'N/A'}\n```\n"
        )
