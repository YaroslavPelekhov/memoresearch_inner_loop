from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.repo_harness.benchmark_evidence import build_benchmark_evidence
from gigaevo.repo_harness.git_utils import ensure_git_repo, rev_parse, run_git
from gigaevo.repo_harness.manifest import RepoCandidateManifest


class RepoCandidateCardBuilder:
    """Build bounded, benchmark-agnostic summaries for repo meta-agents.

    ProgramStorage remains the source of truth. Cards are prompt-time views over
    program metrics, metadata, reflection, lineage, Git changes, and normalized
    benchmark evidence.
    """

    def __init__(
        self,
        *,
        primary_key: str,
        higher_is_better: bool,
        source_repo: str | Path | None = None,
        metrics_context: MetricsContext | None = None,
        max_diff_chars: int = 12000,
        max_reflection_chars: int = 6000,
        max_raw_feedback_chars: int = 10000,
    ) -> None:
        self.primary_key = primary_key
        self.higher_is_better = higher_is_better
        self.source_repo = (
            Path(source_repo).expanduser().resolve() if source_repo else None
        )
        self.metrics_context = metrics_context
        self.max_diff_chars = max(0, int(max_diff_chars))
        self.max_reflection_chars = max(0, int(max_reflection_chars))
        self.max_raw_feedback_chars = max(0, int(max_raw_feedback_chars))

    def build(
        self,
        program: Program,
        *,
        candidate_id: str,
        parent: Program | None = None,
        archive: list[Program] | tuple[Program, ...] = (),
        origin: str | None = None,
        diff_budget: int | None = None,
        program_aliases: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        reflection = program.metadata.get("repo_reflection")
        manifest = self._manifest_for_program(program)
        changed_files = self._bounded_strings(
            self._changed_files(program, manifest, reflection), limit=128
        )
        primary_delta = self._primary_delta(program, parent=parent)
        evidence = build_benchmark_evidence(
            program,
            parent=parent,
            archive=archive,
            metrics_context=self.metrics_context,
            # Raw feedback contains the same metrics, cases, and diagnostics as
            # the normalized evidence plus benchmark-specific bulk artifacts.
            # Keep the normalized decision evidence instead.
            max_raw_feedback_chars=0,
        )
        aliases = program_aliases or {}

        card: dict[str, Any] = {
            "candidate_id": candidate_id,
            "short_id": program.short_id,
            "program_name": self._clip(str(program.name or ""), 400),
            "origin": origin,
            "metrics": dict(program.metrics),
            "primary_metric_value": program.metrics.get(self.primary_key),
            "primary_metric_delta": primary_delta,
            "lineage": {
                "parents": [
                    aliases.get(parent_id, str(parent_id)[:8])
                    for parent_id in program.lineage.parents
                ],
                "generation": program.lineage.generation,
                "mutation": self._clip(str(program.lineage.mutation or ""), 500),
            },
            "changed_files": changed_files,
            "repo_reflection": self._clip_reflection(reflection),
            "benchmark_evidence": self._compact_benchmark_evidence(
                program,
                evidence,
            ),
            "metadata_context": self._metadata_context(program),
        }
        if not card["program_name"]:
            card.pop("program_name")
        if not card["repo_reflection"]:
            card.pop("repo_reflection")
        if not card["metadata_context"]:
            card.pop("metadata_context")
        if program.failed_stages:
            card["stage_errors"] = self._clip(
                program.format_errors(include_traceback=False), 3000
            )

        if manifest is not None:
            card["repo_candidate"] = {
                # Short revisions are enough to reveal exact code identity and
                # parent/child relationships without repeating paths or branch
                # names shared by the whole run.
                "commit": manifest.commit[:12],
                "parent_commit": (
                    manifest.parent_commit[:12] if manifest.parent_commit else None
                ),
            }
            own_diff = self._own_improvement_diff(
                manifest,
                reflection,
                self.max_diff_chars if diff_budget is None else max(0, diff_budget),
            )
            if own_diff:
                card["own_improvement_diff"] = own_diff
        elif isinstance(reflection, dict):
            card["own_improvement_diff"] = {
                "diff_stat": reflection.get("diff_stat"),
                "name_status": reflection.get("name_status"),
                "changed_files": reflection.get("changed_files"),
                "source": "repo_reflection_metadata",
            }
        return card

    def _metadata_context(self, program: Program) -> dict[str, Any]:
        """Expose compact decision evidence without provenance duplication."""
        context: dict[str, Any] = {}
        descriptors = program.metadata.get("repo_descriptors")
        if isinstance(descriptors, dict):
            compact = self._compact_descriptors(descriptors)
            if compact:
                context["repo_descriptors"] = compact

        roles = program.metadata.get("archive_roles")
        if isinstance(roles, list) and roles:
            context["archive_roles"] = self._bounded_strings(roles, limit=12)
        return context

    def _primary_delta(self, program: Program, *, parent: Program | None) -> float:
        reflection = program.metadata.get("repo_reflection")
        if isinstance(reflection, dict):
            for key in ("metrics_delta", "delta"):
                delta = reflection.get(key)
                if isinstance(delta, dict) and self.primary_key in delta:
                    try:
                        value = float(delta[self.primary_key])
                    except (TypeError, ValueError):
                        return 0.0
                    return value if self.higher_is_better else -value
        if parent is not None:
            value = program.metrics.get(self.primary_key)
            parent_value = parent.metrics.get(self.primary_key)
            if isinstance(value, (int, float)) and isinstance(
                parent_value, (int, float)
            ):
                delta = float(value) - float(parent_value)
                return delta if self.higher_is_better else -delta
        return 0.0

    def _manifest_for_program(self, program: Program) -> RepoCandidateManifest | None:
        try:
            return RepoCandidateManifest.from_program(
                program, default_repo_path=self.source_repo
            )
        except Exception:
            if self.source_repo is None:
                return None
            try:
                return RepoCandidateManifest.from_metadata(
                    program, default_repo_path=self.source_repo
                )
            except Exception:
                return None

    @staticmethod
    def _changed_files(
        program: Program,
        manifest: RepoCandidateManifest | None,
        reflection: Any,
    ) -> list[str]:
        if manifest is not None and manifest.changed_files:
            return list(manifest.changed_files)
        if isinstance(reflection, dict) and isinstance(
            reflection.get("changed_files"), list
        ):
            return [str(path) for path in reflection["changed_files"]]
        for key in ("changed_files", "previous_changed_files"):
            value = program.metadata.get(key)
            if isinstance(value, list):
                return [str(path) for path in value]
        return []

    def _own_improvement_diff(
        self,
        manifest: RepoCandidateManifest,
        reflection: Any,
        diff_budget: int,
    ) -> dict[str, Any] | None:
        base = manifest.parent_commit
        if isinstance(reflection, dict) and reflection.get("parent_commit"):
            base = str(reflection["parent_commit"])
        if not base:
            return None
        try:
            repo = ensure_git_repo(manifest.repo_path)
            head = rev_parse(repo, manifest.commit)
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
                "base": base[:12],
                "head": head[:12],
                "diff_stat": stat,
                "name_status": name_status,
                "diff": self._clip(diff, diff_budget),
                "diff_was_truncated": len(diff) > diff_budget,
            }
        except Exception as exc:
            return {
                "skipped": self._clip(f"could not compute git diff: {exc}", 1000),
                "base": base[:12],
                "head": manifest.commit[:12],
            }

    def _clip_reflection(self, reflection: Any) -> dict[str, Any] | None:
        if not isinstance(reflection, dict):
            if reflection is None:
                return None
            return {
                "summary": self._clip(str(reflection), self.max_reflection_chars)
            }

        clipped: dict[str, Any] = {}
        summary = reflection.get("reflection") or reflection.get("summary")
        if summary:
            if not isinstance(summary, str):
                summary = json.dumps(summary, sort_keys=True, default=str)
            clipped["summary"] = self._clip(summary, self.max_reflection_chars)

        for source_key, output_key, limit in (
            ("reflection_status", "status", 100),
            ("skip_reason", "skip_reason", 1000),
            ("llm_error", "error", 1000),
        ):
            value = reflection.get(source_key)
            if value not in (None, ""):
                clipped[output_key] = self._clip(str(value), limit)

        attempt = reflection.get("attempt_record")
        if isinstance(attempt, dict) and attempt.get("verdict"):
            clipped["verdict"] = self._clip(str(attempt["verdict"]), 100)
        return clipped or None

    def _compact_benchmark_evidence(
        self,
        program: Program,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep behavioral distinctions and reliability without metric copies."""
        compact: dict[str, Any] = {
            "benchmark": evidence.get("benchmark"),
        }
        fingerprint = self._behavior_fingerprint(program, evidence)
        if fingerprint:
            compact["behavior_fingerprint"] = fingerprint
        diagnostics = self._compact_benchmark_diagnostics(program)
        if diagnostics:
            compact["benchmark_specific_diagnostics"] = diagnostics

        scope = evidence.get("scope")
        if isinstance(scope, dict) and any(value is not None for value in scope.values()):
            compact["scope"] = scope

        unit_results = evidence.get("unit_results")
        if isinstance(unit_results, list) and unit_results:
            compact["unit_results"] = self._compact_unit_results(unit_results)

        resources = evidence.get("resource_metrics")
        if isinstance(resources, dict) and resources:
            compact["resource_metrics"] = self._bounded_mapping(resources, limit=24)

        parent_comparison = evidence.get("comparison_with_parent")
        if isinstance(parent_comparison, dict):
            comparison = self._compact_parent_comparison(parent_comparison)
            if comparison:
                compact["comparison_with_parent"] = comparison

        archive_distinction = evidence.get("distinction_from_archive")
        if isinstance(archive_distinction, dict):
            distinction = self._compact_archive_distinction(archive_distinction)
            if distinction:
                compact["distinction_from_archive"] = distinction

        return {key: value for key, value in compact.items() if value is not None}

    def _compact_benchmark_diagnostics(self, program: Program) -> dict[str, Any]:
        """Retain bounded task-specific evidence omitted by the normalizer."""
        max_chars = min(self.max_raw_feedback_chars, 5000)
        if max_chars <= 0:
            return {}
        feedback = program.metadata.get("repo_benchmark_feedback")
        if not isinstance(feedback, dict):
            return {}
        structured = feedback.get("structured_feedback") or feedback.get(
            "structured_failure_feedback"
        )
        if not isinstance(structured, dict):
            return {}

        # These fields are already represented by canonical metrics, normalized
        # unit outcomes, resource evidence, or the behavior fingerprint.
        excluded = {
            "aggregate_metrics",
            "all_cases",
            "all_examples",
            "all_tasks",
            "artifacts",
            "benchmark",
            "case_results",
            "cases",
            "examples",
            "metrics",
            "points",
            "schema_version",
            "scope",
            "trials",
            "unit_results",
            "universe_ids",
        }
        priority = (
            "diagnosis",
            "summary",
            "failure_clusters",
            "worst_triplets",
            "point_pressure",
            "closest_pairs",
        )
        keys = [key for key in priority if key in structured]
        keys.extend(
            sorted(key for key in structured if key not in excluded and key not in keys)
        )

        compact: dict[str, Any] = {}
        remaining = max_chars
        for key in keys:
            if key in excluded or remaining < 80:
                continue
            value = structured[key]
            if key == "summary" and isinstance(value, dict):
                value = {
                    summary_key: summary_value
                    for summary_key, summary_value in value.items()
                    if summary_key
                    not in {
                        "aggregate_metrics",
                        "all_cases",
                        "all_examples",
                        "all_tasks",
                        "metrics",
                        "selected_cases",
                        "selected_examples",
                        "selected_tasks",
                        "universe_ids",
                    }
                }
            if value in (None, "", [], {}):
                continue
            serialized = json.dumps(value, sort_keys=True, default=str)
            field_budget = min(1600, remaining)
            stored: Any = (
                value
                if len(serialized) <= field_budget
                else self._clip(serialized, field_budget)
            )
            compact[key] = stored
            remaining -= len(json.dumps(stored, sort_keys=True, default=str))
        return compact

    def _compact_parent_comparison(self, comparison: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        parent_id = comparison.get("parent_program_id")
        if parent_id:
            compact["parent_short_id"] = str(parent_id)[:8]

        directional = comparison.get("directional_metric_deltas")
        if isinstance(directional, dict) and directional:
            compact["directional_metric_deltas"] = self._bounded_mapping(
                directional, limit=32
            )
        outcomes = comparison.get("metric_outcomes")
        if isinstance(outcomes, dict) and outcomes:
            compact["metric_outcomes"] = self._bounded_mapping(outcomes, limit=32)

        for key in (
            "improved_units",
            "regressed_units",
            "newly_successful_units",
            "lost_success_units",
        ):
            values = comparison.get(key)
            if isinstance(values, list) and values:
                compact[key] = self._bounded_strings(values, limit=64)
        return compact

    def _compact_archive_distinction(self, distinction: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key in ("archive_program_count", "archive_programs_with_unit_evidence"):
            value = distinction.get(key)
            if value is not None:
                compact[key] = value
        for key in (
            "uniquely_successful_units",
            "best_known_units",
            "best_known_metrics",
        ):
            values = distinction.get(key)
            if isinstance(values, list) and values:
                compact[key] = self._bounded_strings(values, limit=64)
        return compact

    def _compact_unit_results(self, rows: list[Any]) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        normalized: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            status = str(raw.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if raw.get("observed") is not True:
                continue
            row: dict[str, Any] = {
                "unit_id": self._clip(str(raw.get("unit_id") or ""), 240),
                "status": status,
                "score": raw.get("score"),
                "repetitions": raw.get("repetitions"),
            }
            metrics = raw.get("metrics")
            if isinstance(metrics, dict) and metrics:
                row["metrics"] = self._bounded_mapping(metrics, limit=16)
            normalized.append(row)

        limit = 64
        return {
            "total_count": len(rows),
            "observed_count": len(normalized),
            "status_counts": status_counts,
            "items": normalized[:limit],
            "truncated_count": max(0, len(normalized) - limit),
        }

    def _compact_descriptors(self, descriptors: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        code = descriptors.get("code")
        if isinstance(code, dict):
            code_summary = {
                "modules": self._bounded_strings(code.get("modules"), limit=32),
                "file_count": code.get("file_count"),
                "extensions": self._bounded_strings(
                    code.get("extensions"), limit=16
                ),
                "diff": code.get("diff"),
                "risk_flags": self._bounded_strings(
                    code.get("risk_flags"), limit=16
                ),
            }
            compact["code"] = {
                key: value
                for key, value in code_summary.items()
                if value not in (None, [], {})
            }

        feedback = descriptors.get("feedback")
        if isinstance(feedback, dict):
            feedback_summary = {
                "returncode": feedback.get("returncode"),
                "duration_seconds": feedback.get("duration_seconds"),
                "failure_count": feedback.get("failure_count"),
                "failure_cluster_labels": self._bounded_strings(
                    feedback.get("failure_cluster_labels"), limit=32
                ),
                "failed_case_ids": self._bounded_strings(
                    feedback.get("failed_case_ids"), limit=64
                ),
                "timeout": feedback.get("timeout"),
            }
            compact["reliability"] = {
                key: value
                for key, value in feedback_summary.items()
                if value not in (None, [], {})
            }

        risk = descriptors.get("risk")
        if isinstance(risk, dict):
            risk_summary = {
                "risk_flags": self._bounded_strings(
                    risk.get("risk_flags"), limit=16
                ),
                "quarantine_flags": self._bounded_strings(
                    risk.get("quarantine_flags"), limit=16
                ),
            }
            risk_summary = {
                key: value for key, value in risk_summary.items() if value
            }
            if risk_summary:
                compact["risk"] = risk_summary

        signature = descriptors.get("signature")
        if isinstance(signature, dict) and signature.get("sha1"):
            compact["semantic_fingerprint"] = str(signature["sha1"])[:16]
        return {key: value for key, value in compact.items() if value}

    def _behavior_fingerprint(
        self,
        program: Program,
        evidence: dict[str, Any],
    ) -> dict[str, str] | None:
        feedback = program.metadata.get("repo_benchmark_feedback")
        if not isinstance(feedback, dict):
            feedback = {}
        structured = feedback.get("structured_feedback") or feedback.get(
            "structured_failure_feedback"
        )
        if not isinstance(structured, dict):
            structured = feedback

        for key in (
            "behavior_fingerprint",
            "solution_fingerprint",
            "output_fingerprint",
        ):
            value = structured.get(key)
            if value not in (None, ""):
                return {"source": key, "value": self._clip(str(value), 200)}

        points = structured.get("points")
        if isinstance(points, list) and points:
            point_values = []
            for point in points:
                if isinstance(point, dict):
                    point_value = point.get(
                        "xy", point.get("coordinates", point.get("point"))
                    )
                    if point_value is None:
                        point_values = []
                        break
                    point_values.append(point_value)
                else:
                    point_values.append(point)
            payload: Any = point_values or None
            source = "points" if point_values else ""
        else:
            payload = None
            source = ""

        if payload is None:
            units = evidence.get("unit_results")
            if not isinstance(units, list) or not units:
                return None
            payload = [
                {
                    "unit_id": row.get("unit_id"),
                    "status": row.get("status"),
                    "score": row.get("score"),
                    "metrics": row.get("metrics"),
                }
                for row in units
                if isinstance(row, dict) and row.get("observed") is True
            ]
            if not payload:
                return None
            payload.sort(key=lambda row: str(row.get("unit_id") or ""))
            source = "observed_unit_results"

        serialized = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        return {
            "source": source,
            "value": hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16],
        }

    @classmethod
    def _bounded_mapping(
        cls, value: dict[Any, Any], *, limit: int
    ) -> dict[str, Any]:
        items = list(value.items())
        bounded: dict[str, Any] = {}
        for key, item in items[:limit]:
            serialized = json.dumps(item, sort_keys=True, default=str)
            bounded[str(key)] = (
                item if len(serialized) <= 1000 else cls._clip(serialized, 1000)
            )
        if len(items) > limit:
            bounded["_truncated_item_count"] = len(items) - limit
        return bounded

    @classmethod
    def _bounded_strings(cls, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        items = [cls._clip(str(item), 240) for item in value]
        bounded = items[:limit]
        if len(items) > limit:
            bounded.append(f"...<{len(items) - limit} more>")
        return bounded

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...<truncated>"
