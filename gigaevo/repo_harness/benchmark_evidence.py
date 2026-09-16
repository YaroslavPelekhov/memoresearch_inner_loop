from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isclose, isfinite
from typing import Any, Protocol

from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program

__all__ = [
    "BenchmarkEvidenceProvider",
    "DefaultBenchmarkEvidenceProvider",
    "build_benchmark_evidence",
]

_UNIT_COLLECTION_KEYS = ("cases", "case_results", "trials", "examples")
_COMPLETENESS_VALUES = {
    "full",
    "partial",
    "projected",
    "aggregate_only",
    "unknown",
}
_SUCCESS_STATUSES = {"passed", "pass", "success", "succeeded", "solved", "ok"}
_FAILURE_STATUSES = {
    "failed",
    "fail",
    "failure",
    "unsolved",
    "incorrect",
}
_ERROR_STATUSES = {
    "error",
    "errored",
    "timeout",
    "timed_out",
    "crashed",
    "exception",
}
_UNKNOWN_STATUSES = {
    "",
    "unknown",
    "not_run",
    "not-run",
    "unevaluated",
    "skipped",
    "pending",
}
_RESOURCE_NAME_PARTS = {
    "bytes",
    "calls",
    "cost",
    "cpu",
    "duration",
    "energy",
    "latency",
    "memory",
    "price",
    "requests",
    "runtime",
    "seconds",
    "steps",
    "time",
    "tokens",
    "turns",
    "usd",
}
_UNIT_RESERVED_KEYS = {
    "attempts",
    "case",
    "case_id",
    "example",
    "example_id",
    "fitness",
    "id",
    "instance",
    "instance_id",
    "metrics",
    "n_trials",
    "name",
    "observed",
    "ok",
    "pass_rate",
    "passed",
    "projected",
    "repetitions",
    "result",
    "reward",
    "runs",
    "scenario",
    "scenario_id",
    "score",
    "status",
    "succeeded",
    "success",
    "success_rate",
    "task",
    "task_id",
    "trial",
    "trial_count",
    "unit_id",
    "value",
}


class BenchmarkEvidenceProvider(Protocol):
    """Convert benchmark-specific program metadata into curator-friendly evidence."""

    def extract(
        self,
        program: Program,
        *,
        parent: Program | None = None,
        archive: Sequence[Program] = (),
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DefaultBenchmarkEvidenceProvider:
    """Default normalizer for the structured feedback emitted by repo benchmarks."""

    metrics_context: MetricsContext | None = None
    max_raw_feedback_chars: int = 12_000

    def __post_init__(self) -> None:
        if self.max_raw_feedback_chars < 0:
            raise ValueError("max_raw_feedback_chars must be non-negative")

    def extract(
        self,
        program: Program,
        *,
        parent: Program | None = None,
        archive: Sequence[Program] = (),
    ) -> dict[str, Any]:
        return build_benchmark_evidence(
            program,
            parent=parent,
            archive=archive,
            metrics_context=self.metrics_context,
            max_raw_feedback_chars=self.max_raw_feedback_chars,
        )


def build_benchmark_evidence(
    program: Program,
    *,
    parent: Program | None = None,
    archive: Sequence[Program] = (),
    metrics_context: MetricsContext | None = None,
    max_raw_feedback_chars: int = 12_000,
) -> dict[str, Any]:
    """Build portable benchmark evidence for an archive curator.

    Benchmark-specific unit rows (tasks, cases, trials, or examples) are
    normalized when present. Aggregate-only benchmarks remain valid inputs.
    Comparisons never interpret a missing or projected unit as a failure.
    """

    if max_raw_feedback_chars < 0:
        raise ValueError("max_raw_feedback_chars must be non-negative")

    snapshot = _snapshot(program, metrics_context=metrics_context)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": snapshot["benchmark"],
        "aggregate_metrics": snapshot["aggregate_metrics"],
        "metric_metadata": snapshot["metric_metadata"],
        "scope": snapshot["scope"],
        "unit_results": snapshot["unit_results"],
        "resource_metrics": snapshot["resource_metrics"],
    }

    if parent is not None:
        parent_snapshot = _snapshot(parent, metrics_context=metrics_context)
        evidence["comparison_with_parent"] = _compare_with_parent(
            snapshot,
            parent_snapshot,
            metrics_context=metrics_context,
        )

    comparison_archive: list[Program] = []
    seen_archive_ids: set[str] = {program.id}
    for candidate in archive:
        if candidate.id in seen_archive_ids:
            continue
        seen_archive_ids.add(candidate.id)
        comparison_archive.append(candidate)
    if comparison_archive:
        archive_snapshots = [
            _snapshot(candidate, metrics_context=metrics_context)
            for candidate in comparison_archive
        ]
        evidence["distinction_from_archive"] = _compare_with_archive(
            snapshot,
            archive_snapshots,
            metrics_context=metrics_context,
        )

    if max_raw_feedback_chars:
        raw_feedback = _bounded_raw_feedback(
            snapshot["raw_feedback"],
            source=snapshot["raw_feedback_source"],
            max_chars=max_raw_feedback_chars,
        )
        if raw_feedback is not None:
            evidence["raw_feedback"] = raw_feedback

    return evidence


def _snapshot(
    program: Program,
    *,
    metrics_context: MetricsContext | None,
) -> dict[str, Any]:
    feedback = _as_dict(program.metadata.get("repo_benchmark_feedback"))
    structured, structured_source = _structured_feedback(feedback)
    aggregate_metrics = _aggregate_metrics(program, feedback, structured)
    unit_results, unit_source = _unit_results(structured)
    universe_ids = _universe_ids(structured)
    unit_results = _add_known_unknown_units(unit_results, universe_ids)
    scope = _scope(
        feedback,
        structured,
        unit_results,
        unit_source=unit_source,
        universe_ids=universe_ids,
    )
    benchmark = (
        structured.get("benchmark")
        or feedback.get("benchmark")
        or program.metadata.get("benchmark")
    )
    raw_feedback: Any = structured if structured else feedback
    raw_source = structured_source if structured else "repo_benchmark_feedback"
    return {
        "program_id": program.id,
        "benchmark": str(benchmark) if benchmark is not None else None,
        "aggregate_metrics": aggregate_metrics,
        "metric_metadata": _metric_metadata(
            aggregate_metrics, metrics_context=metrics_context
        ),
        "scope": scope,
        "unit_results": unit_results,
        "resource_metrics": _resource_metrics(aggregate_metrics, feedback, structured),
        "raw_feedback": raw_feedback,
        "raw_feedback_source": raw_source,
    }


def _structured_feedback(
    feedback: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    for key in ("structured_feedback", "structured_failure_feedback"):
        value = feedback.get(key)
        if isinstance(value, dict):
            return value, key
    if any(
        key in feedback
        for key in (
            "summary",
            "cases",
            "case_results",
            "trials",
            "examples",
            "failure_clusters",
        )
    ):
        return feedback, "repo_benchmark_feedback"
    return {}, "repo_benchmark_feedback"


def _aggregate_metrics(
    program: Program,
    feedback: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, float]:
    summary = _as_dict(structured.get("summary"))
    merged: dict[str, float] = {}
    for source in (
        summary.get("metrics"),
        structured.get("aggregate_metrics"),
        structured.get("metrics"),
        feedback.get("metrics"),
        program.metrics,
    ):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            numeric = _float_or_none(value)
            if numeric is not None:
                merged[str(key)] = numeric
    return dict(sorted(merged.items()))


def _metric_metadata(
    metrics: dict[str, float],
    *,
    metrics_context: MetricsContext | None,
) -> dict[str, dict[str, Any]]:
    if metrics_context is None:
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for key in metrics:
        spec = metrics_context.specs.get(key)
        if spec is None:
            continue
        metadata[key] = {
            "description": spec.description,
            "higher_is_better": spec.higher_is_better,
            "is_primary": spec.is_primary,
            "unit": spec.unit,
        }
    return metadata


def _unit_results(
    structured: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    source_key: str | None = None
    rows: list[dict[str, Any]] = []
    claimed_ids: set[str] = set()
    for key in _UNIT_COLLECTION_KEYS:
        candidate_rows = _collection_rows(structured.get(key))
        parsed = [row for row in candidate_rows if _unit_id(row) is not None]
        if not parsed:
            continue
        if source_key is None:
            source_key = key
        ids_in_source = {
            unit_id for row in parsed if (unit_id := _unit_id(row)) is not None
        }
        new_ids = ids_in_source - claimed_ids
        rows.extend(row for row in parsed if _unit_id(row) in new_ids)
        claimed_ids.update(ids_in_source)
    if not rows:
        return [], None

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        unit_id = _unit_id(row)
        if unit_id is None:
            continue
        if unit_id not in grouped:
            grouped[unit_id] = []
            order.append(unit_id)
        grouped[unit_id].append(_normalized_unit_observation(row))

    return [
        _merge_unit_observations(unit_id, grouped[unit_id]) for unit_id in order
    ], source_key


def _collection_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for unit_id, payload in value.items():
        if isinstance(payload, Mapping):
            row = dict(payload)
            row.setdefault("unit_id", str(unit_id))
        else:
            row = {"unit_id": str(unit_id), "score": payload}
        rows.append(row)
    return rows


def _unit_id(row: Mapping[str, Any]) -> str | None:
    for key in (
        "unit_id",
        "case_id",
        "case",
        "task_id",
        "task",
        "example_id",
        "example",
        "instance_id",
        "instance",
        "scenario_id",
        "scenario",
        "opponent_id",
        "opponent",
        "id",
        "name",
    ):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    trial = row.get("trial")
    if trial is None or not str(trial):
        return None
    return str(trial).rsplit("__", 1)[0]


def _normalized_unit_observation(row: dict[str, Any]) -> dict[str, Any]:
    score = _unit_score(row)
    status = _unit_status(row, score)
    observed = _unit_observed(row, status)
    repetitions = _repetitions(row, observed=observed)

    metrics: dict[str, float] = {}
    explicit_metrics = row.get("metrics")
    if isinstance(explicit_metrics, Mapping):
        for key, value in explicit_metrics.items():
            numeric = _float_or_none(value)
            if numeric is not None:
                metrics[str(key)] = numeric
    for key, value in row.items():
        if key in _UNIT_RESERVED_KEYS:
            continue
        numeric = _float_or_none(value)
        if numeric is not None:
            metrics[str(key)] = numeric

    return {
        "status": status,
        "score": score,
        "repetitions": repetitions,
        "observed": observed,
        "metrics": metrics,
    }


def _unit_score(row: Mapping[str, Any]) -> float | None:
    for key in ("score", "reward", "pass_rate", "success_rate", "fitness", "value"):
        numeric = _float_or_none(row.get(key))
        if numeric is not None:
            return numeric
    for key in ("passed", "success", "succeeded", "ok"):
        value = row.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
    status = str(row.get("status") or row.get("result") or "").strip().lower()
    if status in _SUCCESS_STATUSES:
        return 1.0
    if status in _FAILURE_STATUSES:
        return 0.0
    return None


def _unit_status(row: Mapping[str, Any], score: float | None) -> str:
    raw_status = str(row.get("status") or row.get("result") or "").strip().lower()
    if raw_status in _SUCCESS_STATUSES:
        return "success"
    if raw_status in _FAILURE_STATUSES:
        return "failure"
    if raw_status in _ERROR_STATUSES:
        return "error"
    if raw_status in {"partial", "partially_successful", "incomplete"}:
        return "partial"
    if raw_status in _UNKNOWN_STATUSES and raw_status:
        return "unknown"
    for key in ("passed", "success", "succeeded", "ok"):
        value = row.get(key)
        if isinstance(value, bool):
            return "success" if value else "failure"
    if score is None:
        return "unknown"
    if isclose(score, 1.0, abs_tol=1.0e-12):
        return "success"
    if isclose(score, 0.0, abs_tol=1.0e-12):
        return "failure"
    if 0.0 < score < 1.0:
        return "partial"
    return "unknown"


def _unit_observed(row: Mapping[str, Any], status: str) -> bool:
    explicit = row.get("observed")
    if isinstance(explicit, bool):
        return explicit
    for key in ("projected", "estimated", "inferred"):
        if row.get(key) is True:
            return False
    if status == "unknown":
        raw_status = str(row.get("status") or row.get("result") or "").strip().lower()
        if raw_status in _UNKNOWN_STATUSES:
            return False
    return True


def _repetitions(row: Mapping[str, Any], *, observed: bool) -> int:
    for key in ("repetitions", "n_trials", "trial_count", "attempts", "runs"):
        value = _int_or_none(row.get(key))
        if value is not None and value >= 0:
            return value
    return 1 if observed else 0


def _merge_unit_observations(
    unit_id: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_rows = [row for row in observations if row["observed"]]
    contributing = observed_rows or observations
    repetitions = sum(int(row["repetitions"]) for row in contributing)
    score = _weighted_mean(
        [(row["score"], max(1, int(row["repetitions"]))) for row in contributing]
    )
    metrics: dict[str, float] = {}
    metric_keys = {key for row in contributing for key in row["metrics"]}
    for key in sorted(metric_keys):
        value = _weighted_mean(
            [
                (row["metrics"].get(key), max(1, int(row["repetitions"])))
                for row in contributing
            ]
        )
        if value is not None:
            metrics[key] = value
    return {
        "unit_id": unit_id,
        "status": _merged_status(
            [str(row["status"]) for row in contributing], score=score
        ),
        "score": score,
        "repetitions": repetitions,
        "observed": bool(observed_rows),
        "metrics": metrics,
    }


def _weighted_mean(values: list[tuple[Any, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        numeric = _float_or_none(value)
        if numeric is None or weight <= 0:
            continue
        numerator += numeric * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _merged_status(statuses: list[str], *, score: float | None) -> str:
    known = {status for status in statuses if status != "unknown"}
    if len(known) == 1:
        return next(iter(known))
    if len(known) > 1:
        return "partial"
    if score is not None:
        if isclose(score, 1.0, abs_tol=1.0e-12):
            return "success"
        if isclose(score, 0.0, abs_tol=1.0e-12):
            return "failure"
        if 0.0 < score < 1.0:
            return "partial"
    return "unknown"


def _universe_ids(structured: dict[str, Any]) -> list[str]:
    scope = _as_dict(structured.get("scope"))
    summary = _as_dict(structured.get("summary"))
    candidates = (
        scope.get("universe_ids"),
        scope.get("all_units"),
        scope.get("units"),
        structured.get("universe_ids"),
        structured.get("all_units"),
        structured.get("all_cases"),
        structured.get("all_tasks"),
        structured.get("all_examples"),
        structured.get("selected_units"),
        structured.get("selected_cases"),
        structured.get("selected_tasks"),
        structured.get("selected_examples"),
        summary.get("universe_ids"),
        summary.get("all_units"),
        summary.get("all_cases"),
        summary.get("all_tasks"),
        summary.get("all_examples"),
        summary.get("selected_units"),
        summary.get("selected_cases"),
        summary.get("selected_tasks"),
        summary.get("selected_examples"),
    )
    for candidate in candidates:
        values = _id_list(candidate)
        if values:
            return values
    return []


def _id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return _dedupe_strings(
            [part.strip() for part in value.replace(",", " ").split() if part.strip()]
        )
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item_id = _unit_id(item)
            if item_id is not None:
                result.append(item_id)
        elif item is not None and str(item):
            result.append(str(item))
    return _dedupe_strings(result)


def _add_known_unknown_units(
    units: list[dict[str, Any]],
    universe_ids: list[str],
) -> list[dict[str, Any]]:
    if not universe_ids:
        return list(units)
    by_id = {str(unit["unit_id"]): unit for unit in units}
    expanded: list[dict[str, Any]] = []
    for unit_id in universe_ids:
        if unit_id in by_id:
            expanded.append(by_id[unit_id])
            continue
        expanded.append(
            {
                "unit_id": unit_id,
                "status": "unknown",
                "score": None,
                "repetitions": 0,
                "observed": False,
                "metrics": {},
            }
        )
    universe = set(universe_ids)
    expanded.extend(unit for unit in units if str(unit["unit_id"]) not in universe)
    return expanded


def _scope(
    feedback: dict[str, Any],
    structured: dict[str, Any],
    units: list[dict[str, Any]],
    *,
    unit_source: str | None,
    universe_ids: list[str],
) -> dict[str, Any]:
    explicit_scope = _as_dict(structured.get("scope"))
    summary = _as_dict(structured.get("summary"))
    evaluated = [unit for unit in units if unit["observed"]]
    evaluated_count = len(evaluated)
    universe_count = _universe_count(
        explicit_scope,
        structured,
        summary,
        universe_ids=universe_ids,
        evaluated_count=evaluated_count,
    )
    completeness = _completeness(
        feedback,
        structured,
        explicit_scope,
        summary,
        has_units=bool(units),
        evaluated_count=evaluated_count,
        universe_count=universe_count,
    )
    unit_type = explicit_scope.get("unit_type") or structured.get("unit_type")
    if not unit_type and unit_source:
        unit_type = {
            "cases": "case",
            "case_results": "case",
            "trials": "trial",
            "examples": "example",
        }[unit_source]
    unknown_count = (
        max(0, universe_count - evaluated_count) if universe_count is not None else None
    )
    return {
        "unit_type": str(unit_type) if unit_type else None,
        "completeness": completeness,
        "universe_count": universe_count,
        "evaluated_count": evaluated_count,
        "unknown_count": unknown_count,
        "repetitions": sum(int(unit["repetitions"]) for unit in evaluated),
        "source": unit_source,
    }


def _universe_count(
    scope: dict[str, Any],
    structured: dict[str, Any],
    summary: dict[str, Any],
    *,
    universe_ids: list[str],
    evaluated_count: int,
) -> int | None:
    if universe_ids:
        return max(len(universe_ids), evaluated_count)
    for source in (scope, structured, summary):
        for key in (
            "universe_count",
            "total_units",
            "total_cases",
            "total_tasks",
            "total_examples",
        ):
            value = _int_or_none(source.get(key))
            if value is not None and value >= 0:
                return max(value, evaluated_count)
    return None


def _completeness(
    feedback: dict[str, Any],
    structured: dict[str, Any],
    scope: dict[str, Any],
    summary: dict[str, Any],
    *,
    has_units: bool,
    evaluated_count: int,
    universe_count: int | None,
) -> str:
    # GigaEvo's staged-validation envelope is authoritative about whether the
    # attached benchmark rows came from a full run or only a projected subset.
    staged = _as_dict(feedback.get("staged_validation"))
    if staged.get("enabled") is True:
        if staged.get("promoted_to_full") is True:
            return "full"
        return "projected"

    for value in (
        scope.get("completeness"),
        structured.get("evaluation_completeness"),
        structured.get("completeness"),
        summary.get("evaluation_completeness"),
        summary.get("completeness"),
    ):
        normalized = str(value or "").strip().lower()
        if normalized in _COMPLETENESS_VALUES:
            return normalized

    if structured.get("projected") is True or summary.get("projected") is True:
        return "projected"
    if not has_units:
        return "aggregate_only"
    if universe_count is not None:
        return "full" if evaluated_count >= universe_count else "partial"
    if (
        feedback.get("benchmark_run_label") == "full"
        or structured.get("full_evaluation") is True
        or summary.get("full_evaluation") is True
    ):
        return "full"
    if structured.get("partial") is True or summary.get("partial") is True:
        return "partial"
    return "unknown"


def _resource_metrics(
    aggregate_metrics: dict[str, float],
    feedback: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(structured.get("summary"))
    resources: dict[str, Any] = {}
    for source in (summary, structured, feedback):
        for key in ("resource_metrics", "resources", "resource_usage", "usage"):
            value = source.get(key)
            if isinstance(value, Mapping):
                for resource_key, resource_value in value.items():
                    resources[str(resource_key)] = _json_native(resource_value)
    for source in (summary, structured, feedback, aggregate_metrics):
        for key, value in source.items():
            if not _looks_like_resource_metric(str(key)):
                continue
            numeric = _float_or_none(value)
            if numeric is not None:
                resources[str(key)] = numeric
    return dict(sorted(resources.items()))


def _looks_like_resource_metric(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    parts = {part for part in normalized.split("_") if part}
    return bool(parts & _RESOURCE_NAME_PARTS)


def _compare_with_parent(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    *,
    metrics_context: MetricsContext | None,
) -> dict[str, Any]:
    metric_deltas: dict[str, float] = {}
    directional_deltas: dict[str, float] = {}
    outcomes: dict[str, str] = {}
    candidate_metrics = candidate["aggregate_metrics"]
    parent_metrics = parent["aggregate_metrics"]
    for key in sorted(candidate_metrics.keys() & parent_metrics.keys()):
        delta = candidate_metrics[key] - parent_metrics[key]
        metric_deltas[key] = delta
        higher_is_better = _metric_direction(key, metrics_context)
        if higher_is_better is None:
            continue
        directional_delta = delta if higher_is_better else -delta
        directional_deltas[key] = directional_delta
        outcomes[key] = _delta_outcome(directional_delta)

    unit_comparison = _compare_units(
        candidate["unit_results"],
        parent["unit_results"],
    )
    return {
        "parent_program_id": parent["program_id"],
        "metric_deltas": metric_deltas,
        "directional_metric_deltas": directional_deltas,
        "metric_outcomes": outcomes,
        "improved_metrics": sorted(
            key for key, outcome in outcomes.items() if outcome == "improved"
        ),
        "regressed_metrics": sorted(
            key for key, outcome in outcomes.items() if outcome == "regressed"
        ),
        **unit_comparison,
        "comparison_basis": "observed_unit_results_only",
    }


def _compare_units(
    candidate_units: list[dict[str, Any]],
    baseline_units: list[dict[str, Any]],
) -> dict[str, list[str]]:
    candidate = _observed_unit_map(candidate_units)
    baseline = _observed_unit_map(baseline_units)
    improved: list[str] = []
    regressed: list[str] = []
    newly_successful: list[str] = []
    lost_success: list[str] = []

    for unit_id in sorted(candidate.keys() & baseline.keys()):
        current = candidate[unit_id]
        previous = baseline[unit_id]
        current_success = current["status"] == "success"
        previous_success = previous["status"] == "success"
        if current_success and not previous_success and previous["status"] != "unknown":
            newly_successful.append(unit_id)
        if previous_success and not current_success and current["status"] != "unknown":
            lost_success.append(unit_id)

        outcome = _unit_outcome(current, previous)
        if outcome == "improved":
            improved.append(unit_id)
        elif outcome == "regressed":
            regressed.append(unit_id)

    return {
        "improved_units": improved,
        "regressed_units": regressed,
        "newly_successful_units": newly_successful,
        "lost_success_units": lost_success,
    }


def _unit_outcome(current: dict[str, Any], previous: dict[str, Any]) -> str:
    current_score = _float_or_none(current.get("score"))
    previous_score = _float_or_none(previous.get("score"))
    if current_score is not None and previous_score is not None:
        return _delta_outcome(current_score - previous_score)
    ranks = {"failure": 0, "partial": 1, "success": 2}
    current_rank = ranks.get(str(current.get("status")))
    previous_rank = ranks.get(str(previous.get("status")))
    if current_rank is None or previous_rank is None:
        return "unchanged"
    return _delta_outcome(float(current_rank - previous_rank))


def _compare_with_archive(
    candidate: dict[str, Any],
    archive: list[dict[str, Any]],
    *,
    metrics_context: MetricsContext | None,
) -> dict[str, Any]:
    archive_units = [_observed_unit_map(item["unit_results"]) for item in archive]
    candidate_units = _observed_unit_map(candidate["unit_results"])
    unique_successes: list[str] = []
    best_known_units: list[str] = []

    for unit_id, unit in sorted(candidate_units.items()):
        if unit["status"] == "success" and not any(
            archived.get(unit_id, {}).get("status") == "success"
            for archived in archive_units
        ):
            unique_successes.append(unit_id)
        candidate_score = _float_or_none(unit.get("score"))
        archived_scores = [
            score
            for archived in archive_units
            if unit_id in archived
            for score in [_float_or_none(archived[unit_id].get("score"))]
            if score is not None
        ]
        if (
            candidate_score is not None
            and archived_scores
            and candidate_score > max(archived_scores)
            and not isclose(candidate_score, max(archived_scores), abs_tol=1.0e-12)
        ):
            best_known_units.append(unit_id)

    best_known_metrics: list[str] = []
    for key, value in candidate["aggregate_metrics"].items():
        higher_is_better = _metric_direction(key, metrics_context)
        if higher_is_better is None:
            continue
        archived_values = [
            item["aggregate_metrics"][key]
            for item in archive
            if key in item["aggregate_metrics"]
        ]
        if not archived_values:
            continue
        best = max(archived_values) if higher_is_better else min(archived_values)
        directional_delta = value - best if higher_is_better else best - value
        if directional_delta > 0 and not isclose(
            directional_delta, 0.0, abs_tol=1.0e-12
        ):
            best_known_metrics.append(key)

    return {
        "archive_program_count": len(archive),
        "archive_programs_with_unit_evidence": sum(
            bool(units) for units in archive_units
        ),
        "uniquely_successful_units": unique_successes,
        "best_known_units": best_known_units,
        "best_known_metrics": sorted(best_known_metrics),
        "comparison_basis": "observed_unit_results_only",
    }


def _observed_unit_map(
    units: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(unit["unit_id"]): unit
        for unit in units
        if unit.get("observed") is True and unit.get("status") != "unknown"
    }


def _metric_direction(
    key: str,
    metrics_context: MetricsContext | None,
) -> bool | None:
    if metrics_context is None:
        return None
    spec = metrics_context.specs.get(key)
    return spec.higher_is_better if spec is not None else None


def _delta_outcome(delta: float) -> str:
    if isclose(delta, 0.0, abs_tol=1.0e-12):
        return "unchanged"
    return "improved" if delta > 0 else "regressed"


def _bounded_raw_feedback(
    value: Any,
    *,
    source: str,
    max_chars: int,
) -> dict[str, Any] | None:
    if not value:
        return None
    native = _json_native(value)
    serialized = json.dumps(
        native,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) <= max_chars:
        content: Any = native
        truncated = False
    else:
        content = _clip_middle(serialized, max_chars)
        truncated = True
    return {
        "source": source,
        "truncated": truncated,
        "original_chars": len(serialized),
        "content": content,
    }


def _clip_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = "...<truncated benchmark feedback>..."
    if max_chars <= len(marker):
        return marker[:max_chars]
    remaining = max_chars - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + (text[-tail:] if tail else "")


def _json_native(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value) and value.is_integer():
        return int(value)
    return None


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
