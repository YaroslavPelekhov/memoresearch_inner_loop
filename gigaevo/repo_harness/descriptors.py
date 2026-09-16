from __future__ import annotations

import hashlib
import json
from math import isfinite, sqrt
from pathlib import PurePosixPath
import re
from typing import Any

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.repo_harness.manifest import RepoCandidateManifest

REPO_DESCRIPTORS_METADATA_KEY = "repo_descriptors"
REPO_ARCHIVE_STATUS_METADATA_KEY = "archive_status"
REPO_ARCHIVE_ROLE_METADATA_KEY = "archive_role"
REPO_ARCHIVE_ROLES_METADATA_KEY = "archive_roles"
REPO_ARCHIVE_STATUS_REASONS_METADATA_KEY = "archive_status_reasons"

ROLE_ACTIVE_PARENT = "active_parent"
ROLE_HALL_OF_FAME = "hall_of_fame"
ROLE_LOCAL_PARETO = "local_pareto"
ROLE_NOVELTY = "novelty"
ROLE_FAILURE = "failure"
ROLE_SUPERSEDED = "superseded"
ROLE_QUARANTINED = "quarantined"

ACTIVE_ARCHIVE_ROLES = frozenset(
    {
        ROLE_ACTIVE_PARENT,
        ROLE_HALL_OF_FAME,
        ROLE_LOCAL_PARETO,
        ROLE_NOVELTY,
    }
)
INACTIVE_ARCHIVE_ROLES = frozenset(
    {
        ROLE_FAILURE,
        ROLE_SUPERSEDED,
        ROLE_QUARANTINED,
    }
)

_BENCHMARK_PATH_RE = re.compile(
    r"(^|/)(bench|benchmark|benchmarks|eval|evaluator|metrics?|sandbox|tests?)(/|\.|$)",
    re.IGNORECASE,
)
_DEPENDENCY_PATHS = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.toml",
    "Cargo.lock",
}


def extract_repo_descriptors(
    program: Program,
    *,
    metrics_context: MetricsContext | None = None,
    primary_key: str | None = None,
) -> dict[str, Any]:
    """Extract cheap, JSON-native repo-harness descriptors from existing metadata."""

    feedback = _as_dict(program.metadata.get("repo_benchmark_feedback"))
    reflection = _as_dict(program.metadata.get("repo_reflection"))
    candidate = _candidate_payload(program)
    structured = _structured_feedback(feedback)
    changed_files = _changed_files(program, candidate, reflection)
    diff_stat_text = str(reflection.get("diff_stat") or candidate.get("diff_stat") or "")
    diff = _parse_diff_stat(diff_stat_text)
    failure = _failure_descriptors(feedback, structured)
    semantic = _semantic_descriptors(reflection)
    metrics = _metrics_descriptors(
        program, metrics_context=metrics_context, primary_key=primary_key
    )
    code = {
        "changed_files": changed_files,
        "modules": _modules(changed_files),
        "file_count": len(changed_files),
        "extensions": sorted(
            {
                PurePosixPath(path).suffix.lstrip(".")
                for path in changed_files
                if PurePosixPath(path).suffix
            }
        ),
        "diff_stat_text": diff_stat_text,
        "diff": diff,
        "risk_flags": _code_risk_flags(changed_files, diff),
    }
    descriptors = {
        "schema_version": 1,
        "metrics": metrics,
        "code": code,
        "feedback": failure,
        "semantic": semantic,
        "lineage": _lineage_descriptors(program),
    }
    descriptors["risk"] = _risk_descriptors(descriptors)
    descriptors["signature"] = _signature(descriptors)
    return descriptors


def initial_archive_roles(
    program: Program,
    descriptors: dict[str, Any],
    *,
    validity_key: str = VALIDITY_KEY,
    min_quality_floor: float | None = None,
    regression_tolerance: float = 1.0e-9,
    enable_quarantine: bool = True,
) -> tuple[list[str], list[str]]:
    """Classify candidate before archive-local competition is known."""

    roles: list[str] = []
    reasons: list[str] = []
    metrics = descriptors.get("metrics", {})
    risk = descriptors.get("risk", {})
    feedback = descriptors.get("feedback", {})
    validity = _float_or_none(program.metrics.get(validity_key))

    if enable_quarantine and risk.get("quarantine_score", 0.0) > 0:
        roles.append(ROLE_QUARANTINED)
        reasons.extend(str(flag) for flag in risk.get("quarantine_flags", []))

    if validity is None or validity <= 0:
        roles.append(ROLE_FAILURE)
        reasons.append(f"{validity_key}={validity}")

    returncode = feedback.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        roles.append(ROLE_FAILURE)
        reasons.append(f"benchmark_returncode={returncode}")

    primary_value = _float_or_none(metrics.get("primary_value"))
    if min_quality_floor is not None and primary_value is not None:
        if primary_value < float(min_quality_floor):
            roles.append(ROLE_FAILURE)
            reasons.append(
                f"primary_value {primary_value:.6g} below floor {float(min_quality_floor):.6g}"
            )

    primary_delta = _float_or_none(metrics.get("primary_delta"))
    if primary_delta is not None and primary_delta < -abs(regression_tolerance):
        roles.append(ROLE_FAILURE)
        reasons.append(f"primary_delta={primary_delta:.6g}")

    if enable_quarantine and feedback.get("timeout"):
        roles.append(ROLE_FAILURE)
        reasons.append("benchmark_timeout")

    if not roles:
        roles.append(ROLE_ACTIVE_PARENT)
        reasons.append("passed initial repo archive checks")

    return _dedupe_roles(roles), _dedupe_strings(reasons)


def apply_archive_roles(
    program: Program,
    roles: list[str],
    *,
    reasons: list[str] | None = None,
    source: str,
) -> None:
    roles = _dedupe_roles(roles)
    if not roles:
        roles = [ROLE_ACTIVE_PARENT]
    program.set_metadata(REPO_ARCHIVE_ROLES_METADATA_KEY, roles)
    program.set_metadata(REPO_ARCHIVE_STATUS_METADATA_KEY, roles[0])
    program.set_metadata(REPO_ARCHIVE_ROLE_METADATA_KEY, roles[0])
    program.set_metadata(
        REPO_ARCHIVE_STATUS_REASONS_METADATA_KEY,
        {
            "source": source,
            "items": _dedupe_strings(reasons or []),
        },
    )


def descriptor_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Cheap bounded-ish distance over behavior, code, failure, and semantic descriptors."""

    code_l = left.get("code", {})
    code_r = right.get("code", {})
    fb_l = left.get("feedback", {})
    fb_r = right.get("feedback", {})
    sem_l = left.get("semantic", {})
    sem_r = right.get("semantic", {})
    metrics_l = left.get("metrics", {})
    metrics_r = right.get("metrics", {})

    distances = [
        _jaccard_distance(code_l.get("modules"), code_r.get("modules")),
        _jaccard_distance(code_l.get("changed_files"), code_r.get("changed_files")),
        _jaccard_distance(
            fb_l.get("failure_cluster_labels"), fb_r.get("failure_cluster_labels")
        ),
        _jaccard_distance(fb_l.get("failed_case_ids"), fb_r.get("failed_case_ids")),
        0.0
        if sem_l.get("reflection_hash")
        and sem_l.get("reflection_hash") == sem_r.get("reflection_hash")
        else 1.0,
        _metric_vector_distance(
            metrics_l.get("ordered_vector"), metrics_r.get("ordered_vector")
        ),
    ]
    return sum(distances) / len(distances)


def role_set(program: Program) -> set[str]:
    roles = program.metadata.get(REPO_ARCHIVE_ROLES_METADATA_KEY)
    if isinstance(roles, list):
        return {str(role) for role in roles}
    role = program.metadata.get(REPO_ARCHIVE_STATUS_METADATA_KEY) or program.metadata.get(
        REPO_ARCHIVE_ROLE_METADATA_KEY
    )
    return {str(role)} if role else set()


def is_active_archive_program(program: Program) -> bool:
    roles = role_set(program)
    return bool(roles & ACTIVE_ARCHIVE_ROLES) and not bool(roles & INACTIVE_ARCHIVE_ROLES)


def _candidate_payload(program: Program) -> dict[str, Any]:
    payload = _as_dict(program.metadata.get("repo_candidate"))
    if payload:
        return payload
    try:
        return RepoCandidateManifest.from_program(program).model_dump()
    except Exception:
        return {}


def _metrics_descriptors(
    program: Program,
    *,
    metrics_context: MetricsContext | None,
    primary_key: str | None,
) -> dict[str, Any]:
    metrics = {
        key: float(value)
        for key, value in program.metrics.items()
        if isinstance(value, (int, float)) and isfinite(float(value))
    }
    ordered_keys = sorted(metrics)
    resolved_primary = primary_key
    if resolved_primary is None and metrics_context is not None:
        try:
            resolved_primary = metrics_context.get_primary_key()
        except Exception:
            resolved_primary = None
    if resolved_primary is None and "fitness" in metrics:
        resolved_primary = "fitness"
    reflection = _as_dict(program.metadata.get("repo_reflection"))
    delta = _as_dict(reflection.get("metrics_delta")) or _as_dict(
        reflection.get("delta")
    )
    primary_delta = None
    if resolved_primary and resolved_primary in delta:
        primary_delta = _float_or_none(delta[resolved_primary])
    return {
        "primary_key": resolved_primary,
        "primary_value": metrics.get(resolved_primary) if resolved_primary else None,
        "primary_delta": primary_delta,
        "values": metrics,
        "ordered_keys": ordered_keys,
        "ordered_vector": [metrics[key] for key in ordered_keys],
        "delta": {
            key: float(value)
            for key, value in delta.items()
            if isinstance(value, (int, float)) and isfinite(float(value))
        },
    }


def _changed_files(
    program: Program, candidate: dict[str, Any], reflection: dict[str, Any]
) -> list[str]:
    sources = (
        candidate.get("changed_files"),
        reflection.get("changed_files"),
        program.metadata.get("changed_files"),
        program.metadata.get("previous_changed_files"),
    )
    for value in sources:
        if isinstance(value, list):
            return sorted({str(path) for path in value if str(path)})
    return []


def _modules(paths: list[str]) -> list[str]:
    modules: set[str] = set()
    for path in paths:
        pure = PurePosixPath(path)
        parts = pure.parts
        if len(parts) > 1:
            modules.add(parts[0])
        elif parts:
            modules.add(pure.stem or parts[0])
    return sorted(modules)


def _parse_diff_stat(text: str) -> dict[str, int]:
    files = _first_int(r"(\d+)\s+files?\s+changed", text)
    insertions = _first_int(r"(\d+)\s+insertions?\(\+\)", text)
    deletions = _first_int(r"(\d+)\s+deletions?\(-\)", text)
    return {
        "files_changed": files,
        "insertions": insertions,
        "deletions": deletions,
        "line_delta": insertions + deletions,
    }


def _first_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def _failure_descriptors(
    feedback: dict[str, Any], structured: dict[str, Any]
) -> dict[str, Any]:
    failed_case_ids: set[str] = set()
    cluster_labels: set[str] = set()

    for item in _list_of_dicts(structured.get("failure_clusters")):
        label = (
            item.get("name")
            or item.get("label")
            or item.get("cluster")
            or item.get("type")
            or item.get("id")
        )
        if label is not None:
            cluster_labels.add(str(label))

    for key in ("cases", "case_results", "trials", "examples"):
        for item in _list_of_dicts(structured.get(key)):
            case_id = (
                item.get("case_id")
                or item.get("case")
                or item.get("task_id")
                or item.get("task")
                or item.get("id")
                or item.get("name")
            )
            if case_id is None:
                continue
            if _case_failed(item):
                failed_case_ids.add(str(case_id))

    failure_count = _float_or_none(structured.get("failure_count"))
    if failure_count is None:
        failure_count = float(len(failed_case_ids))

    stderr_tail = str(feedback.get("stderr_tail") or "")
    return {
        "benchmark": structured.get("benchmark") or feedback.get("benchmark"),
        "returncode": feedback.get("returncode"),
        "duration_seconds": _float_or_none(feedback.get("duration_seconds")),
        "failure_cluster_labels": sorted(cluster_labels),
        "failed_case_ids": sorted(failed_case_ids),
        "failure_count": failure_count,
        "timeout": "timed out" in stderr_tail.lower(),
    }


def _structured_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    structured = feedback.get("structured_feedback") or feedback.get(
        "structured_failure_feedback"
    )
    if isinstance(structured, dict):
        return structured
    if any(key in feedback for key in ("summary", "cases", "failure_clusters")):
        return feedback
    return {}


def _semantic_descriptors(reflection: dict[str, Any]) -> dict[str, Any]:
    text = str(reflection.get("reflection") or "")
    summary = _clip(text.strip(), 1600)
    digest = hashlib.sha1(summary.encode("utf-8")).hexdigest() if summary else None
    return {
        "reflection_summary": summary,
        "reflection_hash": digest,
        "changed_files": reflection.get("changed_files"),
    }


def _lineage_descriptors(program: Program) -> dict[str, Any]:
    return {
        "generation": program.lineage.generation,
        "parent_count": program.lineage.parent_count,
        "child_count": program.lineage.child_count,
        "parents": list(program.lineage.parents),
        "children": list(program.lineage.children),
        "mutation": program.lineage.mutation,
    }


def _risk_descriptors(descriptors: dict[str, Any]) -> dict[str, Any]:
    code = descriptors.get("code", {})
    feedback = descriptors.get("feedback", {})
    risk_flags = list(code.get("risk_flags") or [])
    if feedback.get("timeout"):
        risk_flags.append("benchmark_timeout")
    if any("flak" in str(label).lower() for label in feedback.get("failure_cluster_labels") or []):
        risk_flags.append("flaky_failure_cluster")
    if (feedback.get("duration_seconds") or 0.0) > 0:
        # Placeholder for future per-problem timeout normalization.
        pass
    quarantine_flags = [
        flag
        for flag in risk_flags
        if flag.startswith("changed_protected_")
        or flag in {"large_diff", "benchmark_timeout"}
    ]
    return {
        "risk_flags": _dedupe_strings(risk_flags),
        "quarantine_flags": _dedupe_strings(quarantine_flags),
        "quarantine_score": float(len(quarantine_flags)),
    }


def _code_risk_flags(changed_files: list[str], diff: dict[str, int]) -> list[str]:
    flags: list[str] = []
    for path in changed_files:
        normalized = path.replace("\\", "/")
        name = PurePosixPath(normalized).name
        if _BENCHMARK_PATH_RE.search(normalized):
            flags.append("changed_protected_benchmark_or_evaluator")
        if name in _DEPENDENCY_PATHS:
            flags.append("dependency_change")
    if diff.get("line_delta", 0) > 2000 or diff.get("files_changed", 0) > 50:
        flags.append("large_diff")
    return _dedupe_strings(flags)


def _signature(descriptors: dict[str, Any]) -> dict[str, Any]:
    code = descriptors.get("code", {})
    feedback = descriptors.get("feedback", {})
    semantic = descriptors.get("semantic", {})
    payload = {
        "modules": code.get("modules") or [],
        "failure_cluster_labels": feedback.get("failure_cluster_labels") or [],
        "failed_case_ids": feedback.get("failed_case_ids") or [],
        "reflection_hash": semantic.get("reflection_hash"),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return {"sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(), **payload}


def _case_failed(item: dict[str, Any]) -> bool:
    for key in ("passed", "success", "ok"):
        if key in item:
            return not bool(item[key])
    status = str(item.get("status") or item.get("result") or "").lower()
    if status:
        return status in {"failed", "fail", "error", "timeout", "crashed"}
    score = _float_or_none(
        item.get("score", item.get("reward", item.get("fitness", None)))
    )
    return score is not None and score <= 0.0


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    fval = float(value)
    return fval if isfinite(fval) else None


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _jaccard_distance(left: Any, right: Any) -> float:
    left_set = {str(item) for item in left} if isinstance(left, list) else set()
    right_set = {str(item) for item in right} if isinstance(right, list) else set()
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - (len(left_set & right_set) / len(union))


def _metric_vector_distance(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return 0.0
    n = min(len(left), len(right))
    diffs = []
    for idx in range(n):
        lval = _float_or_none(left[idx])
        rval = _float_or_none(right[idx])
        if lval is None or rval is None:
            continue
        denom = max(abs(lval), abs(rval), 1.0)
        diffs.append(((lval - rval) / denom) ** 2)
    if not diffs:
        return 0.0
    return min(1.0, sqrt(sum(diffs) / len(diffs)))


def _dedupe_roles(roles: list[str]) -> list[str]:
    return _dedupe_strings(roles)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
