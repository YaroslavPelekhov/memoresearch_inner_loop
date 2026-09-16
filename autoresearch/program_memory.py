"""Project every GigaEvo execution into a human-verifiable ProgramCard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gigaevo.memory.shared_memory.models import ProgramCard
from gigaevo.programs.program import Program
from gigaevo.repo_harness.manifest import RepoCandidateManifest


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _number_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, (int, float))
    }


def _manifest(
    program: Program, source_repo: str | Path | None
) -> RepoCandidateManifest | None:
    try:
        return RepoCandidateManifest.from_program(
            program,
            default_repo_path=Path(source_repo).resolve() if source_repo else None,
        )
    except Exception:
        return None


def _metric_deltas(
    program: Program, parent: Program | None, reflection: dict[str, Any]
) -> dict[str, float]:
    reflected = _number_map(
        reflection.get("metrics_delta") or reflection.get("metric_deltas")
    )
    if reflected:
        return reflected
    if parent is None:
        return {}
    return {
        key: float(value) - float(parent.metrics[key])
        for key, value in program.metrics.items()
        if isinstance(value, (int, float))
        and isinstance(parent.metrics.get(key), (int, float))
    }


def _verdict(
    program: Program,
    deltas: dict[str, float],
    reflection: dict[str, Any],
    primary_key: str,
) -> str:
    validity = program.metrics.get("is_valid")
    if isinstance(validity, (int, float)) and validity <= 0:
        return "invalid"
    explicit = reflection.get("verdict")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    delta = deltas.get(primary_key)
    if delta is None:
        return "baseline" if not program.lineage.parents else "unknown"
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "neutral"


def program_to_card(
    program: Program,
    *,
    parent: Program | None,
    primary_key: str,
    task_description: str,
    source_repo: str | Path | None = None,
) -> ProgramCard:
    """Create an exact evidence card without filtering on fitness or validity."""

    reflection = _dict(program.metadata.get("repo_reflection"))
    manifest = _manifest(program, source_repo)
    deltas = _metric_deltas(program, parent, reflection)
    verdict = _verdict(program, deltas, reflection, primary_key)
    fitness = program.metrics.get(primary_key)
    commit = manifest.commit if manifest is not None else ""
    parent_commit = manifest.parent_commit if manifest is not None else ""
    changed_files = manifest.changed_files if manifest is not None else []
    duration = program.metrics.get("wall_time_seconds")
    approved_idea_id = program.metadata.get("approved_idea_id")
    if not approved_idea_id:
        mutation_output = _dict(program.metadata.get("mutation_output"))
        approved_idea_id = mutation_output.get("approved_idea_id")
    artifact_refs = _dict(program.metadata.get("repo_evaluation_artifacts"))
    benchmark_feedback = _dict(program.metadata.get("repo_benchmark_feedback"))
    structured_feedback = benchmark_feedback.get("structured_feedback")
    if isinstance(structured_feedback, dict):
        artifact_refs["benchmark"] = structured_feedback
        if structured_feedback.get("status") == "smoke_complete":
            verdict = "smoke"
        elif structured_feedback.get("status") == "diagnostic_complete":
            verdict = "diagnostic"

    description = (
        f"Generation {program.lineage.generation} execution was {verdict}; "
        f"{primary_key}={fitness!r}."
    )
    return ProgramCard(
        id=f"program-{program.id}",
        program_id=program.id,
        task_description=task_description,
        description=description,
        fitness=float(fitness) if isinstance(fitness, (int, float)) else None,
        code=program.code,
        parent_program_ids=list(program.lineage.parents),
        generation=program.lineage.generation,
        metrics=_number_map(program.metrics),
        metric_deltas=deltas,
        verdict=verdict,
        commit=commit,
        parent_commit=parent_commit or "",
        changed_files=list(changed_files),
        reflection=reflection,
        artifact_refs=artifact_refs,
        duration_seconds=(
            float(duration) if isinstance(duration, (int, float)) else None
        ),
        approved_idea_id=(
            str(approved_idea_id) if approved_idea_id is not None else ""
        ),
        keywords=["execution", verdict],
        strategy="execution-grounded",
    )
