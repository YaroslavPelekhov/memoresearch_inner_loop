from __future__ import annotations

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.repo_archive import RepoHarnessTopKArchiveStrategy
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.repo_harness.descriptors import (
    REPO_ARCHIVE_ROLES_METADATA_KEY,
    REPO_DESCRIPTORS_METADATA_KEY,
    ROLE_ACTIVE_PARENT,
    ROLE_FAILURE,
    ROLE_LOCAL_PARETO,
    ROLE_QUARANTINED,
    ROLE_SUPERSEDED,
    extract_repo_descriptors,
)


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Primary score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "is_valid": MetricSpec(
                description="Validity",
                is_primary=False,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _behavior_space() -> BehaviorSpace:
    return BehaviorSpace(
        bins={
            "fitness": LinearBinning(min_val=0.0, max_val=1.0, num_bins=1),
        }
    )


def _repo_program(
    fitness: float,
    *,
    changed_files: list[str],
    diff_stat: str = " 1 file changed, 10 insertions(+), 2 deletions(-)",
    failure_clusters: list[dict] | None = None,
    failed_cases: list[str] | None = None,
) -> Program:
    program = Program(code="{}", state=ProgramState.DONE)
    program.add_metrics({"fitness": fitness, "is_valid": 1.0})
    program.metadata["repo_candidate"] = {"changed_files": changed_files}
    program.metadata["repo_reflection"] = {
        "changed_files": changed_files,
        "diff_stat": diff_stat,
        "metrics_delta": {"fitness": 0.0},
        "reflection": "Changed parser behavior based on benchmark feedback.",
    }
    cases = [
        {"case_id": case_id, "passed": False, "reward": 0.0}
        for case_id in (failed_cases or [])
    ]
    program.metadata["repo_benchmark_feedback"] = {
        "returncode": 0,
        "duration_seconds": 1.5,
        "structured_feedback": {
            "benchmark": "repo-test",
            "failure_clusters": failure_clusters or [],
            "cases": cases,
        },
    }
    return program


def test_extract_repo_descriptors_from_existing_metadata():
    program = _repo_program(
        0.5,
        changed_files=["agent/loop.py"],
        diff_stat=" 2 files changed, 42 insertions(+), 8 deletions(-)",
        failure_clusters=[{"name": "parser", "count": 2}],
        failed_cases=["task-a"],
    )

    descriptors = extract_repo_descriptors(
        program,
        metrics_context=_metrics_context(),
    )

    assert descriptors["metrics"]["primary_key"] == "fitness"
    assert descriptors["metrics"]["ordered_vector"] == [0.5, 1.0]
    assert descriptors["code"]["modules"] == ["agent"]
    assert descriptors["code"]["diff"]["line_delta"] == 50
    assert descriptors["feedback"]["failure_cluster_labels"] == ["parser"]
    assert descriptors["feedback"]["failed_case_ids"] == ["task-a"]
    assert descriptors["semantic"]["reflection_hash"]
    assert descriptors["lineage"]["child_count"] == 0


async def test_repo_topk_archive_keeps_top_k_active_and_supersedes_overflow(
    fakeredis_storage,
):
    strategy = RepoHarnessTopKArchiveStrategy(
        program_storage=fakeredis_storage,
        behavior_space=_behavior_space(),
        primary_key="fitness",
        cell_size=2,
        redis_prefix="test_repo_topk",
    )
    weak = _repo_program(
        0.2,
        changed_files=["agent/a.py"],
        diff_stat=" 1 file changed, 200 insertions(+), 0 deletions(-)",
    )
    strong = _repo_program(
        0.8,
        changed_files=["agent/a.py"],
        diff_stat=" 1 file changed, 20 insertions(+), 0 deletions(-)",
    )
    compact = _repo_program(
        0.7,
        changed_files=["agent/b.py"],
        diff_stat=" 1 file changed, 5 insertions(+), 0 deletions(-)",
    )

    for program in (weak, strong, compact):
        await fakeredis_storage.add(program)

    assert await strategy.add(weak) is True
    assert await strategy.add(strong) is True
    assert await strategy.add(compact) is True

    active_ids = set(await strategy.get_program_ids())
    assert active_ids == {strong.id, compact.id}

    refreshed_weak = await fakeredis_storage.get(weak.id)
    assert refreshed_weak is not None
    assert refreshed_weak.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY] == [
        ROLE_SUPERSEDED
    ]

    selected = await strategy.select_elites(total=10)
    assert {program.id for program in selected} == {strong.id, compact.id}
    for program in selected:
        roles = set(program.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY])
        assert ROLE_ACTIVE_PARENT in roles
        assert ROLE_LOCAL_PARETO in roles


async def test_repo_topk_archive_reset_state_clears_persisted_cells(
    fakeredis_storage,
):
    strategy = RepoHarnessTopKArchiveStrategy(
        program_storage=fakeredis_storage,
        behavior_space=_behavior_space(),
        primary_key="fitness",
        cell_size=2,
        redis_prefix="test_repo_topk_reset",
    )
    program = _repo_program(0.8, changed_files=["agent/a.py"])
    await fakeredis_storage.add(program)
    assert await strategy.add(program) is True
    assert await strategy.get_program_ids() == [program.id]

    await fakeredis_storage.save_run_state("repo_archive:selection_round", 7)
    await fakeredis_storage.save_run_state("repo_archive:last_curation_round", 6)
    strategy._selection_round = 7
    strategy._last_curation_round = 6

    await strategy.reset_state()

    assert await strategy.get_program_ids() == []
    assert strategy._selection_round == 0
    assert strategy._last_curation_round == -1
    assert await fakeredis_storage.load_run_state("repo_archive:selection_round") == 0
    assert (
        await fakeredis_storage.load_run_state("repo_archive:last_curation_round")
        == -1
    )


async def test_repo_topk_archive_quarantines_protected_benchmark_changes(
    fakeredis_storage,
):
    strategy = RepoHarnessTopKArchiveStrategy(
        program_storage=fakeredis_storage,
        behavior_space=_behavior_space(),
        primary_key="fitness",
        cell_size=2,
        redis_prefix="test_repo_topk_quarantine",
    )
    program = _repo_program(0.9, changed_files=["bench.py"])
    program.metadata["repo_benchmark_feedback"][
        "stderr_tail"
    ] = "Agent execution timed out after 900.0 seconds"
    await fakeredis_storage.add(program)

    assert await strategy.add(program) is False
    active_ids = await strategy.get_program_ids()
    assert active_ids == []
    refreshed = await fakeredis_storage.get(program.id)
    assert refreshed is not None
    assert ROLE_QUARANTINED in refreshed.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY]


async def test_repo_topk_archive_can_disable_quarantine(fakeredis_storage):
    strategy = RepoHarnessTopKArchiveStrategy(
        program_storage=fakeredis_storage,
        behavior_space=_behavior_space(),
        primary_key="fitness",
        cell_size=2,
        redis_prefix="test_repo_topk_no_quarantine",
        enable_quarantine=False,
    )
    program = _repo_program(0.9, changed_files=["bench.py"])
    program.metadata["repo_benchmark_feedback"][
        "stderr_tail"
    ] = "Agent execution timed out after 900.0 seconds"
    await fakeredis_storage.add(program)

    assert await strategy.add(program) is True
    active_ids = await strategy.get_program_ids()
    assert active_ids == [program.id]
    refreshed = await fakeredis_storage.get(program.id)
    assert refreshed is not None
    assert ROLE_QUARANTINED not in refreshed.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY]
    assert ROLE_FAILURE not in refreshed.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY]
    assert ROLE_ACTIVE_PARENT in refreshed.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY]


async def test_repo_topk_archive_marks_invalid_candidate_failure(fakeredis_storage):
    strategy = RepoHarnessTopKArchiveStrategy(
        program_storage=fakeredis_storage,
        behavior_space=_behavior_space(),
        primary_key="fitness",
        cell_size=2,
        redis_prefix="test_repo_topk_failure",
    )
    program = _repo_program(0.9, changed_files=["agent/loop.py"])
    program.metrics["is_valid"] = 0.0
    await fakeredis_storage.add(program)

    assert await strategy.add(program) is False
    refreshed = await fakeredis_storage.get(program.id)
    assert refreshed is not None
    assert refreshed.metadata[REPO_ARCHIVE_ROLES_METADATA_KEY] == [ROLE_FAILURE]
    assert refreshed.metadata[REPO_DESCRIPTORS_METADATA_KEY]["metrics"][
        "primary_value"
    ] == 0.9
