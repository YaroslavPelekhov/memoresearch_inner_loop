from __future__ import annotations

import json
from math import isfinite
from typing import Any

from loguru import logger

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.strategies.base import EvolutionStrategy, StrategyMetrics
from gigaevo.evolution.strategies.models import BehaviorSpace, DynamicBehaviorSpace
from gigaevo.evolution.strategies.utils import weighted_sample_without_replacement
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.repo_harness.descriptors import (
    ACTIVE_ARCHIVE_ROLES,
    REPO_ARCHIVE_ROLES_METADATA_KEY,
    REPO_DESCRIPTORS_METADATA_KEY,
    ROLE_ACTIVE_PARENT,
    ROLE_FAILURE,
    ROLE_HALL_OF_FAME,
    ROLE_LOCAL_PARETO,
    ROLE_NOVELTY,
    ROLE_QUARANTINED,
    ROLE_SUPERSEDED,
    apply_archive_roles,
    descriptor_distance,
    extract_repo_descriptors,
    initial_archive_roles,
    is_active_archive_program,
    role_set,
)

_RUN_STATE_SELECTION_ROUND = "repo_archive:selection_round"
_RUN_STATE_CURATION_ROUND = "repo_archive:last_curation_round"


class RepoHarnessTopKArchiveStrategy(EvolutionStrategy):
    """Repo-harness strategy with cell -> top-K active program IDs.

    Redis stores only the active parent pool view. Every evaluated Program still
    remains in the normal ProgramStorage history, including candidates demoted to
    failure, superseded, or quarantined roles.
    """

    def __init__(
        self,
        *,
        program_storage: RedisProgramStorage,
        behavior_space: DynamicBehaviorSpace | BehaviorSpace,
        primary_key: str = "fitness",
        higher_is_better: bool = True,
        cell_size: int = 8,
        redis_prefix: str = "repo_topk_archive",
        novelty_threshold: float = 0.55,
        min_quality_floor: float | None = None,
        regression_tolerance: float = 1.0e-9,
        enable_quarantine: bool = True,
        hall_of_fame_size: int = 8,
        curation_interval: int = 1,
        stale_child_count_threshold: int = 8,
        selection_temperature: float = 1.0,
    ):
        if cell_size < 1:
            raise ValueError(f"cell_size must be >= 1, got {cell_size}")
        if hall_of_fame_size < 1:
            raise ValueError(
                f"hall_of_fame_size must be >= 1, got {hall_of_fame_size}"
            )
        if curation_interval < 1:
            raise ValueError(
                f"curation_interval must be >= 1, got {curation_interval}"
            )
        self.program_storage = program_storage
        self.state_manager = ProgramStateManager(program_storage)
        self.behavior_space = behavior_space
        self.primary_key = primary_key
        self.higher_is_better = higher_is_better
        self.cell_size = cell_size
        self.novelty_threshold = novelty_threshold
        self.min_quality_floor = min_quality_floor
        self.regression_tolerance = regression_tolerance
        self.enable_quarantine = enable_quarantine
        self.hall_of_fame_size = hall_of_fame_size
        self.curation_interval = curation_interval
        self.stale_child_count_threshold = stale_child_count_threshold
        self.selection_temperature = max(float(selection_temperature), 1.0e-9)

        self._cells_key = f"{redis_prefix}:cells"
        self._program_cell_key = f"{redis_prefix}:program_cell"
        self._selection_round = 0
        self._last_curation_round = -1

        logger.info(
            "Initialized repo top-K archive (cell_size={}, curation_interval={})",
            self.cell_size,
            self.curation_interval,
        )

    async def add(self, program: Program) -> bool:
        descriptors = self._ensure_descriptors(program)
        initial_roles, reasons = initial_archive_roles(
            program,
            descriptors,
            min_quality_floor=self.min_quality_floor,
            regression_tolerance=self.regression_tolerance,
            enable_quarantine=self.enable_quarantine,
        )
        if ROLE_QUARANTINED in initial_roles or ROLE_FAILURE in initial_roles:
            apply_archive_roles(
                program,
                initial_roles,
                reasons=reasons,
                source="repo_topk_archive_admission",
            )
            await self.state_manager.update_program(program)
            logger.info(
                "[RepoTopKArchive] {} inactive at admission roles={} reasons={}",
                program.short_id,
                initial_roles,
                reasons,
            )
            return False

        if isinstance(self.behavior_space, DynamicBehaviorSpace):
            if self.behavior_space.check_and_expand(program.metrics):
                await self.reindex_archive()

        cell = self.behavior_space.get_cell(program.metrics)
        field = self._field(cell)
        current = await self._cell_programs(field)
        reference_pool = await self._active_programs()
        novelty_score = self._novelty_score(descriptors, reference_pool)
        self._set_selection_descriptor(program, "novelty_score", novelty_score)

        ranked = self._rank_cell([*current, program])
        selected = ranked[: self.cell_size]
        demoted = [p for p in ranked[self.cell_size :] if p.id != program.id]
        selected_ids = [p.id for p in selected]
        accepted = program.id in selected_ids

        if accepted:
            await self._set_cell_ids(field, selected_ids)
            hall_ids = self._hall_of_fame_ids([*reference_pool, program])
            for selected_program in selected:
                roles = [ROLE_ACTIVE_PARENT, ROLE_LOCAL_PARETO]
                if selected_program.id in hall_ids:
                    roles.append(ROLE_HALL_OF_FAME)
                if self._program_novelty_score(selected_program) >= self.novelty_threshold:
                    roles.append(ROLE_NOVELTY)
                apply_archive_roles(
                    selected_program,
                    roles,
                    reasons=[
                        f"active in cell {field}",
                        f"cell_rank={selected.index(selected_program) + 1}",
                    ],
                    source="repo_topk_archive_admission",
                )
                await self.state_manager.update_program(selected_program)
            for old in demoted:
                await self._demote(
                    old,
                    [ROLE_SUPERSEDED],
                    [f"outside top-{self.cell_size} for cell {field}"],
                    discard=True,
                )
            logger.info(
                "[RepoTopKArchive] accepted {} in cell {} ({}/{}) roles={}",
                program.short_id,
                field,
                selected_ids.index(program.id) + 1,
                len(selected_ids),
                program.metadata.get(REPO_ARCHIVE_ROLES_METADATA_KEY),
            )
            return True

        apply_archive_roles(
            program,
            [ROLE_SUPERSEDED],
            reasons=[f"not in top-{self.cell_size} for cell {field}"],
            source="repo_topk_archive_admission",
        )
        await self.state_manager.update_program(program)
        logger.info(
            "[RepoTopKArchive] superseded {} in cell {}",
            program.short_id,
            field,
        )
        return False

    async def select_elites(self, total: int) -> list[Program]:
        if total <= 0:
            return []
        await self._maybe_curate()
        active = await self._active_programs()
        active = [program for program in active if is_active_archive_program(program)]
        if not active:
            return []
        ranked = sorted(active, key=self._selection_score, reverse=True)
        if len(ranked) <= total:
            selected = ranked
        else:
            min_score = min(self._selection_score(program) for program in ranked)
            weights = [
                max(1.0e-9, (self._selection_score(program) - min_score) + 1.0)
                ** (1.0 / self.selection_temperature)
                for program in ranked
            ]
            selected = weighted_sample_without_replacement(ranked, weights, total)
        self._selection_round += 1
        await self.program_storage.save_run_state(
            _RUN_STATE_SELECTION_ROUND, self._selection_round
        )
        logger.debug(
            "[RepoTopKArchive] selected {} / {} active parents",
            len(selected),
            len(active),
        )
        return selected

    async def get_program_ids(self) -> list[str]:
        cells = await self._cell_mapping()
        ids: set[str] = set()
        for cell_ids in cells.values():
            ids.update(cell_ids)
        return sorted(ids)

    async def remove_program_by_id(self, program_id: str) -> bool:
        field = await self._cell_for_program(program_id)
        if field is None:
            return False
        ids = await self._cell_ids(field)
        if program_id not in ids:
            return False
        ids = [pid for pid in ids if pid != program_id]
        await self._set_cell_ids(field, ids)
        program = await self.program_storage.get(program_id)
        if program is not None:
            await self._demote(
                program,
                [ROLE_SUPERSEDED],
                ["removed from repo top-K archive"],
                discard=True,
            )
        return True

    async def get_metrics(self) -> StrategyMetrics:
        cells = await self._cell_mapping()
        ids = {pid for cell_ids in cells.values() for pid in cell_ids}
        return StrategyMetrics(
            total_programs=len(ids),
            active_populations=max(1, len(cells)),
            strategy_specific_metrics={
                "repo_archive/cells": len(cells),
                "repo_archive/cell_size": self.cell_size,
                "repo_archive/selection_round": self._selection_round,
                "repo_archive/last_curation_round": self._last_curation_round,
            },
        )

    async def restore_state(self) -> None:
        selection_round = await self.program_storage.load_run_state(
            _RUN_STATE_SELECTION_ROUND
        )
        curation_round = await self.program_storage.load_run_state(
            _RUN_STATE_CURATION_ROUND
        )
        if selection_round is not None:
            self._selection_round = int(selection_round)
        if curation_round is not None:
            self._last_curation_round = int(curation_round)
        logger.info(
            "[RepoTopKArchive] restored selection_round={} last_curation_round={}",
            self._selection_round,
            self._last_curation_round,
        )

    async def reset_state(self) -> None:
        self._selection_round = 0
        self._last_curation_round = -1
        await self._clear_cells()
        await self.program_storage.save_run_state(_RUN_STATE_SELECTION_ROUND, 0)
        await self.program_storage.save_run_state(_RUN_STATE_CURATION_ROUND, -1)
        logger.info("[RepoTopKArchive] reset persisted archive state")

    async def reindex_archive(self) -> None:
        active = await self._active_programs()
        await self._clear_cells()
        placements: dict[str, list[Program]] = {}
        for program in active:
            descriptors = self._ensure_descriptors(program)
            roles, reasons = initial_archive_roles(
                program,
                descriptors,
                min_quality_floor=self.min_quality_floor,
                regression_tolerance=self.regression_tolerance,
                enable_quarantine=self.enable_quarantine,
            )
            if ROLE_FAILURE in roles or ROLE_QUARANTINED in roles:
                await self._demote(program, roles, reasons, discard=True)
                continue
            try:
                field = self._field(self.behavior_space.get_cell(program.metrics))
            except Exception as exc:
                await self._demote(
                    program,
                    [ROLE_SUPERSEDED],
                    [f"could not map behavior cell during reindex: {exc}"],
                    discard=True,
                )
                continue
            placements.setdefault(field, []).append(program)

        for field, programs in placements.items():
            ranked = self._rank_cell(programs)
            selected = ranked[: self.cell_size]
            await self._set_cell_ids(field, [p.id for p in selected])
            for idx, program in enumerate(selected, start=1):
                roles = [ROLE_ACTIVE_PARENT, ROLE_LOCAL_PARETO]
                if self._program_novelty_score(program) >= self.novelty_threshold:
                    roles.append(ROLE_NOVELTY)
                apply_archive_roles(
                    program,
                    roles,
                    reasons=[f"active in cell {field}", f"cell_rank={idx}"],
                    source="repo_topk_archive_reindex",
                )
                await self.state_manager.update_program(program)
            for program in ranked[self.cell_size :]:
                await self._demote(
                    program,
                    [ROLE_SUPERSEDED],
                    [f"outside top-{self.cell_size} for cell {field} during reindex"],
                    discard=True,
                )
        await self._mark_hall_of_fame()

    async def _maybe_curate(self) -> None:
        if self._selection_round - self._last_curation_round < self.curation_interval:
            return
        await self._curate_archive()
        self._last_curation_round = self._selection_round
        await self.program_storage.save_run_state(
            _RUN_STATE_CURATION_ROUND, self._last_curation_round
        )

    async def _curate_archive(self) -> None:
        cells = await self._cell_mapping()
        curated_cells = 0
        demoted = 0
        for field, ids in cells.items():
            programs = await self._programs_by_ids(ids)
            keep_candidates: list[Program] = []
            for program in programs:
                descriptors = self._ensure_descriptors(program)
                roles, reasons = initial_archive_roles(
                    program,
                    descriptors,
                    min_quality_floor=self.min_quality_floor,
                    regression_tolerance=self.regression_tolerance,
                    enable_quarantine=self.enable_quarantine,
                )
                if ROLE_QUARANTINED in roles or ROLE_FAILURE in roles:
                    await self._demote(program, roles, reasons, discard=True)
                    demoted += 1
                    continue
                keep_candidates.append(program)

            ranked = self._rank_cell(keep_candidates)
            selected, stale_demotions = self._split_stale_selected(ranked)
            await self._set_cell_ids(field, [p.id for p in selected])
            curated_cells += 1
            for idx, program in enumerate(selected, start=1):
                roles = [ROLE_ACTIVE_PARENT, ROLE_LOCAL_PARETO]
                if self._program_novelty_score(program) >= self.novelty_threshold:
                    roles.append(ROLE_NOVELTY)
                apply_archive_roles(
                    program,
                    roles,
                    reasons=[f"curated active in cell {field}", f"cell_rank={idx}"],
                    source="repo_topk_archive_curation",
                )
                await self.state_manager.update_program(program)
            for program in ranked[self.cell_size :]:
                await self._demote(
                    program,
                    [ROLE_SUPERSEDED],
                    [f"curation outside top-{self.cell_size} for cell {field}"],
                    discard=True,
                )
                demoted += 1
            for program in stale_demotions:
                await self._demote(
                    program,
                    [ROLE_SUPERSEDED],
                    [
                        "stale active parent demoted during curation",
                        f"child_count>={self.stale_child_count_threshold}",
                    ],
                    discard=True,
                )
                demoted += 1
        await self._mark_hall_of_fame()
        logger.info(
            "[RepoTopKArchive] curated {} cell(s), demoted {} program(s)",
            curated_cells,
            demoted,
        )

    def _ensure_descriptors(self, program: Program) -> dict[str, Any]:
        existing = program.metadata.get(REPO_DESCRIPTORS_METADATA_KEY)
        if isinstance(existing, dict):
            return existing
        descriptors = extract_repo_descriptors(program, primary_key=self.primary_key)
        program.set_metadata(REPO_DESCRIPTORS_METADATA_KEY, descriptors)
        return descriptors

    def _rank_cell(self, programs: list[Program]) -> list[Program]:
        candidates = [
            program
            for program in programs
            if not (role_set(program) & {ROLE_FAILURE, ROLE_QUARANTINED})
        ]
        ranked: list[Program] = []
        remaining = list(dict.fromkeys(candidates))
        while remaining:
            front = [
                program
                for program in remaining
                if not any(
                    other is not program and self._dominates(other, program)
                    for other in remaining
                )
            ]
            front.sort(key=self._archive_score, reverse=True)
            ranked.extend(front)
            front_ids = {program.id for program in front}
            remaining = [program for program in remaining if program.id not in front_ids]
        return ranked

    def _split_stale_selected(
        self, ranked: list[Program]
    ) -> tuple[list[Program], list[Program]]:
        selected = ranked[: self.cell_size]
        if len(selected) <= 1:
            return selected, []
        stale_demotions = [
            program
            for program in selected[1:]
            if self._is_stale(program) and ROLE_HALL_OF_FAME not in role_set(program)
        ]
        if not stale_demotions:
            return selected, []
        stale_ids = {program.id for program in stale_demotions}
        return [program for program in selected if program.id not in stale_ids], stale_demotions

    def _is_stale(self, program: Program) -> bool:
        descriptors = self._ensure_descriptors(program)
        lineage = descriptors.get("lineage", {})
        child_count = lineage.get("child_count")
        return (
            isinstance(child_count, int)
            and child_count >= self.stale_child_count_threshold
        )

    def _dominates(self, left: Program, right: Program) -> bool:
        left_values = self._objective_values(left)
        right_values = self._objective_values(right)
        return all(
            left_value >= right_value
            for left_value, right_value in zip(left_values, right_values)
        ) and any(
            left_value > right_value
            for left_value, right_value in zip(left_values, right_values)
        )

    def _objective_values(self, program: Program) -> tuple[float, ...]:
        descriptors = self._ensure_descriptors(program)
        metrics = descriptors.get("metrics", {})
        feedback = descriptors.get("feedback", {})
        code = descriptors.get("code", {})
        diff = code.get("diff", {})
        risk = descriptors.get("risk", {})
        primary = self._oriented_primary(program)
        validity = self._metric(program, "is_valid", default=1.0)
        diff_size = float(diff.get("line_delta") or 0.0)
        duration = float(feedback.get("duration_seconds") or 0.0)
        failures = float(feedback.get("failure_count") or 0.0)
        risk_score = float(risk.get("quarantine_score") or 0.0)
        primary_delta = float(metrics.get("primary_delta") or 0.0)
        return (
            primary,
            validity,
            primary_delta,
            -failures,
            -risk_score,
            -diff_size,
            -duration,
        )

    def _archive_score(self, program: Program) -> float:
        descriptors = self._ensure_descriptors(program)
        code = descriptors.get("code", {})
        diff = code.get("diff", {})
        risk = descriptors.get("risk", {})
        lineage = descriptors.get("lineage", {})
        novelty = self._program_novelty_score(program)
        return (
            self._oriented_primary(program)
            + 0.25 * novelty
            - 0.01 * float(lineage.get("child_count") or 0)
            - 0.001 * float(diff.get("line_delta") or 0)
            - 0.25 * float(risk.get("quarantine_score") or 0)
        )

    def _selection_score(self, program: Program) -> float:
        roles = role_set(program)
        bonus = 0.0
        if ROLE_HALL_OF_FAME in roles:
            bonus += 0.5
        if ROLE_NOVELTY in roles:
            bonus += 0.2
        return self._archive_score(program) + bonus

    def _oriented_primary(self, program: Program) -> float:
        value = self._metric(program, self.primary_key, default=0.0)
        return value if self.higher_is_better else -value

    @staticmethod
    def _metric(program: Program, key: str, *, default: float) -> float:
        value = program.metrics.get(key, default)
        if not isinstance(value, (int, float)):
            return default
        fval = float(value)
        return fval if isfinite(fval) else default

    def _novelty_score(
        self, descriptors: dict[str, Any], reference_pool: list[Program]
    ) -> float:
        references = [
            program.metadata.get(REPO_DESCRIPTORS_METADATA_KEY)
            for program in reference_pool
            if isinstance(program.metadata.get(REPO_DESCRIPTORS_METADATA_KEY), dict)
        ]
        if not references:
            return 1.0
        return min(descriptor_distance(descriptors, ref) for ref in references)

    @staticmethod
    def _set_selection_descriptor(program: Program, key: str, value: Any) -> None:
        descriptors = program.metadata.setdefault(REPO_DESCRIPTORS_METADATA_KEY, {})
        if isinstance(descriptors, dict):
            selection = descriptors.setdefault("selection", {})
            if isinstance(selection, dict):
                selection[key] = value

    def _program_novelty_score(self, program: Program) -> float:
        descriptors = self._ensure_descriptors(program)
        selection = descriptors.get("selection", {})
        if isinstance(selection, dict):
            value = selection.get("novelty_score")
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    async def _mark_hall_of_fame(self) -> None:
        active = await self._active_programs()
        if not active:
            return
        ranked = sorted(active, key=self._oriented_primary, reverse=True)
        hall_ids = {program.id for program in ranked[: self.hall_of_fame_size]}
        for program in ranked:
            roles = role_set(program)
            if not roles & ACTIVE_ARCHIVE_ROLES:
                continue
            new_roles = [role for role in program.metadata.get(REPO_ARCHIVE_ROLES_METADATA_KEY, [])]
            if program.id in hall_ids and ROLE_HALL_OF_FAME not in new_roles:
                new_roles.append(ROLE_HALL_OF_FAME)
            if program.id not in hall_ids and ROLE_HALL_OF_FAME in new_roles:
                new_roles = [role for role in new_roles if role != ROLE_HALL_OF_FAME]
            apply_archive_roles(
                program,
                new_roles or [ROLE_ACTIVE_PARENT, ROLE_LOCAL_PARETO],
                reasons=["hall-of-fame curation"],
                source="repo_topk_archive_hall_of_fame",
            )
            await self.state_manager.update_program(program)

    def _hall_of_fame_ids(self, programs: list[Program]) -> set[str]:
        ranked = sorted(programs, key=self._oriented_primary, reverse=True)
        return {program.id for program in ranked[: self.hall_of_fame_size]}

    async def _demote(
        self,
        program: Program,
        roles: list[str],
        reasons: list[str],
        *,
        discard: bool,
    ) -> None:
        apply_archive_roles(
            program,
            roles,
            reasons=reasons,
            source="repo_topk_archive_curation",
        )
        await self.state_manager.update_program(program)
        if discard and program.state == ProgramState.DONE:
            try:
                await self.state_manager.set_program_state(
                    program, ProgramState.DISCARDED
                )
            except ValueError:
                logger.debug(
                    "[RepoTopKArchive] could not discard {} from state {}",
                    program.short_id,
                    program.state,
                )

    async def _active_programs(self) -> list[Program]:
        ids = await self.get_program_ids()
        return await self._programs_by_ids(ids)

    async def _programs_by_ids(self, ids: list[str]) -> list[Program]:
        if not ids:
            return []
        programs = await self.program_storage.mget(ids)
        return [program for program in programs if program is not None]

    async def _cell_programs(self, field: str) -> list[Program]:
        return await self._programs_by_ids(await self._cell_ids(field))

    async def _cell_mapping(self) -> dict[str, list[str]]:
        async def _op(r):
            return await r.hgetall(self._cells_key)

        raw = await self.program_storage.with_redis("repo_archive:hgetall", _op) or {}
        mapping: dict[str, list[str]] = {}
        for field, text in raw.items():
            try:
                ids = json.loads(text)
            except Exception:
                ids = []
            if isinstance(ids, list):
                mapping[str(field)] = [str(pid) for pid in ids]
        return mapping

    async def _cell_ids(self, field: str) -> list[str]:
        async def _op(r):
            return await r.hget(self._cells_key, field)

        raw = await self.program_storage.with_redis("repo_archive:hget", _op)
        if not raw:
            return []
        try:
            ids = json.loads(raw)
        except Exception:
            return []
        return [str(pid) for pid in ids] if isinstance(ids, list) else []

    async def _cell_for_program(self, program_id: str) -> str | None:
        async def _op(r):
            return await r.hget(self._program_cell_key, program_id)

        value = await self.program_storage.with_redis("repo_archive:reverse", _op)
        return str(value) if value else None

    async def _set_cell_ids(self, field: str, ids: list[str]) -> None:
        ids = list(dict.fromkeys(ids))

        async def _op(r):
            old_raw = await r.hget(self._cells_key, field)
            old_ids: list[str] = []
            if old_raw:
                try:
                    old = json.loads(old_raw)
                    if isinstance(old, list):
                        old_ids = [str(pid) for pid in old]
                except Exception:
                    old_ids = []
            pipe = r.pipeline(transaction=False)
            if ids:
                pipe.hset(self._cells_key, field, json.dumps(ids))
            else:
                pipe.hdel(self._cells_key, field)
            for pid in set(old_ids) - set(ids):
                pipe.hdel(self._program_cell_key, pid)
            for pid in ids:
                pipe.hset(self._program_cell_key, pid, field)
            await pipe.execute()

        await self.program_storage.with_redis("repo_archive:set_cell", _op)

    async def _clear_cells(self) -> None:
        async def _op(r):
            pipe = r.pipeline(transaction=False)
            pipe.delete(self._cells_key)
            pipe.delete(self._program_cell_key)
            await pipe.execute()

        await self.program_storage.with_redis("repo_archive:clear", _op)

    @staticmethod
    def _field(cell: tuple[int, ...]) -> str:
        return ",".join(map(str, cell))
