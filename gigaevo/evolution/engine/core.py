from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import math
import time
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.hooks import NullPostRunHook, PostRunHook
from gigaevo.evolution.engine.metrics import EngineMetrics
from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.evolution.engine.stopper import EvolutionStopper, StopContext
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.mutation.mutation_operator import (
    LLMMutationOperator,
)
from gigaevo.evolution.strategies.base import (
    BatchAdmissionResult,
    EvolutionStrategy,
)
from gigaevo.llm.bandit import BanditModelRouter, MutationOutcome
from gigaevo.monitoring.emit import emit as _emit_event
from gigaevo.monitoring.events import GenerationBoundary
from gigaevo.programs.metrics.context import DEFAULT_DECIMALS, VALIDITY_KEY
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.metrics_tracker import MetricsTracker
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from typing import Any

# Redis run-state field names (used for resume persistence)
_RUN_STATE_TOTAL_GENERATIONS = "engine:total_generations"
_RUN_STATE_PROGRAMS_PROCESSED = "engine:programs_processed"
_RUN_STATE_COMPLETION_REASON = "engine:completion_reason"

# A mutation operator that repeatedly produces no persisted candidates cannot make
# progress.  Bound retries so infrastructure failures (for example expired LLM
# credentials) do not spin generations indefinitely.
_MAX_CONSECUTIVE_EMPTY_MUTATION_GENERATIONS = 3


class EvolutionStopRequested(RuntimeError):
    """Signal that a hook reached a valid terminal condition for this run."""


class EvolutionEngine:
    """
      1) Wait until no DAGs are running (idle)
      2) Select elites & create mutants
      3) Wait for mutants' DAGs to finish (idle again)
      4) Ingest completed mutants
      5) Refresh all archive programs (DONE -> QUEUED)
      6) Wait for refresh DAGs to finish (idle)
    All state writes go through ProgramStateManager; storage is read-oriented here.
    """

    def __init__(
        self,
        storage: ProgramStorage,
        strategy: EvolutionStrategy,
        mutation_operator: MutationOperator,
        config: EngineConfig,
        writer: LogWriter,
        metrics_tracker: MetricsTracker,
        pre_step_hook: Callable[[], Awaitable[None]] | None = None,
        post_run_hook: PostRunHook | None = None,
        post_step_hook: Callable[[], Awaitable[None]] | None = None,
    ):
        self.storage = storage
        self.strategy = strategy
        self.mutation_operator = mutation_operator
        self.config = config
        self._writer = writer.bind(path=["evolution_engine"])

        self._running = False
        self._paused = False
        self._resume_ingest_pending = False
        self._last_pending_dags_counts: tuple[int, int] | None = None

        self._task: asyncio.Task | None = None
        self._metrics_collector_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # ETA tracking — set at the start of run()
        self._run_start_time: float | None = None
        self._run_start_gen: int = 0

        # Archive stagnation tracking
        self._prev_archive_size: int = 0
        self._stagnant_gens: int = 0
        self._best_fitness_for_stopping: float | None = None
        self._consecutive_empty_mutation_generations: int = 0

        self.metrics = EngineMetrics()
        self.state = ProgramStateManager(self.storage)
        self._metrics_tracker = metrics_tracker
        self._pre_step_hook = pre_step_hook
        self._post_step_hook = post_step_hook
        self._post_run_hook = post_run_hook or NullPostRunHook()

        logger.info(
            "[EvolutionEngine] Init | strategy={}, acceptor={}, stopper={}",
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
            type(self.stopper).__name__,
        )

    def start(self) -> None:
        """Start the evolution engine in a background task."""
        if self._task and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._task = asyncio.create_task(self.run(), name="evolution-engine")
        self._metrics_tracker.start(self._loop)

        async def _collect_metrics() -> dict[str, Any]:
            out = self.metrics.model_dump(mode="json")
            strategy_metrics = await self.strategy.get_metrics()
            if strategy_metrics:
                out.update(strategy_metrics.to_dict())
            if isinstance(self.mutation_operator, LLMMutationOperator) and isinstance(
                self.mutation_operator.llm_wrapper, BanditModelRouter
            ):
                out["bandit"] = self.mutation_operator.llm_wrapper.get_bandit_stats()
            return out

        self._metrics_collector_task = start_metrics_collector(
            writer=self._writer,
            collect_fn=_collect_metrics,
            interval=self.config.metrics_collection_interval,
            stop_flag=lambda: not self._running,
            task_name="evolution-metrics-collector",
        )
        logger.info("[EvolutionEngine] Task started")

    async def stop(self) -> None:
        """Stop the evolution engine and await task completion."""
        self._running = False
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self._metrics_collector_task:
            self._metrics_collector_task.cancel()
            self._metrics_collector_task = None

        if self._metrics_tracker:
            await self._metrics_tracker.stop()

        await self.storage.close()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_running(self) -> bool:
        return self._running

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    async def run(self) -> None:
        logger.info(
            "[EvolutionEngine] Start | stopper={} strategy={} acceptor={}"
            " | max_elites={} max_mutations={} loop_interval={}s",
            type(self.config.stopper).__name__,
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
            self.config.max_elites_per_generation,
            self.config.max_mutations_per_generation,
            self.config.loop_interval,
        )
        self._running = True
        self._run_start_time = time.monotonic()
        self._run_start_gen = self.metrics.total_generations
        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(self.config.loop_interval)
                    continue

                stop_decision = self.stopper.should_stop(self._build_stop_context())
                if stop_decision.stop:
                    logger.info(
                        "[EvolutionEngine] Stop: {}",
                        stop_decision.reason,
                    )
                    await self.storage.save_run_state(
                        _RUN_STATE_COMPLETION_REASON, stop_decision.reason
                    )
                    break

                try:
                    await self.step()
                except asyncio.CancelledError:
                    # Propagate so shutdown stays clean.
                    raise
                except EvolutionStopRequested as exc:
                    reason = str(exc) or type(exc).__name__
                    logger.info("[EvolutionEngine] Terminal stop: {}", reason)
                    await self.storage.save_run_state(
                        _RUN_STATE_COMPLETION_REASON, reason
                    )
                    break
                except Exception as e:
                    # Don’t crash the engine on a single bad step; just log and continue.
                    logger.exception("[EvolutionEngine] step() failed: {}", e)

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Task is being cancelled during shutdown.
            logger.debug("[EvolutionEngine] run() cancelled")
            raise
        finally:
            self._running = False
            try:
                await self._post_run_hook.on_run_complete(self.storage)
            except Exception as e:
                logger.error("[EvolutionEngine] post-run hook failed: {}", e)
            logger.info("[EvolutionEngine] Stopped")

    async def step(self) -> None:
        """One generation step (idle → mutate → idle → ingest → refresh → idle)."""
        gen = self.metrics.total_generations
        step_t0 = time.monotonic()

        if self._pre_step_hook:
            await self._pre_step_hook()

        self.storage.snapshot.bump()

        # Phase 1: wait until engine is idle (no QUEUED/RUNNING programs)
        await self._await_idle()
        logger.debug("[EvolutionEngine] gen={} Phase 1: Idle confirmed", gen)

        # On resume, RUNNING programs are recovered to QUEUED before the engine
        # starts. Wait for those DAGs above, then ingest their completed children
        # before creating a new mutation batch; otherwise the new mutation IDs
        # would make recovered children look like stale leftovers.
        if self._resume_ingest_pending:
            await self._ingest_completed_programs(mutation_ids=None)
            self._resume_ingest_pending = False
            self.storage.snapshot.bump(incremental=True)
            logger.debug(
                "[EvolutionEngine] gen={} Resume ingestion completed before mutation",
                gen,
            )

        # Phase 2: select elites & create mutants
        elites = await self._select_elites_for_mutation()
        mutation_ids = await self._create_mutants(elites) if elites else None
        logger.debug(
            "[EvolutionEngine] gen={} Phase 2: Created {} mutant(s)",
            gen,
            len(mutation_ids) if mutation_ids is not None else 0,
        )

        # Phase 3: wait for the mutants' DAGs to finish
        await self._await_idle()
        logger.debug(
            "[EvolutionEngine] gen={} Phase 3: Mutant DAGs finished (idle)", gen
        )

        # Phase 4: ingest newly completed programs (typically the mutants)
        await self._ingest_completed_programs(mutation_ids=mutation_ids)
        logger.debug("[EvolutionEngine] gen={} Phase 4: Ingestion done", gen)

        # Incremental bump: ingestion only changes program states (DONE→DISCARDED)
        # and set membership — not the data fields the collector reads (metrics,
        # lineage, generation).  Allows the snapshot to reuse cached Program
        # objects and only fetch newly added/removed IDs.
        self.storage.snapshot.bump(incremental=True)

        # Phase 5: refresh all archive programs (to re-run lineage/descendant-aware stages)
        refreshed = await self._refresh_archive_programs()
        logger.debug(
            "[EvolutionEngine] gen={} Phase 5: Refreshed {} program(s)", gen, refreshed
        )

        # Phase 6: wait for refresh DAGs to finish
        if refreshed:
            await self._await_idle()
            logger.debug(
                "[EvolutionEngine] gen={} Phase 6: Refresh DAGs finished (idle)", gen
            )

            # Phase 7: reindex archive with updated metrics (e.g., prompt fitness)
            await self.strategy.reindex_archive()
            logger.debug("[EvolutionEngine] gen={} Phase 7: Archive reindexed", gen)

        # Post-step hook: composition injection, cross-population analysis, etc.
        if self._post_step_hook:
            try:
                await self._post_step_hook()
            except Exception as e:
                logger.error("[EvolutionEngine] post_step_hook failed: {}", e)

        self.metrics.total_generations += 1
        try:
            _emit_event(GenerationBoundary(gen=self.metrics.total_generations))
        except Exception:  # pragma: no cover — never fail the engine on logging
            logger.opt(exception=True).debug(
                "[EvolutionEngine] GENERATION_BOUNDARY emission failed"
            )
        await self.storage.save_run_state(
            _RUN_STATE_TOTAL_GENERATIONS, self.metrics.total_generations
        )
        await self.storage.save_run_state(
            _RUN_STATE_PROGRAMS_PROCESSED, self.metrics.programs_processed
        )

        # Log generation summary for easy diagnosis
        step_elapsed = time.monotonic() - step_t0
        archive_program_ids = await self.strategy.get_program_ids()
        archive_size = len(archive_program_ids)
        best_str = await self._format_archive_best_summary(archive_program_ids)

        archive_delta = archive_size - self._prev_archive_size
        self._prev_archive_size = archive_size
        if archive_delta == 0:
            self._stagnant_gens += 1
        else:
            self._stagnant_gens = 0

        logger.info(
            "[EvolutionEngine] gen={} done | elites={} mutants={} refreshed={}"
            " | archive={} ({:+d}){} ({:.1f}s)",
            gen,
            len(elites),
            len(mutation_ids) if mutation_ids is not None else 0,
            refreshed,
            archive_size,
            archive_delta,
            best_str,
            step_elapsed,
        )

        if self._stagnant_gens >= 5:
            logger.warning(
                "[EvolutionEngine] Archive stagnant for {} consecutive generations",
                self._stagnant_gens,
            )

    async def _format_archive_best_summary(
        self, program_ids: list[str] | None = None
    ) -> str:
        """Return a compact best-metrics summary from the current archive.

        The background MetricsTracker is eventually consistent, so generation
        summaries must read the archive directly after ingestion/refresh.
        """
        try:
            if program_ids is None:
                program_ids = await self.strategy.get_program_ids()
            if not program_ids:
                return self._metrics_tracker.format_best_summary()

            programs = await self.storage.mget(
                program_ids, exclude=EXCLUDE_STAGE_RESULTS
            )
            ctx = self._metrics_tracker.metrics_context
            best: dict[str, float] = {}

            for program in programs:
                if not program:
                    continue
                metrics = program.metrics or {}
                try:
                    if not ctx.is_valid(metrics):
                        continue
                except Exception:
                    continue

                for key, value in metrics.items():
                    if key == VALIDITY_KEY or not isinstance(value, (int, float)):
                        continue
                    fval = float(value)
                    if not math.isfinite(fval):
                        continue

                    spec = ctx.specs.get(key)
                    higher_is_better = (
                        True if spec is None else bool(spec.higher_is_better)
                    )
                    current = best.get(key)
                    if (
                        current is None
                        or (higher_is_better and fval > current)
                        or (not higher_is_better and fval < current)
                    ):
                        best[key] = fval

            if not best:
                return self._metrics_tracker.format_best_summary()

            primary_key = ctx.get_primary_key()
            self._best_fitness_for_stopping = best.get(primary_key)

            parts = []
            for key in sorted(best):
                spec = ctx.specs.get(key)
                decimals = spec.decimals if spec else DEFAULT_DECIMALS
                parts.append(f"{key}={best[key]:.{decimals}f}")
            return " best=[" + ", ".join(parts) + "]"
        except Exception:
            logger.opt(exception=True).debug(
                "[EvolutionEngine] archive best summary failed; falling back to tracker"
            )
            return self._metrics_tracker.format_best_summary()

    async def _await_idle(self) -> None:
        """Block until there are no programs in QUEUED or RUNNING."""
        t0 = time.monotonic()
        ghost_checked = False
        while True:
            has_active = await self._has_active_dags()
            if not has_active:
                break

            elapsed = time.monotonic() - t0
            if elapsed > 30 and int(elapsed) % 60 < self.config.loop_interval:
                logger.info(
                    "[EvolutionEngine] gen={} Waiting for idle ({:.0f}s elapsed)",
                    self.metrics.total_generations,
                    elapsed,
                )
            # Ghost safety: after 30s, verify counts with full fetch (once)
            if elapsed > 30 and not ghost_checked:
                ghost_checked = True
                real_q = len(
                    await self.storage.get_all_by_status(
                        ProgramState.QUEUED.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                real_r = len(
                    await self.storage.get_all_by_status(
                        ProgramState.RUNNING.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                if real_q == 0 and real_r == 0:
                    # Clean up ghost IDs from status sets
                    queued_ids = await self.storage.get_ids_by_status(
                        ProgramState.QUEUED.value
                    )
                    running_ids = await self.storage.get_ids_by_status(
                        ProgramState.RUNNING.value
                    )
                    if queued_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.QUEUED.value, queued_ids
                        )
                    if running_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.RUNNING.value, running_ids
                        )
                    ghost_count = len(queued_ids) + len(running_ids)
                    logger.warning(
                        "[EvolutionEngine] Ghost IDs detected — SCARD says active "
                        "but no real programs found. Cleaned {} ghost ID(s) from "
                        "status sets. Breaking idle wait.",
                        ghost_count,
                    )
                    break
            await asyncio.sleep(self.config.loop_interval)

    async def _select_elites_for_mutation(self) -> list[Program]:
        elites = await self.strategy.select_elites(
            total=self.config.max_elites_per_generation
        )
        logger.debug(
            "[EvolutionEngine] gen={} Elites selected: {}",
            self.metrics.total_generations,
            len(elites),
        )
        self.metrics.record_elite_selection_metrics(len(elites), 0)
        return elites

    async def _create_mutants(self, elites: list[Program]) -> list[str]:
        """Create mutants and return their program IDs."""
        logger.debug(
            "[EvolutionEngine] gen={} Mutate from {} elite(s)",
            self.metrics.total_generations,
            len(elites),
        )
        mutation_ids = await generate_mutations(
            elites,
            mutator=self.mutation_operator,
            storage=self.storage,
            state_manager=self.state,
            parent_selector=self.config.parent_selector,
            limit=self.config.max_mutations_per_generation,
            iteration=self.metrics.total_generations,
        )

        self.metrics.record_mutation_metrics(len(mutation_ids), 0)
        if mutation_ids:
            self._consecutive_empty_mutation_generations = 0
        else:
            self._consecutive_empty_mutation_generations += 1
            attempts = self._consecutive_empty_mutation_generations
            logger.warning(
                "[EvolutionEngine] Mutation produced no persisted candidates "
                "({}/{} consecutive generation(s))",
                attempts,
                _MAX_CONSECUTIVE_EMPTY_MUTATION_GENERATIONS,
            )
            if attempts >= _MAX_CONSECUTIVE_EMPTY_MUTATION_GENERATIONS:
                raise EvolutionStopRequested(
                    "Mutation produced no persisted candidates for "
                    f"{attempts} consecutive generations"
                )
        return mutation_ids

    async def _ingest_completed_programs(
        self,
        *,
        mutation_ids: list[str] | None = None,
    ) -> None:
        """
        Validate and hand over any DONE programs to the strategy.
        Programs already in the archive stay DONE (they arrived from a refresh DAG).
        New programs are added if accepted, otherwise discarded.

        Args:
            mutation_ids: IDs of programs created during this generation's mutation
                phase. When None (mutation was skipped or resume recovery is being
                ingested), all non-archive DONE programs are deserialized and validated
                normally. When a list (mutation ran), non-archive DONE programs that are
                NOT in this set are batch-discarded without deserialization — they are
                stale leftovers from previous generations or initial population.
        """
        # Fetch only IDs first (SMEMBERS — no deserialization), then filter
        # out archive programs before doing the expensive mget+deserialize.
        done_ids = await self.storage.get_ids_by_status(ProgramState.DONE.value)
        if not done_ids:
            logger.debug(
                "[EvolutionEngine] gen={} No completed programs to ingest",
                self.metrics.total_generations,
            )
            return

        archive_program_ids = set(await self.strategy.get_program_ids())
        non_archive_ids = [pid for pid in done_ids if pid not in archive_program_ids]

        if not non_archive_ids:
            logger.debug(
                "[EvolutionEngine] gen={} {} DONE programs all in archive, skipping",
                self.metrics.total_generations,
                len(done_ids),
            )
            return

        # Fast path: when mutation_ids are known, batch-discard stale DONE
        # programs (those not created this generation) without deserializing
        # them.  This avoids O(N) mget + from_dict on the initial population.
        if mutation_ids is not None:
            mutation_id_set = set(mutation_ids)
            stale_ids = [pid for pid in non_archive_ids if pid not in mutation_id_set]
            new_ids = [pid for pid in non_archive_ids if pid in mutation_id_set]
            if stale_ids:
                logger.info(
                    "[EvolutionEngine] gen={} Fast-discard {} stale DONE program(s)",
                    self.metrics.total_generations,
                    len(stale_ids),
                )
                try:
                    await self.storage.batch_move_status_sets(
                        stale_ids,
                        ProgramState.DONE.value,
                        ProgramState.DISCARDED.value,
                    )
                except Exception as e:
                    logger.error(
                        "[EvolutionEngine] gen={} stale batch discard failed: {}",
                        self.metrics.total_generations,
                        e,
                    )
        else:
            new_ids = non_archive_ids

        if not new_ids:
            return

        # Only deserialize the new (non-archive) programs.
        # Exclude stage_results (~10% of payload) — ingestion only needs
        # metrics, state, metadata, and lineage.  The merge strategy in
        # storage.update() preserves existing stage_results from Redis.
        completed = await self.storage.mget(new_ids, exclude=EXCLUDE_STAGE_RESULTS)
        # Filter to actual DONE state (mget may return stale status)
        completed = [p for p in completed if p.state == ProgramState.DONE]

        if not completed:
            return

        logger.info(
            "[EvolutionEngine] gen={} Ingest {} program(s) ({} in archive skipped)",
            self.metrics.total_generations,
            len(completed),
            len(done_ids) - len(new_ids),
        )
        logger.debug(
            "[EvolutionEngine] Program IDs: {}",
            [p.short_id for p in completed[:8]]
            + (["..."] if len(completed) > 8 else []),
        )

        added = 0
        rej_valid = 0
        rej_strategy = 0

        # Deterministic checks run before the strategy sees any candidate.
        eligible: list[Program] = []
        acceptor_rejected: list[Program] = []
        for prog in completed:
            try:
                accepted = self.config.program_acceptor.is_accepted(prog)
            except Exception as exc:
                logger.error(
                    "[EvolutionEngine] Acceptor failed for program {}: {} — rejecting",
                    prog.short_id,
                    exc,
                )
                accepted = False

            if accepted:
                eligible.append(prog)
                continue

            rej_valid += 1
            acceptor_rejected.append(prog)
            logger.info(
                "[EvolutionEngine] Program {} REJECTED by acceptor (metrics={})",
                prog.short_id,
                prog.metrics,
            )
            await self._notify_hook(prog, MutationOutcome.REJECTED_ACCEPTOR)

        admission = BatchAdmissionResult()
        if eligible:
            admission = await self._admit_generation_batch(
                eligible,
                incumbent_program_ids=archive_program_ids,
            )

        accepted_ids = set(admission.accepted_new_program_ids)
        rejected_ids = set(admission.rejected_new_program_ids)
        added = len(accepted_ids)
        rej_strategy = len(rejected_ids)

        admission_errors = admission.metadata.get("admission_errors")
        if admission_errors:
            logger.error(
                "[EvolutionEngine] Strategy admission failed for {} program(s): {}",
                len(admission_errors),
                admission_errors,
            )

        for prog in eligible:
            if prog.id in accepted_ids:
                await self._notify_hook(prog, MutationOutcome.ACCEPTED)
                logger.debug(
                    "[EvolutionEngine] Program {} added to strategy (metrics={})",
                    prog.short_id,
                    prog.metrics,
                )
            else:
                await self._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)
                logger.debug(
                    "[EvolutionEngine] Program {} rejected by strategy (metrics={})",
                    prog.short_id,
                    prog.metrics,
                )

        # Apply all terminal state changes together after the generation-wide
        # decision. Strategies may already have transitioned an incumbent; the
        # storage operation only affects IDs that are still in DONE.
        discard_ids = list(
            dict.fromkeys(
                [
                    *(prog.id for prog in acceptor_rejected),
                    *admission.rejected_new_program_ids,
                    *admission.evicted_incumbent_program_ids,
                ]
            )
        )

        # Batch DONE → DISCARDED (raw JSON patch, no Pydantic serialization).
        # Also update in-memory state so any downstream code sees DISCARDED.
        if discard_ids:
            discard_set = set(discard_ids)
            for prog in completed:
                if prog.id in discard_set:
                    prog.state = ProgramState.DISCARDED
            try:
                await self.storage.batch_transition_by_ids(
                    discard_ids,
                    ProgramState.DONE.value,
                    ProgramState.DISCARDED.value,
                )
            except Exception as e:
                logger.error(
                    "[EvolutionEngine] batch discard failed for {} programs: {}",
                    len(discard_ids),
                    e,
                )

        self.metrics.programs_processed += added
        self.metrics.record_ingestion_metrics(added, rej_valid, rej_strategy)
        logger.info(
            "[EvolutionEngine] gen={} Ingest done | added={}, rejected_validation={}, "
            "rejected_strategy={}, evicted_incumbents={}",
            self.metrics.total_generations,
            added,
            rej_valid,
            rej_strategy,
            len(admission.evicted_incumbent_program_ids),
        )

    async def _admit_generation_batch(
        self,
        programs: list[Program],
        *,
        incumbent_program_ids: set[str],
    ) -> BatchAdmissionResult:
        """Run and validate one strategy admission decision.

        A strategy-level exception or malformed result is reconciled against
        the strategy's current active IDs. This prevents a partial write from
        leaving a program active in the archive while the engine discards it.
        """
        try:
            # A few unit/integration callers use a duck-typed strategy mock
            # instead of an EvolutionStrategy subclass. Use the base default in
            # that case so legacy ``add`` behavior remains testable.
            if getattr(type(self.strategy), "add_batch", None) is None:
                result = await self._admit_via_legacy_add(programs)
            else:
                result = await self.strategy.add_batch(programs)
            self._validate_batch_admission_result(
                result,
                programs=programs,
                incumbent_program_ids=incumbent_program_ids,
            )
            return result
        except Exception as exc:
            logger.error(
                "[EvolutionEngine] Generation batch admission failed: {}. "
                "Reconciling against the active archive.",
                exc,
            )
            return await self._reconcile_batch_admission_failure(
                programs,
                incumbent_program_ids=incumbent_program_ids,
                error=exc,
            )

    async def _admit_via_legacy_add(
        self,
        programs: list[Program],
    ) -> BatchAdmissionResult:
        """Compatibility adapter for duck-typed strategies without ``add_batch``.

        Production strategies inherit :class:`EvolutionStrategy` and use its
        final-archive reconciliation. This adapter retains the historical
        ``add`` return-value semantics for external duck-typed implementations.
        """
        accepted_ids: list[str] = []
        rejected_ids: list[str] = []
        errors: dict[str, str] = {}
        for program in programs:
            try:
                if await self.strategy.add(program):
                    accepted_ids.append(program.id)
                else:
                    rejected_ids.append(program.id)
            except Exception as exc:
                rejected_ids.append(program.id)
                errors[program.id] = f"{type(exc).__name__}: {exc}"

        metadata: dict[str, Any] = {}
        if errors:
            metadata["admission_errors"] = errors

        return BatchAdmissionResult(
            accepted_new_program_ids=accepted_ids,
            rejected_new_program_ids=rejected_ids,
            # Legacy implementations expose no atomic archive snapshot. Existing
            # behavior did not report evictions, so leave this empty.
            evicted_incumbent_program_ids=[],
            metadata=metadata,
        )

    @staticmethod
    def _validate_batch_admission_result(
        result: BatchAdmissionResult,
        *,
        programs: list[Program],
        incumbent_program_ids: set[str],
    ) -> None:
        if not isinstance(result, BatchAdmissionResult):
            raise TypeError(
                "strategy.add_batch() must return BatchAdmissionResult, "
                f"got {type(result).__name__}"
            )

        new_ids = [program.id for program in programs]
        if len(new_ids) != len(set(new_ids)):
            raise ValueError("generation batch contains duplicate program IDs")

        expected = set(new_ids)
        accepted = set(result.accepted_new_program_ids)
        rejected = set(result.rejected_new_program_ids)
        actual = accepted | rejected
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                "batch admission must partition every new program exactly once "
                f"(missing={missing}, unknown={unknown})"
            )

        evicted = set(result.evicted_incumbent_program_ids)
        unknown_evictions = evicted - incumbent_program_ids
        if unknown_evictions:
            raise ValueError(
                "batch admission evicted IDs that were not incumbents: "
                f"{sorted(unknown_evictions)}"
            )

    async def _reconcile_batch_admission_failure(
        self,
        programs: list[Program],
        *,
        incumbent_program_ids: set[str],
        error: Exception,
    ) -> BatchAdmissionResult:
        """Derive a safe result after a failed/invalid batch admission."""
        metadata: dict[str, Any] = {
            "batch_admission_error": f"{type(error).__name__}: {error}",
            "reconciled_from_active_archive": True,
        }
        try:
            active_ids = set(await self.strategy.get_program_ids())
        except Exception as snapshot_exc:
            # If even reconciliation is unavailable, preserve every incumbent
            # and reject every newcomer. This is the least destructive outcome.
            active_ids = set(incumbent_program_ids)
            metadata["reconciliation_snapshot_error"] = (
                f"{type(snapshot_exc).__name__}: {snapshot_exc}"
            )

        accepted_ids = [program.id for program in programs if program.id in active_ids]
        accepted_set = set(accepted_ids)
        rejected_ids = [
            program.id for program in programs if program.id not in accepted_set
        ]
        evicted_ids = sorted(incumbent_program_ids - active_ids)
        return BatchAdmissionResult(
            accepted_new_program_ids=accepted_ids,
            rejected_new_program_ids=rejected_ids,
            evicted_incumbent_program_ids=evicted_ids,
            metadata=metadata,
        )

    async def _refresh_archive_programs(self) -> int:
        """Flip all archive programs from DONE to QUEUED so lineage/descendant-aware stages re-run."""
        program_ids_to_refresh = await self.strategy.get_program_ids()

        if not program_ids_to_refresh:
            return 0

        try:
            count = await self.storage.batch_transition_by_ids(
                program_ids_to_refresh,
                ProgramState.DONE.value,
                ProgramState.QUEUED.value,
            )
        except Exception as e:
            logger.error(
                "[EvolutionEngine] gen={} batch_transition_by_ids failed: {}",
                self.metrics.total_generations,
                e,
            )
            return 0

        if count:
            logger.info(
                "[EvolutionEngine] gen={} Submitted {} program(s) for refresh",
                self.metrics.total_generations,
                count,
            )
            self.metrics.record_reprocess_metrics(count)
        return count

    async def _has_active_dags(self) -> bool:
        """True if any programs are QUEUED or RUNNING (i.e., engine not idle).

        Uses count_by_status (SCARD, O(1)) for the fast path.  Falls back to
        the expensive get_all_by_status after 30s of continuous waiting to
        detect ghost IDs that would otherwise stall _await_idle forever.
        """
        queued, running = await asyncio.gather(
            self.storage.count_by_status(ProgramState.QUEUED.value),
            self.storage.count_by_status(ProgramState.RUNNING.value),
        )

        if queued or running:
            current_counts = (queued, running)
            if self._last_pending_dags_counts != current_counts:
                logger.debug(
                    "[EvolutionEngine] Pending DAGs: queued={}, running={}",
                    queued,
                    running,
                )
                self._last_pending_dags_counts = current_counts
            return True

        self._last_pending_dags_counts = None
        return False

    async def _set_state(self, program: Program, state: ProgramState) -> None:
        await self.state.set_program_state(program, state)

    async def _notify_hook(self, prog: Program, outcome: MutationOutcome) -> None:
        """Call on_program_ingested with fault isolation.

        Hook failures are non-fatal: they must never cause a program that was
        already accepted by the strategy to be discarded (which would create a
        ghost entry in the archive).
        """
        try:
            await self.mutation_operator.on_program_ingested(
                prog, self.storage, outcome=outcome
            )
        except Exception as exc:
            logger.warning(
                "[EvolutionEngine] on_program_ingested hook failed for {}: {} "
                "(non-fatal, program state unchanged)",
                prog.short_id,
                exc,
            )

    async def restore_state(self) -> None:
        """Restore total_generations and programs_processed from storage after a resume."""
        gen = await self.storage.load_run_state(_RUN_STATE_TOTAL_GENERATIONS)
        if gen is not None:
            self.metrics.total_generations = gen
            logger.info("[EvolutionEngine] Restored total_generations={}", gen)
        pp = await self.storage.load_run_state(_RUN_STATE_PROGRAMS_PROCESSED)
        if pp is not None:
            self.metrics.programs_processed = pp
            logger.info("[EvolutionEngine] Restored programs_processed={}", pp)
        # The first resumed step must admit any non-archive programs that were
        # DONE already or recovered from RUNNING before it assigns a fresh
        # mutation-ID boundary.
        self._resume_ingest_pending = True

    @property
    def stopper(self) -> EvolutionStopper:
        return self.config.stopper

    def _build_stop_context(self) -> StopContext:
        elapsed = (
            time.monotonic() - self._run_start_time
            if self._run_start_time is not None
            else 0.0
        )
        return StopContext(
            total_generations=self.metrics.total_generations,
            elapsed_seconds=elapsed,
            best_fitness=self._best_fitness_for_stopping,
            programs_processed=self.metrics.programs_processed,
        )

    def _reached_generation_cap(self) -> bool:
        return self.stopper.should_stop(self._build_stop_context()).stop
