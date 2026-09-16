#!/usr/bin/env python3
"""Multi-run contention simulation for SteadyStateEvolutionEngine.

Simulates N concurrent evolution runs sharing:
- M mutation servers (LLM calls for producing mutants)
- K chain servers (DAG evaluation / CallValidatorFunction)

Two-phase program lifecycle (calibrated from production HoVer logs):
1. LLM Mutation (~90-110s): produces mutant, subject to mutation server contention
2. DAG Evaluation (~25 min): validates program on chain servers, subject to chain contention

Both phases use scipy lognorm distributions fitted to real experiment data.
Calibration data loaded from benchmarks/calibration_data.json.

Measured mutation server contention (Qwen3-235B):
  per_server_concurrent=1: 5.6s mean  (baseline)
  per_server_concurrent=2: 4.3s mean  (batching sweet spot)
  per_server_concurrent=4: 5.2s mean  (still OK)
  per_server_concurrent=8: 10.6s mean (saturated)

Chain server contention (from DAG eval concurrency analysis):
  conc 1-4:  0.3-0.65x base  (less GPU contention)
  conc 8-15: 1.0-1.05x base  (normal operating point)
  conc >15:  1.05-1.15x base (moderate increase)

Usage:
    # 4 runs, 3 mutation servers, 4 chain servers (production config)
    PYTHONPATH=. python tools/benchmarks/bench_multirun.py

    # Sweep max_in_flight
    PYTHONPATH=. python tools/benchmarks/bench_multirun.py --sweep

    # Custom config
    PYTHONPATH=. python tools/benchmarks/bench_multirun.py --runs 4 --mut-servers 3 --chain-servers 4
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import statistics
import time
from unittest.mock import AsyncMock, MagicMock

import scipy.stats

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxGenerationsStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
TIMESCALE = 0.001  # 1ms bench time = 1s real time

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------
_CALIBRATION_PATH = Path(__file__).parent / "calibration_data.json"


def _load_calibration() -> dict:
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


_CAL = _load_calibration()

# ---------------------------------------------------------------------------
# Fitted distributions
# ---------------------------------------------------------------------------
_LLM_FIT = _CAL.get("llm_mutation", {}).get("fit", {})
_LLM_DIST = scipy.stats.lognorm(
    s=_LLM_FIT.get("s", 0.2784),
    loc=_LLM_FIT.get("loc", 0),
    scale=_LLM_FIT.get("scale", 102.87),
)

_DAG_FIT = _CAL.get("dag_eval", {}).get("fit", {})
_DAG_DIST = scipy.stats.lognorm(
    s=_DAG_FIT.get("s", 0.4616),
    loc=_DAG_FIT.get("loc", 0),
    scale=_DAG_FIT.get("scale", 1375.6),
)

# ---------------------------------------------------------------------------
# Contention models
# ---------------------------------------------------------------------------
_MUTATION_CONTENTION = [
    (1, 5.6),
    (2, 4.3),
    (4, 5.2),
    (8, 10.6),
    (12, 18.0),
    (16, 28.0),
]

# Chain contention: multiplier on DAG eval base duration
# Distribution already models typical operating point (conc ~12)
_CHAIN_CONTENTION = [
    (1, 0.3),
    (4, 0.65),
    (8, 1.0),
    (15, 1.05),
    (22, 1.15),
]


def _piecewise_interp(curve: list[tuple[float, float]], x: float) -> float:
    if x <= curve[0][0]:
        return curve[0][1]
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    x0, y0 = curve[-2]
    x1, y1 = curve[-1]
    slope = (y1 - y0) / (x1 - x0)
    return y1 + slope * (x - x1)


def _mutation_contention_multiplier(per_server_n: float) -> float:
    baseline = _MUTATION_CONTENTION[0][1]
    lat = _piecewise_interp(_MUTATION_CONTENTION, max(1, per_server_n))
    return lat / baseline


def _dag_contention_multiplier(global_concurrent: int) -> float:
    return _piecewise_interp(_CHAIN_CONTENTION, max(1, global_concurrent))


# ---------------------------------------------------------------------------
# Shared server pools
# ---------------------------------------------------------------------------


class ServerPool:
    """Simulates servers shared by all runs with round-robin assignment."""

    def __init__(self, num_servers: int):
        self.num_servers = num_servers
        self._load = [0] * num_servers
        self._next = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        async with self._lock:
            sid = self._next
            self._next = (self._next + 1) % self.num_servers
            self._load[sid] += 1
            return sid

    async def release(self, sid: int) -> None:
        async with self._lock:
            self._load[sid] -= 1

    def per_server_load(self, sid: int) -> int:
        return self._load[sid]

    @property
    def total_load(self) -> int:
        return sum(self._load)


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    label: str
    mutations_completed: int = 0
    evals_completed: int = 0
    programs_ingested: int = 0
    max_concurrent_mutations: int = 0
    max_concurrent_evals: int = 0
    _concurrent_mutations: int = 0
    _concurrent_evals: int = 0
    mutation_latencies: list[float] = field(default_factory=list)
    eval_latencies: list[float] = field(default_factory=list)

    def ingestion_rate(self, sim_wall_s: float) -> float:
        return self.programs_ingested / sim_wall_s * 60 if sim_wall_s > 0 else 0

    def mutation_rate(self, sim_wall_s: float) -> float:
        return self.mutations_completed / sim_wall_s * 60 if sim_wall_s > 0 else 0

    def summary(self, sim_wall_s: float) -> str:
        irate = self.ingestion_rate(sim_wall_s)
        avg_mut = (
            statistics.mean(self.mutation_latencies) if self.mutation_latencies else 0
        )
        avg_eval = statistics.mean(self.eval_latencies) if self.eval_latencies else 0
        return (
            f"  {self.label}: {irate:>5.2f} ingested/min | "
            f"mut={self.mutations_completed} eval={self.evals_completed} "
            f"ingested={self.programs_ingested} | "
            f"avg_mut={avg_mut:.0f}s avg_eval={avg_eval:.0f}s"
        )


# ---------------------------------------------------------------------------
# Single-run engine factory
# ---------------------------------------------------------------------------


async def create_run(
    label: str,
    mutation_pool: ServerPool,
    chain_pool: ServerPool,
    max_in_flight: int,
    epoch_size: int,
    max_generations: int,
) -> tuple[SteadyStateEvolutionEngine, RunMetrics]:
    """Create one engine instance wired to shared server pools."""
    metrics = RunMetrics(label=label)
    pending_programs: dict[str, Program] = {}  # pid -> prog (QUEUED, awaiting eval)
    done_programs: dict[str, Program] = {}
    discarded: set[str] = set()

    # Background DAG evaluator queue
    eval_queue: asyncio.Queue[tuple[str, Program]] = asyncio.Queue()

    async def dag_evaluator():
        while True:
            pid, prog = await eval_queue.get()
            try:
                metrics._concurrent_evals += 1
                metrics.max_concurrent_evals = max(
                    metrics.max_concurrent_evals, metrics._concurrent_evals
                )

                sid = await chain_pool.acquire()
                global_conc = chain_pool.total_load

                base_dur = max(60.0, _DAG_DIST.rvs())
                mult = _dag_contention_multiplier(global_conc)
                real_dur = base_dur * mult
                sim_dur = real_dur * TIMESCALE

                await asyncio.sleep(sim_dur)
                await chain_pool.release(sid)

                metrics.eval_latencies.append(real_dur)
                metrics.evals_completed += 1
                metrics._concurrent_evals -= 1

                prog.state = ProgramState.DONE
                done_programs[pid] = prog
            except asyncio.CancelledError:
                metrics._concurrent_evals -= 1
                raise

    # Start eval workers
    num_eval_workers = chain_pool.num_servers * 4
    eval_tasks = [asyncio.create_task(dag_evaluator()) for _ in range(num_eval_workers)]

    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    mt = MagicMock()
    mt.format_best_summary.return_value = ""

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    strategy.get_program_ids.return_value = []
    strategy.add.return_value = True
    strategy.select_elites.return_value = [
        Program(code="def solve(): return 42", state=ProgramState.DONE)
    ]

    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        max_mutations_per_generation=epoch_size,
        stopper=MaxGenerationsStopper(max_generations),
        loop_interval=0.001,
    )

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=mt,
    )
    engine.config.program_acceptor = MagicMock()
    engine.config.program_acceptor.is_accepted.return_value = True

    async def fake_generate(elites, **kwargs):
        metrics._concurrent_mutations += 1
        metrics.max_concurrent_mutations = max(
            metrics.max_concurrent_mutations, metrics._concurrent_mutations
        )

        sid = await mutation_pool.acquire()
        per_server = mutation_pool.per_server_load(sid)

        base_dur = max(30.0, _LLM_DIST.rvs())
        mult = _mutation_contention_multiplier(per_server)
        real_dur = base_dur * mult
        sim_dur = real_dur * TIMESCALE

        await asyncio.sleep(sim_dur)
        await mutation_pool.release(sid)

        metrics.mutation_latencies.append(real_dur)
        metrics.mutations_completed += 1
        metrics._concurrent_mutations -= 1

        prog = Program(code="def solve(): return 42", state=ProgramState.QUEUED)
        pending_programs[prog.id] = prog
        await eval_queue.put((prog.id, prog))
        return [prog.id]

    def mget_side(ids, **kw):
        result = []
        for pid in ids:
            if pid in done_programs:
                result.append(done_programs[pid])
            elif pid in pending_programs:
                result.append(pending_programs[pid])
            else:
                result.append(None)
        return result

    storage.mget.side_effect = mget_side

    def get_ids_side(status):
        if status == ProgramState.DONE.value:
            return list(done_programs.keys())
        if status == ProgramState.QUEUED.value:
            return [pid for pid in pending_programs if pid not in done_programs]
        return []

    storage.get_ids_by_status.side_effect = get_ids_side

    orig_ingest = engine._ingest_batch

    async def tracked_ingest(pids):
        result = await orig_ingest(pids)
        added, _ = result
        metrics.programs_ingested += added
        for pid in pids:
            done_programs.pop(pid, None)
            pending_programs.pop(pid, None)
            discarded.add(pid)
        return result

    engine._ingest_batch = tracked_ingest

    # Monkey-patch _create_single_mutant for this engine instance
    async def patched_create_single_mutant(elites):
        ids = await fake_generate(elites)
        if ids:
            engine.metrics.record_mutation_metrics(len(ids), 0)
        return ids

    engine._create_single_mutant = patched_create_single_mutant

    # Store eval_tasks on engine for cleanup
    engine._bench_eval_tasks = eval_tasks  # type: ignore[attr-defined]

    return engine, metrics


# ---------------------------------------------------------------------------
# Multi-run simulation
# ---------------------------------------------------------------------------


async def run_simulation(
    num_runs: int = 4,
    num_mutation_servers: int = 3,
    num_chain_servers: int = 4,
    max_in_flight: int = 5,
    epoch_size: int = 8,
    duration_s: float = 5.0,
    max_generations: int = 500,
) -> tuple[list[RunMetrics], float]:
    """Run num_runs engines concurrently sharing server pools."""
    mutation_pool = ServerPool(num_mutation_servers)
    chain_pool = ServerPool(num_chain_servers)

    engines = []
    all_metrics = []

    for i in range(num_runs):
        label = f"R{i + 1}"
        engine, metrics = await create_run(
            label, mutation_pool, chain_pool, max_in_flight, epoch_size, max_generations
        )
        engines.append(engine)
        all_metrics.append(metrics)

    async def run_one(engine):
        try:
            await asyncio.wait_for(engine.run(), timeout=duration_s)
        except TimeoutError:
            engine._running = False
        finally:
            # Cleanup eval tasks
            for t in getattr(engine, "_bench_eval_tasks", []):
                t.cancel()
            await asyncio.gather(
                *getattr(engine, "_bench_eval_tasks", []), return_exceptions=True
            )

    tasks = [asyncio.create_task(run_one(eng)) for eng in engines]
    t0 = time.monotonic()
    await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.monotonic() - t0
    sim_wall = wall / TIMESCALE

    return all_metrics, sim_wall


def print_results(
    label: str,
    all_metrics: list[RunMetrics],
    sim_wall: float,
):
    total_ingested = sum(m.programs_ingested for m in all_metrics)
    total_mutations = sum(m.mutations_completed for m in all_metrics)
    total_irate = total_ingested / sim_wall * 60 if sim_wall > 0 else 0
    total_mrate = total_mutations / sim_wall * 60 if sim_wall > 0 else 0

    all_mut_lats = [lat for m in all_metrics for lat in m.mutation_latencies]
    all_eval_lats = [lat for m in all_metrics for lat in m.eval_latencies]
    avg_mut = statistics.mean(all_mut_lats) if all_mut_lats else 0
    avg_eval = statistics.mean(all_eval_lats) if all_eval_lats else 0

    print(f"\n=== {label} ===")
    print(f"Simulated wall clock: {sim_wall:.0f}s ({sim_wall / 60:.1f}min)")
    print(f"Total mutations:      {total_mutations}")
    print(f"Total ingested:       {total_ingested}")
    print(f"Total mutation rate:  {total_mrate:.2f} mutants/min (across all runs)")
    print(f"Total ingestion rate: {total_irate:.2f} programs/min (across all runs)")
    print(f"Avg LLM mutation:     {avg_mut:.0f}s")
    print(f"Avg DAG evaluation:   {avg_eval:.0f}s")
    print()
    for m in all_metrics:
        print(m.summary(sim_wall))


def main():
    infra = _CAL.get("infrastructure", {})
    default_mut = infra.get("mutation_servers", 3)
    default_chain = infra.get("chain_servers", 4)
    default_mif = infra.get("max_in_flight", 5)
    default_epoch = infra.get("epoch_size", 8)

    parser = argparse.ArgumentParser(
        description="Multi-run contention simulation (two-phase lifecycle)"
    )
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--mut-servers", type=int, default=default_mut)
    parser.add_argument("--chain-servers", type=int, default=default_chain)
    parser.add_argument("--max-in-flight", type=int, default=default_mif)
    parser.add_argument("--epoch-size", type=int, default=default_epoch)
    parser.add_argument("--duration", type=float, default=5.0, help="Bench seconds")
    parser.add_argument("--max-generations", type=int, default=500)
    parser.add_argument("--sweep", action="store_true", help="Sweep max_in_flight")
    args = parser.parse_args()

    if args.sweep:
        flight_values = [2, 3, 4, 5, 6, 8, 10, 12, 16]
        print(
            f"=== Sweep: {args.runs} runs, {args.mut_servers} mut servers, "
            f"{args.chain_servers} chain servers ==="
        )
        for mif in flight_values:
            random.seed(42)
            metrics, sim_wall = asyncio.run(
                run_simulation(
                    args.runs,
                    args.mut_servers,
                    args.chain_servers,
                    mif,
                    args.epoch_size,
                    args.duration,
                    args.max_generations,
                )
            )
            total_ingested = sum(m.programs_ingested for m in metrics)
            total_mutations = sum(m.mutations_completed for m in metrics)
            irate = total_ingested / sim_wall * 60 if sim_wall > 0 else 0
            mrate = total_mutations / sim_wall * 60 if sim_wall > 0 else 0
            all_eval = [lat for m in metrics for lat in m.eval_latencies]
            avg_eval = statistics.mean(all_eval) if all_eval else 0
            print(
                f"  max_in_flight={mif:>2d} | "
                f"{irate:>5.2f} ingested/min | "
                f"{mrate:>5.2f} mutants/min | "
                f"ingested={total_ingested} mut={total_mutations} | "
                f"avg_eval={avg_eval:.0f}s"
            )
    else:
        random.seed(42)
        metrics, sim_wall = asyncio.run(
            run_simulation(
                args.runs,
                args.mut_servers,
                args.chain_servers,
                args.max_in_flight,
                args.epoch_size,
                args.duration,
                args.max_generations,
            )
        )
        print_results(
            f"{args.runs} runs, {args.mut_servers} mut servers, "
            f"{args.chain_servers} chain servers, "
            f"max_in_flight={args.max_in_flight}",
            metrics,
            sim_wall,
        )


if __name__ == "__main__":
    main()
