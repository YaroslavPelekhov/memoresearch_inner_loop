"""Realistic benchmark for SteadyStateEvolutionEngine throughput.

Simulates real-world two-phase program lifecycle calibrated from production
HoVer experiment logs (hover/steady-state-validation):

Phase 1 — LLM Mutation (~90s mean):
  - Produces a new mutant program via LLM call
  - Subject to mutation server contention (piecewise-linear from measured data)

Phase 2 — DAG Evaluation (~25 min mean):
  - Runs CallValidatorFunction (300 samples × 3 hops on chain LLM servers)
  - Subject to chain server contention (measured: 284s at conc=1, ~1500s at conc=12)
  - Programs remain in PENDING state until evaluation completes

Both phases use scipy lognorm distributions fitted to real experiment data.
Calibration data loaded from benchmarks/calibration_data.json.

Infrastructure: 3 mutation servers, 4 chain servers, 4 concurrent runs.

Usage:
    PYTHONPATH=. python tools/benchmarks/bench_steady_state.py
    PYTHONPATH=. python tools/benchmarks/bench_steady_state.py --sweep
    PYTHONPATH=. python tools/benchmarks/bench_steady_state.py --duration 5.0 --max-in-flight 5
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
from unittest.mock import AsyncMock, MagicMock, patch

import scipy.stats

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxGenerationsStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Timing: 1ms bench time = 1s real time
# ---------------------------------------------------------------------------
TIMESCALE = 0.001

# ---------------------------------------------------------------------------
# Load calibration data (or use defaults)
# ---------------------------------------------------------------------------
_CALIBRATION_PATH = Path(__file__).parent / "calibration_data.json"


def _load_calibration() -> dict:
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


_CAL = _load_calibration()

# ---------------------------------------------------------------------------
# Fitted distributions from production logs
# ---------------------------------------------------------------------------
# LLM mutation: lognorm(s=0.2784, scale=102.87) → mean ~107s, range 62-172s
_LLM_FIT = _CAL.get("llm_mutation", {}).get("fit", {})
_LLM_DIST = scipy.stats.lognorm(
    s=_LLM_FIT.get("s", 0.2784),
    loc=_LLM_FIT.get("loc", 0),
    scale=_LLM_FIT.get("scale", 102.87),
)

# DAG eval: lognorm(s=0.4616, scale=1375.6) → mean ~1513s (~25 min), range 284-3000s
_DAG_FIT = _CAL.get("dag_eval", {}).get("fit", {})
_DAG_DIST = scipy.stats.lognorm(
    s=_DAG_FIT.get("s", 0.4616),
    loc=_DAG_FIT.get("loc", 0),
    scale=_DAG_FIT.get("scale", 1375.6),
)

# ---------------------------------------------------------------------------
# Contention models (piecewise linear interpolation)
# ---------------------------------------------------------------------------
# Mutation server contention (from measured Qwen3-235B data)
_MUTATION_CONTENTION = [
    (1, 5.6),
    (2, 4.3),
    (4, 5.2),
    (8, 10.6),
    (12, 18.0),
    (16, 28.0),
]

# Chain server contention (from DAG eval concurrency analysis)
# The fitted lognorm distribution already captures the TYPICAL eval duration at the
# normal operating point (concurrency ~10-15 across 4 runs).  The contention multiplier
# only adjusts for DEVIATIONS from that baseline.
#
# Measured data shows:
#   conc 1-4:   0.19-0.67x of mean  (very fast — less GPU contention)
#   conc 5-8:   0.96-1.28x          (approaching normal)
#   conc 8-22:  0.72-1.57x          (noisy, centered ~1.0)
#
# We model this as a simple piecewise curve relative to the normal operating point.
_CHAIN_CONTENTION = [
    (1, 0.3),  # very low concurrency: much faster
    (4, 0.65),  # light load
    (8, 1.0),  # normal operating point — distribution mean applies as-is
    (15, 1.05),  # slight increase at higher load
    (22, 1.15),  # moderate increase at heavy load
]


def _piecewise_interp(curve: list[tuple[float, float]], x: float) -> float:
    """Piecewise linear interpolation."""
    if x <= curve[0][0]:
        return curve[0][1]
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    # Extrapolate from last two points
    x0, y0 = curve[-2]
    x1, y1 = curve[-1]
    slope = (y1 - y0) / (x1 - x0)
    return y1 + slope * (x - x1)


def _mutation_contention_multiplier(per_server_n: float) -> float:
    """Multiplier relative to baseline (N=1)."""
    baseline = _MUTATION_CONTENTION[0][1]
    lat = _piecewise_interp(_MUTATION_CONTENTION, max(1, per_server_n))
    return lat / baseline


def _dag_contention_multiplier(global_concurrent: int) -> float:
    """Multiplier on DAG eval base duration.

    The fitted distribution already models the typical operating point (conc ~12).
    This returns a multiplier that adjusts for deviation from that baseline:
    <1.0 at low concurrency (less GPU contention), ~1.0 at normal, >1.0 at high.
    """
    return _piecewise_interp(_CHAIN_CONTENTION, max(1, global_concurrent))


# ---------------------------------------------------------------------------
# Server pools
# ---------------------------------------------------------------------------


class ServerPool:
    """Round-robin server pool tracking per-server load."""

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
# Instrumentation
# ---------------------------------------------------------------------------


@dataclass
class BenchMetrics:
    mutations_completed: int = 0
    evals_completed: int = 0
    programs_ingested: int = 0
    epochs_completed: int = 0
    max_concurrent_mutations: int = 0
    max_concurrent_evals: int = 0
    _concurrent_mutations: int = 0
    _concurrent_evals: int = 0
    pipeline_fill_snapshots: list[tuple[float, int]] = field(default_factory=list)
    mutation_latencies: list[float] = field(default_factory=list)
    eval_latencies: list[float] = field(default_factory=list)
    t0: float = 0.0
    max_in_flight: int = 5
    num_mutation_servers: int = 3
    num_chain_servers: int = 4

    def summary(self, wall_clock: float, compact: bool = False) -> str:
        sim_wall = wall_clock / TIMESCALE
        sim_rate = self.mutations_completed / sim_wall * 60 if sim_wall > 0 else 0
        ingest_rate = self.programs_ingested / sim_wall * 60 if sim_wall > 0 else 0

        avg_mut_lat = (
            statistics.mean(self.mutation_latencies) if self.mutation_latencies else 0
        )
        avg_eval_lat = (
            statistics.mean(self.eval_latencies) if self.eval_latencies else 0
        )

        if compact:
            return (
                f"max_in_flight={self.max_in_flight:>2d} | "
                f"{ingest_rate:>5.2f} ingested/min | "
                f"mut={self.mutations_completed} eval={self.evals_completed} | "
                f"avg_mut={avg_mut_lat:.0f}s avg_eval={avg_eval_lat:.0f}s"
            )

        fills = [f for _, f in self.pipeline_fill_snapshots]
        avg_fill = statistics.mean(fills) if fills else 0

        lines = [
            "=== Steady-State Engine Benchmark (Two-Phase) ===",
            f"Wall clock (bench):       {wall_clock:.2f}s",
            f"Simulated wall clock:     {sim_wall:.0f}s ({sim_wall / 60:.1f}min)",
            "",
            "--- Throughput ---",
            f"Mutations produced:       {self.mutations_completed}",
            f"Evals completed:          {self.evals_completed}",
            f"Programs ingested:        {self.programs_ingested}",
            f"Epochs completed:         {self.epochs_completed}",
            f"Ingestion rate:           {ingest_rate:.2f} programs/min",
            f"Mutation rate:            {sim_rate:.2f} mutants/min",
            "",
            "--- Latency ---",
            f"Avg LLM mutation:         {avg_mut_lat:.0f}s",
            f"Avg DAG evaluation:       {avg_eval_lat:.0f}s",
            "",
            "--- Pipeline ---",
            f"Max concurrent mutations: {self.max_concurrent_mutations}",
            f"Max concurrent evals:     {self.max_concurrent_evals}",
            f"Avg pipeline fill:        {avg_fill:.1f} / {self.max_in_flight}",
            "",
            "--- Infrastructure ---",
            f"Mutation servers:         {self.num_mutation_servers}",
            f"Chain servers:            {self.num_chain_servers}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Two-phase benchmark
# ---------------------------------------------------------------------------


async def run_benchmark(
    duration_s: float = 3.0,
    max_in_flight: int = 5,
    epoch_size: int = 8,
    max_generations: int = 200,
    num_mutation_servers: int = 3,
    num_chain_servers: int = 4,
) -> BenchMetrics:
    """Run benchmark with two-phase program lifecycle."""
    metrics = BenchMetrics(
        t0=time.monotonic(),
        max_in_flight=max_in_flight,
        num_mutation_servers=num_mutation_servers,
        num_chain_servers=num_chain_servers,
    )

    mutation_pool = ServerPool(num_mutation_servers)
    chain_pool = ServerPool(num_chain_servers)

    # Programs in various states
    # After LLM mutation: program exists but is PENDING (awaiting DAG eval)
    # After DAG eval: program becomes DONE (ready for ingestion)
    pending_programs: dict[
        str, tuple[Program, float]
    ] = {}  # pid -> (prog, eval_finish_time)
    done_programs: dict[str, Program] = {}
    discarded: set[str] = set()

    # Background DAG evaluator
    eval_queue: asyncio.Queue[tuple[str, Program]] = asyncio.Queue()

    async def dag_evaluator():
        """Simulates DAG evaluation for programs after LLM mutation."""
        while True:
            pid, prog = await eval_queue.get()
            try:
                metrics._concurrent_evals += 1
                metrics.max_concurrent_evals = max(
                    metrics.max_concurrent_evals, metrics._concurrent_evals
                )

                # Acquire chain server
                sid = await chain_pool.acquire()
                global_conc = chain_pool.total_load

                # Sample DAG eval duration with contention
                base_dur = max(60.0, _DAG_DIST.rvs())  # floor at 60s
                mult = _dag_contention_multiplier(global_conc)
                real_dur = base_dur * mult
                sim_dur = real_dur * TIMESCALE

                await asyncio.sleep(sim_dur)
                await chain_pool.release(sid)

                metrics.eval_latencies.append(real_dur)
                metrics.evals_completed += 1
                metrics._concurrent_evals -= 1

                # Move program to DONE
                prog.state = ProgramState.DONE
                done_programs[pid] = prog
            except asyncio.CancelledError:
                metrics._concurrent_evals -= 1
                raise

    # Start eval workers (one per chain server slot — they share the pool)
    # More workers than chain servers is fine; contention model handles slowdown
    num_eval_workers = num_chain_servers * 4  # enough to not bottleneck on workers
    eval_tasks = [asyncio.create_task(dag_evaluator()) for _ in range(num_eval_workers)]

    # Mocks for the engine
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
        """Phase 1: LLM mutation (produces PENDING program)."""
        metrics._concurrent_mutations += 1
        metrics.max_concurrent_mutations = max(
            metrics.max_concurrent_mutations, metrics._concurrent_mutations
        )

        # Acquire mutation server
        sid = await mutation_pool.acquire()
        per_server = mutation_pool.per_server_load(sid)

        # Sample LLM mutation duration with contention
        base_dur = max(30.0, _LLM_DIST.rvs())  # floor at 30s
        mult = _mutation_contention_multiplier(per_server)
        real_dur = base_dur * mult
        sim_dur = real_dur * TIMESCALE

        await asyncio.sleep(sim_dur)
        await mutation_pool.release(sid)

        metrics.mutation_latencies.append(real_dur)
        metrics.mutations_completed += 1
        metrics._concurrent_mutations -= 1

        # Program starts as PENDING (not yet DONE — needs DAG eval)
        # But the engine expects DONE programs for ingestion.
        # We create as QUEUED and let the dag_evaluator move it to DONE.
        prog = Program(code="def solve(): return 42", state=ProgramState.QUEUED)
        pending_programs[prog.id] = (prog, 0)

        # Queue for DAG evaluation
        await eval_queue.put((prog.id, prog))

        return [prog.id]

    def mget_side(ids, **kw):
        result = []
        for pid in ids:
            if pid in done_programs:
                result.append(done_programs[pid])
            elif pid in pending_programs:
                result.append(pending_programs[pid][0])
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

    # Track epochs
    orig_epoch = engine._epoch_refresh

    async def tracked_epoch():
        await orig_epoch()
        metrics.epochs_completed += 1

    engine._epoch_refresh = tracked_epoch

    # Track ingestion
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

    # Monitor pipeline fill
    async def monitor():
        while engine._running:
            async with engine._in_flight_lock:
                fill = len(engine._in_flight)
            metrics.pipeline_fill_snapshots.append((time.monotonic(), fill))
            await asyncio.sleep(0.005)

    with patch(
        "gigaevo.evolution.engine.steady_state.generate_mutations",
        side_effect=fake_generate,
    ):
        monitor_task = asyncio.create_task(monitor())
        try:
            await asyncio.wait_for(engine.run(), timeout=duration_s)
        except TimeoutError:
            engine._running = False
        finally:
            monitor_task.cancel()
            for t in eval_tasks:
                t.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            await asyncio.gather(*eval_tasks, return_exceptions=True)

    return metrics


async def sweep_max_in_flight(
    duration_s: float,
    epoch_size: int,
    max_generations: int,
    num_mutation_servers: int,
    num_chain_servers: int,
    flight_values: list[int],
) -> list[tuple[BenchMetrics, float]]:
    """Run benchmark for each max_in_flight value."""
    results = []
    for mif in flight_values:
        random.seed(42)
        t0 = time.monotonic()
        m = await run_benchmark(
            duration_s=duration_s,
            max_in_flight=mif,
            epoch_size=epoch_size,
            max_generations=max_generations,
            num_mutation_servers=num_mutation_servers,
            num_chain_servers=num_chain_servers,
        )
        wall = time.monotonic() - t0
        results.append((m, wall))
    return results


def main():
    infra = _CAL.get("infrastructure", {})
    default_mut_servers = infra.get("mutation_servers", 3)
    default_chain_servers = infra.get("chain_servers", 4)
    default_mif = infra.get("max_in_flight", 5)
    default_epoch = infra.get("epoch_size", 8)

    parser = argparse.ArgumentParser(
        description="Benchmark steady-state engine (two-phase lifecycle)"
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Bench seconds")
    parser.add_argument("--max-in-flight", type=int, default=default_mif)
    parser.add_argument("--epoch-size", type=int, default=default_epoch)
    parser.add_argument("--max-generations", type=int, default=200)
    parser.add_argument("--mutation-servers", type=int, default=default_mut_servers)
    parser.add_argument("--chain-servers", type=int, default=default_chain_servers)
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep max_in_flight values"
    )
    args = parser.parse_args()

    if args.sweep:
        flight_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
        print(
            f"=== Sweep: max_in_flight "
            f"(mut_servers={args.mutation_servers}, chain_servers={args.chain_servers}) ==="
        )
        results = asyncio.run(
            sweep_max_in_flight(
                args.duration,
                args.epoch_size,
                args.max_generations,
                args.mutation_servers,
                args.chain_servers,
                flight_values,
            )
        )
        best_rate = 0
        best_m = None
        for m, wall in results:
            sim_wall = wall / TIMESCALE
            rate = m.programs_ingested / sim_wall * 60 if sim_wall > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_m = m
            marker = ""
            if m is best_m:
                marker = " <-- BEST"
            print(m.summary(wall, compact=True) + marker)
        if best_m:
            print(f"\nSweet spot: max_in_flight={best_m.max_in_flight}")
            print(f"Peak ingestion rate: {best_rate:.2f} programs/min")
    else:
        random.seed(42)
        t0 = time.monotonic()
        m = asyncio.run(
            run_benchmark(
                duration_s=args.duration,
                max_in_flight=args.max_in_flight,
                epoch_size=args.epoch_size,
                max_generations=args.max_generations,
                num_mutation_servers=args.mutation_servers,
                num_chain_servers=args.chain_servers,
            )
        )
        wall = time.monotonic() - t0
        print(m.summary(wall))


if __name__ == "__main__":
    main()
