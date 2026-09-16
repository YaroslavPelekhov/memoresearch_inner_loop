#!/usr/bin/env python3
"""Benchmark: FIFO vs LPT scheduling for DAG evaluation.

Simulates the DagRunner's launch ordering with different scheduling
strategies using a discrete-event model (no asyncio timing artifacts).

Strategies compared:
- **FIFO**: programs launched in arrival order (baseline)
- **Oracle LPT**: perfect knowledge of eval times (upper bound)
- **Predicted LPT**: uses SimpleHeuristicPredictor (realistic)
- **SJF**: shortest-predicted first (negative control)

The benchmark varies noise levels to show the spectrum from
"perfect prediction" to "noisy prediction" to "random order".

Usage:
    PYTHONPATH=. python tools/benchmarks/bench_lpt.py
    PYTHONPATH=. python tools/benchmarks/bench_lpt.py --programs 40 --servers 4
    PYTHONPATH=. python tools/benchmarks/bench_lpt.py --sweep-noise
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import random
import statistics

from gigaevo.evolution.scheduling import (
    EvalTimePredictor,
    FIFOPrioritizer,
    LPTPrioritizer,
    ProgramPrioritizer,
    SimpleHeuristicPredictor,
    SJFPrioritizer,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CALIBRATION_PATH = Path(__file__).parent / "calibration_data.json"


def _load_calibration() -> dict:
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


_CAL = _load_calibration()


# ---------------------------------------------------------------------------
# Oracle predictor (for upper bound)
# ---------------------------------------------------------------------------


class OraclePredictor(EvalTimePredictor):
    """Knows exact eval times.  Upper bound for LPT benefit."""

    def __init__(self, dur_map: dict[str, float]) -> None:
        self._dur_map = dur_map

    def predict(self, program: Program) -> float:
        return self._dur_map.get(program.id, 1.0)

    def update(self, program: Program, actual_duration: float) -> None:
        pass

    def is_warm(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Program generation
# ---------------------------------------------------------------------------


def _make_programs(
    n: int, rng: random.Random, *, noise_ratio: float = 0.3
) -> list[tuple[Program, float]]:
    """Create n programs with variable code lengths and correlated eval times.

    Args:
        noise_ratio: Controls how much base_dur noise there is relative
            to the code-length signal.  0.0 = eval_time purely determined
            by code_length.  1.0 = high noise (real-world-like).
    """
    programs = []

    # Mean eval time from calibration (default ~1500s)
    mean_eval = 1500.0

    for _ in range(n):
        # Variable code length: 200-5000 chars (realistic range)
        code_len = int(rng.lognormvariate(7.0, 0.6))
        code_len = max(200, min(5000, code_len))
        code = "def solve():\n" + "    x = 1\n" * (code_len // 12)

        prog = Program(code=code, state=ProgramState.QUEUED)

        # Signal: code length determines base eval time
        # Normalize so mean code_len (~1100) maps to mean_eval
        signal = (code_len / 1100.0) * mean_eval

        # Noise: LogNormal multiplicative noise
        if noise_ratio > 0:
            noise_mult = rng.lognormvariate(0, noise_ratio)
        else:
            noise_mult = 1.0

        true_dur = max(60.0, signal * noise_mult)
        programs.append((prog, true_dur))

    return programs


# ---------------------------------------------------------------------------
# Discrete-event simulation (no asyncio timing artifacts)
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    strategy: str
    total_programs: int
    makespan_s: float
    throughput_per_min: float
    tail_idle_s: float
    server_utilization: float
    completion_times: list[float] = field(default_factory=list)


def simulate_discrete(
    programs_with_durations: list[tuple[Program, float]],
    prioritizer: ProgramPrioritizer,
    num_servers: int,
    strategy_name: str,
    *,
    warmup_programs: list[tuple[Program, float]] | None = None,
) -> SimResult:
    """Discrete-event simulation of parallel machine scheduling.

    Models K identical parallel machines (servers).  Programs are assigned
    to machines in prioritizer order.  When a machine finishes, the next
    program in queue starts.  No asyncio — pure math.
    """
    # Warm up predictor
    predictor = prioritizer.predictor
    if predictor is not None and warmup_programs:
        for prog, dur in warmup_programs:
            predictor.update(prog, dur)

    # Prioritize
    progs_only = [p for p, _ in programs_with_durations]
    dur_map = {p.id: dur for p, dur in programs_with_durations}
    ordered = prioritizer.prioritize(progs_only)

    if not ordered:
        return SimResult(strategy_name, 0, 0, 0, 0, 0)

    # Min-heap of (finish_time, server_id) — tracks when each server is free
    servers: list[tuple[float, int]] = [(0.0, i) for i in range(num_servers)]
    heapq.heapify(servers)

    completion_times: list[float] = []
    total_busy_time = 0.0

    for prog in ordered:
        dur = dur_map[prog.id]
        # Assign to earliest-free server
        free_at, sid = heapq.heappop(servers)
        start = free_at
        finish = start + dur
        heapq.heappush(servers, (finish, sid))
        completion_times.append(finish)
        total_busy_time += dur

    makespan = max(finish for finish, _ in servers)
    completion_times.sort()

    tail_idle = 0.0
    if len(completion_times) >= 2:
        tail_idle = completion_times[-1] - completion_times[-2]

    utilization = total_busy_time / (makespan * num_servers) if makespan > 0 else 0

    return SimResult(
        strategy=strategy_name,
        total_programs=len(ordered),
        makespan_s=makespan,
        throughput_per_min=len(ordered) / makespan * 60 if makespan > 0 else 0,
        tail_idle_s=tail_idle,
        server_utilization=utilization,
        completion_times=completion_times,
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_comparison(
    num_programs: int = 20,
    num_servers: int = 4,
    seed: int = 42,
    noise_ratio: float = 0.3,
    num_warmup: int = 10,
) -> list[SimResult]:
    """Run FIFO vs Oracle LPT vs Predicted LPT vs SJF comparison."""
    rng = random.Random(seed)

    warmup = _make_programs(num_warmup, rng, noise_ratio=noise_ratio)
    test_programs = _make_programs(num_programs, rng, noise_ratio=noise_ratio)

    dur_map = {p.id: dur for p, dur in test_programs}

    strategies: list[tuple[str, ProgramPrioritizer]] = [
        ("FIFO", FIFOPrioritizer()),
        ("Oracle", LPTPrioritizer(OraclePredictor(dur_map))),
        ("LPT", LPTPrioritizer(SimpleHeuristicPredictor())),
        ("SJF", SJFPrioritizer(SimpleHeuristicPredictor())),
    ]

    results = []
    for name, prioritizer in strategies:
        result = simulate_discrete(
            test_programs,
            prioritizer,
            num_servers,
            name,
            warmup_programs=warmup,
        )
        results.append(result)

    return results


def print_results(results: list[SimResult], num_servers: int, noise: float) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"LPT Scheduling  ({results[0].total_programs} programs, "
        f"{num_servers} servers, noise={noise:.1f})"
    )
    print(f"{'=' * 78}")
    print(
        f"{'Strategy':<10} | {'Makespan':>10} | {'Throughput':>14} | "
        f"{'Tail Idle':>10} | {'Utilization':>12} | {'vs FIFO':>8}"
    )
    print("-" * 78)

    fifo_makespan = results[0].makespan_s if results else 1

    for r in results:
        delta = (fifo_makespan - r.makespan_s) / fifo_makespan * 100
        delta_str = f"{delta:+.1f}%" if r.strategy != "FIFO" else ""
        print(
            f"{r.strategy:<10} | {r.makespan_s:>8.0f}s | "
            f"{r.throughput_per_min:>10.2f}/min | "
            f"{r.tail_idle_s:>8.0f}s | "
            f"{r.server_utilization:>10.1%} | "
            f"{delta_str:>8}"
        )

    print()
    # Show eval time distribution
    durs = [
        dur for _, dur in _make_programs(results[0].total_programs, random.Random(42))
    ]
    if durs:
        print(
            f"  Eval time stats: min={min(durs):.0f}s  "
            f"mean={statistics.mean(durs):.0f}s  "
            f"max={max(durs):.0f}s  "
            f"stdev={statistics.stdev(durs):.0f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="FIFO vs LPT scheduling benchmark")
    parser.add_argument("--programs", type=int, default=40)
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.3, help="Noise ratio (0-1)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--sweep-noise",
        action="store_true",
        help="Sweep noise levels to show prediction quality impact",
    )
    parser.add_argument(
        "--sweep-servers",
        action="store_true",
        help="Sweep server count",
    )
    parser.add_argument(
        "--sweep-programs",
        action="store_true",
        help="Sweep program count (batch size)",
    )
    args = parser.parse_args()

    if args.sweep_noise:
        print(
            f"\n=== Noise sweep ({args.programs} programs, {args.servers} servers) ==="
        )
        print(
            f"{'Noise':>8} | {'FIFO make':>10} | {'Oracle LPT':>11} | "
            f"{'Pred LPT':>10} | {'Oracle gain':>12} | {'Pred gain':>10}"
        )
        print("-" * 75)

        for noise in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            results = run_comparison(
                args.programs, args.servers, args.seed, noise, args.warmup
            )
            fifo = results[0].makespan_s
            oracle = results[1].makespan_s
            pred = results[2].makespan_s
            oracle_gain = (fifo - oracle) / fifo * 100
            pred_gain = (fifo - pred) / fifo * 100
            print(
                f"{noise:>8.1f} | {fifo:>8.0f}s | {oracle:>9.0f}s | "
                f"{pred:>8.0f}s | {oracle_gain:>+10.1f}% | {pred_gain:>+8.1f}%"
            )

    elif args.sweep_servers:
        print(f"\n=== Server sweep ({args.programs} programs, noise={args.noise}) ===")
        print(
            f"{'Servers':>8} | {'FIFO make':>10} | {'Oracle':>10} | "
            f"{'Pred LPT':>10} | {'Oracle gain':>12} | {'Pred gain':>10}"
        )
        print("-" * 75)

        for n_servers in [1, 2, 4, 8, 12, 16]:
            results = run_comparison(
                args.programs, n_servers, args.seed, args.noise, args.warmup
            )
            fifo = results[0].makespan_s
            oracle = results[1].makespan_s
            pred = results[2].makespan_s
            oracle_gain = (fifo - oracle) / fifo * 100
            pred_gain = (fifo - pred) / fifo * 100
            print(
                f"{n_servers:>8} | {fifo:>8.0f}s | {oracle:>8.0f}s | "
                f"{pred:>8.0f}s | {oracle_gain:>+10.1f}% | {pred_gain:>+8.1f}%"
            )

    elif args.sweep_programs:
        print(
            f"\n=== Program count sweep ({args.servers} servers, noise={args.noise}) ==="
        )
        print(
            f"{'Programs':>9} | {'FIFO make':>10} | {'Oracle':>10} | "
            f"{'Pred LPT':>10} | {'Oracle gain':>12} | {'Pred gain':>10}"
        )
        print("-" * 75)

        for n_progs in [8, 16, 32, 64, 128]:
            results = run_comparison(
                n_progs, args.servers, args.seed, args.noise, args.warmup
            )
            fifo = results[0].makespan_s
            oracle = results[1].makespan_s
            pred = results[2].makespan_s
            oracle_gain = (fifo - oracle) / fifo * 100
            pred_gain = (fifo - pred) / fifo * 100
            print(
                f"{n_progs:>9} | {fifo:>8.0f}s | {oracle:>8.0f}s | "
                f"{pred:>8.0f}s | {oracle_gain:>+10.1f}% | {pred_gain:>+8.1f}%"
            )

    else:
        results = run_comparison(
            args.programs, args.servers, args.seed, args.noise, args.warmup
        )
        print_results(results, args.servers, args.noise)


if __name__ == "__main__":
    main()
