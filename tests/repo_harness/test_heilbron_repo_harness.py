from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

from gigaevo.problems.initial_loaders import RepoSeedLoader
from gigaevo.repo_harness.auto_seed import create_auto_seed_repo
from gigaevo.repo_harness.git_utils import rev_parse

PROBLEM_DIR = Path(__file__).resolve().parents[2] / "problems" / "heilbron"
BENCHMARK_PATH = PROBLEM_DIR / "repo_benchmark.py"


def test_heilbron_auto_seed_benchmarks_with_repo_harness_adapter(tmp_path: Path):
    result = create_auto_seed_repo(
        problem_dir=PROBLEM_DIR,
        problem_name="heilbron",
        seed_root=tmp_path / "seeds",
        seed_name="run",
        variant="grid",
    )

    assert (result.source_repo / "solution.py").is_file()
    assert rev_parse(result.source_repo)

    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_PATH),
            "--candidate-repo",
            str(result.source_repo),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    metrics = json.loads(completed.stdout)
    assert metrics["is_valid"] == 1
    assert metrics["fitness"] > 0.0


def test_heilbron_repo_harness_loads_all_five_seed_parents(tmp_path: Path):
    variants = ["grid", "random_arr", "cluster", "fan", "arc"]
    result = create_auto_seed_repo(
        problem_dir=PROBLEM_DIR,
        problem_name="heilbron",
        seed_root=tmp_path / "seeds",
        seed_name="run",
        variants=variants,
    )

    class MemoryStorage:
        def __init__(self):
            self.programs = []

        async def add(self, program):
            self.programs.append(program)

    storage = MemoryStorage()
    loader = RepoSeedLoader(
        source_repo=result.source_repo,
        refs=result.refs,
        entrypoint="solution.py",
    )
    programs = asyncio.run(loader.load(storage))

    assert len(programs) == 5
    assert [program.metadata["seed_ref"] for program in programs] == list(result.refs)
    assert len({program.metadata["git_commit"] for program in programs}) == 5
    assert all(program.lineage.is_root() for program in programs)
