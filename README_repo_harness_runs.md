# Repo-Harness Runbook

This file is a practical runbook for the four repo-harness experiment modes:
`repo_harness`, `autoresearch`, `repo_harness_archive`, and
`repo_harness_archival_curator`. All four evolve a whole Git repository instead
of a single code string: each candidate is a Git commit, mutation is performed
by a coding-agent CLI, and evaluation is performed by a benchmark command
against a temporary checkout of that commit.

More detailed design notes live in `docs/REPO_HARNESS_EVOLUTION.md`.

## Basic Requirements

Run commands from the repo root unless a command says otherwise:

```bash
cd /home/projects/giga_harness/gigaevo-repo-harness
```

Before launching a run, make sure:

- the Python environment for GigaEvo is active;
- Redis is running and `redis.db` / `redis.prefix` do not collide with a run you
  want to keep;
- `repo_harness.source_repo` points at an initialized Git repository with at
  least one commit, or `repo_harness.auto_seed.enabled=true` is set and the
  problem provides a `repo_harness_seed.yaml` spec;
- the mutation backend command is installed and authenticated, for example
  `codex`;
- the benchmark command can run outside GigaEvo as a smoke test;
- expensive benchmark dependencies are ready, for example Docker images, blobs,
  Harbor, ProgramBench, or Terminal-Bench task files.

## Choosing A Repo-Harness Mode

The modes share the same auto-seeding, Git mutation, benchmark, and optional
visualization machinery. The autoresearch mode intentionally omits reflection;
the other modes can use it. The modes also differ in how evaluated candidates
become the active parent pool:

| Experiment | Active archive | Extra LLM curation | Best fit |
| --- | --- | --- | --- |
| `repo_harness` | Single-island MAP-Elites over the primary-metric behavior space | No | Compatibility with the standard GigaEvo algorithm and runs where primary-metric cells are useful diversity buckets |
| `autoresearch` | One global incumbent, replaced only by a strict primary-fitness improvement | No; only mutation invokes an agent | Greedy batched search where every generation samples sibling mutations from the best repository so far |
| `repo_harness_archive` | Deterministic repo-aware top-K archive per primary-metric cell, with novelty and hall-of-fame roles | No | Larger, inexpensive portfolios selected from fitness, repo descriptors, novelty, and lineage |
| `repo_harness_archival_curator` | One capacity-limited portfolio chosen after each completed generation | Yes, one curator decision per non-empty eligible batch | Agent-harness and benchmark-solving runs where distinct ideas, unit-level capabilities, tradeoffs, and reflections matter more than fixed fitness cells |

### Default MAP-Elites Mode

Select the default mode with:

```text
experiment=repo_harness
```

This uses the standard single-island MAP-Elites strategy. The behavior space is
built from the primary metric, so candidates at different primary-metric levels
can occupy different cells. Within a cell, the stronger candidate wins. This is
simple and compatible with normal GigaEvo runs, but on a single-island
repo-harness task it can retain low-fitness cells while rejecting candidates
that collide with an already strong cell.

The minimal command shape is:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=<problem_name> \
  redis.prefix=<unique_run_prefix> \
  repo_harness.source_repo=/abs/path/to/seed-or-agent-repo \
  'repo_harness.benchmark_command=["python","/abs/path/to/benchmark.py","--candidate-repo","{worktree}"]' \
  repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.CommandCodingAgentBackend \
  +repo_harness.mutation_backend.name=codex \
  '+repo_harness.mutation_backend.command=["codex","exec","--ephemeral","--sandbox","workspace-write","--skip-git-repo-check","--cd","{worktree}","{prompt}"]' \
  stopper=max_generations_or_target_fitness \
  target_fitness=1.0 \
  max_generations=10 \
  max_mutations_per_generation=1 \
  max_elites_per_generation=2 \
  num_parents=1 \
  redis.db=0 \
  post_step_hook=repo_viz
```

### Autoresearch Mode

Select the greedy mutation-only loop with:

```text
experiment=autoresearch
```

Each generation creates `max_mutations_per_generation` independent mutations
from the same best repository. It benchmarks the complete batch, keeps the best
strict improvement, and otherwise starts the next generation from the unchanged
incumbent. The evaluation DAG contains only the repo benchmark and deterministic
metric validation; it does not call reflection, insight, lineage, memory,
parent-selection, or archive-curation agents.

In pseudocode, the loop is:

```text
incumbent = best evaluated seed
repeat:
    children = mutate incumbent K times
    evaluate every child
    candidate = best child by the configured primary metric
    if candidate is strictly better than incumbent:
        incumbent = candidate
```

All `K` children in a generation are siblings: their worktrees start from the
same incumbent commit. Selection happens only after the generation becomes
idle, so a fast-finishing child cannot become the parent of another child in
the same batch. If a mutation agent fails or produces no admissible change, the
actual batch can contain fewer than `K` children.

The preset fixes the selection-related settings as follows:

| Setting | Value | Purpose |
| --- | ---: | --- |
| `num_parents` | `1` | Disable crossover; each mutation receives only the incumbent repository |
| `max_elites_per_generation` | `1` | Expose only the incumbent to parent selection |
| `island_max_size` | `1` | Keep only one active repository |
| `primary_resolution` | `1` | Put every score in the same archive cell, where replacement requires strict improvement |
| `enable_migration` | `false` | Disable irrelevant multi-island behavior |
| `repo_harness.staged_validation.enabled` | `false` | Compare complete benchmark fitness instead of a projected failed-case score |

The parent selector is named `RandomParentSelector`, but it is deterministic in
this mode because its available pool contains exactly one program. Its infinite
iterator is used only to repeat that incumbent `K` times. The mutation coding
agent receives the incumbent checkout, task description, and benchmark feedback
available on that program. No separate LLM selects parents or summarizes the
result afterward.

The command shape is the same as the default repo harness:

```bash
python run.py \
  experiment=autoresearch \
  problem.name=<problem_name> \
  redis.prefix=<unique_run_prefix> \
  repo_harness.source_repo=/abs/path/to/seed-or-agent-repo \
  'repo_harness.benchmark_command=["python","/abs/path/to/benchmark.py","--candidate-repo","{worktree}"]' \
  repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.CommandCodingAgentBackend \
  +repo_harness.mutation_backend.name=codex \
  '+repo_harness.mutation_backend.command=["codex","exec","--ephemeral","--sandbox","workspace-write","--skip-git-repo-check","--cd","{worktree}","{prompt}"]' \
  max_mutations_per_generation=4 \
  max_generations=20
```

`max_mutations_per_generation` is the autoresearch batch size. Keep the
generational engine: the steady-state engine would admit results as they finish
instead of comparing one complete sibling batch. `max_concurrent_dags` controls
how many child benchmarks can execute concurrently; it does not change the
selection semantics.

### Deterministic Top-K Archive Mode

Select the deterministic repo-aware archive with:

```text
experiment=repo_harness_archive
```

A complete command has the same shape as the default command; change the
experiment name:

```bash
python run.py \
  experiment=repo_harness_archive \
  problem.name=<problem_name> \
  redis.prefix=<unique_run_prefix> \
  repo_harness.source_repo=/abs/path/to/seed-or-agent-repo \
  'repo_harness.benchmark_command=["python","/abs/path/to/benchmark.py","--candidate-repo","{worktree}"]' \
  ...
```

This variant keeps the same Git-backed mutation and benchmark path, but adds a
cheap descriptor stage and uses a deterministic repo-specific active archive
with top-K candidates per primary-metric cell. It also tracks local Pareto,
novelty, and hall-of-fame roles. Historical candidates remain in
ProgramStorage, including candidates marked `failure`, `superseded`, or
`quarantined`.

Useful knobs are:

```text
repo_archive.cell_size=8
repo_archive.hall_of_fame_size=8
repo_archive.novelty_threshold=0.55
repo_archive.min_quality_floor=null
repo_archive.enable_quarantine=false
```

Use this mode when deterministic admission and lower LLM cost are important,
but the default one-elite-per-cell behavior is too restrictive.

### LLM Archival Curator Mode

Select generation-level archival curation with:

```text
experiment=repo_harness_archival_curator
```

A complete command again changes only the experiment-specific settings:

```bash
python run.py \
  experiment=repo_harness_archival_curator \
  problem.name=<problem_name> \
  redis.prefix=<unique_run_prefix> \
  repo_harness.source_repo=/abs/path/to/seed-or-agent-repo \
  'repo_harness.benchmark_command=["python","/abs/path/to/benchmark.py","--candidate-repo","{worktree}"]' \
  ...
```

The curator maintains a capacity-limited active parent portfolio over immutable
ProgramStorage history. It compares every deterministically valid child from a
completed generation with the current archive using metrics, generic benchmark
evidence, lineage, implementation diffs, and repo reflections. The default
capacity is 10 and can be changed with
`archival_curator.capacity=<count>`.

The deterministic filter runs before the curator. Candidates with missing or
sentinel primary metrics or invalid results are excluded without spending an
LLM call. Protected-file quarantine is disabled by default; when enabled,
protected-file quarantine flags are also excluded by this filter. The best
observed candidate is protected by default. If the curator call fails and
`archival_curator.fail_open=true`, eligible incumbents are preserved and
remaining capacity is filled deterministically.

Capacity is an upper bound, not a target. The curator may leave slots unused
when the remaining programs have no credible quality, diversity, robustness, or
recombination value. A lower-fitness program can remain active when it preserves
a distinct mechanism, solves different evaluation units, has useful resource
tradeoffs, or provides complementary parent material.

Benchmark evidence is expressed in generic evaluation units rather than
Terminal-Bench-specific task fields. When available, cards include observed
unit outcomes, evaluation completeness, changes versus the parent, distinctions
from the incumbent archive, aggregate metrics, and resource usage. Benchmarks
that expose only aggregate metrics are supported as well.

Useful curator knobs are:

```text
archival_curator.capacity=10
archival_curator.protect_champion=true
archival_curator.enable_quarantine=false
archival_curator.fail_open=true
```

Because this mode deliberately considers regressions that may contain useful
ideas, it sets `repo_harness.reflection_skip_clear_regressions=false`. Budget
for one reflection call per eligible evaluated child and one curator call per
completed non-empty child batch.

### Settings Shared By All Modes

Important settings are:

- `experiment=repo_harness` selects the Git-backed evolution pipeline.
- `experiment=autoresearch` selects greedy, mutation-only `(1 + K)` repo
  evolution.
- `experiment=repo_harness_archive` selects the Git-backed pipeline with the
  top-K repo archive variant.
- `experiment=repo_harness_archival_curator` selects the Git-backed pipeline
  with generation-batch Codex archive curation.
- `problem.name` selects `problems/<name>/task_description.txt` and
  `metrics.yaml`.
- `repo_harness.source_repo` is the explicit seed repository that GigaEvo checks
  out, mutates, and commits from. With auto-seed, GigaEvo fills
  `repo_harness.effective_source_repo` with the generated seed path.
- `{worktree}` is replaced with the temporary checkout for the candidate being
  evaluated or mutated.
- `repo_harness.benchmark_command` should emit metrics JSON on stdout, usually
  including `fitness` and `is_valid`.
- `post_step_hook=repo_viz` writes a static dashboard for inspecting the run.

`max_mutations_per_generation` is a limit, not a guaranteed count. The engine
creates at most one mutation for each parent set returned by `parent_selector`.
For example, with `num_parents=1`, one active archive member produces only one
unique single-parent selection even when
`max_mutations_per_generation=2`. Once two active parents exist, the selector
can return two distinct single-parent selections. Increasing the mutation limit
does not make the meta selector repeat the same parent set.

`num_parents` is the number of repositories combined into each mutation, not
the number of mutations. With `num_parents=2`, each selection normally contains
a primary parent whose worktree is edited and a secondary parent whose
reflection and diff are supplied as source material.

Start with small values for `max_generations`, `max_mutations_per_generation`,
and benchmark concurrency. Increase them only after the smoke benchmark and the
mutation backend both work.

### Archival-Curator Auto-Seed Smoke Example

The bundled Heilbron problem is a convenient end-to-end smoke test:

```bash
python run.py \
  experiment=repo_harness_archival_curator \
  problem.name=heilbron \
  redis.prefix=heilbron_curator_smoke_01 \
  redis.db=0 \
  repo_harness.auto_seed.enabled=true \
  repo_harness.auto_seed.variant=grid \
  repo_harness.entrypoint=solution.py \
  'repo_harness.benchmark_command=["python","${problem.dir}/repo_benchmark.py","--candidate-repo","{worktree}"]' \
  repo_harness.structured_feedback_path=feedback.json \
  repo_harness.staged_validation.enabled=false \
  repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.CommandCodingAgentBackend \
  +repo_harness.mutation_backend.name=codex \
  '+repo_harness.mutation_backend.command=["codex","exec","--ephemeral","--sandbox","workspace-write","--skip-git-repo-check","--cd","{worktree}","{prompt}"]' \
  archival_curator.capacity=3 \
  max_mutations_per_generation=2 \
  num_parents=1 \
  stopper=max_generations_or_target_fitness \
  max_generations=4 \
  target_fitness=0.0365 \
  post_step_hook=repo_viz
```

Four generations are used so that the archive has time to grow beyond one
parent and a later generation can exercise multiple distinct parent selections.
Use a fresh `redis.prefix` for a clean run. Reusing a prefix resumes the
persisted generation and archive state.

## Auto-Creating Seed Repos

Repo-harness can create the initial placeholder Git repo when
`repo_harness.source_repo` is omitted:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=programbench_task \
  redis.prefix=quick_programbench_auto_seed \
  repo_harness.auto_seed.enabled=true \
  'repo_harness.benchmark_command=["python","/abs/path/to/benchmark.py","--candidate-repo","{worktree}"]' \
  ...
```

By default, generated seeds are written next to this repository under:

```text
../repo_harness_seeds/<problem-name>/<redis-prefix>/<variant>/
```

For example, from `/home/petranokhin/projects/codex_evo/gigaevo-core-internal`,
the default seed root is:

```text
/home/petranokhin/projects/codex_evo/repo_harness_seeds/
```

Each repo-harness problem that wants auto-seeding must provide exactly one
`repo_harness_seed.yaml` or `repo_harness_seed.yml` somewhere under its
`problems/<problem.name>/` directory, unless the run sets
`repo_harness.auto_seed.spec_path` explicitly. The spec is deliberately
deterministic: it describes placeholder files, not an LLM-generated solution, so
generation 0 remains an empty scaffold and the first real solution attempt still
happens in the first mutation cycle.

Minimal spec shape:

```yaml
version: 1
default_variant: default
variants:
  default:
    description: Importable placeholder harness.
    files:
      README.md:
        content: |
          Empty seed repo. First mutation should implement the task.
      main.py:
        executable: true
        content: |
          #!/usr/bin/env python3
          raise SystemExit(0)
```

Seed specs can also vendor an installed package into the generated repo. This
is useful when the mutation agent should edit a real architecture rather than a
thin wrapper around a site-package dependency:

```yaml
variants:
  default:
    package_files:
      - package: harbor.agents.terminus_2
        destination: agents/terminus_2
        include: ["*.py", "*.sh", "templates/*.txt"]
        import_rewrites:
          harbor.agents.terminus_2: agents.terminus_2
    files:
      agents/baseline_terminus2.py:
        content: |
          from agents.terminus_2.terminus_2 import Terminus2
```

Use variants when one repo-harness problem can run different inner tasks that
need different starting repo shapes:

```bash
repo_harness.auto_seed.variant=rust-cli
repo_harness.auto_seed.spec_path=terminal_bench2/repo_harness_seed.yaml
```

New repo-harness problem implementations should document and commit their seed
spec beside the benchmark adapter. The spec should create only the minimum files
needed for the benchmark to import, compile, or execute the candidate; it should
not contain task-solving code.

## ProgramBench

ProgramBench support lives in:

```text
problems/programbench_task/programbench/
```

The candidate repo is packed as a ProgramBench submission, evaluated with
`programbench eval`, and converted to GigaEvo metrics JSON.

### Local Environment Variables

Set these before running ProgramBench repo-harness commands from this checkout:

```bash
export GE=/home/petranokhin/projects/codex_evo/gigaevo-core-internal
export PB=/home/petranokhin/projects/codex_evo/ProgramBench
export INSTANCE=testorg__calculator.abc1234
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python

cd "$GE"
```

`GE`, `PB`, and `INSTANCE` are used in the commands below. The `UV_*` variables
keep `uv` from writing to read-only home cache locations in restricted
environments.

This optional check catches a wrong ProgramBench path or instance id before a
GigaEvo run:

```bash
test -d "$PB/src/programbench/data/tasks/$INSTANCE" && echo "task ok"
```

For the bundled calculator fixture (`testorg__calculator.abc1234`), the adapter
uses local fixture tests and can build the local Docker task image on demand.
For real ProgramBench tasks, pre-sync blobs and optionally pre-pull the Docker
image:

```bash
cd "$PB"
uv run programbench blob sync "$INSTANCE"

IMAGE="programbench/$(python -c 'import os; print(os.environ["INSTANCE"].replace("__", "_1776_"))'):task"
docker pull "$IMAGE"
```

### Prepare A ProgramBench Task

From the ProgramBench checkout, sync blobs and pull the task image when needed:

```bash
cd /home/projects/giga_harness/ProgramBench
programbench blob sync xampprocky__tokei.505d648
docker pull programbench/xampprocky_1776_tokei.505d648:task
```

### Smoke A Candidate

Run the benchmark directly before starting evolution:

```bash
cd "$GE"

python "$GE/problems/programbench_task/programbench/benchmark.py" \
  --candidate-repo /path/to/seed-or-auto-seed-repo \
  --programbench-root "$PB" \
  --instance-id "$INSTANCE" \
  --workers 1 \
  --branch-workers 1 \
  --docker-cpus 2
```

Detailed benchmark outputs are written under:

```text
problems/programbench_task/programbench/jobs/
```

### Run ProgramBench Evolution

This uses auto-seeding, runs Codex as the mutation backend, and stops either at
`fitness=1.0` or `max_generations`:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=programbench_task \
  redis.prefix=pb_codex_calculator \
  repo_harness.auto_seed.enabled=true \
  repo_harness.auto_seed.variant=python-cli \
  "repo_harness.benchmark_command=[\"python\",\"$GE/problems/programbench_task/programbench/benchmark.py\",\"--candidate-repo\",\"{worktree}\",\"--programbench-root\",\"$PB\",\"--instance-id\",\"$INSTANCE\",\"--workers\",\"1\",\"--branch-workers\",\"1\"]" \
  'repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.CommandCodingAgentBackend' \
  +repo_harness.mutation_backend.name=codex \
  '+repo_harness.mutation_backend.command=["codex","exec","--ephemeral","--sandbox","workspace-write","--skip-git-repo-check","--cd","{worktree}","{prompt}"]' \
  stopper=max_generations_or_target_fitness \
  target_fitness=1.0 \
  max_generations=3 \
  max_mutations_per_generation=1 \
  max_elites_per_generation=2 \
  num_parents=1 \
  redis.db=2 \
  post_step_hook=repo_viz
```

For another ProgramBench instance, change:

- `redis.prefix`, so the run has its own Redis namespace;
- `INSTANCE`, so the benchmark targets the desired ProgramBench task;
- `repo_harness.auto_seed.variant`, if the task needs a different initial repo
  shape such as `rust-cli`;
- `--workers`, `--branch-workers`, and `--docker-cpus` to fit available CPU and
  Docker capacity;
- `redis.db`, if you separate runs by Redis database.

## Terminal-Bench 2

Terminal-Bench support lives in:

```text
problems/repo_harness_template/terminal_bench2/
```

It expects Harbor to run Terminal-Bench 2 tasks in containers. The candidate repo
is a Harbor-compatible agent harness repository.

### Smoke A Candidate

```bash
cd /home/projects/giga_harness/gigaevo-repo-harness

python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --runs 1 \
  --concurrency 1
```

For OpenRouter-backed Harbor runs:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...

python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --openrouter
```

For a local OpenAI-compatible llama.cpp server:

```bash
python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --model openai/qwen-local \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --local-llama-defaults
```

### Run Terminal-Bench Evolution

```bash
python run.py \
  experiment=repo_harness \
  problem.name=repo_harness_template \
  redis.prefix=terminal_bench2_smoke \
  repo_harness.source_repo=/home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  'repo_harness.benchmark_command=["python","/home/projects/giga_harness/gigaevo-repo-harness/problems/repo_harness_template/terminal_bench2/benchmark.py","--candidate-repo","{worktree}","--task-set","smoke","--runs","1","--concurrency","1"]' \
  repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.CommandCodingAgentBackend \
  +repo_harness.mutation_backend.name=codex \
  '+repo_harness.mutation_backend.command=["codex","exec","--ephemeral","--sandbox","workspace-write","--skip-git-repo-check","--cd","{worktree}","{prompt}"]' \
  stopper=max_generations_or_target_fitness \
  target_fitness=1.0 \
  max_generations=10 \
  max_mutations_per_generation=1 \
  max_elites_per_generation=2 \
  num_parents=1 \
  redis.db=0 \
  post_step_hook=repo_viz
```

Useful Terminal-Bench benchmark selectors:

- `--task-set smoke` for a one-task debugging run;
- `--task-set balanced20` for a deterministic 20-task mix across difficulties;
- `--task-set hard` for the cheaper hard subset;
- `--task-set full` for all tasks;
- `--tasks task-a,task-b` or repeated `--task task-a` for explicit tasks;
- `--runs` and `--concurrency` for Harbor attempt count and parallelism.

Detailed Terminal-Bench outputs are written under:

```text
problems/repo_harness_template/terminal_bench2/jobs/
```

## Inspecting Results

With `post_step_hook=repo_viz`, each generation refreshes a static repo-harness
dashboard under the run output directory.

Useful places to inspect:

```text
outputs/<date>/<time>/
outputs/<date>/<time>/viz/
problems/programbench_task/programbench/jobs/
problems/repo_harness_template/terminal_bench2/jobs/
```

For individual benchmark summaries:

```bash
python -m json.tool problems/programbench_task/programbench/jobs/<job-name>/gigaevo_summary.json
python -m json.tool problems/repo_harness_template/terminal_bench2/jobs/<job-name>/gigaevo_summary.json
```

## Common Problems

- If the initial candidate cannot be loaded, check that
  `repo_harness.source_repo` is a Git repo with at least one commit, or that
  auto-seed found exactly one valid `repo_harness_seed.yaml` under the problem.
- If Hydra rejects an override, quote list-valued overrides exactly as shown in
  the examples.
- If the benchmark cannot find the candidate, include
  `"--candidate-repo","{worktree}"` in `repo_harness.benchmark_command`.
- If ProgramBench fails before evaluation, pre-sync blobs and pre-pull or build
  the task Docker image.
- If Terminal-Bench fails before tasks run, verify Harbor, model credentials,
  Docker, and the candidate agent import path.
- If runs mix together in Redis, use a fresh `redis.prefix` or a separate
  `redis.db`.
