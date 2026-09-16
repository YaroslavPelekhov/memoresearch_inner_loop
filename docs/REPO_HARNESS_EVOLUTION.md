# Repo Harness Evolution

This fork adds a Git-backed mode for evolving an entire coding-agent harness
repository with GigaEvo.

## Mental Model

GigaEvo still owns the loop:

```text
select parent -> mutate -> evaluate -> score -> archive -> repeat
```

The candidate artifact is now a Git commit:

```text
parent commit + benchmark feedback -> coding-agent CLI edits repo -> child commit
```

Redis stores population state, metrics, lineage, and a small manifest in
`Program.code`. Git stores the real harness files.

## Candidate Manifest

`Program.code` is JSON like:

```json
{
  "kind": "repo_snapshot",
  "schema_version": 1,
  "repo_path": "/path/to/agent-repo",
  "commit": "abc123...",
  "parent_commit": "def456...",
  "branch": "gigaevo/candidate/...",
  "entrypoint": "agent_harness:AgentHarness",
  "changed_files": ["agent/loop.py"]
}
```

## New Pieces

- `RepoSeedLoader`: seeds the first population member from an existing Git repo.
- `RepoHarnessMutationOperator`: creates a Git worktree at the parent commit,
  writes a benchmark feedback brief, runs a coding-agent backend, commits the
  edit, and returns a child manifest.
- `RepoBenchmarkStage`: checks out a candidate commit and runs your benchmark
  command. The command should emit metrics JSON on stdout, or write a JSON file
  configured with `repo_harness.metrics_path`.
- `RepoReflectionStage`: computes the parent-to-child Git diff, compares metrics
  and benchmark feedback, asks the configured LLM for a compact repo-level
  insight/lineage summary, and feeds that markdown into the next mutation brief.
- `CommandCodingAgentBackend`: generic subprocess backend for Codex, Claude Code,
  OpenCode, Pi, or any local mutation agent.
- `ClaudeCodeBackend`: convenience wrapper for `claude -p`.

## Minimal Run Shape

Use the repo experiment config:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=my_repo_harness_problem \
  repo_harness.source_repo=/abs/path/to/agent-repo \
  'repo_harness.benchmark_command=["python","bench.py"]' \
  max_generations=3
```

The benchmark command runs from a temporary checkout of each candidate commit.
By default it must print:

```json
{"fitness": 0.42, "is_valid": 1.0}
```

For richer mutation guidance, the benchmark may also emit generic structured
feedback to stderr:

```text
[gigaevo] structured feedback:
{"schema_version": 1, "benchmark": "my_benchmark", "summary": {}, "failure_clusters": [], "examples": [], "hints": []}
```

Alternatively, set `repo_harness.structured_feedback_path` to a JSON file
written by the benchmark inside the candidate worktree. Legacy ProgramBench
logs using `[programbench] structured failure feedback:` are still parsed, but
new adapters should use the generic marker or file path.

Your problem still needs `task_description.txt` and `metrics.yaml` so GigaEvo
knows which metric is primary and how to compare candidates.
There is a starter problem at `problems/repo_harness_template/`.

## Using Claude Code

Override the backend:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=my_repo_harness_problem \
  repo_harness.source_repo=/abs/path/to/agent-repo \
  'repo_harness.benchmark_command=["python","bench.py"]' \
  'repo_harness.mutation_backend._target_=gigaevo.repo_harness.backends.ClaudeCodeBackend' \
  repo_harness.mutation_backend.model=opus
```

## Using Any Coding-Agent CLI

Use the generic command backend. The command may reference `{prompt}`,
`{prompt_path}`, `{cwd}`, `{brief_path}`, `{worktree}`, `{parent_commit}`, and
`{branch}`.

Codex example shape:

```yaml
repo_harness:
  mutation_backend:
    _target_: gigaevo.repo_harness.backends.CommandCodingAgentBackend
    name: codex
    command:
      - codex
      - exec
      - --skip-git-repo-check
      - --dangerously-bypass-approvals-and-sandbox
      - "{prompt}"
```

Adjust the command to your installed Codex CLI version.
`CommandCodingAgentBackend` automatically adds `--json` to `codex exec` so it
can retain the final response while recording machine-readable token usage.

OpenCode/Pi/etc. use the same backend; replace only `name` and `command`.
The command must edit files in `{cwd}` and exit when done.

## Replacing Codex For Reflection And Parent Selection

The repo-harness experiment defaults to `llm=codex` for non-mutating LLM work
such as repo reflection and meta parent selection. That preset now uses the
generic `CommandCLIChatModel`, so any CLI that prints its response to stdout can
be used:

```bash
python run.py \
  experiment=repo_harness \
  llm=command_cli \
  'llm.models.0.command=["opencode","run","--prompt","{prompt}"]' \
  llm.models.0.model_name=opencode-cli \
  llm.models.0.provider_name=opencode \
  ...
```

For CLI LLM stages, the command may reference `{prompt}`, `{prompt_path}`, and
`{cwd}`; set `llm.models.0.stdin_prompt=true` for CLIs that read the prompt
from stdin. For repo mutation, `CommandCodingAgentBackend` additionally supports
`{brief_path}`, `{worktree}`, `{parent_commit}`, and `{branch}`.

## Cost Controls

The repo experiment config defaults to:

```yaml
max_mutations_per_generation: 1
max_elites_per_generation: 2
num_parents: 1
```

Keep this low until the benchmark and mutation brief are stable. Add cheap
`repo_harness.smoke_commands` before expensive benchmark runs.

### Codex Token And API-Cost Accounting

Codex-backed mutation, reflection, parent selection, and archive curation append
one normalized record per call to `<run_dir>/codex_usage.jsonl`. Each record
contains input, cached-input, uncached-input, output, reasoning-output, and total
tokens, plus the model and an estimated API-equivalent USD price. Cached input
is a subset of input and reasoning is a subset of output, so the estimator does
not charge either category twice.

The dashboard reads that ledger and shows:

- total run tokens and estimated API price in the header;
- a token and price breakdown for every evolution generation;
- token and price totals for each candidate, with call-level details in its
  `usage` tab.

The price is an estimate based on the dated rate card in
`gigaevo/llm/codex_usage.py`, not an OpenAI invoice. Subscription-backed Codex
runs have no per-call API charge, but the same number is useful as the estimated
cost if those calls were made with API billing. Unknown models remain
`unpriced` instead of silently using the wrong rate.

The `llm=codex` preset defaults to `gpt-5.6-sol` and standard, short-context
pricing. Override the accounting inputs when needed:

```bash
CODEX_MODEL=gpt-5.6-terra \
CODEX_SERVICE_TIER=standard \
CODEX_CONTEXT_TIER=short \
python run.py experiment=repo_harness ...
```

For a custom mutation backend, set `model` in the Codex command and optionally
set `repo_harness.mutation_backend.service_tier` or
`repo_harness.mutation_backend.context_tier`.

## Repo Insights And Lineage

The default repo-harness DAG is:

```text
RepoBenchmarkStage -> EnsureMetricsStage -> RepoReflectionStage -> MutationContextStage
```

`RepoReflectionStage` gives the coding-agent mutation more than raw scalar
metrics. For every candidate it gathers:

- parent and child commit ids
- changed files, `git diff --stat`, `git diff --name-status`, and a capped raw diff
- child benchmark feedback and parent benchmark feedback
- parent metrics, child metrics, and metric deltas
- generic structured feedback when the benchmark emits it

It then produces a concise retrospective markdown reflection with change
summary, lineage assessment, likely mechanism, and risks. That reflection is
routed into `MutationContextStage.formatted`, so
`RepoHarnessMutationOperator` includes it in `mutation_brief.md` for the next
coding-agent edit.

The reflection stage fails open by default. If the LLM call is unavailable, it
still returns a deterministic fallback containing diffstat and metric deltas, so
evolution can continue.

Useful knobs:

```yaml
repo_harness:
  reflection_max_diff_chars: 20000
  reflection_max_feedback_chars: 8000
  reflection_max_prompt_tokens: 150000
  reflection_prompt_token_encoding: o200k_base
  reflection_fail_open: true
```

Reflection prompt construction canonicalizes structured benchmark feedback,
removes repeated metrics and artifact manifests, and applies the final limit to
the complete rendered prompt with the configured tokenizer. Essential task,
metric-delta, diffstat, and changed-file evidence is kept as separate fields so
large optional diagnostics cannot remove it through whole-payload truncation.

## Safety Boundary

The mutation brief tells the coding agent not to edit benchmarks, metric parsers,
or sandbox policy. For stronger protection, point `source_repo` at a harness repo
that does not contain benchmark data, or enforce checks in `smoke_commands`.
