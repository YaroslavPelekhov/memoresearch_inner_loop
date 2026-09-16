# Terminal-Bench 2 Repo-Harness Benchmark

This folder adds a Terminal-Bench 2 benchmark command for the Git-backed
repo-harness problem. It expects Harbor to run TB2 tasks in containers, prints
GigaEvo metrics JSON on stdout, and emits structured mutation feedback on stderr
using the generic `[gigaevo] structured feedback:` marker.

By default, detailed Harbor job outputs are written under:

```text
problems/repo_harness_template/terminal_bench2/jobs/
```

## One-Task Smoke Run

```bash
python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --runs 1 \
  --concurrency 1
```

The default candidate repo is the local TB2 reference example, whose
`agents.baseline_terminus2:AgentHarness` starts from Harbor's Terminus-2
baseline.

If you are routing Harbor to a local llama.cpp OpenAI-compatible server, use the
local preset so Terminus-2 advertises the real context window to LiteLLM and
summarizes before llama.cpp reaches `n_ctx`:

```bash
python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --model openai/qwen-local \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --local-llama-defaults
```

Tune the preset with `TB2_LOCAL_CONTEXT_TOKENS`, `TB2_LOCAL_RESPONSE_TOKENS`,
and `TB2_LOCAL_SUMMARIZATION_THRESHOLD` if your server uses a different
`--n_ctx`. You can also pass raw Harbor agent kwargs with repeated
`--agent-kwarg key=value`.

To use OpenRouter instead of a local model, set an OpenRouter key and pass the
OpenRouter preset. The preset defaults to `openrouter/openai/gpt-oss-120b`,
`https://openrouter.ai/api/v1`, and the first key found in
`TB2_OPENROUTER_API_KEY`, `OPENROUTER_API_KEY`, `TB2_OPENAI_API_KEY`, or
`OPENAI_API_KEY`:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python problems/repo_harness_template/terminal_bench2/test_terminal_bench2_smoke.py \
  --candidate-repo /home/projects/giga_harness/meta-harness/reference_examples/terminal_bench_2 \
  --task extract-elf \
  --openrouter
```

Override the routed model with `--openrouter-model openrouter/<provider>/<model>`
or `TB2_OPENROUTER_MODEL`. Explicit `--model`, `--api-base`, and `--api-key`
still win if you need a custom OpenAI-compatible proxy.

## GigaEvo Usage

Pass the benchmark script as an absolute command so it is still visible from
GigaEvo's temporary candidate checkout:

```bash
python run.py \
  experiment=repo_harness \
  problem.name=repo_harness_template \
  repo_harness.auto_seed.enabled=true \
  'repo_harness.benchmark_command=["python","/home/projects/giga_harness/gigaevo-repo-harness/problems/repo_harness_template/terminal_bench2/benchmark.py","--task-set","smoke","--runs","1","--concurrency","1"]'
```

Use `--task-set full` for the full 89-task benchmark, `--task-set hard` for the
30-task cheaper subset, `--task-set balanced20` for a deterministic 20-task
mix across difficulties, or `--tasks task-a,task-b` / repeated `--task task-a`
for explicit task selection.

The auto-seed spec for this adapter is
`problems/repo_harness_template/terminal_bench2/repo_harness_seed.yaml`. It
creates an importable `agents.baseline_terminus2:AgentHarness` and vendors
Harbor's Terminus-2 architecture under `agents/terminus_2/`, so generation 0 is
a real baseline run and later mutations can edit the actual loop, parsers,
session handling, and prompt templates. Existing auto-seed repositories are
reused once they have a Git HEAD; use a new `redis.prefix` or remove the old
generated seed directory if you need an existing run name to pick up seed-spec
changes.

## Local Qwen / llama.cpp

Harbor agents need a chat-completions endpoint for repeated LLM calls. The
one-shot `qwen_llamacpp_infer.py` proves the GGUF works locally; for TB2, serve
the same GGUF through an OpenAI-compatible llama.cpp server and point Harbor at
it:

```bash
export TB2_MODEL=openai/qwen-local
export TB2_OPENAI_API_BASE=http://127.0.0.1:8000/v1
export TB2_OPENAI_API_KEY=dummy
export TB2_LOCAL_LLAMA_DEFAULTS=1
```

Then run the smoke test above. If Harbor is installed only in the candidate
repo's uv environment, pass `--runner "uv run"` or set `TB2_RUNNER="uv run"`.

## OpenRouter

The shorter environment-only form is:

```bash
export TB2_OPENROUTER=1
export OPENROUTER_API_KEY=sk-or-v1-...
export TB2_OPENROUTER_MODEL=openrouter/openai/gpt-oss-120b
```

## Results

Inspect the newest job with:

```bash
ls -lt problems/repo_harness_template/terminal_bench2/jobs
python -m json.tool problems/repo_harness_template/terminal_bench2/jobs/<job-name>/gigaevo_summary.json
```
