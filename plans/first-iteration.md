# First iteration: run contract and open decisions

Status date: 2026-08-20. This is the operational state for producing the first
real baseline memory item and then one reviewed candidate memory item.

## Fixed for iteration 0

- Outer repository: this GigaEvo summer-school fork.
- Training/evaluation library: the vendored GigaChat LLM Foundry fork.
- Search unit: one decoder-layer implementation plus its constructor kwargs.
- Immutable experiment contract: model scale (candidate parameter count within
  1% of the 143,710,848 baseline), data, optimizer, schedule, evaluation tasks,
  seed, and resource limit.
- Execution: one experiment at a time; one GPU for the initial calibration.
- Evidence: every execution, including invalid and regressed runs, becomes an
  exact `ProgramCard`.
- Control flow: baseline execution -> baseline memory review -> idea proposal ->
  idea review -> Codex implementation -> candidate execution -> candidate memory
  review.
- Review mode: mandatory. Pending or rejected memory is never visible to the
  ideator.
- First-run objective: inspect the evidence and review interfaces. It is not yet
  evidence that short-horizon CORE reliably ranks architectures.

The old 500-batch/16.4M-token contract has been removed. The provisional
screening contract is 1,000,013,824 tokens (`30,518 * 16 * 2,048`). DCLM 1x
confirmation targets exactly 2,874,216,960 tokens (20 tokens per measured model
parameter); its `87,715` complete batches consume 2,874,245,120 tokens after
rounding.

## Remote state

- Orchestration environment: `/home/bulatov/envs/evo`.
- Isolated training environment: `/home/bulatov/envs/evo-lmf`.
- Codex CLI: `/home/bulatov/.local/bin/codex`; authentication and the external
  proxy runtime are installed and a real proxied request succeeded.
- Data root: `/home/bulatov/rmt/autoresearch/data`.
- Raw training input: one deterministic DCLM Baseline 1.0 smoke shard at
  `data/dclm-raw/shard_00000000_processed.jsonl.zst`.
- DCLM CORE fixtures: `data/dclm-core`; all 15 configured files are present.
- LLM Foundry runtime: Torch 2.7.0+cu128; a 12-batch GPU-1 smoke completed at
  about 34k warmed tokens/s with 143,710,848 parameters.

## Decisions and remaining calibration

### 1. Runtime dependency decision

The supplied LLM Foundry checkout declares these private submodules:

- `contrib/composer`, branch `gigachat3.5`
- `contrib/streaming`, branch `main`

Their directories were empty in the supplied source and anonymous access to the
declared GitLab URLs is denied. The current decision is an explicit focused
compatibility layer over public Composer/Streaming. Dense Gigar training is
verified; private-only features fail clearly instead of being silently faked.
If the private submodules become available, replace this compatibility layer
and rerun the smoke test.

### 2. Tokenizer/model template decision

Use `EleutherAI/gpt-neox-20b`, the public tokenizer named by DCLM's reference
recipe. The generated Gigar template pads its 50,277-token vocabulary to 50,304
and the instantiated model is 143,710,848 parameters.

### 3. Horizon calibration

The smoke establishes execution speed, not statistical significance. Run the
same baseline and a small set of nontrivial candidates to 0.25B, 0.5B, 1B, and
the final 2.874B-token checkpoint, evaluate CORE, and measure ranking agreement
and repeated-seed noise. Keep 1B only if its rankings predict the final result
and candidate deltas exceed noise. Until then, 1B is a provisional screen and
2.874B is the confirmation horizon.

### 4. Full training corpus

The converted smoke corpus is one shard, about 76.6M tokens. It is sufficient
for the execution smoke only. Download and convert enough distinct DCLM shards
before a 1B or 2.874B-token run; do not obtain the token budget by repeatedly
cycling this single shard.
The scientific loader must also leave Mosaic Streaming `epoch_size`,
`proportion`, `repeat`, and `choose` unset so that sufficient physical samples
are consumed without virtual-epoch resampling.

### 5. Fitness after the interface test

Keep equal-weight DCLM CORE as the primary fitness. If the calibration shows
that CORE at 1B tokens cannot rank candidates reliably, increase the screening
horizon. Adding held-out loss may be useful diagnostic evidence, but changing
the selection objective is a separate human decision.

## Bring-up sequence

1. Download and convert enough distinct DCLM shards for the chosen horizon.
2. Run the baseline benchmark and stop at the baseline memory review queue.
3. Review/redact/approve the baseline card.
4. Review one Codex-proposed idea and run one candidate.
5. Inspect the exact candidate `ProgramCard` before approving it for subsequent
   retrieval.
6. Use checkpoint results to validate or revise the 1B screening horizon.
