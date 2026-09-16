# Memory Autoresearch — Implementation Draft v1

**Updated:** 2026-08-20
**Status:** first implementation draft wired; dense GPU smoke passed

## 1. The first system we are actually building

The first implementation is a small extension of the GigaEvo summer-school repo harness around a normal LLM Foundry experiment.

It performs one fixed-scale loop:

```text
approved memory + incumbent evidence
                |
                v
      Codex proposes one idea
                |
       mandatory human review
                |
                v
       GigaEvo asks Codex to implement it
                |
                v
       normal LLM Foundry train/eval
                |
                v
 exact attempt evidence + Codex reflection
                |
                v
 exact ProgramCard enters the memory index
                |
       mandatory human review/redaction
                +--------------------> next iteration
```

For the initial run:

- fixed 143,710,848-parameter model derived from the supplied 7B config;
- DCLM Baseline 1.0 training data;
- the supplied LLM Foundry `core` evaluation task set;
- one candidate at a time;
- one GPU per candidate;
- `K=1`;
- a 1B-token screen should fit within roughly two to three hours on four H100s;
- Codex is both ideator and implementer;
- all idea and memory changes require human approval.

Scale, token budget, GPU allocation, and parallelism are human-controlled experiment parameters. They are not evolved by this loop.

## 2. What K means

`K` is simply the number of code implementations attempted for one approved
research idea. It is GigaEvo's existing `max_mutations_per_generation`
parameter, not another subsystem.

- `K=1`: one Codex implementation and one training run.
- `K=4`: four independent Codex implementations of the same approved idea.

K is not the number of ideas and not the number of GPUs. Candidate concurrency and GPUs per candidate are separate parameters:

```yaml
max_mutations_per_generation: 1  # K
candidate_concurrency: 1
gpus_per_candidate: 1
```

The summer-school `experiment=autoresearch` default currently has `K=4`; the first experiment must override it to 1. A later 8-GPU-node configuration could use `gpus_per_candidate=4` and `candidate_concurrency=2` without changing the research loop.

## 3. Simplicity rule

The implementation adds only three concepts to the repositories:

1. **A candidate decoder-layer seam in LLM Foundry.** Normal LLM Foundry configuration and training remain authoritative.
2. **A reviewed idea before GigaEvo mutation.** One approved idea is shared by all K implementations in a generation.
3. **A reviewed memory view around existing GigaEvo cards.** Existing card storage and retrieval remain in use; no separate semantic analyzer runs in v1.

There is no second evolution engine, custom training loop, custom experiment database, nested candidate repository, or large autoresearch controller.

The operational unit remains a normal GigaEvo problem in the outer repository:

```text
problems/llm_foundry_autoresearch/
├── task_description.txt
├── metrics.yaml
└── README.md
```

The outer repository, initially cloned from the summer-school fork, is the repository being evolved. Its focused `vendor/llm-foundry/` subtree is an immutable library during candidate evolution. Git commits and GigaEvo Programs remain the candidate and experiment ledger.

`python -m autoresearch.review` is intentionally a small file-based CLI, not a UI framework. It supports:

```text
autoresearch.review list
autoresearch.review show <id>
autoresearch.review approve <id> --note ...
autoresearch.review reject <id> --note ...
autoresearch.review redact <id> --field key=value
```

The same commands operate on idea proposals and memory cards.

## 4. What the summer-school fork already provides

We reuse these parts unchanged:

- evolution engine and stop policies;
- greedy `(1 + K)` autoresearch selection;
- Git worktrees and commits for candidate isolation;
- Codex coding-agent backend;
- deterministic repo benchmark stage;
- metric validation and archive replacement;
- repo-level Codex reflection;
- MemoryCard and ProgramCard storage;
- MemoryProvider retrieval;
- Redis state, lineage, logs, and usage accounting.

One correction to the earlier mental model: the current `repo_harness_autoresearch` pipeline does not contain separate ideator and implementer components. Its only agent call is the coding backend, which chooses a change and edits the repository in one step. It also explicitly disables reflection and memory.

Therefore strict idea approval before code requires one narrow new hook. It cannot be obtained by configuration alone.

## 5. Minimal GigaEvo changes

### 5.1 `ReviewedIdeaPreStepHook`

Use the engine's existing `pre_step_hook` seam.

Once per generation it:

1. on the first generation, waits for the seed baseline evaluation and invokes
   the same memory post-step so its ProgramCard is reviewed before ideation;
2. loads the incumbent's metrics, last repo reflection, and approved memory cards;
3. calls the existing Codex LLM wrapper in read-only mode;
4. validates one structured idea proposal;
5. writes it to the run review queue with status `pending`;
6. waits for human approval, rejection, or edit;
7. stores the approved idea as the active idea for that generation.

The idea contract is deliberately small:

```json
{
  "id": "idea-...",
  "generation": 3,
  "title": "...",
  "hypothesis": "...",
  "proposed_change": "...",
  "expected_mechanism": "...",
  "memory_ids": ["..."],
  "success_signal": "...",
  "risks": ["..."],
  "status": "pending"
}
```

An idea without valid approved memory citations is rejected by validation. The first baseline/root is the exception because it establishes experiment evidence rather than testing a generated idea.

Later, `review_mode=auto` changes only the gate policy; it does not change the proposal format or audit log.

### 5.2 Extend `RepoHarnessMutationOperator`, do not replace it

The existing operator still creates the worktree, calls the coding backend, runs smoke tests, commits changes, and records usage.

The extension only makes it:

- load the generation's approved idea;
- include that idea and the cited memory text in its existing mutation brief;
- require Codex to treat the idea as fixed and produce an implementation variant;
- record the approved idea and cited memory IDs in `mutation_output` for provenance.

For example:

```json
{
  "archetype": "approved_research_idea",
  "changes": [{
    "description": "<approved proposed_change>",
    "motivation": "<hypothesis and expected mechanism>"
  }],
  "insights_used": ["<memory id>"],
  "approved_idea_id": "idea-...",
  "changed_files": ["..."]
}
```

This closes a provenance gap: the repo operator otherwise records only
`repo_coding_agent_edit` and changed filenames, so the resulting memory card
would not identify which reviewed idea caused the execution.

With `K>1`, the pre-step hook still runs once and all K operator calls implement the same approved idea. This gives a clean distinction between research ideation and implementation variance.

### 5.3 Use the full repo evidence path

The problem pipeline should be the normal repo-harness evidence pipeline, not the stripped `repo_harness_autoresearch` pipeline. It should contain:

```text
RepoBenchmarkStage
  -> EnsureMetricsStage
  -> RepoReflectionStage
  -> MutationContextStage
```

`RepoReflectionStage` already calls the configured Codex LLM and records:

- parent and child metrics;
- metric deltas and verdict;
- Git diff and changed files;
- benchmark evidence and artifacts;
- a retrospective change/mechanism/risk analysis.

No new experiment summarizer is needed.

### 5.4 `ReviewedMemoryPostStepHook`

Use the engine's existing `post_step_hook` seam. It:

1. materializes exact ProgramCards for all completed candidate attempts;
2. detects new or changed cards and marks them `pending`;
3. waits for human approval, redaction, or rejection;
4. makes only the approved/redacted view available to the next ideator.

## 6. Memory extraction

### Exact execution evidence

`RepoReflectionStage` produces the canonical structured attempt record for every completed candidate and links to the full artifacts. This is the source of truth for:

- exact metrics;
- parent/child relation;
- validity and verdict;
- commit and diff;
- duration and artifacts;
- retrospective Codex reflection.

`autoresearch.program_memory.program_to_card` converts every attempt record into
a ProgramCard. This includes improved, neutral, regressed, and invalid attempts.
Negative results do not disappear merely because they did not enter the archive.

ProgramCard has been backward-compatibly enriched with defaulted fields:

```json
{
  "program_id": "...",
  "parent_program_id": "...",
  "fitness": 0.0,
  "metrics": {},
  "metric_deltas": {},
  "verdict": "improved|neutral|regressed|invalid",
  "commit": "...",
  "changed_files": [],
  "reflection": "...",
  "artifact_refs": {}
}
```

This is the one existing memory schema change required for v1. It makes experiment memory representative and human-verifiable without inventing a parallel experiment-record format.

### Human review overlay

Approval/redaction is stored as a sidecar keyed by card or idea ID:

```json
{
  "item_id": "...",
  "kind": "idea|memory|program",
  "status": "pending|approved|rejected|redacted",
  "original_hash": "...",
  "field_overrides": {},
  "reviewer_note": "...",
  "reviewed_at": "..."
}
```

The original extracted item stays immutable for auditability. After each review
gate, the post-step hook atomically publishes a second, approved-only card
index. `ReviewedMemoryProvider` searches only that index and applies human text
overrides. Rejected or still-pending cards therefore cannot influence search
ranking, synthesis, or the next idea. This avoids changing the existing
A-MEM/GAM retrieval internals.

There is no EvoMem semantic analyzer or second synthesized-memory layer in v1.
The exact reviewed cards are the memory. The result is a single
execution-grounded memory:

```text
ProgramCard: exact experiment evidence + reviewed human-readable fields
```

`ProgramCard` is already a memory-card type in the GigaEvo schema. No LLM
classification, clustering, merging, or semantic rewriting is required in the
first implementation.

### Papers and pre-existing experiments

Papers and old experiment reports may later enter through the same reviewed card
boundary, but they must not pretend to be executed ProgramCards. That importer
is outside the first executable loop.

## 7. LLM Foundry integration

### 7.1 Keep normal training conventions

The benchmark invokes the repository's normal training entrypoint with
PyTorch's standard distributed launcher:

```text
torchrun --standalone --nproc_per_node N scripts/train/train.py <effective-config.yaml>
```

Training, StreamingDataset/MDS loading, Composer distributed launch, callbacks, logging, checkpoints, and ICL evaluation remain LLM Foundry responsibilities.

GigaEvo's benchmark command is only a thin adapter that:

1. validates the allowed candidate diff;
2. builds an ordinary effective YAML from the frozen base and candidate fragment;
3. invokes the normal LLM Foundry command with the configured GPU count;
4. reads LLM Foundry's logged metrics;
5. prints GigaEvo's small metrics JSON.

LLM Foundry users can inspect and rerun the effective YAML directly without running GigaEvo.

### 7.2 Mutable decoder layer

The outer repository contains one explicit candidate area. The vendored LLM
Foundry package loads it through a registry seam:

```text
autoresearch/
├── model/decoder.py       # mutable implementation
└── config/candidate.yaml  # mutable decoder kwargs only
```

The allowed diff is these two files. The causal-LM wrapper, training code, evaluation, dataset settings, optimizer, and base model dimensions are immutable within the experiment definition.

There is one small, permanent LLM Foundry integration patch:

```python
# GigarConfig
decoder_layer_type: str = "LlamaDecoderLayer"
decoder_layer_kwargs: dict = {}

# GigarModel
layer_cls = BLOCK_CLASS_REGISTRY[config.decoder_layer_type]
self.layers = nn.ModuleList([
    layer_cls(config, layer_idx, **config.decoder_layer_kwargs)
    for layer_idx in range(config.num_hidden_layers)
])
```

`autoresearch/model/decoder.py` registers its class in the existing `BLOCK_CLASS_REGISTRY`. `candidate.yaml` can add arbitrary constructor arguments under `decoder_layer_kwargs`; the deterministic benchmark adapter merges only this allowed subtree into the frozen base config.

This is preferable to runtime monkeypatching because normal LLM Foundry training scripts construct the selected layer directly, the effective config records every argument, and downstream users can use the mechanism without GigaEvo.

The baseline value remains `LlamaDecoderLayer`, so existing configurations are unaffected.

### 7.3 Supplied config findings

The added `dclm-7b-140b-baseline_exp.yaml` is a valuable source config, but it is not yet the first-run config:

- it defines a 7B model (`hidden_size=4096`, 32 layers), not a 100–140M model;
- it schedules 140B tokens (`max_duration: 33500ba`), not a two-hour candidate;
- it has an unresolved local MDS path placeholder;
- it sets `hybrid_group_size: 8`, which is not the initial one-GPU setup;
- it uses `model.name: giga_causal_lm`, while this checkout registers `gigar_causal_lm`.

The first implementation therefore derives rather than hand-reinterprets a
small config from this file. It preserves the field names, data-loader form,
optimizer/scheduler style, callbacks, loggers, and ICL task definitions while
changing the explicitly fixed scale, budget, and distributed parameters.

### 7.4 DCLM data

The user-facing dataset identity is `mlfoundations/dclm-baseline-1.0`, but this LLM Foundry pretraining path expects tokenized Mosaic Streaming/MDS data. The supplied YAML confirms that convention with a local MDS stream and split name `dclm-baseline-1.0`.

Therefore data preparation is a one-time immutable prerequisite:

```text
HF DCLM Baseline 1.0
  -> fixed snapshot/shard subset
  -> LLM Foundry tokenization/conversion
  -> fixed MDS directory
  -> train_loader.dataset.streams.dclm-baseline.local
```

Dataset download, shard selection, tokenizer, sequence length, and resulting MDS manifest hash are frozen outside evolution. Candidate code never controls them.

### 7.5 DCLM-CORE fitness naming

The supplied LLM Foundry config tags the intended tasks with `gauntlet_tags: [core]` and uses equal weighting. In this checkout `EvalGauntlet` defaults to neither subtracting nor rescaling random baselines, so `metrics_gauntlet/core` is a raw equal average of the listed task accuracies.

For the first run, use that exact LLM Foundry metric because compatibility is the goal, but name it explicitly in records, for example:

```text
llmfoundry_core_equal_raw
```

Do not silently label it as the official DCLM Core v2 score. An official-v2 calculation can later be added as a second diagnostic without changing the primary v1 metric.

## 8. Initial run parameters

```yaml
autoresearch:
  review_mode: mandatory
  ideas_per_generation: 1
  implementations_per_idea: 1   # K
  candidate_concurrency: 1
  gpus_per_candidate: 1
  candidate_wall_time: human-owned timeout
  mutable_paths:
    - autoresearch/model/decoder.py
    - autoresearch/config/candidate.yaml
```

The GPU settings must become launcher parameters now even though the first values are 1. Later configurations can change allocation without modifying GigaEvo or LLM Foundry code.

The run uses two independent bounds:

- a per-candidate timeout, defaulting to 12 hours so a one-GPU 1B run is not
  killed, while the intended four-GPU screen is approximately two hours;
- a run-level wall-clock stopper, initially one or two weeks.

## 9. Build and validation order

### Milestone 1 — native baseline (implemented; smoke verified)

1. Correct the supplied model registry name.
2. Derive the fixed small-scale/small-duration config.
3. Prepare and hash the DCLM MDS dataset.
4. Run the baseline through normal LLM Foundry on one GPU.
5. Confirm the configured `core` metric and artifact locations.

### Milestone 2 — mutable layer through repo harness (implemented)

6. Add the registry/config seam for decoder type and kwargs.
7. Create a baseline-equivalent autoresearch decoder.
8. Add forward/backward and config-construction smoke tests.
9. Run the same baseline through one GigaEvo repo-harness candidate.
10. Enforce the two-file diff allowlist.

### Milestone 3 — reviewed idea (implemented; live review still to exercise)

11. Add `ReviewedIdeaPreStepHook` and the review CLI.
12. Make one Codex ideator call and manually approve it.
13. Pass it through the existing mutation brief to one Codex implementer.
14. Store the structured idea in `mutation_output`.

### Milestone 4 — reviewed memory (implemented; live review still to exercise)

15. Re-enable RepoReflectionStage.
16. Adapt every attempt record into an enriched ProgramCard.
17. Add the approval/redaction sidecar and reviewed provider.
18. Verify that the next idea cites only approved ProgramCards.

### Milestone 5 — endurance

20. Test interruption/resume while waiting for review and during training.
21. Parameterize GPU allocation and concurrency.
22. Run a short multi-iteration trial.
23. Start the bounded week-long run.

## 10. Remaining decisions before the first scientific baseline

Two empirical values still need to be frozen:

1. whether the provisional 1B-token screen predicts the 2.874B-token
   confirmation ranking above repeated-run noise;
2. the exact multi-shard DCLM snapshot used for those budgets.

The fixed model shape is 16 layers, hidden size 640, MLP size 1,728, and ten
attention/KV heads with a 50,304-entry padded public GPT-NeoX vocabulary. LLM
Foundry measured 143,710,848 parameters and validates that count against the
130–150M guard before training. A 12-batch GPU-1 smoke reached about 34k warmed
tokens/s. The remaining work is ranking/noise calibration, not another model
shape or system redesign.

The repository prerequisite is now resolved: the outer project is a Git clone of the summer-school fork, and candidate commits are made there. LLM Foundry is a focused vendored library rather than a nested candidate repository.

## 11. First code milestone

The first code milestone now includes the reviewed memory loop:

> One reviewed idea produces one allowlisted Codex implementation; an ordinary
> LLM Foundry config trains and evaluates it; every attempt becomes an exact
> ProgramCard; and each card passes through mandatory review before the next
> idea.

The remaining proof is operational rather than architectural: prepare the full
training snapshot, run the baseline, and exercise at least two reviewed
iterations while checking whether the 1B screen predicts the 1x result.
