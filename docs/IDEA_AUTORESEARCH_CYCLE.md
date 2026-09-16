# Idea-driven DCLM autoresearch: code and diagnostic map

The system has two levels:

1. The outer loop selects and studies a durable research idea.
2. The inner GigaEvo loop searches for implementations of that fixed idea.

```text
fixed task + previous ideas + previous takeaways
                         |
                         v
              generate five proposals
                         |
                         v
                 approve one idea
                         |
                         v
       select a suitable implementation parent
                         |
                         v
                 create one seed
                         |
                         v
             evolve implementations
                         |
                         v
           benchmark implementation attempts
              at 1,024 training batches
                         |
                         v
       confirm the best attempt at 4,096 batches
                         |
                         v
        verified evidence -> idea-level takeaways
                         |
                         +----> next outer round
```

Raw run results do not feed the next idea-generation call. The next call sees
the fixed task, the complete idea catalog, and the idea-level takeaway bank.

## 1. Main loop

Start with `autoresearch/ideas/campaign.py`:

- `IdeaCampaign.run()` controls outer rounds and campaign state.
- `_run_round()` contains the complete idea-to-takeaway cycle.
- `_filter()` implements Codex or human selection.
- `_parent_candidates()` returns canonical plus a score-, idea-diversity-, and
  recency-aware set of valid implementations.
- `_create_seed()` creates exactly one implementation seed.
- `_run_evolution()` starts fixed-idea GigaEvo.
- `_load_evidence()` verifies Git lineage and builds the evidence ledger.
- `_record_failure()` records campaign, active-idea, and version failures.

The executable entrypoint is `tools/run-idea-campaign`. It selects the campaign
directory, data paths, Redis, Python environment, and visible GPU before calling
`python -m autoresearch.ideas.campaign`.

## 2. Research task and fixed boundary

`problems/llm_foundry_autoresearch/research_task.yaml` defines:

- the fixed Transformer/GDN task;
- immutable training, data, evaluation, topology, and scale constraints;
- `autoresearch/model/gdn.py` as the only mutable file;
- five idea proposals per round;
- eight eligible evolved parents plus canonical;
- four inner evolution generations by default.

`problems/llm_foundry_autoresearch/task_description.txt` is the longer fixed task
description inserted into every mutation brief.

## 3. Typed research objects

`autoresearch/ideas/models.py` defines:

- `Idea`: hypothesis, mechanism, implementation direction, success criteria,
  risks, source idea IDs, takeaway IDs, and status.
- `IdeaVersion`: selected parent, immutable base commit, implementation prompt,
  seed commit, mutable files, evolution budget, and status.
- `Implementation`: a concrete Git commit, parent, 1,024-step screen metrics,
  optional 4,096-step confirmation metrics, validity, changed files, and
  evidence references.
- `Takeaway`: a typed, deduplicated conclusion with supporting or contradicting
  implementation IDs. Operational feasibility is separate from scientific
  mechanism support.
- `CampaignState`: campaign progress and the active idea.

An idea is therefore not just a prompt. Its initialization is the
`IdeaVersion`: selected parent plus exactly one seed, together with the mutable
and immutable boundary.

## 4. Codex prompts

`autoresearch/ideas/agents.py` contains the outer structured calls:

- `generate_ideas()`: task + idea catalog + takeaway bank -> five proposals.
- `select_idea()`: proposals + research history -> one approved idea.
- `select_parent()`: approved idea + top implementations -> suitable parent.
- `synthesize_takeaways()`: complete verified lineage -> typed takeaways.

`IdeaCampaign._implementation_prompt()` builds the seed prompt. The exact prompt
and agent output are persisted under `prompts/seed/`.

The generic inner mutation prompt is `DEFAULT_MUTATION_PROMPT` in
`gigaevo/evolution/mutation/repo_harness_operator.py`. That operator writes a
`mutation_brief.md` containing the fixed task, frozen approved idea, selected
parent set, parent feedback, and benchmark evidence.

## 5. Model boundary

`autoresearch/model/hybrid_decoder.py` owns the immutable topology:

```python
GDN_LAYER_INDICES = frozenset(range(0, 16, 2))
```

Layers 0, 2, ..., 14 use GDN. Layers 1, 3, ..., 15 use unchanged Transformer
attention. Evolution cannot change this placement.

`autoresearch/model/gdn.py` is the canonical mutable baseline. It contains the
GDN wrapper and DCLM-compatible initialization. Evolved implementations do not
overwrite the main checkout; each is stored as a separate Git commit.

Inspect an implementation with:

```bash
git show <commit>:autoresearch/model/gdn.py
git diff <parent-commit> <commit> -- autoresearch/model/gdn.py
```

## 6. Inner evolution configuration

`config/experiment/llm_foundry_idea_evolution.yaml` fixes:

- `pre_step_hook: null`, so inner evolution cannot replace the idea;
- one parent and one mutation per generation;
- the GDN-only file allowlist;
- Codex mutation backend;
- dry-run validation and benchmark command;
- automatic extraction/review of implementation evidence.

`autoresearch/config/dclm-140m.yaml` contains the fixed model/training setup and
WSD schedule: warmup to 25%, stable through 90%, cosine decay over the final
10%, ending at 10% of peak LR rather than zero.

Every evolution attempt follows the checkpoint cascade in
`config/multifidelity/dclm-140m.yaml`. The checked-in protocol runs to 128, 256,
1,024, 2,048, and 4,096 batches while preserving one fixed 4,096-batch main LR
schedule. Each non-final rung also runs an isolated LR-to-zero convergence
probe. The initial campaign is observe-only: every candidate reaches the final
rung so that later probability models and gates can be calibrated without
survivorship bias. The fixed, physically disjoint DCLM validation split is
prepared once with `tools/prepare-dclm-holdout.py`. The detailed decision and
calibration contract is in `docs/MULTIFIDELITY_CONVERGENCE_PROBES.md`.

## 7. Two kinds of memory

### Outer research memory

At campaign root:

- `idea-catalog.jsonl`: every proposed, rejected, approved, completed, or failed
  idea.
- `takeaway-bank.jsonl`: the only experimental conclusions supplied to later
  idea generation.
- `implementation-index.jsonl`: concrete commits used for lineage inspection and
  future parent selection.

Storage is implemented in `autoresearch/ideas/store.py`.

### Inner evolution evidence

Under each idea version:

- `evidence/extracted/api_index.json`: all extracted attempts, including invalid
  and regressed implementations.
- `evidence/approved/api_index.json`: reviewed GigaEvo memory cards.
- `evidence/ledger.json`: the outer loop's compact, Git-verified lineage.

The outer loop recomputes parent-to-child Git diffs. This prevents a seed that
is unchanged relative to itself inside GigaEvo from being incorrectly described
as unchanged relative to its selected parent.

## 8. Accepted and rejected ideas

All proposals remain in `idea-catalog.jsonl`. The chosen proposal becomes
`approved`, then `completed` after takeaways are written. All other proposals
from the same batch become `rejected`.

For each accepted idea, `selection.json` records:

- Codex or human filter mode;
- why the idea was selected;
- why a particular parent implementation was selected.

Derived ideas carry `source_idea_ids` and `takeaway_ids`, making combinations and
research ancestry explicit.

## 9. Per-idea diagnostic directory

Inspect `runs/<campaign>/ideas/<idea-id>-<title>/v001/` in this order:

```text
selection.json                 why this idea and parent were selected
idea.json                      exact frozen idea seen by all mutations
idea-version.json              parent, seed, prompt, budget, status
prompts/seed/prompt.md         exact seed-agent prompt
prompts/seed/stderr.txt        seed-agent tool trace
evolution.log                  outer view of the inner GigaEvo process
system/run.log                 detailed GigaEvo log
system/repo_mutation/logs/     per-mutation briefs, prompts, traces, smoke checks
implementations/impl-*/        benchmark outputs grouped by commit
evidence/ledger.json           verified lineage sent to takeaway synthesis
confirmations/impl-*/          one 4,096-step confirmation for the idea winner
takeaways/final.json           conclusions for this idea version
failure.json                   failure record, when present
```

Inside each implementation's `training/` directory, inspect:

```text
effective-config.yaml
train.stdout.log
train.stderr.log
launch.log
tensorboard/
```

Each evidence row contains structured screen/confirmation metrics, deltas from
the same-horizon canonical baseline, failure stage and type, stderr tail, and
peak GPU memory when available. `completed` means that the cycle finished;
`scientific_status` separately records preliminary, supported, refuted, or
inconclusive evidence.

For overnight operation, `tools/run-idea-campaign-supervised.py` runs one round
at a time to a fixed total, retries failed rounds, and takes a file lock.
`tools/watch-idea-campaign.py` records GPU/campaign state and restarts a dead or
hour-stale supervisor. Its monitor exits after the target round count.

## 10. Fast diagnosis commands

```bash
cd /home/bulatov/rmt/autoresearch/autoresearch-dclm-ideas
run=runs/idea-multiround-validation

cat "$run/campaign-state.json"
jq . "$run/idea-catalog.jsonl"
jq . "$run/takeaway-bank.jsonl"
jq . "$run/implementation-index.jsonl"

find "$run/ideas" -name selection.json -print
find "$run/ideas" -name idea-version.json -print
find "$run/ideas" -path '*/agent/prompt.md' -print
find "$run/ideas" -path '*/evidence/ledger.json' -print

tail -f runs/idea-multiround-validation.launch.log
tmux has-session -t dclm-idea-multiround && echo running
nvidia-smi
```

For a first code-reading pass, use this order:

1. `autoresearch/ideas/campaign.py`
2. `autoresearch/ideas/agents.py`
3. `autoresearch/ideas/models.py`
4. `autoresearch/ideas/store.py`
5. `config/experiment/llm_foundry_idea_evolution.yaml`
6. `gigaevo/evolution/mutation/repo_harness_operator.py`
7. `autoresearch/model/hybrid_decoder.py`
8. `autoresearch/model/gdn.py`
9. One real directory under `runs/<campaign>/ideas/`
