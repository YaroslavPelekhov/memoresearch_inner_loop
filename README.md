# Idea-driven DCLM autoresearch

This branch moves the unit of research above individual code mutations. The
outer loop studies a durable **research idea**; the inner GigaEvo loop searches
for implementations of that one idea.

```text
idea catalog + takeaway bank
            |
            v
 generate proposals -> filter one idea -> select a suitable implementation base
                                              |
                                              v
                                   create exactly one seed
                                              |
                                              v
                                  evolve implementations
                                              |
                                              v
                              evidence -> idea-level takeaways
                                              |
                                              +----> next idea round
```

Raw run results never feed idea generation directly. At the end of an inner
evolution, Codex sees the complete implementation lineage and produces typed,
provenance-carrying takeaways. The next proposal call receives only the idea
catalog, the takeaway bank, and the fixed research task. Ideas may cite and
combine multiple earlier ideas.

## Fixed and mutable boundaries

The model is a fixed 16-layer alternating hybrid:

- odd layers use unchanged Llama Transformer attention;
- even layers use `AutoresearchGatedDeltaNet`;
- evolution may edit only `autoresearch/model/gdn.py`;
- layer placement, Transformer implementation, training loop, DCLM data,
  evaluation, model-scale guard, optimizer peak LR, and scheduler are immutable.

The scheduler is warmup-stable-decay (WSD): warm up to the peak LR over the
first 25%, stay at peak through 90%, then cosine-decay during the final 10%.
The run ends at 10% of peak rather than zero, leaving continuation headroom.

Candidate evaluation now uses an observe-only multi-fidelity cascade. The main
trajectory keeps that fixed full-horizon scheduler, while isolated checkpoint
forks briefly decay LR to zero and measure reachable local quality. See
[`docs/MULTIFIDELITY_CONVERGENCE_PROBES.md`](docs/MULTIFIDELITY_CONVERGENCE_PROBES.md)
for the hypotheses, invariants, gate calibration, and activation criteria.

## One outer round

Each research stage is a separate Codex call except the benchmark itself:

1. Generate five candidate ideas from prior ideas and takeaways.
2. Select one idea. `codex` is the default filter; `human` is also supported.
3. Select the most suitable parent from the current top valid implementations
   plus the canonical baseline. Suitability can outweigh a small fitness gap.
4. Create one seed implementation from that parent.
5. Run a fixed four-generation evolution whose mutation prompts all contain
   the same approved idea.
6. Index every implementation attempt, including invalid and regressed ones.
7. Synthesize typed idea-level takeaways with implementation-ID citations.

The fixed task is readable in
`problems/llm_foundry_autoresearch/research_task.yaml`; the inner evolution is
`config/experiment/llm_foundry_idea_evolution.yaml`.

## Run layout

A campaign is intentionally file-first and inspectable:

```text
runs/<campaign>/
  campaign-state.json
  idea-catalog.jsonl
  takeaway-bank.jsonl
  implementation-index.jsonl
  codex-usage.jsonl
  ideas/<idea-id>-<title>/v001/
    selection.json
    idea.json
    idea-version.json
    prompts/seed/
    evolution.log
    system/
    implementations/impl-<commit>/training/
      multifidelity-trajectory.json
      main/budget-*/
      probes/budget-*/
    evidence/
      extracted/api_index.json
      approved/api_index.json
      ledger.json
    takeaways/final.json
```

The naming boundary is `idea / idea-version / implementation`. A new idea gets
one version initially; later versions can reuse the idea without mixing their
evidence.

## Run it

Use Python 3.12 and provide the fixed external artifacts:

```bash
export AUTORESEARCH_PYTHON=/path/to/python3.12
export DCLM_MODEL_PATH=/path/to/model-template-gpt-neox-140m
export DCLM_MDS_PATH=/path/to/dclm-mds-3b-v1
export DCLM_CORE_PATH=/path/to/dclm-core
export DCLM_EVAL_CACHE_PATH=/path/to/eval-cache
```

Validate composition without a GPU:

```bash
$AUTORESEARCH_PYTHON -m autoresearch.benchmark --dry-run
$AUTORESEARCH_PYTHON -m pytest -q tests/autoresearch
```

Start one Codex-filtered idea round on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
AUTORESEARCH_CAMPAIGN_NAME=idea-campaign-001 \
tools/run-idea-campaign --rounds 1
```

For a cheap pipeline check, use smoke benchmarks and one evolution generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
AUTORESEARCH_SMOKE_BATCHES=12 \
AUTORESEARCH_CAMPAIGN_NAME=idea-smoke \
tools/run-idea-campaign --rounds 1 --evolution-generations 1
```

Smoke runs skip CORE and report zero fitness. They validate execution and
evidence plumbing only and must not be treated as scientific comparisons.

For human filtering, the first call persists proposals and exits cleanly:

```bash
tools/run-idea-campaign --filter-mode human --rounds 1
```

Inspect `pending-human-selection.json`, then resume the same campaign root:

```bash
AUTORESEARCH_CAMPAIGN_ROOT=/path/to/the/campaign \
tools/run-idea-campaign \
  --filter-mode human \
  --selected-idea-id idea-0003 \
  --rounds 1
```

## Scientific boundary

Shortened runs are useful while debugging the idea machinery, but their
takeaways must state the shortened horizon. A single winning implementation is
not confirmation: takeaway synthesis records uncertainty whenever replication
or a noise-aware comparison is absent.

The original small MDS fixture is suitable only for smoke tests. Use
`tools/prepare-dclm-corpus.py --convert` before claiming a representative
long-horizon result.
