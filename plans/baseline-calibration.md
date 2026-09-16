# Baseline optimizer and score calibration

**Decision date:** 2026-08-21

Architecture ideation remains disabled until this calibration produces a fixed,
finite baseline contract and a representative DCLM CORE score distribution.

## Fixed contract

- Model: the 143,710,848-parameter baseline decoder and initialization.
- Data: pinned `dclm-baseline-3b-v1` source manifest and its verified MDS form,
  sampled once without virtual-epoch upsampling or weighted stream repetition.
- Tokenizer: the frozen local GPT-NeoX tokenizer recorded by corpus hash, with
  no implicit special tokens and exactly one explicit EOS between documents.
- Sequence/global batch: 2,048 tokens and 16 sequences (32,768 tokens/batch).
- Optimizer: Composer `DecoupledAdamW`, betas `(0.9, 0.95)`, epsilon `1e-8`,
  decoupled per-step weight decay `1e-4`.
- Schedule: cosine, 25% warmup, final/peak LR ratio `0.015`.
- Gradient clipping: Composer norm clipping at 1.0 with pre/post diagnostics.
- Fitness: raw equal-weight LLM Foundry CORE; checkpoints disabled.

These choices reproduce the supplied LLM Foundry contract. The warmup fraction
and effective weight decay also match the public DCLM 411M-1x recipe: 2,000 of
approximately 7,850 steps of warmup and `0.003 * 0.033 ~= 1e-4` decay per peak
step. Only peak learning rate is calibrated below.

## Phase A: numerical boundary

1. Preserve the observed `2e-3` failure at batch 3,151 as invalid evidence.
2. Require `1.5e-3` to clear 3,300 batches with finite loss/gradients.
3. On the pinned full corpus, run 3,300-batch diagnostics for `1.625e-3` and
   `1.75e-3`, with the fixed 25% schedule reaching peak LR after 825 batches.
   A setting is eliminated on any non-finite value.

## Phase B: representative 1B selection

Run the stable settings `1.25e-3`, `1.5e-3`, and the best stable Phase-A upper
setting for 30,518 batches with full CORE evaluation and seed 2048. Runs are
direct baseline benchmarks, never autoresearch elites or ideation memory.

Decision rule:

1. Eliminate invalid/non-finite runs.
2. Rank by `llmfoundry_core_equal_raw`.
3. Treat absolute CORE differences below `0.001` as unresolved rather than a
   win, matching `problems/llm_foundry_autoresearch/metrics.yaml`.
4. Repeat the leading settings with seed 2049. Select a default only when the
   ordering exceeds observed seed noise; otherwise retain the lower stable LR
   and record the tie.

## Phase C: confirmation and architecture gate

Run the selected default for 87,715 batches (rounded DCLM-1x) and full CORE.
Only after that run is finite and its artifacts contain validated data/tokenizer
hashes may the normal mandatory-review autoresearch baseline start. Candidate
architecture scores must use the same frozen contract and are not comparable to
the earlier smoke-corpus executions.

The handoff is fail-closed: it launches from the detached calibration source
commit only after revalidating the finite confirmation CORE score, suite and
confirmation contracts, corpus/evaluation hashes, clean source worktree, and
the reserved idle GPU UUID. It passes the calibrated LR and all data paths
explicitly; an invalid or ambiguous prior handoff cannot start autoresearch.

## Failure-triggered protocol amendment (2026-08-22)

The corrected one-EOS corpus invalidated every originally proposed upper
setting observed so far: `1.625e-3` and `1.75e-3` failed at batch 635, and the
representative `1.25e-3` run failed at batch 5,045. The `1.5e-3` screen then
failed at batch 4,218. The latter two failures occurred at nearly the same
effective LR (`8.263e-4` and approximately `8.29e-4`) despite different batch
indices. All failures were finite in the forward pass and then produced
non-finite gradients in the embedding and the same layer-0 attention/norm
parameters. Neither representative run produced CORE or can enter selection or
memory.

The failure-triggered `1.0e-3` screen subsequently failed at batch 5,977 with
the identical six-parameter signature. Batch 5,960 was still ordinary (loss
`4.4177`, global gradient norm `0.3159`, and maximum update/parameter ratio
`0.6231%`), so the transition again had no useful warning in the preceding
logged norms. Its effective LR was approximately `7.83e-4`, which broadens the
observed instability band downward and makes fail-fast rejection before the
optimizer mutation a required part of the contract. The `0.75e-3` screen
cleared its 7,630-batch warmup on seed 2048 with finite loss and gradients; this
is encouraging boundary evidence, not final stability evidence until its full
screen, seed repeat, and 87,715-batch confirmation complete.

If every original representative screen is invalid, extend Phase B without
changing any other contract field:

1. Screen `1.0e-3` and `0.75e-3` for the same 30,518 batches and exact CORE.
2. Only if both are invalid, screen `0.5e-3` as the last-resort stable anchor.
3. Apply the original seed-repeat, `<0.001` tie, and 87,715-batch confirmation
   rules to the first group containing at least one valid run.

This is a failure-triggered stability extension, not a post-hoc score search:
architecture ideation remains closed, invalid runs receive no CORE score, and
lower groups are never opened after a valid higher group is found. Direct runs
that bypass the calibrated handoff now default conservatively to `0.75e-3`
instead of the proven-invalid `1.5e-3`; scientific runs continue to receive an
explicit LR bound to their calibration contract.

The calibrated handoff also binds the confirmation's own structured-feedback
artifact, rather than accepting only a valid metric plus the current files. It
requires the emitted 87,715-batch physical-corpus contract and manifest hash,
143,710,848 parameter count, 15-task `EQUAL` CORE task contract and fixture
hashes, raw CORE event/tag, and exact effective-config artifact before launch.
