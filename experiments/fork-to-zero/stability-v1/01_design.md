# Experimental Design: Fork-to-zero as an instability detector

**Date**: 2026-09-21
**Status**: Prepared locally; not launched

## Research question

Does a short learning-rate descent to zero reveal candidates that later suffer
model-intrinsic optimization collapse earlier or more reliably than an
equal-compute continuation of the original schedule?

This is deliberately narrower than the rejected ranking claim. The treatment
is a stress test, and the target is catastrophic instability rather than final
candidate quality.

## Hypotheses

- **H0:** At specificity at least 0.90, the preregistered Fork-to-zero alarm has
  no higher collapse sensitivity than the matched continuation alarm.
- **H1:** At specificity at least 0.90, Fork-to-zero improves collapse
  sensitivity by at least 0.15 and the campaign-bootstrap 95% lower confidence
  bound for the paired improvement is above zero.

## Treatment and controls

At main checkpoints 512 and 1,024, two read-only branches run for exactly 256
batches:

- matched control: unchanged 4,096-batch LR schedule;
- treatment: cosine LR descent from the checkpoint LR to zero.

The main branch is untouched and targets batch 4,096. Main-path gradient
telemetry is a secondary comparator.

## Frozen outcome definition

A candidate is a positive collapse case after a checkpoint if either:

1. the later main branch reports `nonfinite_training_gradient` or
   `nonfinite_training_loss`; or
2. it completes batch 4,096 and its final held-out loss is at least 10% above
   the held-out loss at that checkpoint.

CUDA OOM, checkpoint incompatibility, storage, network, timeout, and
orchestration failures are censored rather than labeled as collapse. A
completed run below the 10% loss-increase threshold is negative.

## Frozen detector definitions

- Fork alarm: its branch is intrinsically non-finite, or
  `probe_fitness - control_fitness <= -0.002`.
- Matched-control alarm: its branch is intrinsically non-finite, or
  `control_fitness - main_fitness <= -0.002`.
- Gradient alarm, secondary only: main-path `gradient_clipping_fraction >= 0.5`
  or `gradient_norm_pre_clip_p95 >= 10.0`, or a non-finite main gradient.

The `0.002` margin is far above measured replicate noise in the exploratory
campaign and was fixed after seeing that campaign but before collecting any
confirmation data.

## Primary and secondary metrics

The unit of inference is the independently started campaign, not a candidate
commit or an evolutionary lineage.

- Primary: candidate-level sensitivity difference, Fork minus matched control,
  using each detector's earliest alarm across checkpoints.
- Required safety constraint: Fork specificity at least 0.90.
- Secondary: precision, false-positive rate, balanced accuracy, AUROC and
  average precision of continuous detector scores, and detection lead time in
  main-branch batches.
- Descriptive comparator: the same metrics for gradient telemetry.

Uncertainty uses a paired bootstrap over whole campaigns. All candidates and
checkpoints from a sampled campaign move together. Candidate-level intervals
may be reported only as descriptive sensitivity analysis.

## Run design and sample sufficiency

- Eight fresh campaign roots, four assigned to each of GPUs 0 and 1.
- Each campaign starts from the same frozen canonical commit, runs one idea
  round, and shares no candidate parents or memory with another campaign.
- Six inner evolution generations per campaign, targeting roughly 48 new
  experimental candidates in total.
- Campaigns are launched in four sequential two-GPU waves.

The result is automatically **inconclusive** unless all eight campaigns are
present, at least 10 eligible positive candidates and 20 eligible negative
candidates are observed, and both classes occur on each GPU. No extra
candidates may be added after inspecting the primary effect; a new protocol
version is required.

## Decision rule

The instability-detector claim is supported only when all conditions hold:

1. sample-sufficiency conditions pass;
2. Fork specificity is at least 0.90;
3. Fork sensitivity minus control sensitivity is at least 0.15;
4. the campaign-bootstrap 95% lower bound of that difference is above zero;
5. the direction is positive separately on both GPUs.

Otherwise the result is null or inconclusive according to the frozen sample
rules. Candidate-ranking performance is explicitly outside this experiment.

## Compute budget

Each candidate uses 4,096 main batches plus two pairs of 256-batch shadows:
5,120 batch-equivalents, a 25% overhead over full training. The planned target
is approximately 48 new candidates plus one canonical baseline per campaign.
No GPU work starts as part of repository preparation.

## Known risks

- Stability-boundary proposals may still produce too few positive cases.
- Evolution within a campaign creates dependent candidates; campaign bootstrap
  preserves that dependence but eight clusters remain a modest sample.
- A stress response can be real without being specific to future collapse;
  the specificity constraint prevents that from becoming a positive claim.
- Failures before checkpoint 512 are not detectable by this protocol and are
  reported separately.
