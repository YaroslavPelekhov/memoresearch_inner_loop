# Pre-registration: Fork-to-zero instability detector v1

**Date**: 2026-09-21
**Protocol version**: 1.0
**Status**: Frozen for code review; launch commit to be filled after commit
**Design**: `experiments/fork-to-zero/stability-v1/01_design.md`
**Analysis**: `analysis/fork-to-zero-stability-v1/analyze.py`

## Frozen claim

Fork-to-zero is an early stress test for later model-intrinsic optimization
collapse. It is not claimed to improve general candidate ranking.

## Frozen constants

| Quantity | Value |
|---|---:|
| Main horizon | 4,096 batches |
| Fork checkpoints | 512 and 1,024 batches |
| Shadow length | 256 batches per branch |
| Fitness alarm margin | -0.002 |
| Terminal loss increase defining collapse | 10% |
| Gradient clipping-fraction alarm | 0.50 |
| Gradient p95 norm alarm | 10.0 |
| Campaigns | 8 |
| Inner generations per campaign | 6 |
| Bootstrap resamples | 20,000 |
| Random seed | 20260921 |

## Frozen exclusions

- Canonical baselines are assay controls and excluded from primary inference.
- Repeated commits are aggregated within campaign before inference.
- CUDA OOM, checkpoint loading or serialization errors, storage exhaustion,
  network failures, timeouts, and orchestration failures are censored.
- Missing or invalid matched branches make that checkpoint ineligible, unless
  the branch itself failed with a model-intrinsic non-finite loss or gradient.
- A run failing before checkpoint 512 is counted in the audit but cannot enter
  detector evaluation.

## Frozen analysis order

1. Verify that every campaign uses the preregistered plan hash.
2. Classify failures without using branch detector values.
3. Create one candidate record with earliest alarms across eligible checkpoints.
4. Aggregate repeated commits within campaign conservatively: any intrinsic
   collapse is positive and any alarm is positive.
5. Check sample sufficiency before interpreting the primary effect.
6. Compute specificity, sensitivity and the paired campaign-bootstrap interval.
7. Check the two GPU strata for direction consistency.
8. Report secondary continuous-score and lead-time metrics.

## Success rule

The claim is supported only if all five checks in the design document pass.
No threshold tuning, alternate collapse label, subgroup removal, or additional
campaign may replace this primary result. Such analyses must be labeled
exploratory and committed as a new protocol version.

## Launch matrix

Campaign names are frozen as:

- `fork-zero-stability-v1-gpu0-pool1` through `pool4`;
- `fork-zero-stability-v1-gpu1-pool1` through `pool4`.

Each pool uses `--target-rounds 1 --evolution-generations 6`, the stability-v1
task, and the stability-v1 multi-fidelity plan. Pool N+1 on a GPU starts only
after pool N has fully exited. GPU availability must be checked immediately
before every launch. The frozen wrapper is
`tools/launch-fork-to-zero-stability-v1 --gpu GPU --pool POOL`; without
`--start` it performs validation only.

## Amendments

None.
