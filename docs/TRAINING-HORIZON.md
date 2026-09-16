# Training horizon for the first experiment

**Decision date:** 2026-08-20

The old 500-batch setting was only 16.4M tokens and is not a defensible model
comparison. The first experiment now separates two human-controlled horizons.

| Use | Tokens | Batches at global batch 16 × 2,048 | Default |
| --- | ---: | ---: | --- |
| Screening | 1,000,013,824 | 30,518 | yes, provisional |
| DCLM 1x confirmation | 2,874,245,120 consumed | 87,715 | no |

The confirmation value follows the public DCLM 1x convention of 20 training
tokens per parameter. The instantiated model has 143,710,848 parameters, so
its exact target is 2,874,216,960 tokens; rounding up to 87,715 complete global
batches consumes 2,874,245,120 tokens. DCLM itself publishes 1x recipes from
412M parameters upward; 140M is our extrapolation of the same convention, not
an official DCLM model scale.

Primary references: the [DCLM repository and model
table](https://github.com/mlfoundations/dclm) and its [411M 1x training
config](https://github.com/mlfoundations/dclm/blob/main/training/configs/411m_1x.json).
The accompanying paper is [DataComp-LM](https://arxiv.org/abs/2406.11794).

## Measured runtime

A 12-batch GPU-1 smoke on `2h100-airi` reached 34,018 tokens/s after the first
kernel compilation. At that observed single-GPU rate:

- 1B tokens takes about 8.2 hours;
- 2.874B tokens takes about 23.5 hours.

Near-linear scaling would put the 1B screen around 2.0 hours on four H100s, but
that must be measured because data loading, distributed communication, CORE
evaluation, and another process already sharing GPU 1 were not included in the
projection.

## What “significant difference” requires

No fixed token count guarantees that two decoder variants will separate on
DCLM CORE. The cutoff is an empirical property of the candidate effect size,
evaluation variance, and whether early rankings agree with final rankings.

Use independent baseline and representative-candidate runs at 0.25B, 0.5B,
1B, and 2.874B tokens. Repeat enough seeds at the proposed cutoff to estimate
score noise. Accept the earliest screening cutoff only when candidate score
deltas exceed that noise and its ranking agrees with the 2.874B confirmation
ranking. Until this calibration exists, 1B is a practical provisional
screen—not a claim of statistical sufficiency. Screening runs do not save
model checkpoints; this avoids unnecessary serialization and makes each run's
training/evaluation contract explicit.

The defaults are controlled outside candidate code:

```bash
# 1B screening
export AUTORESEARCH_MAX_DURATION=30518ba
export AUTORESEARCH_WARMUP_DURATION=7630ba
export AUTORESEARCH_EVAL_INTERVAL=30518ba

# DCLM 1x confirmation
export AUTORESEARCH_MAX_DURATION=87715ba
export AUTORESEARCH_WARMUP_DURATION=21929ba
export AUTORESEARCH_EVAL_INTERVAL=87715ba
```

## Baseline gradient diagnostic

Before selecting the fixed optimizer schedule, reproduce the current failure
on the unchanged baseline without evaluation or autoresearch:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m autoresearch.benchmark \
  --diagnostic-batches 3300 \
  --run-dir runs/baseline-gradient-diagnostic
```

The diagnostic run retains TensorBoard and records these global scalars every
batch:

- `l2_norm/grad/pre_clip_global`;
- `l2_norm/grad/post_clip_global`;
- `gradient_clipping/clip_coefficient`;
- `gradient_clipping/was_applied`;
- `gradient_clipping/nonfinite`.

Composer's native optimizer monitor additionally records post-clipping
per-parameter gradient, parameter, Adam moment, and update norms every 10
batches. This run is diagnostic evidence only and is excluded from scientific
baseline and ideation selection.

The 2026-08-21 sweep reproduced the first non-finite backward pass at batch
3,151 with peak LR `2e-3`; the preceding batch had finite loss and a raw global
gradient norm of only `0.3860`, so norm clipping could not prevent the abrupt
non-finite backward. Peak LRs `1e-3` and `1.5e-3` both completed 3,300 batches
with zero non-finite gradients. Over matched batches 3,001–3,150, `1.5e-3`
averaged loss `4.452657` versus `4.595235` for `1e-3`, so `1.5e-3` is the
selected stable baseline default.

## Corpus sufficiency gate

The first 37,422-sequence MDS conversion contains only 76,640,256 tokens. It is
a smoke/diagnostic fixture, not a scientific 1B-token corpus. The benchmark now
reads every configured MDS `index.json` before a normal run and rejects the
execution as `insufficient_training_corpus` unless the distinct tokenized
samples cover the complete requested horizon. Smoke and gradient-diagnostic
modes remain exempt because their evidence is explicitly excluded from
scientific selection.

`tools/prepare-dclm-corpus.py --convert` creates the pinned calibration corpus:
32 deterministic global/local partitions, one SHA-256-selected shard from each,
source LFS hash verification, a frozen local tokenizer, and source/corpus
manifests. Conversion fails if the result cannot cover the rounded DCLM-1x
confirmation horizon. Normal benchmarks verify the source-manifest, tokenizer,
and MDS-index hashes before launching a GPU process and include those identities
in structured execution evidence.
