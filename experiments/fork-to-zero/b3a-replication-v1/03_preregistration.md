# Pre-registration: b3a210 replication and onset map v1

**Date**: 2026-09-22
**Protocol version**: 1.0
**Status**: Frozen before launch

## Frozen matrix

| Quantity | Value |
|---|---:|
| Candidate source | `b3a210ded839250e22eb148718fc9d401ec90052` |
| Parent source | `0369584bf978a6850f96d29d3dc07fe00f32aaa9` |
| Seeds | `31001, 31002, 31003, 31004, 31005, 31006` |
| Main horizon | 4,096 batches |
| Fork checkpoints | 512, 768, 1,024 |
| Shadow length | 256 batches per branch |
| Alarm margin | -0.002 fitness |
| Collapse loss ratio | 1.10 |
| Replicates per variant | 6 |

The GPU assignment is balanced and frozen: candidate/parent are assigned to
0/1 on odd-numbered pairs and 1/0 on even-numbered pairs.

## Frozen analysis

1. Verify plan, harness, model-file hashes, seeds, and assignment from the
   campaign manifest.
2. Censor infrastructure failures without consulting alarm values.
3. Determine collapse separately at every eligible checkpoint.
4. Record the earliest Fork and continuation alarm for every run.
5. Evaluate the candidate-parent collapse contrast.
6. Evaluate whether the candidate's Fork alarm precedes its matched control.
7. Report all six pairs, including failures and ties; do not replace runs after
   inspecting outcomes.

## Success rules

- Replicated mutation effect: candidate collapse count at least 4/6 and parent
  collapse count at most 1/6.
- Fork early-warning effect: Fork alarms before final collapse and strictly
  earlier than continuation in at least 4/6 candidate runs.
- Full support requires both rules. A result that passes only the first rule is
  evidence for the mutation mechanism, not for a unique Fork-to-zero detector.

Any rerun caused by a censored infrastructure failure must keep the same seed,
variant, GPU assignment, source hashes, and output directory. A changed design
requires a new protocol version.
