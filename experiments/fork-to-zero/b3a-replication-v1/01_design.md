# Experimental Design: b3a210 replication and onset map

**Date**: 2026-09-22
**Status**: Frozen before launch

## Question

Does the retention-gradient-floor mutation `b3a210d` reproducibly collapse,
and does a 256-batch Fork-to-zero branch reveal that collapse earlier than an
equal-compute continuation of the normal learning-rate schedule?

This is a fixed-code challenge experiment. It does not use Codex, generate
mutations, tune thresholds, or prune the main trajectory.

## Variants and pairing

- Treatment: model file from `b3a210ded839250e22eb148718fc9d401ec90052`.
- Negative control: model file from its immediate parent
  `0369584bf978a6850f96d29d3dc07fe00f32aaa9`.
- Six paired seeds: `31001` through `31006`.
- Both variants in a pair use the same seed and data order.
- Odd pairs put treatment on GPU 0 and parent on GPU 1; even pairs swap them.
- Pairs run sequentially. The two variants inside a pair run concurrently.

Each variant runs the current frozen benchmark harness, with only
`autoresearch/model/gdn.py` overlaid from the specified historical commit.
The launcher records both source-file hashes and the harness commit.

## Trajectory

The untouched main branch runs to 4,096 batches. At checkpoints 512, 768, and
1,024, two read-only 256-batch branches are launched from the same checkpoint:

1. matched continuation under the original 4,096-batch schedule;
2. Fork-to-zero, a cosine learning-rate descent to zero.

Per variant this costs 4,096 main batches plus 1,536 shadow batches, or 5,632
batch-equivalents. The 37.5% overhead is deliberate: the 768 checkpoint maps
the onset missed by the earlier two-checkpoint experiment.

## Frozen labels and alarms

A run collapses when the later main branch is intrinsically non-finite or its
held-out loss at batch 4,096 is at least 10% above the checkpoint loss. Hardware,
storage, timeout, and orchestration errors are censored.

- Fork alarm: intrinsic failure on the Fork branch, or
  `fork_fitness - continuation_fitness <= -0.002`.
- Continuation alarm: intrinsic failure on the continuation branch, or
  `continuation_fitness - checkpoint_fitness <= -0.002`.

The earliest alarm checkpoint is used. No threshold is changed after launch.

## Primary decision

The mechanism is replicated only if treatment collapses in at least four of
six pairs and the parent collapses in at most one of six. The distinctive
early-warning claim is supported only if, in at least four of six treatment
runs, Fork-to-zero alarms before final collapse and strictly earlier than the
matched continuation alarm. A tie is not a Fork advantage.

If the collapse replicates but alarms are simultaneous, the mutation mechanism
is supported and the unique early-warning value of Fork-to-zero is not.

## Execution and isolation

The one-shot coordinator creates retained worktrees and all outputs under
`/home/bulatov/yaroslav-multifidelity`. It refuses to start unless GPUs 0 and 1
are both idle immediately before launch. No path outside that private root is
used for environments, code, caches, data lanes, checkpoints, or logs.
