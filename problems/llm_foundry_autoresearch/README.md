# Fixed-hybrid DCLM research task

The outer campaign selects a research idea, chooses a suitable parent from the
best valid implementations, creates one seed, and then gives that fixed idea to
the inner evolution loop.

Only `autoresearch/model/gdn.py` is mutable. Transformer layers, alternating
layer placement, training configuration, and evaluation remain fixed. The
machine-readable contract is `research_task.yaml`; the complete mutation
boundary is `task_description.txt`.

Prepare one fixed train/validation view from disjoint compressed MDS shards:

```bash
python tools/prepare-dclm-holdout.py \
  --source-root /path/to/dclm-mds \
  --output-root /path/to/dclm-mds-heldout
```

Set both `DCLM_MDS_PATH` and `DCLM_SCREEN_MDS_PATH` to that output root, and set
`DCLM_TRAIN_SPLIT=train`. Its `train` split excludes its `validation` split. The inner evolution ranks four
attempts at 1,024 batches; the outer campaign confirms only the winner at 4,096
batches. CORE is intentionally left for later final confirmation. Artifacts are
grouped by idea ID, idea version, implementation commit, and evaluation horizon
under `runs/<campaign>/ideas/`.
