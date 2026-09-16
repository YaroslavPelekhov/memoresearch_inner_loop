# One fixed-idea evolution

This experiment isolates the inner loop. It runs one approved idea through at
most 500 GigaEvo mutation attempts. Every implementation must pass a cached,
content-verified 128-batch executable smoke test. A 1024-batch short-loss record
earns a 4096-batch confirmation; only a long-loss improvement replaces the
confirmed parent.

Prepare an independent candidate repository and editable idea template:

```bash
python tools/setup-one-idea-experiment.py --output prepared/one-idea-test
```

Edit `prepared/one-idea-test/idea.yaml`, then run the custom idea:

```bash
bash prepared/one-idea-test/run-custom.sh
```

To have Codex generate proposals and select exactly one idea to evolve:

```bash
bash prepared/one-idea-test/run-generated.sh
```

The two focused charts are under the TensorBoard custom-scalars `Evolution`
layout: `Validation loss` and `CORE`. CORE remains empty until a genuine CORE
evaluation is enabled.
