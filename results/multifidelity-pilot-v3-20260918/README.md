# Multifidelity pilot v3 results

Final snapshot of the two four-round campaigns completed on 2026-09-18.

## Campaigns

- `multifidelity-pilot-v3-gpu0`: completed, 4/4 rounds, exit code 0.
- `multifidelity-pilot-v3-gpu1`: completed, 4/4 rounds, exit code 0.
- Eight hypotheses were evaluated. All remain scientifically `inconclusive`
  pending paired multi-seed validation and a calibrated noise estimate.

## Main observations

- Lowest observed final held-out loss: `4.0520` in the GPU 0 lineage.
- Best attributable child improvement: `4.0531 -> 4.0521` for selective
  norm control, but the delta is too small to distinguish from run noise.
- GPU 1's best final-round change was `4.0609 -> 4.0606`, with a worse loss
  AUC, so it is not a clear improvement.
- The 1024-step screen did not reliably predict the 4096-step ordering.

## Preserved artifacts

The authoritative server copy is under:

`/home/bulatov/yaroslav-multifidelity/archives/multifidelity-pilot-v3-20260918`

A verified portable copy is also stored locally outside the Git repository at:

`/Users/yaroslavpelehov/Downloads/multifidelity-pilot-v3-20260918`

It contains:

- `portable-results-no-checkpoints.tar.zst`: all campaign artifacts except
  `*.pt` checkpoint payloads;
- `repository.bundle`: a verified Git bundle containing the repository and all
  candidate commits;
- `candidate-git-refs.txt`: 30 protected candidate commit references;
- `checkpoint-files.tsv`: paths, sizes, and timestamps for all 195 checkpoints;
- `portable-files.tsv`: paths, sizes, and timestamps for all 44,331 portable
  files;
- `SHA256SUMS`: checksums for the portable archive, Git bundle, and manifests.

The full checkpoint trees remain in their original server locations:

- `/home/bulatov/yaroslav-multifidelity/runs/multifidelity-pilot-v3-gpu0`
  (approximately 137 GiB);
- `/home/bulatov/yaroslav-multifidelity/runs/multifidelity-pilot-v3-gpu1`
  (approximately 144 GiB).

All 195 checkpoint files also have a hard-link snapshot under
`full-checkpoints-hardlinks/` in the authoritative server archive. This keeps
the checkpoint data recoverable if an original run path is removed, without
duplicating the underlying 281 GiB of storage.

## Restore

Verify the portable package with `sha256sum -c SHA256SUMS`. Extract it from a
directory that should receive the `runs/` tree:

```bash
tar --zstd -xf portable-results-no-checkpoints.tar.zst
```

Restore the complete repository history into a new directory:

```bash
git clone repository.bundle restored-repository
```

The portable archive intentionally omits only checkpoint files. Use
`checkpoint-files.tsv` to locate the authoritative copies on the server.
