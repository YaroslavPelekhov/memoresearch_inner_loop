# Pre-Registration: <Experiment Name>

**Date**: YYYY-MM-DD
**Protocol version**: 1.0
**Pre-registration commit**: `<hash>` ← commit this file BEFORE any code changes
**GitHub PR**: #<number> (branch: `exp/<name>`)
**Tracking issue**: #<number> — or N/A
**Design doc**: `experiments/<task>/<name>/01_design.md`
**Review doc**: `experiments/<task>/<name>/02_review.md` (verdict: APPROVED)
**Evaluation script**: `experiments/<task>/<name>/run_test_eval.sh` (sha256: `<hash>`) — or N/A if no test split

---

## Hypothesis

**H₀**:
**H₁**:
**Primary metric**: <metric> at gen <N> on <val/test> set
**Significance threshold**: α =

---

## Run Design Table

| Run | Label | `redis.db` | `pipeline` | `prompts` | `problem.name` | `llm_base_url` | Seed | Val set |
|-----|-------|------------|-----------|-----------|----------------|----------------|------|---------|

---

## Controlled Variables

| Field | Value |
|-------|-------|

---

## Reproducibility Notes

**This experiment uses stochastic LLM-based evolution. Exact trajectory reproduction is not possible.**

Known sources of non-determinism (document and accept):
- `random.sample` in `FormatterStage` (failure sampling per generation)
- LLM sampling temperature and nucleus sampling in chain and mutation LLMs
- Non-deterministic GPU floating point across hardware

**Global seed** (if Hydra config supports it): _(fill in or write N/A)_

A fresh run with identical config will produce a different fitness trajectory but should
land in a statistically similar fitness range. Cross-experiment comparisons use effect-size
thresholds (from `01_design.md`) rather than exact trajectory matching.

---

## Dataset Checksums

Cryptographic anchor for the data used in this experiment.
Compute at pre-registration time and verify before final evaluation.

```bash
sha256sum <path/to/train_file> <path/to/test_file>
```

| File | sha256 |
|------|--------|
| _(train/val data file)_ | |
| _(test data file)_ | |

---

## Success Criteria

---

## Monitoring Plan

`max_generations`: ___

- Gen ___ (~10%): smoke check — all PIDs alive, Redis keys growing
- Gen ___ (~20%): first checkpoint — extract best-by-val, run test eval (if applicable), record metrics
- Gen ___ (~50%): midpoint checkpoint
- Gen ___ (100%): final evaluation + analysis

Early termination rule:

---

## Actual Launch Record

| Run | PID | Launch time (UTC) | Notes |
|-----|-----|-------------------|-------|

Watchdog PID:
Launch commit: `<hash>`

---

## Checkpoint Log

| Gen | Date (UTC) | Notes |
|-----|-----------|-------|

---

## Amendments

_(Add numbered entries here for any post-registration changes.)_
