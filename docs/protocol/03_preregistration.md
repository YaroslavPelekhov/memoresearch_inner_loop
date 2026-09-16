# Phase 3: Pre-Registration
<!-- Protocol version: 1.0 -->

**Actor**: Researcher (with Claude Code)
**Gate**: Committed to git **before any code changes or implementation begins**
**Output**: `experiments/<task>/<name>/03_plan.md` — locked document

---

## Rules

1. Commit `03_plan.md` before touching any code. Record the commit hash in the document.
2. After commitment, the plan is **locked**. Any change requires a numbered amendment (see below).
3. The pre-registration commit hash is the experiment's identity — include it in all reports.
4. Create the experiment branch and GitHub PR at this phase (see steps below).
5. Record dataset file checksums at registration time:
   ```bash
   sha256sum <path/to/train_file> <path/to/test_file>
   ```
   Record in the Dataset Checksums section of `03_plan.md`. These are the cryptographic
   anchor for cross-experiment reference distributions. If a prior experiment's results
   are used as a baseline or noise floor calibration, verify its checksums match.
6. Pin the evaluation script hash at registration time:
   ```bash
   sha256sum experiments/<task>/<name>/run_test_eval.sh
   ```
   Record the hash in `03_plan.md`. If the script changes before the final evaluation, record it as an amendment. If there is no test split, write `N/A`.

---

## Template for `experiments/<task>/<name>/03_plan.md`

```markdown
# Pre-Registration: <Experiment Name>

**Date**: YYYY-MM-DD
**Pre-registration commit**: `<hash>` (committed before implementation)
**Design doc**: `experiments/<task>/<name>/01_design.md`
**Review doc**: `experiments/<task>/<name>/02_review.md` (verdict: APPROVED)

---

## Hypothesis

**H₀**: <copy from 01_design.md>
**H₁**: <copy from 01_design.md>
**Primary metric**: <metric> at gen <N> on <val/test> set
**Significance threshold**: α = <value>

---

## Run Design Table

| Run | Label | `redis.db` | `pipeline` | `prompts` | `problem.name` | `llm_base_url` | Seed | Val set |
|-----|-------|------------|-----------|-----------|----------------|----------------|------|---------|

---

## Controlled Variables

| Field | Value |
|-------|-------|

---

## Success Criteria

> (Exact numeric thresholds. Pre-specified. Will not be changed after registration.)

---

## Monitoring Plan

`max_generations`: ___

- Gen ___ (~10%): smoke check — all PIDs alive, Redis keys growing
- Gen ___ (~20%): first checkpoint — extract best-by-val, run test eval (if applicable), record metrics
- Gen ___ (~50%): midpoint checkpoint
- Gen ___ (100%): final evaluation + analysis

Early termination rule: <condition, e.g., val EM < X at gen 10%>

---

## Actual Launch Record

Filled in at launch time (not pre-registered):

| Run | PID | Launch time (UTC) | Notes |
|-----|-----|-------------------|-------|

Watchdog PID: ___
Launch commit: `<hash>`

---

## Checkpoint Log

| Gen | Date (UTC) | K val EM | L val EM | … | Notes |
|-----|-----------|----------|----------|---|-------|

---

## Amendments

Any post-registration change requires a numbered entry here.

### Amendment 1 — <title>

**Date**: YYYY-MM-DD
**Commit**: `<hash>`
**Change**: <what changed>
**Reason**: <why>
**Impact**: No confound / Confound introduced / Run invalidated
**Details**: <elaboration>
```

---

## GitHub Steps (do these at pre-registration time)

```bash
# 1. Create experiment branch and switch to it
git checkout -b exp/<name>

# 2. Initialize PR description from template
cp experiments/_template/PR_DESCRIPTION.md experiments/<task>/<name>/PR_DESCRIPTION.md
# Edit: fill in hypothesis, design table, status = 🔵 Pre-registered

# 3. Pin evaluation script hash (if applicable)
sha256sum experiments/<task>/<name>/run_test_eval.sh  # paste into 03_plan.md

# 4. Commit 03_plan.md and PR_DESCRIPTION.md (the pre-registration commit)
git add experiments/<task>/<name>/03_plan.md experiments/<task>/<name>/PR_DESCRIPTION.md
git commit -m "preregister: <experiment name>"
# Record this commit hash in 03_plan.md and PR_DESCRIPTION.md, then amend or note it

# 5. Push and create the PR
git push -u origin exp/<name>
gh pr create \
  --title "exp: <experiment name>" \
  --base main \
  --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)" \
  --label "experiment-running"

# 6. Record PR number in 03_plan.md and link to tracking issue (if one exists)
gh issue comment <issue-N> --body "PR created: #$(gh pr view --json number -q .number)"
```

See `docs/protocol/00_github.md` for full GitHub workflow details.

---

## Notes on Scope

- `03_plan.md` is the **authoritative locked record**. It must be self-contained enough
  that someone can reproduce the experiment from it alone.
- Scientific rationale lives in `01_design.md`. Do not duplicate it here — reference it.
- Operational details (PIDs, launch times, checkpoint metrics) are appended to `03_plan.md`
  as the experiment progresses.
