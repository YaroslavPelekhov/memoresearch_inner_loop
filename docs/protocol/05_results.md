# Phase 5: Results and Post-Mortem
<!-- Protocol version: 1.0 -->

**Actor**: `ml-research-methodologist` agent + Researcher
**Input**: Final metrics, `01_design.md`, `03_plan.md` (with checkpoint log)
**Output**: `experiments/<task>/<name>/05_results.md`
**Gate**: Written, committed, `experiments/INDEX.md` updated, Claude memory updated — before starting the next experiment

---

## Prerequisites (verify before starting Phase 5)

- [ ] **Step 8 (Archive Run Data) of Phase 4 is complete.** GitHub Release `exp/<name>`
  exists with archives for every run. Without this, the data needed for analysis may
  already be gone. Check: `gh release view exp/<name>`
- [ ] All test evaluations have been run (`run_test_eval.sh`)
- [ ] `03_plan.md` checkpoint log is up to date through the final generation
- [ ] `environment_freeze.txt` is committed to git

## Invocation

Provide the agent with:
- The completed `03_plan.md` (design table, success criteria, amendment log)
- The completed `01_design.md` (hypotheses, statistical test, primary metric)
- Final metrics for all runs (primary metric at final generation, val EM trajectory)
- GitHub Release URL for the archives (for reference in paper supplementary)
- Any anomalies observed during the run

The agent fills in all sections below.

---

## Template

### 1. Final Metrics

| Run | Condition | Best val fitness | Test EM (gen 50) | Mutations | Notes |
|-----|-----------|--------------|-----------------|-----------|-------|
| | | | | | |

**Baseline / GEPA reference**: _(test EM for comparison)_

### 2. Hypothesis Test

**H₀**: _(copy from 01_design.md)_
**H₁**: _(copy from 01_design.md)_
**Primary metric**: _(value)_
**Statistical test**: _(test name, statistic, p-value)_
**Result**: H₀ **rejected** / **not rejected** at α = ___

### 3. Effect Size

> _(Magnitude of improvement over control/baseline. Practical significance beyond p-value.)_

### 4. Secondary Observations

> _(Patterns in val EM trajectory, program structure changes, unexpected behavior.)_

### 5. Deviations from Pre-Registration

**Mandatory.** For every item pre-registered in `03_plan.md`, state whether execution
followed the plan exactly. This section must be completed before any results are interpreted.

| Pre-registered item | Followed? | Notes |
|---------------------|-----------|-------|
| Primary metric and threshold | Yes / Deviated: ___ | |
| Statistical test / decision rule | Yes / Deviated: ___ | |
| Evaluation script | Yes (sha256: ___) / Changed: ___ | |
| Run design table (pipeline, prompts, seed) | Yes / Deviated: ___ | |
| Monitoring plan | Yes / Deviated: ___ | |
| Early termination rule | Not triggered / Triggered: ___ | |

**If there are no deviations**: state explicitly — *"Execution followed 03_plan.md exactly."*

**Each deviation** must be one of:
- **(A) Pre-registered amendment** — numbered entry exists in `03_plan.md` Amendments section. Reference it.
- **(B) Unrecorded deviation** — this is a protocol violation. Document what happened and why it was not amended. Assess impact on validity.

### 6. Amendment Impact Assessment

For each amendment in `03_plan.md`, assess whether it affected result validity:

| Amendment | Impact on validity | Assessment |
|-----------|-------------------|-----------|
| | | |

### 7. Run Validity

| Run | Valid for analysis? | Reason if excluded |
|-----|--------------------|--------------------|
| | | |

### 8. Lessons Learned

**What worked**:
-

**What didn't work**:
-

**Bugs / infrastructure issues**:
-

### 9. Next Steps

> _(Follow-up experiments to run, hypotheses to test, improvements to the framework.
> Reference `docs/plans/` for any already-written follow-up proposals.)_

### 10. Paper / Report Notes

> _(Key results and framing for inclusion in a paper or report. What claim does this
> experiment support or refute?)_

---

## GitHub Closeout (after all sections above are complete)

### Step A — Commit results and update PR

```bash
# 1. Commit 05_results.md and updated PR_DESCRIPTION.md
git add experiments/<task>/<name>/05_results.md experiments/<task>/<name>/PR_DESCRIPTION.md
git commit -m "results: <experiment name> — verdict: <POSITIVE/NULL/etc>"
# Edit PR_DESCRIPTION.md first: fill in Final Result section, set status to 🟢 Complete

# 2. Update PR description on GitHub
gh pr edit --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)"

# 3. Post results summary comment
gh pr comment --body "Phase 5 complete.
Verdict: <POSITIVE/SUGGESTIVE/NULL/NEGATIVE>
delta: <+X.Xpp>
Full analysis: experiments/<task>/<name>/05_results.md"
```

### Step B — Review branch before merging

**Do not merge blindly.** This branch may contain a mix of:

| Category | Examples | Goes to `main`? |
|----------|----------|-----------------|
| Experiment records | `01_design.md`, `03_plan.md`, `05_results.md`, `launch.sh`, `run_watchdog.py`, `PR_DESCRIPTION.md` | **Yes** — permanent scientific record |
| Protocol / tooling improvements | `docs/protocol/`, `tools/`, bug fixes, new features | **Yes** — but consider a separate focused PR if the changes are large and generally useful |
| Run logs and generated artifacts | `*.log`, PID files not embedded in `.md` | **No** — add to `.gitignore` or exclude |

**Review the diff before merging:**
```bash
git log main..HEAD --oneline        # commits not yet on main
git diff main...HEAD --stat         # files changed vs main
```

Decide:
- Are there **general-purpose code or infra changes** (new tools, framework fixes) that should be
  a separate, focused PR to `main` first? If so, extract those commits into their own PR.
- Are there **log files or large artifacts** that should not go to `main`?
  Add them to `.gitignore` and commit before merging.

### Step C — Update experiment index

```bash
# Add a row (or update status) in experiments/INDEX.md
# Fill in: status → ✅ Complete, key finding (one sentence), PR number, archives URL
git add experiments/INDEX.md
git commit -m "index: mark <experiment name> complete"
```

### Step D — Update Claude memory

`experiments/INDEX.md` is the canonical results ledger. Claude's memory files store only active
run state and cross-cutting findings. Keep them in sync at every closeout:

```bash
# File: .claude/projects/*/memory/MEMORY.md
# - Remove this experiment from the "Active Experiments" table
# - If a new experiment is launching next, add its entry

# File: .claude/projects/*/memory/findings.md
# - If a new cross-cutting finding emerged (a pattern confirmed across ≥2 runs,
#   or a previously SUGGESTIVE result now resolved), add to §Validated Findings
# - Update §Open Questions if a question was answered or a new one opened
# - Update §Permanently CUT if an approach was definitively ruled out

# File: .claude/projects/*/memory/infrastructure.md
# - If server IPs, context windows, or proxy config changed during the experiment

# File: .claude/projects/*/memory/patterns.md
# - If a new bug, config pitfall, or workflow pattern was discovered
```

**Rule**: never copy per-experiment result rows into memory files. `INDEX.md` owns that data.
Memory stores only what cannot be retrieved by reading a project file.

### Step F — Merge and close

```bash
# Run full lifecycle gate before merging
bash tools/experiment/check_experiment_complete.sh <experiment-name>

# Regular merge (not squash — preserve experiment commit history)
gh pr merge --merge --delete-branch

# Close tracking issue
gh issue close <N> --comment "Experiment complete. PR #<M> merged."

# Kill health-check loop (crontab is not available on this machine — see Step 6 of Phase 4)
kill $HEALTH_PID  # PID noted when health-check loop was started in Phase 4 Step 6
```

> **Why not squash?** Squashing collapses the experiment's pre-registration commit, amendment
> commits, and checkpoint commits into one — destroying the audit trail that proves the
> hypothesis was registered before data was collected. Use `--merge` to preserve history.
