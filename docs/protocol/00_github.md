# Phase 0: GitHub Tracking
<!-- Protocol version: 1.0 -->

**This phase runs in parallel with the five experiment phases — not instead of them.**
GitHub is the public audit trail. The `.md` files in `experiments/<task>/<name>/` are the
authoritative scientific record. Both must stay in sync.

---

## Overview

| GitHub artifact | Purpose | Created at |
|-----------------|---------|-----------|
| **Issue** | Tracks the experiment idea from inception through completion | When idea is formed |
| **Branch** `exp/<name>` | Isolates experiment code and docs | Phase 3 (pre-registration) |
| **PR** | Living experiment record: hypothesis, design, checkpoints, final result | Phase 3 (pre-registration) |

---

## Issues — Research Backlog

Use GitHub Issues to track experiment ideas before they are designed.

### Creating an issue

When a new experiment idea arises (from Elena's `Next Steps`, from a result, from discussion):

```bash
gh issue create \
  --title "Experiment idea: <brief description>" \
  --label "experiment-idea" \
  --body "$(cat <<'EOF'
## Motivation
<Why is this worth testing? What prior result or gap does it address?>

## Hypothesis (sketch)
<H₀ / H₁ in plain language — not yet rigorous>

## Estimate
<Compute cost, wall time, number of runs>

## References
<Prior experiment PRs, docs/plans/ files, papers>
EOF
)"
```

### Issue lifecycle

| State | Label | Meaning |
|-------|-------|---------|
| Open | `experiment-idea` | Idea recorded, not yet designed |
| Open | `experiment-running` | Phase 3 complete; PR open; runs active |
| Closed | `experiment-complete` | Phase 5 complete; PR merged |
| Closed | `experiment-invalid` | Experiment invalidated; documented in PR |

Update the label when the experiment transitions between phases:
```bash
gh issue edit <N> --remove-label "experiment-idea" --add-label "experiment-running"
```

Close the issue when the PR is merged:
```bash
gh issue close <N> --comment "Completed. See PR #<M> for results."
```

---

## Releases — Artifact Storage

Large experiment artifacts (evolved programs, fitness CSVs, environment snapshots) are stored
as **GitHub Release assets**, not in the git repository.

| Artifact | Storage | Uploaded by |
|----------|---------|-------------|
| `<label>_archive.tar.gz` (evolution CSV + all programs + top50 JSON) | GitHub Release `exp/<name>` | `tools/experiment/archive_run.sh --upload` |
| `environment.txt` (pip freeze, OS, GPU) | GitHub Release `exp/<name>` | `tools/experiment/archive_run.sh --upload` |
| Watchdog plots (PNG/PDF) | *(attach manually to PR comment via GitHub web UI)* | Researcher |

Release naming: `exp/<experiment-name>` (one release per experiment, all runs as assets).

```bash
# View releases for this experiment
gh release view exp/<name>

# List all experiment releases
gh release list | grep "^exp/"
```

---

## Branch and PR — Experiment Record

### Phase 3: Create branch and PR

At pre-registration time, after `03_plan.md` is written and ready to commit:

```bash
# 1. Create and push the experiment branch
git checkout -b exp/<name>
git push -u origin exp/<name>

# 2. Commit 03_plan.md (this is the pre-registration commit)
git add experiments/<task>/<name>/03_plan.md
git commit -m "preregister: <experiment name> — commit 03_plan.md before code"
PREREG_HASH=$(git rev-parse HEAD)

# 3. Create the PR using the template
gh pr create \
  --title "exp: <experiment name>" \
  --base main \
  --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)" \
  --label "experiment-running"

# 4. Link the PR to the tracking issue (if one exists)
gh issue comment <issue-N> --body "PR created: #$(gh pr view --json number -q .number)"
```

Record the PR number in `03_plan.md`:
```
**GitHub PR**: #<number>
```

### PR description template

Each experiment has `experiments/<task>/<name>/PR_DESCRIPTION.md` — the living PR description.
It is initialized from `experiments/_template/PR_DESCRIPTION.md` and updated throughout
the experiment. To push updates to the GitHub PR:

```bash
gh pr edit --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)"
```

**Status badge convention**:
- 🔵 `Pre-registered` — 03_plan.md committed, code not yet launched
- 🟡 `Running (gen X/N)` — runs active; update badge at each checkpoint
- 🟢 `Complete` — all runs finished, Phase 5 done
- 🔴 `Invalid` — one or more runs invalidated; see PR for details

### Phase 4: Checkpoint updates

At each monitoring checkpoint (gen ~20%, ~50%), update the PR description with the
latest checkpoint results and push:

```bash
# Edit experiments/<task>/<name>/PR_DESCRIPTION.md — add checkpoint row
gh pr edit --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)"
```

The watchdog already posts automated status comments. The PR description update is
for the high-level checkpoint table (val EM per run, date).

### Phase 5: Final result and merge

After Phase 5 (`05_results.md` committed):

1. Update `PR_DESCRIPTION.md` with final results, set status to 🟢 Complete:
   ```bash
   # Edit PR_DESCRIPTION.md — fill in Final Result section, update status badge
   gh pr edit --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)"
   ```

2. Post a summary comment linking to `05_results.md`:
   ```bash
   gh pr comment --body "Phase 5 complete. Results: [05_results.md](experiments/<task>/<name>/05_results.md)
   Verdict: <POSITIVE/SUGGESTIVE/NULL/NEGATIVE>. delta = <+X.Xpp>."
   ```

3. **Review before merging** — the branch may contain experiment records, general-purpose
   code changes, and run artifacts. Decide what belongs in `main`:
   ```bash
   git log main..HEAD --oneline   # what commits are on the branch
   git diff main...HEAD --stat    # what files changed
   ```
   - Experiment records (`experiments/<task>/<name>/`) → merge to `main`
   - General-purpose code/infra changes → consider a separate focused PR if substantial
   - Log files and generated artifacts → exclude (`.gitignore` if needed)

4. Merge with `--merge` (not `--squash`) to preserve the pre-registration audit trail:
   ```bash
   gh pr merge --merge --delete-branch
   ```

5. Close tracking issue:
   ```bash
   gh issue close <N> --comment "Completed. PR #<M> merged."
   ```

---

## PR Naming Convention

| Experiment | Branch | PR title |
|-----------|--------|----------|
| NLP prompts | `exp/hotpotqa-nlp-prompts` | `exp: HotpotQA NLP mutation prompts` |
| P3 crossover | `exp/hotpotqa-p3-crossover` | `exp: HotpotQA P3 crossover (num_parents=2)` |

---

## Quick Reference

```bash
# Create issue for new experiment idea
gh issue create --title "Experiment idea: ..." --label "experiment-idea"

# Check open experiment issues
gh issue list --label "experiment-running"

# View current PR status
gh pr view

# Update PR description from file
gh pr edit --body "$(cat experiments/<task>/<name>/PR_DESCRIPTION.md)"

# Post checkpoint comment
gh pr comment --body "Gen 10 checkpoint: K=64.0% L=65.2% M=66.1% N=63.8%"

# Review branch diff before merging (see 05_results.md GitHub Closeout)
git diff main...HEAD --stat
# Merge after Phase 5 (--merge, not --squash, to preserve audit trail)
gh pr merge --merge --delete-branch
```
