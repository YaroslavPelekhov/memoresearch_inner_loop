# Phase 1: Experimental Design
<!-- Protocol version: 1.0 -->

**Actor**: `ml-research-methodologist` agent
**Output**: Completed `experiments/<task>/<name>/01_design.md`
**Gate**: Researcher approves design before proceeding to Phase 2

---

## Invocation

Provide the agent with:
- The research question you want to answer
- Relevant prior results (benchmark numbers, previous experiment outcomes)
- Available compute (number of runs, max generations, GPU hours)
- Any hard constraints (fixed seed, fixed dataset split, required controls)

The agent fills in all sections below.

---

## Template

### 1. Research Question

> _(One sentence. Must be answerable from the experiment's metrics alone.)_

### 2. Hypotheses

**Null hypothesis (H₀)**:
> _(The default assumption — no effect, no difference.)_

**Alternative hypothesis (H₁)**:
> _(The claim being tested. Must be falsifiable and directional if possible.)_

### 3. Independent Variable(s)

| Variable | Control value | Treatment value(s) |
|----------|---------------|--------------------|
| | | |

### 4. Dependent Variable(s)

| Metric | How measured | Primary? |
|--------|-------------|----------|
| | | |

**Primary metric**: _(single metric used for hypothesis test — must be pre-specified)_

### 5. Controlled Variables

Every config field held constant across all runs, with its value:

| Field | Value | Rationale |
|-------|-------|-----------|
| | | |

### 6. Run Design Table

One row per run:

| Run | Label | `redis.db` | `pipeline` | `prompts` | `problem.name` | `llm_base_url` | Seed | Val set |
|-----|-------|------------|-----------|-----------|----------------|----------------|------|---------|
| | | | | | | | | |

### 7. Sample Size Justification

> _(Why this many runs per condition? What effect size is detectable? Reference prior variance
> if available.)_

### 8. Statistical Test

**Test**: _(e.g., one-sided t-test on test EM at gen 50)_
**Significance threshold**: α = ___
**How computed**: _(exact procedure — bootstrapping, paired, etc.)_

### 9. Known Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|-----------|
| | | |

### 10. Stop Criteria

**Early termination** (if any):
> _(Conditions under which a run is stopped before gen 50 — e.g., val EM < X at gen 10,
> crash with no recovery path.)_

**Run invalidation**:
> _(Conditions under which a completed run is excluded from analysis.)_

### 11. Compute Budget

| Resource | Estimated usage |
|----------|----------------|
| GPU hours | |
| Wall time | |
| Redis DBs used | |

### 12. Open Questions / Risks

> _(Anything unresolved that could affect validity. Must be addressed before Phase 2 approval.)_
