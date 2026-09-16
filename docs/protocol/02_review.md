# Phase 2: Adversarial Review
<!-- Protocol version: 1.0 -->

**Actor**: `reviewer-2-adversary` agent
**Input**: Completed `experiments/<task>/<name>/01_design.md`
**Output**: Completed `experiments/<task>/<name>/02_review.md`
**Gate**: Verdict = **APPROVED** before proceeding to Phase 3

If verdict is NEEDS REVISION: revise `01_design.md`, re-invoke reviewer, repeat until approved.

---

## Invocation

Provide the agent with the completed `01_design.md`. Ask it to act as a hostile but fair
peer reviewer: find every methodological weakness, confound, and statistical issue, then
give a verdict.

---

## Template

### Summary of Design

> _(1–2 sentences: what is being tested and how.)_

### Methodological Concerns

For each concern, rate severity: **Critical** / **Major** / **Minor**

| # | Concern | Severity | Recommendation |
|---|---------|----------|---------------|
| | | | |

### Hypothesis and Falsifiability

- [ ] H₀ is clearly stated
- [ ] H₁ is falsifiable and directional
- [ ] Primary metric is pre-specified and sufficient to test H₁
- [ ] Success criteria are numeric and unambiguous

**Notes**:

### Confound Analysis

- [ ] All controlled variables are genuinely controlled
- [ ] IV is isolated (no other differences between conditions)
- [ ] Known confounds are mitigated or acknowledged
- [ ] Val/test split is not contaminated

**Unaddressed confounds** (if any):

### Statistical Validity

- [ ] Sample size is justified
- [ ] Statistical test is appropriate for the data
- [ ] Significance threshold is pre-specified
- [ ] Multiple comparison correction applied if testing multiple hypotheses

**Notes**:

### Evaluation Protocol

- [ ] Metric is computed identically across all conditions
- [ ] Val set and test set are fixed and identical for all runs
- [ ] No metric is cherry-picked post-hoc
- [ ] Thinking mode is consistent across all evaluations

**Notes**:

### Required Changes Before Approval

> _(List specific changes to `01_design.md` required before APPROVED verdict.
> Leave blank if none.)_

1.
2.

---

## Verdict

**[ ] APPROVED** — proceed to Phase 3

**[ ] NEEDS REVISION** — address required changes, re-submit for review

**[ ] REJECTED** — fundamental flaw; redesign required

**Reviewer notes**:
