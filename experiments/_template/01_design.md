# Experimental Design: <Experiment Name>

**Date**: YYYY-MM-DD
**Researcher**:
**Status**: Draft / Approved

> Fill this in by invoking the `ml-research-methodologist` agent.
> See `docs/protocol/01_design.md` for instructions and field descriptions.

---

## 1. Research Question

## 2. Hypotheses

**H₀**:
**H₁**:

## 3. Independent Variable(s)

| Variable | Control value | Treatment value(s) |
|----------|---------------|--------------------|

## 4. Dependent Variable(s)

| Metric | How measured | Primary? |
|--------|-------------|----------|

**Primary metric**:

## 5. Controlled Variables

| Field | Value | Rationale |
|-------|-------|-----------|

## 6. Run Design Table

> **Framework convention (I-06)** — every run's Redis key prefix MUST equal its
> `problem.name`. The watchdog `adversarial` plugin matches opponent populations
> by pop_a/pop_b suffix, and `dataset_snapshot.json` is keyed by the problem
> path — inventing a URL-like namespace (e.g. `<task>/<exp-name>/pop_a`) breaks
> both. For adversarial runs, `opponent_redis_prefix=<other_population>` must
> similarly equal the opponent's `problem.name`.
>
> Concretely, for an adversarial pair the `runs[]` block should read:
>   `prefix: <task>_<name>/pop_a` paired with `problem_name: <task>_<name>/pop_a`
>   and `opponent_redis_prefix=<task>_<name>/pop_b`.

| Run | Label | `redis.db` | `pipeline` | `prompts` | `problem.name` | Redis `prefix` (== `problem.name`) | `llm_base_url` | Seed | Val set |
|-----|-------|------------|-----------|-----------|----------------|------------------------------------|----------------|------|---------|

## 7. Sample Size Justification

## 8. Statistical Test

**Test**:
**Significance threshold**: α =
**How computed**:

## 9. Known Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|-----------|

## 10. Stop Criteria

**Early termination**:
**Run invalidation**:

## 11. Compute Budget

| Resource | Estimated usage |
|----------|----------------|
| GPU hours | |
| Wall time | |
| Redis DBs used | |

## 12. Open Questions / Risks
