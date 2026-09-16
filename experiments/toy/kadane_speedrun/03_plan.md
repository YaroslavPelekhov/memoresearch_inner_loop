# Pre-Registration: Kadane Speedrun (Pipeline Dry Run)

**Date**: 2026-03-21
**Protocol version**: 1.0
**Pre-registration commit**: `<hash>` -- filled after commit
**Design doc**: `experiments/toy/kadane_speedrun/01_design.md`
**Review doc**: `experiments/toy/kadane_speedrun/02_review.md` (verdict: APPROVED)

---

## Hypothesis

**H0**: Gemini Flash does not reach fitness=1.0 within 3 generations.
**H1**: Gemini Flash reaches fitness=1.0 within 3 generations.
**Primary metric**: fitness at gen 3
**Significance threshold**: N/A (descriptive)

---

## Run Design Table

| Run | Label | `redis.db` | `pipeline` | `problem.name` | `llm_base_url` |
|-----|-------|------------|-----------|----------------|----------------|
| 1 | R1 | 15 | standard | toy_kadane | https://openrouter.ai/api/v1 |
| 2 | R2 | 14 | standard | toy_kadane | https://openrouter.ai/api/v1 |

---

## Controlled Variables

| Field | Value |
|-------|-------|
| max_generations | 3 |
| max_mutations_per_generation | 2 |
| max_elites_per_generation | 2 |
| num_parents | 1 |
| mutation_mode | rewrite |
| model_name | google/gemini-2.0-flash-001 |

---

## Success Criteria

fitness=1.0 reached by at least one run within 3 generations.

---

## Monitoring Plan

`max_generations`: 3

- Gen 1: smoke check
- Gen 3: final evaluation

---

## Amendments

_(None -- pipeline dry run.)_
