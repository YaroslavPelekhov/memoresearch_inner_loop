# Experimental Design: Kadane Speedrun (Pipeline Dry Run)

**Date**: 2026-03-21
**Researcher**: Pipeline test
**Status**: Approved

> This is a toy experiment to validate the full experiment lifecycle pipeline.
> Not real research -- the problem is trivially solvable.

---

## 1. Research Question

Can Gemini Flash one-shot the max-subarray-sum problem from a buggy seed, reaching fitness=1.0 within 3 generations?

## 2. Hypotheses

**H0**: Gemini Flash does not reach fitness=1.0 within 3 generations.
**H1**: Gemini Flash reaches fitness=1.0 within 3 generations (expected outcome).

## 3. Independent Variable(s)

| Variable | Control value | Treatment value(s) |
|----------|---------------|--------------------|
| N/A -- single arm | -- | Gemini Flash via OpenRouter |

## 4. Dependent Variable(s)

| Metric | How measured | Primary? |
|--------|-------------|----------|
| fitness | Fraction of 20 test cases passed | Yes |

**Primary metric**: fitness (0.0-1.0)

## 5. Controlled Variables

| Field | Value | Rationale |
|-------|-------|-----------|
| problem.name | toy_kadane | Fixed problem |
| pipeline | standard | validate.py returns dict |
| max_generations | 3 | Toy -- should converge in 1 |
| max_mutations_per_generation | 2 | Minimal compute |
| max_elites_per_generation | 2 | Minimal archive |
| num_parents | 1 | Standard |
| mutation_mode | rewrite | Standard |

## 6. Run Design Table

| Run | Label | `redis.db` | `pipeline` | `problem.name` | `llm_base_url` |
|-----|-------|------------|-----------|----------------|----------------|
| 1 | R1 | 15 | standard | toy_kadane | https://openrouter.ai/api/v1 |
| 2 | R2 | 14 | standard | toy_kadane | https://openrouter.ai/api/v1 |

## 7. Sample Size Justification

N=2 -- minimum for the pipeline. This is a dry run, not real research.

## 8. Statistical Test

**Test**: None -- descriptive only (did fitness reach 1.0?).
**Significance threshold**: N/A
**How computed**: Direct observation.

## 9. Known Confounds and Mitigations

| Confound | Risk | Mitigation |
|----------|------|-----------|
| OpenRouter rate limits | Low | Use cheap model |
| Network latency | Low | Acceptable for toy |

## 10. Stop Criteria

**Early termination**: fitness=1.0 reached.
**Run invalidation**: N/A -- toy experiment.

## 11. Compute Budget

| Resource | Estimated usage |
|----------|----------------|
| GPU hours | 0 (API-based) |
| Wall time | ~5 minutes |
| Redis DBs used | 14, 15 |

## 12. Treatment Verification

**Observable evidence**: fitness metric in Redis reaches 1.0 (seed starts at 0.9).
**extra_overrides**: None -- single-arm design, no treatment vs control.

## 13. Open Questions / Risks

This is a pipeline test. The only risk is if GigaEvo's framework can't handle OpenRouter as a mutation LLM endpoint.
