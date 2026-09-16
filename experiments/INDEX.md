# Experiment Index

One row per experiment, updated by `/project-pm`.
Columns: name, status, key finding, PR link

**Server inventory**: [`infrastructure.yaml`](infrastructure.yaml)

---

## HotpotQA

| Experiment | Status | Key Finding | PR |
|-----------|--------|-------------|-----|
| `hotpotqa/thinking` | ✅ Complete | Established ddce37b4 warm-start seed (val 62.7%) with Qwen3-8B thinking mode. | #66 |
| `hotpotqa/p1p2` | ✅ Complete | E+G invalid (repr-contamination). F+H valid: ASI pipeline confirmed. | #67 |
| `hotpotqa/nlp_prompts` | ✅ Complete | NULL — NLP-specific mutation prompts no gain; val-test gaps 8-14pp; confounded compound treatment | #69 |
| `hotpotqa/push` | ✅ Complete | POSITIVE — GEPA attack with F1 x Prompts x Val-N (test EM 63.0%) | #73 |
| `hotpotqa/crossover` | ✅ Complete | NULL — num_parents=2 stagnation attack no improvement | #74 |
| `hotpotqa/cold_start` | ✅ Complete | SUGGESTIVE — basin escape via n=4 replication | #75 |
| `hotpotqa/colbert_feedback` | ✅ Complete | NEGATIVE — ColBERT + rich feedback no test improvement | #76 |
| `hotpotqa/gemini_mutation` | ✅ Complete | INCONCLUSIVE — Gemini as mutation LLM mixed results | #79 |
| `hotpotqa/prompt_coevolution` | ✅ Complete | NULL — prompt co-evolution no meaningful improvement | #84 |
| `hotpotqa/val_gap` | ❌ Invalid | Val-test gap reduction invalidated | #70 |
| `hotpotqa/generalization` | ❌ Invalid | Generalization via held-out validation invalidated | #81 |

## HoVer

| Experiment | Status | Key Finding | PR |
|-----------|--------|-------------|-----|
| `hover/baseline` | ✅ Complete | POSITIVE — cold-start baseline established (grand mean 51.65%) | #90 |
| `hover/feedback_softfit` | ✅ Complete | POSITIVE — soft fitness +3.42pp over baseline; failure feedback alone NULL (-0.18pp) | #92 |
| `hover/prompt_coevolution` | ✅ Complete | NULL — prompt co-evolution with soft fitness no gain | #93 |
| `hover/co-evolution-bus` | ✅ Complete | REGRESSIVE — high-throughput prompt meta-evolution did not help | #109 |
| `hover/dynamic-topology` | ✅ Complete | POSITIVE — dynamic topology beats fixed-topology and GEPA benchmark (+8.5pp test) | #116 |
| `hover/steady-state-validation` | ✅ Complete | INCONCLUSIVE — non-inferiority test underpowered at N=2; CI too wide | #133 |
| `hover/steady-state-v2` | ✅ Complete | POSITIVE — steady-state + LPT viable replacement, throughput benefits, no fitness penalty | #138 |
| `hover/map-elites-topology` | ✅ Complete | NULL — 3D structural BC no fitness gain (p=0.413) | #142 |
| `hover/7step-dynamic` | ✅ Complete | STRONG POSITIVE — +5.05pp val, +8.33pp test | #144 |
| `hover/no-deep-retrieval` | ✅ Complete | NULL — retrieve_deep unnecessary for dynamic chains (0.13pp delta); topology dominant | #150 |
| `hover/memory` | ✅ Complete | NULL/SUGGESTIVE — +0.61pp val, +1.47pp test, p=0.31 | #161 |
| `hover/dynamic-crossover` | ❌ Invalid | Import error bug differentially degraded runs, comparison uninterpretable | #130 |

## Adversarial

| Experiment | Status | Key Finding | PR |
|-----------|--------|-------------|-----|
| `adversarial/optimizer-coevo` | ✅ Complete | ASYMMETRIC — landscapes improve (+67.7pp), optimizers stagnate (+1.8pp); arms race not supported | #169 |
| `adversarial/heilbron-prover` | ✅ Complete | POSITIVE — Constructors reach 97% of Heilbronn target (min_area 0.0355); arms race asymmetric but actual_fitness STRO... | #183 |
| `adversarial/adversarial-vs-solo` | ✅ Complete | INCONCLUSIVE — solo mean 0.03267 vs adversarial 0.03449 (+0.00182, p=0.365); underpowered at N=4; d=0.70 medium effect | #203 |

## Toy / Validation

| Experiment | Status | Key Finding | PR |
|-----------|--------|-------------|-----|
| `toy/kadane_speedrun` | ✅ Complete | Validation run (no formal results) | — |

## Heilbron

| Experiment | Status | Key Finding | PR |
|-----------|--------|-------------|-----|
| `heilbron/adversarial-v2` | ✅ Complete | SUGGESTIVE POSITIVE — K=3 bidirectional feedback marginal (+0.00038 actual_fitness); K=1 regressed below baseline; Im... | #188 |
| `heilbron/adversarial-dynamic-updates` | ✅ Complete | NEGATIVE — archive re-eval hurts actual_fitness (-0.011 gap); GAN resistance gaming confirmed; all cells below baseline | #197 |
| `heilbron/baseline-repro` | ✅ Complete | SUGGESTIVE — baseline reproducible on best-overall (mean=0.03449 vs 0.03464); Constructor-only CI wide; 3 near-Q_MAX ... | #201 |
| `heilbron/asymmetric-iterations` | ✅ Complete | NEUTRAL (between arms) — Both feedback modes produced >=105% SOTA (0.03648, 0.03650). Composition vs gradient-in-prompt equivalent (delta 0.00081). K=5 NOT tested (K=1 amendment). Source code access accelerates early optimization. | #204 |
| `heilbron/asymmetric-iterations-v2` | ✅ Complete | INCONCLUSIVE — Replication with bug fixes. Feedback mode NULL (cross-arm delta 0.00066). Best max(G,D)=0.03588 (C1, 104% baseline). v2 underperformed v1 despite 4-6x more gens; min_delta sync fix identified as hidden confound. | #206 |
| `heilbron/k5-budget-v3` | ✅ Complete | INCONCLUSIVE — K=3 vs K=5 HoF-lag effect -0.00164 [-0.00393, +0.00065] 95% CI; K3_1 pair deadlocked on ProgressBasedSyncHook at gen 29/50; v3 2D MAP-Elites eliminated D fitness collapse (all D fitness > 0.50). N=2 underpowered. | (branch `exp/heilbron/k5-budget-v3`; PR not yet opened) |
| `heilbron/adversarial-repro-v1` | ✅ Complete | NULL — v1 recipe under bug-fixed library; mean best-ever actual_fitness 0.0341 [0.0306, 0.0377] 95% CI, in v2/baseline band (< v1=0.03574). A1_G reached 0.0365 evolutionarily (earliest at gen=5 during I-16-broken phase, D-to-G feedback inactive). Confound: `redis.resume=false` does not flush Redis, so broken-phase programs persisted post-relaunch. Systemic bugs discovered: I-17 CompositionInjectionHook mis-labels injected programs as `gen=1/iter=0/is_root=True` (fixed this experiment); relaunch protocol must flush DBs, not rely on resume flag. | #211 |

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| 📋 Designed | Phase 1-3 complete; not yet launched |
| 🟡 Running | Runs active |
| ✅ Complete | Phase 5 done; PR merged |
| ❌ Invalid | Invalidated; see PR for details |

---

*Auto-generated by `/project-pm`. Do not edit manually.*
