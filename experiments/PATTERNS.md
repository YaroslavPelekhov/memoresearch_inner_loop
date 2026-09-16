# Research Patterns

Cross-cutting insights consolidated from all completed experiments. Updated by `/experiment-closeout` and `/experiment-retrospective`.

All agents read this file before design, review, and retrospective analysis. Do not duplicate this content in agent-specific memories.

---

## Confirmed Patterns

Supported by 2+ experiments with consistent results.

| Pattern | Evidence | Confidence | Tasks |
|---|---|---|---|
| Dynamic topology >> fixed topology | hover/dynamic-topology (+8.5pp test vs GEPA), hover/7step-dynamic (+8.33pp test) | HIGH | HoVer |
| Prompt-level interventions fail consistently | hotpotqa/prompt_coevolution (NULL), hover/prompt_coevolution (NULL), hover/co-evolution-bus (REGRESSIVE) | HIGH | HoVer, HotpotQA |
| Soft fitness functions help modestly | hover/feedback_softfit (+2.72pp test, p~0.03) | MEDIUM | HoVer |
| Feedback alone is insufficient | hover/feedback_softfit: feedback-only cell NULL (-0.18pp) | MEDIUM | HoVer |
| Mutation prompt quality is NOT a binding constraint | 3 co-evolution experiments, 2 tasks, 8 treatment runs, 0 positive results | HIGH | HoVer, HotpotQA |
| Steady-state engine is non-inferior to generational | hover/steady-state-v2 (POSITIVE), hover/steady-state-validation (INCONCLUSIVE but CI includes parity) | MEDIUM | HoVer |
| Adversarial co-evolution drives actual_fitness despite asymmetric dynamics | optimizer-coevo: landscapes +67.7pp; heilbron-prover: Constructors +0.03462 (97% of Heilbronn target); baseline-repro: 0.03449 best-overall (N=4). adversarial-vs-solo: +0.00182 vs solo (p=0.365, d=0.70, underpowered). asymmetric-iterations: both arms produced >=105% SOTA (0.03648, 0.03650). asymmetric-iterations-v2: replication with tight coupling produced only 98-104% baseline (best 0.03588). k5-budget-v3 (2D MAP-Elites): all D populations fitness > 0.50 (collapse eliminated) but grand mean G 3.3% below v2 baseline — diversity cost. 10 experiments, 72+ runs confirm actual_fitness improves; magnitude is sensitive to coupling dynamics AND to fitness-function formulation (see D hard-floor pattern below). | HIGH | Adversarial |
| Improver stagnation is structural — ROOT CAUSE: D hard-floor fitness | Confirmed across 10 adversarial experiments (optimizer-coevo, heilbron-prover, adversarial-v2, adversarial-dynamic-updates, baseline-repro, adversarial-vs-solo, asymmetric-iterations, asymmetric-iterations-v2, k5-budget-loose, k5-budget-v3). K=0/1/3/5, bidirectional code, soft fitness, GAN resistance, archive re-eval, source code access, composition injection, gradient-in-prompt, loose coupling, tight coupling, 1D/2D BD — none break stagnation. **ROOT CAUSE IDENTIFIED 2026-04-16** (k5-budget-loose): D hard-floor fitness (`max(delta,0)/Q_MAX` at `pop_b/evaluate.py:78`) produces 60–90% point mass at fitness=0.0 with K=1. All 10 experiments ran on this structural defect; prior NULL results are expected under this hypothesis. G parallel flaw identified (`pop_a/evaluate.py:95`): binary `float(delta<=0)` resistance masked by 50% quality weight. Compute budget K=5 tested INCONCLUSIVE under broken fitness (k5-budget-v3). **The ONLY untested intervention class is FITNESS FUNCTION SMOOTHING** — REDESIGN bundle in `experiments/heilbron/k5-budget-loose/REDESIGN.md` (smoothed `tanh` fitness + deterministic HoF + K=L=3 + `cache_on` edges). | HIGH | Adversarial |
| D hard-floor fitness formulation produces point-mass collapse at 0.0 | k5-budget-loose empirical: 60–90% of D programs at fitness=0.0 exactly across all 4 D runs at gen 1–2; only 10–21 unique fitness values across 79–104 programs. Strategy rejection 56–79% on D vs 0–14% on G. With K=1 and `scores.append(min(max(delta,0.0)/Q_MAX, 1.0))`, fitness becomes binary {win=positive, lose=0.0}. MAP-Elites cannot diversify when majority of cells contain identical-fitness programs. Retroactive explanation for all 10 prior heilbron-adversarial experiments' Improver stagnation. Binding constraint — supersedes information-flow, coupling, and budget hypotheses until fixed. | HIGH | Adversarial |
| 2D MAP-Elites BD (fitness, wins) eliminates D collapse | k5-budget-v3: all 4 D populations maintained frontier fitness in [0.52, 0.68] throughout 50-gen runs; pre-registered failure criterion (D fitness → 0) NOT triggered. Qualitative structural fix vs 1D BD where D collapsed to 0.0 point mass across 9 prior experiments. Does NOT repair underlying fitness formula — diversifies niches instead. Grand mean G actual_fitness 3.3% below v2 baseline suggests diversity is traded for peak. | HIGH | Adversarial |
| Opponent context has a minimum effective dose | heilbron/adversarial-v2: K=1 regressed below K=0 baseline (0.03247 vs 0.03464). K=3 marginally above baseline (0.03502). Ordering: K=3 > K=0 > K=1. Single opponent exemplar may cause LLM over-anchoring, worse than no context. Use K>=3 or parsed critique. | MEDIUM | Adversarial |
| Simpler adversarial setups outperform complex ones | Best Heilbronn actual_fitness was 0.03548 from simplest config (K=0, no re-eval, generational engine) until asymmetric-iterations. Source-code-access arms produced 0.03648/0.03650 (v1, loose coupling) but only 0.03426/0.03588 (v2, tight coupling). Both feedback modes (composition vs gradient-in-prompt) are equivalent (8 pairs, 16 runs, cross-arm delta 0.00066-0.00081). Added mechanism complexity beyond source code access does not differentiate. 8 experiments confirm. | MEDIUM | Adversarial |
| Soft fitness resists metric gaming better than pure GAN resistance | adversarial-dynamic-updates: SOFT_C actual_fitness 0.03186 vs GAN_C 0.02854. GAN_C_A reached 99.6% resistance with poor geometry -- MAP-Elites analog of GAN mode collapse. | MEDIUM | Adversarial |
| Coupling granularity affects search effectiveness | asymmetric-iterations (min_delta=1, loose): best 0.03648/0.03650 (>=105% SOTA). asymmetric-iterations-v2 (min_delta=8, tight): best 0.03426/0.03588 (98-104% baseline). Tight coupling over-constrains search by forcing D to wait full G epochs. v2 ran 4-6x more generations but achieved lower peaks. The accidental loose coupling in v1 was a performance-enabling "bug." | MEDIUM | Adversarial |
| Feedback mode (composition vs gradient-in-prompt) does not affect outcomes | asymmetric-iterations: cross-arm delta 0.00081. asymmetric-iterations-v2: cross-arm delta 0.00066. Across 8 pairs, 16 runs, two independent experiments. Direction CLOSED. | HIGH | Adversarial |

## Refuted Hypotheses

Tried and failed — do NOT pursue further without new theoretical justification.

| Hypothesis | Evidence | Tasks |
|---|---|---|
| Prompt co-evolution improves fitness | 3 NULL/REGRESSIVE results (PR #84, #93, #109). Mean delta: -1.37pp. Research line CLOSED. | HoVer, HotpotQA |
| retrieve_deep is necessary for dynamic chains | hover/no-deep-retrieval: 0.13pp delta, topology dominant | HoVer |
| 3D structural behavior characterization adds value | hover/map-elites-topology: NULL (p=0.413) | HoVer |
| Archive re-evaluation improves adversarial co-evolution | heilbron/adversarial-dynamic-updates: NEGATIVE. Re-eval ON actual_fitness 0.0194 vs control 0.0302 (gap=0.011, 5.5x NEGATIVE threshold). Re-evaluation destabilizes archive in adversarial settings where opponent changes are frequent. | Adversarial |
| Pure GAN resistance is a viable fitness signal for MAP-Elites | GAN_C_A reached 99.6% resistance with actual_fitness only 0.02854 (below baseline 0.03464). Constructor learned to game resistance metric -- MAP-Elites analog of GAN mode collapse. | Adversarial |
| Feedback mode (composition vs gradient-in-prompt) affects adversarial outcomes | 8 pairs, 16 runs across asymmetric-iterations (delta 0.00081) and asymmetric-iterations-v2 (delta 0.00066). Both well below 0.002 threshold. Direction CLOSED — do not revisit. | Adversarial |
| Hard-floor fitness formulation (`max(delta,0)` / binary `float(delta<=0)`) is a viable signal for adversarial MAP-Elites co-evolution | k5-budget-loose empirical: 60–90% of D programs at fitness=0.0 point mass; D strategy rejection 56–79% vs 0–14% on G. 10 prior heilbron-adversarial experiments showed D stagnation under this formulation — retroactively explained by the structural defect. Hard floor eliminates evolutionary gradient for "almost improved" programs. Replace with smoothed `tanh(delta/Q_MAX)` rescaled to `(score+1)/2` per `experiments/heilbron/k5-budget-loose/REDESIGN.md`. | Adversarial |

## Suggestive Signals

Weak evidence warranting follow-up with stronger design.

| Signal | Evidence | Suggested Follow-Up |
|---|---|---|
| Memory mechanisms may help retrieval | hover/memory: +0.61pp val, +1.47pp test (p=0.31) | Different memory mechanism, larger N, or combine with topology |
| K=3 bidirectional feedback may improve Constructor actual_fitness | heilbron/adversarial-v2 K=3: 0.03502 (+1.1% vs baseline), best 0.03568 via Improver. Premature stop at 45% -- trend was upward. | Run K=3 to completion (gen 75), or test K=3 + D re-evaluation (issue #195) |
| Adversarial pressure may regularize fitness variance | adversarial-vs-solo: adversarial SD=0.00212 vs solo SD=0.00300. Solo bimodal (2/4 matched adversarial, 2/4 well below). Stagnation (S4: 17-gen plateau) and elevated invalidity (S2: 31%) observed only in solo arm. | N=8 replication to confirm variance reduction. Opponent pressure may provide escape from local optima. |
| Improver polishing exceeds Constructor quality | baseline-repro: In 3/4 pairs, Improver's best actual_fitness >= Constructor's (P1_B=0.0361 vs P1_A=0.0340; P2 tied; P4_B=0.0326 vs P4_A=0.0302). Mean Improver=0.0340, mean Constructor=0.0334. | Track best-overall (either population) as standard metric. |
| Source code access accelerates early optimization | asymmetric-iterations (v1, loose coupling): Both arms >=105% SOTA by gen 8-12. v2 (tight coupling): best found at gen 22-36, NOT reproducing the acceleration. WEAKENED by v2 replication failure — acceleration requires loose coupling + source code together, not source alone. Deprioritize below REDESIGN bundle and min_delta dose-response. | Isolate source code access from coupling granularity in factorial design (2x2: source ON/OFF x loose/tight), AFTER REDESIGN bundle resolves fitness signal. |
| High invalidity correlates with best results under aggressive feedback | asymmetric-iterations v1: C2_G 50.2% invalidity, best overall (0.03650). v2: C1_G 33% invalidity, best v2 result (0.03588). Runs with moderate-to-high invalidity (30-50%) tend to find best solutions. Pattern holds weakly across both experiments. | Monitor invalidity as a search strategy indicator, not just a quality metric. |
| Loose G/D coupling outperforms tight coupling | asymmetric-iterations v1 (min_delta=1): >=105% SOTA by gen 8-12. v2 (min_delta=8): 98-104% baseline despite 4-6x more generations. STRENGTHENED by v2 replication failure: tight coupling fails to reproduce v1 peaks across 4 runs × 2 feedback modes. Confounded with 6 v1→v2 bug fixes AND broken fitness (H3) — clean isolation blocked on REDESIGN bundle. Loose coupling allows D to run many micro-steps per G epoch, creating implicit compute budget asymmetry. Most actionable finding from v1/v2 comparison. | Test min_delta as IV (1 vs 4 vs 8) under REDESIGN bundle (`adversarial_014` — now contingent on REDESIGN rather than on adversarial_013). |
| G hard-floor resistance masks adversarial signal (parallel to D) | k5-budget-loose code analysis: `pop_a/evaluate.py:95` uses binary `float(delta<=0)` resistance per opponent, averaged over K opponents then weighted 50% with continuous quality. Creates discrete point masses masked by quality signal — never visible in standard monitoring because quality carries enough signal to keep MAP-Elites diversifying on the quality axis. Parallel pathology to D's hard floor; not yet tested for fix. | Apply `tanh(-delta/Q_MAX)` symmetrically with D fix (REDESIGN bundle Change 2). Fix cannot be tested in isolation — always paired with D fix. |
| ProgressBasedSyncHook deadlock may be K-dependent | k5-budget-v3: K3_1 pair deadlocked for ~2h at gen 29/50 (both sides waiting on each other's `min_progress` target with no timeout fallback). No deadlock observed in any K=5 pair. N=1 occurrence — cannot distinguish systematic K-dependence from stochastic bad luck. If K-dependent, smaller K may lead to tighter sync requirements. | Add deadlock guard with configurable timeout BEFORE any future co-evolution experiment (KF-07). Track deadlock frequency across K values in future runs. |

## GigaEvo Platform Failure Modes

Discovered during experiment design/execution. Check these proactively in every new experiment.

| Failure Mode | Status | Impact | Check |
|---|---|---|---|
| `include_in_prompts` as hidden IV | ACTIVE | Treatment/control get different mutation prompts when metrics.yaml differs | Compare prompt content parity across conditions |
| `model_name` config drift | ACTIVE | `endpoints.yaml` defaults to OpenRouter; local experiments must override | Verify with `--cfg job` |
| validate.py return type / pipeline mismatch | ACTIVE | dict vs tuple return silently corrupts with wrong pipeline | Audit all pipeline/problem.name pairings |
| Silent YAML wiring failures | ACTIVE | Config keys silently ignored if not wired in pipeline YAML | Always verify with `--cfg job` |
| Stale workers repopulating Redis | FIXED | Workers repopulate flushed DBs | Use `tools/flush.py` (kills workers first) |
| `pipeline=standard` stage_timeout not wired | FIXED | Timeout not passed to DefaultPipelineBuilder | Fixed in standard.yaml |

## Known Failures

Structured entries for `/experiment-implement` and `/experiment-launch` to check proactively. Each entry describes a specific failure mode with enough detail to detect and prevent it. Updated by `/post-experiment-fixes` and `/experiment-closeout`.

| ID | Trigger Condition | Symptoms | Root Cause | Fix | Status | Affected Types | Source |
|----|-------------------|----------|------------|-----|--------|----------------|--------|
| KF-01 | adversarial pipeline + missing `evolution=steady_state` override | All runs deadlock at gen 0 | MainRunSyncHook waits on `total_generations` incremented after hook call; circular wait in SteadyState | Override `pre_step_hook` to `ProgressBasedSyncHook`; or add `evolution=steady_state` to extra_overrides | FIXED (e69021c0) | adversarial | heilbron/asymmetric-iterations |
| KF-02 | extra_overrides containing `${}` Hydra interpolation refs | launch.sh expands `${...}` as empty shell variable | `generate_launch.py` does not quote Hydra interpolation refs | Single-quote all `${}` refs in extra_overrides in `_build_run_cmd` when `shell_escape=True`. **Note**: writing `\${...}` in experiment.yaml is cosmetic — ruamel.yaml silently strips the backslash, so the real defense is the bash-emit-time quoting, not the YAML author's escape attempt (I-04). | FIXED (fdd3dae1) | all | heilbron/asymmetric-iterations |
| KF-03 | Missing `population_role` in adversarial_asymmetric runs | Pipeline cannot differentiate G vs D roles; wrong stages run | `experiment.yaml` missing per-run `population_role=constructor/improver` overrides | Add `population_role=constructor` or `population_role=improver` per run in experiment.yaml | FIXED (config default + experiment-implement skill) | adversarial_asymmetric | heilbron/asymmetric-iterations |
| KF-04 | CompositionInjectionHook programs missing `iteration` field | MetricsTracker crashes on KeyError, kills entire tracker async task; frontier never updates | Programs created without `iteration` in metadata | Promoted `iteration` to typed Program field with `default=0` | FIXED | adversarial with composition | heilbron/asymmetric-iterations |
| KF-05 | `min_delta=1` in ProgressBasedSyncHook with asymmetric runs | D runs 2.5x faster than G; massive desync | D processes 1 program and unblocks immediately; no epoch-level sync | Set `min_delta` to `${max_mutations_per_generation}` Hydra ref for proper epoch-level sync | FIXED | adversarial_asymmetric | heilbron/asymmetric-iterations |
| KF-06 | Telegram sendPhoto 400 on large plots | Plot not delivered to researcher; monitoring gap | Plot file exceeds Telegram 10 MB photo limit | Fall back to sendDocument (50 MB limit) on HTTP 400 | FIXED | all | heilbron/asymmetric-iterations |
| KF-07 | Two co-evolving populations both blocked on each other's `min_progress` target with no grace period | Both runs stall indefinitely; no forward progress; wall-clock budget wasted; external cancel required | Symmetric wait condition `opponent_progress >= my_last_sync + min_delta` allows mutual deadlock when both populations reach their hooks simultaneously | Replaced with asymmetric drift-cap semantics: `while own_progress - min(opponent_progress) > drift_cap: sleep()`. Only the ahead side ever blocks; behind side's condition is always satisfied. Deadlock impossible by construction. PR #??? | FIXED | adversarial (all co-evolution) | heilbron/k5-budget-v3 |
| KF-08 | Programs with legacy fields (e.g., `iteration`) or task-specific pickle refs processed through `archive_run.sh` | Empty `evolution_data.csv` with exit 0; silent data loss | (A1) Pydantic `extra="forbid"` on Program model rejects programs with legacy fields. (A2) pickle metadata refs fail outside task Python path | `extra="forbid"` → `extra="ignore"` on Program; add pickle fallback; add CSV-emptiness guard with non-zero exit | OPEN | all (any evolving Program fields) | heilbron/k5-budget-v3 |
| KF-09 | Checkpoint or monitoring read occurs before first `step()` completes | Checkpoint records show `best_fitness=0.0`, `gen=null`; polluted monitoring data | `EvolutionEngine.run()` does not persist `engine:total_generations=0` to Redis before entering step loop | Write `engine:total_generations=0` in `run()` before first `step()` call | OPEN | all | heilbron/k5-budget-v3 |
| KF-10 | Frozen `problems/<task>/evaluate.py` from an older commit returns bare `dict` instead of `(dict, artifact)` tuple | `DGImprovementTracker` keys empty across all runs; `CompositionInjectionHook` logs `injected=0` forever; `GradientInPromptStage` skips every iteration with `no D to inject`; zero Lamarckian/gradient feedback over the whole run | `CallValidatorFunction.parse_output` silently wraps a bare `dict` as `(dict, None)`; `DGTrackerStage.compute` then sees `artifact=None`, reads `per_opp_delta=[]`, hits length-mismatch guard, skips batch | Update frozen `evaluate.py` to return `(metrics, artifact)` with `artifact={"role": "<constructor|improver>", "per_opp_delta": [...]}` aligned index-wise to `opponent_results`. Add treatment verification that asserts non-empty `dg_best_pairs` Redis key within first N steps | FIXED | adversarial (all co-evolution with DGTrackerStage) | heilbron/adversarial-repro-v1 |
| KF-11 | `CompositionInjectionHook` constructs `Program(code=..., metadata={...})` without passing `lineage=` | Every injected D∘G program stamped `lineage.generation=1`, `iteration=0`, `parents=[]`, `is_root=True` regardless of its G parent's actual generation; all `is_root=True` / gen-1 archive slices contaminated with Lamarckian descendants | `Program.__init__` falls back to `Lineage()` defaults when no lineage supplied; `Lineage.is_root` derives `True` from empty `parents`. Hook never invokes the `create_child` factory | Use `Program.create_child(parents=[g_prog], code=..., mutation="d_improvement")` so `generation = G.generation + 1`, `parents = [g_id]`, `is_root=False`. D NOT added to `parents` (separate Redis DB); D reference stays in `metadata.d_source_id`. Tests: `TestInjectedLineage` in `tests/adversarial_pipeline/test_composition_injection.py` | FIXED (b2d8bdf9) | adversarial_asymmetric with CompositionInjectionHook | heilbron/adversarial-repro-v1 |
| KF-12 | Relaunch protocol uses `redis.resume=false` to signal "fresh start" without flushing Redis DBs | Pre-relaunch programs persist into post-relaunch archive; fitness and lineage data from broken (pre-fix) phase contaminates post-fix run; closeout analysis mis-attributes pre-relaunch discoveries as post-relaunch results | `redis.resume=false` only controls engine state reload (total_generations, etc.); it does NOT flush the Program storage DB. Launch-time hook had no "fresh Redis" sanity check | Flush DBs explicitly via `gigaevo flush --db N --confirm` (or bulk via manifest) as part of the relaunch skill. Add startup assertion: if `redis.resume=false` AND Program storage DB is non-empty, WARN or require explicit `--force-resume-on-dirty-db` flag | OPEN | all (but especially post-bug-fix relaunches) | heilbron/adversarial-repro-v1 |

## Recurring Design Flaws

Patterns that Volkov flags repeatedly — address at template level.

| Flaw | Occurrences | Fix |
|---|---|---|
| MDE formula missing t_beta power term | 3 consecutive designs | Fix in design template section 7 |
| task_description.txt / validator behavior contradictions | 2 occurrences | Pre-launch check: constraints in task_description must match validator |
| initial_programs/baseline.py not byte-identical across conditions | 1 occurrence | Preflight check: verify seed identity |

## Protocol Gaps

From methodology audits (Hiroshi, 2026-03-20 and 2026-04-07).

| Gap | Severity | Status |
|---|---|---|
| Stopping rules unenforced — free-text in 01_design.md §10 | Critical | OPEN |
| N=2-4 systematically underpowered (σ≈0.63pp) | Critical | ACCEPTED — formal power not required, acknowledge honestly |
| Test set reuse across 10+ experiments | Major | OPEN — no rotation or family-wise correction |
| Val-test gap systematic (4-8pp) | Major | OPEN — no watchdog alert |
| Dataset checksums not verified at launch | Minor | OPEN |
| Partial blinding structurally limited (labels identify conditions) | Major | OPEN — acknowledged limitation |

## Open Questions (highest expected information gain)

| Question | Why It Matters | Suggested Design |
|---|---|---|
| **Does the REDESIGN bundle (smoothed `tanh` fitness + deterministic HoF + K=L=3 + `cache_on` edges) break D-collapse and lift G actual_fitness above baseline?** | **NEW #1 PRIORITY** (2026-04-19 retrospective). Addresses root cause of Improver stagnation (D hard-floor fitness). First intervention targeting the actual bottleneck — all 10 prior experiments tested info/coupling/budget on a broken fitness landscape. Positive result reopens entire intervention space for proper re-testing. Negative result implies fundamental ceiling, escalate to Pareto coevolution. Full spec in `experiments/heilbron/k5-budget-loose/REDESIGN.md` (~40–70 LOC diff, mostly config + edge wiring). | 2×2 factorial: fitness (linear vs smoothed) × feedback mode (composition vs gradient-in-prompt as calibration axis). N=2/cell. Primary: D fitness distribution at gen 25 (point mass check). Secondary: G actual_fitness at gen 50. 8 runs × gen 50 ≈ 24h. Requires KF-08/09 fixes pre-launch (KF-07 deadlock fixed via drift-cap redesign). |
| Does asymmetric iteration ratio (WGAN-GP K:1) break Improver stagnation? | **CONFOUNDED-TESTED** (k5-budget-v3 2026-04-19): K=3 vs K=5 effect −0.00164 [−0.00393, +0.00065] 95% CI — INCONCLUSIVE. Confounded by (a) K3_1 ProgressBasedSyncHook deadlock truncating at 58% of target gens, (b) N=2 underpowered, (c) **ran under broken hard-floor fitness — uninterpretable against fitness-smoothing hypothesis even if clean**. K=5 is no longer "untested" but still effectively unanswered. `adversarial_013` should be marked BLOCKED-ON-REDESIGN. | Re-run under REDESIGN bundle at N=4/arm with KF-07 deadlock guard + pre-registered `max_runtime_hours` cap. Contingent on REDESIGN bundle being positive. |
| Does coupling granularity (min_delta) affect search effectiveness? | **CONTINGENT-ON-REDESIGN**: min_delta=1 (v1, accidental) produced >=105% SOTA; min_delta=8 (v2, intentional) produced only 98-104% despite 4-6x more generations. Tight coupling may over-constrain search; loose coupling may create implicit D compute asymmetry. Testing min_delta under broken fitness is uninterpretable. `adversarial_014` conditional should be updated from "contingent on adversarial_013" to "contingent on REDESIGN bundle". | 3-arm design: min_delta=1 vs 4 vs 8 with REDESIGN bundle applied to all arms. |
| Does source code access (white-box) drive the early acceleration seen in asymmetric-iterations? | **WEAKENED by v2**: v1 acceleration (>=105% SOTA by gen 8-12) NOT reproduced in v2 despite identical source code access. Deprioritize below REDESIGN bundle and min_delta dose-response. | 2x2 factorial: source code ON/OFF × loose/tight coupling, under REDESIGN bundle. |
| Does 2D BD (fitness, wins) specifically prevent D collapse, or is smoothed fitness alone sufficient with 1D BD? | **NEW from k5-budget-v3 (2026-04-19)**: v3 2D BD eliminated D collapse (all D > 0.50) but ran under hard-floor fitness — cannot disentangle BD dimensionality from fitness smoothing. | 2-arm ablation (1D vs 2D) under REDESIGN bundle. N=3/arm. Contingent on REDESIGN positive. If smoothed fitness alone prevents collapse, 1D may yield higher peaks (no diversity tax). |
| ~~Does adversarial co-evolution beat non-adversarial MAP-Elites on Heilbronn?~~ | **ANSWERED INCONCLUSIVE** (adversarial-vs-solo PR #203): Solo mean 0.03267 vs adversarial 0.03449 (+0.00182, p=0.365, d=0.70). Underpowered at N=4 (55% power). Solo bimodal. **Line should NOT be closed.** | Increase to N=8/arm. 4 additional solo runs needed. |
| ~~Does D re-evaluation fix Improver stagnation?~~ | **ANSWERED NEGATIVE** (adversarial-dynamic-updates): per-program fingerprint re-eval hurt actual_fitness (-0.011). | CLOSED. Do not pursue without fundamentally different re-eval mechanism. |
| Can topology + soft fitness stack? | Two strongest HoVer interventions untested together | 2x1: dynamic-topology with/without soft fitness |
| Can structured Improver operators break stagnation? | 7 experiments confirm stagnation is robust to information changes. K=5 budget hypothesis untested. If K=5 also fails, stagnation is a search space problem. Perturbation-policy Improver reduces search space. Priority: after K=5 re-test. | adversarial_003: structured move Improver. Run after K=5 budget test resolves budget-vs-search-space question. |
| Is HoVer approaching saturation after topology gains? | Determines whether to continue HoVer or switch tasks | Compare current SOTA to theoretical ceiling |
| Can chain LLM scaling (Qwen3-32B) unlock gains? | Stronger base model may dominate algorithmic tricks | Simple A/B: 8B vs 32B on same topology |

---

*Updated by `/experiment-closeout`, `/experiment-retrospective`, and `/post-experiment-fixes`. All agents read; no agent duplicates in its own memory.*
