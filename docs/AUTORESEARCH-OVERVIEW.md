# Autoresearch System Overview

**Snapshot date:** 2026-08-20
**Evidence covered:** Zotero `autoresearch` collection, the latest relevant screencast transcript (2026-08-17), recent Obsidian todos through 2026-08-18, the shared Google Slides deck, and the current GigaEvo summer-school repository.

## Executive summary

The planned system is a **memory-centered, human-steerable, execution-grounded autoresearch system**. It should integrate with an existing experiment repository, transform prior papers and experiment results into an interpretable research memory, propose evidence-grounded next ideas, and improve both the target code and its own memory organization over time.

GigaChat architecture research is an important initial application and supplies the first LLM Foundry experiment environment, but it is not the primary product definition.

It should not be merely GigaEvo with a larger prompt or a collection of successful mutations. The intended research loop is:

```text
papers + new models + training logs + human R&D experience
                            |
                            v
problem/reason -> hypothesis -> method -> implementation specification
                            |              [human approval initially]
                            v
                 multiple compliant implementations
                            |
                            v
                  controlled experiments
                            |
                            v
             report + verdict + failure analysis
                            |
                            v
          persistent research memory + meta-analysis
                            |
                            +---------------------------> next hypotheses
```

Its strongest prospective differentiator is memory that models **the research process itself**: hypotheses, mechanisms, failed implementations, negative results, scale-transfer evidence, provenance, and human judgment. This goes beyond retrieving previously successful code or tactics.

The concept is coherent, but the implementation and experimental protocol are not yet settled. The first implementation deliberately fixes one model scale, training setup, dataset, and evaluation. Changing scale or promoting experiments across scales belongs to a human-controlled meta-loop, not to the core v0 system.

## Evidence inspected

### Zotero collection

The Zotero collection `autoresearch` contains eight records: seven PDFs and one saved webpage.

1. **EvoMem: Memory-Augmented Evolution for Code Optimization**
   - Local PDF: `/home/booydar/Sync/zotero/storage/SSSHNIIK/Gigaevo_memory (4).pdf`
   - Persistent cross-run and cross-task memory for GigaEvo.
   - Stores abstract idea/tactic cards and strong program exemplars.
   - Performs offline extraction after a run and bounded retrieval before mutations in later runs.
   - Uses multi-view retrieval and conservative merging, and logs retrieved memory IDs for attribution.
   - Evidence is promising but variable. The study does not cleanly isolate memory from extra LLM tokens and judgments with token-matched, shuffled-memory, and component controls.

2. **Towards Execution-Grounded Automated AI Research**
   - Local PDF: `/home/booydar/Sync/zotero/storage/BZNXG6IT/Si et al. - Towards Execution-Grounded Automated AI Research.pdf`
   - Converts natural-language ideas into code changes, schedules GPU jobs, and searches using execution results.
   - Uses an Implementer, Scheduler, and Worker, with parallel implementation attempts and limited repair.
   - Demonstrates that execution-grounded search can improve nanoGPT pretraining and GRPO post-training.
   - Identifies small-to-large transfer, noisy scalar rewards, idea/code mismatch, executor quality, and missing novelty objectives as important limitations.

3. **GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms**
   - Local PDF: `/home/booydar/Sync/zotero/storage/GT5KTMXH/Khrulkov et al. - 2025 - GigaEvo An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms.pdf`
   - Provides the asynchronous evolutionary substrate: program storage, DAG execution, MAP-Elites/islands, LLM mutation, metrics, and lineage.
   - It is useful infrastructure, but does not itself define a research methodology, scale-promotion protocol, or durable scientific memory.

4. **AlphaGo Moment for Model Architecture Discovery / ASI-Arch**
   - Local PDF: `/home/booydar/Sync/zotero/storage/MM7NQZTM/Liu et al. - 2025 - AlphaGo Moment for Model Architecture Discovery.pdf`
   - Uses a Researcher -> Engineer -> Analyst pipeline, literature-derived knowledge, experiment history, and real training.
   - A close conceptual competitor to the planned end-to-end architecture factory.
   - Its most reusable pattern is the explicit separation of proposal, implementation, execution, and analysis.

5. **Language Modeling by Language Models / Genesys**
   - Local PDF: `/home/booydar/Sync/zotero/storage/T9WHARIM/Cheng et al. - 2025 - Language Modeling by Language Models.pdf`
   - Literature-aware, multi-agent architecture discovery with structured components, unit tests, smoke training, and staged verification.
   - Demonstrates an important failure mode: generated code can simplify or deviate from the proposal to pass automated checks.
   - Supports the need for an explicit specification-to-implementation verifier and selective human review.

6. **OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery**
   - Local PDF: `/home/booydar/Sync/zotero/storage/SMKTGRQE/Liu et al. - 2026 - OR-Agent Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery.pdf`
   - Organizes work as a structured research tree rather than simple mutation/crossover.
   - Uses short-term reflection, long-term verbal momentum, and compressed memory.
   - Closely resembles the proposed ResearcherOS hierarchy, although it targets operations-research algorithms rather than model architecture and scaling.

7. **Can I Borrow Your Graph? Revisiting Feature Engineering with Evolved DAG Representations**
   - Local PDF: `/home/booydar/Sync/zotero/storage/6GNK92LB/main (48) (2).pdf`
   - Uses typed DAGs and schema-constrained diffs to reject invalid candidates before evaluation.
   - Suggests that autoresearch should evolve typed architectural specifications and components, not unrestricted code whenever possible.

8. **AIDE²: The First Evidence of Recursive Self-Improvement**
   - Saved Zotero webpage: `/home/booydar/Sync/zotero/storage/QEM8C556/2607.html`
   - Official post: <https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement>
   - Places an outer autoresearch loop around an inner research agent.
   - Reports seven retained agent improvements over 100 steps and layered defenses against reward hacking.
   - Also exposes a major cost of unrestricted evolution: the evolved harness becomes complex, difficult to understand, and difficult to integrate into production.

### Presentation

Original deck:
<https://docs.google.com/presentation/d/1t1PaniFhecN94dxRpfJ54A3aK9z6hfLGU6RY0QmTbb0/edit>

Rendered local export:
`tmp/presentation/shared-autoresearch.pptx`

The deck describes:

- An ideas factory fed by papers, new model releases, internal R&D experience, and training logs.
- Human formalization and filtering of ideas before multiple implementations enter evolution.
- Research priorities including long-context quality, generation speed, speculative decoding, layer ordering, norms, MoE, mixed precision, and analysis of training logs.
- A ResearcherOS hierarchy: reason/problem -> hypothesis -> method -> implementation -> experiment -> report/reflection.
- Multi-stage testing, with broad cheap search followed by progressively larger validation.
- Population islands for distinct research directions such as looped models, latent reasoning, alternative attention/memory, sparse MoE, and diffusion.

The linked slide anchor points to an `Approach draft` divider; the substantive architecture is spread across the surrounding slides.

### Screencasts

Most relevant raw transcript:

- `/home/booydar/Videos/Screencasts/2026-08-17 17-09-24.txt`
- English summary: `/home/booydar/Videos/Screencasts/2026-08-17 17-09-24_recurrent_memory_experiments_and_autoresearch_system_design_summary_en.txt`

Important conclusions from the raw discussion:

- Human ownership should initially extend through the idea/method level; agents should begin by exploring implementations of an approved idea.
- One approved idea may produce several implementations, making the early loop a kind of intelligent grid/search process.
- LLM parallel search produces many duplicates and has weak native originality. The system needs a solution library, duplicate detection, diverse subpopulations, and deliberately unusual proposals.
- Candidate implementations may reward-hack the metric—for example, silently adding full attention. A separate compliance checker and tests are required.
- Memory should store concise human-readable records of what changed, why it changed, and what happened, rather than injecting raw histories into context.
- Implementation errors and conceptual/methodological failures should be separated in an error bank.
- A meta-analysis or thinking agent should examine history, predict promising directions, and steer hypotheses. This is intended to differentiate the system from pure evolution.
- The ultimate output is a GigaChat architecture that survives scaling laws and can be adopted in the product, not merely a strong proxy score.

Other relevant summaries:

- 2026-08-13: integration options, modular Full Attention/Gated DeltaNet components, resource limits, and paper-derived hypotheses.
- 2026-08-06: scale risk, multi-stage validation, metrics beyond perplexity, human filtering, and GPU planning.
- 2026-07-30: shift toward Autoarcher, GigaEvo backend, explicit component interfaces, and honest quality/compute tradeoffs.

### Obsidian todos

Most relevant notes:

- `/home/booydar/Sync/obsidian-db/_todo/2026-08-18.md`
- `/home/booydar/Sync/obsidian-db/_todo/2026-08-10.md`
- `/home/booydar/Sync/obsidian-db/_todo/2026-08-03.md`
- `/home/booydar/Sync/obsidian-db/_todo/2026-07-27.md`

The latest operational target is a simple system that can run iteratively, be repaired if it stalls, and produce useful samples for approximately two weeks. Assigned concerns include the solution library, human idea decisions, low-level evolution, implementation compliance tests, human-readable summaries, and an error bank.

## Consolidated planned system

### 1. Research control plane

Use durable, typed research objects:

- **Source:** paper, new model, log observation, or human insight.
- **Problem/Reason:** observed limitation that motivates research.
- **Hypothesis:** falsifiable claim and expected mechanism.
- **Method:** proposed architectural intervention.
- **Implementation specification:** allowed files/components, invariants, forbidden shortcuts, and expected resource envelope.
- **Candidate:** exact code or commit implementing the specification.
- **Experiment:** model size, data, schedule, seeds, budget, environment, and hashes.
- **Result:** quality, throughput, memory, stability, and failure metadata.
- **Verdict:** supported, refuted, open, or invalid implementation.
- **Memory claim:** compact reusable conclusion with provenance and confidence.

Important relationships should be explicit: `derived-from`, `tests`, `implements`, `parent-of`, `contradicts`, `transfers-to`, and `promoted-to-scale`.

### 2. Human/autonomy boundary

Initial boundary:

- Humans choose and approve the problem, hypothesis, method, and implementation specification.
- Agents generate multiple candidate implementations, repair them, execute experiments, and prepare reports.
- Humans review high-uncertainty, high-cost, or suspicious cases.

The boundary can move upward only after the lower-level system demonstrates reliable specification compliance and useful experimental decisions.

### 3. Execution plane

Suggested roles:

- **Ideator:** proposes hypotheses from sources and memory.
- **Specifier:** turns an approved method into a testable implementation contract.
- **Implementer:** creates candidate code.
- **Verifier:** checks code against the specification and detects shortcuts.
- **Scheduler/Executor:** runs isolated experiments and captures complete provenance.
- **Analyzer:** compares results against noise and the preregistered decision rule.
- **Meta-researcher:** studies history and recommends the next experiment or branch.

Evolution should initially operate **inside an approved method/specification**, exploring alternative compliant implementations. Research-direction islands may be added after the basic loop is reliable.

### 4. Verification

Use deterministic checks before LLM judgment:

- AST and dependency checks.
- Allowed/forbidden component checks.
- Unit and property tests.
- Parameter, FLOP, memory, latency, and throughput checks.
- Dataset, tokenizer, evaluation, and schedule hashes.
- Code-diff-to-specification comparison.
- Duplicate and near-duplicate detection.
- Checks for hidden fallback components such as unrestricted full attention.

An LLM verifier and human review are useful for semantic mismatches that cannot be expressed deterministically, but should not be the first or only defense.

### 5. Memory

Keep raw evidence immutable and expose several purpose-specific views:

1. **Experiment ledger:** all candidates, protocols, results, artifacts, and provenance.
2. **Working/frontier memory:** active branches, unresolved questions, and current best evidence.
3. **Tactic memory:** reusable EvoMem-style mechanisms and implementation advice.
4. **Failure/anti-pattern bank:** invalid code, reward hacks, numerical failures, and refuted mechanisms.
5. **Environment-transfer memory:** which effects and rankings transfer when a human changes model size, data, schedule, or another fixed experiment definition.
6. **Human-preference record:** accepted/rejected ideas and explanations of research taste or product constraints.

Retrieval should be bounded, logged, attributable, and allowed to abstain. The meta-researcher may revise summaries and beliefs, but must never rewrite the raw experiment record.

### 6. Objective and selection

Avoid a single unconstrained fitness value. Use hard constraints plus a Pareto view of:

- validation quality;
- generation/training throughput;
- GPU and CPU memory;
- numerical stability;
- implementation complexity and maintainability;
- compatibility with the fixed experiment's human-defined constraints.

Every experiment should have a preregistered decision rule and an explicit noise model. One mutation should ideally correspond to one identifiable mechanism; combined changes require automatic or scheduled ablations.

## Future human-controlled scale meta-loop

The source material currently contains three incompatible scale ladders:

- Presentation: approximately **100M -> 400M -> 1-1.3B**.
- 2026-08-10 todo: **3B-MoE -> 9B -> 27B**, with 27B considered a better production proxy.
- 2026-08-17 recording: **1B as a minimum useful signal**, with approximately 32B mentioned as a possible proxy for 100B-scale decisions.

This disagreement does not need to be resolved inside the first autoresearch implementation. Version 0 fixes one scale and one experiment definition. A human may later initiate a separate calibration or promotion study.

Recommended calibration experiment:

1. Select 10-20 already understood architectural changes, including positive, neutral, and negative interventions.
2. Fix the data mixture, tokenizer, optimization schedule, and evaluation suite.
3. Test all changes at affordable sizes with repeated seeds.
4. Promote a smaller subset to larger sizes.
5. Measure Spearman/Kendall rank correlation, winner survival, effect sizes relative to seed noise, and calibration of early learning curves.
6. Select the cheapest stage that predicts promotion decisions with sufficient confidence.
7. Preserve an explicit `insufficient evidence / do not promote` outcome.

This calibration remains important before applying small-scale discoveries to much larger models, but it is not a property or responsibility of the v0 autoresearch loop.

## Comparison with related systems

### EvoMem

EvoMem provides persistent tactic retrieval for evolution. The planned system should additionally represent hypotheses, negative results, invalid implementations, scale transfer, human decisions, and changes in the memory/research policy itself. It should also evaluate memory causally with no-memory, shuffled-memory, and token-matched controls.

### Execution-Grounded Automated AI Research

Execution-Grounded provides a strong implementation/execution engine and simple evolutionary or RL search. The planned system adds a durable and evolvable research memory, explicit human gates, specification compliance, easy experiment-repository integration, and long-term meta-analysis.

### ASI-Arch and Genesys

These are close to the full architecture-discovery vision. A meaningful differentiation requires more than literature retrieval and multi-agent implementation. The distinguishing features should be a continually updated and evolvable research memory, typed provenance from evidence to idea to code, easy integration with an existing experiment repository, and explicit human collaboration.

### OR-Agent

OR-Agent is close in its structured research tree and hierarchical reflection. The planned system adds expensive multi-fidelity model experiments, product constraints, implementation compliance, and a deliberate human/autonomy boundary.

### AIDE²

AIDE² evolves the research harness itself and demonstrates anti-reward-hacking adaptations. It also warns that unrestricted evolution can create opaque and hard-to-maintain systems. The planned system should preserve modular contracts and typed components rather than evolving the entire harness without structural constraints.

### TabDAG

TabDAG demonstrates that typed, schema-constrained representations save evaluations and improve provenance. This pattern should be adopted for model component composition and experiment specifications.

## Current repository status

Relevant project documents:

- [PLAN-model-autoresearch.md](PLAN-model-autoresearch.md)
- [GIGAEVO-FORK-MAP.md](GIGAEVO-FORK-MAP.md)

Summer-school repository:

- `remote/tools/gigaevo-summer-school`

The repository already provides useful substrate:

- candidate lifecycle and execution;
- LLM mutation;
- metrics and archives;
- concurrency;
- repository worktrees/diffs in repo-harness mode;
- reflections and mutation context in the fuller pipeline.

The minimal autoresearch preset is currently a bring-up harness rather than the planned research system:

- `config/experiment/autoresearch.yaml` uses a small greedy `1+K` setup with one island/cell;
- `config/pipeline/repo_harness_autoresearch.yaml` omits reflection, insight, lineage, memory, meta-selection, and archive curation;
- the default repo-harness mutation backend may be a dry-run backend unless explicitly overridden;
- persistent memory is primarily an offline two-run retrieval mechanism, not an online research ledger and meta-analysis loop;
- no typed idea/specification objects or compliance gate are present;
- no scale-promotion, noise, or product-objective layer is present.

The root implementation plan sensibly recommends beginning in classic program mode with one mutable component, a fresh subprocess per candidate to prevent CUDA/process leakage, low-variance greedy search, and simple JSONL memory. Repo-harness mode is likely more useful later when integration requires changes across LM Foundry or another multi-file training repository.

### Gap summary

| Layer | Current summer-school fork | Planned system |
|---|---|---|
| Candidate execution | Mostly present | Reuse and harden |
| Evolution/archive | Present | Organize by research direction and evidence |
| Reflection | Partial; full pipeline only | Structured verdicts and hypothesis updates |
| Persistent memory | Offline tactic retrieval | Continual research graph and failure memory |
| Idea governance | Prompt-level | Typed specifications plus human gate |
| Correctness | Smoke checks and prompt rules | Deterministic and semantic compliance verification |
| Scaling | Absent | Fixed in v0; changed by a human meta-loop |
| Objectives | Generic scalar metrics | Experiment-specific fixed objectives and constraints |
| Meta-research | Absent | Historical analysis and experiment steering |

ResearcherOS is a plausible human-readable control plane because it already represents problem -> hypothesis -> method -> experiment -> report -> knowledge. It is not currently connected to GigaEvo and should not be mistaken for the execution/search engine.

## Recommended two-week MVP

Choose one approved idea family, such as layer ordering, norm removal, or Gated DeltaNet/full-attention placement. Do not begin with all proposed islands.

1. One mutable component or file with a typed configuration.
2. Fixed data mixture, tokenizer, schedule, and evaluation suite.
3. Greedy `1+4` search on two GPUs, one generation at a time.
4. Human approval of each method/specification before implementation.
5. Deterministic compliance checks followed by smoke training.
6. Repeated seeds for candidates whose initial gain may exceed measured noise.
7. Quality, throughput, memory, stability, and maintainability metrics.
8. Structured JSONL experiment records plus a daily human-readable report.
9. Offline or next-day memory initially; no self-modifying memory architecture yet.
10. Randomized memory/no-memory or shuffled-memory proposals to measure whether memory helps.
11. Keep scale, data, schedule, and evaluation fixed for the whole run.
12. Long-running monitoring, failure triage, restartability, and complete artifact paths.

MVP success should mean:

- high valid-run rate;
- high specification-compliance rate;
- duplicate rate under control;
- effects distinguishable from seed noise;
- reproducible artifacts and decisions;
- tolerable human review time per candidate;
- evidence that memory improves proposal quality or sample efficiency.

Finding one lower validation loss is not sufficient evidence that the research system works.

## Recommended refinement priorities

1. **Freeze one experiment contract.** Keep scale, data, schedule, evaluation, and fitness immutable during v0.
2. **Narrow the change surface.** One idea family and one component boundary for the MVP.
3. **Define the typed research schema.** Make hypothesis, method, specification, candidate, experiment, and verdict first-class objects.
4. **Build compliance before creativity.** Reward hacking and idea/code mismatch are already observed failure modes.
5. **Preserve failures and rejected candidates.** Successful elites alone create survivorship bias and an overly optimistic memory.
6. **Measure memory causally.** Do not infer value merely because memory-assisted runs sometimes win.
7. **Use evolution only where it helps.** At low throughput, batched experimental design or Bayesian selection may be more efficient than MAP-Elites. Evolution is most valuable for diversity, recombination, and multiple viable branches.
8. **Control combined mutations.** Prefer atomic mechanisms or schedule ablations so results remain scientifically interpretable.
9. **Quantify the human bottleneck.** Record reviewer minutes, disagreement, reversals, and which cases actually required human judgment.
10. **Keep the system maintainable.** Constrain self-modification to stable interfaces and separately version the research policy, memory policy, and experiment code.

## Overall assessment

The most credible framing is:

> A memory-centered, execution-grounded, human-steerable system that integrates with existing experiments and continually learns how to conduct research.

If the result is only EvoMem cards added to GigaEvo, it will overlap heavily with existing work. The genuinely distinctive contribution is the combination of:

- typed idea-to-code provenance;
- explicit human/autonomy boundaries;
- memory of failures and negative evidence;
- a versioned memory structure that can itself be refined;
- long-running meta-analysis over a research program;
- simple integration with external experiment code and maintainability from the beginning.

The first scientific milestone is therefore not autonomous architecture discovery. It is demonstrating that explicit memory produces understandable, evidence-grounded next-step ideas and that the system can turn approved ideas into compliant, reproducible experiments over a long run.
