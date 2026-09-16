# Experiment Skills Redesign — Research Synthesis

> Date: 2026-04-07
> Sources: Claude Code skills docs, SOTA auto-research papers, GigaEvo skills inventory

---

## 1. Current State (Inventory)

### Skills Architecture (18 total)

| Category | Skills | Pattern |
|---|---|---|
| Experiment lifecycle | 8 (design → implement → launch → checkpoint → diagnose → restart → closeout + orchestrator) | Linear state machine via `experiment.yaml` |
| GitNexus | 6 (exploring, debugging, impact, refactoring, guide, cli) | Read-only MCP tool wrappers |
| Testing/PM | 3 (run-tests, auto-optimize-loop, project-pm) | Standalone utilities |
| Explain | 1 (explain-code) | Reference |

**Agents (8):** Elena (ml-research-methodologist), Volkov (reviewer-2-adversary), systems-architect, chaos-hacker, checkpoint-analyst, treatment-verifier, anomaly-detector, project-pm.

### What works well
- **Status machine** (`experiment.yaml`) as single source of truth — clean sequencing
- **Human gates** at design / launch / results — prevents premature automation
- **Distributed monitoring** (watchdog + anomaly-detector cron + checkpoint cron) — self-healing
- **Persistent agent memory** (Elena, Volkov) — reduces re-learning across experiments
- **Opaque label blinding** in checkpoint-analyst — statistical hygiene

### Structural weaknesses
1. **No `context: fork`** on any skill — heavy exploration (diagnose, checkpoint analysis) pollutes main conversation context
2. **No `skills` preloading** for agents — Elena/Volkov don't receive current CONTEXT.md, experiment state at invocation time
3. **No dynamic context injection** (`!`cmd``) — status tables, gen counts, log snippets embedded by copy-paste not preprocessing
4. **Monolithic skill bodies** — `experiment-implement` and `experiment-closeout` are 500+ line single files; no supporting file structure
5. **`run-experiment` orchestrator** always runs in main context despite just reading manifest and dispatching

---

## 2. SOTA Landscape

### Architecturally Closest to GigaEvo

**AlphaEvolve (Google DeepMind, June 2025)**
- MAP-Elites evolutionary database + dual LLM ensemble (Gemini Flash for breadth/throughput, Gemini Pro for depth/quality)
- Beat Strassen (1969) on 4×4 complex matrix multiplication
- Directly validates GigaEvo's core architecture. Key delta: **dual-model ensemble** for mutation

**FunSearch (DeepMind, Nature 2023)**
- Showing top-K elites in mutation context is the core operator — GigaEvo already does this ✓
- Validates the "best programs as context" mutation strategy

**AI Scientist v2 (Sakana AI, April 2025)**
- **Progressive agentic tree search** over experiment paths — branches on promising directions
- One paper passed ICLR workshop peer review (first fully AI-generated)
- Tree search ≈ MAP-Elites: branching on diverse solution paths rather than flat mutation

**TextGrad (Stanford, 2024 — published in Nature)**
- **Textual gradients**: LLM provides natural language critiques that propagate upstream through compound AI systems
- Formalizes exactly what `heilbron-prover-v2` is: Improver critique flows into Constructor mutation prompt = LLM-native backpropagation
- Theoretical grounding for the GAN-like adversarial setup

**AI Co-Scientist (Google, February 2025)**
- Multi-agent: generate → debate → evolve with **tournament evolution** for hypothesis quality
- Tournament ≈ MAP-Elites selection pressure
- Asynchronous task execution framework for parallel hypothesis exploration

**AgentRxiv (2025)**
- Agents without cumulative knowledge sharing plateau at 73.5%; with it, reach 78.2% (MATH-500)
- GigaEvo's lineage traces + archive ARE this mechanism — but it's not being fully exploited in mutation prompts

### SOTA Pattern → GigaEvo Gap Analysis

| SOTA Pattern | Paper | GigaEvo Status | Gap |
|---|---|---|---|
| Elite context in mutation prompt | FunSearch, AlphaEvolve, OPRO | ✅ Implemented | — |
| MAP-Elites for quality+diversity | AlphaEvolve, ShinkaEvolve | ✅ Core mechanism | — |
| Dual-model ensemble (fast+slow) | AlphaEvolve (Flash+Pro) | ❌ Missing | Single LLM only |
| Textual gradients (adversarial critique→mutation) | TextGrad | 🔶 Partially (heilbron-prover-v2 design) | Not formalized |
| Tournament evolution | AI Co-Scientist | ❌ Missing | MAP-Elites is fitness-proportional, not tournament |
| Tree search over experiment paths | AI Scientist v2 | ❌ Missing | Flat mutation loop |
| Cumulative lineage in mutation context | AgentRxiv | 🔶 Lineage tools exist, not used in prompts | Lineage not fed to mutation LLM |
| Zero-cost proxy fitness | RZ-NAS | ❌ Missing | Full eval every time |
| Island-based diversity | CodeEvolve | ❌ Missing | Single MAP-Elites archive |

---

## 3. Claude Code Skills — Key Capabilities (Underused)

### A. `context: fork` — Isolated Subagents
Skills with `context: fork` run in a forked subagent context with independent window:
```yaml
context: fork
agent: Explore   # Read-only (Haiku), fast, cheap
```
Critical for expensive exploration that shouldn't pollute the main session.

### B. `skills` Field in Agent Definitions
Agents can preload skill content at startup:
```yaml
# .claude/agents/checkpoint-analyst.md
---
skills:
  - protocol-requirements
  - experiment-context-hover
---
```
This gives agents domain knowledge without copy-paste in prompts.

### C. Dynamic Context Injection (`!`cmd``)
Shell commands run **before** Claude sees the prompt:
```markdown
Current status: !`PYTHONPATH=. $GIGAEVO_PYTHON tools/status.py --experiment $ARGUMENTS`
Last checkpoint: !`tail -20 experiments/$ARGUMENTS/04_issues_log.md`
```
Claude receives the output, not the command — live data at zero manual effort.

### D. Supporting Files
Skills can reference external `.md`, `.sh`, `.py` files:
```
.claude/skills/experiment-implement/
├── SKILL.md          # Orchestration only (~150 lines)
├── references/
│   ├── treatment-verification.md
│   ├── config-patterns.md
│   └── failure-modes.md
└── scripts/
    └── smoke-test.sh
```

---

## 4. Redesign Recommendations

### Priority 1: Context Isolation (High Impact, Low Effort)

Add `context: fork` to read-heavy skills:

```yaml
# experiment-diagnose: pure exploration, no side effects
---
name: experiment-diagnose
context: fork
agent: Explore
allowed-tools: "Read Grep Bash(redis-cli *) Bash(PYTHONPATH=*)"
---
```

```yaml
# checkpoint-analyst invocation: isolated per-run analysis
context: fork
agent: general-purpose
```

### Priority 2: Dynamic Context Injection (High Impact, Medium Effort)

Replace manual copy-paste in checkpoint/diagnose with preprocessing:

```markdown
## Current Experiment State
!`PYTHONPATH=$PROJ $GIGAEVO_PYTHON tools/status.py --experiment $ARGUMENTS 2>/dev/null`

## Recent Log Errors
!`tail -50 experiments/$ARGUMENTS/run_*.log 2>/dev/null | grep -E "ERROR|CRITICAL" | tail -20`

## Last Checkpoint
!`tail -30 experiments/$ARGUMENTS/04_issues_log.md 2>/dev/null`
```

### Priority 3: Agent Knowledge Injection (Medium Impact, Low Effort)

Add `skills` field to key agents so they receive live context:

```yaml
# .claude/agents/checkpoint-analyst.md
---
name: checkpoint-analyst
skills:
  - experiment-context-hover   # CONTEXT.md as skill
  - protocol-requirements      # Stopping rules, gates
---
```

Create `experiment-context-hover` and `experiment-context-hotpotqa` as `user-invocable: false` skills that embed CONTEXT.md content.

### Priority 4: Skill Decomposition (Medium Impact, High Effort)

Split `experiment-implement` (most complex skill, ~600 lines) into:
- `experiment-implement` — orchestrator only (~100 lines)
- Supporting files: `references/config-patterns.md`, `references/treatment-guide.md`, `scripts/`

### Priority 5: SOTA Research Gaps (High Impact, Research Effort)

Three gaps worth addressing as experiments:

**A. Dual-model ensemble mutation (AlphaEvolve pattern)**
- Fast model (Qwen-235B or Gemini Flash) for high-throughput breadth mutations
- Slow model (Claude Opus / Gemini Pro) for quality "depth" mutations (e.g., 1 in 5)
- Expected: 3-5x throughput increase with maintained quality

**B. Lineage-aware mutation context**
- Feed ancestor chain (not just current program) into mutation prompt
- AgentRxiv shows cumulative knowledge = +4.7pp
- `tools/lineage.py` already extracts this; just needs to be injected into mutation operator context

**C. TextGrad for adversarial feedback**
- Formalize heilbron-prover-v2 as TextGrad: Improver critique as "textual gradient" upstream to Constructor mutation
- Structured critique format: {worst_triangle, moved_points, strategy_summary}
- Expected: better-targeted mutations vs. score-only feedback

---

## 5. Proposed Revised Skill Architecture

```
Tier 1: Orchestration (1 skill)
  run-experiment       → routes by experiment.yaml status

Tier 2: Phase Skills (5 skills, context: fork where applicable)
  experiment-design    → agent: general-purpose (Elena + Volkov)
  experiment-implement → agent: general-purpose
  experiment-launch    → agent: general-purpose
  experiment-checkpoint→ agent: general-purpose, preloads experiment-context-{task}
  experiment-closeout  → agent: general-purpose (Elena)

Tier 3: Diagnostic Skills (3 skills, context: fork, read-only)
  experiment-diagnose  → agent: Explore (read-only, cheap)
  experiment-restart   → agent: general-purpose
  anomaly-detector     → (stays as agent, invoked by cron)

Tier 4: Knowledge Skills (user-invocable: false, no side effects)
  experiment-context-hover     → CONTEXT.md + infrastructure as skill
  experiment-context-hotpotqa  → CONTEXT.md as skill
  protocol-requirements        → docs/protocol/ summary
  safety-gates                 → preflight check reference

Tier 5: Utilities (unchanged)
  run-tests, auto-optimize-loop, project-pm, explain-code
  gitnexus-* (6 skills)
```

---

## 6. Quick Wins (Can Do Today)

1. Add `context: fork` to `experiment-diagnose` — 2-line change
2. Add dynamic status injection to `experiment-checkpoint` Step 1 — replace manual status paste
3. Create `experiment-context-hover.md` as `user-invocable: false` skill from CONTEXT.md
4. Add `skills: [experiment-context-hover]` to `checkpoint-analyst.md`
5. Add `argument-hint` to skills missing it (experiment-restart, experiment-diagnose)

---

## Sources

- Claude Code Skills documentation (fetched live, April 2026)
- AlphaEvolve — arXiv 2506.13131
- AI Scientist v2 — arXiv 2504.08066
- TextGrad — arXiv 2406.07496 (Nature)
- AI Co-Scientist — arXiv 2502.18864
- AgentRxiv — arXiv 2503.18102
- FunSearch — Nature 2023
- GigaEvo skills inventory — `.claude/skills/` + `.claude/agents/`
