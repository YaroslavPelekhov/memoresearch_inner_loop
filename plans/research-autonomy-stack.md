# Research Autonomy Stack — Implementation Plan

**Branch**: `feat/research-autonomy-stack`
**Goal**: Close-loop research pipeline: ideas → design → implement → run → results → paper → next ideas. Human gates via Telegram (async). GPU scheduling automated.

---

## Problem Statement (from researcher interview 2026-04-09)

1. **Trust gap**: Agent designs experiments without understanding codebase → implements the wrong thing. No check that code matches design.
2. **Blocking gates**: 3 gates require terminal presence. Runs can't proceed overnight unattended.
3. **No ideas queue**: Research direction requires manual intervention. No persistent ranked backlog.
4. **Manual resource assignment**: Redis DBs and servers assigned by hand.
5. **Results → paper gap**: Closeout doesn't produce paper-ready output.

---

## SOTA Grounding (papers 2024-2026)

| Paper | Insight | Applied Here |
|---|---|---|
| AI Scientist v2 (Sakana AI, 2025) | Tree search, VLM figure feedback, agentic loops prune bad branches | implementation-aligner agent |
| R&D-Agent (Microsoft, 2024) | Decouple Research (ideation) from Development (implementation) | code-archaeologist → Elena split |
| ADAS (Liu et al., ICLR 2025) | Agent design as searchable code space; ideas need lineage edges | IDEAS.yaml with `builds_on`/`contradicts` |
| AIDE (Weco AI, 2025) | Solution tree with metric-feedback pruning; 4× Kaggle medal rate | implementation-aligner + smoke test as verifier |
| Agent Laboratory (AMD, 2025) | Role-based agents; PhD+Engineer+Professor separation | existing Elena/Volkov pattern; adds paper-section-writer |
| MemGPT/Letta | Three-tier memory: working / episodic / archival | formalize existing INDEX.md + PATTERNS.md + memory files |
| PaperOrchestra (Google, 2026) | Structured JSON → LaTeX, not LLM → PDF; outline → lit → plot → section agents | paper_data.json schema + paper-section-writer |
| LangGraph async HITL | Telegram buttons + persistent checkpoint state; async not blocking | tools/telegram_notify.py + gate pattern |
| Goal Drift (AAAI/ACM AIES, 2025) | Agents lose research goal after 600+ interactions; re-anchor each checkpoint | goal-anchor-check in checkpoint agent |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         IDEAS.yaml                              │
│  Ranked queue: hypothesis + task + rank + lineage graph         │
│  builds_on / contradicts / alternative_to edges                 │
│  Fed by: literature-scout, idea-generate, human, retrospective  │
└──────────────────┬──────────────────────────────────────────────┘
                   │ research-scheduler (autonomous pick)
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  experiment-design  [ENHANCED]                                  │
│  Step 2b: code-archaeologist → codebase_map.md                  │
│  Elena reads: literature_brief + codebase_map → designs         │
│  Volkov adversarial review → auto-resolve minors                │
│  Gate 1: Telegram notify (saves state) → researcher approves    │
└──────────────────┬──────────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  experiment-implement  [ENHANCED]                               │
│  Step 10b: implementation-aligner → design↔code check           │
│  Hard gate: MISALIGNED → fix gaps → re-check                    │
│  treatment-verifier → catches silent fallbacks                   │
│  Gate 2: Telegram notify → researcher confirms                  │
└──────────────────┬──────────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  experiment-launch + run-experiment  [ENHANCED]                 │
│  resource-manager auto-assigns free DBs + servers               │
│  Watchdog posts Telegram fitness updates (hourly)               │
│  Checkpoint cron: goal-anchor-check at each checkpoint          │
└──────────────────┬──────────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  experiment-closeout  [ENHANCED]                                │
│  experiment-paper-draft: results → paper_data.json → sections   │
│  Gate 3: Telegram notify → researcher signs off                 │
│  → idea-generate updates IDEAS.yaml with new proposals          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Three-Tier Memory Architecture (MemGPT pattern)

| Tier | Contents | Storage | Access pattern |
|---|---|---|---|
| **Working** | Current experiment state, active agent context | `experiment.yaml`, `MEMORY.md` | Always in context |
| **Episodic** | Recent experiment summaries, pattern consolidations | `experiments/INDEX.md`, `experiments/PATTERNS.md` | Loaded at session start |
| **Archival** | All past results, literature briefs, full designs | `experiments/<task>/<name>/`, `codebase_map.md` | Loaded on demand |

Goal drift prevention: at every checkpoint, checkpoint-analyst re-reads `01_design.md` hypothesis and scores trajectory alignment.

---

## Implementation Phases

### Phase 1 — Trust Gap (Critical, highest ROI)

**New files:**
- `.claude/agents/code-archaeologist.md` ← maps codebase before Elena designs
- `.claude/agents/implementation-aligner.md` ← verifies design↔code after implement

**Modified files:**
- `.claude/skills/experiment-design/SKILL.md` ← add Step 2b (code-archaeologist)
- `.claude/skills/experiment-implement/SKILL.md` ← add Step 10b (implementation-aligner)

### Phase 2 — Telegram Async Gates

**New files:**
- `tools/telegram_notify.py` ← send + async wait for reply
- `docs/setup/telegram-bot.md` ← setup instructions

**Modified files:**
- `.claude/skills/run-experiment/SKILL.md` ← replace blocking gates with Telegram async
- `.claude/agents/checkpoint-analyst.md` ← add goal-anchor-check step

### Phase 3 — Ideas Pool + Auto-scheduler

**New files:**
- `experiments/IDEAS.yaml` ← structured idea queue with lineage graph
- `.claude/skills/idea-generate/SKILL.md` ← generates ranked ideas post-retrospective
- `.claude/skills/research-scheduler/SKILL.md` ← autonomous idea → experiment

**Modified files:**
- `.claude/skills/experiment-retrospective/SKILL.md` ← call idea-generate at end

### Phase 4 — Resource Auto-detection

**New files:**
- `tools/resource_manager.py` ← queries servers + Redis DBs for available capacity

**Modified files:**
- `.claude/skills/experiment-implement/SKILL.md` ← Step 7 uses resource_manager

### Phase 5 — Paper Writing Pipeline

**New files:**
- `.claude/agents/paper-section-writer.md` ← PaperOrchestra-style section generation
- `.claude/skills/experiment-paper-draft/SKILL.md` ← orchestrates paper_data.json → sections

**Modified files:**
- `.claude/skills/experiment-closeout/SKILL.md` ← add paper-draft step

---

## Non-Goals

- Do NOT remove human gates — make them async, not absent
- Do NOT automate paper submission — researcher reviews first
- Do NOT auto-launch multiple experiments simultaneously (resource contention)
- Do NOT replace Elena with a tree-search agent now — too much architectural disruption; tree search is a future upgrade
