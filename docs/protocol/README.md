# GigaEvo Experimental Protocol

**Version**: 1.0 (2026-03-04)
**Every experiment must complete all five phases in order. No exceptions.**

---

## Phase Overview

| # | File | Actor | Gate (before proceeding) |
|---|------|-------|--------------------------|
| 0 | [00_github.md](00_github.md) | Researcher | GitHub Issue created for idea; PR created at Phase 3 |
| 1 | [01_design.md](01_design.md) | `ml-research-methodologist` agent | Design approved by researcher |
| 2 | [02_review.md](02_review.md) | `reviewer-2-adversary` agent | Verdict = **APPROVED** |
| 3 | [03_preregistration.md](03_preregistration.md) | Researcher | Committed to git; branch + PR created |
| 4 | [04_launch.md](04_launch.md) | Researcher + Claude Code | All dry-run checks pass, preflight green |
| 5 | [05_results.md](05_results.md) | `ml-research-methodologist` agent | Written + PR merged before next experiment |

---

## How to Start a New Experiment

1. **Create a GitHub Issue** for the experiment idea (`gh issue create --label experiment-idea`)
2. Copy `experiments/_template/` to `experiments/<task>/<name>/`
3. Ask Claude Code to invoke `/agent:ml-research-methodologist` with the research question, prior results, and compute budget
4. Elena fills in `01_design.md` and ends with *"Ready for Reviewer-2's scrutiny"*
5. **Researcher reads `01_design.md` and approves the handoff** (or asks Elena to revise first)
6. Tell Claude Code to invoke `/agent:reviewer-2-adversary` — it will run the review
7. If verdict is NEEDS REVISION: tell Claude Code to send the concerns back to Elena, repeat from step 4
8. Fill in `03_plan.md`, **create branch `exp/<name>` and GitHub PR**, commit pre-registration
9. Implement code changes, then follow `04_launch.md` checklist
10. After final generation: ask Claude Code to invoke `/agent:ml-research-methodologist` for results analysis
11. **Merge PR and close tracking issue**

**Phase 1→2 handoff rule (Option A)**: Elena self-declares readiness; researcher approves;
Claude Code invokes Volkov. The researcher's approval is the gate — not automatic.

---

## Experiment Directory Layout

```
experiments/
  _template/               ← generic phase templates
  <task>/                  ← research project directory
    CONTEXT.md             ← task knowledge (benchmarks, chain, infra)
    tools/                 ← shared project tools
    <experiment-name>/     ← per-experiment directory
      01_design.md         ← scientific design (methodologist output)
      02_review.md         ← adversarial review (reviewer-2 output)
      03_plan.md           ← pre-registration (locked before code)
      04_launch.sh         ← launch script
      05_results.md        ← post-experiment analysis
      run_watchdog.py
      tools/               ← experiment-specific tools (optional)
      *.log
```

The `docs/protocol/` files are **templates and instructions**. The per-experiment files
under `experiments/<task>/<name>/` are the **filled-out instances**.

---

## What Does NOT Belong Here

Task-specific knowledge (benchmarks, infrastructure, chain topology, known bugs) lives in
`experiments/<task>/CONTEXT.md` — for HotpotQA: `experiments/hotpotqa/CONTEXT.md`.

Experiment-specific scripts live in `experiments/<task>/<name>/tools/`, not in `tools/`.
The `tools/` directory holds generic GigaEvo tools that work for any run.
Shared project tools (useful across experiments within the same task) live in
`experiments/<task>/tools/`.

---

## Amendment Protocol

When a pre-registered plan must change after registration:

1. Record the amendment in `experiments/<task>/<name>/03_plan.md` under a dedicated **Amendments** section
2. Include: what changed, why, relevant commit hash, impact classification:
   - **No confound** — change applied uniformly to all runs
   - **Confound introduced** — change applies differently across runs; document and assess
   - **Run invalidated** — affected run(s) must be excluded from analysis

---

## Agent Invocation Reference

Agents live in `.claude/agents/` and run on Opus 4.6 with persistent memory.
**Invoke by telling Claude Code** — it handles the actual `/agent:` call.

| Agent | Phase | Trigger | What to tell Claude Code |
|-------|-------|---------|--------------------------|
| `ml-research-methodologist` | 1 — Design | Researcher requests | "Invoke Elena with: [research question, prior results, compute budget]" |
| `reviewer-2-adversary` | 2 — Review | Researcher approves Elena's draft | "Send to Volkov for review" |
| `ml-research-methodologist` | 2 — Revision | Volkov returns NEEDS REVISION | "Send Volkov's concerns back to Elena" |
| `reviewer-2-adversary` | 2 — Re-review | Elena resubmits | "Ask Volkov to re-review" |
| `ml-research-methodologist` | 5 — Results | Final gen complete | "Invoke Elena for results analysis" |
| `research-methodology-expert` | Any | Protocol audit requested | "Ask Prof. Nakamura to review the protocol" |

**Elena's sign-off**: *"Ready for Reviewer-2's scrutiny."* — signals Phase 1 complete.
**Volkov's sign-off**: *"The science demands nothing less."* — always ends his reviews.
**Revision loop**: repeats until Volkov issues APPROVED. No cap on rounds.
