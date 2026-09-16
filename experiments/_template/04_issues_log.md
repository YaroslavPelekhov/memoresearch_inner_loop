# Issues Log

Track ALL errors, crashes, unexpected behavior, and manual interventions during this experiment.
This includes: run crashes, tool/script failures, watchdog issues, skill execution errors,
config mistakes, git problems, Redis issues, helper script bugs, and anything else that
did not execute as expected and required a manual fix or workaround.

This log is for post-experiment reflection — use it to identify bugs to fix and process improvements.

## Format

Each entry should include:
- **When**: timestamp or phase (e.g., "launch", "checkpoint #3", "gen 12")
- **What**: brief description of the issue
- **Category**: run crash | tool bug | config mistake | infra issue | skill bug | watchdog | git | other
- **Impact**: how it affected the experiment (data loss, wasted compute, delayed launch, etc.)
- **Root cause**: why it happened (if known)
- **Fix applied**: what was done to resolve it (including manual workarounds)
- **Systemic fix needed**: whether a code/process/tool change would prevent recurrence (YES/NO + description)

## Entry Types

### Events (auto-captured by lifecycle skills)

Brief entries auto-appended by `/experiment-launch`, `/experiment-restart`, `/experiment-checkpoint`, and `/experiment-diagnose`. One-line summary with structured metadata. Format:

```
### [EVENT <ISO-timestamp>] -- <one-line description>

- **When**: <ISO-timestamp>
- **What**: <one-line description>
- **Category**: launch | restart | checkpoint | watchdog | diagnose
- **Impact**: <brief impact or "automated capture">
```

### Issues (manual or escalated entries)

Detailed entries for things that went wrong and required intervention. Use the full format from above (When, What, Category, Impact, Root cause, Fix applied, Systemic fix needed).

```
### <timestamp> -- <description>

- **When**: <timestamp or phase>
- **What**: <brief description>
- **Category**: run crash | tool bug | config mistake | infra issue | skill bug | watchdog | git | other
- **Impact**: <how it affected the experiment>
- **Root cause**: <why it happened>
- **Fix applied**: <what was done>
- **Systemic fix needed**: YES/NO + description
```

---

<!-- Add entries below, newest first -->
