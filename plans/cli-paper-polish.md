# CLI Paper-Quality Polish Plan

## User Requests (Priority Order)

### P0 — Tables: Don't format fitness as %
- `tools/status.py`: percentage formatting driven by `metrics.yaml upper_bound == 1.0` — heilbron has `upper_bound: 1.0` which triggers `%` display for `actual_fitness` (e.g., `3.5%` instead of `0.03538`)
- `tools/trajectory.py`: hardcoded `f"{val * 100:.1f}%"` — always shows percentage
- Fix: respect a new `display_as_pct` field in metrics.yaml (default: infer from upper_bound), or add `--raw` flag
- Scope: `tools/status.py:346-378`, `tools/trajectory.py:185-189`, `gigaevo/cli/status.py`, `gigaevo/cli/trajectory.py`

### P1 — Adversarial: Disable cummax for D (Improver) runs
- `tools/utils.py:334-356`: frontier always computed as cummax/cummin
- For Improver populations, fitness is non-monotonic (depends on current Constructor quality)
- Add `--no-frontier` flag to `gigaevo plot comparison` and per-run frontier control
- Add ability to tag runs as "non-monotonic" (e.g., via `--no-frontier-for label1,label2`)
- Scope: `gigaevo/cli/plot_group.py`, `tools/comparison.py`

### P2 — Frontier annotation
- `tools/comparison.py:612-703` already has `annotate_frontier` logic
- Not exposed in `gigaevo/cli/plot_group.py`
- Add `--annotate-frontier` flag with `--max-annotations N`
- Scope: `gigaevo/cli/plot_group.py`

### P3 — Plot max(G, D) — best-overall across paired populations
- New feature: given paired runs (Pop_A + Pop_B), plot the element-wise max of their frontiers
- Needs: `--paired label_a:label_b` flag or automatic detection from experiment.yaml
- Scope: new logic in `tools/comparison.py` or `gigaevo/cli/plot_group.py`

## Paper-Quality Additions (from research)

### P4 — Publication styling defaults
- Larger fonts, thicker lines, proper axis labels
- `--paper` flag that sets: 300 DPI, larger fonts, colorblind-safe palette, grayscale-friendly line styles
- Scope: `tools/comparison.py`, `gigaevo/cli/plot_group.py`

### P5 — Box/violin plots for condition comparison
- New `gigaevo plot box` command
- Input: experiment results (actual_fitness per run per condition)
- Scope: new command in `gigaevo/cli/plot_group.py`

### P6 — Dual-panel arms race plots
- New `gigaevo plot arms-race` command
- Stacked panels: Constructor fitness (top) + Improver fitness (bottom), shared X-axis
- Scope: new command

## Implementation Order

1. P0 (fitness formatting) — most visible, easiest fix
2. P1 (disable cummax for D) — critical for adversarial paper
3. P2 (frontier annotation) — already exists, just wire up
4. P3 (max(G,D) plot) — new feature, medium complexity
5. P4 (paper styling) — polish pass
6. P5/P6 (new plot types) — if time permits
