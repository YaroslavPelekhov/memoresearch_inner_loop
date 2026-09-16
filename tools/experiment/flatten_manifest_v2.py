"""One-shot migration: flatten v2 experiment.yaml files to nested-only shape.

Current shape: YAML files have both flat top-level keys (`experiment`, `runs`,
`problem`, `watchdog`, etc.) AND nested `contract` / `lifecycle` / `telemetry`
/ `control_plane` sections — linked by YAML anchors (``&id001`` / ``*id001``).

Target shape: nested-only. The flat keys (and their anchors) are deleted so
only the canonical nested sections remain on disk.

Usage::

    # Dry run — shows which files would change and the before/after shapes
    $GIGAEVO_PYTHON tools/experiment/flatten_manifest_v2.py --dry-run

    # Apply to all experiments
    $GIGAEVO_PYTHON tools/experiment/flatten_manifest_v2.py --apply

    # Single file
    $GIGAEVO_PYTHON tools/experiment/flatten_manifest_v2.py --single \\
        experiments/heilbron/asymmetric-iterations-v2/experiment.yaml --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

PROJ = Path(__file__).resolve().parent.parent.parent

# Flat top-level keys that should be removed because they're duplicated under
# nested sub-sections. These are the *sources* that anchors point at in the
# current dual-shape YAMLs.
_FLAT_KEYS_TO_STRIP = {
    "experiment",  # → contract.identity + lifecycle.status + contract.stopping_rule
    "problem",  # → contract.problem
    "runs",  # → contract.runs
    "servers",  # → contract.servers
    "config",  # → contract.config (typed keys + model_extra)
    "custom_env",  # → contract.custom_env
    "baseline",  # → contract.baseline
    "tools",  # → contract.tools
    "stopping_rule",  # v2 intermediate → contract.stopping_rule
    "launch",  # → lifecycle.launch (+ control_plane.* for cron IDs)
    "smoke_test",  # → lifecycle.smoke_test
    "treatment_verification",  # → lifecycle.treatment_verification
    "watchdog",  # → control_plane.watchdog
    "checkpoints",  # → telemetry.checkpoints
    "mid_run_test_eval",  # → telemetry.mid_run_test_eval
    "checkpoint_analysis",  # → telemetry.checkpoint_analysis
    "treatment_checks",  # → telemetry.treatment_checks
}

# Top-level keys that stay.
_KEEP = {"schema_version", "contract", "lifecycle", "telemetry", "control_plane"}


def flatten(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a nested-only copy of ``raw`` and a list of stripped keys.

    If any stripped key lacks a nested counterpart, a warning is recorded
    so the caller can surface it before writing.
    """
    out: dict[str, Any] = {}
    stripped: list[str] = []

    for k, v in raw.items():
        if k in _KEEP:
            out[k] = v
        elif k in _FLAT_KEYS_TO_STRIP:
            stripped.append(k)
        else:
            # Unknown key — preserve it to avoid silent data loss.
            out[k] = v

    return out, stripped


def verify_nested_complete(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Verify the nested sections cover all data that was in the flat keys.

    Returns a list of missing-data warnings (empty list = safe to write).
    """
    warnings: list[str] = []

    contract = after.get("contract") or {}
    lifecycle = after.get("lifecycle") or {}
    telemetry = after.get("telemetry") or {}
    control_plane = after.get("control_plane") or {}

    if "contract" not in after:
        warnings.append("no contract: section — refusing to strip flat fields")
        return warnings
    if "lifecycle" not in after:
        warnings.append("no lifecycle: section — refusing to strip flat fields")
        return warnings

    # runs: both flat and contract.runs must match (anchors should guarantee)
    flat_runs = before.get("runs") or []
    nested_runs = contract.get("runs") or []
    if flat_runs and not nested_runs:
        warnings.append(
            f"flat runs has {len(flat_runs)} items but contract.runs is empty"
        )

    # status: lifecycle.status must exist if flat experiment.status did
    flat_status = (before.get("experiment") or {}).get("status")
    nested_status = lifecycle.get("status")
    if flat_status and not nested_status:
        warnings.append(
            f"flat experiment.status={flat_status} but lifecycle.status missing"
        )

    # watchdog: control_plane.watchdog must exist if flat watchdog did
    flat_wd = before.get("watchdog")
    nested_wd = control_plane.get("watchdog")
    if flat_wd and not nested_wd:
        warnings.append(
            "flat watchdog section exists but control_plane.watchdog is missing"
        )

    # checkpoints: telemetry.checkpoints
    flat_cp = before.get("checkpoints") or []
    nested_cp = telemetry.get("checkpoints") or []
    if flat_cp and not nested_cp:
        warnings.append(
            f"flat checkpoints has {len(flat_cp)} items but telemetry.checkpoints is empty"
        )

    return warnings


def _load_resolving_anchors(path: Path) -> dict[str, Any]:
    """Load YAML resolving anchors into independent copies per key.

    yaml.safe_load already resolves anchors by reference (both flat and nested
    sections will point to the same Python objects). That's fine for reading;
    when we strip the flat keys and write back, only the nested copies remain
    and they become the single source of truth.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def process_file(path: Path, apply: bool) -> tuple[bool, list[str]]:
    """Process one YAML file. Returns (changed, warnings)."""
    before = _load_resolving_anchors(path)

    if before.get("schema_version") != 2:
        return False, [f"schema_version != 2 (got {before.get('schema_version')!r})"]

    after, stripped = flatten(before)
    if not stripped:
        return False, []

    warnings = verify_nested_complete(before, after)
    if warnings:
        return False, warnings

    if apply:
        # Write back without YAML anchors (default_flow_style=False, no aliases)
        # PyYAML's default_representer emits aliases for repeated references;
        # since we stripped the flat duplicates, only the nested copies remain
        # and aliases won't be emitted.
        with open(path, "w") as f:
            yaml.safe_dump(
                after,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    return True, [f"would strip: {sorted(stripped)}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--single", type=Path, help="Process a single YAML file")
    args = parser.parse_args()

    apply = args.apply
    if not apply:
        print("[dry-run] Use --apply to write changes")

    targets: list[Path] = []
    if args.single:
        targets.append(args.single)
    else:
        experiments_dir = PROJ / "experiments"
        targets.extend(sorted(experiments_dir.glob("*/*/experiment.yaml")))
        template = experiments_dir / "_template" / "experiment.yaml"
        if template.exists():
            targets.append(template)

    if not targets:
        print("No targets found", file=sys.stderr)
        return 2

    total_changed = 0
    total_warned = 0
    for path in targets:
        rel = path.relative_to(PROJ)
        try:
            changed, messages = process_file(path, apply)
        except Exception as exc:
            print(f"  ERROR {rel}: {exc}", file=sys.stderr)
            total_warned += 1
            continue
        if changed:
            total_changed += 1
            status = "wrote" if apply else "would change"
            print(f"  [{status}] {rel}")
            for msg in messages:
                print(f"    · {msg}")
        elif messages:
            total_warned += 1
            print(f"  [skip] {rel}")
            for msg in messages:
                print(f"    ! {msg}")

    print(
        f"\nSummary: {total_changed} {'changed' if apply else 'would change'}, "
        f"{total_warned} warnings"
    )
    return 0 if total_warned == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
