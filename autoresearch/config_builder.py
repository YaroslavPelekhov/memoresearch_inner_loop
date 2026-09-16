"""Build a normal LLM Foundry config from a frozen base and candidate overlay."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ALLOWED_CANDIDATE_KEYS = frozenset(
    {
        "decoder_layer_module",
        "decoder_layer_type",
        "decoder_layer_kwargs",
    }
)


def _load_mapping(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return value


def candidate_overrides(candidate: Mapping[str, Any]) -> dict[str, Any]:
    try:
        overrides = candidate["model"]["config_overrides"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Candidate config must contain model.config_overrides"
        ) from exc
    if not isinstance(overrides, Mapping):
        raise ValueError("model.config_overrides must be a mapping")
    extra = set(overrides) - ALLOWED_CANDIDATE_KEYS
    if extra:
        raise ValueError(
            "Candidate config changes immutable keys: " + ", ".join(sorted(extra))
        )
    if set(overrides) != ALLOWED_CANDIDATE_KEYS:
        missing = ALLOWED_CANDIDATE_KEYS - set(overrides)
        raise ValueError(
            "Candidate config is missing required keys: " + ", ".join(sorted(missing))
        )
    if not isinstance(overrides["decoder_layer_kwargs"], Mapping):
        raise ValueError("decoder_layer_kwargs must be a mapping")
    return dict(overrides)


def build_effective_config(
    base_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, Any]:
    """Return a base config with only the decoder plugin subtree replaced."""

    base = _load_mapping(Path(base_path))
    candidate = _load_mapping(Path(candidate_path))
    try:
        model_overrides = base["model"]["config_overrides"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Base config must contain model.config_overrides") from exc
    if not isinstance(model_overrides, dict):
        raise ValueError("Base model.config_overrides must be a mapping")
    model_overrides.update(candidate_overrides(candidate))
    return base


def write_effective_config(
    base_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    effective = build_effective_config(base_path, candidate_path)
    with output.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(effective, config_file, sort_keys=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(write_effective_config(args.base, args.candidate, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
