#!/usr/bin/env python3
"""Create immutable train/validation MDS views from disjoint source shards."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _index(source: dict[str, object], shards: list[dict[str, object]]) -> dict[str, object]:
    result = dict(source)
    result["shards"] = shards
    return result


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="dclm-baseline-1.0")
    parser.add_argument("--validation-shards", type=int, default=2)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_split = source_root / args.source_split
    source_index_path = source_split / "index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    shards = source_index.get("shards")
    if not isinstance(shards, list) or len(shards) <= args.validation_shards:
        parser.error("source must contain more shards than --validation-shards")
    if args.validation_shards < 1:
        parser.error("--validation-shards must be positive")

    if output_root.exists():
        manifest = output_root / "holdout-split-manifest.json"
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                existing.get("source_index_sha256") == _sha256(source_index_path)
                and existing.get("validation_shards") == args.validation_shards
            ):
                print(output_root)
                return 0
        parser.error(f"output already exists with a different contract: {output_root}")

    train_shards = shards[: -args.validation_shards]
    validation_shards = shards[-args.validation_shards :]
    output_root.mkdir(parents=True)
    for split_name, split_shards in (
        ("train", train_shards),
        ("validation", validation_shards),
    ):
        split_dir = output_root / split_name
        split_dir.mkdir()
        _write_json(split_dir / "index.json", _index(source_index, split_shards))
        for shard in split_shards:
            zip_data = shard.get("zip_data")
            if not isinstance(zip_data, dict) or not isinstance(
                zip_data.get("basename"), str
            ):
                raise ValueError("every source shard must name compressed zip_data")
            basename = zip_data["basename"]
            source = source_split / basename
            if not source.is_file():
                raise FileNotFoundError(source)
            (split_dir / basename).symlink_to(source)

    train_index = output_root / "train" / "index.json"
    validation_index = output_root / "validation" / "index.json"
    source_manifest_path = source_root / "corpus-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    # Import after constructing the views so this script also works when called
    # directly from a fresh checkout.
    from autoresearch.mds_provenance import (
        MDS_PROVENANCE_POLICY,
        mds_source_tree_sha256,
    )

    train_samples = sum(int(shard["samples"]) for shard in train_shards)
    sequence_length = int(source_manifest["sequence_length"])
    corpus_manifest = dict(source_manifest)
    corpus_manifest.update(
        {
            "samples": train_samples,
            "tokens": train_samples * sequence_length,
            "mds_index_sha256": _sha256(train_index),
            "mds_tree_sha256": mds_source_tree_sha256(output_root / "train"),
            "mds_provenance_policy": MDS_PROVENANCE_POLICY,
            "derived_from_corpus_manifest": str(source_manifest_path),
        }
    )
    _write_json(output_root / "corpus-manifest.json", corpus_manifest)
    _write_json(
        output_root / "holdout-split-manifest.json",
        {
            "version": 1,
            "source_index": str(source_index_path),
            "source_index_sha256": _sha256(source_index_path),
            "source_corpus_manifest": str(source_manifest_path),
            "source_corpus_manifest_sha256": _sha256(source_manifest_path),
            "validation_shards": args.validation_shards,
            "train_index_sha256": _sha256(train_index),
            "validation_index_sha256": _sha256(validation_index),
            "validation_tree_sha256": mds_source_tree_sha256(
                output_root / "validation"
            ),
            "train_samples": train_samples,
            "validation_samples": sum(
                int(shard["samples"]) for shard in validation_shards
            ),
            "overlapping_compressed_shards": [],
        },
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
