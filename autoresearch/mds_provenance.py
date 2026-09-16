"""Stable provenance for compressed Mosaic Streaming training corpora."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

MDS_PROVENANCE_POLICY = "index-and-compressed-shards-sha256-v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mds_source_tree_sha256(split_dir: Path, index: dict[str, Any] | None = None) -> str:
    """Hash index-declared compressed sources, excluding derived raw caches."""

    index_path = split_dir / "index.json"
    if index is None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") != 2:
        raise ValueError(f"Unsupported MDS index version: {index.get('version')!r}")
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"MDS index has no shards: {index_path}")

    expected_compressed: set[str] = set()
    source_files = [index_path]
    for position, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"MDS shard {position} is not a mapping")
        if shard.get("format") != "mds" or shard.get("compression") != "zstd":
            raise ValueError(f"MDS shard {position} has an unexpected format")
        if shard.get("column_names") != ["dtype", "tokens"] or shard.get(
            "column_encodings"
        ) != ["int8", "bytes"]:
            raise ValueError(f"MDS shard {position} has unexpected token columns")
        if not isinstance(shard.get("samples"), int) or shard["samples"] <= 0:
            raise ValueError(f"MDS shard {position} has an invalid sample count")
        zip_data = shard.get("zip_data")
        if not isinstance(zip_data, dict):
            raise ValueError(f"MDS shard {position} has no compressed source")
        basename = zip_data.get("basename")
        expected_bytes = zip_data.get("bytes")
        if (
            not isinstance(basename, str)
            or Path(basename).name != basename
            or not basename.endswith(".mds.zstd")
        ):
            raise ValueError(f"MDS shard {position} has an invalid basename")
        if basename in expected_compressed:
            raise ValueError(f"MDS index repeats compressed shard {basename}")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"MDS shard {position} has an invalid byte count")
        shard_path = split_dir / basename
        if not shard_path.is_file():
            raise FileNotFoundError(f"MDS compressed shard is missing: {shard_path}")
        if shard_path.stat().st_size != expected_bytes:
            raise ValueError(f"MDS compressed shard size mismatch: {shard_path}")
        expected_compressed.add(basename)
        source_files.append(shard_path)

    actual_compressed = {item.name for item in split_dir.glob("*.mds.zstd")}
    if actual_compressed != expected_compressed:
        raise ValueError(
            "Compressed MDS files differ from index: "
            f"missing={sorted(expected_compressed - actual_compressed)}, "
            f"extra={sorted(actual_compressed - expected_compressed)}"
        )

    digest = sha256()
    for file_path in sorted(source_files, key=lambda item: item.name):
        digest.update(file_path.name.encode())
        digest.update(b"\0")
        digest.update(_file_sha256(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def clear_derived_mds_cache(
    split_dir: Path, index: dict[str, Any] | None = None
) -> dict[str, int]:
    """Remove only index-declared raw shards derived from verified zip data."""

    index_path = split_dir / "index.json"
    if index is None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"MDS index has no shards: {index_path}")
    expected_raw: set[str] = set()
    for position, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"MDS shard {position} is not a mapping")
        raw_data = shard.get("raw_data")
        if not isinstance(raw_data, dict):
            raise ValueError(f"MDS shard {position} has no raw-cache description")
        basename = raw_data.get("basename")
        if (
            not isinstance(basename, str)
            or Path(basename).name != basename
            or not basename.endswith(".mds")
        ):
            raise ValueError(f"MDS shard {position} has an invalid raw basename")
        if basename in expected_raw:
            raise ValueError(f"MDS index repeats raw shard {basename}")
        expected_raw.add(basename)

    actual_raw = {item.name for item in split_dir.glob("*.mds")}
    unexpected = actual_raw - expected_raw
    if unexpected:
        raise ValueError(f"Unexpected derived MDS files: {sorted(unexpected)}")
    removed_files = 0
    removed_bytes = 0
    for basename in sorted(actual_raw):
        path = split_dir / basename
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_files += 1
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}
