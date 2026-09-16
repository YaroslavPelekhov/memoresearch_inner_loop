from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.mds_provenance import clear_derived_mds_cache, mds_source_tree_sha256


def _write_mds_fixture(split: Path) -> Path:
    split.mkdir(exist_ok=True)
    shard = split / "shard.00000.mds.zstd"
    shard.write_bytes(b"compressed immutable source")
    (split / "index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "shards": [
                    {
                        "format": "mds",
                        "compression": "zstd",
                        "column_names": ["dtype", "tokens"],
                        "column_encodings": ["int8", "bytes"],
                        "samples": 10,
                        "zip_data": {
                            "basename": shard.name,
                            "bytes": shard.stat().st_size,
                        },
                        "raw_data": {
                            "basename": "shard.00000.mds",
                            "bytes": 100,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return shard


def test_derived_raw_mds_cache_does_not_change_source_hash(tmp_path: Path) -> None:
    _write_mds_fixture(tmp_path)
    original = mds_source_tree_sha256(tmp_path)

    raw = tmp_path / "shard.00000.mds"
    raw.write_bytes(b"derived decompression cache")

    assert mds_source_tree_sha256(tmp_path) == original
    assert clear_derived_mds_cache(tmp_path) == {
        "removed_files": 1,
        "removed_bytes": len(b"derived decompression cache"),
    }
    assert not raw.exists()


def test_unindexed_compressed_shard_is_rejected(tmp_path: Path) -> None:
    _write_mds_fixture(tmp_path)
    (tmp_path / "shard.99999.mds.zstd").write_bytes(b"stale source")

    with pytest.raises(ValueError, match="extra=.*99999"):
        mds_source_tree_sha256(tmp_path)


def test_unindexed_raw_cache_is_rejected(tmp_path: Path) -> None:
    _write_mds_fixture(tmp_path)
    (tmp_path / "stale.mds").write_bytes(b"unknown cache")

    with pytest.raises(ValueError, match="Unexpected derived MDS"):
        clear_derived_mds_cache(tmp_path)
