from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE = runpy.run_path(str(ROOT / "tools/prepare-dclm-corpus.py"))
JSON_CONVERTER = ROOT / "vendor/llm-foundry/scripts/data_prep/convert_dataset_json.py"


def _manifest() -> dict[str, object]:
    partitions = PREPARE["_partition_paths"]()
    return {
        "dataset": PREPARE["DATASET"],
        "revision": PREPARE["REVISION"],
        "selection_seed": PREPARE["SELECTION_SEED"],
        "files": [
            {
                "path": f"{partition}/shard_00000000_processed.jsonl.zst",
                "bytes": 123,
                "sha256": "a" * 64,
                "local_file": f"source-{index}.zst",
            }
            for index, partition in enumerate(partitions)
        ],
    }


def test_existing_pinned_manifest_avoids_api_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setitem(
        PREPARE,
        "_select_files",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected API selection")),
    )

    selected = PREPARE["_load_or_select_files"](tmp_path)

    assert selected == manifest["files"]


def test_existing_manifest_rejects_duplicate_partition(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["files"][1]["path"] = manifest["files"][0]["path"]
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Duplicate DCLM partition"):
        PREPARE["_load_or_select_files"](tmp_path)


def test_json_converter_does_not_force_duplicate_special_tokens() -> None:
    source = JSON_CONVERTER.read_text(encoding="utf-8")

    assert "add_bos_token=True" not in source
    assert "add_eos_token=True" not in source
    assert "AutoTokenizer.from_pretrained(args.tokenizer)" in source
    assert 'data_files = sorted(glob(f"{path}/*"))' in source


def test_raw_input_set_must_exactly_match_manifest(tmp_path: Path) -> None:
    expected = tmp_path / "selected.zst"
    expected.write_bytes(b"selected")
    files = [{"local_file": expected.name}]

    PREPARE["_verify_raw_input_set"](tmp_path, files)

    (tmp_path / "stale.zst").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="extra=.*stale.zst"):
        PREPARE["_verify_raw_input_set"](tmp_path, files)
