#!/usr/bin/env python3
"""Prepare a pinned, partition-diverse DCLM corpus for baseline calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import quote

import requests

from autoresearch.mds_provenance import MDS_PROVENANCE_POLICY, mds_source_tree_sha256

DATASET = "mlfoundations/dclm-baseline-1.0"
REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
SELECTION_SEED = 2048
PARTITION_COUNT = 32
# 87,715 complete global batches * 16 sequences * 2,048 tokens. The exact
# 20-tokens/parameter target is 2,874,216,960; batch rounding consumes 28,160
# additional tokens and the physical corpus must cover those too.
MIN_CONFIRMATION_TOKENS = 2_874_245_120
SPLIT = "dclm-baseline-1.0"


def _api_json(url: str) -> Any:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _partition_paths() -> list[str]:
    partitions = [
        f"global-shard_{global_id:02d}_of_10/local-shard_{local_id}_of_10"
        for global_id in range(1, 11)
        for local_id in range(10)
    ]
    return sorted(
        partitions,
        key=lambda path: sha256(f"{SELECTION_SEED}:partition:{path}".encode()).digest(),
    )[:PARTITION_COUNT]


def _select_files() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for partition in _partition_paths():
        encoded = quote(partition, safe="/")
        url = (
            f"https://huggingface.co/api/datasets/{DATASET}/tree/"
            f"{REVISION}/{encoded}?limit=1000&expand=false"
        )
        files = [
            item
            for item in _api_json(url)
            if item.get("type") == "file"
            and str(item.get("path", "")).endswith("_processed.jsonl.zst")
        ]
        if not files:
            raise RuntimeError(f"No processed shards found in {partition}")
        files.sort(key=lambda item: item["path"])
        digest = sha256(f"{SELECTION_SEED}:file:{partition}".encode()).digest()
        item = files[int.from_bytes(digest[:8], "big") % len(files)]
        lfs = item.get("lfs") or {}
        selected.append(
            {
                "path": item["path"],
                "bytes": int(item["size"]),
                "sha256": lfs.get("oid"),
            }
        )
    return selected


def _load_or_select_files(raw_root: Path) -> list[dict[str, Any]]:
    """Reuse a validated pinned manifest without requiring dataset API access."""

    manifest_path = raw_root / "source-manifest.json"
    if not manifest_path.is_file():
        return _select_files()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "dataset": DATASET,
        "revision": REVISION,
        "selection_seed": SELECTION_SEED,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Pinned source manifest has unexpected {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != PARTITION_COUNT:
        raise RuntimeError(
            f"Pinned source manifest must contain {PARTITION_COUNT} files"
        )
    expected_partitions = set(_partition_paths())
    observed_partitions: set[str] = set()
    validated: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("Pinned source manifest file records must be mappings")
        path = str(item.get("path", ""))
        partition = str(Path(path).parent)
        size = item.get("bytes")
        checksum = item.get("sha256")
        if partition not in expected_partitions:
            raise RuntimeError(f"Unexpected DCLM partition in manifest: {partition}")
        if partition in observed_partitions:
            raise RuntimeError(f"Duplicate DCLM partition in manifest: {partition}")
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Invalid source size for {path}: {size!r}")
        if not (
            isinstance(checksum, str)
            and len(checksum) == 64
            and all(character in "0123456789abcdef" for character in checksum)
        ):
            raise RuntimeError(f"Invalid source SHA-256 for {path}")
        observed_partitions.add(partition)
        validated.append(dict(item))
    if observed_partitions != expected_partitions:
        raise RuntimeError("Pinned source manifest does not cover expected partitions")
    print(f"Reusing pinned source manifest without API lookup: {manifest_path}")
    return validated


def _destination(raw_dir: Path, source_path: str) -> Path:
    return raw_dir / source_path.replace("/", "__")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(_file_sha256(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _download_one(raw_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    destination = _destination(raw_dir, item["path"])
    expected_hash = item.get("sha256")
    if destination.exists():
        if destination.stat().st_size == item["bytes"] and (
            expected_hash is None or _file_sha256(destination) == expected_hash
        ):
            return {**item, "local_file": destination.name}
        raise RuntimeError(f"Existing shard failed verification: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    url = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/"
        f"{REVISION}/{quote(item['path'], safe='/')}"
    )
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Downloaded shard has wrong size: {partial}")
    if expected_hash is not None and _file_sha256(partial) != expected_hash:
        raise RuntimeError(f"Downloaded shard has wrong SHA-256: {partial}")
    partial.replace(destination)
    return {**item, "local_file": destination.name}


def _write_manifest(raw_root: Path, files: list[dict[str, Any]]) -> Path:
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "selection_seed": SELECTION_SEED,
        "selection_strategy": (
            "SHA-256-ranked 32-of-100 global/local partitions, then one "
            "SHA-256-indexed processed shard per partition"
        ),
        "files": sorted(files, key=lambda item: item["path"]),
        "compressed_bytes": sum(item["bytes"] for item in files),
    }
    destination = raw_root / "source-manifest.json"
    partial = destination.with_suffix(".json.part")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.replace(destination)
    return destination


def _verify_raw_input_set(raw_dir: Path, files: list[dict[str, Any]]) -> None:
    expected = {str(item["local_file"]) for item in files}
    actual = {item.name for item in raw_dir.iterdir() if item.is_file()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            "Raw DCLM directory differs from the pinned manifest: "
            f"missing={missing}, extra={extra}"
        )


def _verify_tokenizer_boundary(tokenizer: Path, python: str) -> dict[str, Any]:
    """Require exactly one explicit EOS separator and no implicit specials."""

    script = """
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
empty_ids = tokenizer("", add_special_tokens=True)["input_ids"]
eos_ids = tokenizer(sys.argv[2], add_special_tokens=False)["input_ids"]
if empty_ids:
    raise RuntimeError(f"Tokenizer inserts implicit special tokens: {empty_ids}")
if eos_ids != [tokenizer.eos_token_id]:
    raise RuntimeError(
        f"Explicit EOS must encode to exactly the EOS token: "
        f"{eos_ids} != {[tokenizer.eos_token_id]}"
    )
print(json.dumps({
    "policy": "no_implicit_special_tokens_plus_one_explicit_eos",
    "implicit_empty_ids": empty_ids,
    "explicit_eos_ids": eos_ids,
    "eos_token_id": tokenizer.eos_token_id,
}))
"""
    completed = subprocess.run(
        [python, "-c", script, str(tokenizer), "<|endoftext|>"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _convert(repo_root: Path, raw_dir: Path, mds_root: Path, python: str) -> None:
    split_dir = mds_root / SPLIT
    tokenizer = repo_root.parent / "data" / "model-template-gpt-neox-140m"
    tokenizer_boundary = _verify_tokenizer_boundary(tokenizer, python)
    index_path = split_dir / "index.json"
    if not index_path.is_file():
        if split_dir.exists() and any(split_dir.iterdir()):
            raise RuntimeError(f"Incomplete MDS output requires cleanup: {split_dir}")
        split_dir.mkdir(parents=True, exist_ok=True)
        command = [
            python,
            "-c",
            (
                "from autoresearch.lmfoundry_compat import install; install(); "
                "import runpy; "
                "runpy.run_module('scripts.data_prep.convert_dataset_json', "
                "run_name='__main__')"
            ),
            "--path",
            str(raw_dir),
            "--out_root",
            str(split_dir),
            "--split",
            "train",
            "--concat_tokens",
            "2048",
            "--tokenizer",
            str(tokenizer),
            "--eos_text",
            "<|endoftext|>",
            "--compression",
            "zstd",
            "--tokens_dtype",
            "uint16",
        ]
        python_paths = [str(repo_root), str(repo_root / "vendor" / "llm-foundry")]
        if os.environ.get("PYTHONPATH"):
            python_paths.append(os.environ["PYTHONPATH"])
        subprocess.run(
            command,
            cwd=repo_root / "vendor" / "llm-foundry",
            check=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(python_paths)},
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    samples = sum(int(shard["samples"]) for shard in index["shards"])
    tokens = samples * 2048
    if tokens < MIN_CONFIRMATION_TOKENS:
        raise RuntimeError(
            f"Converted corpus has only {tokens:,} tokens; expected at least "
            f"{MIN_CONFIRMATION_TOKENS:,}"
        )
    corpus = {
        "source_manifest": str(raw_dir.parent / "source-manifest.json"),
        "source_manifest_sha256": _file_sha256(raw_dir.parent / "source-manifest.json"),
        "mds_index_sha256": _file_sha256(index_path),
        "mds_tree_sha256": mds_source_tree_sha256(split_dir, index),
        "mds_provenance_policy": MDS_PROVENANCE_POLICY,
        "sequence_length": 2048,
        "samples": samples,
        "tokens": tokens,
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": _tree_sha256(tokenizer),
        "tokenizer_boundary": tokenizer_boundary,
        "eos_text": "<|endoftext|>",
        "tokens_dtype": "uint16",
        "compression": "zstd",
        "converter_sha256": _file_sha256(
            repo_root
            / "vendor"
            / "llm-foundry"
            / "scripts"
            / "data_prep"
            / "convert_dataset_json.py"
        ),
        "compatibility_layer_sha256": _file_sha256(
            repo_root / "autoresearch" / "lmfoundry_compat.py"
        ),
    }
    manifest_path = mds_root / "corpus-manifest.json"
    partial = manifest_path.with_suffix(".json.part")
    partial.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    partial.replace(manifest_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = repo_root.parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--convert", action="store_true")
    parser.add_argument(
        "--training-python",
        default=os.environ.get("AUTORESEARCH_TRAINING_PYTHON", sys.executable),
    )
    args = parser.parse_args()

    raw_root = args.data_root / "dclm-raw-3b-v1"
    raw_dir = raw_root / "shards"
    mds_root = args.data_root / "dclm-mds-3b-v1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_or_select_files(raw_root)
    print(
        f"Downloading {len(selected)} pinned shards "
        f"({sum(item['bytes'] for item in selected) / 1e9:.2f} GB)"
    )
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, raw_dir, item): item for item in selected}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"[{len(completed)}/{len(selected)}] {result['local_file']}")
    _verify_raw_input_set(raw_dir, completed)
    manifest = _write_manifest(raw_root, completed)
    print(f"Verified source manifest: {manifest}")
    if args.convert:
        _convert(repo_root, raw_dir, mds_root, args.training_python)
        print(f"Verified MDS corpus: {mds_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
