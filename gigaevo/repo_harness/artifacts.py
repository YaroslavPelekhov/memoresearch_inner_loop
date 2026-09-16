from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import re
import shutil
from typing import Any

from loguru import logger


ARTIFACT_LIST_KEYS = ("artifacts", "artifact_paths")
ARTIFACT_PATH_KEYS = ("artifact_path",)


def _slug(text: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return (slug[:max_len] or "artifact").strip(".") or "artifact"


def _descriptor_from_path(
    path: str, *, inherited: dict[str, Any] | None = None
) -> dict[str, Any]:
    inherited = inherited or {}
    return {
        "name": inherited.get("name") or Path(path).name or "artifact",
        "kind": inherited.get("kind"),
        "role": inherited.get("role"),
        "description": inherited.get("description"),
        "path": path,
    }


def iter_artifact_descriptors(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield generic artifact descriptors embedded in a structured feedback object.

    Benchmarks can expose artifacts with either of these shapes:

    - ``{"artifacts": [{"name": "trial.log", "path": "/abs/trial.log"}]}``
    - ``{"artifact_paths": ["/abs/trial.log", "/abs/result.json"]}``
    - ``{"artifact_path": "/abs/summary.json"}``

    The scan is recursive so descriptors can live at top-level, per task, per
    example, or inside benchmark-specific result objects.
    """

    if isinstance(payload, dict):
        inherited = {
            key: payload.get(key)
            for key in ("role", "kind", "description")
            if payload.get(key) is not None
        }
        for key in ARTIFACT_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield _descriptor_from_path(item, inherited=inherited)
                    elif isinstance(item, dict):
                        item_path = item.get("path") or item.get("file")
                        if isinstance(item_path, str):
                            descriptor = dict(inherited)
                            descriptor.update(item)
                            descriptor["path"] = item_path
                            descriptor.setdefault("name", Path(item_path).name)
                            yield descriptor
                        paths = item.get("paths")
                        if isinstance(paths, list):
                            item_inherited = dict(inherited)
                            item_inherited.update(
                                {
                                    k: item.get(k)
                                    for k in ("name", "role", "kind", "description")
                                    if item.get(k) is not None
                                }
                            )
                            for path in paths:
                                if isinstance(path, str):
                                    yield _descriptor_from_path(
                                        path, inherited=item_inherited
                                    )

        for key in ARTIFACT_PATH_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                yield _descriptor_from_path(value, inherited=inherited)

        for value in payload.values():
            yield from iter_artifact_descriptors(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_artifact_descriptors(item)


def _resolve_artifact_path(path_text: str, base_dirs: Iterable[Path]) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    base_dir_list = list(base_dirs)
    for base_dir in base_dir_list:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    base_dir = base_dir_list[0] if base_dir_list else Path.cwd()
    return base_dir / path


def _unique_destination(
    root: Path, index: int, descriptor: dict[str, Any], source: Path
) -> Path:
    raw_name = str(descriptor.get("name") or source.name or "artifact")
    suffix = "".join(source.suffixes)
    name = _slug(raw_name)
    if suffix and not name.endswith(suffix):
        name = f"{name}{suffix}"
    return root / f"{index:03d}-{name}"


def _copy_file(
    source: Path, destination: Path, *, max_file_bytes: int
) -> dict[str, Any]:
    size = source.stat().st_size
    if size > max_file_bytes:
        return {
            "captured": False,
            "bytes": size,
            "error": f"file exceeds max_artifact_file_bytes={max_file_bytes}",
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"captured": True, "bytes": size}


def _copy_dir(
    source: Path,
    destination: Path,
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    current_total_bytes: int,
    current_file_count: int,
) -> dict[str, Any]:
    copied_files = 0
    copied_bytes = 0
    skipped_files = 0
    destination.mkdir(parents=True, exist_ok=True)

    for child in sorted(source.rglob("*")):
        if not child.is_file():
            continue
        if current_file_count + copied_files >= max_files:
            skipped_files += 1
            continue
        child_size = child.stat().st_size
        if child_size > max_file_bytes:
            skipped_files += 1
            continue
        if current_total_bytes + copied_bytes + child_size > max_total_bytes:
            skipped_files += 1
            continue
        relative = child.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)
        copied_files += 1
        copied_bytes += child_size

    return {
        "captured": copied_files > 0,
        "bytes": copied_bytes,
        "file_count": copied_files,
        "skipped_files": skipped_files,
    }


def capture_evaluation_artifacts(
    *,
    artifact_root: str | Path,
    structured_feedback: dict[str, Any] | None,
    base_dirs: Iterable[Path],
    stdout: str | None = None,
    stderr: str | None = None,
    max_files: int = 200,
    max_file_bytes: int = 50_000_000,
    max_total_bytes: int = 250_000_000,
) -> dict[str, Any]:
    """Copy benchmark-declared artifacts into a stable run directory.

    The returned manifest is intentionally generic and can be attached to
    ``Program.metadata``. Reflection and mutation agents can use its copied
    paths without depending on benchmark-specific job directory conventions.
    """

    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_dir_list = [Path(base).expanduser().resolve() for base in base_dirs]
    items: list[dict[str, Any]] = []
    total_bytes = 0
    file_count = 0

    def add_text_artifact(name: str, kind: str, content: str | None) -> None:
        nonlocal total_bytes, file_count
        if content is None:
            return
        if file_count >= max_files:
            return
        encoded = content.encode(errors="replace")
        if (
            len(encoded) > max_file_bytes
            or total_bytes + len(encoded) > max_total_bytes
        ):
            items.append(
                {
                    "name": name,
                    "kind": kind,
                    "captured": False,
                    "bytes": len(encoded),
                    "error": "benchmark stream exceeds artifact size limits",
                }
            )
            return
        path = root / name
        path.write_text(content, encoding="utf-8", errors="replace")
        total_bytes += len(encoded)
        file_count += 1
        items.append(
            {
                "name": name,
                "kind": kind,
                "captured": True,
                "captured_path": str(path),
                "bytes": len(encoded),
            }
        )

    add_text_artifact("benchmark_stdout.txt", "benchmark_stdout", stdout)
    add_text_artifact("benchmark_stderr.txt", "benchmark_stderr", stderr)

    seen_sources: set[str] = set()
    descriptors = list(iter_artifact_descriptors(structured_feedback or {}))
    for descriptor in descriptors:
        if file_count >= max_files or total_bytes >= max_total_bytes:
            break
        source_text = descriptor.get("path")
        if not isinstance(source_text, str) or not source_text:
            continue
        source = _resolve_artifact_path(source_text, base_dir_list)
        source_key = str(source)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        item: dict[str, Any] = {
            "name": descriptor.get("name") or source.name,
            "kind": descriptor.get("kind"),
            "role": descriptor.get("role"),
            "description": descriptor.get("description"),
            "source_path": str(source),
            "captured": False,
        }
        try:
            if not source.exists():
                item["error"] = "source path does not exist"
            else:
                destination = _unique_destination(
                    root, len(items) + 1, descriptor, source
                )
                if source.is_dir():
                    result = _copy_dir(
                        source,
                        destination,
                        max_files=max_files,
                        max_file_bytes=max_file_bytes,
                        max_total_bytes=max_total_bytes,
                        current_total_bytes=total_bytes,
                        current_file_count=file_count,
                    )
                    item.update(result)
                    copied_files = int(result.get("file_count") or 0)
                    file_count += copied_files
                    total_bytes += int(result.get("bytes") or 0)
                else:
                    source_size = source.stat().st_size
                    if total_bytes + source_size > max_total_bytes:
                        item.update(
                            {
                                "captured": False,
                                "bytes": source_size,
                                "error": (
                                    "file exceeds remaining "
                                    f"max_artifact_total_bytes={max_total_bytes}"
                                ),
                            }
                        )
                    else:
                        result = _copy_file(
                            source,
                            destination,
                            max_file_bytes=max_file_bytes,
                        )
                        item.update(result)
                        if result.get("captured"):
                            file_count += 1
                            total_bytes += int(result.get("bytes") or 0)
                if item.get("captured"):
                    item["captured_path"] = str(destination)
        except Exception as exc:
            logger.warning(
                "[repo_harness.artifacts] Failed to capture artifact {}: {}",
                source,
                exc,
            )
            item["error"] = f"{type(exc).__name__}: {exc}"
        items.append(item)

    manifest = {
        "schema_version": 1,
        "root": str(root),
        "items": items,
        "limits": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
        },
        "stats": {
            "captured_items": sum(1 for item in items if item.get("captured")),
            "total_items": len(items),
            "bytes": total_bytes,
            "files": file_count,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
