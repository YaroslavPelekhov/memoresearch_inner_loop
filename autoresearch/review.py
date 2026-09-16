"""Durable file-backed review queue for ideas and memory items."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

FINAL_STATUSES = frozenset({"approved", "rejected", "redacted"})
VALID_STATUSES = FINAL_STATUSES | {"pending"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_item(item_path: str | Path) -> dict[str, Any]:
    path = Path(item_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Review item must be a JSON object: {path}")
    return value


def write_item(item_path: str | Path, item: dict[str, Any]) -> Path:
    """Atomically write one review item."""

    path = Path(item_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(item, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def update_status(
    item_path: str | Path,
    status: str,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid review status: {status}")
    item = load_item(item_path)
    item["status"] = status
    item["reviewed_at"] = _now() if status in FINAL_STATUSES else None
    if note is not None:
        item["reviewer_note"] = note
    write_item(item_path, item)
    return item


def find_item(review_dir: str | Path, item_id: str) -> Path:
    matches: list[Path] = []
    for candidate in Path(review_dir).rglob("*.json"):
        try:
            if load_item(candidate).get("id") == item_id:
                matches.append(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not matches:
        raise FileNotFoundError(f"No review item with id {item_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple review items have id {item_id!r}: {matches}")
    return matches[0]


def _items(review_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for item_path in review_dir.rglob("*.json"):
        if item_path.name == "active_idea.json":
            continue
        try:
            item = load_item(item_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if "id" in item and "status" in item:
            items.append((item_path, item))
    return sorted(items, key=lambda value: str(value[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=Path("review"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("id")
    for command in ("approve", "reject"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("id")
        command_parser.add_argument("--note")
    redact_parser = subparsers.add_parser("redact")
    redact_parser.add_argument("id")
    redact_parser.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Replace a human-readable top-level field; may be repeated.",
    )
    redact_parser.add_argument("--note")
    args = parser.parse_args()

    if args.command == "list":
        for item_path, item in _items(args.review_dir):
            print(
                f"{item.get('id')}\t{item.get('kind')}\t{item.get('status')}\t"
                f"{item_path}"
            )
        return 0

    item_path = find_item(args.review_dir, args.id)
    if args.command == "show":
        print(json.dumps(load_item(item_path), indent=2, sort_keys=True))
        return 0

    if args.command == "redact":
        item = load_item(item_path)
        protected = {
            "id",
            "kind",
            "status",
            "generation",
            "attempt",
            "available_memory_ids",
            "original_proposal",
            "source_card",
            "source_hash",
        }
        memory_fields = {"description", "keywords", "strategy", "task_description"}
        for assignment in args.field:
            if "=" not in assignment:
                parser.error(f"Invalid --field value: {assignment!r}")
            key, raw_value = assignment.split("=", 1)
            if key in protected:
                parser.error(f"Review field {key!r} cannot be replaced")
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            if item.get("kind") in {"memory", "program"}:
                if key not in memory_fields:
                    parser.error(
                        "Memory redaction may replace only: "
                        + ", ".join(sorted(memory_fields))
                    )
                overrides = item.setdefault("field_overrides", {})
                if not isinstance(overrides, dict):
                    parser.error("Memory item has invalid field_overrides")
                overrides[key] = value
            else:
                item[key] = value
        item["status"] = "redacted"
        item["reviewed_at"] = _now()
        item["reviewer_note"] = args.note
        write_item(item_path, item)
        print(json.dumps(item, indent=2, sort_keys=True))
        return 0

    status = "approved" if args.command == "approve" else "rejected"
    item = update_status(item_path, status, note=args.note)
    print(json.dumps(item, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
