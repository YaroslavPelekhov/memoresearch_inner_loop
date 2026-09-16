"""Transparent on-disk storage for an idea campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autoresearch.ideas.models import (
    CampaignState,
    Idea,
    IdeaVersion,
    Implementation,
    Takeaway,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
) -> list[ModelT]:
    if not path.is_file():
        return []
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _upsert_jsonl[ModelT: BaseModel](
    path: Path,
    item: ModelT,
    *,
    key: str = "id",
) -> None:
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    item_value = item.model_dump(mode="json")
    identity = item_value[key]
    replaced = False
    for index, value in enumerate(existing):
        if value.get(key) == identity:
            existing[index] = item_value
            replaced = True
            break
    if not replaced:
        existing.append(item_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in existing),
        encoding="utf-8",
    )
    temporary.replace(path)


class CampaignStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ideas_path = self.root / "idea-catalog.jsonl"
        self.takeaways_path = self.root / "takeaway-bank.jsonl"
        self.implementations_path = self.root / "implementation-index.jsonl"
        self.state_path = self.root / "campaign-state.json"

    def load_state(self, campaign_id: str) -> CampaignState:
        if self.state_path.is_file():
            state = CampaignState.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
            if state.campaign_id != campaign_id:
                raise ValueError(
                    f"Campaign directory belongs to {state.campaign_id!r}, "
                    f"not {campaign_id!r}"
                )
            return state
        state = CampaignState(campaign_id=campaign_id)
        self.save_state(state)
        return state

    def save_state(self, state: CampaignState) -> None:
        _write_json(self.state_path, state.model_dump(mode="json"))

    def ideas(self) -> list[Idea]:
        return _read_jsonl(self.ideas_path, Idea)

    def takeaways(self) -> list[Takeaway]:
        return _read_jsonl(self.takeaways_path, Takeaway)

    def implementations(self) -> list[Implementation]:
        return _read_jsonl(self.implementations_path, Implementation)

    def save_idea(self, idea: Idea) -> None:
        _upsert_jsonl(self.ideas_path, idea)

    def save_takeaway(self, takeaway: Takeaway) -> None:
        _upsert_jsonl(self.takeaways_path, takeaway)

    def save_implementation(self, implementation: Implementation) -> None:
        _upsert_jsonl(self.implementations_path, implementation)

    def idea_version_root(
        self,
        idea_id: str,
        title: str,
        version: int,
    ) -> Path:
        slug = "".join(
            character.lower() if character.isalnum() else "-" for character in title
        )
        slug = "-".join(part for part in slug.split("-") if part)[:60]
        return self.root / "ideas" / f"{idea_id}-{slug}" / f"v{version:03d}"

    def save_version(self, root: Path, version: IdeaVersion) -> None:
        _write_json(root / "idea-version.json", version.model_dump(mode="json"))
