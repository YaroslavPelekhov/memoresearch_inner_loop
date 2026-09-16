from __future__ import annotations

from pathlib import Path
import sys

from autoresearch.review import load_item, main, update_status, write_item


def test_review_status_round_trip(tmp_path: Path) -> None:
    review_path = tmp_path / "ideas" / "proposal.json"
    write_item(
        review_path,
        {"id": "idea-1", "kind": "idea", "status": "pending"},
    )

    approved = update_status(review_path, "approved", note="looks testable")

    assert approved["status"] == "approved"
    assert approved["reviewer_note"] == "looks testable"
    assert approved["reviewed_at"]
    assert load_item(review_path) == approved


def test_memory_redaction_keeps_source_card_immutable(
    tmp_path: Path, monkeypatch
) -> None:
    review_path = tmp_path / "memory" / "program-1.json"
    source_card = {"id": "program-1", "description": "raw evidence"}
    write_item(
        review_path,
        {
            "id": "program-1",
            "kind": "program",
            "status": "pending",
            "source_card": source_card,
            "field_overrides": {},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autoresearch.review",
            "--review-dir",
            str(tmp_path),
            "redact",
            "program-1",
            "--field",
            "description=reviewed takeaway",
        ],
    )

    assert main() == 0
    redacted = load_item(review_path)
    assert redacted["source_card"] == source_card
    assert redacted["field_overrides"] == {
        "description": "reviewed takeaway"
    }
