from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from gigaevo.programs.program import Program

REPO_CANDIDATE_KIND = "repo_snapshot"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class RepoCandidateManifest(BaseModel):
    """Small Redis-safe pointer to a Git-backed evolved repository snapshot.

    GigaEvo still stores a Program, but the candidate artifact is a Git commit.
    ``Program.code`` contains this manifest as JSON; Git stores the actual files.
    """

    kind: Literal["repo_snapshot"] = REPO_CANDIDATE_KIND
    schema_version: int = 1
    repo_path: str = Field(description="Path to the Git repository that owns commits")
    commit: str = Field(description="Candidate commit SHA")
    parent_commit: str | None = Field(
        default=None, description="Parent candidate commit SHA"
    )
    branch: str | None = Field(default=None, description="Branch containing commit")
    entrypoint: str | None = Field(
        default=None, description="Optional harness import path or command entrypoint"
    )
    mutation_agent: str | None = Field(
        default=None, description="Mutation backend that produced this commit"
    )
    mutation_session_log: str | None = Field(
        default=None, description="Path to stdout/stderr/session log for mutation"
    )
    changed_files: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    extra: dict[str, Any] = Field(default_factory=dict)

    def repo(self) -> Path:
        return Path(self.repo_path).expanduser().resolve()

    def to_program_code(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_program(
        cls, program: Program, *, default_repo_path: str | Path | None = None
    ) -> RepoCandidateManifest:
        """Load a manifest from Program.code, with legacy metadata fallback."""
        try:
            payload = json.loads(program.code)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Program.code is not a repo candidate manifest JSON object"
            ) from exc

        if not isinstance(payload, dict) or payload.get("kind") != REPO_CANDIDATE_KIND:
            raise ValueError(
                f"Program.code manifest kind must be {REPO_CANDIDATE_KIND!r}"
            )

        if default_repo_path is not None and not payload.get("repo_path"):
            payload["repo_path"] = str(default_repo_path)
        return cls.model_validate(payload)

    @classmethod
    def from_metadata(
        cls, program: Program, *, default_repo_path: str | Path
    ) -> RepoCandidateManifest:
        """Build a manifest from metadata keys for compatibility/ad hoc seeds."""
        commit = program.metadata.get("git_commit")
        if not commit:
            raise ValueError("Program metadata does not contain git_commit")
        return cls(
            repo_path=str(default_repo_path),
            commit=str(commit),
            parent_commit=(
                str(program.metadata["parent_commit"])
                if program.metadata.get("parent_commit")
                else None
            ),
            branch=(
                str(program.metadata["git_branch"])
                if program.metadata.get("git_branch")
                else None
            ),
            entrypoint=(
                str(program.metadata["repo_entrypoint"])
                if program.metadata.get("repo_entrypoint")
                else None
            ),
            mutation_agent=(
                str(program.metadata["mutation_agent"])
                if program.metadata.get("mutation_agent")
                else None
            ),
            changed_files=list(program.metadata.get("changed_files") or []),
        )
