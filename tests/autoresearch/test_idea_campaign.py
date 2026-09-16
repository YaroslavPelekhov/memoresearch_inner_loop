from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from autoresearch.ideas.campaign import IdeaCampaign
from autoresearch.ideas.models import (
    Idea,
    IdeaVersion,
    Implementation,
    ParentSelection,
    ResearchTask,
)


class FakeAgent:
    def __init__(self) -> None:
        self.stages: list[str] = []

    async def call(self, *, stage: str, system: str, payload: dict[str, Any]) -> Any:
        self.stages.append(stage)
        if stage == "idea_generation":
            base = {
                "hypothesis": "A testable GDN change improves the fixed task.",
                "mechanism": "It changes recurrent state control.",
                "implementation_direction": "Add one bounded GDN mechanism.",
                "success_criteria": ["valid implementation", "fitness improves"],
                "risks": ["optimization instability"],
                "source_idea_ids": [],
                "takeaway_ids": [],
            }
            return {
                "proposals": [
                    {**base, "title": "First proposal"},
                    {**base, "title": "Selected proposal"},
                ]
            }
        if stage == "idea_filter":
            return {"selected_idea_id": "idea-0002", "reason": "Clear mechanism."}
        if stage == "implementation_parent_selection":
            return {"implementation_id": "canonical", "reason": "Clean attribution."}
        if stage == "idea_takeaway_synthesis":
            return {
                "takeaways": [
                    {
                        "kind": "uncertainty",
                        "claim": "The integration path works but needs a longer run.",
                        "scope_and_conditions": "Fake benchmark only.",
                        "confidence": "low",
                        "supporting_implementation_ids": ["impl-test"],
                        "contradicting_implementation_ids": [],
                        "recommended_next_action": "Run the real benchmark.",
                    }
                ]
            }
        raise AssertionError(stage)


class FakeCampaign(IdeaCampaign):
    async def _create_seed(self, **_kwargs: Any) -> str:
        return self.task.canonical_ref

    def _run_evolution(self, **_kwargs: Any) -> None:
        return None

    def _load_evidence(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "implementation_id": "impl-test",
                "commit": "abc123",
                "fitness": 0.1,
                "is_valid": True,
                "verdict": "improved",
            }
        ]


class FailingCampaign(FakeCampaign):
    def _load_evidence(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_complete_round_has_visible_idea_to_takeaway_lineage(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "gdn.py").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")

    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref=commit,
        proposal_count=2,
        evolution_generations=1,
    )
    agent = FakeAgent()
    campaign = FakeCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=agent,
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
    )

    await campaign.run(1)

    assert agent.stages == [
        "idea_generation",
        "idea_filter",
        "implementation_parent_selection",
        "idea_takeaway_synthesis",
    ]
    ideas = campaign.store.ideas()
    assert [idea.id for idea in ideas] == ["idea-0001", "idea-0002"]
    assert ideas[0].status == "rejected"
    assert ideas[1].status == "completed"
    takeaway = campaign.store.takeaways()[0]
    assert takeaway.idea_id == "idea-0002"
    assert takeaway.supporting_implementation_ids == ["impl-test"]
    state = json.loads(campaign.store.state_path.read_text(encoding="utf-8"))
    assert state["completed_rounds"] == 1
    assert state["status"] == "completed"


@pytest.mark.asyncio
async def test_custom_fixed_idea_skips_generation_and_filter(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "gdn.py").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref=commit,
        proposal_count=2,
        evolution_generations=1,
    )
    fixed_idea = Idea(
        id="idea-custom",
        title="Fixed custom idea",
        hypothesis="A fixed hypothesis.",
        mechanism="A fixed mechanism.",
        implementation_direction="Implement the fixed mechanism.",
        success_criteria=["Valid implementation"],
        risks=["Instability"],
    )
    agent = FakeAgent()
    campaign = FakeCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=agent,
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
        fixed_idea=fixed_idea,
    )

    await campaign.run(1)

    assert agent.stages == [
        "implementation_parent_selection",
        "idea_takeaway_synthesis",
    ]
    assert campaign.store.ideas()[0].id == "idea-custom"


@pytest.mark.asyncio
async def test_failed_round_records_the_active_idea_and_version(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "gdn.py").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")

    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref=commit,
        proposal_count=2,
        evolution_generations=1,
    )
    campaign = FailingCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=FakeAgent(),
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
    )

    with pytest.raises(RuntimeError, match="no indexed implementation evidence"):
        await campaign.run(1)

    selected = next(idea for idea in campaign.store.ideas() if idea.id == "idea-0002")
    assert selected.status == "failed"
    failure_path = next((tmp_path / "campaign" / "ideas").rglob("failure.json"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["active_idea_id"] == "idea-0002"
    version_path = failure_path.with_name("idea-version.json")
    version = json.loads(version_path.read_text(encoding="utf-8"))
    assert version["status"] == "failed"


def test_seed_evidence_uses_verified_parent_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    mutable = repo / "gdn.py"
    mutable.write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    parent = _git(repo, "rev-parse", "HEAD")
    mutable.write_text("normalized keys\n", encoding="utf-8")
    _git(repo, "commit", "-am", "seed")
    seed = _git(repo, "rev-parse", "HEAD")

    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref=parent,
        proposal_count=2,
        evolution_generations=1,
    )
    campaign = IdeaCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=FakeAgent(),
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
    )
    idea = Idea(
        id="idea-0001",
        title="Normalize keys",
        hypothesis="Normalized keys help.",
        mechanism="They control update scale.",
        implementation_direction="Normalize keys.",
        success_criteria=["Valid run"],
        risks=["Lost scale information"],
    )
    version = IdeaVersion(
        idea_id=idea.id,
        version=1,
        implementation_prompt="Normalize keys.",
        parent=ParentSelection(
            implementation_id="canonical",
            commit=parent,
            reason="Clean baseline.",
        ),
        mutable_files=["gdn.py"],
        immutable_base_commit=parent,
        evolution_generations=1,
        seed_commit=seed,
    )
    version_root = tmp_path / "campaign" / "ideas" / "idea-0001" / "v001"
    index_path = version_root / "evidence" / "extracted" / "api_index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "memory_cards": {
                    "program-seed": {
                        "commit": seed,
                        "parent_commit": None,
                        "changed_files": [],
                        "fitness": 0.0,
                        "metrics": {"is_valid": 1.0},
                        "verdict": "smoke",
                        "reflection": {"summary": "No repository changes."},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = campaign._load_evidence(version_root, idea, version)

    assert evidence[0]["parent_commit"] == parent
    assert evidence[0]["changed_files"] == ["gdn.py"]
    assert evidence[0]["lineage_basis"] == "verified git diff parent_commit..commit"
    implementation = campaign.store.implementations()[0]
    assert implementation.changed_files == ["gdn.py"]
    assert implementation.summary.startswith("Verified repository lineage:")


def test_parent_candidates_preserve_idea_breadth_and_recent_ties(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "gdn.py").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref=commit,
        proposal_count=2,
        parent_candidate_count=4,
        evolution_generations=1,
    )
    campaign = IdeaCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=FakeAgent(),
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
    )
    campaign._ensure_canonical_implementation()
    for number, idea_id in enumerate(
        ["idea-a", "idea-a", "idea-b", "idea-c", "idea-d"], start=1
    ):
        campaign.store.save_implementation(
            Implementation(
                id=f"impl-{number}",
                idea_id=idea_id,
                commit=f"commit-{number}",
                fitness=0.0,
                screen_fitness=0.0,
                is_valid=True,
            )
        )

    candidates = campaign._parent_candidates()

    evolved = [item for item in candidates if item.id != "canonical"]
    assert len(evolved) == 4
    assert len({item.idea_id for item in evolved}) >= 3
    assert "impl-5" in {item.id for item in evolved}
    assert candidates[-1].id == "canonical"


def test_implementation_prompt_contains_complete_fixed_contract_and_smoke(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "gdn.py").write_text("baseline\n")
    _git(repo, "add", "gdn.py")
    _git(repo, "commit", "-m", "baseline")
    task = ResearchTask(
        id="test-task",
        title="test",
        description="test",
        fixed_contract=["fixed"],
        mutable_files=["gdn.py"],
        canonical_ref="HEAD",
        proposal_count=2,
        evolution_generations=1,
    )
    campaign = IdeaCampaign(
        repo_root=repo,
        campaign_root=tmp_path / "campaign",
        task=task,
        agent=FakeAgent(),
        filter_mode="codex",
        selected_idea_id=None,
        model="fake",
        agent_timeout=1,
        evolution_generations=None,
        skip_seed_smoke=True,
    )
    idea = Idea(
        id="idea-fixed",
        title="Bound the gate",
        hypothesis="Bounded gates train better.",
        mechanism="Clamp the recurrent write gate.",
        implementation_direction="Apply a smooth bound before the state update.",
        success_criteria=["Lower heldout loss"],
        risks=["Saturation"],
    )

    prompt = campaign._implementation_prompt(idea)

    assert prompt.startswith("FIXED IDEA CONTRACT — MUST NOT BE REPLACED")
    assert "Clamp the recurrent write gate." in prompt
    assert "Lower heldout loss" in prompt
    assert "Saturation" in prompt
    assert "run-autoresearch-smoke.py" in prompt
    assert "--batches 128" in prompt
    assert "fidelity checklist" in prompt
