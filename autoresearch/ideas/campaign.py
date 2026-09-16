"""Run the outer idea loop around a fixed-idea GigaEvo evolution.

One round is deliberately explicit:

idea generation -> filtering -> parent selection -> one seed -> evolution
-> evidence indexing -> idea-level takeaways.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml

from autoresearch.ideas.agents import (
    CodexJSONAgent,
    JSONAgent,
    generate_ideas,
    select_idea,
    select_parent,
    synthesize_takeaways,
)
from autoresearch.ideas.models import (
    CampaignState,
    Idea,
    IdeaVersion,
    Implementation,
    ParentSelection,
    ResearchTask,
    Takeaway,
)
from autoresearch.ideas.promotion import initialize_promotion_state
from autoresearch.ideas.store import CampaignStore, _write_json
from gigaevo.repo_harness.backends import CommandCodingAgentBackend
from gigaevo.repo_harness.git_utils import (
    add_worktree,
    changed_files,
    commit_all,
    ensure_git_repo,
    remove_worktree,
    rev_parse,
)


class HumanSelectionRequired(RuntimeError):
    """A human must choose one of the persisted proposals before resuming."""


def _load_task(path: Path) -> ResearchTask:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ResearchTask.model_validate(value)


def _load_idea(path: Path) -> Idea:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Idea.model_validate(value)


def _bounded(value: Any, max_chars: int = 6000) -> Any:
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return value
    return {"truncated": True, "excerpt": serialized[:max_chars]}


STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"


def _last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _structured_feedback(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if STRUCTURED_FEEDBACK_MARKER not in line:
            continue
        value = line.split(STRUCTURED_FEEDBACK_MARKER, 1)[1].strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _card_feedback(card: dict[str, Any]) -> dict[str, Any]:
    artifact_refs = card.get("artifact_refs")
    if not isinstance(artifact_refs, dict):
        return {}
    items = artifact_refs.get("items")
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, dict) or item.get("name") != "benchmark_stderr.txt":
            continue
        path = Path(str(item.get("captured_path", "")))
        if path.is_file():
            return _structured_feedback(
                path.read_text(encoding="utf-8", errors="replace")
            )
    return {}


def _numeric_deltas(
    candidate: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    return {
        key: candidate[key] - baseline[key]
        for key in sorted(candidate.keys() & baseline.keys())
        if key not in {"is_valid", "heldout_evaluations"}
    }


class IdeaCampaign:
    def __init__(
        self,
        *,
        repo_root: Path,
        campaign_root: Path,
        task: ResearchTask,
        agent: JSONAgent,
        filter_mode: str,
        selected_idea_id: str | None,
        model: str,
        agent_timeout: float,
        evolution_generations: int | None,
        skip_seed_smoke: bool,
        fixed_idea: Idea | None = None,
    ) -> None:
        if filter_mode not in {"codex", "human"}:
            raise ValueError("filter_mode must be 'codex' or 'human'")
        self.repo_root = ensure_git_repo(repo_root)
        self.store = CampaignStore(campaign_root)
        self.task = task
        self.agent = agent
        self.filter_mode = filter_mode
        self.selected_idea_id = selected_idea_id
        self.model = model
        self.agent_timeout = agent_timeout
        self.evolution_generations = (
            evolution_generations
            if evolution_generations is not None
            else task.evolution_generations
        )
        self.skip_seed_smoke = skip_seed_smoke
        self.fixed_idea = fixed_idea

    async def run(self, rounds: int) -> None:
        state = self.store.load_state(self.store.root.name)
        self._ensure_canonical_implementation()
        self._ensure_canonical_evaluations()
        for _ in range(rounds):
            state.status = "running"
            self.store.save_state(state)
            try:
                await self._run_round(state)
            except HumanSelectionRequired:
                return
            except Exception as exc:
                self._record_failure(state, exc)
                state.status = "failed"
                self.store.save_state(state)
                raise
            state.completed_rounds += 1
            state.active_idea_id = None
            state.status = "ready"
            self.store.save_state(state)
        state.status = "completed"
        self.store.save_state(state)

    async def _run_round(self, state: CampaignState) -> None:
        existing_ideas = self.store.ideas()
        existing_takeaways = self.store.takeaways()
        fixed_idea_reason: str | None = None
        if self.fixed_idea is not None:
            if state.completed_rounds > 0:
                raise RuntimeError(
                    "A fixed-idea campaign runs exactly one idea version"
                )
            idea = self.fixed_idea
            previous = next(
                (item for item in existing_ideas if item.id == idea.id), None
            )
            if previous is not None:
                idea = previous
            else:
                self.store.save_idea(idea)
            proposals = [idea]
            fixed_idea_reason = "Loaded from the custom fixed-idea file."
        else:
            proposals = self._pending_human_proposals(existing_ideas)
        if not proposals:
            raw_proposals = await generate_ideas(
                self.agent,
                task=self.task,
                ideas=existing_ideas,
                takeaways=existing_takeaways,
            )

            known_idea_ids = {item.id for item in existing_ideas}
            known_takeaway_ids = {item.id for item in existing_takeaways}
            proposals = []
            for raw in raw_proposals:
                idea_id = f"idea-{state.next_idea_number:04d}"
                state.next_idea_number += 1
                raw = dict(raw)
                raw["id"] = idea_id
                proposal = Idea.model_validate(raw)
                unknown_ideas = set(proposal.source_idea_ids) - known_idea_ids
                unknown_takeaways = set(proposal.takeaway_ids) - known_takeaway_ids
                if unknown_ideas or unknown_takeaways:
                    raise ValueError(
                        f"{idea_id} cites unknown provenance: "
                        f"ideas={sorted(unknown_ideas)}, "
                        f"takeaways={sorted(unknown_takeaways)}"
                    )
                self.store.save_idea(proposal)
                proposals.append(proposal)
            self.store.save_state(state)

        if fixed_idea_reason is not None:
            selected, reason = proposals[0].id, fixed_idea_reason
        else:
            selected, reason = await self._filter(
                proposals=proposals,
                existing_ideas=existing_ideas,
                takeaways=existing_takeaways,
                state=state,
            )
        idea = next(item for item in proposals if item.id == selected)
        for proposal in proposals:
            if proposal.id != selected:
                proposal.status = "rejected"
                self.store.save_idea(proposal)
        idea.status = "approved"
        self.store.save_idea(idea)

        state.active_idea_id = idea.id
        self.store.save_state(state)

        candidates = self._parent_candidates()
        parent_id, parent_reason = await select_parent(
            self.agent,
            task=self.task,
            idea=idea,
            candidates=candidates,
            takeaways=existing_takeaways,
        )
        parent = next(item for item in candidates if item.id == parent_id)
        version_number = self._next_version(idea.id)
        version_root = self.store.idea_version_root(
            idea.id,
            idea.title,
            version_number,
        )
        version_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            version_root / "selection.json",
            {
                "idea_filter_mode": self.filter_mode,
                "idea_filter_reason": reason,
                "parent_selection_reason": parent_reason,
            },
        )

        implementation_prompt = self._implementation_prompt(idea)
        version = IdeaVersion(
            idea_id=idea.id,
            version=version_number,
            implementation_prompt=implementation_prompt,
            parent=ParentSelection(
                implementation_id=parent.id,
                commit=parent.commit,
                reason=parent_reason,
            ),
            mutable_files=self.task.mutable_files,
            immutable_base_commit=parent.commit,
            evolution_generations=self.evolution_generations,
        )
        self.store.save_version(version_root, version)
        active_idea_path = version_root / "idea.json"
        _write_json(active_idea_path, self._active_idea(idea, version_number))

        version.seed_commit = await self._create_seed(
            idea=idea,
            version=version,
            version_root=version_root,
        )
        version.status = "evolving"
        self.store.save_version(version_root, version)

        if (
            self.task.screen_batches is not None
            and self.task.confirmation_batches is not None
        ):
            self._initialize_promotion(version_root, idea, version)

        self._run_evolution(
            version=version,
            version_root=version_root,
            active_idea_path=active_idea_path,
        )
        evidence = self._load_evidence(version_root, idea, version)
        if not evidence:
            raise RuntimeError("Evolution produced no indexed implementation evidence")
        takeaway_values = await synthesize_takeaways(
            self.agent,
            task=self.task,
            idea=idea,
            version=version_number,
            evidence=evidence,
            existing_takeaways=existing_takeaways,
        )
        takeaways: list[Takeaway] = []
        evidence_ids = {str(item["implementation_id"]) for item in evidence}
        confirmed_ids = {
            str(item["implementation_id"])
            for item in evidence
            if item.get("confirmation", {}).get("is_valid")
        }
        existing_deduplication_keys = {
            item.deduplication_key
            for item in existing_takeaways
            if item.deduplication_key
        }
        for number, raw in enumerate(takeaway_values, start=1):
            value = dict(raw)
            value.setdefault(
                "deduplication_key",
                "-".join(str(value.get("claim", "")).lower().split())[:120],
            )
            cited_ids = set(value.get("supporting_implementation_ids", [])) | set(
                value.get("contradicting_implementation_ids", [])
            )
            if value.get("kind") in {
                "supported_mechanism",
                "refuted_mechanism",
            } and not (cited_ids & confirmed_ids):
                value["kind"] = "uncertainty"
                value["confidence"] = "low"
                value["claim"] = "Preliminary only: " + str(value.get("claim", ""))
            if (
                value.get("kind") == "infrastructure_failure"
                and value["deduplication_key"] in existing_deduplication_keys
            ):
                continue
            value.update(
                {
                    "id": f"takeaway-{idea.id}-{version_number:03d}-{number:03d}",
                    "idea_id": idea.id,
                    "idea_version": version_number,
                }
            )
            takeaway = Takeaway.model_validate(value)
            cited = set(takeaway.supporting_implementation_ids) | set(
                takeaway.contradicting_implementation_ids
            )
            if not cited <= evidence_ids:
                raise ValueError(
                    f"Takeaway {takeaway.id} cites unknown implementations: "
                    f"{sorted(cited - evidence_ids)}"
                )
            self.store.save_takeaway(takeaway)
            takeaways.append(takeaway)
            if takeaway.deduplication_key:
                existing_deduplication_keys.add(takeaway.deduplication_key)
        _write_json(
            version_root / "takeaways" / "final.json",
            [item.model_dump(mode="json") for item in takeaways],
        )
        version.status = "completed"
        idea.status = "completed"
        version.scientific_status = self._scientific_status(takeaways, confirmed_ids)
        idea.scientific_status = version.scientific_status
        self.store.save_version(version_root, version)
        self.store.save_idea(idea)

    def _record_failure(self, state: CampaignState, exc: Exception) -> None:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "active_idea_id": state.active_idea_id,
        }
        failure_path = self.store.root / "failure.json"
        if state.active_idea_id:
            for idea in self.store.ideas():
                if idea.id == state.active_idea_id:
                    idea.status = "failed"
                    self.store.save_idea(idea)
                    break
            versions = sorted(
                self.store.root.glob(
                    f"ideas/{state.active_idea_id}-*/v*/idea-version.json"
                )
            )
            if versions:
                version_path = versions[-1]
                version = IdeaVersion.model_validate_json(
                    version_path.read_text(encoding="utf-8")
                )
                version.status = "failed"
                self.store.save_version(version_path.parent, version)
                failure_path = version_path.parent / "failure.json"
        _write_json(failure_path, failure)

    async def _filter(
        self,
        *,
        proposals: list[Idea],
        existing_ideas: list[Idea],
        takeaways: list[Takeaway],
        state: CampaignState,
    ) -> tuple[str, str]:
        if self.filter_mode == "codex":
            return await select_idea(
                self.agent,
                task=self.task,
                proposals=proposals,
                ideas=existing_ideas,
                takeaways=takeaways,
            )
        if self.selected_idea_id in {item.id for item in proposals}:
            pending_path = self.store.root / "pending-human-selection.json"
            pending_path.unlink(missing_ok=True)
            return self.selected_idea_id or "", "Selected by the human filter."
        state.status = "awaiting_human"
        self.store.save_state(state)
        _write_json(
            self.store.root / "pending-human-selection.json",
            {
                "proposal_ids": [item.id for item in proposals],
                "instruction": "Resume with --filter-mode human --selected-idea-id IDEA_ID",
            },
        )
        raise HumanSelectionRequired("Human idea selection is required")

    def _pending_human_proposals(self, ideas: list[Idea]) -> list[Idea]:
        pending_path = self.store.root / "pending-human-selection.json"
        if self.filter_mode != "human" or not self.selected_idea_id:
            return []
        if not pending_path.is_file():
            return []
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        ids = pending.get("proposal_ids", []) if isinstance(pending, dict) else []
        proposals = [item for item in ideas if item.id in ids]
        if self.selected_idea_id not in {item.id for item in proposals}:
            raise ValueError(f"Selected idea {self.selected_idea_id!r} is not pending")
        return proposals

    def _ensure_canonical_implementation(self) -> None:
        canonical_commit = rev_parse(self.repo_root, self.task.canonical_ref)
        if any(item.id == "canonical" for item in self.store.implementations()):
            return
        self.store.save_implementation(
            Implementation(
                id="canonical",
                commit=canonical_commit,
                verdict="baseline",
                summary="Canonical fixed hybrid starting point.",
            )
        )

    def _ensure_canonical_evaluations(self) -> None:
        if self.task.screen_batches is None or self.task.confirmation_batches is None:
            return
        canonical = next(
            item for item in self.store.implementations() if item.id == "canonical"
        )
        if self.task.multifidelity_plan is not None and (
            canonical.screen_fitness is None or canonical.confirmation_fitness is None
        ):
            result = self._benchmark_multifidelity_commit(
                implementation_id="canonical",
                commit=canonical.commit,
                root=self.store.root / "baselines" / "canonical" / "multifidelity",
            )
            if result["metrics"].get("is_valid", 0.0) <= 0:
                raise RuntimeError("Canonical multi-fidelity evaluation failed")
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                raise RuntimeError("Canonical multi-fidelity trajectory is missing")
            screen = next(
                (
                    item
                    for item in trajectory
                    if item.get("budget_batches") == self.task.screen_batches
                ),
                None,
            )
            if not isinstance(screen, dict):
                raise RuntimeError("Canonical screen rung is missing")
            screen_metrics = screen.get("main_metrics")
            confirmation_metrics = trajectory[-1].get("main_metrics")
            if not isinstance(screen_metrics, dict) or not isinstance(
                confirmation_metrics, dict
            ):
                raise RuntimeError("Canonical rung metrics are missing")
            canonical.screen_metrics = screen_metrics
            canonical.confirmation_metrics = confirmation_metrics
            canonical.screen_feedback = result["feedback"]
            canonical.confirmation_feedback = result["feedback"]
            screen_loss = screen_metrics.get("heldout_loss_final")
            confirmation_loss = confirmation_metrics.get("heldout_loss_final")
            if not isinstance(screen_loss, int | float) or not isinstance(
                confirmation_loss, int | float
            ):
                raise RuntimeError("Canonical rung losses are missing")
            canonical.screen_fitness = 1.0 / (1.0 + float(screen_loss))
            canonical.confirmation_fitness = 1.0 / (1.0 + float(confirmation_loss))
            canonical.fitness = canonical.confirmation_fitness
            canonical.is_valid = True
            canonical.evidence_refs.extend(
                [
                    str(result["result_path"]),
                    str(result["trajectory_path"]),
                ]
            )
            self.store.save_implementation(canonical)
            return
        changed = False
        if canonical.screen_fitness is None:
            result = self._benchmark_commit(
                implementation_id="canonical",
                commit=canonical.commit,
                batches=self.task.screen_batches,
                root=self.store.root
                / "baselines"
                / "canonical"
                / f"screen-{self.task.screen_batches}",
            )
            if result["metrics"].get("is_valid", 0.0) <= 0:
                raise RuntimeError("Canonical 1024-step screen failed")
            canonical.screen_metrics = result["metrics"]
            canonical.screen_feedback = result["feedback"]
            canonical.screen_fitness = result["metrics"].get("fitness")
            changed = True
        if canonical.confirmation_fitness is None:
            result = self._benchmark_commit(
                implementation_id="canonical",
                commit=canonical.commit,
                batches=self.task.confirmation_batches,
                root=self.store.root
                / "baselines"
                / "canonical"
                / f"confirm-{self.task.confirmation_batches}",
            )
            if result["metrics"].get("is_valid", 0.0) <= 0:
                raise RuntimeError("Canonical 4096-step confirmation failed")
            canonical.confirmation_metrics = result["metrics"]
            canonical.confirmation_feedback = result["feedback"]
            canonical.confirmation_fitness = result["metrics"].get("fitness")
            changed = True
        if changed:
            canonical.fitness = (
                canonical.confirmation_fitness
                if canonical.confirmation_fitness is not None
                else canonical.screen_fitness
            )
            canonical.is_valid = bool(
                canonical.screen_metrics.get("is_valid")
                and canonical.confirmation_metrics.get("is_valid")
            )
            canonical.verdict = "confirmed_baseline"
            self.store.save_implementation(canonical)

    def _parent_candidates(self) -> list[Implementation]:
        implementations = self.store.implementations()
        canonical = [item for item in implementations if item.id == "canonical"]
        evolved = [
            item
            for item in implementations
            if item.id != "canonical" and item.is_valid and item.fitness is not None
        ]
        positions = {item.id: number for number, item in enumerate(implementations)}

        def rank(item: Implementation) -> tuple[float, float, int, str]:
            confirmation = (
                item.confirmation_fitness
                if item.confirmation_fitness is not None
                else float("-inf")
            )
            fitness = item.fitness if item.fitness is not None else float("-inf")
            return (confirmation, fitness, positions[item.id], item.id)

        evolved.sort(key=rank, reverse=True)
        best_per_idea: dict[str, Implementation] = {}
        for item in evolved:
            key = item.idea_id or "unknown"
            if key not in best_per_idea:
                best_per_idea[key] = item
        result: list[Implementation] = []

        def add(item: Implementation) -> None:
            if item.id not in {existing.id for existing in result}:
                result.append(item)

        # First preserve breadth across research ideas, then global score, then
        # recency. This avoids insertion-order starvation when scores tie.
        for item in sorted(best_per_idea.values(), key=rank, reverse=True):
            add(item)
            if len(result) >= max(1, self.task.parent_candidate_count // 2):
                break
        for item in evolved:
            add(item)
            if len(result) >= self.task.parent_candidate_count:
                break
        for item in reversed(implementations):
            if item in evolved:
                add(item)
            if len(result) >= self.task.parent_candidate_count:
                break
        result = result[: self.task.parent_candidate_count]
        if canonical:
            result.append(canonical[0])
        return result

    def _next_version(self, idea_id: str) -> int:
        versions = self.store.root.glob(f"ideas/{idea_id}-*/v*/idea-version.json")
        return 1 + sum(1 for _ in versions)

    def _implementation_prompt(self, idea: Idea) -> str:
        success = "\n".join(f"- {item}" for item in idea.success_criteria)
        risks = "\n".join(f"- {item}" for item in idea.risks) or "- None listed."
        return (
            "FIXED IDEA CONTRACT — MUST NOT BE REPLACED\n\n"
            f"Title: {idea.title}\n\n"
            f"Hypothesis: {idea.hypothesis}\n\n"
            f"Required mechanism: {idea.mechanism}\n\n"
            f"Implementation requirements: {idea.implementation_direction}\n\n"
            f"Success criteria:\n{success}\n\n"
            f"Known risks:\n{risks}\n\n"
            "Implement every part of this idea faithfully in the mutable GDN file. "
            "You may repair or refine implementation details, but you must not replace "
            "the mechanism with a different research idea. Keep the fixed Transformer "
            "layers, hybrid placement, training loop, evaluation, and immutable files "
            "unchanged.\n\n"
            "Before finishing, run this exact executable smoke test and keep fixing the "
            "implementation until both the process succeeds and its structured result "
            "reports is_valid == 1:\n\n"
            f"{shlex.join(self._smoke_command())}\n\n"
            "Finish with a compact fidelity checklist mapping every idea requirement "
            "to the code that implements it, followed by the smoke-test result."
        )

    def _smoke_command(self) -> list[str]:
        return [
            sys.executable,
            "tools/run-autoresearch-smoke.py",
            "--batches",
            "128",
            "--run-dir",
            "runs/codex-smoke",
            "--check-file",
            "autoresearch/model/gdn.py",
            "--gpus",
            "1",
            "--train-microbatch-size",
            os.environ.get("AUTORESEARCH_TRAIN_MICROBATCH_SIZE", "16"),
            "--eval-batch-size",
            os.environ.get("AUTORESEARCH_EVAL_BATCH_SIZE", "32"),
            "--loader-workers",
            os.environ.get("AUTORESEARCH_LOADER_WORKERS", "8"),
            "--gradient-log-interval",
            os.environ.get("AUTORESEARCH_GRADIENT_LOG_INTERVAL", "20"),
            "--disable-optimizer-metrics",
        ]

    def _active_idea(self, idea: Idea, version: int) -> dict[str, Any]:
        return {
            "id": idea.id,
            "idea_version": version,
            "status": "approved",
            "title": idea.title,
            "hypothesis": idea.hypothesis,
            "proposed_change": idea.implementation_direction,
            "expected_mechanism": idea.mechanism,
            "success_signal": idea.success_criteria,
            "risks": idea.risks,
            "source_idea_ids": idea.source_idea_ids,
            "memory_ids": idea.takeaway_ids,
            "available_memory_ids": idea.takeaway_ids,
        }

    async def _create_seed(
        self,
        *,
        idea: Idea,
        version: IdeaVersion,
        version_root: Path,
    ) -> str:
        worktree = version_root / "seed-worktree"
        campaign_id = "-".join(
            part
            for part in "".join(
                character if character.isalnum() else "-"
                for character in self.store.root.name
            ).split("-")
            if part
        )
        branch = f"campaign/{campaign_id}/{idea.id}/v{version.version:03d}/seed"
        if worktree.exists():
            raise RuntimeError(f"Seed worktree already exists: {worktree}")
        add_worktree(self.repo_root, worktree, version.parent.commit, branch=branch)
        try:
            backend = CommandCodingAgentBackend(
                name="codex-seed",
                stdin_prompt=True,
                command=[
                    str(self.repo_root / "tools/codex-proxy"),
                    "exec",
                    "--model",
                    self.model,
                    "--ephemeral",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--cd",
                    "{worktree}",
                    "-",
                ],
            )
            prompt = (
                version.implementation_prompt
                + "\n\nAllowed changed files:\n"
                + "\n".join(f"- {path}" for path in version.mutable_files)
                + "\n\nRead the repository and edit the seed directly. Run cheap relevant checks."
            )
            result = await backend.run(
                prompt=prompt,
                cwd=worktree,
                log_dir=version_root / "prompts" / "seed",
                timeout=self.agent_timeout,
                variables={"worktree": str(worktree)},
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Seed implementation failed with exit code {result.exit_code}: "
                    f"{result.stderr[-2000:]}"
                )
            commit = commit_all(
                worktree,
                message=f"Seed {idea.id} v{version.version:03d}",
            )
            files = changed_files(self.repo_root, version.parent.commit, commit)
            if not files:
                raise RuntimeError("Seed implementation made no changes")
            disallowed = sorted(set(files) - set(version.mutable_files))
            if disallowed:
                raise RuntimeError(
                    "Seed changed immutable files: " + ", ".join(disallowed)
                )
            if not self.skip_seed_smoke:
                smoke_log = version_root / "seed-smoke.log"
                with smoke_log.open("w", encoding="utf-8") as output:
                    smoke = subprocess.run(
                        self._smoke_command(),
                        cwd=worktree,
                        env=os.environ.copy(),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if smoke.returncode != 0:
                    raise RuntimeError(f"Seed smoke failed; see {smoke_log}")
            return commit
        finally:
            if worktree.exists():
                remove_worktree(self.repo_root, worktree)

    def _initialize_promotion(
        self, version_root: Path, idea: Idea, version: IdeaVersion
    ) -> None:
        parent = next(
            item
            for item in self.store.implementations()
            if item.id == version.parent.implementation_id
        )
        if self.task.screen_batches is None or self.task.confirmation_batches is None:
            raise RuntimeError("Fixed-idea promotion requires short and long horizons")
        if not parent.screen_metrics or not parent.confirmation_metrics:
            raise RuntimeError(
                "Selected parent must have completed short and long heldout evaluations"
            )
        initialize_promotion_state(
            version_root / "promotion-state.json",
            idea_contract=self._active_idea(idea, version.version),
            confirmed_commit=parent.commit,
            screen_metrics=parent.screen_metrics,
            confirmation_metrics=parent.confirmation_metrics,
            screen_batches=self.task.screen_batches,
            confirmation_batches=self.task.confirmation_batches,
            screen_eval_batches=self.task.screen_eval_batches,
            tensorboard_dir=version_root / "system" / "tb_logs" / "evolution",
        )

    def _run_evolution(
        self,
        *,
        version: IdeaVersion,
        version_root: Path,
        active_idea_path: Path,
    ) -> None:
        system_root = version_root / "system"
        environment = os.environ.copy()
        environment.update(
            {
                "AUTORESEARCH_ACTIVE_IDEA_PATH": str(active_idea_path),
                "AUTORESEARCH_RUN_ROOT": str(version_root / "implementations"),
                "AUTORESEARCH_MEMORY_DIR": str(version_root / "evidence"),
                "AUTORESEARCH_REVIEW_MODE": "auto",
                "AUTORESEARCH_PROMOTION_STATE_PATH": str(
                    version_root / "promotion-state.json"
                ),
            }
        )
        environment.pop("AUTORESEARCH_SMOKE_BATCHES", None)
        environment.pop("AUTORESEARCH_DIAGNOSTIC_BATCHES", None)
        if self.task.multifidelity_plan is not None:
            environment["AUTORESEARCH_MULTIFIDELITY_PLAN"] = (
                self.task.multifidelity_plan
            )
        if self.task.screen_batches is not None:
            environment["AUTORESEARCH_SCREEN_BATCHES"] = str(self.task.screen_batches)
            environment["AUTORESEARCH_SCREEN_EVAL_BATCHES"] = str(
                self.task.screen_eval_batches
            )
        redis_prefix = (
            f"{self.store.root.name}-{version.idea_id}-v{version.version:03d}"
        )
        command = [
            sys.executable,
            "run.py",
            "experiment=llm_foundry_idea_evolution",
            f"redis.prefix={redis_prefix}",
            f"hydra.run.dir={system_root}",
            f"repo_harness.seed_ref={version.parent.commit}",
            f"repo_harness.seed_refs=[{version.parent.commit},{version.seed_commit}]",
            f"max_generations={version.evolution_generations}",
        ]
        log_path = version_root / "evolution.log"
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Inner evolution failed; see {log_path}")

    def _load_evidence(
        self,
        version_root: Path,
        idea: Idea,
        version: IdeaVersion,
    ) -> list[dict[str, Any]]:
        index_path = version_root / "evidence" / "extracted" / "api_index.json"
        if not index_path.is_file():
            return []
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        cards = raw.get("memory_cards", {})
        if not isinstance(cards, dict):
            return []
        promotion_path = version_root / "promotion-state.json"
        promotion_state = (
            json.loads(promotion_path.read_text(encoding="utf-8"))
            if promotion_path.is_file()
            else {}
        )
        promotion_results = promotion_state.get("results", {})
        if not isinstance(promotion_results, dict):
            promotion_results = {}
        canonical = next(
            (item for item in self.store.implementations() if item.id == "canonical"),
            None,
        )
        evidence: list[dict[str, Any]] = []
        for card_id, card in cards.items():
            if not isinstance(card, dict):
                continue
            commit = str(card.get("commit", ""))
            # The confirmed parent is loaded alongside the idea seed so the
            # archive always has a valid fallback. It is context, not a new
            # implementation of this idea.
            if commit == version.parent.commit:
                continue
            implementation_id = f"impl-{commit[:12]}" if commit else str(card_id)
            reflection = card.get("reflection")
            summary = ""
            if isinstance(reflection, dict):
                summary = str(
                    reflection.get("reflection") or reflection.get("summary") or ""
                )
            parent_commit = str(card.get("parent_commit") or "") or None
            if commit == version.seed_commit and parent_commit is None:
                parent_commit = version.parent.commit
            reported_changed_files = [
                str(path) for path in card.get("changed_files", [])
            ]
            verified_changed_files = reported_changed_files
            if commit and parent_commit:
                verified_changed_files = changed_files(
                    self.repo_root,
                    parent_commit,
                    commit,
                )
            if (
                parent_commit != (str(card.get("parent_commit") or "") or None)
                or verified_changed_files != reported_changed_files
            ):
                summary = (
                    "Verified repository lineage: "
                    f"{parent_commit}..{commit} changes "
                    f"{verified_changed_files}. The evolution reflection below may use "
                    "the evaluated seed itself as its comparison point.\n\n" + summary
                )
            promotion_result = promotion_results.get(commit, {})
            if not isinstance(promotion_result, dict):
                promotion_result = {}
            stored_screen_metrics = promotion_result.get("screen_metrics")
            metric_source = (
                stored_screen_metrics
                if isinstance(stored_screen_metrics, dict)
                else card.get("metrics", {})
            )
            metrics = {
                str(key): float(value)
                for key, value in metric_source.items()
                if isinstance(value, int | float)
            }
            feedback = _card_feedback(card)
            stored_feedback = promotion_result.get("feedback")
            if isinstance(stored_feedback, dict):
                feedback = stored_feedback
            stored_confirmation_metrics = promotion_result.get("confirmation_metrics")
            trajectory_path = (
                version_root
                / "implementations"
                / f"impl-{commit}"
                / "training"
                / "multifidelity-trajectory.json"
            )
            trajectory: list[dict[str, Any]] = []
            if trajectory_path.is_file():
                loaded_trajectory = json.loads(
                    trajectory_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded_trajectory, list):
                    trajectory = [
                        item for item in loaded_trajectory if isinstance(item, dict)
                    ]
            if trajectory:
                screen_observation = next(
                    (
                        item
                        for item in trajectory
                        if item.get("budget_batches") == self.task.screen_batches
                    ),
                    trajectory[0],
                )
                final_observation = trajectory[-1]
                observed_screen = screen_observation.get("main_metrics")
                observed_confirmation = final_observation.get("main_metrics")
                if isinstance(observed_screen, dict):
                    metrics = {
                        str(key): float(value)
                        for key, value in observed_screen.items()
                        if isinstance(value, int | float)
                    }
                if isinstance(observed_confirmation, dict):
                    stored_confirmation_metrics = observed_confirmation
                feedback = {
                    **feedback,
                    "status": "multifidelity_complete",
                    "trajectory": str(trajectory_path),
                    "screen_batches": screen_observation.get("budget_batches"),
                    "multifidelity_observations": [
                        {
                            "budget_batches": item.get("budget_batches"),
                            "features": item.get("features"),
                            "probability": item.get("probability"),
                            "decision": item.get("decision"),
                        }
                        for item in trajectory
                    ],
                }
            confirmation_metrics = {
                str(key): float(value)
                for key, value in (
                    stored_confirmation_metrics.items()
                    if isinstance(stored_confirmation_metrics, dict)
                    else []
                )
                if isinstance(value, int | float)
            }
            screen_loss = metrics.get(
                "heldout_loss_final" if trajectory else "heldout_loss_auc"
            )
            screen_fitness = (
                1.0 / (1.0 + screen_loss)
                if screen_loss is not None and screen_loss >= 0.0
                else None
            )
            confirmation_loss = confirmation_metrics.get(
                "heldout_loss_final" if trajectory else "heldout_loss_auc"
            )
            confirmation_fitness = (
                1.0 / (1.0 + confirmation_loss)
                if confirmation_loss is not None and confirmation_loss >= 0.0
                else None
            )
            implementation = Implementation(
                id=implementation_id,
                idea_id=idea.id,
                idea_version=version.version,
                commit=commit,
                parent_commit=parent_commit,
                fitness=card.get("fitness")
                if isinstance(card.get("fitness"), int | float)
                else None,
                screen_fitness=screen_fitness,
                confirmation_fitness=confirmation_fitness,
                screen_metrics=metrics,
                confirmation_metrics=confirmation_metrics,
                screen_feedback=feedback,
                confirmation_feedback=(feedback if confirmation_metrics else {}),
                is_valid=(
                    float(card.get("metrics", {}).get("is_valid", 0.0)) > 0
                    if isinstance(card.get("metrics"), dict)
                    else None
                ),
                verdict=str(card.get("verdict", "unknown")),
                summary=summary[:4000],
                changed_files=verified_changed_files,
                evidence_refs=[str(index_path), str(card_id)],
            )
            if trajectory:
                implementation.evidence_refs.append(str(trajectory_path))
            self.store.save_implementation(implementation)
            evidence.append(
                {
                    "implementation_id": implementation.id,
                    "commit": implementation.commit,
                    "parent_commit": implementation.parent_commit,
                    "fitness": implementation.fitness,
                    "is_valid": implementation.is_valid,
                    "verdict": implementation.verdict,
                    "changed_files": implementation.changed_files,
                    "lineage_basis": "verified git diff parent_commit..commit",
                    "metric_deltas": card.get("metric_deltas", {}),
                    "screen": {
                        "batches": feedback.get("screen_batches"),
                        "metrics": metrics,
                        "feedback": _bounded(feedback, 5000),
                        "canonical_metrics": (
                            canonical.screen_metrics if canonical is not None else {}
                        ),
                        "deltas_vs_canonical": _numeric_deltas(
                            metrics,
                            canonical.screen_metrics if canonical is not None else {},
                        ),
                    },
                    **(
                        {
                            "confirmation": {
                                "batches": self.task.confirmation_batches,
                                "is_valid": True,
                                "metrics": confirmation_metrics,
                                "feedback": _bounded(feedback, 5000),
                                "promotion": feedback.get("promotion"),
                            }
                        }
                        if confirmation_metrics
                        else {}
                    ),
                    "reflection": _bounded(reflection),
                    "artifact_refs": _bounded(card.get("artifact_refs", {}), 3000),
                }
            )
        _write_json(version_root / "evidence" / "ledger.json", evidence)
        return evidence

    def _benchmark_commit(
        self, *, implementation_id: str, commit: str, batches: int, root: Path
    ) -> dict[str, dict[str, Any]]:
        result_path = root / "result.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        root.mkdir(parents=True, exist_ok=True)
        worktree = root / "worktree"
        training = root / "training"
        add_worktree(self.repo_root, worktree, commit, detach=True)
        try:
            environment = os.environ.copy()
            for name in (
                "AUTORESEARCH_SMOKE_BATCHES",
                "AUTORESEARCH_DIAGNOSTIC_BATCHES",
                "AUTORESEARCH_SCREEN_BATCHES",
            ):
                environment.pop(name, None)
            environment["AUTORESEARCH_RUN_ROOT"] = str(training)
            command = [
                sys.executable,
                "tools/run-autoresearch-benchmark.py",
                "--gpus",
                "1",
                "--train-microbatch-size",
                os.environ.get("AUTORESEARCH_TRAIN_MICROBATCH_SIZE", "16"),
                "--eval-batch-size",
                os.environ.get("AUTORESEARCH_EVAL_BATCH_SIZE", "32"),
                "--loader-workers",
                os.environ.get("AUTORESEARCH_LOADER_WORKERS", "8"),
                "--gradient-log-interval",
                os.environ.get("AUTORESEARCH_GRADIENT_LOG_INTERVAL", "20"),
                "--disable-optimizer-metrics",
                "--screen-batches",
                str(batches),
                "--screen-eval-batches",
                str(self.task.screen_eval_batches),
                "--run-dir",
                str(training),
            ]
            process = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.agent_timeout + 43_200,
            )
            (root / "benchmark.stdout.log").write_text(process.stdout, encoding="utf-8")
            (root / "benchmark.stderr.log").write_text(process.stderr, encoding="utf-8")
            metrics = _last_json_object(process.stdout)
            feedback = _structured_feedback(process.stderr)
            value = {
                "implementation_id": implementation_id,
                "commit": commit,
                "batches": batches,
                "wrapper_returncode": process.returncode,
                "metrics": metrics,
                "feedback": feedback,
            }
            _write_json(result_path, value)
            return value
        finally:
            if worktree.exists():
                remove_worktree(self.repo_root, worktree)

    def _benchmark_multifidelity_commit(
        self, *, implementation_id: str, commit: str, root: Path
    ) -> dict[str, Any]:
        result_path = root / "result.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        if self.task.multifidelity_plan is None:
            raise RuntimeError("Research task has no multi-fidelity plan")
        root.mkdir(parents=True, exist_ok=True)
        worktree = root / "worktree"
        training = root / "training"
        add_worktree(self.repo_root, worktree, commit, detach=True)
        try:
            environment = os.environ.copy()
            for name in (
                "AUTORESEARCH_SMOKE_BATCHES",
                "AUTORESEARCH_DIAGNOSTIC_BATCHES",
                "AUTORESEARCH_SCREEN_BATCHES",
            ):
                environment.pop(name, None)
            command = [
                sys.executable,
                "tools/run-multifidelity-autoresearch-benchmark.py",
                "--multifidelity-plan",
                self.task.multifidelity_plan,
                "--commit",
                commit,
                "--run-dir",
                str(training),
                "--screen-eval-batches",
                str(self.task.screen_eval_batches),
                "--gpus",
                "1",
                "--train-microbatch-size",
                os.environ.get("AUTORESEARCH_TRAIN_MICROBATCH_SIZE", "16"),
                "--eval-batch-size",
                os.environ.get("AUTORESEARCH_EVAL_BATCH_SIZE", "32"),
                "--loader-workers",
                os.environ.get("AUTORESEARCH_LOADER_WORKERS", "8"),
                "--gradient-log-interval",
                os.environ.get("AUTORESEARCH_GRADIENT_LOG_INTERVAL", "20"),
                "--disable-optimizer-metrics",
            ]
            process = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.agent_timeout + 43_200,
            )
            (root / "benchmark.stdout.log").write_text(process.stdout, encoding="utf-8")
            (root / "benchmark.stderr.log").write_text(process.stderr, encoding="utf-8")
            metrics = _last_json_object(process.stdout)
            feedback = _structured_feedback(process.stderr)
            trajectory_path = training / "multifidelity-trajectory.json"
            trajectory = (
                json.loads(trajectory_path.read_text(encoding="utf-8"))
                if trajectory_path.is_file()
                else []
            )
            value = {
                "implementation_id": implementation_id,
                "commit": commit,
                "wrapper_returncode": process.returncode,
                "metrics": metrics,
                "feedback": feedback,
                "trajectory": trajectory,
                "trajectory_path": str(trajectory_path),
                "result_path": str(result_path),
            }
            _write_json(result_path, value)
            return value
        finally:
            if worktree.exists():
                remove_worktree(self.repo_root, worktree)

    @staticmethod
    def _scientific_status(takeaways: list[Takeaway], confirmed_ids: set[str]) -> str:
        if not confirmed_ids:
            return "preliminary"
        kinds = {item.kind for item in takeaways}
        if "supported_mechanism" in kinds:
            return "supported"
        if "refuted_mechanism" in kinds:
            return "refuted"
        return "inconclusive"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Independent Git repository whose gdn.py candidates are evolved.",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("problems/llm_foundry_autoresearch/research_task.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--filter-mode", choices=("codex", "human"), default="codex")
    parser.add_argument("--selected-idea-id")
    parser.add_argument(
        "--idea-file",
        type=Path,
        help="Run exactly this custom idea; omit to generate and select one idea.",
    )
    parser.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--agent-timeout", type=float, default=2400.0)
    parser.add_argument("--evolution-generations", type=int)
    parser.add_argument("--skip-seed-smoke", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    campaign_root = args.campaign_root.expanduser().resolve()
    task = _load_task(args.task.expanduser().resolve())
    agent = CodexJSONAgent(
        repo_root=repo_root,
        model=args.model,
        timeout=args.agent_timeout,
        usage_log_path=campaign_root / "codex-usage.jsonl",
    )
    campaign = IdeaCampaign(
        repo_root=repo_root,
        campaign_root=campaign_root,
        task=task,
        agent=agent,
        filter_mode=args.filter_mode,
        selected_idea_id=args.selected_idea_id,
        model=args.model,
        agent_timeout=args.agent_timeout,
        evolution_generations=args.evolution_generations,
        skip_seed_smoke=args.skip_seed_smoke,
        fixed_idea=(
            _load_idea(args.idea_file.expanduser().resolve())
            if args.idea_file is not None
            else None
        ),
    )
    asyncio.run(campaign.run(args.rounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
