"""Small GigaEvo hooks that add reviewed research decisions."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from autoresearch.program_memory import program_to_card
from autoresearch.review import find_item, load_item, write_item
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.evolution.engine.core import EvolutionStopRequested
from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.memory.provider import MemoryProvider
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState

_BOOTSTRAP_MEMORY_READY = "autoresearch_bootstrap_memory_ready"


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _program_has_valid_result(program: Program, primary_key: str) -> bool:
    """Require an explicitly valid execution with a finite objective."""

    validity = program.metrics.get("is_valid")
    fitness = program.metrics.get(primary_key)
    if not (_finite_number(validity) and float(validity) > 0):
        return False
    if not _finite_number(fitness):
        return False
    feedback = program.metadata.get("repo_benchmark_feedback")
    if isinstance(feedback, dict):
        structured = feedback.get("structured_feedback")
        if isinstance(structured, dict) and structured.get("status") in {
            "diagnostic_complete",
            "dry_run",
            "smoke_complete",
        }:
            return False
    return True


def _card_has_valid_result(card: dict[str, Any], primary_key: str) -> bool:
    """Apply the same validity boundary to serialized execution cards."""

    if card.get("verdict") in {"diagnostic", "invalid", "smoke"}:
        return False
    metrics = card.get("metrics")
    if not isinstance(metrics, dict):
        return False
    validity = metrics.get("is_valid")
    fitness = card.get("fitness", metrics.get(primary_key))
    return bool(
        _finite_number(validity) and float(validity) > 0 and _finite_number(fitness)
    )


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if match is None:
            raise ValueError("Ideator did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Ideator response must be a JSON object")
    return value


def _proposal_payload(value: dict[str, Any]) -> dict[str, Any]:
    required_strings = (
        "title",
        "hypothesis",
        "proposed_change",
        "expected_mechanism",
        "success_signal",
    )
    payload: dict[str, Any] = {}
    for key in required_strings:
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"Ideator response has an invalid {key!r}")
        payload[key] = field.strip()
    for key in ("memory_ids", "risks"):
        field = value.get(key, [])
        if not isinstance(field, list) or not all(
            isinstance(item, str) for item in field
        ):
            raise ValueError(f"Ideator response has an invalid {key!r}")
        payload[key] = [item.strip() for item in field if item.strip()]
    return payload


class ReviewedIdeaPreStepHook:
    """Propose and approve one shared research idea before each generation."""

    def __init__(
        self,
        *,
        storage: ProgramStorage,
        llm: Any,
        memory_provider: MemoryProvider,
        task_description: str,
        review_dir: str | Path,
        primary_key: str = "fitness",
        review_mode: str = "mandatory",
        poll_interval_seconds: float = 2.0,
        bootstrap_memory_hook: Any | None = None,
    ) -> None:
        if review_mode not in {"mandatory", "auto"}:
            raise ValueError("review_mode must be 'mandatory' or 'auto'")
        self.storage = storage
        self.llm = llm
        self.memory_provider = memory_provider
        self.task_description = task_description
        self.review_dir = Path(review_dir)
        self.primary_key = primary_key
        self.review_mode = review_mode
        self.poll_interval_seconds = poll_interval_seconds
        self.bootstrap_memory_hook = bootstrap_memory_hook

    async def __call__(self) -> None:
        generation = int(await self.storage.load_run_state("total_generations") or 0)
        await self._bootstrap_initial_evidence(generation)
        attempt = 0
        while True:
            attempt += 1
            proposal_path = self._proposal_path(generation, attempt)
            if proposal_path.exists():
                item = load_item(proposal_path)
            else:
                item = await self._propose(generation, attempt)
                write_item(proposal_path, item)
            if self.review_mode == "auto" and item.get("status") == "pending":
                item["status"] = "approved"
                item["reviewer_note"] = "Automatically approved by configured policy."
                write_item(proposal_path, item)

            item = await self._wait_for_decision(proposal_path)
            if item["status"] == "rejected":
                logger.info("[ReviewedIdea] Idea {} rejected", item["id"])
                continue
            if item["status"] not in {"approved", "redacted"}:
                raise ValueError(f"Unexpected final idea status: {item['status']}")
            self._validate_memory_citations(item)
            write_item(self.review_dir / "active_idea.json", item)
            logger.info(
                "[ReviewedIdea] Generation {} uses approved idea {}",
                generation,
                item["id"],
            )
            return

    async def _bootstrap_initial_evidence(self, generation: int) -> None:
        if generation != 0:
            return
        while True:
            incomplete = sum(
                [
                    await self.storage.count_by_status(ProgramState.QUEUED.value),
                    await self.storage.count_by_status(ProgramState.RUNNING.value),
                ]
            )
            if incomplete == 0:
                break
            await asyncio.sleep(self.poll_interval_seconds)
        if await self.storage.load_run_state(_BOOTSTRAP_MEMORY_READY):
            return
        if self.bootstrap_memory_hook is not None:
            logger.info(
                "[ReviewedIdea] Extracting and reviewing baseline evidence "
                "before the first idea"
            )
            await self.bootstrap_memory_hook()
        await self.storage.save_run_state(_BOOTSTRAP_MEMORY_READY, 1)

    def _proposal_path(self, generation: int, attempt: int) -> Path:
        return (
            self.review_dir
            / "ideas"
            / f"generation-{generation:05d}-attempt-{attempt:03d}.json"
        )

    async def _propose(self, generation: int, attempt: int) -> dict[str, Any]:
        incumbent = await self._incumbent()
        if incumbent is None:
            raise EvolutionStopRequested(
                "Cannot generate an architecture idea without a valid, finite "
                "scientific baseline or elite"
            )
        selection = await self.memory_provider.select_cards(
            incumbent,
            task_description=self.task_description,
            metrics_description=f"Primary metric: {self.primary_key}",
        )
        memory_ids = selection.card_ids if selection is not None else []
        memory_cards = selection.cards if selection is not None else []
        context = {
            "generation": generation,
            "task": self.task_description,
            "incumbent": self._program_context(incumbent),
            "approved_memory_ids": memory_ids,
            "approved_memory_cards": memory_cards,
        }
        messages = [
            SystemMessage(
                content=(
                    "You are the ideator in an execution-grounded model research "
                    "loop. Propose exactly one coherent decoder-layer experiment. "
                    "Use only the supplied evidence, keep the fixed training and "
                    "evaluation contract unchanged, and return only one JSON object."
                )
            ),
            HumanMessage(
                content=(
                    "Return keys title, hypothesis, proposed_change, "
                    "expected_mechanism, memory_ids, success_signal, and risks. "
                    "memory_ids and risks must be arrays of strings. Every cited "
                    "memory ID must appear in approved_memory_ids.\n\nContext:\n"
                    + json.dumps(context, indent=2, default=str)
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        payload = _proposal_payload(_json_object(_message_text(response)))
        unknown_memory = set(payload["memory_ids"]) - set(memory_ids)
        if unknown_memory:
            raise ValueError(
                "Ideator cited unavailable memory: " + ", ".join(sorted(unknown_memory))
            )
        return {
            "id": f"idea-{uuid4().hex[:12]}",
            "kind": "idea",
            "generation": generation,
            "attempt": attempt,
            "status": "pending",
            "reviewer_note": None,
            "reviewed_at": None,
            "available_memory_ids": memory_ids,
            "original_proposal": payload,
            **payload,
        }

    async def _incumbent(self) -> Program | None:
        programs = await self.storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        candidates = [
            program
            for program in programs
            if _program_has_valid_result(program, self.primary_key)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda program: program.metrics[self.primary_key])

    def _program_context(self, program: Program | None) -> dict[str, Any] | None:
        if program is None:
            return None
        reflection = program.metadata.get("repo_reflection")
        return {
            "program_id": program.id,
            "generation": program.generation,
            "metrics": dict(program.metrics),
            "repo_reflection": reflection if isinstance(reflection, dict) else None,
        }

    async def _wait_for_decision(self, proposal_path: Path) -> dict[str, Any]:
        while True:
            item = load_item(proposal_path)
            if item.get("status") in {"approved", "rejected", "redacted"}:
                return item
            await asyncio.sleep(self.poll_interval_seconds)

    @staticmethod
    def _validate_memory_citations(item: dict[str, Any]) -> None:
        memory_ids = item.get("memory_ids")
        if not isinstance(memory_ids, list) or not all(
            isinstance(memory_id, str) for memory_id in memory_ids
        ):
            raise ValueError("Approved idea has invalid memory_ids")
        available = item.get("available_memory_ids", [])
        if not isinstance(available, list) or not all(
            isinstance(memory_id, str) for memory_id in available
        ):
            raise ValueError("Approved idea has invalid available_memory_ids")
        unknown = set(memory_ids) - set(available)
        if unknown:
            raise ValueError(
                "Approved idea cites unavailable memory: " + ", ".join(sorted(unknown))
            )
        if available and not memory_ids:
            raise ValueError("Approved idea must cite at least one available memory")


class ReviewedMemoryProvider(MemoryProvider):
    """Expose only cards accepted by the human review overlay."""

    def __init__(
        self,
        *,
        provider: MemoryProvider,
        review_dir: str | Path,
        review_mode: str = "mandatory",
    ) -> None:
        if review_mode not in {"mandatory", "auto"}:
            raise ValueError("review_mode must be 'mandatory' or 'auto'")
        self.provider = provider
        self.review_dir = Path(review_dir)
        self.review_mode = review_mode

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        selection = await self.provider.select_cards(
            program,
            task_description=task_description,
            metrics_description=metrics_description,
        )
        cards: list[str] = []
        card_ids: list[str] = []
        pair_count = min(len(selection.card_ids), len(selection.cards))
        if len(selection.card_ids) != len(selection.cards):
            logger.warning(
                "[ReviewedMemory] Selector returned {} IDs for {} cards; "
                "using only the {} attributable pairs",
                len(selection.card_ids),
                len(selection.cards),
                pair_count,
            )
        for card_id, card in zip(
            selection.card_ids[:pair_count], selection.cards[:pair_count]
        ):
            try:
                review_path = find_item(self.review_dir / "memory", card_id)
                review = load_item(review_path)
            except FileNotFoundError:
                # Auto mode may create approvals, but it must never treat a
                # missing review sidecar as approval. The sidecar binds the
                # visible text to a validated source execution card.
                continue
            source_card = review.get("source_card")
            if not isinstance(source_card, dict) or not _card_has_valid_result(
                source_card, "fitness"
            ):
                continue
            if review.get("status") == "approved":
                cards.append(card)
                card_ids.append(card_id)
            elif review.get("status") == "redacted":
                override = review.get("content_override")
                cards.append(override if isinstance(override, str) else card)
                card_ids.append(card_id)
        return MemorySelection(cards=cards, card_ids=card_ids)

    def refresh(self) -> None:
        refresh = getattr(self.provider, "refresh", None)
        if callable(refresh):
            refresh()


class ReviewedMemoryPostStepHook:
    """Materialize exact execution cards and review changed cards."""

    def __init__(
        self,
        *,
        storage: ProgramStorage,
        checkpoint_dir: str | Path,
        approved_checkpoint_dir: str | Path,
        review_dir: str | Path,
        review_mode: str = "mandatory",
        poll_interval_seconds: float = 2.0,
        memory_provider: MemoryProvider | None = None,
        primary_key: str = "fitness",
        task_description: str = "",
        source_repo: str | Path | None = None,
    ) -> None:
        if review_mode not in {"mandatory", "auto"}:
            raise ValueError("review_mode must be 'mandatory' or 'auto'")
        self.storage = storage
        self.checkpoint_dir = Path(checkpoint_dir)
        self.approved_checkpoint_dir = Path(approved_checkpoint_dir)
        self.review_dir = Path(review_dir)
        self.review_mode = review_mode
        self.poll_interval_seconds = poll_interval_seconds
        self.memory_provider = memory_provider
        self.primary_key = primary_key
        self.task_description = task_description
        self.source_repo = source_repo
        self._card_hashes: dict[str, str] = {}

    async def __call__(self) -> None:
        await self._materialize_program_cards()
        cards = self._load_cards()
        queued: list[Path] = []
        for card_id, card in cards.items():
            serialized = json.dumps(card, sort_keys=True, default=str)
            fingerprint = sha256(serialized.encode("utf-8")).hexdigest()
            if self._card_hashes.get(card_id) == fingerprint:
                continue
            self._card_hashes[card_id] = fingerprint
            review_path = self.review_dir / "memory" / f"{card_id}.json"
            existing = load_item(review_path) if review_path.exists() else None
            if existing is not None and existing.get("source_hash") == fingerprint:
                continue
            card_is_valid = _card_has_valid_result(card, self.primary_key)
            item = {
                "id": card_id,
                "kind": "program" if card.get("category") == "program" else "memory",
                "status": (
                    "rejected"
                    if not card_is_valid
                    else "approved"
                    if self.review_mode == "auto"
                    else "pending"
                ),
                "source_hash": fingerprint,
                "source_card": card,
                "field_overrides": {},
                "reviewer_note": (
                    "Rejected automatically: execution is invalid or non-scientific."
                    if not card_is_valid
                    else "Automatically approved by configured policy."
                    if self.review_mode == "auto"
                    else None
                ),
                "reviewed_at": None,
            }
            write_item(review_path, item)
            if item["status"] == "pending":
                queued.append(review_path)

        for review_path in queued:
            await self._wait_for_decision(review_path)
        self._publish_approved_cards(cards)
        refresh = getattr(self.memory_provider, "refresh", None)
        if callable(refresh):
            refresh()

    async def _materialize_program_cards(self) -> None:
        """Persist one exact ProgramCard for every attempt, including failures."""

        programs = await self.storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        if not programs:
            return
        by_id = {program.id: program for program in programs}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        store = CardStore(index_file=self.checkpoint_dir / "api_index.json")
        for program in programs:
            parent = next(
                (
                    by_id[parent_id]
                    for parent_id in program.lineage.parents
                    if parent_id in by_id
                ),
                None,
            )
            card = program_to_card(
                program,
                parent=parent,
                primary_key=self.primary_key,
                task_description=self.task_description,
                source_repo=self.source_repo,
            )
            store.put(card.id, card)
        store.persist()

    def _publish_approved_cards(self, cards: dict[str, dict[str, Any]]) -> None:
        """Atomically build the only memory index visible to the ideator."""

        approved: dict[str, dict[str, Any]] = {}
        for card_id, card in cards.items():
            if not _card_has_valid_result(card, self.primary_key):
                continue
            try:
                review_path = find_item(self.review_dir / "memory", card_id)
                review = load_item(review_path)
            except FileNotFoundError:
                continue
            if review.get("status") not in {"approved", "redacted"}:
                continue
            visible_card = dict(card)
            if review.get("status") == "redacted":
                overrides = review.get("field_overrides", {})
                if isinstance(overrides, dict):
                    visible_card.update(overrides)
            approved[card_id] = normalize_memory_card(
                visible_card, fallback_id=card_id
            ).model_dump()

        self.approved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_item(
            self.approved_checkpoint_dir / "api_index.json",
            {
                "entity_by_card_id": {},
                "entity_version_by_entity": {},
                "memory_cards": approved,
            },
        )

    def _load_cards(self) -> dict[str, dict[str, Any]]:
        index_path = self.checkpoint_dir / "api_index.json"
        if not index_path.exists():
            logger.info("[ReviewedMemory] No memory index at {}", index_path)
            return {}
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        cards = payload.get("memory_cards", {})
        if not isinstance(cards, dict):
            raise ValueError(f"Invalid memory card index: {index_path}")
        return {
            str(card_id): card
            for card_id, card in cards.items()
            if isinstance(card, dict)
        }

    async def _wait_for_decision(self, review_path: Path) -> None:
        while True:
            status = load_item(review_path).get("status")
            if status in {"approved", "rejected", "redacted"}:
                return
            await asyncio.sleep(self.poll_interval_seconds)
