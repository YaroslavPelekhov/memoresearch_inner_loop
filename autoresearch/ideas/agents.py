"""Codex calls used by the idea-level research loop."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from autoresearch.ideas.models import Idea, Implementation, ResearchTask, Takeaway
from gigaevo.llm.command_cli import CommandCLIChatModel


class JSONAgent(Protocol):
    async def call(
        self, *, stage: str, system: str, payload: dict[str, Any]
    ) -> Any: ...


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, str | dict)
        )
    return str(content)


def _json_value(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Agent response did not contain JSON")


class CodexJSONAgent:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        model: str,
        timeout: float,
        usage_log_path: str | Path,
    ) -> None:
        root = Path(repo_root).resolve()
        self._model = CommandCLIChatModel(
            command=[
                str(root / "tools/codex-proxy"),
                "exec",
                "--json",
                "--model",
                model,
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-",
            ],
            cwd=root,
            timeout=timeout,
            model_name=model,
            provider_name="codex",
            stdin_prompt=True,
            json_output=True,
            usage_log_path=usage_log_path,
        )

    async def call(self, *, stage: str, system: str, payload: dict[str, Any]) -> Any:
        response = await self._model.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"Stage: {stage}\nReturn JSON only.\n\n"
                        + json.dumps(payload, indent=2, sort_keys=True)
                    )
                ),
            ]
        )
        return _json_value(_response_text(response))


async def generate_ideas(
    agent: JSONAgent,
    *,
    task: ResearchTask,
    ideas: list[Idea],
    takeaways: list[Takeaway],
) -> list[dict[str, Any]]:
    result = await agent.call(
        stage="idea_generation",
        system=(
            "You generate research ideas, not code mutations. Use the existing idea "
            "catalog and idea-level takeaways as the research history. New ideas may "
            "combine several prior ideas and such combinations are encouraged when "
            "their mechanism is coherent. Do not merely restate an implementation "
            "detail. Each idea must be falsifiable and narrow enough for one fixed-budget "
            "implementation evolution. Do not infer conclusions from raw run data; only "
            "use the supplied takeaways."
        ),
        payload={
            "task": task.model_dump(mode="json"),
            "idea_catalog": [idea.model_dump(mode="json") for idea in ideas],
            "takeaway_bank": [item.model_dump(mode="json") for item in takeaways],
            "required_output": {
                "proposals": [
                    {
                        "title": "string",
                        "hypothesis": "string",
                        "mechanism": "string",
                        "implementation_direction": "string",
                        "success_criteria": ["string"],
                        "risks": ["string"],
                        "source_idea_ids": ["existing idea id"],
                        "takeaway_ids": ["existing takeaway id"],
                    }
                ]
            },
            "proposal_count": task.proposal_count,
        },
    )
    if not isinstance(result, dict) or not isinstance(result.get("proposals"), list):
        raise ValueError("Idea generator must return an object with proposals")
    proposals = result["proposals"]
    if len(proposals) != task.proposal_count:
        raise ValueError(
            f"Idea generator returned {len(proposals)} proposals; expected {task.proposal_count}"
        )
    return proposals


async def select_idea(
    agent: JSONAgent,
    *,
    task: ResearchTask,
    proposals: list[Idea],
    ideas: list[Idea],
    takeaways: list[Takeaway],
) -> tuple[str, str]:
    result = await agent.call(
        stage="idea_filter",
        system=(
            "Select one research idea for the next implementation evolution. Prefer a "
            "clear mechanism, useful combinations of earlier ideas, novelty relative to "
            "the catalog, feasibility inside the mutable GDN file, and evidence value. "
            "Do not select an idea that changes the fixed Transformer or training contract."
        ),
        payload={
            "task": task.model_dump(mode="json"),
            "proposals": [item.model_dump(mode="json") for item in proposals],
            "existing_idea_catalog": [item.model_dump(mode="json") for item in ideas],
            "takeaway_bank": [item.model_dump(mode="json") for item in takeaways],
            "required_output": {"selected_idea_id": "idea id", "reason": "string"},
        },
    )
    if not isinstance(result, dict):
        raise ValueError("Idea filter must return an object")
    selected = str(result.get("selected_idea_id", ""))
    reason = str(result.get("reason", "")).strip()
    if selected not in {item.id for item in proposals} or not reason:
        raise ValueError("Idea filter selected an unknown idea or omitted its reason")
    return selected, reason


async def select_parent(
    agent: JSONAgent,
    *,
    task: ResearchTask,
    idea: Idea,
    candidates: list[Implementation],
    takeaways: list[Takeaway],
) -> tuple[str, str]:
    result = await agent.call(
        stage="implementation_parent_selection",
        system=(
            "Choose the high-performing existing implementation that is the most suitable "
            "base for this idea. Compatibility with the new mechanism matters more than a "
            "tiny fitness difference. You may choose the canonical implementation when "
            "prior evolved mechanisms would confound or obstruct the idea."
        ),
        payload={
            "task": task.model_dump(mode="json"),
            "idea": idea.model_dump(mode="json"),
            "candidate_implementations": [
                item.model_dump(mode="json") for item in candidates
            ],
            "relevant_takeaways": [item.model_dump(mode="json") for item in takeaways],
            "required_output": {
                "implementation_id": "candidate id",
                "reason": "string",
            },
        },
    )
    if not isinstance(result, dict):
        raise ValueError("Parent selector must return an object")
    selected = str(result.get("implementation_id", ""))
    reason = str(result.get("reason", "")).strip()
    if selected not in {item.id for item in candidates} or not reason:
        raise ValueError("Parent selector selected an unknown implementation")
    return selected, reason


async def synthesize_takeaways(
    agent: JSONAgent,
    *,
    task: ResearchTask,
    idea: Idea,
    version: int,
    evidence: list[dict[str, Any]],
    existing_takeaways: list[Takeaway],
) -> list[dict[str, Any]]:
    result = await agent.call(
        stage="idea_takeaway_synthesis",
        system=(
            "Synthesize idea-level conclusions from the complete implementation lineage. "
            "Treat the structured parent_commit, changed_files, and lineage_basis fields "
            "as authoritative repository evidence. A reflection may describe a diff "
            "relative to the evolution seed itself and must not override those fields. "
            "Separate scientific evidence from implementation and infrastructure failures. "
            "Include negative, invalid, and contradictory attempts. A best single run is "
            "not proof; use uncertainty when replication or a noise-aware comparison is "
            "missing. A 1024-step screen is preliminary evidence used for ranking, "
            "not mechanism support. Only a cited successful 4096-step confirmation "
            "may justify supported_mechanism or refuted_mechanism. Use "
            "operational_feasibility for construction, memory, dtype, finite-gradient, "
            "or execution claims. Do not repeat an infrastructure conclusion already "
            "listed in existing_takeaways. Every conclusion must cite supplied "
            "implementation IDs."
        ),
        payload={
            "task": task.model_dump(mode="json"),
            "idea": idea.model_dump(mode="json"),
            "idea_version": version,
            "implementation_evidence": evidence,
            "existing_takeaways": [
                item.model_dump(mode="json") for item in existing_takeaways
            ],
            "required_output": {
                "takeaways": [
                    {
                        "kind": "supported_mechanism | refuted_mechanism | operational_feasibility | implementation_lesson | infrastructure_failure | uncertainty",
                        "claim": "string",
                        "scope_and_conditions": "string",
                        "confidence": "low | medium | high",
                        "supporting_implementation_ids": ["implementation id"],
                        "contradicting_implementation_ids": ["implementation id"],
                        "recommended_next_action": "string",
                        "deduplication_key": "short stable semantic key",
                    }
                ]
            },
        },
    )
    if not isinstance(result, dict) or not isinstance(result.get("takeaways"), list):
        raise ValueError("Takeaway synthesizer must return an object with takeaways")
    takeaways = result["takeaways"]
    if not takeaways:
        raise ValueError("Takeaway synthesizer must return at least one takeaway")
    return takeaways
