from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import combinations
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any

from loguru import logger
import tiktoken

from gigaevo.llm.codex_usage import codex_usage_scope, response_usage
from gigaevo.programs.program import Program
from gigaevo.repo_harness.candidate_card import RepoCandidateCardBuilder
from gigaevo.repo_harness.descriptors import (
    ACTIVE_ARCHIVE_ROLES,
    INACTIVE_ARCHIVE_ROLES,
    REPO_ARCHIVE_ROLES_METADATA_KEY,
)
from gigaevo.repo_harness.manifest import RepoCandidateManifest

_DEFAULT_PROMPT_TOKEN_ENCODING = "o200k_base"
_PROMPT_BUDGET_CHARS_PER_TOKEN = 3


class ParentSelection(list[Program]):
    """A selected parent set plus optional selector decision metadata."""

    def __init__(
        self,
        parents: list[Program],
        *,
        selection_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parents)
        self.selection_metadata = selection_metadata or {}


class ParentSelector(ABC):
    """Abstract base class for selecting parents for mutation."""

    @abstractmethod
    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create an iterator that yields parent selections.

        Args:
            available_parents: List of programs available for selection

        Returns:
            Iterator that yields selected parents for mutation
        """


class RandomParentSelector(ParentSelector):
    """Randomly selects parents from the available pool."""

    def __init__(self, num_parents: int = 1):
        if num_parents < 1:
            raise ValueError(f"num_parents must be at least 1, got {num_parents}")
        self.num_parents = num_parents

    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create iterator for random parent selection.

        Yields infinite random selections of parents (consumer controls limit via break).
        If fewer parents are available than requested, returns all available.
        """
        if not available_parents:
            return
        while True:
            yield random.sample(
                available_parents, min(self.num_parents, len(available_parents))
            )


class AllCombinationsParentSelector(ParentSelector):
    """Exhaustively iterates through all combinations of parents."""

    def __init__(self, num_parents: int = 1):
        if num_parents < 1:
            raise ValueError(f"num_parents must be at least 1, got {num_parents}")
        self.num_parents = num_parents

    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create iterator for all combinations of parents.

        Yields all possible combinations of the requested number of parents.
        Combinations are shuffled for randomness.
        If fewer parents are available than requested, yields all available parents once.
        """
        if not available_parents:
            return

        parents_copy = available_parents.copy()
        random.shuffle(parents_copy)

        if len(parents_copy) < self.num_parents:
            logger.info(
                f"[AllCombinationsParentSelector] Only {len(parents_copy)} parents available, yielding all"
            )
            yield parents_copy
            return

        for combo in combinations(parents_copy, self.num_parents):
            yield list(combo)


class MetaRepoParentSelector(ParentSelector):
    """Select repo parent sets with an LLM meta-agent and heuristic fallback.

    This selector is intended for Git-backed repo candidates. When ``llm`` is
    provided, it asks a meta-agent to inspect elite summaries, reflections, and
    compact own-improvement diffs, then return ranked parent sets. The
    deterministic score/diversity ranker remains as a fallback and tie-breaker.
    """

    _REFLECTION_KEYS = (
        "reflection",
        "diff_stat",
        "name_status",
        "selection_summary",
        "changed_files",
    )
    _RESPONSE_SET_KEYS = ("parent_sets", "selections", "selected_parent_sets")
    _RESPONSE_IDS_KEYS = (
        "parent_ids",
        "candidate_ids",
        "selected_parent_ids",
        "selected_candidate_ids",
        "parents",
        "program_ids",
        "ids",
    )

    def __init__(
        self,
        num_parents: int = 2,
        primary_key: str = "fitness",
        higher_is_better: bool = True,
        llm: Any | None = None,
        source_repo: str | Path | None = None,
        fitness_weight: float = 1.0,
        improvement_weight: float = 0.6,
        complementarity_weight: float = 0.35,
        novelty_weight: float = 0.08,
        max_candidate_count: int = 12,
        max_ranked_sets: int = 12,
        max_prompt_chars: int | None = None,
        max_prompt_tokens: int = 150000,
        prompt_token_encoding: str = _DEFAULT_PROMPT_TOKEN_ENCODING,
        max_diff_chars: int = 12000,
        max_reflection_chars: int = 5000,
        max_pairing_history_items: int = 20,
        max_pairing_history_reflection_chars: int = 2000,
        active_archive_roles: list[str] | None = None,
        excluded_archive_roles: list[str] | None = None,
        fail_open: bool = True,
        task_description: str = "",
    ):
        if num_parents < 1:
            raise ValueError(f"num_parents must be at least 1, got {num_parents}")
        if max_candidate_count < 1:
            raise ValueError(
                f"max_candidate_count must be at least 1, got {max_candidate_count}"
            )
        if max_ranked_sets < 1:
            raise ValueError(
                f"max_ranked_sets must be at least 1, got {max_ranked_sets}"
            )
        if max_pairing_history_items < 0:
            raise ValueError(
                "max_pairing_history_items must be non-negative, "
                f"got {max_pairing_history_items}"
            )
        if max_pairing_history_reflection_chars < 1:
            raise ValueError(
                "max_pairing_history_reflection_chars must be at least 1, "
                f"got {max_pairing_history_reflection_chars}"
            )
        if max_prompt_chars is not None:
            max_prompt_tokens = max(
                1000, int(max_prompt_chars) // _PROMPT_BUDGET_CHARS_PER_TOKEN
            )
        if max_prompt_tokens < 1000:
            raise ValueError(
                f"max_prompt_tokens must be at least 1000, got {max_prompt_tokens}"
            )
        self.num_parents = num_parents
        self.primary_key = primary_key
        self.higher_is_better = higher_is_better
        self.task_description = str(task_description).strip()
        self.llm = llm
        self.source_repo = (
            Path(source_repo).expanduser().resolve() if source_repo else None
        )
        self.fitness_weight = fitness_weight
        self.improvement_weight = improvement_weight
        self.complementarity_weight = complementarity_weight
        self.novelty_weight = novelty_weight
        self.max_candidate_count = max_candidate_count
        self.max_ranked_sets = max_ranked_sets
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.prompt_token_encoding = str(prompt_token_encoding).strip()
        try:
            self._prompt_tokenizer = tiktoken.get_encoding(
                self.prompt_token_encoding
            )
        except ValueError as exc:
            raise ValueError(
                f"unknown prompt_token_encoding {self.prompt_token_encoding!r}"
            ) from exc
        self.max_diff_chars = max_diff_chars
        self.max_reflection_chars = max_reflection_chars
        self.max_pairing_history_items = max_pairing_history_items
        self.max_pairing_history_reflection_chars = (
            max_pairing_history_reflection_chars
        )
        self.active_archive_roles = set(active_archive_roles or ACTIVE_ARCHIVE_ROLES)
        self.excluded_archive_roles = set(excluded_archive_roles or INACTIVE_ARCHIVE_ROLES)
        self._population_history: list[Program] = []
        self.fail_open = fail_open
        self._card_builder = RepoCandidateCardBuilder(
            primary_key=self.primary_key,
            higher_is_better=self.higher_is_better,
            source_repo=self.source_repo,
            metrics_context=None,
            max_diff_chars=self.max_diff_chars,
            max_reflection_chars=self.max_reflection_chars,
            # The selector needs enough benchmark diagnostics to assess pairing,
            # but less than the long-horizon archive curator.
            max_raw_feedback_chars=2500,
        )

    def set_population_history(self, programs: list[Program]) -> None:
        """Provide recent population context for pairing-history prompt snippets."""
        self._population_history = list(programs)

    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        if not available_parents:
            return
        available_parents = self._active_parent_pool(available_parents)

        if self.llm is not None:
            yielded: set[tuple[str, ...]] = set()
            try:
                for selection in self._llm_parent_selections(available_parents):
                    key = tuple(p.id for p in selection)
                    yielded.add(key)
                    yield selection
            except Exception as exc:
                if not self.fail_open:
                    raise
                logger.warning(
                    "[MetaRepoParentSelector] LLM parent selection failed; "
                    "falling back to heuristic selector: {}",
                    exc,
                )

            for parents in self._heuristic_parent_sets(available_parents):
                key = tuple(p.id for p in parents)
                if key in yielded:
                    continue
                yield ParentSelection(
                    parents,
                    selection_metadata={
                        "source": "heuristic_fallback",
                        "selected_parent_ids": [p.id for p in parents],
                        "selected_parent_short_ids": [p.short_id for p in parents],
                    },
                )
            return

        yield from self._heuristic_parent_sets(available_parents)

    def _heuristic_parent_sets(
        self, available_parents: list[Program]
    ) -> Iterator[ParentSelection]:
        parents = sorted(
            available_parents,
            key=lambda p: (-self._individual_score(p), p.short_id),
        )
        if len(parents) <= self.num_parents:
            yield ParentSelection(
                parents,
                selection_metadata={
                    "source": "heuristic",
                    "selected_parent_ids": [p.id for p in parents],
                    "selected_parent_short_ids": [p.short_id for p in parents],
                },
            )
            return

        if self.num_parents == 1:
            for parent in parents:
                yield ParentSelection(
                    [parent],
                    selection_metadata={
                        "source": "heuristic",
                        "selected_parent_ids": [parent.id],
                        "selected_parent_short_ids": [parent.short_id],
                    },
                )
            return

        ranked = sorted(
            combinations(parents, self.num_parents),
            key=lambda combo: (
                -self._set_score(combo),
                tuple(p.short_id for p in combo),
            ),
        )
        for combo in ranked:
            selection = sorted(
                combo,
                key=lambda p: (-self._individual_score(p), p.short_id),
            )
            yield ParentSelection(
                selection,
                selection_metadata={
                    "source": "heuristic",
                    "selected_parent_ids": [p.id for p in selection],
                    "selected_parent_short_ids": [p.short_id for p in selection],
                    "heuristic_set_score": self._set_score(tuple(selection)),
                },
            )

    def _llm_parent_selections(
        self, available_parents: list[Program]
    ) -> Iterator[ParentSelection]:
        candidate_pool = self._candidate_pool(available_parents)
        if len(candidate_pool) <= self.num_parents:
            yield ParentSelection(
                candidate_pool,
                selection_metadata={
                    "source": "llm_bypassed_all_available",
                    "selected_parent_ids": [p.id for p in candidate_pool],
                    "selected_parent_short_ids": [p.short_id for p in candidate_pool],
                    "rationale": "All available parents are needed to satisfy num_parents.",
                },
            )
            return

        prompt = self._render_llm_prompt(candidate_pool, len(available_parents))
        with codex_usage_scope(
            source="parent_selection",
            generation=max(program.generation for program in available_parents) + 1,
            candidate_program_ids=[program.id for program in candidate_pool],
        ):
            response = self.llm.invoke(prompt)
        codex_usage = response_usage(response)
        parsed = self._extract_json(self._response_text(response))
        decisions = self._normalize_llm_decisions(parsed, candidate_pool)
        if not decisions:
            raise ValueError("LLM did not return any valid parent sets")

        for rank, decision in enumerate(decisions[: self.max_ranked_sets], start=1):
            parents = decision["parents"]
            yield ParentSelection(
                parents,
                selection_metadata={
                    "source": "llm_meta_agent",
                    "rank": rank,
                    "selected_parent_ids": [p.id for p in parents],
                    "selected_parent_short_ids": [p.short_id for p in parents],
                    "rationale": decision.get("rationale"),
                    "analysis": decision.get("analysis"),
                    "codex_usage": codex_usage,
                },
            )

    def _candidate_pool(self, available_parents: list[Program]) -> list[Program]:
        parents = self._active_parent_pool(available_parents)
        return sorted(
            parents,
            key=lambda p: (-self._individual_score(p), p.short_id),
        )[: self.max_candidate_count]

    def _active_parent_pool(self, available_parents: list[Program]) -> list[Program]:
        annotated = [
            parent
            for parent in available_parents
            if isinstance(parent.metadata.get(REPO_ARCHIVE_ROLES_METADATA_KEY), list)
        ]
        if not annotated:
            return available_parents
        active = []
        for parent in available_parents:
            roles = set(map(str, parent.metadata.get(REPO_ARCHIVE_ROLES_METADATA_KEY) or []))
            if roles & self.excluded_archive_roles:
                continue
            if roles & self.active_archive_roles:
                active.append(parent)
        return active or available_parents

    def _render_llm_prompt(
        self, candidate_pool: list[Program], available_parent_count: int
    ) -> str:
        aliases = {
            program.id: f"P{idx}"
            for idx, program in enumerate(candidate_pool, start=1)
        }
        context_by_id = {
            program.id: program for program in [*self._population_history, *candidate_pool]
        }
        diff_budget = max(
            1000,
            min(
                self.max_diff_chars,
                self._prompt_char_budget_hint()
                // max(len(candidate_pool), 1)
                // 4,
            ),
        )
        candidates = [
            self._candidate_summary(
                program,
                idx=idx,
                diff_budget=diff_budget,
                parent=next(
                    (
                        context_by_id[parent_id]
                        for parent_id in program.lineage.parents
                        if parent_id in context_by_id
                    ),
                    None,
                ),
                archive=[
                    candidate
                    for candidate in candidate_pool
                    if candidate.id != program.id
                ],
                aliases=aliases,
            )
            for idx, program in enumerate(candidate_pool, start=1)
        ]
        pairing_history = self._pairing_history(candidate_pool)
        payload = {
            "task": "Select ranked parent sets for the next repo-harness mutation.",
            "task_description": self.task_description
            or "No task description provided.",
            "selection_rules": {
                "num_parents_per_set": min(self.num_parents, len(candidate_pool)),
                "max_ranked_sets": self.max_ranked_sets,
                "mutation_portfolio": {
                    "available_slots": self.max_ranked_sets,
                    "allocation_decided_by": "selector_agent",
                    "allocation": (
                        "Decide how many available slots should exploit versus "
                        "explore from the current evidence. When at least two slots "
                        "are available, include both modes unless the candidate pool "
                        "makes one mode impossible. Increase exploration when the "
                        "best metric has plateaued, recent pairings repeat neutral "
                        "outcomes, or behavior fingerprints have collapsed. Increase "
                        "exploitation when recent improvements expose credible follow-up "
                        "mechanisms. With one slot, choose the mode with the strongest "
                        "evidence."
                    ),
                    "allocation_evidence": [
                        "current primary metrics and own-parent deltas",
                        "behavioral and architectural diversity in the active archive",
                        "recent pairing_history outcomes and repeated combinations",
                        "whether the best metric is improving or plateaued",
                    ],
                },
                "order_matters": (
                    "The first parent is the primary worktree base. Later parents "
                    "are secondary sources whose ideas/diffs will be merged into it."
                ),
                "prefer": [
                    "for exploitation slots, high current primary metric and "
                    "positive metric delta from the candidate's own parent",
                    "for exploration slots, promising lower-scoring primary bases "
                    "with genuinely different architectures, algorithms, or "
                    "behavior fingerprints",
                    "implementation diffs that plausibly caused score increases or "
                    "opened a distinct search basin",
                    "complementary fixes in different files, features, or failure clusters",
                    "pairings whose previous outcomes suggest a useful staged synthesis",
                    "under-explored parents with credible future value, even when "
                    "their current score is not comparable to the champion",
                ],
                "avoid": [
                    "near-duplicate parents unless one is clearly the best base",
                    "using the same champion or behavior fingerprint as the primary "
                    "base for every mutation slot",
                    "putting all architecturally distinct lower-scoring candidates "
                    "only in secondary positions; exploration requires distinct "
                    "primary bases so their search basins can continue",
                    "combinations where secondary parents add no distinct mechanism",
                    "repeating pairings whose prior child did not improve unless "
                    "new evidence makes the pairing worth revisiting",
                    "parents that appear to improve by hardcoding evaluator quirks",
                ],
            },
            "metric_context": {
                "primary_key": self.primary_key,
                "higher_is_better": self.higher_is_better,
            },
            "available_parent_count": available_parent_count,
            "included_candidate_count": len(candidate_pool),
            "candidates": candidates,
            "pairing_history": pairing_history,
        }
        return self._selection_prompt_text(payload)

    def _selection_prompt_text(self, payload: dict[str, Any]) -> str:
        task_description = self._clip(
            str(payload.get("task_description") or "No task description provided."),
            min(12000, max(1000, self._prompt_char_budget_hint() // 6)),
        )
        prefix = (
            "You are the repo-harness meta-agent inside a GigaEvo loop.\n\n"
            "Your job is not to mutate code. Your job is to inspect elite repo "
            "candidates and choose which parents the next mutation agent should "
            "combine. Look for high-quality bases, complementary evidence, and "
            "promising but underexplored candidates. Compare metrics, behavioral "
            "and semantic fingerprints, reflections, compact own-improvement diffs, "
            "and prior pairing outcomes. Equal behavior_fingerprint values mean the "
            "benchmark observed identical behavioral outputs. Treat the requested "
            "parent sets as one generation-level mutation portfolio. Decide the "
            "exploration/exploitation allocation yourself from the number of available "
            "slots, fitness progress, archive diversity, and recent pairing history. "
            "When at least two slots are available, include both modes unless the "
            "candidate pool makes that impossible. Exploitation should continue the "
            "strongest credible mechanisms; exploration should continue promising "
            "solutions with different architectures, algorithms, or behavioral "
            "outputs even when their current score is lower. An exploration set must "
            "put such a candidate first as the "
            "primary worktree base; making it only a secondary source does not "
            "continue its search basin. Mix complementary solution families and do "
            "not assign every primary slot to the same champion phenotype. Do not "
            "prescribe the "
            "code change or mutation strategy; the mutation agent will inspect the "
            "worktree and decide what to implement. Use the task description below "
            "as the authoritative context for judging candidate relevance and "
            "compatibility.\n\n"
            "Task description:\n"
            f"{task_description}\n\n"
            "Return exactly one JSON object and no markdown, prose, comments, or "
            "code fences. The JSON object must use this exact schema:\n"
            "{\n"
            '  "analysis": "brief evidence-based candidate fit summary",\n'
            '  "parent_sets": [\n'
            "    {\n"
            '      "parent_ids": ["P1", "P3"],\n'
            '      "rationale": "why these candidates are a good parent set"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Critical output rules:\n"
            "- Use the key parent_sets for the ranked list.\n"
            "- Use the key parent_ids inside each parent set.\n"
            "- Do not use candidate_ids, selected_parent_ids, parents, program_ids, "
            "or any other key in place of parent_ids.\n"
            "- parent_ids values must be candidate_id strings from the candidate "
            "data, such as P1 or P3, not program_id or short_id values.\n"
            "- Each parent_ids list must contain exactly the requested "
            "num_parents_per_set count.\n"
            "- Return one parent set per available mutation slot whenever enough "
            "valid combinations exist.\n"
            "- Decide and follow the mutation_portfolio exploitation/exploration "
            "allocation; "
            "rank a balanced portfolio rather than sorting every set only by "
            "expected immediate fitness.\n"
            "- Rank the portfolio slots and return at most max_ranked_sets items.\n\n"
            "- Do not include a mutation strategy, implementation plan, patch idea, "
            "or instructions for what code to change.\n\n"
            "Elite candidate data:\n"
        )
        prompt_payload = json.loads(json.dumps(payload, default=str))
        # The authoritative task is already in the instruction prefix.
        prompt_payload.pop("task_description", None)
        prompt = prefix + json.dumps(
            prompt_payload,
            indent=2,
            sort_keys=True,
            default=str,
            ensure_ascii=True,
        )
        if self._count_prompt_tokens(prompt) <= self.max_prompt_tokens:
            return prompt

        self._compress_selection_payload(prompt_payload, emergency=False)
        prompt = prefix + json.dumps(
            prompt_payload, separators=(",", ":"), sort_keys=True, default=str
        )
        if self._count_prompt_tokens(prompt) <= self.max_prompt_tokens:
            return prompt

        self._compress_selection_payload(prompt_payload, emergency=True)
        prompt = prefix + json.dumps(
            prompt_payload, separators=(",", ":"), sort_keys=True, default=str
        )
        token_count = self._count_prompt_tokens(prompt)
        if token_count > self.max_prompt_tokens:
            raise ValueError(
                "parent selector prompt exceeds max_prompt_tokens after structured "
                f"compression ({token_count} > {self.max_prompt_tokens} tokens; "
                f"{len(prompt)} characters)"
            )
        return prompt

    def _compress_selection_payload(
        self,
        payload: dict[str, Any],
        *,
        emergency: bool,
    ) -> None:
        reflection_limit = 800 if emergency else 1800
        diff_limit = 0 if emergency else 1800
        diagnostic_limit = 0 if emergency else 1200
        unit_limit = 12 if emergency else 32

        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                reflection = candidate.get("repo_reflection")
                if isinstance(reflection, dict):
                    for key in ("summary", "reflection"):
                        if isinstance(reflection.get(key), str):
                            reflection[key] = self._clip(
                                reflection[key], reflection_limit
                            )
                diff = candidate.get("own_improvement_diff")
                if isinstance(diff, dict) and isinstance(diff.get("diff"), str):
                    if diff_limit:
                        diff["diff"] = self._clip(diff["diff"], diff_limit)
                    else:
                        diff.pop("diff", None)
                evidence = candidate.get("benchmark_evidence")
                if isinstance(evidence, dict):
                    diagnostics = evidence.get("benchmark_specific_diagnostics")
                    if isinstance(diagnostics, dict):
                        if diagnostic_limit:
                            for key, value in list(diagnostics.items()):
                                serialized = json.dumps(value, default=str)
                                if len(serialized) > diagnostic_limit:
                                    diagnostics[key] = self._clip(
                                        serialized, diagnostic_limit
                                    )
                        else:
                            evidence.pop("benchmark_specific_diagnostics", None)
                    unit_results = evidence.get("unit_results")
                    if isinstance(unit_results, dict) and isinstance(
                        unit_results.get("items"), list
                    ):
                        items = unit_results["items"]
                        if len(items) > unit_limit:
                            unit_results["items"] = items[:unit_limit]
                            unit_results["truncated_count"] = max(
                                int(unit_results.get("truncated_count") or 0),
                                int(
                                    unit_results.get("observed_count") or len(items)
                                )
                                - unit_limit,
                            )

        history = payload.get("pairing_history")
        if not isinstance(history, list):
            return
        if emergency and len(history) > 8:
            del history[8:]
        for entry in history:
            if not isinstance(entry, dict):
                continue
            summary = entry.get("change_summary")
            if isinstance(summary, dict) and isinstance(summary.get("reflection"), str):
                summary["reflection"] = self._clip(
                    summary["reflection"], 400 if emergency else 800
                )
            rationale = entry.get("prior_pairing_hypothesis")
            if isinstance(rationale, str):
                entry["prior_pairing_hypothesis"] = self._clip(
                    rationale, 200 if emergency else 400
                )

    def _candidate_summary(
        self,
        program: Program,
        *,
        idx: int,
        diff_budget: int,
        parent: Program | None = None,
        archive: list[Program] | tuple[Program, ...] = (),
        aliases: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        summary = self._card_builder.build(
            program,
            candidate_id=f"P{idx}",
            parent=parent,
            archive=archive,
            origin="active_archive",
            diff_budget=diff_budget,
            program_aliases=aliases,
        )
        summary["heuristic_individual_score"] = self._individual_score(program)
        return summary

    def _pairing_history(self, candidate_pool: list[Program]) -> list[dict[str, Any]]:
        if self.max_pairing_history_items == 0:
            return []

        candidate_ids = {
            program.id: f"P{idx}" for idx, program in enumerate(candidate_pool, start=1)
        }
        source_programs = self._population_history or candidate_pool
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for program in source_programs:
            if program.id in seen:
                continue
            seen.add(program.id)
            selection = program.metadata.get("parent_selection")
            parent_ids = list(program.lineage.parents)
            selected_ids = self._metadata_list(program, "selected_parent_program_ids")
            if not selected_ids:
                selected_ids = self._selection_list(selection, "selected_parent_ids")
            if not parent_ids:
                parent_ids = selected_ids
            if not parent_ids:
                continue

            selected_short_ids = self._selection_list(
                selection, "selected_parent_short_ids"
            )
            reflection = program.metadata.get("repo_reflection")
            child_candidate_id = candidate_ids.get(program.id)
            parent_short_ids = selected_short_ids or [
                parent_id[:8] for parent_id in parent_ids
            ]
            entry: dict[str, Any] = {
                "child_candidate_id": child_candidate_id,
                "child_in_candidate_pool": child_candidate_id is not None,
                "child_short_id": program.short_id,
                "child_generation": program.lineage.generation,
                "child_iteration": program.iteration,
                "primary_metric_value": program.metrics.get(self.primary_key),
                "primary_metric_delta": self._primary_delta(program),
                "mutation": self._clip(
                    str(program.lineage.mutation or program.name or ""), 400
                ),
                "parents": {
                    "candidate_ids": [
                        candidate_ids.get(parent_id, parent_id[:8])
                        for parent_id in parent_ids
                    ],
                    "short_ids": parent_short_ids,
                },
                "changed_files": self._summary_changed_files(
                    program, self._manifest_for_program(program), reflection
                )[:32],
                "change_summary": self._change_summary(reflection),
            }
            if isinstance(selection, dict) and selection.get("rationale"):
                entry["prior_pairing_hypothesis"] = self._clip(
                    str(selection["rationale"]),
                    min(self.max_pairing_history_reflection_chars, 600),
                )
            entries.append(entry)

        entries.sort(
            key=lambda item: (
                int(item.get("child_generation") or 0),
                int(item.get("child_iteration") or 0),
                float(item.get("primary_metric_value") or 0.0),
            ),
            reverse=True,
        )
        return entries[: self.max_pairing_history_items]

    @staticmethod
    def _metadata_list(program: Program, key: str) -> list[str]:
        value = program.metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    @staticmethod
    def _selection_list(selection: Any, key: str) -> list[str]:
        if isinstance(selection, dict) and isinstance(selection.get(key), list):
            return [str(item) for item in selection[key]]
        return []

    def _change_summary(self, reflection: Any) -> dict[str, Any] | None:
        if not isinstance(reflection, dict):
            return None
        summary: dict[str, Any] = {}
        for key in (
            "diff_stat",
            "name_status",
            "reflection_status",
            "llm_skipped",
            "skip_reason",
        ):
            if key in reflection:
                summary[key] = reflection[key]
        attempt = reflection.get("attempt_record")
        if isinstance(attempt, dict) and attempt.get("verdict"):
            summary["verdict"] = self._clip(str(attempt["verdict"]), 100)
        text = reflection.get("reflection")
        if isinstance(text, str):
            summary["reflection"] = self._clip(
                text, min(self.max_pairing_history_reflection_chars, 1000)
            )
        return summary or None

    def _summary_changed_files(
        self,
        program: Program,
        manifest: RepoCandidateManifest | None,
        reflection: Any,
    ) -> list[str]:
        if manifest is not None and manifest.changed_files:
            return list(manifest.changed_files)
        if isinstance(reflection, dict) and isinstance(
            reflection.get("changed_files"), list
        ):
            return [str(path) for path in reflection["changed_files"]]
        for key in ("changed_files", "previous_changed_files"):
            value = program.metadata.get(key)
            if isinstance(value, list):
                return [str(path) for path in value]
        return []

    def _manifest_for_program(self, program: Program) -> RepoCandidateManifest | None:
        try:
            return RepoCandidateManifest.from_program(
                program, default_repo_path=self.source_repo
            )
        except Exception:
            if self.source_repo is None:
                return None
            try:
                return RepoCandidateManifest.from_metadata(
                    program, default_repo_path=self.source_repo
                )
            except Exception:
                return None

    def _normalize_llm_decisions(
        self, parsed: Any, candidate_pool: list[Program]
    ) -> list[dict[str, Any]]:
        candidate_map: dict[str, Program] = {}
        for idx, program in enumerate(candidate_pool, start=1):
            candidate_map[f"P{idx}"] = program
            candidate_map[f"p{idx}"] = program
            candidate_map[program.id] = program
            candidate_map[program.short_id] = program

        response_analysis = None
        raw_sets: Any = parsed
        if isinstance(parsed, dict):
            response_analysis = parsed.get("analysis")
            for key in self._RESPONSE_SET_KEYS:
                if key in parsed:
                    raw_sets = parsed[key]
                    break

        if isinstance(raw_sets, dict):
            raw_sets = [raw_sets]
        if not isinstance(raw_sets, list):
            return []

        expected_count = min(self.num_parents, len(candidate_pool))
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for item in raw_sets:
            parent_ids, rationale = self._decision_fields(item)
            parents: list[Program] = []
            for raw_id in parent_ids:
                key = str(raw_id).strip()
                program = candidate_map.get(key)
                if program is not None and program not in parents:
                    parents.append(program)
            if len(parents) != expected_count:
                logger.debug(
                    "[MetaRepoParentSelector] Ignoring invalid LLM parent set {}",
                    parent_ids,
                )
                continue
            key = tuple(parent.id for parent in parents)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "parents": parents,
                    "rationale": rationale,
                    "analysis": response_analysis,
                }
            )
        return normalized

    def _decision_fields(self, item: Any) -> tuple[list[Any], str | None]:
        if isinstance(item, list):
            return item, None
        if not isinstance(item, dict):
            return [], None
        parent_ids: list[Any] = []
        for key in self._RESPONSE_IDS_KEYS:
            value = item.get(key)
            if isinstance(value, list):
                parent_ids = value
                break
        return parent_ids, item.get("rationale")

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                else:
                    chunks.append(str(part))
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _extract_json(text: str) -> Any:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        decoder = json.JSONDecoder()
        for idx, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[idx:])
                return value
            except json.JSONDecodeError:
                continue
        raise ValueError("LLM response did not contain a JSON object or array")

    def _metric_value(self, program: Program) -> float:
        value = float(program.metrics.get(self.primary_key, 0.0))
        return value if self.higher_is_better else -value

    def _primary_delta(self, program: Program) -> float:
        reflection = program.metadata.get("repo_reflection")
        if isinstance(reflection, dict):
            for key in ("metrics_delta", "delta"):
                delta = reflection.get(key)
                if isinstance(delta, dict) and self.primary_key in delta:
                    return float(delta[self.primary_key]) * (
                        1.0 if self.higher_is_better else -1.0
                    )
        return 0.0

    def _individual_score(self, program: Program) -> float:
        child_count = max(program.lineage.child_count, 0)
        novelty = 1.0 / (1.0 + child_count)
        improvement = max(0.0, self._primary_delta(program))
        return (
            self.fitness_weight * self._metric_value(program)
            + self.improvement_weight * improvement
            + self.novelty_weight * novelty
        )

    def _set_score(self, programs: tuple[Program, ...]) -> float:
        individual = mean(self._individual_score(p) for p in programs)
        improvement = max(max(0.0, self._primary_delta(p)) for p in programs)
        return (
            individual
            + self.improvement_weight * improvement
            + self.complementarity_weight * self._complementarity(programs)
        )

    def _feature_tokens(self, program: Program) -> set[str]:
        tokens: set[str] = set()
        reflection = program.metadata.get("repo_reflection")
        if isinstance(reflection, dict):
            changed_files = reflection.get("changed_files")
            if isinstance(changed_files, list):
                for path in changed_files:
                    text = str(path)
                    tokens.add(f"file:{text}")
                    if "." in text:
                        tokens.add(f"ext:{text.rsplit('.', 1)[-1]}")
            for key in self._REFLECTION_KEYS:
                value = reflection.get(key)
                if value is None:
                    continue
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]{2,}", str(value).lower()
                ):
                    tokens.add(token)

        if not tokens:
            for key in ("changed_files", "previous_changed_files"):
                value = program.metadata.get(key)
                if isinstance(value, list):
                    tokens.update(f"file:{path}" for path in map(str, value))
        return tokens

    def _complementarity(self, programs: tuple[Program, ...]) -> float:
        feature_sets = [self._feature_tokens(p) for p in programs]
        distances: list[float] = []
        for left, right in combinations(feature_sets, 2):
            union = left | right
            if not union:
                distances.append(0.0)
                continue
            distances.append(1.0 - (len(left & right) / len(union)))
        return mean(distances) if distances else 0.0

    def _count_prompt_tokens(self, prompt: str) -> int:
        return len(self._prompt_tokenizer.encode_ordinary(prompt))

    def _prompt_char_budget_hint(self) -> int:
        return self.max_prompt_tokens * _PROMPT_BUDGET_CHARS_PER_TOKEN

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...<truncated>"
