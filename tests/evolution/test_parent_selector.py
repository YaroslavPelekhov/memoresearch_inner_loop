"""Tests for gigaevo/evolution/mutation/parent_selector.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gigaevo.evolution.mutation.parent_selector import (
    AllCombinationsParentSelector,
    MetaRepoParentSelector,
    ParentSelection,
    RandomParentSelector,
)
from gigaevo.programs.program import Program
from gigaevo.repo_harness.git_utils import commit_all, ensure_initial_commit, run_git
from gigaevo.repo_harness.manifest import RepoCandidateManifest


def _mock_programs(n: int) -> list:
    return [MagicMock(id=f"prog-{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# RandomParentSelector
# ---------------------------------------------------------------------------


class TestRandomParentSelector:
    def test_num_parents_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="num_parents must be at least 1"):
            RandomParentSelector(num_parents=0)

    def test_empty_list_yields_nothing(self) -> None:
        sel = RandomParentSelector(num_parents=2)
        results = list(sel.create_parent_iterator([]))
        assert results == []

    def test_yields_correct_count(self) -> None:
        sel = RandomParentSelector(num_parents=2)
        programs = _mock_programs(5)
        it = sel.create_parent_iterator(programs)
        first = next(it)
        assert len(first) == 2

    def test_yields_all_when_fewer_available(self) -> None:
        sel = RandomParentSelector(num_parents=5)
        programs = _mock_programs(2)
        it = sel.create_parent_iterator(programs)
        first = next(it)
        assert len(first) == 2

    def test_infinite_iterator(self) -> None:
        sel = RandomParentSelector(num_parents=1)
        programs = _mock_programs(3)
        it = sel.create_parent_iterator(programs)
        for _ in range(100):
            batch = next(it)
            assert len(batch) == 1
            assert batch[0] in programs


# ---------------------------------------------------------------------------
# AllCombinationsParentSelector
# ---------------------------------------------------------------------------


class TestAllCombinationsParentSelector:
    def test_num_parents_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="num_parents must be at least 1"):
            AllCombinationsParentSelector(num_parents=0)

    def test_empty_list_yields_nothing(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=2)
        results = list(sel.create_parent_iterator([]))
        assert results == []

    def test_all_combinations(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=2)
        programs = _mock_programs(3)
        combos = list(sel.create_parent_iterator(programs))
        # C(3,2) = 3
        assert len(combos) == 3
        for combo in combos:
            assert len(combo) == 2

    def test_fewer_parents_than_requested_yields_all_once(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=5)
        programs = _mock_programs(2)
        combos = list(sel.create_parent_iterator(programs))
        assert len(combos) == 1
        assert len(combos[0]) == 2

    def test_finite_iterator(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=1)
        programs = _mock_programs(4)
        combos = list(sel.create_parent_iterator(programs))
        # C(4,1) = 4
        assert len(combos) == 4


# ---------------------------------------------------------------------------
# MetaRepoParentSelector
# ---------------------------------------------------------------------------


def _repo_program(
    *,
    fitness: float,
    delta: float,
    reflection: str,
    changed_files: list[str],
) -> Program:
    program = Program(code="repo-manifest")
    program.add_metrics({"fitness": fitness})
    program.metadata["repo_reflection"] = {
        "changed_files": changed_files,
        "metrics_delta": {"fitness": delta},
        "reflection": reflection,
    }
    return program


class _FakeLLM:
    def __init__(self, payload: object):
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(self.payload))


class TestMetaRepoParentSelector:
    def test_num_parents_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="num_parents must be at least 1"):
            MetaRepoParentSelector(num_parents=0)

    def test_empty_list_yields_nothing(self) -> None:
        sel = MetaRepoParentSelector(num_parents=2)
        assert list(sel.create_parent_iterator([])) == []

    def test_prefers_complementary_high_potential_pair(self) -> None:
        ungron = _repo_program(
            fitness=0.70,
            delta=0.20,
            reflection="Implemented ungron parser and assignment reconstruction.",
            changed_files=["main.go"],
        )
        ungron_clone = _repo_program(
            fitness=0.69,
            delta=0.19,
            reflection="Another ungron parser refinement with similar behavior.",
            changed_files=["main.go"],
        )
        url = _repo_program(
            fitness=0.68,
            delta=0.18,
            reflection="Implemented URL fetch, proxy, timeout, and redirect handling.",
            changed_files=["fetch.go"],
        )

        sel = MetaRepoParentSelector(num_parents=2)
        first = next(sel.create_parent_iterator([ungron, ungron_clone, url]))

        assert {p.id for p in first} == {ungron.id, url.id}
        assert first[0] is ungron

    def test_single_parent_mode_is_deterministic_by_meta_score(self) -> None:
        weak = _repo_program(
            fitness=0.2,
            delta=0.1,
            reflection="Small stream fix.",
            changed_files=["main.go"],
        )
        strong = _repo_program(
            fitness=0.7,
            delta=0.05,
            reflection="Core parser improvement.",
            changed_files=["main.go"],
        )

        sel = MetaRepoParentSelector(num_parents=1)
        combos = list(sel.create_parent_iterator([weak, strong]))

        assert combos == [[strong], [weak]]

    def test_llm_selector_uses_meta_agent_ranked_parent_set(self) -> None:
        parser = _repo_program(
            fitness=0.80,
            delta=0.15,
            reflection="Parser fix.",
            changed_files=["parser.py"],
        )
        fetch = _repo_program(
            fitness=0.74,
            delta=0.24,
            reflection="Fetch timeout and redirect fix.",
            changed_files=["fetch.py"],
        )
        close_clone = _repo_program(
            fitness=0.79,
            delta=0.14,
            reflection="Similar parser fix.",
            changed_files=["parser.py"],
        )
        llm = _FakeLLM(
            {
                "analysis": "Fetch and parser fixes are complementary.",
                "parent_sets": [
                    {
                        "parent_ids": ["P2", "P1"],
                        "rationale": (
                            "Fetch and parser parents cover complementary failure clusters."
                        ),
                    }
                ],
            }
        )

        sel = MetaRepoParentSelector(
            num_parents=2,
            task_description="Build a general-purpose network and parsing harness.",
            llm=llm,
        )
        first = next(sel.create_parent_iterator([parser, fetch, close_clone]))

        assert isinstance(first, ParentSelection)
        assert first == [fetch, parser]
        assert first.selection_metadata["source"] == "llm_meta_agent"
        assert first.selection_metadata["rationale"]
        assert "own-improvement diffs" in llm.prompts[0]
        assert "P2" in llm.prompts[0]
        assert "Return exactly one JSON object" in llm.prompts[0]
        assert "Use the key parent_ids inside each parent set" in llm.prompts[0]
        assert "Build a general-purpose network and parsing harness." in llm.prompts[0]

    def test_llm_selector_accepts_candidate_ids_alias(self) -> None:
        parser = _repo_program(
            fitness=0.80,
            delta=0.15,
            reflection="Parser fix.",
            changed_files=["parser.py"],
        )
        fetch = _repo_program(
            fitness=0.74,
            delta=0.24,
            reflection="Fetch timeout and redirect fix.",
            changed_files=["fetch.py"],
        )
        close_clone = _repo_program(
            fitness=0.79,
            delta=0.14,
            reflection="Similar parser fix.",
            changed_files=["parser.py"],
        )
        llm = _FakeLLM(
            {
                "analysis": "Fetch and parser fixes are complementary.",
                "parent_sets": [
                    {
                        "candidate_ids": ["P2", "P1"],
                        "rationale": (
                            "Fetch and parser parents cover complementary failure clusters."
                        ),
                    }
                ],
            }
        )

        sel = MetaRepoParentSelector(num_parents=2, llm=llm)
        first = next(sel.create_parent_iterator([parser, fetch, close_clone]))

        assert first == [fetch, parser]
        assert first.selection_metadata["source"] == "llm_meta_agent"

    def test_llm_selector_falls_back_when_response_is_invalid(self) -> None:
        ungron = _repo_program(
            fitness=0.70,
            delta=0.20,
            reflection="Implemented ungron parser and assignment reconstruction.",
            changed_files=["main.go"],
        )
        ungron_clone = _repo_program(
            fitness=0.69,
            delta=0.19,
            reflection="Another ungron parser refinement with similar behavior.",
            changed_files=["main.go"],
        )
        url = _repo_program(
            fitness=0.68,
            delta=0.18,
            reflection="Implemented URL fetch, proxy, timeout, and redirect handling.",
            changed_files=["fetch.go"],
        )
        llm = _FakeLLM({"parent_sets": [{"parent_ids": ["missing", "P1"]}]})

        sel = MetaRepoParentSelector(num_parents=2, llm=llm)
        first = next(sel.create_parent_iterator([ungron, ungron_clone, url]))

        assert {p.id for p in first} == {ungron.id, url.id}
        assert first.selection_metadata["source"] == "heuristic_fallback"

    def test_llm_prompt_includes_own_improvement_diff(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        run_git(repo, ["init"])
        source = repo / "program.py"
        source.write_text("VALUE = 1\n")
        base = ensure_initial_commit(repo, message="base")
        source.write_text("VALUE = 2\n")
        child = commit_all(repo, message="child")

        improved = Program(
            code=RepoCandidateManifest(
                repo_path=str(repo),
                commit=child,
                parent_commit=base,
                changed_files=["program.py"],
            ).to_program_code()
        )
        improved.add_metrics({"fitness": 0.8})
        improved.metadata["repo_reflection"] = {
            "changed_files": ["program.py"],
            "metrics_delta": {"fitness": 0.3},
            "reflection": "Changed VALUE from one to two.",
        }
        other = _repo_program(
            fitness=0.4,
            delta=0.0,
            reflection="No useful change.",
            changed_files=["other.py"],
        )
        llm = _FakeLLM({"parent_sets": [{"parent_ids": ["P1"]}]})

        sel = MetaRepoParentSelector(num_parents=1, llm=llm, source_repo=repo)
        first = next(sel.create_parent_iterator([improved, other]))

        assert first == [improved]
        assert "-VALUE = 1" in llm.prompts[0]
        assert "+VALUE = 2" in llm.prompts[0]

    def test_llm_prompt_includes_pairing_history(self) -> None:
        json_parent = _repo_program(
            fitness=0.65,
            delta=0.12,
            reflection="Implemented JSON output.",
            changed_files=["output.py"],
        )
        table_parent = _repo_program(
            fitness=0.60,
            delta=0.08,
            reflection="Implemented table alignment.",
            changed_files=["table.py"],
        )
        child = Program.create_child(
            parents=[json_parent, table_parent],
            code="repo-manifest",
            mutation="Repo mutation: combined JSON and table paths",
        )
        child.add_metrics({"fitness": 0.82})
        child.iteration = 7
        child.metadata["selected_parent_program_ids"] = [
            json_parent.id,
            table_parent.id,
        ]
        child.metadata["parent_selection"] = {
            "source": "llm_meta_agent",
            "selected_parent_short_ids": [
                json_parent.short_id,
                table_parent.short_id,
            ],
            "rationale": "JSON and table fixes are complementary.",
        }
        child.metadata["repo_reflection"] = {
            "changed_files": ["output.py", "table.py"],
            "metrics_delta": {"fitness": 0.17},
            "reflection": (
                "Merged JSON serialization with table formatting; score improved."
            ),
        }
        unrelated = _repo_program(
            fitness=0.50,
            delta=0.05,
            reflection="Small CLI help fix.",
            changed_files=["cli.py"],
        )

        llm = _FakeLLM({"parent_sets": [{"parent_ids": ["P1", "P2"]}]})

        sel = MetaRepoParentSelector(num_parents=2, llm=llm)
        sel.set_population_history([child, json_parent, table_parent, unrelated])
        first = next(
            sel.create_parent_iterator([json_parent, table_parent, unrelated])
        )

        assert first == [json_parent, table_parent]
        prompt = llm.prompts[0]
        assert "pairing_history" in prompt
        assert '"child_in_candidate_pool": false' in prompt
        assert "JSON and table fixes are complementary" in prompt
        assert "Merged JSON serialization with table formatting" in prompt
        assert '"primary_metric_value": 0.82' in prompt
        assert child.short_id in prompt
        assert json_parent.short_id in prompt

    def test_llm_prompt_lets_agent_allocate_exploitation_and_exploration(self) -> None:
        champion = _repo_program(
            fitness=0.90,
            delta=0.05,
            reflection="Strong incumbent implementation.",
            changed_files=["solver.py"],
        )
        alternate = _repo_program(
            fitness=0.72,
            delta=-0.03,
            reflection="Different planner architecture with a promising mechanism.",
            changed_files=["planner.py"],
        )
        specialist = _repo_program(
            fitness=0.68,
            delta=0.10,
            reflection="Behaviorally distinct verifier-driven design.",
            changed_files=["verifier.py"],
        )

        selector = MetaRepoParentSelector(
            num_parents=2,
            max_ranked_sets=5,
            task_description="Improve the candidate without losing diversity.",
        )
        prompt = selector._render_llm_prompt([champion, alternate, specialist], 3)
        payload = json.loads(prompt.split("Elite candidate data:\n", 1)[1])
        portfolio = payload["selection_rules"]["mutation_portfolio"]

        assert portfolio["available_slots"] == 5
        assert portfolio["allocation_decided_by"] == "selector_agent"
        assert "exploitation_slots" not in portfolio
        assert "exploration_slots" not in portfolio
        assert "best metric has plateaued" in portfolio["allocation"]
        assert any(
            "recent pairing_history outcomes" in item
            for item in portfolio["allocation_evidence"]
        )
        assert "Decide the exploration/exploitation allocation yourself" in prompt
        assert "An exploration set must put such a candidate first" in prompt
        assert "do not assign every primary slot to the same champion phenotype" in (
            prompt
        )
        assert "rank a balanced portfolio" in prompt

    def test_llm_prompt_keeps_all_fifteen_candidates_and_valid_history(self) -> None:
        parents: list[Program] = []
        for index in range(15):
            parent = _repo_program(
                fitness=0.5 + index / 100,
                delta=index / 1000,
                reflection=(
                    f"KEEP-MECHANISM-{index}: distinct bounded implementation. "
                    + "technical evidence " * 350
                ),
                changed_files=[f"module_{index}.py"],
            )
            parent.metadata["repo_reflection"]["attempt_record"] = {
                "verdict": "improved" if index else "specialist",
                "pipeline_bookkeeping": f"DROP-ATTEMPT-{index} " * 5_000,
            }
            parent.metadata["parent_selection"] = {
                "analysis": f"DROP-SELECTOR-{index} " * 5_000,
            }
            parent.metadata["repo_descriptors"] = {
                "metrics": {"duplicate": f"DROP-DESCRIPTOR-{index} " * 5_000},
                "code": {
                    "modules": [f"module_{index}"],
                    "file_count": 1,
                    "extensions": ["py"],
                    "diff": {"files_changed": 1, "insertions": index + 1},
                    "risk_flags": [],
                },
                "feedback": {
                    "returncode": 0,
                    "duration_seconds": 1.0 + index / 10,
                    "failure_count": 0,
                    "failure_cluster_labels": [],
                    "failed_case_ids": [],
                    "timeout": False,
                },
                "risk": {"risk_flags": [], "quarantine_flags": []},
                "signature": {"sha1": f"{index:040x}"},
            }
            parents.append(parent)

        children: list[Program] = []
        for index in range(10):
            child = Program.create_child(
                parents=[parents[index], parents[index + 1]],
                code="repo-manifest",
                mutation=f"combined mechanisms {index} and {index + 1}",
            )
            child.add_metrics({"fitness": 0.6 + index / 100})
            child.iteration = index
            child.metadata["parent_selection"] = {
                "rationale": f"KEEP-PAIRING-{index}: complementary mechanisms.",
                "selected_parent_short_ids": [
                    parents[index].short_id,
                    parents[index + 1].short_id,
                ],
            }
            child.metadata["repo_reflection"] = {
                "metrics_delta": {"fitness": 0.01},
                "reflection": f"KEEP-OUTCOME-{index}: pairing improved the child.",
                "attempt_record": {
                    "verdict": "improved",
                    "pipeline_bookkeeping": "DROP-HISTORY-ATTEMPT " * 5_000,
                },
            }
            children.append(child)

        selector = MetaRepoParentSelector(
            num_parents=2,
            max_candidate_count=15,
            max_prompt_tokens=30_000,
            max_pairing_history_items=20,
            task_description="Choose complementary parents.",
        )
        selector.set_population_history([*children, *parents])
        prompt = selector._render_llm_prompt(parents, len(parents))

        prompt_json = prompt.split("Elite candidate data:\n", 1)[1]
        payload = json.loads(prompt_json)
        aliases = {card["candidate_id"] for card in payload["candidates"]}

        assert aliases == {f"P{index}" for index in range(1, 16)}
        assert payload["pairing_history"]
        assert selector._count_prompt_tokens(prompt) <= 30_000
        assert "KEEP-MECHANISM-0" in prompt
        assert "KEEP-MECHANISM-14" in prompt
        assert "KEEP-PAIRING-0" in prompt
        assert "KEEP-OUTCOME-9" in prompt
        assert "DROP-ATTEMPT" not in prompt
        assert "DROP-SELECTOR" not in prompt
        assert "DROP-DESCRIPTOR" not in prompt
        assert "DROP-HISTORY-ATTEMPT" not in prompt
        assert not prompt.endswith("...<truncated>")
