from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from gigaevo.evolution.mutation.parent_selector import ParentSelection
from gigaevo.evolution.mutation.repo_harness_operator import RepoHarnessMutationOperator
from gigaevo.programs.core_types import StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.repo_benchmark import RepoBenchmarkStage
from gigaevo.programs.stages.repo_reflection import RepoReflectionStage
from gigaevo.repo_harness.backends import CodingAgentRun
from gigaevo.repo_harness.git_utils import changed_files, rev_parse, run_git
from gigaevo.repo_harness.manifest import RepoCandidateManifest


def _load_programbench_benchmark_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "problems"
        / "programbench_task"
        / "programbench"
        / "benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("programbench_benchmark", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_cybergym_benchmark_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "problems"
        / "cybergym_harness"
        / "cybergym"
        / "benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("cybergym_benchmark", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    (path / "harness.py").write_text("VALUE = 1\n")
    (path / "bench.py").write_text(
        "import json\n"
        "import harness\n"
        "print(json.dumps({'fitness': float(harness.VALUE), 'is_valid': 1.0}))\n"
    )
    run_git(path, ["add", "-A"])
    run_git(path, ["commit", "-m", "initial"])
    return rev_parse(path)


class _EditingBackend:
    name = "test-editor"

    async def run(self, *, prompt, cwd, log_dir, timeout, variables=None):
        target = cwd / "harness.py"
        target.write_text("VALUE = 2\n")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.txt").write_text("edited harness.py\n")
        return CodingAgentRun(
            backend_name=self.name,
            exit_code=0,
            stdout="edited harness.py\n",
            stderr="",
            duration_seconds=0.01,
            log_path=str(log_dir),
        )


class _PromptCapturingBackend:
    name = "prompt-capturing-editor"

    def __init__(self):
        self.prompt = None
        self.variables = None

    async def run(self, *, prompt, cwd, log_dir, timeout, variables=None):
        self.prompt = prompt
        self.variables = variables or {}
        (cwd / "harness.py").write_text("VALUE = 4\n")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.txt").write_text("edited from selected parents\n")
        return CodingAgentRun(
            backend_name=self.name,
            exit_code=0,
            stdout="edited from selected parents\n",
            stderr="",
            duration_seconds=0.01,
            log_path=str(log_dir),
        )


class _FakeStorage:
    def __init__(self, *programs: Program):
        self.programs = {p.id: p for p in programs}

    async def get(self, program_id: str) -> Program | None:
        return self.programs.get(program_id)


class _FakeReflectionLLM:
    def __init__(self, content: str = "## Change Summary\n- Edited harness.py"):
        self.content = content
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


class _FailingReflectionLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("reflection unavailable")


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Primary score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=10.0,
            ),
            "is_valid": MetricSpec(
                description="Validity",
                is_primary=False,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _commit_value(repo: Path, value: int, message: str) -> str:
    (repo / "harness.py").write_text(f"VALUE = {value}\n")
    run_git(repo, ["add", "-A"])
    run_git(repo, ["commit", "-m", message])
    return rev_parse(repo)


async def test_repo_harness_mutation_operator_commits_backend_edit(tmp_path):
    repo = tmp_path / "agent_repo"
    parent_commit = _init_repo(repo)
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=parent_commit)
    parent = Program(code=manifest.to_program_code())
    parent.add_metrics({"fitness": 1.0, "is_valid": 1.0})
    parent.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 1.0, "is_valid": 1.0},
        "stderr_tail": "baseline is weak",
    }

    operator = RepoHarnessMutationOperator(
        source_repo=repo,
        backend=_EditingBackend(),
        worktree_root=tmp_path / "worktrees",
        log_root=tmp_path / "logs",
        keep_worktrees=True,
    )

    spec = await operator.mutate_single([parent])

    assert spec is not None
    child = RepoCandidateManifest.model_validate_json(spec.code)
    assert child.parent_commit == parent_commit
    assert child.commit != parent_commit
    assert "harness.py" in child.changed_files
    assert changed_files(repo, parent_commit, child.commit) == ["harness.py"]
    assert spec.metadata["git_commit"] == child.commit


async def test_fixed_idea_smoke_and_recent_failure_are_unavoidable(tmp_path):
    repo = tmp_path / "agent_repo"
    parent_commit = _init_repo(repo)
    parent = Program(
        code=RepoCandidateManifest(
            repo_path=str(repo), commit=parent_commit
        ).to_program_code()
    )
    parent.add_metrics({"fitness": 1.0, "is_valid": 1.0})
    idea_path = tmp_path / "idea.json"
    idea_path.write_text(
        json.dumps(
            {
                "id": "idea-fixed",
                "status": "approved",
                "hypothesis": "fixed hypothesis",
                "proposed_change": "fixed implementation",
                "expected_mechanism": "fixed mechanism",
                "success_signal": ["lower loss"],
                "risks": ["instability"],
            }
        )
    )
    state_path = tmp_path / "promotion-state.json"
    state_path.write_text(
        json.dumps(
            {
                "last_failure": {
                    "commit": "deadbeef",
                    "phase": "long_confirmation",
                    "status": "train_failed",
                }
            }
        )
    )
    backend = _PromptCapturingBackend()
    operator = RepoHarnessMutationOperator(
        source_repo=repo,
        backend=backend,
        worktree_root=tmp_path / "worktrees",
        log_root=tmp_path / "logs",
        keep_worktrees=True,
        active_idea_path=idea_path,
        recent_failure_path=state_path,
        smoke_commands=[
            [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'is_valid': 1.0}))",
            ]
        ],
    )

    spec = await operator.mutate_single([parent])

    assert spec is not None
    brief_path = (
        Path(spec.metadata["mutation_session_log"]).parent / "mutation_brief.md"
    )
    brief = brief_path.read_text()
    assert brief.index("FIXED IDEA CONTRACT") < brief.index("Task Description")
    assert "fixed mechanism" in brief
    assert "Required Executable Smoke Test" in brief
    assert "is_valid == 1" in brief
    assert "Most Recent Incomplete Experiment" in brief
    assert "deadbeef" in brief
    assert "not scientific evidence against the fixed idea" in brief
    assert backend.prompt is not None
    assert "must not replace, omit, or drift away" in backend.prompt


async def test_repo_harness_mutation_operator_briefs_multiple_parents(tmp_path):
    repo = tmp_path / "agent_repo"
    primary_commit = _init_repo(repo)
    secondary_commit = _commit_value(repo, 3, "secondary direction")

    primary_manifest = RepoCandidateManifest(repo_path=str(repo), commit=primary_commit)
    primary = Program(code=primary_manifest.to_program_code())
    primary.add_metrics({"fitness": 1.0, "is_valid": 1.0})
    structured_feedback = {
        "diagnosis": {"primary_issue": "primary feedback"},
        "metrics": {"fitness": 1.0, "is_valid": 1.0},
        "artifacts": [{"captured_path": "duplicated-primary-trial.log"}],
    }
    primary.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 1.0, "is_valid": 1.0},
        "stdout_tail": '{"fitness": 1.0, "is_valid": 1.0}',
        "structured_feedback": structured_feedback,
        "structured_failure_feedback": structured_feedback,
    }
    primary.metadata["repo_reflection"] = {
        "reflection": "Primary parent retained the stable implementation.",
        "metrics_child": {"fitness": 1.0, "is_valid": 1.0},
        "prompt_token_count": 12345,
        "codex_usage": {"input_tokens": 99999, "estimated_cost_usd": 12.34},
    }

    secondary_manifest = RepoCandidateManifest(
        repo_path=str(repo),
        commit=secondary_commit,
        parent_commit=primary_commit,
        changed_files=["harness.py"],
    )
    secondary = Program.create_child(
        parents=[primary],
        code=secondary_manifest.to_program_code(),
        mutation="secondary repo edit",
    )
    secondary.add_metrics({"fitness": 3.0, "is_valid": 1.0})
    secondary.metadata["repo_reflection"] = {
        "commit": secondary_commit,
        "parent_commit": primary_commit,
        "changed_files": ["harness.py"],
        "metrics_delta": {"fitness": 2.0},
        "reflection": "Secondary parent discovered a complementary value path.",
        "prompt_token_count": 54321,
        "codex_usage": {"input_tokens": 88888, "estimated_cost_usd": 23.45},
    }
    secondary.metadata["repo_evaluation_artifacts"] = {
        "items": [
            {
                "kind": "trial_log",
                "captured": True,
                "captured_path": str(tmp_path / "secondary-trial.log"),
            }
        ],
        "limits": {"max_total_bytes": 250000000},
        "stats": {"bytes": 1234, "files": 1},
    }

    backend = _PromptCapturingBackend()
    operator = RepoHarnessMutationOperator(
        source_repo=repo,
        backend=backend,
        problem_context=SimpleNamespace(
            task_description="TASK: maximize harness.VALUE without breaking validity."
        ),
        worktree_root=tmp_path / "worktrees",
        log_root=tmp_path / "logs",
        keep_worktrees=True,
    )

    selection = ParentSelection(
        [primary, secondary],
        selection_metadata={
            "source": "llm_meta_agent",
            "rationale": "Primary is stable; secondary contains a stronger value path.",
            "selected_parent_ids": [primary.id, secondary.id],
            "selected_parent_short_ids": [primary.short_id, secondary.short_id],
            "created_at": "2026-08-09T00:00:00+00:00",
            "codex_usage": {
                "input_tokens": 77777,
                "estimated_cost_usd": 34.56,
            },
        },
    )

    spec = await operator.mutate_single(selection)

    assert spec is not None
    assert spec.parents == [primary, secondary]
    child = RepoCandidateManifest.model_validate_json(spec.code)
    assert child.parent_commit == primary_commit
    assert child.extra["primary_parent_program_id"] == primary.id
    assert child.extra["selected_parent_program_ids"] == [primary.id, secondary.id]
    assert child.extra["selected_parent_commits"] == [
        primary_commit,
        secondary_commit,
    ]
    assert child.extra["parent_selection"]["source"] == "llm_meta_agent"
    assert spec.metadata["parent_commits"] == [primary_commit, secondary_commit]
    assert spec.metadata["parent_selection"]["rationale"]

    brief_path = (
        Path(spec.metadata["mutation_session_log"]).parent / "mutation_brief.md"
    )
    brief = brief_path.read_text()
    assert "Task Description" in brief
    assert "TASK: maximize harness.VALUE without breaking validity." in brief
    assert brief.count("TASK: maximize harness.VALUE without breaking validity.") == 1
    assert "Selected Parent Set" in brief
    assert "Parent Selection" in brief
    assert "Primary is stable; secondary contains a stronger value path" in brief
    assert (
        brief.count("Primary is stable; secondary contains a stronger value path") == 1
    )
    assert "- Parent count: 2" in brief
    assert '"role": "primary_base"' in brief
    assert '"role": "secondary_source"' in brief
    assert "diff_from_primary_parent" in brief
    assert "primary feedback" in brief
    assert "Secondary parent discovered a complementary value path" in brief
    assert "secondary-trial.log" in brief
    assert "VALUE = 3" in brief
    assert '"structured_feedback"' in brief
    assert '"structured_failure_feedback"' not in brief
    assert '"codex_usage"' not in brief
    assert '"prompt_token_count"' not in brief
    assert '"selected_parent_ids"' not in brief
    assert '"selected_parent_short_ids"' not in brief
    assert '"task_description"' not in brief
    assert '"children"' not in brief
    assert '"limits"' not in brief
    assert '"stats"' not in brief
    assert "duplicated-primary-trial.log" not in brief
    assert backend.prompt is not None
    assert str(brief_path) in backend.prompt
    assert "Return the phenotype produced by the mutation's central experiment" in (
        backend.prompt
    )
    assert 'incumbent output, add an "accept only if better" score guard' in (
        backend.prompt
    )
    assert "Remove or bypass inherited score-only guards" in backend.prompt
    assert "Keep fallbacks only for crashes" in backend.prompt


def test_repo_harness_mutation_operator_ignores_inherited_llm_kwargs(tmp_path):
    repo = tmp_path / "agent_repo"
    _init_repo(repo)

    operator = RepoHarnessMutationOperator(
        source_repo=repo,
        backend=_EditingBackend(),
        llm_wrapper=object(),
        mutation_mode="rewrite",
        problem_context=object(),
        strip_comments_and_docstrings=False,
        prompts_dir=None,
        prompt_fetcher=None,
    )

    assert operator.source_repo == repo.resolve()


def test_cybergym_benchmark_uses_candidate_agent_config(tmp_path):
    module = _load_cybergym_benchmark_module()
    candidate_repo = tmp_path / "candidate"
    candidate_repo.mkdir()
    runner = candidate_repo / "run.py"
    runner.write_text("print('runner')\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    job_dir = tmp_path / "job"
    (candidate_repo / "gigaevo_agent_config.json").write_text(
        json.dumps(
            {
                "runner_args": {
                    "max_iter": 37,
                    "inner_timeout": 456,
                    "silent": True,
                },
                "llm": {
                    "api_base": "http://127.0.0.1:8001/v1",
                    "api_key": "secret-key",
                    "temperature": 0.2,
                    "max_output_tokens": 4096,
                },
            }
        )
    )

    argv = [
        "--candidate-repo",
        str(candidate_repo),
        "--data-dir",
        str(data_dir),
        "--task",
        "arvo:10400",
        "--model",
        "openai/Qwen/Qwen3.6-35B-A3B",
    ]
    args = module.build_parser().parse_args(argv)
    args._supplied_options = module.supplied_cli_options(argv)

    command = module.build_candidate_command(
        args,
        candidate_repo=candidate_repo,
        task="arvo:10400",
        attempt=1,
        job_dir=job_dir,
    )

    assert command[0] == sys.executable
    assert command[1] == str(runner.resolve())
    assert command[command.index("--max_iter") + 1] == "37"
    assert command[command.index("--timeout") + 1] == "456"
    assert command[command.index("--base_url") + 1] == "http://127.0.0.1:8001/v1"
    assert "--api_key" not in command
    assert command[command.index("--temperature") + 1] == "0.2"
    assert command[command.index("--max_output_tokens") + 1] == "4096"
    assert command[command.index("--silent") + 1] == "true"
    env = module.build_env(args, candidate_repo)
    assert env["LLM_API_KEY"] == "secret-key"
    assert env["OPENAI_API_KEY"] == "secret-key"

    summary = module.agent_config_summary(candidate_repo, args)
    assert summary["candidate_config_path"] == str(
        candidate_repo / "gigaevo_agent_config.json"
    )
    assert summary["candidate_config"]["llm"]["api_key"] == "<redacted>"


def test_cybergym_benchmark_cli_overrides_candidate_agent_config(tmp_path):
    module = _load_cybergym_benchmark_module()
    candidate_repo = tmp_path / "candidate"
    candidate_repo.mkdir()
    (candidate_repo / "run.py").write_text("print('runner')\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (candidate_repo / "gigaevo_agent_config.json").write_text(
        json.dumps({"runner_args": {"max_iter": 37}})
    )
    argv = [
        "--candidate-repo",
        str(candidate_repo),
        "--data-dir",
        str(data_dir),
        "--task",
        "arvo:10400",
        "--max-iter",
        "12",
    ]
    args = module.build_parser().parse_args(argv)
    args._supplied_options = module.supplied_cli_options(argv)

    command = module.build_candidate_command(
        args,
        candidate_repo=candidate_repo,
        task="arvo:10400",
        attempt=1,
        job_dir=tmp_path / "job",
    )

    assert command[command.index("--max_iter") + 1] == "12"


async def test_repo_benchmark_stage_checks_out_commit_and_parses_stdout_json(tmp_path):
    repo = tmp_path / "agent_repo"
    commit = _init_repo(repo)
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=commit)
    program = Program(code=manifest.to_program_code())

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=["python", "bench.py"],
        worktree_root=tmp_path / "eval_worktrees",
        timeout=30.0,
    )
    stage.attach_inputs({})

    result = await stage.compute(program)

    assert result.data == {"fitness": 1.0, "is_valid": 1.0}
    assert program.metadata["repo_benchmark_feedback"]["returncode"] == 0
    assert program.metrics["fitness"] == 1.0


async def test_repo_benchmark_stage_extracts_structured_failure_feedback(tmp_path):
    repo = tmp_path / "agent_repo"
    _init_repo(repo)
    (repo / "bench.py").write_text(
        "import json, sys\n"
        "print(json.dumps({'fitness': 0.5, 'is_valid': 1.0}))\n"
        "print('[gigaevo] structured feedback:', file=sys.stderr)\n"
        "print(json.dumps({\n"
        "    'schema_version': 1,\n"
        "    'benchmark': 'example',\n"
        "    'failure_clusters': [{'name': 'parser', 'count': 2}],\n"
        "    'failure_count': 2,\n"
        "    'failure_cluster_summary': {'total_failures': 2},\n"
        "}), file=sys.stderr)\n"
        "print('[example] summary: /tmp/summary.json', file=sys.stderr)\n"
    )
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "structured feedback bench"])
    commit = rev_parse(repo)
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=commit)
    program = Program(code=manifest.to_program_code())

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=["python", "bench.py"],
        worktree_root=tmp_path / "eval_worktrees",
        timeout=30.0,
    )
    stage.attach_inputs({})

    await stage.compute(program)

    feedback = program.metadata["repo_benchmark_feedback"]
    assert feedback["structured_feedback"]["benchmark"] == "example"
    assert feedback["structured_feedback"]["failure_clusters"] == [
        {"name": "parser", "count": 2}
    ]
    assert feedback["structured_failure_feedback"]["failure_count"] == 2
    assert (
        feedback["structured_failure_feedback"]["failure_cluster_summary"][
            "total_failures"
        ]
        == 2
    )


async def test_repo_benchmark_stage_captures_generic_evaluation_artifacts(tmp_path):
    repo = tmp_path / "agent_repo"
    _init_repo(repo)
    (repo / "bench.py").write_text(
        "import json, pathlib, sys\n"
        "log_dir = pathlib.Path('logs')\n"
        "log_dir.mkdir(exist_ok=True)\n"
        "(log_dir / 'trial.log').write_text('full trial log\\n')\n"
        "(log_dir / 'result.json').write_text(json.dumps({'reward': 0.0}))\n"
        "feedback = {\n"
        "    'schema_version': 1,\n"
        "    'benchmark': 'generic-example',\n"
        "    'examples': [{\n"
        "        'task': 'demo',\n"
        "        'artifacts': [{\n"
        "            'name': 'trial.log',\n"
        "            'path': 'logs/trial.log',\n"
        "            'kind': 'trial_log',\n"
        "            'role': 'failure',\n"
        "        }],\n"
        "    }],\n"
        "    'artifact_paths': ['logs/result.json'],\n"
        "}\n"
        "print(json.dumps({'fitness': 0.0, 'is_valid': 1.0}))\n"
        "print('[gigaevo] structured feedback:', file=sys.stderr)\n"
        "print(json.dumps(feedback), file=sys.stderr)\n"
    )
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "artifact feedback bench"])
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=rev_parse(repo))
    program = Program(code=manifest.to_program_code())

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=["python", "bench.py"],
        worktree_root=tmp_path / "eval_worktrees",
        artifact_root=tmp_path / "artifacts",
        timeout=30.0,
    )
    stage.attach_inputs({})

    await stage.compute(program)

    feedback = program.metadata["repo_benchmark_feedback"]
    artifacts = feedback["evaluation_artifacts"]
    assert program.metadata["repo_evaluation_artifacts"] == artifacts
    assert Path(artifacts["manifest_path"]).is_file()

    items = artifacts["items"]
    assert any(item["kind"] == "benchmark_stdout" for item in items)
    assert any(item["kind"] == "benchmark_stderr" for item in items)
    trial_log = next(item for item in items if item.get("kind") == "trial_log")
    result_json = next(item for item in items if item.get("name") == "result.json")
    assert Path(trial_log["captured_path"]).read_text() == "full trial log\n"
    assert json.loads(Path(result_json["captured_path"]).read_text()) == {"reward": 0.0}
    assert artifacts["stats"]["captured_items"] >= 4


def _write_case_benchmark(
    repo: Path, log_path: Path, *, subset_c_reward: float
) -> None:
    (repo / "bench.py").write_text(
        "import argparse, json, pathlib, sys\n"
        f"LOG_PATH = pathlib.Path({str(log_path)!r})\n"
        f"SUBSET_C_REWARD = {subset_c_reward!r}\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--cases', default='a,b,c')\n"
        "args = parser.parse_args()\n"
        "cases = [part for part in args.cases.split(',') if part]\n"
        "LOG_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "with LOG_PATH.open('a') as fh:\n"
        "    fh.write(json.dumps(cases) + '\\n')\n"
        "full_rewards = {'a': 1.0, 'b': 1.0, 'c': 0.0}\n"
        "if cases == ['c']:\n"
        "    rewards = {'c': SUBSET_C_REWARD}\n"
        "else:\n"
        "    rewards = full_rewards\n"
        "case_rows = [\n"
        "    {\n"
        "        'case_id': case,\n"
        "        'reward': rewards.get(case, 0.0),\n"
        "        'passed': rewards.get(case, 0.0) > 0,\n"
        "        'status': 'passed' if rewards.get(case, 0.0) > 0 else 'failed',\n"
        "    }\n"
        "    for case in cases\n"
        "]\n"
        "fitness = sum(row['reward'] for row in case_rows) / len(case_rows)\n"
        "metrics = {'fitness': fitness, 'is_valid': 1.0}\n"
        "feedback = {\n"
        "    'schema_version': 1,\n"
        "    'benchmark': 'case-benchmark',\n"
        "    'summary': {'selected_cases': cases, 'metrics': metrics},\n"
        "    'cases': case_rows,\n"
        "    'failure_clusters': [\n"
        "        {'name': row['case_id'], 'count': 1}\n"
        "        for row in case_rows\n"
        "        if not row['passed']\n"
        "    ],\n"
        "}\n"
        "print(json.dumps(metrics))\n"
        "print('[gigaevo] structured feedback:', file=sys.stderr)\n"
        "print(json.dumps(feedback), file=sys.stderr)\n"
    )


def _parent_with_case_feedback(repo: Path, commit: str) -> Program:
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=commit)
    parent = Program(code=manifest.to_program_code())
    parent.add_metrics({"fitness": 2.0 / 3.0, "is_valid": 1.0})
    parent.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 2.0 / 3.0, "is_valid": 1.0},
        "structured_feedback": {
            "schema_version": 1,
            "benchmark": "case-benchmark",
            "summary": {"selected_cases": ["a", "b", "c"]},
            "cases": [
                {"case_id": "a", "reward": 1.0, "passed": True},
                {"case_id": "b", "reward": 1.0, "passed": True},
                {"case_id": "c", "reward": 0.0, "passed": False},
            ],
            "failure_clusters": [{"name": "c", "count": 1}],
        },
    }
    return parent


async def test_repo_benchmark_stage_stops_after_parent_failed_cases_without_improvement(
    tmp_path,
):
    repo = tmp_path / "agent_repo"
    commit = _init_repo(repo)
    log_path = tmp_path / "case_calls.jsonl"
    _write_case_benchmark(repo, log_path, subset_c_reward=0.0)
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "case benchmark"])
    commit = rev_parse(repo)

    parent = _parent_with_case_feedback(repo, commit)
    child = Program.create_child(
        parents=[parent],
        code=RepoCandidateManifest(
            repo_path=str(repo), commit=commit
        ).to_program_code(),
        mutation="noop",
    )

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=[sys.executable, "bench.py", "--cases", "a,b,c"],
        storage=_FakeStorage(parent),
        worktree_root=tmp_path / "eval_worktrees",
        staged_validation_enabled=True,
        staged_case_arg_names=["--cases"],
        timeout=30.0,
    )
    stage.attach_inputs({})

    result = await stage.compute(child)

    assert result.data["fitness"] == 0.666667
    assert json.loads(log_path.read_text().strip()) == ["c"]
    staged = child.metadata["repo_benchmark_feedback"]["staged_validation"]
    assert staged["promoted_to_full"] is False
    assert staged["failed_cases"] == ["c"]


async def test_repo_benchmark_stage_promotes_to_full_after_failed_case_improvement(
    tmp_path,
):
    repo = tmp_path / "agent_repo"
    commit = _init_repo(repo)
    log_path = tmp_path / "case_calls.jsonl"
    _write_case_benchmark(repo, log_path, subset_c_reward=1.0)
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "case benchmark"])
    commit = rev_parse(repo)

    parent = _parent_with_case_feedback(repo, commit)
    child = Program.create_child(
        parents=[parent],
        code=RepoCandidateManifest(
            repo_path=str(repo), commit=commit
        ).to_program_code(),
        mutation="noop",
    )

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=[sys.executable, "bench.py", "--cases", "a,b,c"],
        storage=_FakeStorage(parent),
        worktree_root=tmp_path / "eval_worktrees",
        staged_validation_enabled=True,
        staged_case_arg_names=["--cases"],
        timeout=30.0,
    )
    stage.attach_inputs({})

    result = await stage.compute(child)

    assert result.data["fitness"] == 2.0 / 3.0
    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert calls == [["c"], ["a", "b", "c"]]
    staged = child.metadata["repo_benchmark_feedback"]["staged_validation"]
    assert staged["promoted_to_full"] is True
    assert staged["comparison"]["delta"] == 1.0


async def test_repo_benchmark_stage_extracts_legacy_programbench_feedback(tmp_path):
    repo = tmp_path / "agent_repo"
    _init_repo(repo)
    (repo / "bench.py").write_text(
        "import json, sys\n"
        "print(json.dumps({'fitness': 0.5, 'is_valid': 1.0}))\n"
        "print('[programbench] structured failure feedback:', file=sys.stderr)\n"
        "print(json.dumps({'failure_count': 1}), file=sys.stderr)\n"
    )
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "legacy structured feedback bench"])
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=rev_parse(repo))
    program = Program(code=manifest.to_program_code())

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=["python", "bench.py"],
        worktree_root=tmp_path / "eval_worktrees",
        timeout=30.0,
    )
    stage.attach_inputs({})

    await stage.compute(program)

    feedback = program.metadata["repo_benchmark_feedback"]
    assert feedback["structured_feedback"]["failure_count"] == 1


async def test_repo_benchmark_stage_reads_structured_feedback_path(tmp_path):
    repo = tmp_path / "agent_repo"
    _init_repo(repo)
    (repo / "bench.py").write_text(
        "import json, pathlib\n"
        "pathlib.Path('feedback.json').write_text(json.dumps({\n"
        "    'schema_version': 1,\n"
        "    'benchmark': 'file-benchmark',\n"
        "    'summary': {'failed_trials': 3},\n"
        "}))\n"
        "print(json.dumps({'fitness': 0.25, 'is_valid': 1.0}))\n"
    )
    run_git(repo, ["add", "bench.py"])
    run_git(repo, ["commit", "-m", "structured feedback file bench"])
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=rev_parse(repo))
    program = Program(code=manifest.to_program_code())

    stage = RepoBenchmarkStage(
        source_repo=repo,
        benchmark_command=["python", "bench.py"],
        structured_feedback_path="feedback.json",
        worktree_root=tmp_path / "eval_worktrees",
        timeout=30.0,
    )
    stage.attach_inputs({})

    await stage.compute(program)

    feedback = program.metadata["repo_benchmark_feedback"]
    assert feedback["structured_feedback"]["benchmark"] == "file-benchmark"
    assert feedback["structured_feedback"]["summary"]["failed_trials"] == 3


def test_programbench_feedback_includes_failure_cluster_summary(tmp_path):
    benchmark = _load_programbench_benchmark_module()
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "executable_hash": "abc",
                "test_results": [
                    {
                        "branch": "b1",
                        "name": "tests.test_harvest.test_fixture_counts[tests/data/a.rs-expected1]",
                        "status": "failure",
                        "extra": {"message": "AssertionError: wrong counts"},
                    },
                    {
                        "branch": "b1",
                        "name": "tests.test_harvest.test_fixture_counts[tests/data/b.py-expected2]",
                        "status": "failure",
                        "extra": {"message": "AssertionError: wrong counts"},
                    },
                    {
                        "branch": "b1",
                        "name": "tests.test_embedded_languages.test_vue_component",
                        "status": "failure",
                        "extra": {"message": "AssertionError: vue total"},
                    },
                    {
                        "branch": "b1",
                        "name": "tests.test_cli_utils.test_files_list_individual_files",
                        "status": "failure",
                        "extra": {"message": "AssertionError: table mismatch"},
                    },
                    {
                        "branch": "b1",
                        "name": "tests.test_smoke.test_help",
                        "status": "passed",
                        "extra": {},
                    },
                ],
            }
        )
    )

    metrics, details = benchmark.metrics_from_eval(
        eval_json,
        {"branches": {}},
        0,
        failure_limit=2,
        cluster_examples_per_cluster=1,
    )

    summary = details["failure_cluster_summary"]
    assert metrics["n_resolved"] == 1.0
    assert summary["total_failures"] == 4
    assert summary["by_module"][0] == {"name": "tests.test_harvest", "count": 2}
    assert any(item["name"] == "harvest_fixtures" for item in summary["by_theme"])
    assert all(cluster["example_limit"] == 1 for cluster in summary["clusters"])
    assert "failure_cluster_summary" in benchmark.render_failure_feedback(details)
    rendered = json.loads(benchmark.render_failure_feedback(details))
    assert rendered["benchmark"] == "programbench"
    assert rendered["failure_clusters"][0]["count"] == 2
    assert rendered["raw"]["failure_cluster_summary"]["total_failures"] == 4
    assert rendered["summary"]["failure_count"] == 4


async def test_repo_reflection_stage_uses_diff_metrics_and_parent_feedback(tmp_path):
    repo = tmp_path / "agent_repo"
    parent_commit = _init_repo(repo)
    child_commit = _commit_value(repo, 2, "raise value")

    parent_manifest = RepoCandidateManifest(repo_path=str(repo), commit=parent_commit)
    parent = Program(code=parent_manifest.to_program_code())
    parent.add_metrics({"fitness": 1.0, "is_valid": 1.0})
    parent.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 1.0, "is_valid": 1.0},
        "stderr_tail": "baseline weak on easy tasks",
    }
    parent.metadata["repo_reflection"] = {
        "commit": parent_commit,
        "changed_files": ["harness.py"],
        "reflection": "Earlier lineage note",
    }

    child_manifest = RepoCandidateManifest(
        repo_path=str(repo),
        commit=child_commit,
        parent_commit=parent_commit,
        changed_files=["harness.py"],
    )
    child = Program.create_child(
        parents=[parent],
        code=child_manifest.to_program_code(),
        mutation="repo edit",
    )
    child.add_metrics({"fitness": 2.0, "is_valid": 1.0})
    child.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 2.0, "is_valid": 1.0},
        "stdout_tail": '{"fitness": 2.0, "is_valid": 1.0}',
    }
    child.metadata["repo_evaluation_artifacts"] = {
        "items": [
            {
                "kind": "trial_log",
                "captured": True,
                "captured_path": str(tmp_path / "child-trial.log"),
            }
        ]
    }

    llm = _FakeReflectionLLM("## Lineage Assessment\n- Fitness improved.")
    stage = RepoReflectionStage(
        llm=llm,
        storage=_FakeStorage(parent),
        task_description="Improve the agent harness.",
        metrics_context=_metrics_context(),
        timeout=30.0,
    )
    stage.attach_inputs(
        {"metrics": FloatDictContainer(data={"fitness": 2.0, "is_valid": 1.0})}
    )

    result = await stage.execute(child)

    assert result.status == StageState.COMPLETED
    text = result.output.data
    assert "Repo Reflection Context" in text
    assert "harness.py" in text
    assert "Fitness improved" in text
    assert "VALUE = 2" in text
    assert child.metadata["repo_reflection"]["parent_commit"] == parent_commit
    assert child.metadata["repo_reflection"]["metrics_delta"]["fitness"] == 1.0

    prompt_text = "\n".join(m.content for m in llm.messages)
    assert "Improve the agent harness." in prompt_text
    assert "Next Mutation Guidance" not in prompt_text
    assert "Do not propose follow-up edits" in prompt_text
    assert "baseline weak on easy tasks" in prompt_text
    assert "Earlier lineage note" in prompt_text
    assert "child-trial.log" in prompt_text
    assert "Evaluation Artifacts" in text
    assert '"delta"' in prompt_text
    assert "VALUE = 1" in prompt_text
    assert "VALUE = 2" in prompt_text
    assert "do not invoke tools" in prompt_text
    assert child.metadata["repo_reflection"]["prompt_token_count"] > 0
    assert child.metadata["repo_reflection"]["prompt_token_budget"] == 150000


async def test_repo_reflection_prompt_is_token_bounded_and_deduplicated(tmp_path):
    repo = tmp_path / "agent_repo"
    parent_commit = _init_repo(repo)
    child_commit = _commit_value(repo, 2, "raise value")

    parent_manifest = RepoCandidateManifest(repo_path=str(repo), commit=parent_commit)
    parent = Program(code=parent_manifest.to_program_code())
    parent.add_metrics({"fitness": 1.0, "is_valid": 1.0})

    child_manifest = RepoCandidateManifest(
        repo_path=str(repo),
        commit=child_commit,
        parent_commit=parent_commit,
        changed_files=["harness.py"],
    )
    child = Program.create_child(
        parents=[parent],
        code=child_manifest.to_program_code(),
        mutation="repo edit",
    )
    child.add_metrics({"fitness": 2.0, "is_valid": 1.0})

    structured = {
        "summary": {"fitness": 2.0, "status": "valid"},
        "metrics": {"fitness": 2.0, "is_valid": 1.0},
        "artifacts": [{"path": "duplicated-trial.log"}],
        "diagnosis": {"primary_issue": "bounded evidence"},
        "large_diagnostic": "diagnostic evidence " * 8000,
    }
    child.metadata["repo_benchmark_feedback"] = {
        "metrics": {"fitness": 2.0, "is_valid": 1.0},
        "stdout_tail": '{"fitness": 2.0, "is_valid": 1.0}',
        "structured_feedback": structured,
        "structured_failure_feedback": structured,
        "evaluation_artifacts": {"items": [{"captured_path": "duplicated-trial.log"}]},
    }
    child.metadata["repo_evaluation_artifacts"] = {
        "items": [
            {
                "kind": "trial_log",
                "name": "trial.log",
                "captured": True,
                "captured_path": "canonical-trial.log",
            }
        ]
    }

    llm = _FakeReflectionLLM()
    stage = RepoReflectionStage(
        llm=llm,
        storage=_FakeStorage(parent),
        task_description="Improve the agent harness.",
        metrics_context=_metrics_context(),
        max_feedback_chars=24000,
        max_prompt_tokens=2000,
        timeout=30.0,
    )
    stage.attach_inputs(
        {"metrics": FloatDictContainer(data={"fitness": 2.0, "is_valid": 1.0})}
    )

    result = await stage.execute(child)

    assert result.status == StageState.COMPLETED
    prompt_text = "\n".join(m.content for m in llm.messages)
    reflection = child.metadata["repo_reflection"]
    assert reflection["prompt_token_count"] <= 2000
    assert reflection["prompt_token_budget"] == 2000
    assert '"structured_failure_feedback"' not in prompt_text
    assert '"structured_feedback"' in prompt_text
    assert "canonical-trial.log" in prompt_text
    assert "duplicated-trial.log" not in prompt_text
    assert '"delta"' in prompt_text
    assert "harness.py" in prompt_text
    assert "VALUE = 2" in prompt_text


async def test_repo_reflection_stage_skips_llm_for_clear_regression(tmp_path):
    repo = tmp_path / "agent_repo"
    parent_commit = _init_repo(repo)
    child_commit = _commit_value(repo, 0, "lower value")

    parent_manifest = RepoCandidateManifest(repo_path=str(repo), commit=parent_commit)
    parent = Program(code=parent_manifest.to_program_code())
    parent.add_metrics({"fitness": 1.0, "is_valid": 1.0})

    child_manifest = RepoCandidateManifest(
        repo_path=str(repo),
        commit=child_commit,
        parent_commit=parent_commit,
        changed_files=["harness.py"],
    )
    child = Program.create_child(
        parents=[parent],
        code=child_manifest.to_program_code(),
        mutation="repo edit",
    )
    child.add_metrics({"fitness": 0.0, "is_valid": 1.0})

    llm = _FakeReflectionLLM("this should not be used")
    stage = RepoReflectionStage(
        llm=llm,
        storage=_FakeStorage(parent),
        task_description="Improve the agent harness.",
        metrics_context=_metrics_context(),
        skip_llm_for_clear_regressions=True,
        regression_skip_tolerance=0.000001,
        timeout=30.0,
    )
    stage.attach_inputs(
        {"metrics": FloatDictContainer(data={"fitness": 0.0, "is_valid": 1.0})}
    )

    result = await stage.execute(child)

    assert result.status == StageState.COMPLETED
    assert llm.messages is None
    assert "LLM Repo Insight" in result.output.data
    assert "Reflection Skip" in result.output.data
    reflection = child.metadata["repo_reflection"]
    assert reflection["llm_skipped"] is True
    assert reflection["reflection_status"] == "skipped"
    assert reflection["metrics_delta"]["fitness"] == -1.0
    assert reflection["attempt_record"]["verdict"] == "regressed"
    assert reflection["attempt_record"]["objective_primary_delta"] == -1.0
    assert "clear primary-metric regression" in reflection["skip_reason"]
    assert "Next Mutation Guidance" not in reflection["reflection"]


async def test_repo_reflection_stage_fails_open_with_fallback(tmp_path):
    repo = tmp_path / "agent_repo"
    commit = _init_repo(repo)
    manifest = RepoCandidateManifest(repo_path=str(repo), commit=commit)
    program = Program(code=manifest.to_program_code())
    program.add_metrics({"fitness": 1.0, "is_valid": 1.0})

    stage = RepoReflectionStage(
        llm=_FailingReflectionLLM(),
        storage=_FakeStorage(),
        task_description="Improve the agent harness.",
        metrics_context=_metrics_context(),
        timeout=30.0,
    )
    stage.attach_inputs(
        {"metrics": FloatDictContainer(data={"fitness": 1.0, "is_valid": 1.0})}
    )

    result = await stage.execute(program)

    assert result.status == StageState.COMPLETED
    assert "LLM reflection failed open" in result.output.data
    assert "reflection unavailable" in result.output.data
    assert program.metadata["repo_reflection"]["llm_error"] is not None
