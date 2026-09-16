from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "problems"
    / "repo_harness_template"
    / "terminal_bench2"
    / "benchmark.py"
)
SMOKE_PATH = BENCHMARK_PATH.with_name("test_terminal_bench2_smoke.py")
SEED_SPEC_PATH = BENCHMARK_PATH.with_name("repo_harness_seed.yaml")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    return _load_module("tb2_benchmark", BENCHMARK_PATH)


def test_local_llama_defaults_append_agent_kwargs_without_overriding_user_values():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        agent_kwarg=["max_turns=3"],
        local_llama_defaults=True,
        local_context_tokens=4096,
        local_output_tokens=512,
        local_response_tokens=384,
        local_summarization_threshold=1500,
        local_max_turns=8,
        local_temperature=0.1,
    )

    kwargs = benchmark.build_agent_kwargs(args)

    assert "max_turns=3" in kwargs
    assert (
        'model_info={"max_input_tokens":4096,"max_output_tokens":512,'
        '"input_cost_per_token":0,"output_cost_per_token":0}'
    ) in kwargs
    assert "proactive_summarization_threshold=1500" in kwargs
    assert "temperature=0.1" in kwargs
    assert 'llm_call_kwargs={"max_tokens":384}' in kwargs
    assert "max_turns=8" not in kwargs


def test_candidate_agent_config_overrides_local_llama_defaults(tmp_path):
    benchmark = _load_benchmark_module()
    candidate_repo = tmp_path / "candidate"
    candidate_repo.mkdir()
    (candidate_repo / "gigaevo_agent_config.json").write_text(
        json.dumps(
            {
                "agent_kwargs": {
                    "max_turns": 30,
                    "proactive_summarization_threshold": 6000,
                    "llm_call_kwargs": {"max_tokens": 2048},
                }
            }
        )
    )
    args = argparse.Namespace(
        agent_kwarg=[],
        local_llama_defaults=True,
        local_context_tokens=4096,
        local_output_tokens=512,
        local_response_tokens=384,
        local_summarization_threshold=1500,
        local_max_turns=8,
        local_temperature=0.1,
    )

    kwargs = benchmark.build_agent_kwargs(args, candidate_repo=candidate_repo)

    assert "max_turns=30" in kwargs
    assert "max_turns=8" not in kwargs
    assert "proactive_summarization_threshold=6000" in kwargs
    assert 'llm_call_kwargs={"max_tokens":2048}' in kwargs


def test_explicit_agent_kwarg_protects_against_candidate_agent_config(tmp_path):
    benchmark = _load_benchmark_module()
    candidate_repo = tmp_path / "candidate"
    candidate_repo.mkdir()
    (candidate_repo / "gigaevo_agent_config.json").write_text(
        json.dumps({"agent_kwargs": {"max_turns": 30}})
    )
    args = argparse.Namespace(
        agent_kwarg=["max_turns=12"],
        local_llama_defaults=True,
        local_context_tokens=4096,
        local_output_tokens=512,
        local_response_tokens=384,
        local_summarization_threshold=1500,
        local_max_turns=8,
        local_temperature=0.1,
    )

    kwargs = benchmark.build_agent_kwargs(args, candidate_repo=candidate_repo)

    assert "max_turns=12" in kwargs
    assert "max_turns=30" not in kwargs


def test_openrouter_defaults_override_local_model_unless_cli_supplied():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        openrouter=True,
        model="openai/qwen-local",
        api_base="http://127.0.0.1:8000/v1",
        api_key="local-key",
        openrouter_model="openrouter/openai/gpt-oss-120b",
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key="openrouter-key",
    )

    benchmark.apply_openrouter_defaults(args, set())

    assert args.model == "openrouter/openai/gpt-oss-120b"
    assert args.api_base == "https://openrouter.ai/api/v1"
    assert args.api_key == "openrouter-key"


def test_openrouter_defaults_preserve_explicit_cli_values():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        openrouter=True,
        model="openai/custom-model",
        api_base="https://proxy.example/v1",
        api_key="explicit-key",
        openrouter_model="openrouter/openai/gpt-oss-120b",
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key="openrouter-key",
    )

    benchmark.apply_openrouter_defaults(args, {"--model", "--api-base", "--api-key"})

    assert args.model == "openai/custom-model"
    assert args.api_base == "https://proxy.example/v1"
    assert args.api_key == "explicit-key"


def test_openrouter_env_sets_openrouter_and_openai_keys():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        env_file=None,
        dotenv_override=False,
        model="openrouter/openai/gpt-oss-120b",
        api_base="https://openrouter.ai/api/v1",
        api_key="openrouter-key",
        openrouter=True,
    )

    env = benchmark.build_env(args, Path("."))

    assert env["HARBOR_MODEL"] == "openrouter/openai/gpt-oss-120b"
    assert env["OPENAI_API_BASE"] == "https://openrouter.ai/api/v1"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert env["OPENAI_API_KEY"] == "openrouter-key"
    assert env["OPENROUTER_API_KEY"] == "openrouter-key"


def test_balanced20_task_set_is_deterministic_and_mixed():
    benchmark = _load_benchmark_module()
    args = benchmark.build_parser().parse_args(["--task-set", "balanced20"])

    tasks = benchmark.selected_tasks(args)

    assert tasks == [
        "fix-git",
        "extract-elf",
        "configure-git-webserver",
        "cobol-modernization",
        "log-summary-date-ranges",
        "cancel-async-tasks",
        "prove-plus-comm",
        "sqlite-db-truncate",
        "regex-chess",
        "overfull-hbox",
        "git-multibranch",
        "model-extraction-relu-logits",
        "code-from-image",
        "path-tracing",
        "mailman",
        "fix-code-vulnerability",
        "polyglot-c-py",
        "train-fasttext",
        "query-optimize",
        "llm-inference-batching-scheduler",
    ]
    assert len(tasks) == 20


def test_auto_seed_uses_terminus2_baseline_agent():
    import yaml

    spec = yaml.safe_load(SEED_SPEC_PATH.read_text(encoding="utf-8"))
    variant = spec["variants"]["minimal-agent"]
    content = variant["files"]["agents/baseline_terminus2.py"]["content"]
    config = json.loads(
        variant["files"]["gigaevo_agent_config.json"]["content"]
    )

    assert "from agents.terminus_2.terminus_2 import Terminus2" in content
    assert "class AgentHarness(Terminus2):" in content
    assert config["agent_kwargs"]["max_turns"] == 30
    assert config["agent_kwargs"]["llm_call_kwargs"]["max_tokens"] == 2048
    assert 'return "gigaevo-terminus2-seed"' in content
    assert "return None" not in content
    assert variant["package_files"] == [
        {
            "package": "harbor.agents.terminus_2",
            "destination": "agents/terminus_2",
            "include": ["*.py", "*.sh", "templates/*.txt"],
            "import_rewrites": {
                "harbor.agents.terminus_2": "agents.terminus_2",
            },
        }
    ]


def test_smoke_openrouter_does_not_forward_local_env_defaults():
    smoke = _load_module("tb2_smoke", SMOKE_PATH)
    args = smoke.build_parser().parse_args(["--openrouter"])
    args.model = "openai/qwen-local"
    args.api_base = "http://127.0.0.1:8000/v1"
    args.api_key = "local-key"
    args.supplied_options = {"--openrouter"}

    cmd = smoke.build_command(args, [])

    assert "--openrouter" in cmd
    assert "--model" not in cmd
    assert "--api-base" not in cmd
    assert "--api-key" not in cmd


def test_llama_cpp_broadcast_error_is_reported_as_context_error():
    benchmark = _load_benchmark_module()

    errors = benchmark.extract_context_errors(
        "ValueError: could not broadcast input array from shape (248320,) into shape (0,)"
    )

    assert errors == [
        {
            "type": "llama_cpp_context_overflow",
            "message": (
                "llama.cpp raised a broadcast-shape ValueError, which usually "
                "means the request exceeded the server n_ctx window before it "
                "could return a normal context-length error."
            ),
        }
    ]


def test_terminal_bench2_builds_generic_structured_feedback():
    benchmark = _load_benchmark_module()
    result = benchmark.CommandResult(returncode=0, stdout="", stderr="")
    records = [
        benchmark.TrialRecord(
            task="extract-elf",
            reward=0.0,
            cost_usd=0.01,
            result_path="/tmp/result.json",
        ),
        benchmark.TrialRecord(
            task="fix-git",
            reward=1.0,
            cost_usd=0.02,
            result_path="/tmp/fix-git/result.json",
        )
    ]
    failure_summaries = [
        {
            "task": "extract-elf",
            "trial": "extract-elf__1",
            "reward": 0.0,
            "exception_type": "AgentTimeoutError",
            "artifacts": [
                {
                    "name": "trial.log",
                    "path": "/tmp/trial.log",
                    "kind": "trial_log",
                }
            ],
            "mutation_hints": ["Shorten the agent loop"],
        }
    ]

    feedback = benchmark.build_structured_feedback(
        result=result,
        tasks=["extract-elf"],
        records=records,
        metrics={"fitness": 0.0, "is_valid": 1.0, "cost": 0.01},
        failure_summaries=failure_summaries,
        top_level_result={"n_errors": 1},
        task_set="smoke",
    )

    assert feedback["schema_version"] == 1
    assert feedback["benchmark"] == "terminal_bench_2"
    assert feedback["summary"]["selected_cases"] == ["extract-elf", "fix-git"]
    assert feedback["summary"]["failed_trials"] == 1
    assert feedback["summary"]["llm_call_failures"] == 0
    assert feedback["failure_clusters"] == [{"name": "extract-elf", "count": 1}]
    assert feedback["cases"] == [
        {
            "case_id": "extract-elf",
            "task": "extract-elf",
            "reward": 0.0,
            "passed": False,
            "status": "failed",
            "n_trials": 1,
            "mean_cost_usd": 0.01,
            "result_paths": ["/tmp/result.json"],
        },
        {
            "case_id": "fix-git",
            "task": "fix-git",
            "reward": 1.0,
            "passed": True,
            "status": "passed",
            "n_trials": 1,
            "mean_cost_usd": 0.02,
            "result_paths": ["/tmp/fix-git/result.json"],
        },
    ]
    assert feedback["examples"][0]["exception_type"] == "AgentTimeoutError"
    assert feedback["examples"][0]["artifacts"][0]["kind"] == "trial_log"
    assert feedback["hints"] == ["Shorten the agent loop"]


def test_terminal_bench2_trial_failure_summary_exposes_artifacts_and_trajectory(
    tmp_path,
):
    benchmark = _load_benchmark_module()
    trial_dir = tmp_path / "jobs" / "extract-elf__1"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "extract-elf",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_result": {
                    "n_input_tokens": 10,
                    "n_output_tokens": 20,
                    "n_cache_tokens": 0,
                    "cost_usd": None,
                    "metadata": {
                        "n_episodes": 1,
                        "summarization_count": 0,
                        "api_request_times_msec": [12.0],
                    },
                },
                "exception_info": {
                    "exception_type": "AgentTimeoutError",
                    "exception_message": "timed out",
                },
            }
        )
    )
    (trial_dir / "trial.log").write_text("trial log body\n")
    (trial_dir / "exception.txt").write_text("AgentTimeoutError: timed out\n")
    (trial_dir / "agent" / "terminus_2.pane").write_text("$ ls\n")
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "system", "message": "start"},
                    {
                        "source": "agent",
                        "message": "I will inspect the binary.",
                        "tool_calls": [
                            {"arguments": {"keystrokes": "file mystery\n"}},
                            {"arguments": {"keystrokes": "readelf -h mystery\n"}},
                        ],
                        "observation": {
                            "results": [
                                {"content": "ELF header output with wrong path"}
                            ]
                        },
                    },
                ]
            }
        )
    )
    (trial_dir / "verifier" / "test-stdout.txt").write_text("missing answer file\n")
    (trial_dir / "verifier" / "ctrf.json").write_text(json.dumps({"tests": []}))

    summary = benchmark.extract_trial_failure_summary(trial_dir)

    artifact_kinds = {item["kind"] for item in summary["artifacts"]}
    assert artifact_kinds == {
        "trial_result",
        "trial_log",
        "exception",
        "agent_trajectory",
        "terminal_pane",
        "verifier_stdout",
        "verifier_result",
    }
    episode = summary["episode_summary"]
    assert episode["trajectory_step_count"] == 2
    assert episode["agent_step_count"] == 1
    assert episode["last_commands"] == [
        "file mystery\n",
        "readelf -h mystery\n",
    ]
    assert "wrong path" in episode["last_observation_excerpt"]


def test_llm_internal_errors_make_metrics_invalid():
    benchmark = _load_benchmark_module()
    records = [
        benchmark.TrialRecord(
            task="fix-git",
            reward=1.0,
            cost_usd=None,
            result_path="/tmp/fix-git/result.json",
        ),
        benchmark.TrialRecord(
            task="extract-elf",
            reward=0.0,
            cost_usd=None,
            result_path="/tmp/extract-elf/result.json",
        ),
    ]
    failure_summaries = [
        {
            "task": "extract-elf",
            "exception_type": "InternalServerError",
            "exception_message": (
                "litellm.InternalServerError: InternalServerError: "
                "OpenAIException - Connection error."
            ),
        }
    ]

    metrics = benchmark.compute_metrics(
        records,
        returncode=0,
        failure_summaries=failure_summaries,
    )

    assert metrics == {"fitness": 0.5, "is_valid": 0.0, "cost": 0.0}


def test_docker_infrastructure_errors_make_metrics_invalid():
    benchmark = _load_benchmark_module()
    records = [
        benchmark.TrialRecord(
            task="extract-elf",
            reward=0.0,
            cost_usd=None,
            result_path="/tmp/extract-elf/result.json",
        )
    ]
    failure_summaries = [
        {
            "task": "extract-elf",
            "trial": "extract-elf__abc1234",
            "exception_type": "RuntimeError",
            "exception_message": (
                "Docker compose command failed for environment extract-elf. "
                "failed to create network extract-elf__abc1234_default: "
                "Error response from daemon: all predefined address pools "
                "have been fully subnetted"
            ),
        }
    ]

    metrics = benchmark.compute_metrics(
        records,
        returncode=0,
        failure_summaries=failure_summaries,
    )
    feedback = benchmark.build_structured_feedback(
        result=benchmark.CommandResult(returncode=0, stdout="", stderr=""),
        tasks=["extract-elf"],
        records=records,
        metrics=metrics,
        failure_summaries=failure_summaries,
        top_level_result={"n_errors": 1},
        task_set="smoke",
    )

    assert metrics == {"fitness": 0.0, "is_valid": 0.0, "cost": 0.0}
    assert feedback["summary"]["infrastructure_failures"] == 1
    assert feedback["examples"][0]["infrastructure_failed"] is True


def test_non_llm_agent_timeout_can_remain_valid():
    benchmark = _load_benchmark_module()
    records = [
        benchmark.TrialRecord(
            task="fix-git",
            reward=0.0,
            cost_usd=None,
            result_path="/tmp/fix-git/result.json",
        )
    ]
    failure_summaries = [
        {
            "task": "fix-git",
            "exception_type": "AgentTimeoutError",
            "exception_message": "Agent execution timed out after 900.0 seconds",
        }
    ]

    metrics = benchmark.compute_metrics(
        records,
        returncode=0,
        failure_summaries=failure_summaries,
    )

    assert metrics == {"fitness": 0.0, "is_valid": 1.0, "cost": 0.0}


def test_structured_feedback_counts_llm_call_failures():
    benchmark = _load_benchmark_module()
    result = benchmark.CommandResult(returncode=0, stdout="", stderr="")
    records = [
        benchmark.TrialRecord(
            task="regex-log",
            reward=0.0,
            cost_usd=None,
            result_path="/tmp/regex-log/result.json",
        )
    ]
    failure_summaries = [
        {
            "task": "regex-log",
            "trial": "regex-log__1",
            "reward": None,
            "exception_type": "APIConnectionError",
            "exception_message": "Cannot connect to host 127.0.0.1:8001",
            "mutation_hints": ["Inspect LLM endpoint health"],
        }
    ]

    feedback = benchmark.build_structured_feedback(
        result=result,
        tasks=["regex-log"],
        records=records,
        metrics=benchmark.compute_metrics(
            records,
            returncode=0,
            failure_summaries=failure_summaries,
        ),
        failure_summaries=failure_summaries,
        top_level_result={"n_errors": 1},
        task_set="smoke",
    )

    assert feedback["summary"]["metrics"]["is_valid"] == 0.0
    assert feedback["summary"]["llm_call_failures"] == 1
    assert feedback["examples"][0]["llm_call_failed"] is True


def test_docker_cleanup_discovers_job_compose_projects(tmp_path):
    benchmark = _load_benchmark_module()
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "extract-elf__AbC1234"
    trial_dir.mkdir(parents=True)
    (trial_dir / "exception.txt").write_text(
        "docker compose --project-name fix-git__XYZ987 up failed"
    )

    projects = benchmark.docker_compose_projects_from_job_dir(job_dir)

    assert projects == {"extract-elf__abc1234", "fix-git__xyz987"}


def test_runtime_limit_hint_uses_effective_max_turns():
    benchmark = _load_benchmark_module()
    feedback = {"hints": []}

    benchmark.add_runtime_limit_hints(
        feedback,
        failure_summaries=[
            {
                "task": "regex-log",
                "agent_metrics": {"n_episodes": 30},
            }
        ],
        agent_config={"effective_agent_kwargs": {"max_turns": "30"}},
    )

    assert "max_turns=30" in feedback["hints"][0]
    assert "regex-log" in feedback["hints"][0]
