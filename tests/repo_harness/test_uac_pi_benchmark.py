from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess

from problems.uac_pi_harness.uac_pi import (
    benchmark,
    mutation_runner,
    pi_runner,
)
from problems.uac_pi_harness.uac_pi import (
    test_uac_pi_smoke as smoke,
)


def write_result(
    trial_dir: Path,
    *,
    task: str,
    reward: object,
    metadata: dict[str, object] | None = None,
) -> None:
    trial_dir.mkdir(parents=True)
    payload = {
        "task_name": f"local/{task}",
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {
            "cost_usd": 0,
            "n_input_tokens": 10,
            "n_output_tokens": 5,
            "metadata": metadata or {},
        },
    }
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def command_result(tmp_path: Path, returncode: int = 0) -> benchmark.CommandResult:
    return benchmark.CommandResult(
        returncode=returncode,
        timed_out=False,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )


def test_private_manifest_has_37_unique_direct_task_ids() -> None:
    tasks = benchmark.read_task_manifest(benchmark.PRIVATE_TASKS_FILE)
    assert len(tasks) == 37
    assert len(set(tasks)) == 37
    assert all(benchmark.TASK_ID_RE.fullmatch(task) for task in tasks)
    assert "forensics-cloud-oauth-pivot" in tasks


def test_explicit_empty_task_manifest_fails_instead_of_using_private37(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "empty.txt"
    manifest.write_text("# no tasks selected\n", encoding="utf-8")
    args = benchmark.build_parser().parse_args(["--tasks-file", str(manifest)])
    try:
        benchmark.select_tasks(args)
    except benchmark.BenchmarkError as exc:
        assert "zero task ids" in str(exc)
    else:
        raise AssertionError("empty explicit manifest fell back to private37")


def test_combined_explicit_task_duplicates_are_rejected() -> None:
    args = benchmark.build_parser().parse_args(
        ["--task", "task-a", "--tasks", "task-a,task-b"]
    )
    try:
        benchmark.select_tasks(args)
    except benchmark.BenchmarkError as exc:
        assert "duplicates" in str(exc)
        assert "task-a" in str(exc)
    else:
        raise AssertionError("duplicate explicit task selectors were accepted")


def test_models_config_uses_exact_local_model_and_qwen_compat() -> None:
    config = pi_runner.build_models_config(
        base_url="http://172.18.0.1:8001/v1",
        api_key="dummy",
        model="Qwen/Qwen3.6-35B-A3B",
        context_window=32768,
        max_tokens=8192,
        thinking="medium",
    )
    provider = config["providers"]["uac-local"]
    model = provider["models"][0]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == "http://172.18.0.1:8001/v1"
    assert provider["compat"]["thinkingFormat"] == "qwen-chat-template"
    assert provider["compat"]["maxTokensField"] == "max_tokens"
    assert model["id"] == "Qwen/Qwen3.6-35B-A3B"
    assert model["contextWindow"] == 32768
    assert model["maxTokens"] == 8192


def test_models_config_disables_thinking_compat_when_off() -> None:
    config = pi_runner.build_models_config(
        base_url="http://model:8001/v1",
        api_key="dummy",
        model="Qwen/Qwen3.6-35B-A3B",
        context_window=32768,
        max_tokens=8192,
        thinking="off",
    )
    provider = config["providers"]["uac-local"]
    assert "thinkingFormat" not in provider["compat"]
    assert provider["models"][0]["reasoning"] is False


def test_loopback_api_base_rewrites_to_docker_gateway(monkeypatch) -> None:
    monkeypatch.setattr(pi_runner, "default_gateway", lambda: "172.19.0.1")
    assert (
        pi_runner.resolve_api_base("http://127.0.0.1:8001/v1")
        == "http://172.19.0.1:8001/v1"
    )


def test_proc_route_fallback_reads_little_endian_gateway(tmp_path: Path) -> None:
    route = tmp_path / "route"
    route.write_text(
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        "eth0 00000000 010013AC 0003 0 0 0 00000000 0 0 0\n",
        encoding="ascii",
    )
    assert pi_runner.proc_default_gateway(route) == "172.19.0.1"


def test_streaming_preflight_forces_and_accepts_named_tool(monkeypatch) -> None:
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_health",
                            "type": "function",
                            "function": {
                                "name": "health_check",
                                "arguments": "{}",
                            },
                        }
                    ]
                }
            }
        ]
    }
    response = BytesIO(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())

    def fake_urlopen(req, timeout):
        assert timeout == 90.0
        payload = json.loads(req.data)
        assert payload["stream"] is True
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "health_check"},
        }
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return response

    monkeypatch.setattr(smoke.request, "urlopen", fake_urlopen)
    smoke.preflight_streaming_tools(
        api_base="http://127.0.0.1:8001/v1",
        api_key="dummy",
        model="Qwen/Qwen3.6-35B-A3B",
    )


def test_smoke_benchmark_command_keeps_api_key_out_of_argv() -> None:
    args = smoke.build_parser().parse_args(["--api-key", "super-secret"])

    command = smoke.build_command(args, [])

    assert "super-secret" not in command
    assert "--api-key" not in command


def test_finalized_trace_discards_cumulative_stream_updates() -> None:
    cumulative = {
        "role": "assistant",
        "content": [{"type": "text", "text": "large cumulative response"}],
    }
    update = {
        "type": "message_update",
        "message": cumulative,
        "assistantMessageEvent": {
            "type": "text_delta",
            "delta": "e",
            "partial": cumulative,
        },
    }
    assert pi_runner.finalized_trace_event(json.dumps(update)) is None
    assert (
        pi_runner.finalized_trace_event(
            json.dumps({"type": "tool_execution_update", "partialResult": cumulative})
        )
        is None
    )


def test_finalized_trace_keeps_completed_messages_and_compacts_run_end() -> None:
    message_end = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "final response"}],
            "stopReason": "stop",
        },
    }
    assert pi_runner.finalized_trace_event(json.dumps(message_end)) == message_end
    assert pi_runner.finalized_trace_event(
        json.dumps(
            {
                "type": "agent_end",
                "messages": [message_end["message"]],
                "willRetry": False,
            }
        )
    ) == {"type": "agent_end", "willRetry": False}
    compaction = {
        "type": "compaction_end",
        "reason": "threshold",
        "aborted": False,
    }
    assert pi_runner.finalized_trace_event(json.dumps(compaction)) == compaction


def test_event_summary_detects_success_and_accumulates_usage(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "usage": {
                        "input": 11,
                        "output": 7,
                        "totalTokens": 18,
                        "cost": {"total": 0},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    usage, summary = pi_runner.summarize_events(events, 0)
    assert summary["completed"] is True
    assert summary["failed"] is False
    assert summary["assistant_messages"] == 1
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_event_summary_reads_incrementally(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}\n',
        encoding="utf-8",
    )

    def fail_read_text(*args, **kwargs):
        raise AssertionError("summarize_events must not load the full trace")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    _, summary = pi_runner.summarize_events(events, 0)
    assert summary["completed"] is True


def test_event_summary_rejects_assistant_error(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "errorMessage": "endpoint disconnected",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _, summary = pi_runner.summarize_events(events, 0)
    assert summary["completed"] is False
    assert summary["failed"] is True
    assert "endpoint disconnected" in summary["failure"]


def test_event_summary_accepts_success_after_retried_error(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "error",
                            "errorMessage": "temporarily overloaded",
                        },
                    }
                ),
                json.dumps({"type": "auto_retry_start", "attempt": 1}),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "stop",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _, summary = pi_runner.summarize_events(events, 0)
    assert summary["completed"] is True
    assert summary["failed"] is False
    assert summary["last_stop_reason"] == "stop"
    assert summary["failure"] is None


def test_event_summary_rejects_missing_assistant_message(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text('{"type":"agent_start"}\n', encoding="utf-8")
    _, summary = pi_runner.summarize_events(events, 0)
    assert summary["failed"] is True
    assert "no completed assistant message" in summary["failure"]


def test_event_summary_rejects_nonzero_process_exit(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}\n',
        encoding="utf-8",
    )
    _, summary = pi_runner.summarize_events(events, 137)
    assert summary["failed"] is True
    assert "exited with code 137" in summary["failure"]


def test_reward_zero_is_a_valid_completed_trial(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task-a__one"
    write_result(trial_dir, task="task-a", reward=0)
    record = benchmark.parse_trial_record(trial_dir)
    assert record.valid is True
    assert record.reward == 0.0
    assert record.status == "failed"


def test_pi_json_failure_invalidates_reward(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task-a__one"
    write_result(
        trial_dir,
        task="task-a",
        reward=0,
        metadata={"failed": True, "exit_code": 1, "failure": "API connection error"},
    )
    record = benchmark.parse_trial_record(trial_dir)
    assert record.valid is False
    assert record.status == "agent_failure"
    assert "API connection error" in str(record.error)


def test_expected_agent_timeout_remains_a_valid_task_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task-a__one"
    write_result(trial_dir, task="task-a", reward=0)
    (trial_dir / "exception.txt").write_text(
        "AgentTimeoutError: agent timed out after 1800 seconds",
        encoding="utf-8",
    )
    record = benchmark.parse_trial_record(trial_dir)
    assert record.valid is True
    assert record.reward == 0.0
    assert record.status == "agent_timeout"


def test_agent_timeout_without_result_is_still_a_valid_zero(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task-a__one"
    trial_dir.mkdir()
    (trial_dir / "exception.txt").write_text(
        "AgentTimeoutError: agent execution timed out",
        encoding="utf-8",
    )
    record = benchmark.parse_trial_record(trial_dir)
    assert record.valid is True
    assert record.reward == 0.0
    assert record.status == "agent_timeout"


def test_trial_exception_takes_priority_over_missing_verifier_reward(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "task-a__one"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "local/task-a",
                "verifier_result": None,
                "exception_info": {"exception_type": "RuntimeError"},
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "exception.txt").write_text(
        "RuntimeError: model proxy is unhealthy",
        encoding="utf-8",
    )

    record = benchmark.parse_trial_record(trial_dir)

    assert record.valid is False
    assert record.status == "trial_exception"
    assert "model proxy is unhealthy" in str(record.error)


def test_missing_result_uses_trial_config_to_recover_truncated_task_id(
    tmp_path: Path,
) -> None:
    task = "forensics-postgres-replication-exfil"
    truncated = task[:32].rstrip("_-")
    trial_dir = tmp_path / f"{truncated}__abc1234"
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": f"/benchmarks/local_task/{task}"}}),
        encoding="utf-8",
    )
    (trial_dir / "exception.txt").write_text(
        "AgentTimeoutError: agent execution timed out",
        encoding="utf-8",
    )

    record = benchmark.parse_trial_record(trial_dir)

    assert record.task == task
    assert record.valid is True
    assert record.reward == 0.0


def test_negative_pi_exit_code_invalidates_reward(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task-a__one"
    write_result(
        trial_dir,
        task="task-a",
        reward=0,
        metadata={"exit_code": -9},
    )
    record = benchmark.parse_trial_record(trial_dir)
    assert record.valid is False
    assert record.status == "agent_failure"


def test_partial_run_is_valid_and_missing_trial_stays_in_denominator(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "task-a__one"
    write_result(trial_dir, task="task-a", reward=1)
    evaluation = benchmark.evaluate_records(
        tasks=["task-a", "task-b"],
        runs=1,
        records=[benchmark.parse_trial_record(trial_dir)],
        harbor_result=command_result(tmp_path),
        elapsed_seconds=2.0,
    )
    assert evaluation.metrics["fitness"] == 0.5
    assert evaluation.metrics["is_valid"] == 1.0
    assert evaluation.metrics["solved_trials"] == 1.0
    assert evaluation.metrics["completed_trials"] == 1.0
    assert evaluation.metrics["expected_trials"] == 2.0
    assert evaluation.metrics["invalid_trials"] == 1.0


def test_run_is_invalid_only_when_no_trial_is_usable(tmp_path: Path) -> None:
    evaluation = benchmark.evaluate_records(
        tasks=["task-a", "task-b"],
        runs=1,
        records=[],
        harbor_result=command_result(tmp_path, returncode=1),
        elapsed_seconds=2.0,
    )
    assert evaluation.metrics["fitness"] == 0.0
    assert evaluation.metrics["is_valid"] == 0.0
    assert evaluation.metrics["completed_trials"] == 0.0
    assert evaluation.metrics["expected_trials"] == 2.0
    assert evaluation.metrics["invalid_trials"] == 2.0


def test_all_binary_results_are_valid_even_with_reward_zero(tmp_path: Path) -> None:
    first = tmp_path / "task-a__one"
    second = tmp_path / "task-b__one"
    write_result(first, task="task-a", reward=1)
    write_result(second, task="task-b", reward=0)
    evaluation = benchmark.evaluate_records(
        tasks=["task-a", "task-b"],
        runs=1,
        records=[
            benchmark.parse_trial_record(first),
            benchmark.parse_trial_record(second),
        ],
        harbor_result=command_result(tmp_path),
        elapsed_seconds=2.0,
    )
    assert evaluation.metrics["fitness"] == 0.5
    assert evaluation.metrics["is_valid"] == 1.0
    assert evaluation.metrics["invalid_trials"] == 0.0


def test_harbor_command_redaction_removes_api_key(tmp_path: Path) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--tasks-root",
            str(tmp_path),
            "--api-key",
            "super-secret",
            "--runner",
            "uv run",
        ]
    )
    command = benchmark.build_harbor_command(
        args,
        tasks=["task-a"],
        job_name="job",
        jobs_dir=tmp_path,
    )
    rendered = " ".join(benchmark.redact_command(command, args.api_key))
    assert "super-secret" not in rendered
    assert "OPENAI_API_KEY=***" in rendered
    assert benchmark.DEFAULT_ENVIRONMENT in command
    assert "--extra-docker-compose" in command
    assert str(benchmark.MODEL_PROXY_COMPOSE) in command
    assert "--allow-agent-host" in command
    assert benchmark.DEFAULT_AGENT_ALLOWED_HOST in command
    assert "super-secret" not in command


def test_build_contract_rejects_protected_file_changes(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "package.json").write_text('{"name":"pi"}', encoding="utf-8")
    (candidate / "package.json").write_text('{"name":"changed"}', encoding="utf-8")
    monkeypatch.setattr(benchmark, "PROTECTED_BUILD_FILES", ("package.json",))
    try:
        benchmark.validate_build_contract(candidate, baseline)
    except benchmark.CandidateBuildError as exc:
        assert "package.json" in str(exc)
    else:
        raise AssertionError("protected build-contract mutation was accepted")


def test_build_contract_rejects_added_protected_script(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "package.json").write_text('{"name":"pi"}', encoding="utf-8")
    (candidate / "package.json").write_text('{"name":"pi"}', encoding="utf-8")
    script = candidate / "scripts" / "unexpected.mjs"
    script.parent.mkdir()
    script.write_text("process.exit(0);\n", encoding="utf-8")
    try:
        benchmark.validate_build_contract(candidate, baseline)
    except benchmark.CandidateBuildError as exc:
        assert "scripts/unexpected.mjs" in str(exc)
    else:
        raise AssertionError("candidate-added protected script was accepted")


def test_build_contract_protects_root_gitignore(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (candidate / ".gitignore").write_text("!node_modules/\n", encoding="utf-8")

    try:
        benchmark.validate_build_contract(candidate, baseline)
    except benchmark.CandidateBuildError as exc:
        assert ".gitignore" in str(exc)
    else:
        raise AssertionError("candidate changed the protected root .gitignore")


def test_build_contract_ignores_installed_workspace_dependencies(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for root in (baseline, candidate):
        package = root / "packages" / "agent"
        package.mkdir(parents=True)
        (package / "package.json").write_text(
            '{"name":"agent"}', encoding="utf-8"
        )
    installed = (
        baseline
        / "packages"
        / "agent"
        / "node_modules"
        / "@types"
        / "node"
        / "package.json"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text('{"name":"@types/node"}', encoding="utf-8")

    benchmark.validate_build_contract(candidate, baseline)


def test_mutation_cleanup_unstages_forced_ignored_dependencies(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "config", "user.email", "uac-pi@example.invalid"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "UAC Pi Test"],
        cwd=worktree,
        check=True,
    )
    (worktree / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    source = worktree / "source.ts"
    source.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=worktree, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
    ).strip()

    source.write_text("after\n", encoding="utf-8")
    dependency = worktree / "node_modules" / "forced.txt"
    dependency.parent.mkdir()
    dependency.write_text("do not commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.ts"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "add", "-f", "node_modules/forced.txt"],
        cwd=worktree,
        check=True,
    )

    mutation_runner.clean_mutation_index(worktree, head)

    cached = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=worktree, check=False
    )
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=worktree, text=True
    )
    assert cached.returncode == 0
    assert "source.ts" in status
    assert "node_modules" not in status


def test_candidate_tree_rejects_symlink_outside_worktree(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "escape").symlink_to(tmp_path / "outside")
    try:
        benchmark.validate_candidate_tree_safety(candidate)
    except benchmark.CandidateBuildError as exc:
        assert "escapes its worktree" in str(exc)
    else:
        raise AssertionError("external candidate symlink was accepted")


def test_candidate_dependency_tree_is_replaced_from_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for root in (baseline, candidate):
        root.mkdir()
        (root / "package.json").write_text('{"name":"pi"}', encoding="utf-8")
    baseline_tsc = baseline / "node_modules" / ".bin" / "tsc"
    baseline_tsc.parent.mkdir(parents=True)
    baseline_tsc.write_text("trusted\n", encoding="utf-8")
    candidate_tsc = candidate / "node_modules" / ".bin" / "tsc"
    candidate_tsc.parent.mkdir(parents=True)
    candidate_tsc.write_text("modified\n", encoding="utf-8")
    (candidate / "node_modules" / "injected").write_text(
        "untrusted\n", encoding="utf-8"
    )
    baseline_workspace_dependency = (
        baseline / "packages" / "agent" / "node_modules" / "nested.txt"
    )
    baseline_workspace_dependency.parent.mkdir(parents=True)
    baseline_workspace_dependency.write_text("trusted nested\n", encoding="utf-8")
    candidate_workspace_dependency = (
        candidate / "packages" / "agent" / "node_modules" / "injected.txt"
    )
    candidate_workspace_dependency.parent.mkdir(parents=True)
    candidate_workspace_dependency.write_text("untrusted nested\n", encoding="utf-8")

    benchmark.ensure_candidate_node_modules(candidate, baseline)

    assert candidate_tsc.read_text(encoding="utf-8") == "trusted\n"
    assert not (candidate / "node_modules" / "injected").exists()
    assert (
        candidate / "packages" / "agent" / "node_modules" / "nested.txt"
    ).read_text(encoding="utf-8") == "trusted nested\n"
    assert not candidate_workspace_dependency.exists()


def test_dependency_fingerprint_changes_with_installed_lock(tmp_path: Path) -> None:
    root = tmp_path / "pi"
    installed_lock = root / "node_modules" / ".package-lock.json"
    installed_lock.parent.mkdir(parents=True)
    (root / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    installed_lock.write_text('{"packages":{}}', encoding="utf-8")
    first = benchmark.dependency_fingerprint(root)
    installed_lock.write_text('{"packages":{"node_modules/x":{}}}', encoding="utf-8")
    assert benchmark.dependency_fingerprint(root) != first


def test_candidate_model_data_is_copied_from_dependency_root(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    source = baseline / benchmark.MODEL_DATA_RELATIVE
    target = candidate / benchmark.MODEL_DATA_RELATIVE
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / ".manifest.json").write_text('{"version":1}', encoding="utf-8")
    (source / "openai.json").write_text('{"model":{}}', encoding="utf-8")
    (target / "injected.json").write_text('{"bad":true}', encoding="utf-8")

    benchmark.ensure_candidate_model_data(candidate, baseline)

    assert (target / ".manifest.json").is_file()
    assert (target / "openai.json").read_text(encoding="utf-8") == '{"model":{}}'
    assert not (target / "injected.json").exists()


def test_candidate_model_data_requires_hydrated_manifest(tmp_path: Path) -> None:
    root = tmp_path / "pi"
    (root / benchmark.MODEL_DATA_RELATIVE).mkdir(parents=True)
    try:
        benchmark.ensure_candidate_model_data(root, root)
    except benchmark.CandidateBuildError as exc:
        assert "hydrate:model-data" in str(exc)
    else:
        raise AssertionError("missing offline model-data manifest was accepted")


def test_dependency_fingerprint_changes_with_model_data(tmp_path: Path) -> None:
    root = tmp_path / "pi"
    data = root / benchmark.MODEL_DATA_RELATIVE
    data.mkdir(parents=True)
    (data / ".manifest.json").write_text('{"version":1}', encoding="utf-8")
    provider = data / "openai.json"
    provider.write_text('{"a":1}', encoding="utf-8")
    first = benchmark.dependency_fingerprint(root)
    provider.write_text('{"a":2}', encoding="utf-8")
    assert benchmark.dependency_fingerprint(root) != first


def test_default_model_proxy_compose_isolates_main_from_upstream() -> None:
    compose = (benchmark.HERE / "model_proxy.compose.yaml").read_text(encoding="utf-8")
    assert "uac-pi-model-proxy:" in compose
    assert "host.docker.internal:host-gateway" in compose
    assert "\n  main:" not in compose
    assert "default:\n    internal: true" in compose
    assert "uac-pi-upstream:" in compose


def test_build_environment_drops_keys_tokens_and_proxies(monkeypatch) -> None:
    monkeypatch.setenv("UAC_PI_API_KEY", "secret")
    monkeypatch.setenv("EXAMPLE_TOKEN", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:password@proxy")
    monkeypatch.setenv("SAFE_BUILD_VALUE", "kept")
    env = benchmark.sanitized_build_env()
    assert "UAC_PI_API_KEY" not in env
    assert "EXAMPLE_TOKEN" not in env
    assert "HTTPS_PROXY" not in env
    assert env["SAFE_BUILD_VALUE"] == "kept"
    assert env["npm_config_offline"] == "true"


def test_parse_only_main_emits_metrics_and_summary(tmp_path: Path, capsys) -> None:
    tasks_root = tmp_path / "tasks"
    for task in ("task-a", "task-b"):
        task_dir = tasks_root / task
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text("[task]\n", encoding="utf-8")
    job_dir = tmp_path / "job"
    write_result(job_dir / "task-a__one", task="task-a", reward=1)
    write_result(job_dir / "task-b__one", task="task-b", reward=0)
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    returncode = benchmark.main(
        [
            "--candidate-repo",
            str(candidate),
            "--tasks-root",
            str(tasks_root),
            "--tasks",
            "task-a,task-b",
            "--parse-job-dir",
            str(job_dir),
            "--job-name",
            "synthetic",
        ]
    )

    captured = capsys.readouterr()
    metrics = json.loads(captured.out.strip().splitlines()[-1])
    assert returncode == 0
    assert metrics["fitness"] == 0.5
    assert metrics["is_valid"] == 1.0
    summary = json.loads((job_dir / "gigaevo_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_tasks"] == ["task-a", "task-b"]
    assert benchmark.STRUCTURED_FEEDBACK_MARKER in captured.err
