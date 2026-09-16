from __future__ import annotations

import json

import pytest

from gigaevo.llm.codex_usage import (
    ensure_codex_json_flag,
    estimate_cost_usd,
    parse_codex_jsonl,
    summarize_usage_records,
)
from gigaevo.llm.command_cli import CommandCLIChatModel
from gigaevo.repo_harness.backends import CommandCodingAgentBackend


def _codex_jsonl() -> str:
    events = [
        {"type": "thread.started", "thread_id": "thread-123"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "finished"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "output_tokens": 200,
                "reasoning_output_tokens": 50,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def test_parse_codex_jsonl_and_estimate_standard_cost():
    content, usage = parse_codex_jsonl(_codex_jsonl(), model_hint="gpt-5.6-sol")

    assert content == "finished"
    assert usage is not None
    assert usage["input_tokens"] == 1000
    assert usage["cached_input_tokens"] == 400
    assert usage["uncached_input_tokens"] == 600
    assert usage["output_tokens"] == 200
    assert usage["reasoning_output_tokens"] == 50
    assert usage["total_tokens"] == 1200
    assert usage["estimated_cost_usd"] == pytest.approx(0.0092)


def test_estimate_cost_does_not_double_charge_cached_or_reasoning_tokens():
    cost, _ = estimate_cost_usd(
        model="gpt-5.6-sol",
        input_tokens=1000,
        cached_input_tokens=1000,
        output_tokens=100,
    )

    assert cost == pytest.approx(0.0035)


def test_summarize_usage_deduplicates_embedded_ledger_records():
    _, usage = parse_codex_jsonl(_codex_jsonl(), model_hint="gpt-5.6-sol")
    assert usage is not None
    usage["generation"] = 2

    result = summarize_usage_records([usage, dict(usage)])

    assert result["summary"]["calls"] == 1
    assert result["summary"]["total_tokens"] == 1200
    assert result["by_generation"][0]["generation"] == 2


def test_ensure_codex_json_flag_is_idempotent():
    assert ensure_codex_json_flag(["codex", "exec", "--ephemeral", "prompt"]) == [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "prompt",
    ]
    assert ensure_codex_json_flag(["codex", "exec", "--json", "prompt"]) == [
        "codex",
        "exec",
        "--json",
        "prompt",
    ]


def test_command_cli_exposes_usage_metadata_and_appends_ledger(tmp_path):
    payload = _codex_jsonl()
    model = CommandCLIChatModel(
        command=["python", "-c", "import sys; print(sys.argv[1])", payload],
        model_name="gpt-5.6-sol",
        provider_name="codex",
        json_output=True,
        usage_log_path=tmp_path / "codex_usage.jsonl",
    )

    response = model.invoke("ignored")

    assert response.content == "finished"
    assert response.response_metadata["token_usage"]["total_tokens"] == 1200
    records = [
        json.loads(line)
        for line in (tmp_path / "codex_usage.jsonl").read_text().splitlines()
    ]
    assert records[0]["estimated_cost_usd"] == pytest.approx(0.0092)


async def test_coding_agent_backend_keeps_raw_jsonl_and_returns_final_message(
    tmp_path,
):
    payload = _codex_jsonl().replace("{", "{{").replace("}", "}}")
    backend = CommandCodingAgentBackend(
        command=["python", "-c", "import sys; print(sys.argv[1])", payload],
        name="codex-test",
    )

    run = await backend.run(
        prompt="ignored",
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        timeout=10,
    )

    assert run.stdout == "finished"
    assert run.usage is not None
    assert run.usage["total_tokens"] == 1200
    assert "turn.completed" in (tmp_path / "logs" / "stdout.txt").read_text()
    usage_file = json.loads((tmp_path / "logs" / "codex_usage.json").read_text())
    assert usage_file["estimated_cost_usd"] is None
