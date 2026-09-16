from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import threading
import tomllib
from typing import Any

CODEX_USAGE_SCHEMA_VERSION = 1
CODEX_USAGE_FILENAME = "codex_usage.jsonl"
PRICING_VERSION = "2026-08-09"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


@dataclass(frozen=True)
class _RateCard:
    input: float
    cached_input: float
    output: float
    cache_write: float | None = None
    long_input: float | None = None
    long_cached_input: float | None = None
    long_output: float | None = None
    long_cache_write: float | None = None


# USD per one million tokens. Keep this deliberately explicit: prices are
# external policy, not a model-name heuristic. Unknown models remain unpriced.
_STANDARD_RATES: dict[str, _RateCard] = {
    "gpt-5.6-sol": _RateCard(5.00, 0.50, 30.00, 6.25, 10.00, 1.00, 45.00, 12.50),
    "gpt-5.6-terra": _RateCard(2.00, 0.20, 12.00, 2.50, 4.00, 0.40, 18.00, 5.00),
    "gpt-5.6-luna": _RateCard(0.20, 0.02, 1.20, 0.25, 0.40, 0.04, 1.80, 0.50),
    "gpt-5.5": _RateCard(
        5.00, 0.50, 30.00, long_input=10.00, long_cached_input=1.00, long_output=45.00
    ),
    "gpt-5.4": _RateCard(
        2.50, 0.25, 15.00, long_input=5.00, long_cached_input=0.50, long_output=22.50
    ),
    "gpt-5.4-mini": _RateCard(0.75, 0.075, 4.50),
    "gpt-5.4-nano": _RateCard(0.20, 0.02, 1.25),
    "gpt-5.2": _RateCard(1.75, 0.175, 14.00),
    "gpt-5.1": _RateCard(1.25, 0.125, 10.00),
    "gpt-5": _RateCard(1.25, 0.125, 10.00),
    "gpt-5-mini": _RateCard(0.25, 0.025, 2.00),
    "gpt-5-nano": _RateCard(0.05, 0.005, 0.40),
}

_FAST_RATES: dict[str, _RateCard] = {
    "gpt-5.6-sol": _RateCard(10.00, 1.00, 60.00, 12.50, 20.00, 2.00, 90.00, 25.00),
    "gpt-5.6-terra": _RateCard(4.00, 0.40, 24.00, 5.00, 8.00, 0.80, 36.00, 10.00),
    "gpt-5.6-luna": _RateCard(0.40, 0.04, 2.40, 0.50, 0.80, 0.08, 3.60, 1.00),
    "gpt-5.5": _RateCard(12.50, 1.25, 75.00),
    "gpt-5.4": _RateCard(5.00, 0.50, 30.00),
    "gpt-5.4-mini": _RateCard(1.50, 0.15, 9.00),
    "gpt-5.2": _RateCard(3.50, 0.35, 28.00),
    "gpt-5.1": _RateCard(2.50, 0.25, 20.00),
    "gpt-5": _RateCard(2.50, 0.25, 20.00),
    "gpt-5-mini": _RateCard(0.45, 0.045, 3.60),
}

_USAGE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "codex_usage_context", default={}
)
_LEDGER_LOCK = threading.Lock()


@contextmanager
def codex_usage_scope(**fields: Any) -> Iterator[None]:
    """Attach source/generation/program context to a Codex invocation."""
    current = dict(_USAGE_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _USAGE_CONTEXT.set(current)
    try:
        yield
    finally:
        _USAGE_CONTEXT.reset(token)


def current_usage_context() -> dict[str, Any]:
    return dict(_USAGE_CONTEXT.get())


def normalize_model_name(model: str | None) -> str | None:
    if not model:
        return None
    name = str(model).strip()
    for prefix in ("openai/", "openai-codex/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    aliases = {
        "gpt-5.6": "gpt-5.6-sol",
    }
    return aliases.get(name, name) or None


def _to_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _rate_card(model: str | None, service_tier: str) -> _RateCard | None:
    normalized = normalize_model_name(model)
    tier = str(service_tier or "standard").lower()
    if tier == "priority":
        tier = "fast"
    rates = (
        _FAST_RATES if tier == "fast" else _STANDARD_RATES if tier == "standard" else {}
    )
    if normalized in rates:
        return rates[normalized]
    # Dated snapshots use the base model's published rate unless OpenAI lists a
    # distinct price. Prefer longest names so gpt-5.4-mini does not match gpt-5.4.
    for base in sorted(rates, key=len, reverse=True):
        if normalized and normalized.startswith(base + "-"):
            return rates[base]
    return None


def estimate_cost_usd(
    *,
    model: str | None,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    service_tier: str = "standard",
    context_tier: str = "short",
) -> tuple[float | None, dict[str, Any] | None]:
    """Estimate token cost using the pinned OpenAI rate card.

    Codex reports cached input as a subset of input. Reasoning output is also a
    subset of output, so neither is double-counted here.
    """
    card = _rate_card(model, service_tier)
    if card is None:
        return None, None
    use_long = context_tier == "long"
    input_rate = card.long_input if use_long else card.input
    cached_rate = card.long_cached_input if use_long else card.cached_input
    output_rate = card.long_output if use_long else card.output
    cache_write_rate = card.long_cache_write if use_long else card.cache_write
    if input_rate is None or cached_rate is None or output_rate is None:
        return None, None

    total_input = _to_nonnegative_int(input_tokens)
    cached = min(total_input, _to_nonnegative_int(cached_input_tokens))
    uncached = total_input - cached
    writes = _to_nonnegative_int(cache_write_tokens)
    cost = (
        uncached * input_rate
        + cached * cached_rate
        + _to_nonnegative_int(output_tokens) * output_rate
        + writes * (cache_write_rate or input_rate)
    ) / 1_000_000
    pricing = {
        "version": PRICING_VERSION,
        "source": PRICING_SOURCE,
        "service_tier": "fast" if service_tier == "priority" else service_tier,
        "context_tier": context_tier,
        "currency": "USD",
        "per_million_tokens": {
            "input": input_rate,
            "cached_input": cached_rate,
            "cache_write": cache_write_rate,
            "output": output_rate,
        },
    }
    return cost, pricing


def _jsonl_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            events.append(value)
    return events


def is_codex_jsonl(text: str) -> bool:
    return any(
        event.get("type") in {"thread.started", "turn.completed", "turn.failed"}
        for event in _jsonl_events(text)
    )


def _final_agent_message(events: Sequence[Mapping[str, Any]]) -> str:
    messages: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str):
            messages.append(text)
            continue
        content = item.get("content")
        if isinstance(content, str):
            messages.append(content)
    return messages[-1] if messages else ""


def parse_codex_jsonl(
    text: str,
    *,
    model_hint: str | None = None,
    service_tier: str = "standard",
    context_tier: str = "short",
) -> tuple[str, dict[str, Any] | None]:
    """Return the final agent text and normalized usage from Codex JSONL."""
    events = _jsonl_events(text)
    if not events:
        return text.strip(), None

    thread_id = next(
        (
            str(event["thread_id"])
            for event in events
            if event.get("type") == "thread.started" and event.get("thread_id")
        ),
        None,
    )
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not completed:
        return _final_agent_message(events) or text.strip(), None

    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    event_model: str | None = None
    event_service_tier: str | None = None
    for event in completed:
        raw = event.get("usage")
        if not isinstance(raw, Mapping):
            continue
        for key in totals:
            totals[key] += _to_nonnegative_int(raw.get(key))
        event_model = event_model or (
            str(raw.get("model")) if raw.get("model") else None
        )
        event_service_tier = event_service_tier or (
            str(raw.get("service_tier")) if raw.get("service_tier") else None
        )

    model = normalize_model_name(event_model or model_hint)
    resolved_service_tier = event_service_tier or service_tier or "standard"
    input_tokens = totals["input_tokens"]
    cached = min(input_tokens, totals["cached_input_tokens"])
    output_tokens = totals["output_tokens"]
    estimated_cost, pricing = estimate_cost_usd(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=totals["cache_write_tokens"],
        output_tokens=output_tokens,
        service_tier=resolved_service_tier,
        context_tier=context_tier,
    )
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    call_id = f"{thread_id}:{digest}" if thread_id else f"codex-{digest}"
    usage: dict[str, Any] = {
        "schema_version": CODEX_USAGE_SCHEMA_VERSION,
        "call_id": call_id,
        "thread_id": thread_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "service_tier": resolved_service_tier,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "uncached_input_tokens": max(0, input_tokens - cached),
        "cache_write_tokens": totals["cache_write_tokens"],
        "output_tokens": output_tokens,
        "reasoning_output_tokens": min(
            output_tokens, totals["reasoning_output_tokens"]
        ),
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimated_cost,
        "pricing": pricing,
    }
    return _final_agent_message(events) or text.strip(), usage


def with_usage_context(
    usage: Mapping[str, Any] | None, **fields: Any
) -> dict[str, Any] | None:
    if not usage:
        return None
    enriched = dict(usage)
    enriched.update(current_usage_context())
    enriched.update({key: value for key, value in fields.items() if value is not None})
    return enriched


def response_usage(response: Any) -> dict[str, Any] | None:
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    detailed = metadata.get("codex_usage")
    if isinstance(detailed, Mapping):
        return dict(detailed)
    return None


def token_usage_metadata(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": _to_nonnegative_int(usage.get("input_tokens")),
        "completion_tokens": _to_nonnegative_int(usage.get("output_tokens")),
        "total_tokens": _to_nonnegative_int(usage.get("total_tokens")),
        "prompt_tokens_details": {
            "cached_tokens": _to_nonnegative_int(usage.get("cached_input_tokens"))
        },
        "completion_tokens_details": {
            "reasoning_tokens": _to_nonnegative_int(
                usage.get("reasoning_output_tokens")
            )
        },
    }


def append_usage_record(
    path: str | Path | None, usage: Mapping[str, Any] | None
) -> None:
    if not path or not usage:
        return
    ledger = Path(path).expanduser().resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(usage), separators=(",", ":"), sort_keys=True) + "\n"
    with _LEDGER_LOCK, ledger.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def read_usage_records(path: str | Path) -> list[dict[str, Any]]:
    ledger = Path(path)
    if not ledger.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("call_id"):
            records.append(value)
    return records


def summarize_usage_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        call_id = str(record.get("call_id") or "")
        if call_id:
            unique[call_id] = record

    def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        costs = [
            float(item["estimated_cost_usd"])
            for item in items
            if isinstance(item.get("estimated_cost_usd"), (int, float))
        ]
        models = sorted({str(item["model"]) for item in items if item.get("model")})
        return {
            "calls": len(items),
            "priced_calls": len(costs),
            "unpriced_calls": len(items) - len(costs),
            "estimated_cost_is_partial": bool(costs) and len(costs) < len(items),
            "input_tokens": sum(
                _to_nonnegative_int(item.get("input_tokens")) for item in items
            ),
            "cached_input_tokens": sum(
                _to_nonnegative_int(item.get("cached_input_tokens")) for item in items
            ),
            "uncached_input_tokens": sum(
                _to_nonnegative_int(item.get("uncached_input_tokens")) for item in items
            ),
            "output_tokens": sum(
                _to_nonnegative_int(item.get("output_tokens")) for item in items
            ),
            "reasoning_output_tokens": sum(
                _to_nonnegative_int(item.get("reasoning_output_tokens"))
                for item in items
            ),
            "total_tokens": sum(
                _to_nonnegative_int(item.get("total_tokens")) for item in items
            ),
            "estimated_cost_usd": sum(costs) if costs else None,
            "models": models,
        }

    items = list(unique.values())
    by_generation: dict[int, list[Mapping[str, Any]]] = {}
    unassigned: list[Mapping[str, Any]] = []
    for item in items:
        try:
            generation = int(item.get("generation"))
        except (TypeError, ValueError):
            unassigned.append(item)
            continue
        by_generation.setdefault(generation, []).append(item)
    rows = [
        {"generation": generation, **_summary(by_generation[generation])}
        for generation in sorted(by_generation)
    ]
    return {
        "pricing_version": PRICING_VERSION,
        "pricing_source": PRICING_SOURCE,
        "summary": _summary(items),
        "by_generation": rows,
        "unassigned": _summary(unassigned),
        "records": [dict(item) for item in items],
    }


def _codex_config_path(env: Mapping[str, str] | None = None) -> Path:
    environment = env or os.environ
    configured = environment.get("CODEX_HOME")
    return (
        Path(configured).expanduser() if configured else Path.home() / ".codex"
    ) / "config.toml"


def load_codex_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    path = _codex_config_path(env)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    result: dict[str, str] = {}
    if data.get("model"):
        result["model"] = str(data["model"])
    if data.get("service_tier"):
        result["service_tier"] = str(data["service_tier"])
    return result


def codex_command_defaults(
    command: str | Sequence[str], env: Mapping[str, str] | None = None
) -> dict[str, str]:
    argv = shlex.split(command) if isinstance(command, str) else list(map(str, command))
    result: dict[str, str] = {}
    for idx, arg in enumerate(argv):
        if arg in {"-m", "--model"} and idx + 1 < len(argv):
            result["model"] = argv[idx + 1]
        elif arg.startswith("--model="):
            result["model"] = arg.split("=", 1)[1]
        elif arg in {"-c", "--config"} and idx + 1 < len(argv):
            key, _, raw = argv[idx + 1].partition("=")
            if key in {"model", "service_tier"} and raw:
                result[key] = raw.strip("\"'")
    defaults = load_codex_defaults(env)
    return {**defaults, **result}


def is_codex_exec_command(argv: Sequence[str]) -> bool:
    return bool(argv) and Path(str(argv[0])).name == "codex" and "exec" in argv[1:]


def ensure_codex_json_flag(argv: Sequence[str]) -> list[str]:
    rendered = list(map(str, argv))
    if not is_codex_exec_command(rendered) or "--json" in rendered:
        return rendered
    exec_idx = rendered.index("exec")
    rendered.insert(exec_idx + 1, "--json")
    return rendered


def infer_usage_ledger_from_mutation_log_root(log_root: str | Path) -> Path:
    root = Path(log_root).expanduser().resolve()
    if root.name == "logs" and root.parent.name == "repo_mutation":
        return root.parent.parent / CODEX_USAGE_FILENAME
    return root / CODEX_USAGE_FILENAME
