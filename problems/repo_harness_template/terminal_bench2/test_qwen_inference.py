#!/usr/bin/env python3
"""Simple local Qwen inference smoke test.

This does not use Harbor, Docker, or Terminal-Bench. It only checks that the
OpenAI-compatible llama.cpp server answers chat-completion requests for the
model endpoint you plan to use in TB2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gpt-3.5-turbo-16k"


def post_json(url: str, payload: dict, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{error_body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}") from exc


def get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{error_body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test simple inference against a local OpenAI-compatible Qwen server."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TB2_OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TB2_MODEL") or os.environ.get("HARBOR_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TB2_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Return exactly this JSON and nothing else: "
            "{\"status\":\"ok\",\"message\":\"hello\"}"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional stop sequence. May be repeated.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip GET /models before the chat-completion request.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")

    if not args.skip_models:
        models = get_json(f"{base_url}/models")
        print("=== /models ===")
        print(json.dumps(models, indent=2, sort_keys=True))

    payload = {
        "model": args.model.removeprefix("openai/"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict test endpoint. Output only the requested "
                    "final answer. Do not include analysis or thinking."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.stop:
        payload["stop"] = args.stop

    print("\n=== Request ===")
    print(json.dumps(payload, indent=2, sort_keys=True))

    result = post_json(f"{base_url}/chat/completions", payload, args.api_key)
    print("\n=== Raw Response ===")
    print(json.dumps(result, indent=2, sort_keys=True))

    content = (
        ((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    )
    print("\n=== Assistant Text ===")
    print(content)

    if "status" not in content.lower() and "hello" not in content.lower():
        print(
            "\nWARNING: response did not contain the expected JSON-ish content. "
            "The server works, but the model may need prompt/template tuning.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
