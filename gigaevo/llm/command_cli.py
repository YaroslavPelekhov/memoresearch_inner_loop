from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from gigaevo.llm.codex_usage import (
    append_usage_record,
    is_codex_jsonl,
    parse_codex_jsonl,
    token_usage_metadata,
    with_usage_context,
)


def _message_role(message: BaseMessage) -> str:
    role = getattr(message, "type", None) or message.__class__.__name__
    return str(role).upper()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, sort_keys=True, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _input_to_prompt(input: LanguageModelInput) -> str:
    if isinstance(input, str):
        return input
    if isinstance(input, PromptValue):
        messages = input.to_messages()
    elif isinstance(input, list):
        messages = input
    else:
        return str(input)

    chunks: list[str] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            chunks.append(
                f"{_message_role(message)}:\n{_content_to_text(message.content)}"
            )
        else:
            chunks.append(str(message))
    return "\n\n".join(chunks).strip()


def _render_command(
    command: str | list[str], variables: dict[str, str]
) -> str | list[str]:
    def replace_placeholders(text: str) -> str:
        rendered = text
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered

    if isinstance(command, str):
        return replace_placeholders(command)
    return [replace_placeholders(str(part)) for part in command]


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
    raise ValueError("CLI response did not contain a JSON object or array")


class CommandCLIChatModel(Runnable):
    """LangChain-compatible chat model backed by any non-interactive CLI command.

    This wrapper is for GigaEvo stages that expect a LangChain ``Runnable`` while
    the actual model call should go through a local coding-agent or LLM CLI.
    Commands may reference ``{prompt}``, ``{prompt_path}``, and ``{cwd}``.
    """

    def __init__(
        self,
        *,
        command: str | list[str],
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        model_name: str = "command-cli",
        provider_name: str = "command-cli",
        env: dict[str, str] | None = None,
        stdin_prompt: bool = False,
        json_output: bool = False,
        usage_log_path: str | Path | None = None,
        service_tier: str = "standard",
        context_tier: str = "short",
    ) -> None:
        self.command = command
        self.cwd = Path(cwd).expanduser().resolve() if cwd else None
        self.timeout = timeout
        self.model_name = model_name
        self.provider_name = provider_name
        self.env = env or {}
        self.stdin_prompt = stdin_prompt
        self.json_output = json_output
        self.usage_log_path = usage_log_path
        self.service_tier = service_tier
        self.context_tier = context_tier

    @contextmanager
    def _variables(self, prompt: str) -> Iterator[dict[str, str]]:
        with TemporaryDirectory(prefix="gigaevo-cli-prompt-") as tmp:
            prompt_path = Path(tmp) / "prompt.md"
            prompt_path.write_text(prompt)
            yield {
                "prompt": prompt,
                "prompt_path": str(prompt_path),
                "cwd": str(self.cwd or Path.cwd()),
            }

    def _env(self, variables: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {key: str(value).format(**variables) for key, value in self.env.items()}
        )
        return env

    def _response(self, stdout: str, stderr: str, duration: float) -> AIMessage:
        content = stdout.strip()
        usage = None
        if self.json_output or is_codex_jsonl(stdout):
            content, usage = parse_codex_jsonl(
                stdout,
                model_hint=self.model_name,
                service_tier=self.service_tier,
                context_tier=self.context_tier,
            )
            usage = with_usage_context(usage)
            append_usage_record(self.usage_log_path, usage)

        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "duration_seconds": duration,
            "stderr_tail": stderr[-4000:],
        }
        if usage:
            metadata["codex_usage"] = usage
            metadata["token_usage"] = token_usage_metadata(usage)
        return AIMessage(
            content=content,
            response_metadata=metadata,
        )

    def _command_text(self, rendered: str | list[str]) -> str:
        if isinstance(rendered, str):
            return rendered
        return " ".join(map(shlex.quote, rendered))

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        prompt = _input_to_prompt(input)
        with self._variables(prompt) as variables:
            rendered = _render_command(self.command, variables)
            started = time.monotonic()
            if isinstance(rendered, str):
                result = subprocess.run(
                    rendered,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=self._env(variables),
                    shell=True,
                    input=prompt if self.stdin_prompt else None,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            else:
                result = subprocess.run(
                    rendered,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=self._env(variables),
                    input=prompt if self.stdin_prompt else None,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
        duration = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.provider_name} command failed with exit code "
                f"{result.returncode}: {self._command_text(rendered)}\n"
                f"{result.stderr[-4000:]}"
            )
        return self._response(result.stdout, result.stderr, duration)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        prompt = _input_to_prompt(input)
        with self._variables(prompt) as variables:
            rendered = _render_command(self.command, variables)
            env = self._env(variables)
            started = time.monotonic()
            if isinstance(rendered, str):
                proc = await asyncio.create_subprocess_shell(
                    rendered,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=env,
                    stdin=asyncio.subprocess.PIPE if self.stdin_prompt else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *rendered,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=env,
                    stdin=asyncio.subprocess.PIPE if self.stdin_prompt else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(prompt.encode() if self.stdin_prompt else None),
                    timeout=self.timeout,
                )
            except TimeoutError:
                proc.kill()
                stdout_b, stderr_b = await proc.communicate()
                stderr_b += f"\nTimed out after {self.timeout}s".encode()
        duration = time.monotonic() - started
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.provider_name} command failed with exit code "
                f"{proc.returncode}: {self._command_text(rendered)}\n"
                f"{stderr[-4000:]}"
            )
        return self._response(stdout, stderr, duration)

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        yield self.invoke(input, config=config, **kwargs)

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        yield await self.ainvoke(input, config=config, **kwargs)

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> CommandStructuredOutput:
        return CommandStructuredOutput(
            self,
            schema,
            include_raw=bool(kwargs.get("include_raw")),
        )


class CommandStructuredOutput(Runnable):
    def __init__(
        self, model: CommandCLIChatModel, schema: Any, *, include_raw: bool
    ) -> None:
        self.model = model
        self.schema = schema
        self.include_raw = include_raw

    def _schema_text(self) -> str:
        if hasattr(self.schema, "model_json_schema"):
            payload = self.schema.model_json_schema()
        elif isinstance(self.schema, dict):
            payload = self.schema
        else:
            payload = {"schema": str(self.schema)}
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    def _augment(self, input: LanguageModelInput) -> str:
        prompt = _input_to_prompt(input)
        return (
            f"{prompt}\n\n"
            "Return only valid JSON matching this schema. Do not include markdown "
            "fences or explanatory text.\n\n"
            f"JSON schema:\n{self._schema_text()}"
        )

    def _parse(self, raw: AIMessage) -> Any:
        data = _extract_json(_content_to_text(raw.content))
        if hasattr(self.schema, "model_validate"):
            return self.schema.model_validate(data)
        return data

    def _format(self, raw: AIMessage) -> Any:
        parsed = self._parse(raw)
        if self.include_raw:
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return parsed

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._format(
            self.model.invoke(self._augment(input), config=config, **kwargs)
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        raw = await self.model.ainvoke(self._augment(input), config=config, **kwargs)
        return self._format(raw)
