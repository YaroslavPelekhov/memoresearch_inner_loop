from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any, Protocol

from gigaevo.llm.codex_usage import (
    codex_command_defaults,
    ensure_codex_json_flag,
    is_codex_exec_command,
    is_codex_jsonl,
    parse_codex_jsonl,
)


@dataclass(frozen=True)
class CodingAgentRun:
    backend_name: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: str | None = None
    usage: dict[str, Any] | None = None


class CodingAgentBackend(Protocol):
    name: str

    async def run(
        self,
        *,
        prompt: str,
        cwd: Path,
        log_dir: Path,
        timeout: float,
        variables: dict[str, str] | None = None,
    ) -> CodingAgentRun: ...


def _render(value: str, variables: dict[str, str]) -> str:
    return value.format(**variables)


def _argv(command: str | list[str], variables: dict[str, str]) -> list[str]:
    if isinstance(command, str):
        return [_render(part, variables) for part in shlex.split(command)]
    return [_render(str(part), variables) for part in command]


class CommandCodingAgentBackend:
    """Run a coding agent through a configurable subprocess command.

    The command can reference ``{prompt}``, ``{prompt_path}``, ``{cwd}``,
    ``{brief_path}``, and any variables supplied by the mutation operator.
    """

    def __init__(
        self,
        *,
        command: str | list[str],
        name: str = "command",
        env: dict[str, str] | None = None,
        stdin_prompt: bool = False,
        codex_json: bool | None = None,
        service_tier: str = "standard",
        context_tier: str = "short",
    ):
        self.command = command
        self.name = name
        self.env = env or {}
        self.stdin_prompt = stdin_prompt
        self.codex_json = codex_json
        self.service_tier = service_tier
        self.context_tier = context_tier

    async def run(
        self,
        *,
        prompt: str,
        cwd: Path,
        log_dir: Path,
        timeout: float,
        variables: dict[str, str] | None = None,
    ) -> CodingAgentRun:
        log_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = log_dir / "prompt.md"
        prompt_path.write_text(prompt)

        vars_ = {
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "cwd": str(cwd),
            **(variables or {}),
        }
        argv = _argv(self.command, vars_)
        env = os.environ.copy()
        env.update({k: _render(str(v), vars_) for k, v in self.env.items()})
        is_codex = is_codex_exec_command(argv)
        if self.codex_json is True or (self.codex_json is None and is_codex):
            argv = ensure_codex_json_flag(argv)
        codex_defaults = codex_command_defaults(argv, env) if is_codex else {}
        command_text = " ".join(shlex.quote(a) for a in argv)
        (log_dir / "command.txt").write_text(command_text)

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if self.stdin_prompt else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _capture_stream(
            stream: asyncio.StreamReader | None,
            path: Path,
        ) -> bytes:
            chunks: list[bytes] = []
            if stream is None:
                path.write_bytes(b"")
                return b""
            with path.open("wb") as handle:
                while True:
                    chunk = await stream.read(8192)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    handle.write(chunk)
                    handle.flush()
            return b"".join(chunks)

        stdout_task = asyncio.create_task(
            _capture_stream(proc.stdout, log_dir / "stdout.txt")
        )
        stderr_task = asyncio.create_task(
            _capture_stream(proc.stderr, log_dir / "stderr.txt")
        )
        timed_out = False
        try:
            if self.stdin_prompt and proc.stdin is not None:
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
                proc.stdin.close()
                await proc.stdin.wait_closed()
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            proc.kill()
            await proc.wait()

        stdout_b, stderr_b = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
            timeout_message = f"\nTimed out after {timeout}s".encode()
            with (log_dir / "stderr.txt").open("ab") as handle:
                handle.write(timeout_message)
            stderr_b += timeout_message
        elif proc.returncode not in (0, None) and (log_dir / "stderr.txt").exists():
            stderr_b = (log_dir / "stderr.txt").read_bytes()

        duration = time.monotonic() - started
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        content = stdout
        usage = None
        if is_codex_jsonl(stdout):
            content, usage = parse_codex_jsonl(
                stdout,
                model_hint=codex_defaults.get("model"),
                service_tier=codex_defaults.get("service_tier", self.service_tier),
                context_tier=self.context_tier,
            )
            if usage:
                (log_dir / "codex_usage.json").write_text(
                    json.dumps(usage, indent=2, sort_keys=True) + "\n"
                )
        return CodingAgentRun(
            backend_name=self.name,
            exit_code=proc.returncode if proc.returncode is not None else 124,
            stdout=content,
            stderr=stderr,
            duration_seconds=duration,
            log_path=str(log_dir),
            usage=usage,
        )


class ClaudeCodeBackend(CommandCodingAgentBackend):
    """Claude Code CLI backend compatible with the Meta-Harness proposer style."""

    def __init__(
        self,
        *,
        model: str = "opus",
        allowed_tools: list[str] | None = None,
        extra_args: list[str] | None = None,
        name: str = "claude-code",
    ):
        tools = allowed_tools or ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
        command: list[str] = [
            "claude",
            "--dangerously-skip-permissions",
            "-p",
            "{prompt}",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--allowedTools",
            *tools,
        ]
        if extra_args:
            command.extend(extra_args)
        super().__init__(command=command, name=name)


class DryRunCodingAgentBackend:
    """Deterministic no-op backend for config smoke tests and dry runs."""

    name = "dry-run"

    async def run(
        self,
        *,
        prompt: str,
        cwd: Path,
        log_dir: Path,
        timeout: float,
        variables: dict[str, str] | None = None,
    ) -> CodingAgentRun:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "prompt.md").write_text(prompt)
        (log_dir / "stdout.txt").write_text("dry-run backend did not edit files\n")
        return CodingAgentRun(
            backend_name=self.name,
            exit_code=0,
            stdout="dry-run backend did not edit files\n",
            stderr="",
            duration_seconds=0.0,
            log_path=str(log_dir),
        )
