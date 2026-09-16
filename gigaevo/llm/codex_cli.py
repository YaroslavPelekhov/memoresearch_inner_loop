from __future__ import annotations

from pathlib import Path

from gigaevo.llm.command_cli import CommandCLIChatModel, CommandStructuredOutput

DEFAULT_CODEX_LLM_COMMAND = [
    "codex",
    "exec",
    "--ephemeral",
    "--full-auto",
    "--sandbox",
    "read-only",
    "--skip-git-repo-check",
    "{prompt}",
]


class CodexCLIChatModel(CommandCLIChatModel):
    """Backward-compatible Codex preset for ``CommandCLIChatModel``."""

    def __init__(
        self,
        *,
        command: str | list[str] | None = None,
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        model_name: str = "codex-cli",
        env: dict[str, str] | None = None,
        stdin_prompt: bool = False,
    ) -> None:
        super().__init__(
            command=command or DEFAULT_CODEX_LLM_COMMAND,
            cwd=cwd,
            timeout=timeout,
            model_name=model_name,
            provider_name="codex",
            env=env,
            stdin_prompt=stdin_prompt,
        )


CodexStructuredOutput = CommandStructuredOutput
