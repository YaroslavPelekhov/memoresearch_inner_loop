from __future__ import annotations

from pydantic import BaseModel

from gigaevo.llm.command_cli import CommandCLIChatModel


class _Answer(BaseModel):
    answer: int


def test_command_cli_chat_model_invokes_command_with_prompt():
    model = CommandCLIChatModel(
        command=[
            "python",
            "-c",
            "import sys; print('reply:' + sys.argv[1])",
            "{prompt}",
        ],
        model_name="test-cli",
        provider_name="test-provider",
    )

    response = model.invoke("hello")

    assert response.content == "reply:hello"
    assert response.response_metadata["model_name"] == "test-cli"
    assert response.response_metadata["provider_name"] == "test-provider"


def test_command_cli_chat_model_exposes_prompt_path():
    model = CommandCLIChatModel(
        command=[
            "python",
            "-c",
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())",
            "{prompt_path}",
        ],
        model_name="test-cli",
    )

    response = model.invoke("from file")

    assert response.content == "from file"


def test_command_cli_chat_model_supports_stdin_prompt():
    model = CommandCLIChatModel(
        command=["python", "-c", "import sys; print('stdin:' + sys.stdin.read())"],
        stdin_prompt=True,
    )

    response = model.invoke("hello")

    assert response.content == "stdin:hello"


async def test_command_cli_chat_model_ainvoke():
    model = CommandCLIChatModel(
        command=[
            "python",
            "-c",
            "import sys; print('async:' + sys.argv[1])",
            "{prompt}",
        ]
    )

    response = await model.ainvoke("hello")

    assert response.content == "async:hello"


def test_command_cli_structured_output_parses_pydantic_schema():
    model = CommandCLIChatModel(
        command=["python", "-c", "print('{\"answer\": 7}')"],
        model_name="test-cli",
    )

    structured = model.with_structured_output(_Answer, include_raw=True)
    response = structured.invoke("ignored")

    assert response["parsed"] == _Answer(answer=7)
    assert response["raw"].content == '{"answer": 7}'
