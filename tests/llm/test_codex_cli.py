from __future__ import annotations

from pydantic import BaseModel

from gigaevo.llm.codex_cli import CodexCLIChatModel


class _Answer(BaseModel):
    answer: int


def test_codex_cli_chat_model_invokes_command_with_prompt():
    model = CodexCLIChatModel(
        command=[
            "python",
            "-c",
            "import sys; print('reply:' + sys.argv[1])",
            "{prompt}",
        ],
        model_name="test-codex",
    )

    response = model.invoke("hello")

    assert response.content == "reply:hello"
    assert response.response_metadata["model_name"] == "test-codex"
    assert response.response_metadata["provider_name"] == "codex"


async def test_codex_cli_chat_model_ainvoke():
    model = CodexCLIChatModel(
        command=[
            "python",
            "-c",
            "import sys; print('async:' + sys.argv[1])",
            "{prompt}",
        ]
    )

    response = await model.ainvoke("hello")

    assert response.content == "async:hello"


def test_codex_cli_structured_output_parses_pydantic_schema():
    model = CodexCLIChatModel(
        command=["python", "-c", "print('{\"answer\": 7}')"],
        model_name="test-codex",
    )

    structured = model.with_structured_output(_Answer, include_raw=True)
    response = structured.invoke("ignored")

    assert response["parsed"] == _Answer(answer=7)
    assert response["raw"].content == '{"answer": 7}'
