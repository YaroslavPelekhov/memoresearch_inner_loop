from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[2] / "tools/run-autoresearch-smoke.py"
    spec = importlib.util.spec_from_file_location("autoresearch_smoke_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_is_fail_closed_and_reuses_only_matching_content(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    model = tmp_path / "autoresearch/model/gdn.py"
    model.parent.mkdir(parents=True)
    model.write_text("first")
    run_dir = tmp_path / "smoke"
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout='{"fitness": 0.0, "is_valid": 1.0}\n',
            stderr="",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke",
            "--batches",
            "8",
            "--run-dir",
            str(run_dir),
            "--check-file",
            "autoresearch/model/gdn.py",
            "--gpus",
            "1",
        ],
    )

    assert module.main() == 0
    assert module.main() == 0
    assert len(calls) == 1
    assert calls[0][-2:] == ["--gpus", "1"]

    model.write_text("changed")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"fitness": 0.0, "is_valid": 0.0}\n',
            stderr="broken",
        ),
    )
    assert module.main() == 1
