from __future__ import annotations

import json
from pathlib import Path
import subprocess

from gigaevo.repo_harness import viz


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(
        repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test User",
        "commit",
        "-am",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _manifest(repo: Path, commit: str, parent: str | None = None) -> dict:
    return {
        "kind": "repo_snapshot",
        "schema_version": 1,
        "repo_path": str(repo),
        "commit": commit,
        "parent_commit": parent,
        "branch": None,
        "entrypoint": None,
        "mutation_agent": "test-agent" if parent else None,
        "mutation_session_log": None,
        "changed_files": ["main.go"] if parent else [],
        "created_at": "2026-05-10T00:00:00+00:00",
        "extra": {},
    }


def test_build_visualization_enriches_repo_harness_programs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "main.go").write_text("package main\n\nfunc main() {}\n")
    _git(repo, "add", "main.go")
    parent_commit = _commit(repo, "seed")
    (repo / "main.go").write_text(
        "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hi\") }\n"
    )
    child_commit = _commit(repo, "child")

    run_dir = tmp_path / "run"
    log_dir = run_dir / "repo_mutation" / "logs" / "abc123"
    agent_dir = log_dir / "agent"
    agent_dir.mkdir(parents=True)
    (log_dir / "mutation_brief.md").write_text("mutation brief")
    (agent_dir / "prompt.md").write_text("prompt")
    (agent_dir / "stdout.txt").write_text("stdout")
    (agent_dir / "stderr.txt").write_text("stderr")
    (agent_dir / "command.txt").write_text("agent command")

    parent_id = "00000000-0000-4000-8000-000000000001"
    child_id = "00000000-0000-4000-8000-000000000002"
    parent_manifest = _manifest(repo, parent_commit)
    child_manifest = _manifest(repo, child_commit, parent_commit)
    child_manifest["mutation_session_log"] = str(agent_dir)
    mutation_usage = {
        "call_id": "mutation-call",
        "model": "gpt-5.6-sol",
        "generation": 2,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 60,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "total_tokens": 120,
        "estimated_cost_usd": 0.001,
    }
    curator_usage = {
        "call_id": "curator-call",
        "model": "gpt-5.6-sol",
        "generation": 2,
        "source": "archive_curator",
        "input_tokens": 200,
        "cached_input_tokens": 100,
        "uncached_input_tokens": 100,
        "output_tokens": 10,
        "reasoning_output_tokens": 2,
        "total_tokens": 210,
        "estimated_cost_usd": 0.002,
    }
    (run_dir / "codex_usage.jsonl").write_text(
        "\n".join(json.dumps(record) for record in [mutation_usage, curator_usage])
        + "\n"
    )

    programs = [
        {
            "id": parent_id,
            "code": json.dumps(parent_manifest),
            "metrics": {"fitness": 0.1, "n_resolved": 1, "n_tests": 10},
            "metadata": {"repo_candidate": parent_manifest},
            "lineage": {"parents": [], "children": [child_id], "generation": 1},
            "state": "done",
        },
        {
            "id": child_id,
            "code": json.dumps(child_manifest),
            "metrics": {"fitness": 0.5, "n_resolved": 5, "n_tests": 10},
            "metadata": {
                "repo_candidate": child_manifest,
                "codex_usage": mutation_usage,
                "repo_reflection": {"reflection": "Implemented output."},
                "repo_benchmark_feedback": {
                    "stderr_tail": (
                        "[programbench] structured failure feedback:\n"
                        '{"failure_count": 1, "failure_names": ["tests.x"], '
                        '"status_counts": {"failure": 1}}'
                    )
                },
            },
            "lineage": {"parents": [parent_id], "children": [], "generation": 2},
            "state": "done",
        },
    ]
    monkeypatch.setattr(viz, "_fetch_programs", lambda config: programs)

    config = viz.RepoVizConfig(
        run_dir=run_dir,
        out_dir=tmp_path / "viz",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_prefix="test",
        source_repo=repo,
    )
    result = viz.build_visualization(config)

    assert result.node_count == 2
    assert result.edge_count == 1
    assert result.best_value == 0.5
    data = json.loads(result.data_path.read_text())
    assert data["edges"][0]["delta"] == 0.4
    child = next(node for node in data["nodes"] if node["id"] == child_id)
    assert child["parents"] == [parent_id]
    assert "fmt.Println" in child["git"]["diff"]
    assert child["mutation_brief"] == "mutation brief"
    assert child["agent_logs"]["command"] == "agent command"
    assert child["structured_feedback"]["failure_names"] == ["tests.x"]
    assert child["codex_usage"]["total_tokens"] == 120
    assert data["usage"]["summary"]["calls"] == 2
    assert data["usage"]["summary"]["total_tokens"] == 330
    assert data["usage"]["by_generation"][0]["estimated_cost_usd"] == 0.003
    assert result.html_path.exists()
    html = result.html_path.read_text()
    assert "function renderParents(node)" in html
    assert "changed into child" in html
    assert "function renderUsageTable()" in html
    assert "estimated API cost" in html
