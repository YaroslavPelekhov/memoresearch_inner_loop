from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "problems"
    / "cybergym_harness"
    / "cybergym"
    / "benchmark.py"
)
SEED_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "problems"
    / "cybergym_harness"
    / "repo_harness_seed.yaml"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    return _load_module("cybergym_benchmark", BENCHMARK_PATH)


def _make_pocdb(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table poc_records (
                id integer primary key,
                agent_id text,
                task_id text,
                poc_id text,
                poc_hash text,
                poc_length integer,
                vul_exit_code integer,
                fix_exit_code integer
            )
            """
        )
        conn.executemany(
            """
            insert into poc_records (
                agent_id, task_id, poc_id, poc_hash, poc_length,
                vul_exit_code, fix_exit_code
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("agent-ok", "arvo:10400", "poc-ok", "hash-ok", 17, 1, 0),
                ("agent-vul-only", "arvo:10400", "poc-vul", "hash-vul", 9, 1, None),
                ("agent-no-crash", "arvo:10400", "poc-no", "hash-no", 4, 0, 0),
            ],
        )


def test_selected_tasks_defaults_to_smoke():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        tasks=None,
        task=None,
        task_set="smoke",
        smoke_task="arvo:10400",
        max_tasks=None,
    )

    assert benchmark.selected_tasks(args) == ["arvo:10400"]


def test_selected_tasks_supports_subset_and_explicit_overrides():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        tasks=["arvo:1,arvo:2"],
        task=["oss-fuzz:3"],
        task_set="subset",
        smoke_task="arvo:10400",
        max_tasks=2,
    )

    assert benchmark.selected_tasks(args) == ["arvo:1", "arvo:2"]


def test_poc_scoring_distinguishes_verified_and_vul_crash(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "poc.db"
    _make_pocdb(db_path)

    ok = benchmark.load_poc_records(db_path, agent_id="agent-ok", task_id="arvo:10400")
    vul_only = benchmark.load_poc_records(
        db_path, agent_id="agent-vul-only", task_id="arvo:10400"
    )
    no_crash = benchmark.load_poc_records(
        db_path, agent_id="agent-no-crash", task_id="arvo:10400"
    )

    assert benchmark.trial_success(ok, "verified") is True
    assert benchmark.trial_success(vul_only, "verified") is False
    assert benchmark.trial_success(vul_only, "vul-crash") is True
    assert benchmark.trial_success(no_crash, "vul-crash") is False


def test_openhands_command_is_default_and_does_not_pass_cybench_image(tmp_path):
    benchmark = _load_benchmark_module()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    runner = candidate / "run.py"
    runner.write_text("# runner\n")
    openhands_repo = tmp_path / "openhands-repo"
    openhands_repo.mkdir()
    args = argparse.Namespace(
        runner="run.py",
        data_dir=str(tmp_path),
        server="http://127.0.0.1:8666",
        difficulty="level1",
        model="gpt-4.1-2025-04-14",
        max_iter=100,
        inner_timeout=1200,
        mask_map="",
        framework="openhands",
        openhands_repo=str(openhands_repo),
        cybench_repo="",
        cybergym_root="",
        silent="false",
        image="cybergym/cybench:latest",
        container_name=None,
        runner_arg=None,
    )

    cmd = benchmark.build_candidate_command(
        args,
        candidate_repo=candidate,
        task="arvo:10400",
        attempt=1,
        job_dir=tmp_path / "job",
    )

    assert "--repo" in cmd
    assert str(openhands_repo) in cmd
    assert "--silent" in cmd
    assert "--image" not in cmd


def test_compute_metrics_counts_successes_and_submissions(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "poc.db"
    _make_pocdb(db_path)
    ok_records = benchmark.load_poc_records(
        db_path, agent_id="agent-ok", task_id="arvo:10400"
    )
    no_crash_records = benchmark.load_poc_records(
        db_path, agent_id="agent-no-crash", task_id="arvo:10400"
    )

    records = [
        benchmark.TrialRecord(
            task="arvo:10400",
            attempt=1,
            agent_id="agent-ok",
            run_dir="/tmp/ok",
            result=benchmark.CommandResult(0, "", ""),
            poc_records=ok_records,
            success=True,
        ),
        benchmark.TrialRecord(
            task="arvo:10400",
            attempt=2,
            agent_id="agent-no-crash",
            run_dir="/tmp/no",
            result=benchmark.CommandResult(0, "", ""),
            poc_records=no_crash_records,
            success=False,
        ),
    ]

    metrics = benchmark.compute_metrics(records)

    assert metrics["fitness"] == 0.5
    assert metrics["is_valid"] == 1.0
    assert metrics["n_successes"] == 1.0
    assert metrics["n_trials"] == 2.0
    assert metrics["n_submissions"] == 2.0
    assert metrics["n_vul_crashes"] == 1.0
    assert metrics["n_verified_successes"] == 1.0


def test_parse_existing_trials_reads_agent_ids_and_pocdb(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "poc.db"
    _make_pocdb(db_path)
    run_dir = tmp_path / "job" / "logs" / "arvo_10400-agent-ok"
    run_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text(
        json.dumps({"task": {"task_id": "arvo:10400", "agent_id": "agent-ok"}})
    )
    args = argparse.Namespace(pocdb_path=str(db_path), success_mode="verified")

    records = benchmark.parse_existing_trials(args, tmp_path / "job")

    assert len(records) == 1
    assert records[0].agent_id == "agent-ok"
    assert records[0].success is True


def test_structured_feedback_includes_missing_submission_hint():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        framework="openhands",
        task_set="smoke",
        success_mode="verified",
        server="http://127.0.0.1:8666",
        pocdb_path="/tmp/poc.db",
        verify=False,
    )
    records = [
        benchmark.TrialRecord(
            task="arvo:10400",
            attempt=1,
            agent_id="agent-no-submit",
            run_dir="/tmp/run",
            result=benchmark.CommandResult(0, "", ""),
            poc_records=[],
            success=False,
        )
    ]

    feedback = benchmark.build_structured_feedback(
        records=records,
        metrics=benchmark.compute_metrics(records),
        args=args,
        tasks=["arvo:10400"],
    )

    assert feedback["benchmark"] == "cybergym_openhands"
    assert feedback["failure_clusters"] == [{"name": "arvo:10400", "count": 1}]
    assert any("PoC database records" in hint for hint in feedback["hints"])


def test_auto_seed_contains_default_openhands_runner():
    import yaml

    spec = yaml.safe_load(SEED_SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["default_variant"] == "openhands-default"
    files = spec["variants"]["openhands-default"]["files"]
    run_py = files["run.py"]["content"]

    assert "def run_openhands(" in run_py
    assert "openhands.core.main" in run_py
    assert "docker.all-hands.dev/all-hands-ai/runtime:0.33-nikolaik" in files["template/config.toml"]["content"]
    assert "bootstrap_openhands_repo.py" in files
