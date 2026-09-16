from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import html
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

from loguru import logger

from gigaevo.llm.codex_usage import (
    CODEX_USAGE_FILENAME,
    read_usage_records,
    summarize_usage_records,
    with_usage_context,
)
from gigaevo.repo_harness.feedback import extract_structured_feedback

try:
    import redis as redis_lib
except ModuleNotFoundError:  # pragma: no cover - import error is surfaced at runtime
    redis_lib = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - config inference becomes best effort
    yaml = None


DEFAULT_MAX_DIFF_CHARS = 24000
DEFAULT_MAX_TEXT_CHARS = 60000
HTML_FILENAME = "index.html"
DATA_FILENAME = "data.json"


@dataclass(frozen=True)
class RepoVizConfig:
    run_dir: Path
    out_dir: Path
    redis_host: str
    redis_port: int
    redis_db: int
    redis_prefix: str
    source_repo: Path | None = None
    metric: str = "fitness"
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS


@dataclass(frozen=True)
class RepoVizResult:
    out_dir: Path
    data_path: Path
    html_path: Path
    node_count: int
    edge_count: int
    metric: str
    best_value: float | None


class RepoVizPostStepHook:
    """Regenerate the repo_harness dashboard after each evolution generation."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        out_dir: str | Path | None = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_prefix: str,
        source_repo: str | Path | None = None,
        metric: str = "fitness",
        max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        fail_open: bool = True,
    ):
        self.config = infer_config(
            run_dir=run_dir,
            out_dir=out_dir,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_prefix=redis_prefix,
            source_repo=source_repo,
            metric=metric,
            max_diff_chars=max_diff_chars,
            max_text_chars=max_text_chars,
        )
        self.fail_open = fail_open
        self.count = 0

    async def __call__(self) -> None:
        try:
            result = build_visualization(self.config)
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.warning("[RepoVizPostStepHook] dashboard update failed: {}", exc)
            return
        self.count += 1
        logger.info(
            "[RepoVizPostStepHook] update #{} wrote {} (programs={}, edges={}, best_{}={})",
            self.count,
            result.html_path,
            result.node_count,
            result.edge_count,
            result.metric,
            result.best_value,
        )


def infer_config(
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    redis_db: int | None = None,
    redis_prefix: str | None = None,
    source_repo: str | Path | None = None,
    metric: str = "fitness",
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> RepoVizConfig:
    """Build a visualization config, filling omitted fields from Hydra output."""
    run_path = Path(run_dir).expanduser().resolve()
    hydra_cfg = _load_hydra_config(run_path)

    inferred_host = _dig(hydra_cfg, "redis", "host") or "localhost"
    inferred_port = _to_int(_dig(hydra_cfg, "redis", "port"), 6379)
    inferred_db = _to_int(_dig(hydra_cfg, "redis", "db"), 0)
    inferred_prefix = _resolve_interpolation(
        _dig(hydra_cfg, "redis_storage", "config", "key_prefix"),
        hydra_cfg,
    )
    if not inferred_prefix:
        inferred_prefix = _resolve_interpolation(
            _dig(hydra_cfg, "redis", "prefix"),
            hydra_cfg,
        )
    if not inferred_prefix:
        inferred_prefix = _dig(hydra_cfg, "problem", "name")
    if not inferred_prefix:
        inferred_prefix = run_path.name

    inferred_source = source_repo
    if inferred_source is None:
        inferred_source = _dig(hydra_cfg, "repo_harness", "source_repo")

    return RepoVizConfig(
        run_dir=run_path,
        out_dir=Path(out_dir).expanduser().resolve() if out_dir else run_path / "viz",
        redis_host=str(redis_host or inferred_host),
        redis_port=int(redis_port or inferred_port),
        redis_db=int(redis_db if redis_db is not None else inferred_db),
        redis_prefix=str(redis_prefix or inferred_prefix),
        source_repo=Path(inferred_source).expanduser().resolve()
        if inferred_source
        else None,
        metric=metric,
        max_diff_chars=max_diff_chars,
        max_text_chars=max_text_chars,
    )


def build_visualization(config: RepoVizConfig) -> RepoVizResult:
    programs = _fetch_programs(config)
    nodes = [_node_from_program(program, config) for program in programs]
    nodes.sort(
        key=lambda n: (
            _safe_int(n.get("generation"), 0),
            _metric_value(n, config.metric) is None,
            -float(_metric_value(n, config.metric) or 0.0),
            str(n.get("short_id") or n.get("id") or ""),
        )
    )

    node_by_id = {str(node["id"]): node for node in nodes if node.get("id")}
    short_to_id = {
        str(node.get("short_id")): str(node["id"])
        for node in nodes
        if node.get("id") and node.get("short_id")
    }
    for node in nodes:
        parents = []
        for parent_id in node.get("parents", []) or []:
            if parent_id in node_by_id:
                parents.append(parent_id)
            elif parent_id[:8] in short_to_id:
                parents.append(short_to_id[parent_id[:8]])
            else:
                parents.append(parent_id)
        node["parents"] = parents

    edges = _build_edges(nodes, node_by_id, config.metric)
    available_metrics = _available_metrics(nodes)
    trajectories = {
        name: _trajectory(nodes, name)
        for name in available_metrics
        if any(_metric_value(node, name) is not None for node in nodes)
    }
    best_node = _best_node(nodes, config.metric)
    ledger_records = read_usage_records(config.run_dir / CODEX_USAGE_FILENAME)
    embedded_records = [
        record
        for node in nodes
        for record in node.get("codex_usage_records", [])
        if isinstance(record, dict)
    ]
    usage = summarize_usage_records([*ledger_records, *embedded_records])

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(config.run_dir),
        "source_repo": str(config.source_repo) if config.source_repo else None,
        "redis": {
            "host": config.redis_host,
            "port": config.redis_port,
            "db": config.redis_db,
            "prefix": config.redis_prefix,
        },
        "metric": config.metric,
        "available_metrics": available_metrics,
        "summary": {
            "programs": len(nodes),
            "edges": len(edges),
            "generations": sorted(
                {
                    _safe_int(node.get("generation"), 0)
                    for node in nodes
                    if node.get("generation") is not None
                }
            ),
            "best": {
                "program_id": best_node.get("id") if best_node else None,
                "short_id": best_node.get("short_id") if best_node else None,
                "value": _metric_value(best_node, config.metric) if best_node else None,
                "generation": best_node.get("generation") if best_node else None,
            },
            "token_usage": usage["summary"],
        },
        "usage": usage,
        "nodes": nodes,
        "edges": edges,
        "trajectories": trajectories,
    }

    config.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = config.out_dir / DATA_FILENAME
    html_path = config.out_dir / HTML_FILENAME
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")

    return RepoVizResult(
        out_dir=config.out_dir,
        data_path=data_path,
        html_path=html_path,
        node_count=len(nodes),
        edge_count=len(edges),
        metric=config.metric,
        best_value=payload["summary"]["best"]["value"],
    )


def _load_hydra_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / ".hydra" / "config.yaml"
    if not path.exists() or yaml is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _dig(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_interpolation(value: Any, cfg: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    if value == "${problem.name}":
        return _dig(cfg, "problem", "name")
    if value == "${redis.prefix}":
        return _resolve_interpolation(_dig(cfg, "redis", "prefix"), cfg)
    if value == "${redis.db}":
        return _dig(cfg, "redis", "db")
    return value


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fetch_programs(config: RepoVizConfig) -> list[dict[str, Any]]:
    if redis_lib is None:
        raise RuntimeError("redis package is required to build repo visualization")
    r = redis_lib.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        decode_responses=True,
    )
    try:
        keys = sorted(r.scan_iter(f"{config.redis_prefix}:program:*"))
        programs = []
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            try:
                program = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(program, dict):
                programs.append(program)
        return programs
    finally:
        r.close()


def _node_from_program(program: dict[str, Any], config: RepoVizConfig) -> dict[str, Any]:
    metadata = program.get("metadata") if isinstance(program.get("metadata"), dict) else {}
    lineage = program.get("lineage") if isinstance(program.get("lineage"), dict) else {}
    metrics = program.get("metrics") if isinstance(program.get("metrics"), dict) else {}
    manifest = _manifest_from_program(program, metadata)
    repo_path = (
        Path(str(manifest.get("repo_path"))).expanduser().resolve()
        if manifest.get("repo_path")
        else config.source_repo
    )
    commit = _str_or_none(manifest.get("commit") or metadata.get("git_commit"))
    parent_commit = _str_or_none(
        manifest.get("parent_commit") or metadata.get("parent_commit")
    )
    changed_files = _list_of_str(
        manifest.get("changed_files") or metadata.get("changed_files") or []
    )

    git_info = _git_transition_info(
        repo_path=repo_path,
        parent_commit=parent_commit,
        commit=commit,
        max_diff_chars=config.max_diff_chars,
    )
    if not changed_files:
        changed_files = list(git_info.get("changed_files") or [])

    benchmark_feedback = _clip_obj(
        metadata.get("repo_benchmark_feedback"),
        max_chars=config.max_text_chars,
    )
    structured_feedback = _extract_structured_feedback(benchmark_feedback)
    repo_reflection = _clip_obj(
        metadata.get("repo_reflection"),
        max_chars=config.max_text_chars,
    )
    mutation_context = _clip_text(
        str(metadata.get("mutation_context") or ""),
        config.max_text_chars,
    )
    mutation_log_dir = _mutation_log_dir(manifest, metadata)
    mutation_brief = _read_text_if_exists(
        mutation_log_dir / "mutation_brief.md" if mutation_log_dir else None,
        config.max_text_chars,
    )
    agent_logs = _agent_logs(mutation_log_dir, config.max_text_chars)

    program_id = str(program.get("id") or "")
    short_id = program_id[:8] if program_id else "unknown"
    generation = lineage.get("generation") or program.get("generation")
    usage_records = _program_usage_records(
        metadata,
        generation=generation,
        program_id=program_id,
        commit=commit,
    )
    usage_summary = summarize_usage_records(usage_records)["summary"]
    return {
        "id": program_id,
        "short_id": short_id,
        "name": program.get("name"),
        "generation": generation,
        "iteration": program.get("iteration"),
        "state": _state_value(program.get("state")),
        "created_at": program.get("created_at"),
        "parents": _list_of_str(lineage.get("parents") or []),
        "children": _list_of_str(lineage.get("children") or []),
        "mutation": lineage.get("mutation"),
        "metrics": _float_metrics(metrics),
        "commit": commit,
        "parent_commit": parent_commit,
        "branch": manifest.get("branch") or metadata.get("git_branch"),
        "repo_path": str(repo_path) if repo_path else None,
        "worktree_path": metadata.get("worktree_path")
        or _dig(manifest, "extra", "worktree"),
        "changed_files": changed_files,
        "mutation_agent": manifest.get("mutation_agent")
        or metadata.get("mutation_agent"),
        "mutation_duration_seconds": metadata.get("mutation_duration_seconds"),
        "mutation_session_log": manifest.get("mutation_session_log")
        or metadata.get("mutation_session_log"),
        "mutation_log_dir": str(mutation_log_dir) if mutation_log_dir else None,
        "mutation_brief": mutation_brief,
        "agent_logs": agent_logs,
        "repo_reflection": repo_reflection,
        "mutation_context": mutation_context,
        "benchmark_feedback": benchmark_feedback,
        "structured_feedback": structured_feedback,
        "codex_usage": usage_summary,
        "codex_usage_records": usage_records,
        "git": git_info,
    }


def _program_usage_records(
    metadata: dict[str, Any],
    *,
    generation: Any,
    program_id: str,
    commit: str | None,
) -> list[dict[str, Any]]:
    candidates = [
        metadata.get("codex_usage"),
        _dig(metadata, "repo_candidate", "extra", "codex_usage"),
        _dig(metadata, "repo_reflection", "codex_usage"),
        _dig(metadata, "parent_selection", "codex_usage"),
    ]
    records = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("call_id"):
            continue
        record = with_usage_context(
            candidate,
            generation=generation,
            program_id=program_id,
            commit=commit,
        )
        if record:
            records.append(record)
    return summarize_usage_records(records)["records"]


def _manifest_from_program(
    program: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    candidate = metadata.get("repo_candidate")
    if isinstance(candidate, dict):
        return candidate
    code = program.get("code")
    if isinstance(code, str):
        try:
            payload = json.loads(code)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("kind") == "repo_snapshot":
            return payload
    return {}


def _git_transition_info(
    *,
    repo_path: Path | None,
    parent_commit: str | None,
    commit: str | None,
    max_diff_chars: int,
) -> dict[str, Any]:
    if not repo_path or not commit or not parent_commit or not repo_path.exists():
        return {
            "diff_stat": "",
            "name_status": "",
            "diff": "",
            "diff_truncated": False,
            "changed_files": [],
            "error": None,
        }

    rev_range = f"{parent_commit}..{commit}"
    stat = _git(repo_path, ["diff", "--stat", rev_range])
    name_status = _git(repo_path, ["diff", "--name-status", rev_range])
    diff = _git(repo_path, ["diff", "--find-renames", "--find-copies", rev_range])
    error = stat.get("error") or name_status.get("error") or diff.get("error")
    diff_text = diff.get("stdout", "")
    clipped = _clip_text(diff_text, max_diff_chars)
    return {
        "diff_stat": stat.get("stdout", ""),
        "name_status": name_status.get("stdout", ""),
        "diff": clipped,
        "diff_truncated": len(diff_text) > len(clipped),
        "changed_files": _name_status_files(name_status.get("stdout", "")),
        "error": error,
    }


def _git(repo_path: Path, args: list[str]) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except Exception as exc:
        return {"stdout": "", "stderr": "", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.stderr.strip() if result.returncode else None,
    }


def _name_status_files(name_status: str) -> list[str]:
    files = []
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append(parts[-1])
    return files


def _mutation_log_dir(
    manifest: dict[str, Any], metadata: dict[str, Any]
) -> Path | None:
    raw = manifest.get("mutation_session_log") or metadata.get("mutation_session_log")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.name == "agent":
        return path.parent
    if path.name in {"stdout.txt", "stderr.txt", "prompt.md", "command.txt"}:
        return path.parent.parent
    return path


def _agent_logs(log_dir: Path | None, max_chars: int) -> dict[str, str]:
    if log_dir is None:
        return {}
    agent_dir = log_dir / "agent"
    return {
        "prompt": _read_text_if_exists(agent_dir / "prompt.md", max_chars),
        "stdout": _read_text_if_exists(agent_dir / "stdout.txt", max_chars),
        "stderr": _read_text_if_exists(agent_dir / "stderr.txt", max_chars),
        "command": _read_text_if_exists(agent_dir / "command.txt", max_chars),
    }


def _read_text_if_exists(path: Path | None, max_chars: int) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return _clip_text(path.read_text(encoding="utf-8", errors="replace"), max_chars)
    except OSError:
        return ""


def _extract_structured_feedback(feedback: Any) -> dict[str, Any] | None:
    if not isinstance(feedback, dict):
        return None
    direct = feedback.get("structured_feedback") or feedback.get(
        "structured_failure_feedback"
    )
    if isinstance(direct, dict):
        return direct
    stderr = feedback.get("stderr_tail")
    if not isinstance(stderr, str):
        return None
    return extract_structured_feedback(stderr)


def _build_edges(
    nodes: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    edges = []
    for node in nodes:
        child_value = _metric_value(node, metric)
        for parent_id in node.get("parents", []) or []:
            parent = node_by_id.get(parent_id)
            parent_value = _metric_value(parent, metric) if parent else None
            delta = (
                float(child_value) - float(parent_value)
                if child_value is not None and parent_value is not None
                else None
            )
            edges.append(
                {
                    "source": parent_id,
                    "target": node.get("id"),
                    "source_short_id": parent.get("short_id") if parent else parent_id[:8],
                    "target_short_id": node.get("short_id"),
                    "generation": node.get("generation"),
                    "metric": metric,
                    "parent_value": parent_value,
                    "child_value": child_value,
                    "delta": delta,
                    "changed_files": node.get("changed_files") or [],
                    "diff_stat": _dig(node, "git", "diff_stat") or "",
                }
            )
    return edges


def _available_metrics(nodes: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for node in nodes:
        metrics = node.get("metrics")
        if isinstance(metrics, dict):
            names.update(metrics)
    return sorted(names, key=lambda name: (name != "fitness", name))


def _trajectory(nodes: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    by_gen: dict[int, list[float]] = {}
    for node in nodes:
        value = _metric_value(node, metric)
        gen = node.get("generation")
        if value is None or gen is None:
            continue
        by_gen.setdefault(_safe_int(gen, 0), []).append(float(value))
    rows: list[dict[str, Any]] = []
    running_best: float | None = None
    for gen in sorted(by_gen):
        values = by_gen[gen]
        current_best = max(values)
        running_best = (
            current_best
            if running_best is None
            else max(running_best, current_best)
        )
        rows.append(
            {
                "generation": gen,
                "best": running_best,
                "generation_best": current_best,
                "mean": statistics.fmean(values),
                "count": len(values),
            }
        )
    return rows


def _best_node(nodes: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [node for node in nodes if _metric_value(node, metric) is not None]
    if not scored:
        return None
    return max(scored, key=lambda node: float(_metric_value(node, metric) or 0.0))


def _metric_value(node: dict[str, Any] | None, metric: str) -> float | None:
    if not node:
        return None
    metrics = node.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = value.get("value") or value.get("name")
        return str(raw) if raw else None
    return str(value) if value is not None else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def _clip_obj(value: Any, *, max_chars: int) -> Any:
    if value is None:
        return None
    text = json.dumps(value, default=str)
    if len(text) <= max_chars:
        return value
    return _clip_text(text, max_chars)


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...<truncated>...\n"
    keep = max(0, (max_chars - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]


def _render_html(payload: dict[str, Any]) -> str:
    embedded_json = (
        json.dumps(payload, sort_keys=True)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    title = html.escape(
        f"GigaEvo Repo Viz - {payload['redis']['prefix']}@{payload['redis']['db']}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f2;
  --ink: #18201f;
  --muted: #65706b;
  --line: #d7ddd5;
  --panel: #ffffff;
  --teal: #197b7a;
  --berry: #b23a5b;
  --amber: #b77812;
  --blue: #315c99;
  --bad: #c94c4c;
  --good: #21866f;
  --shadow: 0 10px 30px rgba(24, 32, 31, 0.08);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 18px 24px;
  border-bottom: 1px solid var(--line);
  background: #fbfcf8;
  position: sticky;
  top: 0;
  z-index: 5;
}}
h1 {{
  font-size: 20px;
  margin: 0 0 3px;
  letter-spacing: 0;
}}
.subtle {{ color: var(--muted); font-size: 13px; }}
.stats {{
  display: grid;
  grid-template-columns: repeat(6, minmax(86px, 1fr));
  gap: 10px;
  min-width: 650px;
}}
.stat {{
  border: 1px solid var(--line);
  background: var(--panel);
  padding: 8px 10px;
  border-radius: 7px;
}}
.stat b {{ display: block; font-size: 18px; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
main {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 430px;
  min-height: calc(100vh - 76px);
}}
.workspace {{
  min-width: 0;
  padding: 18px 20px 22px;
  display: grid;
  grid-template-rows: auto minmax(360px, 1fr) auto;
  gap: 16px;
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}}
select, input, button {{
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 7px;
  padding: 8px 10px;
  font: inherit;
  min-height: 36px;
}}
button {{
  cursor: pointer;
}}
button.active {{
  background: var(--teal);
  color: white;
  border-color: var(--teal);
}}
.panel {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.graph-wrap {{
  min-height: 360px;
  overflow: auto;
}}
.chart-wrap {{
  height: 220px;
  padding: 10px;
}}
svg {{ display: block; }}
.edge {{ stroke: #9aa7a0; stroke-width: 1.7; fill: none; opacity: 0.7; }}
.edge.positive {{ stroke: var(--good); }}
.edge.negative {{ stroke: var(--bad); }}
.edge-label {{ font-size: 11px; fill: var(--muted); paint-order: stroke; stroke: white; stroke-width: 3px; }}
.node circle {{ stroke: white; stroke-width: 2.5; cursor: pointer; }}
.node text {{ font-size: 12px; fill: var(--ink); pointer-events: none; paint-order: stroke; stroke: white; stroke-width: 3px; }}
.node.selected circle {{ stroke: var(--berry); stroke-width: 4; }}
.axis, .grid {{ stroke: var(--line); stroke-width: 1; }}
.series-best {{ fill: none; stroke: var(--teal); stroke-width: 2.8; }}
.series-mean {{ fill: none; stroke: var(--amber); stroke-width: 2; stroke-dasharray: 5 4; }}
.table-wrap {{ max-height: 320px; overflow: auto; }}
.usage-wrap {{ max-height: 240px; overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ position: sticky; top: 0; background: #fbfcf8; z-index: 1; color: var(--muted); font-weight: 650; }}
tr {{ cursor: pointer; }}
tr:hover {{ background: #f3f7f5; }}
tr.selected {{ background: #f9eef2; }}
aside {{
  border-left: 1px solid var(--line);
  background: #fbfcf8;
  min-width: 0;
  overflow: auto;
}}
.details {{
  padding: 18px;
}}
.details h2 {{
  margin: 0;
  font-size: 18px;
  letter-spacing: 0;
}}
.badge-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0 14px;
}}
.badge {{
  border: 1px solid var(--line);
  background: white;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 12px;
}}
.metric-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 12px 0 16px;
}}
.metric-cell {{
  border: 1px solid var(--line);
  background: white;
  border-radius: 7px;
  padding: 8px 10px;
}}
.metric-cell span {{ display: block; color: var(--muted); font-size: 12px; }}
.metric-cell b {{ font-size: 16px; }}
.lineage-section {{
  margin: 14px 0 16px;
}}
.lineage-section h3 {{
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0;
}}
.parent-list {{
  display: grid;
  gap: 8px;
}}
.parent-card {{
  width: 100%;
  display: grid;
  gap: 5px;
  text-align: left;
  background: white;
}}
.parent-card:hover:not(:disabled) {{
  background: #f3f7f5;
}}
.parent-card:disabled {{
  cursor: default;
  color: var(--ink);
}}
.parent-main {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-weight: 650;
}}
.parent-score {{
  white-space: nowrap;
  color: var(--muted);
  font-weight: 500;
}}
.parent-meta,
.parent-diff {{
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}}
.delta-positive {{ color: var(--good); }}
.delta-negative {{ color: var(--bad); }}
.tabs {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 12px 0;
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  background: #111816;
  color: #eef6f0;
  padding: 12px;
  border-radius: 7px;
  max-height: 520px;
  overflow: auto;
  font-size: 12px;
  line-height: 1.45;
}}
.empty {{
  color: var(--muted);
  padding: 18px;
}}
.files {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.file {{
  background: #eef4f2;
  color: #183c3a;
  padding: 4px 7px;
  border-radius: 5px;
  font-size: 12px;
}}
@media (max-width: 980px) {{
  header {{ align-items: flex-start; flex-direction: column; }}
  .stats {{ min-width: 0; width: 100%; grid-template-columns: repeat(2, 1fr); }}
  main {{ grid-template-columns: 1fr; }}
  aside {{ border-left: 0; border-top: 1px solid var(--line); }}
}}
</style>
</head>
<body>
<script id="evo-data" type="application/json">{embedded_json}</script>
<header>
  <div>
    <h1>GigaEvo Repo Evolution</h1>
    <div class="subtle" id="runMeta"></div>
  </div>
  <div class="stats">
    <div class="stat"><b id="statPrograms">0</b><span>programs</span></div>
    <div class="stat"><b id="statEdges">0</b><span>transitions</span></div>
    <div class="stat"><b id="statBest">-</b><span id="statBestLabel">best</span></div>
    <div class="stat"><b id="statGen">-</b><span>generations</span></div>
    <div class="stat"><b id="statTokens">-</b><span>Codex tokens</span></div>
    <div class="stat"><b id="statCost">-</b><span>estimated API cost</span></div>
  </div>
</header>
<main>
  <section class="workspace">
    <div class="toolbar">
      <label class="subtle" for="metricSelect">Metric</label>
      <select id="metricSelect"></select>
      <input id="searchBox" placeholder="Filter programs, commits, files" autocomplete="off">
      <button id="reloadButton" title="Reload data.json when served over HTTP">Reload</button>
      <button id="autoRefreshButton" title="Refresh this page periodically while evolution runs">Auto refresh</button>
    </div>
    <div class="panel graph-wrap"><svg id="graphSvg"></svg></div>
    <div class="panel chart-wrap"><svg id="chartSvg"></svg></div>
    <div class="panel usage-wrap">
      <table>
        <thead>
          <tr>
            <th>Evolution</th>
            <th>Calls</th>
            <th>Input</th>
            <th>Cached input</th>
            <th>Output</th>
            <th>Total</th>
            <th>Est. API cost</th>
          </tr>
        </thead>
        <tbody id="usageTable"></tbody>
      </table>
    </div>
    <div class="panel table-wrap">
      <table>
        <thead>
          <tr>
            <th>Gen</th>
            <th>Program</th>
            <th>Score</th>
            <th>Resolved</th>
            <th>Parent</th>
            <th>Tokens</th>
            <th>Est. cost</th>
            <th>Changed</th>
          </tr>
        </thead>
        <tbody id="nodeTable"></tbody>
      </table>
    </div>
  </section>
  <aside><div class="details" id="details"></div></aside>
</main>
<script>
let DATA = JSON.parse(document.getElementById("evo-data").textContent);
let selectedId = DATA.summary.best.program_id || (DATA.nodes[0] && DATA.nodes[0].id);
let activeTab = "reflection";
let autoRefresh = localStorage.getItem("repoVizAutoRefresh") !== "off";
let autoRefreshTimer = null;

const svgNS = "http://www.w3.org/2000/svg";
const byId = () => new Map(DATA.nodes.map(n => [n.id, n]));
const fmt = value => value === null || value === undefined || Number.isNaN(Number(value)) ? "-" : Number(value).toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
const fmtInt = value => value === null || value === undefined || Number.isNaN(Number(value)) ? "-" : Math.round(Number(value)).toLocaleString();
const fmtUsd = value => value === null || value === undefined || Number.isNaN(Number(value)) ? "unpriced" : "$" + Number(value).toFixed(4);
const fmtUsageUsd = usage => fmtUsd(usage && usage.estimated_cost_usd) + (usage && usage.estimated_cost_is_partial ? "+" : "");
const escText = value => value === null || value === undefined || value === "" ? "" : String(value);
const metricValue = (node, metric) => node && node.metrics && node.metrics[metric] !== undefined ? Number(node.metrics[metric]) : null;
const short = value => value ? String(value).slice(0, 12) : "-";

function init() {{
  document.getElementById("runMeta").textContent = `${{DATA.redis.prefix}}@${{DATA.redis.db}}  |  ${{DATA.run_dir}}`;
  const metricSelect = document.getElementById("metricSelect");
  metricSelect.innerHTML = "";
  DATA.available_metrics.forEach(name => {{
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    metricSelect.appendChild(opt);
  }});
  if (DATA.available_metrics.includes(DATA.metric)) metricSelect.value = DATA.metric;
  metricSelect.addEventListener("change", () => {{
    DATA.metric = metricSelect.value;
    render();
  }});
  document.getElementById("searchBox").addEventListener("input", render);
  document.getElementById("reloadButton").addEventListener("click", reloadData);
  document.getElementById("autoRefreshButton").addEventListener("click", toggleAutoRefresh);
  scheduleAutoRefresh();
  render();
}}

async function reloadData() {{
  try {{
    const response = await fetch("data.json?ts=" + Date.now(), {{cache: "no-store"}});
    if (!response.ok) return;
    DATA = await response.json();
    if (!byId().has(selectedId)) selectedId = DATA.summary.best.program_id || (DATA.nodes[0] && DATA.nodes[0].id);
    init();
  }} catch (error) {{
    console.warn("Reload works when the dashboard is served over HTTP.", error);
    location.reload();
  }}
}}

function toggleAutoRefresh() {{
  autoRefresh = !autoRefresh;
  localStorage.setItem("repoVizAutoRefresh", autoRefresh ? "on" : "off");
  scheduleAutoRefresh();
  render();
}}

function scheduleAutoRefresh() {{
  if (autoRefreshTimer) clearTimeout(autoRefreshTimer);
  const btn = document.getElementById("autoRefreshButton");
  if (btn) {{
    btn.className = autoRefresh ? "active" : "";
    btn.textContent = autoRefresh ? "Auto refresh on" : "Auto refresh off";
  }}
  if (autoRefresh) {{
    autoRefreshTimer = setTimeout(() => {{
      if (location.protocol === "file:") {{
        location.reload();
      }} else {{
        reloadData().finally(scheduleAutoRefresh);
      }}
    }}, 30000);
  }}
}}

function visibleNodes() {{
  const q = document.getElementById("searchBox").value.trim().toLowerCase();
  if (!q) return DATA.nodes;
  return DATA.nodes.filter(n => {{
    const hay = [
      n.id, n.short_id, n.commit, n.parent_commit, n.branch,
      ...(n.changed_files || [])
    ].join(" ").toLowerCase();
    return hay.includes(q);
  }});
}}

function render() {{
  const nodes = visibleNodes();
  const nodeIds = new Set(nodes.map(n => n.id));
  const edges = DATA.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
  const best = DATA.nodes.reduce((acc, n) => {{
    const v = metricValue(n, DATA.metric);
    if (v === null) return acc;
    return !acc || v > acc.value ? {{node: n, value: v}} : acc;
  }}, null);
  document.getElementById("statPrograms").textContent = DATA.nodes.length;
  document.getElementById("statEdges").textContent = DATA.edges.length;
  document.getElementById("statBest").textContent = best ? fmt(best.value) : "-";
  document.getElementById("statBestLabel").textContent = `best ${{DATA.metric}}`;
  document.getElementById("statGen").textContent = DATA.summary.generations.length || "-";
  const usage = DATA.usage && DATA.usage.summary ? DATA.usage.summary : {{}};
  document.getElementById("statTokens").textContent = fmtInt(usage.total_tokens);
  document.getElementById("statCost").textContent = fmtUsageUsd(usage);
  renderGraph(nodes, edges);
  renderChart(DATA.trajectories[DATA.metric] || []);
  renderUsageTable();
  renderTable(nodes);
  renderDetails(byId().get(selectedId) || nodes[0]);
}}

function scoreColor(value) {{
  if (value === null || value === undefined || Number.isNaN(value)) return "#87918c";
  const v = Math.max(0, Math.min(1, Number(value)));
  if (v < 0.5) return mix("#c94c4c", "#b77812", v * 2);
  return mix("#b77812", "#197b7a", (v - 0.5) * 2);
}}

function mix(a, b, t) {{
  const ah = a.replace("#", ""), bh = b.replace("#", "");
  const ar = parseInt(ah.slice(0,2),16), ag = parseInt(ah.slice(2,4),16), ab = parseInt(ah.slice(4,6),16);
  const br = parseInt(bh.slice(0,2),16), bg = parseInt(bh.slice(2,4),16), bb = parseInt(bh.slice(4,6),16);
  const rr = Math.round(ar + (br - ar) * t).toString(16).padStart(2,"0");
  const rg = Math.round(ag + (bg - ag) * t).toString(16).padStart(2,"0");
  const rb = Math.round(ab + (bb - ab) * t).toString(16).padStart(2,"0");
  return "#" + rr + rg + rb;
}}

function svgEl(name, attrs = {{}}, text = "") {{
  const el = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (text) el.textContent = text;
  return el;
}}

function renderGraph(nodes, edges) {{
  const svg = document.getElementById("graphSvg");
  svg.textContent = "";
  if (!nodes.length) {{
    svg.setAttribute("width", 900);
    svg.setAttribute("height", 360);
    svg.appendChild(svgEl("text", {{x: 24, y: 40, fill: "#65706b"}}, "No programs match the filter."));
    return;
  }}
  const gens = [...new Set(nodes.map(n => Number(n.generation || 0)))].sort((a,b) => a-b);
  const grouped = new Map(gens.map(g => [g, nodes.filter(n => Number(n.generation || 0) === g).sort((a,b) => (metricValue(b, DATA.metric) || -1) - (metricValue(a, DATA.metric) || -1))]));
  const maxRows = Math.max(...[...grouped.values()].map(items => items.length), 1);
  const width = Math.max(980, gens.length * 220 + 120);
  const height = Math.max(380, maxRows * 86 + 100);
  const left = 80, right = 70, top = 54, bottom = 50;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  const positions = new Map();
  gens.forEach((gen, gi) => {{
    const x = left + (gens.length === 1 ? 0 : gi * (width - left - right) / (gens.length - 1));
    svg.appendChild(svgEl("line", {{x1: x, y1: top - 25, x2: x, y2: height - bottom + 15, class: "grid"}}));
    svg.appendChild(svgEl("text", {{x, y: 24, "text-anchor": "middle", fill: "#65706b", "font-size": 12}}, `gen ${{gen}}`));
    const items = grouped.get(gen);
    items.forEach((node, idx) => {{
      const y = top + (idx + 1) * (height - top - bottom) / (items.length + 1);
      positions.set(node.id, {{x, y}});
    }});
  }});
  edges.forEach(edge => {{
    const a = positions.get(edge.source), b = positions.get(edge.target);
    if (!a || !b) return;
    const delta = edge.delta;
    const cls = delta === null || delta === undefined ? "edge" : delta >= 0 ? "edge positive" : "edge negative";
    svg.appendChild(svgEl("path", {{d: `M ${{a.x}} ${{a.y}} C ${{(a.x+b.x)/2}} ${{a.y}}, ${{(a.x+b.x)/2}} ${{b.y}}, ${{b.x}} ${{b.y}}`, class: cls}}));
    if (delta !== null && delta !== undefined) {{
      svg.appendChild(svgEl("text", {{x: (a.x+b.x)/2, y: (a.y+b.y)/2 - 5, class: "edge-label", "text-anchor": "middle"}}, `${{delta >= 0 ? "+" : ""}}${{fmt(delta)}}`));
    }}
  }});
  nodes.forEach(node => {{
    const pos = positions.get(node.id);
    if (!pos) return;
    const value = metricValue(node, DATA.metric);
    const g = svgEl("g", {{class: node.id === selectedId ? "node selected" : "node", tabindex: 0}});
    g.addEventListener("click", () => {{ selectedId = node.id; render(); }});
    g.appendChild(svgEl("circle", {{cx: pos.x, cy: pos.y, r: 17, fill: scoreColor(value)}}));
    g.appendChild(svgEl("text", {{x: pos.x + 24, y: pos.y - 2}}, node.short_id));
    g.appendChild(svgEl("text", {{x: pos.x + 24, y: pos.y + 14, fill: "#65706b"}}, fmt(value)));
    const title = svgEl("title", {{}}, `${{node.short_id}} ${{DATA.metric}}=${{fmt(value)}}`);
    g.appendChild(title);
    svg.appendChild(g);
  }});
}}

function renderChart(rows) {{
  const svg = document.getElementById("chartSvg");
  svg.textContent = "";
  const width = svg.parentElement.clientWidth - 20;
  const height = 200;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  if (!rows.length) {{
    svg.appendChild(svgEl("text", {{x: 16, y: 32, fill: "#65706b"}}, "No trajectory data."));
    return;
  }}
  const pad = {{l: 42, r: 18, t: 20, b: 32}};
  const xs = rows.map(r => Number(r.generation));
  const ys = rows.flatMap(r => [Number(r.best), Number(r.mean)]).filter(Number.isFinite);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
  const x = v => pad.l + (maxX === minX ? 0.5 : (v - minX) / (maxX - minX)) * (width - pad.l - pad.r);
  const y = v => height - pad.b - (maxY === minY ? 0.5 : (v - minY) / (maxY - minY)) * (height - pad.t - pad.b);
  svg.appendChild(svgEl("line", {{x1: pad.l, y1: height - pad.b, x2: width - pad.r, y2: height - pad.b, class: "axis"}}));
  svg.appendChild(svgEl("line", {{x1: pad.l, y1: pad.t, x2: pad.l, y2: height - pad.b, class: "axis"}}));
  const bestPath = rows.map((r, i) => `${{i ? "L" : "M"}} ${{x(Number(r.generation))}} ${{y(Number(r.best))}}`).join(" ");
  const meanPath = rows.map((r, i) => `${{i ? "L" : "M"}} ${{x(Number(r.generation))}} ${{y(Number(r.mean))}}`).join(" ");
  svg.appendChild(svgEl("path", {{d: bestPath, class: "series-best"}}));
  svg.appendChild(svgEl("path", {{d: meanPath, class: "series-mean"}}));
  svg.appendChild(svgEl("text", {{x: pad.l, y: 14, fill: "#197b7a", "font-size": 12}}, `best ${{DATA.metric}}`));
  svg.appendChild(svgEl("text", {{x: pad.l + 110, y: 14, fill: "#b77812", "font-size": 12}}, `mean ${{DATA.metric}}`));
  svg.appendChild(svgEl("text", {{x: pad.l, y: height - 8, fill: "#65706b", "font-size": 11}}, `gen ${{minX}}`));
  svg.appendChild(svgEl("text", {{x: width - pad.r, y: height - 8, fill: "#65706b", "font-size": 11, "text-anchor": "end"}}, `gen ${{maxX}}`));
}}

function renderTable(nodes) {{
  const tbody = document.getElementById("nodeTable");
  tbody.textContent = "";
  const sorted = [...nodes].sort((a,b) => Number(a.generation || 0) - Number(b.generation || 0) || (metricValue(b, DATA.metric) || -1) - (metricValue(a, DATA.metric) || -1));
  sorted.forEach(node => {{
    const tr = document.createElement("tr");
    if (node.id === selectedId) tr.className = "selected";
    tr.addEventListener("click", () => {{ selectedId = node.id; render(); }});
    const resolved = node.metrics && node.metrics.n_resolved !== undefined ? `${{fmt(node.metrics.n_resolved)}}/${{fmt(node.metrics.n_tests)}}` : "-";
    const parent = (node.parents || []).map(p => (byId().get(p) || {{short_id: short(p)}}).short_id).join(", ") || "-";
    const usage = node.codex_usage || {{}};
    const cells = [node.generation || "-", node.short_id, fmt(metricValue(node, DATA.metric)), resolved, parent, fmtInt(usage.total_tokens), fmtUsageUsd(usage), (node.changed_files || []).slice(0,3).join(", ") || "-"];
    cells.forEach(value => {{
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}

function renderUsageTable() {{
  const tbody = document.getElementById("usageTable");
  tbody.textContent = "";
  const rows = DATA.usage && DATA.usage.by_generation ? DATA.usage.by_generation : [];
  rows.forEach(row => {{
    const tr = document.createElement("tr");
    [
      `generation ${{row.generation}}`,
      fmtInt(row.calls),
      fmtInt(row.input_tokens),
      fmtInt(row.cached_input_tokens),
      fmtInt(row.output_tokens),
      fmtInt(row.total_tokens),
      fmtUsageUsd(row),
    ].forEach(value => {{
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  if (!rows.length) {{
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "empty";
    td.textContent = "No Codex usage recorded yet. New API-backed runs use Codex JSONL accounting.";
    tr.appendChild(td);
    tbody.appendChild(tr);
  }}
}}

function renderDetails(node) {{
  const root = document.getElementById("details");
  root.textContent = "";
  if (!node) {{
    const div = document.createElement("div");
    div.className = "empty";
    div.textContent = "No program selected.";
    root.appendChild(div);
    return;
  }}
  const title = document.createElement("h2");
  title.textContent = node.short_id;
  root.appendChild(title);
  const meta = document.createElement("div");
  meta.className = "subtle";
  meta.textContent = `gen ${{node.generation || "-"}}  |  ${{short(node.commit)}}  |  ${{node.state || "-"}}`;
  root.appendChild(meta);
  const badges = document.createElement("div");
  badges.className = "badge-row";
  const usage = node.codex_usage || {{}};
  [`${{DATA.metric}} ${{fmt(metricValue(node, DATA.metric))}}`, `agent ${{node.mutation_agent || "-"}}`, `duration ${{fmt(node.mutation_duration_seconds)}}s`, `tokens ${{fmtInt(usage.total_tokens)}}`, `est. ${{fmtUsageUsd(usage)}}`].forEach(text => {{
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = text;
    badges.appendChild(span);
  }});
  root.appendChild(badges);
  const metricGrid = document.createElement("div");
  metricGrid.className = "metric-grid";
  ["fitness", "n_resolved", "n_tests", "compile_valid"].forEach(name => {{
    if (!node.metrics || node.metrics[name] === undefined) return;
    const cell = document.createElement("div");
    cell.className = "metric-cell";
    const label = document.createElement("span");
    label.textContent = name;
    const val = document.createElement("b");
    val.textContent = fmt(node.metrics[name]);
    cell.appendChild(label);
    cell.appendChild(val);
    metricGrid.appendChild(cell);
  }});
  root.appendChild(metricGrid);
  root.appendChild(renderParents(node));
  const files = document.createElement("div");
  files.className = "files";
  (node.changed_files || []).forEach(file => {{
    const span = document.createElement("span");
    span.className = "file";
    span.textContent = file;
    files.appendChild(span);
  }});
  root.appendChild(files);
  const tabs = ["reflection", "feedback", "usage", "diff", "brief", "agent", "raw"];
  const tabBar = document.createElement("div");
  tabBar.className = "tabs";
  tabs.forEach(name => {{
    const button = document.createElement("button");
    button.textContent = name;
    if (name === activeTab) button.className = "active";
    button.addEventListener("click", () => {{ activeTab = name; renderDetails(node); }});
    tabBar.appendChild(button);
  }});
  root.appendChild(tabBar);
  const content = document.createElement("div");
  root.appendChild(content);
  addTab(content, "reflection", reflectionText(node));
  addTab(content, "feedback", feedbackText(node));
  addTab(content, "usage", usageText(node));
  addTab(content, "diff", `${{node.git && node.git.diff_stat ? node.git.diff_stat + "\\n" : ""}}${{node.git && node.git.diff ? node.git.diff : ""}}`);
  addTab(content, "brief", node.mutation_brief || "");
  addTab(content, "agent", agentText(node));
  addTab(content, "raw", JSON.stringify(node, null, 2));
}}

function renderParents(node) {{
  const section = document.createElement("section");
  section.className = "lineage-section";
  const heading = document.createElement("h3");
  heading.textContent = "Parents";
  section.appendChild(heading);
  const parentIds = node.parents || [];
  if (!parentIds.length) {{
    const empty = document.createElement("div");
    empty.className = "subtle";
    empty.textContent = "No recorded parents.";
    section.appendChild(empty);
    return section;
  }}
  const list = document.createElement("div");
  list.className = "parent-list";
  const nodesById = byId();
  parentIds.forEach(parentId => {{
    const parent = nodesById.get(parentId);
    const edge = DATA.edges.find(e => e.source === parentId && e.target === node.id);
    const edgeMatchesMetric = edge && edge.metric === DATA.metric;
    const value = parent ? metricValue(parent, DATA.metric) : edgeMatchesMetric ? edge.parent_value : null;
    const childValue = metricValue(node, DATA.metric);
    const card = document.createElement("button");
    card.className = "parent-card";
    card.type = "button";
    if (parent) {{
      card.addEventListener("click", () => {{
        selectedId = parent.id;
        render();
      }});
    }} else {{
      card.disabled = true;
    }}

    const main = document.createElement("div");
    main.className = "parent-main";
    const name = document.createElement("span");
    name.textContent = parent ? parent.short_id : short(parentId);
    const score = document.createElement("span");
    score.className = "parent-score";
    score.textContent = `${{DATA.metric}} ${{fmt(value)}}`;
    main.appendChild(name);
    main.appendChild(score);
    card.appendChild(main);

    const meta = document.createElement("div");
    meta.className = "parent-meta";
    if (parent) {{
      meta.textContent = `gen ${{parent.generation || "-"}}  |  ${{short(parent.commit)}}  |  ${{parent.state || "-"}}`;
    }} else {{
      meta.textContent = `${{parentId}}  |  not present in this data set`;
    }}
    card.appendChild(meta);

    if (edge) {{
      const diff = document.createElement("div");
      diff.className = "parent-diff";
      let delta = null;
      if (Number.isFinite(Number(childValue)) && Number.isFinite(Number(value))) {{
        delta = Number(childValue) - Number(value);
      }} else if (edgeMatchesMetric) {{
        delta = edge.delta;
      }}
      const deltaText = delta === null || delta === undefined ? "delta -" : `delta ${{delta >= 0 ? "+" : ""}}${{fmt(delta)}}`;
      const deltaSpan = document.createElement("span");
      deltaSpan.className = delta === null || delta === undefined ? "" : delta >= 0 ? "delta-positive" : "delta-negative";
      deltaSpan.textContent = deltaText;
      diff.appendChild(deltaSpan);
      const files = (edge.changed_files || []).slice(0, 4).join(", ");
      diff.appendChild(document.createTextNode(files ? `  |  changed into child: ${{files}}` : "  |  no changed files recorded"));
      card.appendChild(diff);
    }}

    list.appendChild(card);
  }});
  section.appendChild(list);
  return section;
}}

function addTab(root, name, text) {{
  const wrap = document.createElement("div");
  wrap.className = name === activeTab ? "tab-content active" : "tab-content";
  const pre = document.createElement("pre");
  pre.textContent = text || "N/A";
  wrap.appendChild(pre);
  root.appendChild(wrap);
}}

function reflectionText(node) {{
  const r = node.repo_reflection;
  if (!r) return node.mutation_context || "";
  if (typeof r === "string") return r;
  return [r.reflection || "", r.diff_stat ? "\\nDiff stat:\\n" + r.diff_stat : ""].join("\\n");
}}

function feedbackText(node) {{
  const parts = [];
  if (node.structured_feedback) {{
    parts.push(JSON.stringify(node.structured_feedback, null, 2));
  }}
  if (node.benchmark_feedback) {{
    parts.push(typeof node.benchmark_feedback === "string" ? node.benchmark_feedback : JSON.stringify(node.benchmark_feedback, null, 2));
  }}
  return parts.join("\\n\\n");
}}

function agentText(node) {{
  const logs = node.agent_logs || {{}};
  return [
    logs.command ? "Command:\\n" + logs.command : "",
    logs.stdout ? "\\nStdout:\\n" + logs.stdout : "",
    logs.stderr ? "\\nStderr:\\n" + logs.stderr : "",
    logs.prompt ? "\\nPrompt:\\n" + logs.prompt : ""
  ].filter(Boolean).join("\\n");
}}

function usageText(node) {{
  return JSON.stringify({{
    note: "Estimated API-equivalent USD price; cached input is included in input and reasoning is included in output. A trailing + means some calls use an unpriced model.",
    summary: node.codex_usage || {{}},
    calls: node.codex_usage_records || [],
  }}, null, 2);
}}

init();
</script>
</body>
</html>
"""
