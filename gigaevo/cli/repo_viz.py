"""Repo harness visualization command."""

from __future__ import annotations

from pathlib import Path

import click

from gigaevo.repo_harness.viz import build_visualization, infer_config


@click.command("repo-viz")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Hydra output directory for the run, e.g. outputs/2026-05-10/11-34-19.",
)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for index.html and data.json. Defaults to <run-dir>/viz.",
)
@click.option(
    "--source-repo",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Git repository containing candidate commits. Inferred from .hydra/config.yaml when omitted.",
)
@click.option(
    "--prefix",
    "redis_prefix",
    default=None,
    help="Redis key prefix. Inferred from .hydra/config.yaml when omitted.",
)
@click.option(
    "--db",
    "redis_db",
    type=int,
    default=None,
    help="Redis database. Inferred from .hydra/config.yaml when omitted.",
)
@click.option(
    "--metric",
    default="fitness",
    show_default=True,
    help="Primary metric for coloring, ranking, and summary stats.",
)
@click.option(
    "--max-diff-chars",
    type=int,
    default=24000,
    show_default=True,
    help="Maximum diff text stored per node.",
)
@click.option(
    "--max-text-chars",
    type=int,
    default=60000,
    show_default=True,
    help="Maximum long text stored per node for feedback, reflection, and logs.",
)
@click.pass_context
def repo_viz(
    ctx: click.Context,
    run_dir: Path,
    out_dir: Path | None,
    source_repo: Path | None,
    redis_prefix: str | None,
    redis_db: int | None,
    metric: str,
    max_diff_chars: int,
    max_text_chars: int,
) -> None:
    """Build a static dashboard for a repo_harness evolution run.

    The command reads canonical program lineage and metrics from Redis, then
    enriches each node with artifacts from the Hydra run directory and Git
    diffs from the source repository. The result is a self-contained
    ``index.html`` plus ``data.json``.
    """
    if max_diff_chars < 0:
        raise click.BadParameter(
            "--max-diff-chars must be >= 0", param_hint="--max-diff-chars"
        )
    if max_text_chars < 0:
        raise click.BadParameter(
            "--max-text-chars must be >= 0", param_hint="--max-text-chars"
        )

    config = infer_config(
        run_dir=run_dir,
        out_dir=out_dir,
        redis_host=ctx.obj["redis_host"],
        redis_port=ctx.obj["redis_port"],
        redis_db=redis_db,
        redis_prefix=redis_prefix,
        source_repo=source_repo,
        metric=metric,
        max_diff_chars=max_diff_chars,
        max_text_chars=max_text_chars,
    )
    result = build_visualization(config)
    click.echo(
        "\n".join(
            [
                f"Repo visualization written to {result.html_path}",
                f"Data: {result.data_path}",
                f"Programs: {result.node_count}",
                f"Transitions: {result.edge_count}",
                f"Best {result.metric}: {result.best_value}",
            ]
        )
    )
