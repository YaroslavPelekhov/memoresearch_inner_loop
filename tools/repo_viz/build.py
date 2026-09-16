#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gigaevo.repo_harness.viz import build_visualization, infer_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a static dashboard for a repo_harness evolution run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Hydra output directory, e.g. outputs/2026-05-10/11-34-19.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/viz.",
    )
    parser.add_argument(
        "--source-repo",
        default=None,
        help="Git repository containing candidate commits. Inferred when omitted.",
    )
    parser.add_argument(
        "--redis-host",
        default=None,
        help="Redis host. Inferred from the run config or localhost.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=None,
        help="Redis port. Inferred from the run config or 6379.",
    )
    parser.add_argument(
        "--db",
        type=int,
        default=None,
        help="Redis DB. Inferred from the run config when omitted.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Redis key prefix. Inferred from the run config when omitted.",
    )
    parser.add_argument(
        "--metric",
        default="fitness",
        help="Primary metric for coloring, ranking, and summary stats.",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=24000,
        help="Maximum diff text stored per node.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=60000,
        help="Maximum long text stored per node.",
    )
    args = parser.parse_args()

    config = infer_config(
        run_dir=Path(args.run_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.db,
        redis_prefix=args.prefix,
        source_repo=Path(args.source_repo) if args.source_repo else None,
        metric=args.metric,
        max_diff_chars=args.max_diff_chars,
        max_text_chars=args.max_text_chars,
    )
    result = build_visualization(config)
    print(f"Repo visualization written to {result.html_path}")
    print(f"Data: {result.data_path}")
    print(f"Programs: {result.node_count}")
    print(f"Transitions: {result.edge_count}")
    print(f"Best {result.metric}: {result.best_value}")


if __name__ == "__main__":
    main()
