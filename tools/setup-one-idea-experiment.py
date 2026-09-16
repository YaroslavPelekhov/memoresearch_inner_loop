#!/usr/bin/env python3
"""Prepare a fully isolated, runnable one-idea evolution experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        ".venv312",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "runs",
        "prepared",
    }
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _write_launcher(path: Path, *, use_custom_idea: bool) -> None:
    idea_argument = ' \\\n+  --idea-file "$experiment_root/idea.yaml"' if use_custom_idea else ""
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "experiment_root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "project_root=\"$(cd \"$experiment_root/../..\" && pwd)\"\n"
        "default_python=\"$project_root/../autoresearch-dclm-ideas/.venv/bin/python\"\n"
        "python_bin=\"${AUTORESEARCH_PYTHON:-$default_python}\"\n"
        "if [[ ! -x \"$python_bin\" ]]; then\n"
        "  python_bin=\"${AUTORESEARCH_PYTHON:-python3}\"\n"
        "fi\n"
        "cd \"$project_root\"\n"
        "exec \"$python_bin\" -m autoresearch.ideas.campaign \\\n"
        "  --repo-root \"$experiment_root/candidate-repo\" \\\n"
        "  --campaign-root \"$experiment_root/campaign\" \\\n"
        "  --task \"$project_root/experiments/one_idea/research_task.yaml\" \\\n"
        "  --rounds 1"
        f"{idea_argument}\n"
    )
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    candidate_repo = output / "candidate-repo"
    shutil.copytree(ROOT, candidate_repo, ignore=_ignore)

    _run("git", "init", cwd=candidate_repo)
    _run("git", "config", "user.email", "autoresearch@local", cwd=candidate_repo)
    _run("git", "config", "user.name", "Autoresearch", cwd=candidate_repo)
    _run("git", "add", "-A", cwd=candidate_repo)
    _run("git", "commit", "-m", "isolated one-idea baseline", cwd=candidate_repo)

    idea_path = output / "idea.yaml"
    shutil.copy2(ROOT / "experiments/one_idea/idea.template.yaml", idea_path)
    _write_launcher(output / "run-custom.sh", use_custom_idea=True)
    _write_launcher(output / "run-generated.sh", use_custom_idea=False)

    print(f"Prepared isolated candidate repository: {candidate_repo}")
    print(f"Custom idea template: {idea_path}")
    print(f"Custom launch: {output / 'run-custom.sh'}")
    print(f"Generated-idea launch: {output / 'run-generated.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
