from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess


@dataclass(frozen=True)
class GitCommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class GitError(RuntimeError):
    def __init__(self, message: str, result: GitCommandResult | None = None):
        super().__init__(message)
        self.result = result


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "GigaEvo")
    env.setdefault("GIT_AUTHOR_EMAIL", "gigaevo@example.local")
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


def run_git(
    repo: str | Path,
    args: list[str],
    *,
    check: bool = True,
    timeout: float = 120.0,
) -> GitCommandResult:
    repo_path = Path(repo).expanduser().resolve()
    cmd = ["git", "-C", str(repo_path), *args]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_git_env(),
    )
    result = GitCommandResult(
        args=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}",
            result,
        )
    return result


def ensure_git_repo(repo: str | Path, *, init_if_needed: bool = False) -> Path:
    repo_path = Path(repo).expanduser().resolve()
    if not repo_path.exists():
        raise GitError(f"Repository path does not exist: {repo_path}")
    result = run_git(repo_path, ["rev-parse", "--show-toplevel"], check=False)
    if result.returncode == 0:
        return Path(result.stdout.strip()).resolve()
    if not init_if_needed:
        raise GitError(f"Not a Git repository: {repo_path}", result)
    subprocess.run(["git", "init", str(repo_path)], check=True, env=_git_env())
    return repo_path


def rev_parse(repo: str | Path, ref: str = "HEAD") -> str:
    return run_git(repo, ["rev-parse", ref]).stdout.strip()


def current_branch(repo: str | Path) -> str | None:
    result = run_git(repo, ["branch", "--show-current"], check=False)
    branch = result.stdout.strip()
    return branch or None


def status_porcelain(repo: str | Path) -> str:
    return run_git(repo, ["status", "--porcelain"], check=False).stdout


def ensure_initial_commit(repo: str | Path, *, message: str) -> str:
    """Create an initial commit when a repo has no HEAD yet."""
    if run_git(repo, ["rev-parse", "--verify", "HEAD"], check=False).returncode == 0:
        return rev_parse(repo)
    run_git(repo, ["add", "-A"])
    if not status_porcelain(repo).strip():
        raise GitError(f"Cannot create initial commit for empty repository: {repo}")
    run_git(repo, ["commit", "-m", message])
    return rev_parse(repo)


def add_worktree(
    repo: str | Path,
    worktree: str | Path,
    commit: str,
    *,
    branch: str | None = None,
    detach: bool = False,
) -> Path:
    worktree_path = Path(worktree).expanduser().resolve()
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if worktree_path.exists():
        raise GitError(f"Worktree path already exists: {worktree_path}")
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    if branch:
        args.extend(["-B", branch])
    args.extend([str(worktree_path), commit])
    run_git(repo, args)
    return worktree_path


def remove_worktree(repo: str | Path, worktree: str | Path) -> None:
    run_git(repo, ["worktree", "remove", "--force", str(Path(worktree).resolve())])
    run_git(repo, ["worktree", "prune"], check=False)


def commit_all(repo: str | Path, *, message: str, allow_empty: bool = False) -> str:
    run_git(repo, ["add", "-A"])
    dirty = status_porcelain(repo).strip()
    if not dirty and not allow_empty:
        return rev_parse(repo)
    args = ["commit", "-m", message]
    if allow_empty:
        args.insert(1, "--allow-empty")
    run_git(repo, args)
    return rev_parse(repo)


def changed_files(repo: str | Path, base: str, head: str) -> list[str]:
    result = run_git(repo, ["diff", "--name-only", f"{base}..{head}"], check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def diff_stat(repo: str | Path, base: str, head: str) -> str:
    return run_git(repo, ["diff", "--stat", f"{base}..{head}"], check=False).stdout
