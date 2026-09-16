from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
import yaml

from gigaevo.repo_harness.git_utils import (
    commit_all,
    ensure_git_repo,
    ensure_initial_commit,
    run_git,
)

SEED_SPEC_FILENAMES = ("repo_harness_seed.yaml", "repo_harness_seed.yml")


class AutoSeedError(RuntimeError):
    """Raised when repo-harness auto-seed setup cannot create a seed repo."""


@dataclass(frozen=True)
class AutoSeedResult:
    source_repo: Path
    spec_path: Path
    variant: str
    created: bool
    variants: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeedFile:
    path: Path
    content: str
    executable: bool = False


def configure_repo_harness_auto_seed(
    cfg: DictConfig,
    *,
    repo_root: str | Path,
) -> AutoSeedResult | None:
    """Resolve or create ``repo_harness.effective_source_repo`` in-place.

    Explicit ``repo_harness.source_repo`` wins. If it is absent and
    ``repo_harness.auto_seed.enabled`` is true, create a task-local scaffold repo
    from the problem's ``repo_harness_seed.yaml`` spec.
    """

    if OmegaConf.select(cfg, "repo_harness") is None:
        return None

    source_repo = _optional_str(cfg, "repo_harness.source_repo")
    effective_source_repo = _optional_str(cfg, "repo_harness.effective_source_repo")
    repo_root = Path(repo_root).expanduser().resolve()
    if source_repo:
        OmegaConf.update(
            cfg,
            "repo_harness.effective_source_repo",
            _expand_path(source_repo, base=repo_root),
        )
        return None
    if effective_source_repo:
        return None

    enabled = bool(
        OmegaConf.select(cfg, "repo_harness.auto_seed.enabled", default=False)
    )
    if not enabled:
        raise AutoSeedError(
            "repo_harness.source_repo is required unless "
            "repo_harness.auto_seed.enabled=true"
        )

    problem_dir = Path(_required_str(cfg, "problem.dir")).expanduser().resolve()
    problem_name = _required_str(cfg, "problem.name")
    redis_prefix = _optional_str(cfg, "redis.prefix") or problem_name
    root = _auto_seed_root(cfg, repo_root=repo_root)
    spec_path = _discover_seed_spec(
        problem_dir,
        spec_path=_optional_str(cfg, "repo_harness.auto_seed.spec_path"),
    )
    variant = _optional_str(cfg, "repo_harness.auto_seed.variant")
    variants = _optional_str_list(cfg, "repo_harness.auto_seed.variants")
    seed_name = _optional_str(cfg, "repo_harness.auto_seed.name") or redis_prefix
    commit_message = (
        _optional_str(cfg, "repo_harness.auto_seed.initial_commit_message")
        or "gigaevo initial auto seed"
    )

    result = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name=problem_name,
        seed_root=root,
        seed_name=seed_name,
        spec_path=spec_path,
        variant=variant,
        variants=variants,
        initial_commit_message=commit_message,
    )
    OmegaConf.update(
        cfg,
        "repo_harness.effective_source_repo",
        str(result.source_repo),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "repo_harness.seed_refs",
        list(result.refs),
        merge=False,
    )
    return result


def create_auto_seed_repo(
    *,
    problem_dir: str | Path,
    problem_name: str,
    seed_root: str | Path,
    seed_name: str,
    spec_path: str | Path | None = None,
    variant: str | None = None,
    variants: list[str] | tuple[str, ...] | None = None,
    initial_commit_message: str = "gigaevo initial auto seed",
) -> AutoSeedResult:
    problem_dir = Path(problem_dir).expanduser().resolve()
    seed_root = Path(seed_root).expanduser().resolve()
    selected_spec_path = (
        Path(spec_path).expanduser().resolve()
        if spec_path
        else _discover_seed_spec(problem_dir, spec_path=None)
    )
    spec = _load_seed_spec(selected_spec_path)
    selected_variants = _select_variants(
        spec,
        requested=variant,
        requested_many=variants,
    )
    selected_variant = selected_variants[0]

    if len(selected_variants) == 1:
        files = _normalize_files(spec, selected_variant, selected_spec_path)
        source_repo = (
            seed_root / _slug(problem_name) / _slug(seed_name) / _slug(selected_variant)
        )
        created = _materialize_seed_repo(
            source_repo,
            files=files,
            initial_commit_message=initial_commit_message,
        )
        refs = ("HEAD",)
    else:
        namespace = "multi-" + "-".join(_slug(item) for item in selected_variants)
        source_repo = seed_root / _slug(problem_name) / _slug(seed_name) / namespace
        files_by_variant = [
            (name, _normalize_files(spec, name, selected_spec_path))
            for name in selected_variants
        ]
        created, refs = _materialize_multi_variant_seed_repo(
            source_repo,
            files_by_variant=files_by_variant,
            initial_commit_message=initial_commit_message,
        )
    return AutoSeedResult(
        source_repo=source_repo,
        spec_path=selected_spec_path,
        variant=selected_variant,
        created=created,
        variants=tuple(selected_variants),
        refs=refs,
    )


def _materialize_seed_repo(
    source_repo: Path,
    *,
    files: list[SeedFile],
    initial_commit_message: str,
) -> bool:
    if source_repo.exists():
        if _has_git_head(source_repo):
            ensure_git_repo(source_repo)
            return False
        if any(path.name != ".git" for path in source_repo.iterdir()):
            raise AutoSeedError(
                f"Auto-seed target exists but has no Git HEAD: {source_repo}. "
                "Move it aside or provide repo_harness.source_repo explicitly."
            )
    else:
        source_repo.mkdir(parents=True)

    for item in files:
        path = source_repo / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.content, encoding="utf-8")
        if item.executable:
            path.chmod(path.stat().st_mode | 0o111)

    if not (source_repo / ".git").exists():
        subprocess.run(["git", "init", str(source_repo)], check=True)
    ensure_initial_commit(source_repo, message=initial_commit_message)
    return True


def _materialize_multi_variant_seed_repo(
    source_repo: Path,
    *,
    files_by_variant: list[tuple[str, list[SeedFile]]],
    initial_commit_message: str,
) -> tuple[bool, tuple[str, ...]]:
    refs = tuple(f"gigaevo-seed/{_slug(name)}" for name, _ in files_by_variant)
    if source_repo.exists():
        if _has_git_head(source_repo):
            ensure_git_repo(source_repo)
            missing = [
                ref
                for ref in refs
                if run_git(source_repo, ["rev-parse", "--verify", ref], check=False).returncode
                != 0
            ]
            if missing:
                raise AutoSeedError(
                    f"Multi-variant auto-seed repo is missing refs {missing}: "
                    f"{source_repo}. Move it aside so it can be recreated."
                )
            return False, refs
        if any(path.name != ".git" for path in source_repo.iterdir()):
            raise AutoSeedError(
                f"Auto-seed target exists but has no Git HEAD: {source_repo}. "
                "Move it aside or provide repo_harness.source_repo explicitly."
            )
    else:
        source_repo.mkdir(parents=True)

    if not (source_repo / ".git").exists():
        subprocess.run(["git", "init", str(source_repo)], check=True)

    for name, files in files_by_variant:
        _clear_seed_worktree(source_repo)
        for item in files:
            path = source_repo / item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item.content, encoding="utf-8")
            if item.executable:
                path.chmod(path.stat().st_mode | 0o111)
        commit = commit_all(
            source_repo,
            message=f"{initial_commit_message}: {name}",
        )
        run_git(source_repo, ["tag", f"gigaevo-seed/{_slug(name)}", commit])

    return True, refs


def _clear_seed_worktree(source_repo: Path) -> None:
    for path in source_repo.iterdir():
        if path.name == ".git":
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _discover_seed_spec(problem_dir: Path, *, spec_path: str | None) -> Path:
    if spec_path:
        path = Path(spec_path).expanduser()
        if not path.is_absolute():
            path = problem_dir / path
        path = path.resolve()
        if not path.is_file():
            raise AutoSeedError(f"Repo-harness auto-seed spec not found: {path}")
        return path

    matches = [
        path
        for name in SEED_SPEC_FILENAMES
        for path in problem_dir.rglob(name)
        if path.is_file()
    ]
    if not matches:
        names = ", ".join(SEED_SPEC_FILENAMES)
        raise AutoSeedError(
            f"repo_harness.auto_seed.enabled=true, but no {names} was found "
            f"under problem dir {problem_dir}"
        )
    if len(matches) > 1:
        formatted = "\n".join(f"- {path}" for path in sorted(matches))
        raise AutoSeedError(
            "Multiple repo-harness auto-seed specs found. Set "
            f"repo_harness.auto_seed.spec_path explicitly:\n{formatted}"
        )
    return matches[0].resolve()


def _load_seed_spec(spec_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AutoSeedError(f"Auto-seed spec must be a mapping: {spec_path}")
    return data


def _select_variant(spec: dict[str, Any], requested: str | None) -> str:
    variants = spec.get("variants")
    if variants is None:
        if "files" not in spec:
            raise AutoSeedError(
                "Auto-seed spec must define either top-level files or variants"
            )
        return requested or "default"
    if not isinstance(variants, dict) or not variants:
        raise AutoSeedError("Auto-seed spec variants must be a non-empty mapping")
    selected = requested or str(spec.get("default_variant") or next(iter(variants)))
    if selected not in variants:
        available = ", ".join(sorted(str(key) for key in variants))
        raise AutoSeedError(
            f"Unknown auto-seed variant '{selected}'. Available variants: {available}"
        )
    return selected


def _select_variants(
    spec: dict[str, Any],
    *,
    requested: str | None,
    requested_many: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if requested and requested_many:
        raise AutoSeedError(
            "Set only one of repo_harness.auto_seed.variant or "
            "repo_harness.auto_seed.variants"
        )
    if not requested_many:
        return [_select_variant(spec, requested)]

    selected: list[str] = []
    for item in requested_many:
        name = _select_variant(spec, str(item))
        if name in selected:
            raise AutoSeedError(f"Duplicate auto-seed variant '{name}'")
        selected.append(name)
    return selected


def _normalize_files(
    spec: dict[str, Any],
    variant: str,
    spec_path: Path,
) -> list[SeedFile]:
    variants = spec.get("variants")
    if variants is None:
        files = spec.get("files")
        package_files = spec.get("package_files", [])
    else:
        variant_spec = variants[variant]
        if not isinstance(variant_spec, dict):
            raise AutoSeedError(
                f"Auto-seed variant '{variant}' must be a mapping in {spec_path}"
            )
        files = variant_spec.get("files")
        package_files = variant_spec.get("package_files", [])

    if not isinstance(files, dict) or not files:
        raise AutoSeedError(
            f"Auto-seed variant '{variant}' must define a non-empty files mapping"
        )

    normalized: list[SeedFile] = []
    seen_paths: set[Path] = set()
    for raw_path, raw_spec in files.items():
        rel_path = _validate_relative_path(str(raw_path), spec_path)
        seen_paths.add(rel_path)
        if isinstance(raw_spec, str):
            normalized.append(SeedFile(path=rel_path, content=raw_spec))
            continue
        if not isinstance(raw_spec, dict):
            raise AutoSeedError(
                f"File spec for '{raw_path}' must be a string or mapping"
            )
        content = raw_spec.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise AutoSeedError(f"File content for '{raw_path}' must be a string")
        normalized.append(
            SeedFile(
                path=rel_path,
                content=content,
                executable=bool(raw_spec.get("executable", False)),
            )
        )
    normalized.extend(
        _normalize_package_files(
            package_files,
            variant=variant,
            spec_path=spec_path,
            seen_paths=seen_paths,
        )
    )
    return normalized


def _normalize_package_files(
    package_specs: Any,
    *,
    variant: str,
    spec_path: Path,
    seen_paths: set[Path],
) -> list[SeedFile]:
    if package_specs in (None, []):
        return []
    if not isinstance(package_specs, list):
        raise AutoSeedError(
            f"Auto-seed variant '{variant}' package_files must be a list"
        )

    normalized: list[SeedFile] = []
    for index, raw_spec in enumerate(package_specs):
        if not isinstance(raw_spec, dict):
            raise AutoSeedError(
                f"package_files entry {index} in variant '{variant}' must be a mapping"
            )
        package_name = _required_mapping_str(
            raw_spec,
            "package",
            f"package_files entry {index} in {spec_path}",
        )
        destination = _validate_relative_path(
            _required_mapping_str(
                raw_spec,
                "destination",
                f"package_files entry {index} in {spec_path}",
            ),
            spec_path,
        )
        include = raw_spec.get("include", ["**/*"])
        if not isinstance(include, list) or not all(
            isinstance(item, str) and item for item in include
        ):
            raise AutoSeedError(
                f"package_files entry {index} include must be a list of patterns"
            )
        import_rewrites = raw_spec.get("import_rewrites", {})
        if not isinstance(import_rewrites, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in import_rewrites.items()
        ):
            raise AutoSeedError(
                f"package_files entry {index} import_rewrites must be a mapping"
            )

        package_dir = _package_dir(package_name, spec_path)
        for source_path in sorted(
            path for path in package_dir.rglob("*") if path.is_file()
        ):
            rel_source = source_path.relative_to(package_dir)
            if "__pycache__" in rel_source.parts:
                continue
            rel_source_posix = rel_source.as_posix()
            if not any(
                fnmatch.fnmatch(rel_source_posix, pattern) for pattern in include
            ):
                continue
            dest_path = destination / rel_source
            if dest_path in seen_paths:
                raise AutoSeedError(
                    f"Duplicate auto-seed file path '{dest_path.as_posix()}' in {spec_path}"
                )
            content = source_path.read_text(encoding="utf-8")
            for old, new in import_rewrites.items():
                content = content.replace(old, new)
            seen_paths.add(dest_path)
            normalized.append(
                SeedFile(
                    path=dest_path,
                    content=content,
                    executable=bool(source_path.stat().st_mode & 0o111),
                )
            )
    return normalized


def _package_dir(package_name: str, spec_path: Path) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise AutoSeedError(
            f"Could not import package '{package_name}' referenced by {spec_path}"
        )
    locations = list(spec.submodule_search_locations or [])
    if not locations:
        raise AutoSeedError(
            f"Package '{package_name}' referenced by {spec_path} has no package directory"
        )
    path = Path(locations[0]).resolve()
    if not path.is_dir():
        raise AutoSeedError(
            f"Package directory for '{package_name}' does not exist: {path}"
        )
    return path


def _required_mapping_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AutoSeedError(f"{context} must define non-empty string '{key}'")
    return value.strip()


def _validate_relative_path(raw_path: str, spec_path: Path) -> Path:
    rel_path = Path(raw_path)
    if rel_path.is_absolute() or not raw_path or ".." in rel_path.parts:
        raise AutoSeedError(
            f"Invalid auto-seed file path '{raw_path}' in {spec_path}; paths "
            "must be relative and stay inside the seed repo"
        )
    return rel_path


def _auto_seed_root(cfg: DictConfig, *, repo_root: Path) -> Path:
    raw_root = _optional_str(cfg, "repo_harness.auto_seed.root")
    if raw_root:
        root = Path(raw_root).expanduser()
        return root.resolve() if root.is_absolute() else (repo_root / root).resolve()
    return (repo_root.parent / "repo_harness_seeds").resolve()


def _optional_str(cfg: DictConfig, path: str) -> str | None:
    try:
        value = OmegaConf.select(cfg, path, default=None)
    except MissingMandatoryValue:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_list(cfg: DictConfig, path: str) -> list[str] | None:
    try:
        value = OmegaConf.select(cfg, path, default=None)
    except MissingMandatoryValue:
        return None
    if value is None:
        return None
    items = OmegaConf.to_container(value, resolve=True)
    if not isinstance(items, list) or not items:
        raise AutoSeedError(f"{path} must be a non-empty list")
    normalized = [str(item).strip() for item in items]
    if any(not item for item in normalized):
        raise AutoSeedError(f"{path} entries must be non-empty strings")
    return normalized


def _required_str(cfg: DictConfig, path: str) -> str:
    value = _optional_str(cfg, path)
    if value is None:
        raise AutoSeedError(f"Missing required config value: {path}")
    return value


def _expand_path(path: str, *, base: Path) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return str(expanded.resolve())


def _has_git_head(path: Path) -> bool:
    top_level = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if top_level.returncode != 0:
        return False
    if Path(top_level.stdout.strip()).resolve() != path.resolve():
        return False

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "seed"
