from __future__ import annotations

import importlib
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from gigaevo.repo_harness.auto_seed import (
    AutoSeedError,
    configure_repo_harness_auto_seed,
    create_auto_seed_repo,
)
from gigaevo.repo_harness.git_utils import rev_parse, run_git


def _write_seed_spec(problem_dir: Path) -> Path:
    spec = problem_dir / "repo_harness_seed.yaml"
    spec.write_text(
        """
version: 1
default_variant: default
variants:
  default:
    files:
      README.md:
        content: |
          empty seed
      run.sh:
        executable: true
        content: |
          #!/usr/bin/env bash
          exit 0
  alt:
    files:
      main.py:
        content: |
          raise SystemExit(0)
""".lstrip(),
        encoding="utf-8",
    )
    return spec


def test_create_auto_seed_repo_materializes_variant_and_commit(tmp_path: Path):
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    spec_path = _write_seed_spec(problem_dir)

    result = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name="repo_harness_template",
        seed_root=tmp_path / "seeds",
        seed_name="run/a",
        spec_path=spec_path,
        variant="default",
    )

    assert result.created is True
    assert result.variant == "default"
    assert (result.source_repo / "README.md").read_text() == "empty seed\n"
    assert (result.source_repo / "run.sh").stat().st_mode & 0o111
    assert rev_parse(result.source_repo)

    reused = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name="repo_harness_template",
        seed_root=tmp_path / "seeds",
        seed_name="run/a",
        spec_path=spec_path,
        variant="default",
    )

    assert reused.created is False
    assert reused.source_repo == result.source_repo


def test_create_auto_seed_repo_materializes_multiple_variants_as_refs(tmp_path: Path):
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    spec_path = _write_seed_spec(problem_dir)

    result = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name="repo_harness_template",
        seed_root=tmp_path / "seeds",
        seed_name="run/multi",
        spec_path=spec_path,
        variants=["default", "alt"],
    )

    assert result.created is True
    assert result.variants == ("default", "alt")
    assert result.refs == ("gigaevo-seed/default", "gigaevo-seed/alt")
    assert len({rev_parse(result.source_repo, ref) for ref in result.refs}) == 2
    default_files = run_git(
        result.source_repo, ["ls-tree", "--name-only", result.refs[0]]
    ).stdout.splitlines()
    alt_files = run_git(
        result.source_repo, ["ls-tree", "--name-only", result.refs[1]]
    ).stdout.splitlines()
    assert default_files == ["README.md", "run.sh"]
    assert alt_files == ["main.py"]

    reused = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name="repo_harness_template",
        seed_root=tmp_path / "seeds",
        seed_name="run/multi",
        spec_path=spec_path,
        variants=["default", "alt"],
    )
    assert reused.created is False
    assert reused.refs == result.refs


def test_create_auto_seed_repo_vendors_package_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package_root = tmp_path / "packages"
    package_dir = package_root / "sample_pkg"
    (package_dir / "templates").mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "from sample_pkg.core import Tool\n",
        encoding="utf-8",
    )
    (package_dir / "core.py").write_text(
        "from sample_pkg.helper import help_me\n\nclass Tool: pass\n",
        encoding="utf-8",
    )
    (package_dir / "helper.py").write_text("def help_me(): pass\n", encoding="utf-8")
    (package_dir / "templates" / "prompt.txt").write_text("hello\n", encoding="utf-8")
    (package_dir / "__pycache__").mkdir()
    (package_dir / "__pycache__" / "core.pyc").write_bytes(b"ignored")
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("sample_pkg", None)
    importlib.invalidate_caches()

    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    spec_path = problem_dir / "repo_harness_seed.yaml"
    spec_path.write_text(
        """
version: 1
files:
  README.md:
    content: seed
package_files:
  - package: sample_pkg
    destination: vendored/sample_pkg
    include:
      - "*.py"
      - "templates/*.txt"
    import_rewrites:
      sample_pkg: vendored.sample_pkg
""".lstrip(),
        encoding="utf-8",
    )

    result = create_auto_seed_repo(
        problem_dir=problem_dir,
        problem_name="problem",
        seed_root=tmp_path / "seeds",
        seed_name="run",
        spec_path=spec_path,
    )

    assert (result.source_repo / "vendored/sample_pkg/__init__.py").is_file()
    assert (result.source_repo / "vendored/sample_pkg/templates/prompt.txt").is_file()
    assert not (result.source_repo / "vendored/sample_pkg/__pycache__").exists()
    assert "from vendored.sample_pkg.helper import help_me" in (
        result.source_repo / "vendored/sample_pkg/core.py"
    ).read_text(encoding="utf-8")
    assert rev_parse(result.source_repo)


def test_configure_auto_seed_sets_effective_source_repo(tmp_path: Path):
    repo_root = tmp_path / "gigaevo-core-internal"
    repo_root.mkdir()
    problem_dir = repo_root / "problems" / "repo_harness_template"
    problem_dir.mkdir(parents=True)
    _write_seed_spec(problem_dir)

    cfg = OmegaConf.create(
        {
            "problem": {
                "name": "repo_harness_template",
                "dir": str(problem_dir),
            },
            "redis": {"prefix": "tb2-smoke"},
            "repo_harness": {
                "source_repo": None,
                "effective_source_repo": None,
                "auto_seed": {"enabled": True},
            },
        }
    )

    result = configure_repo_harness_auto_seed(cfg, repo_root=repo_root)

    expected = (
        tmp_path
        / "repo_harness_seeds"
        / "repo_harness_template"
        / "tb2-smoke"
        / "default"
    )
    assert result is not None
    assert result.source_repo == expected
    assert cfg.repo_harness.effective_source_repo == str(expected)
    assert (expected / "README.md").is_file()


def test_configure_auto_seed_sets_multiple_seed_refs(tmp_path: Path):
    repo_root = tmp_path / "gigaevo-core-internal"
    repo_root.mkdir()
    problem_dir = repo_root / "problems" / "repo_harness_template"
    problem_dir.mkdir(parents=True)
    _write_seed_spec(problem_dir)

    cfg = OmegaConf.create(
        {
            "problem": {
                "name": "repo_harness_template",
                "dir": str(problem_dir),
            },
            "redis": {"prefix": "tb2-multi"},
            "repo_harness": {
                "source_repo": None,
                "effective_source_repo": None,
                "seed_refs": None,
                "auto_seed": {
                    "enabled": True,
                    "variants": ["default", "alt"],
                },
            },
        }
    )

    result = configure_repo_harness_auto_seed(cfg, repo_root=repo_root)

    assert result is not None
    assert list(cfg.repo_harness.seed_refs) == list(result.refs)
    assert result.variants == ("default", "alt")


def test_configure_auto_seed_prefers_explicit_source_repo(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run_git(tmp_path, ["init", str(source)])
    (source / "README.md").write_text("manual\n", encoding="utf-8")
    run_git(source, ["add", "-A"])
    run_git(source, ["commit", "-m", "initial"])

    cfg = OmegaConf.create(
        {
            "problem": {"name": "x", "dir": str(tmp_path / "problem")},
            "redis": {"prefix": "x"},
            "repo_harness": {
                "source_repo": str(source),
                "effective_source_repo": None,
                "auto_seed": {"enabled": True},
            },
        }
    )

    result = configure_repo_harness_auto_seed(cfg, repo_root=tmp_path / "repo")

    assert result is None
    assert cfg.repo_harness.effective_source_repo == str(source.resolve())


def test_auto_seed_requires_problem_spec(tmp_path: Path):
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()

    with pytest.raises(AutoSeedError, match="no repo_harness_seed"):
        create_auto_seed_repo(
            problem_dir=problem_dir,
            problem_name="problem",
            seed_root=tmp_path / "seeds",
            seed_name="run",
        )
