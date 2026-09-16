from __future__ import annotations
import sys, os, types as _types, pathlib as _pathlib

import transformer_engine as te


def _setup_standalone_path() -> None:
    _repo = str(_pathlib.Path(__file__).resolve().parents[6])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    _contrib = os.path.join(_repo, "contrib")
    if os.path.isdir(_contrib):
        for _entry in sorted(os.listdir(_contrib)):
            _p = os.path.join(_contrib, _entry)
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
    _llmf = os.path.join(_repo, "llmfoundry")
    for _pkg, _dir in [
        ("llmfoundry",        _llmf),
        ("llmfoundry.models", os.path.join(_llmf, "models")),
    ]:
        if _pkg not in sys.modules:
            _stub = _types.ModuleType(_pkg)
            _stub.__path__ = [_dir]
            _stub.__package__ = _pkg
            _stub.__file__ = os.path.join(_dir, "__init__.py")
            sys.modules[_pkg] = _stub


def _te_version_check():
    major, minor, *_ = te.__version__.split(".")
    assert (int(major), int(minor)) == (2, 8), (
        f"TransformerEngine version must be 2.8.*, got {te.__version__}"
    )
