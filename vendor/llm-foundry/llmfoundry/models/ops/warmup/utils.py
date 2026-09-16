import os
import pathlib as _pathlib
import sys
import types as _types
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml

_BLOCK = 128  # Triton BLOCK_SIZE shared by all kernels here
_MAX_GROUPS_DEEPGEMM = 128  # MAX_NUM_GROUPS constexpr in row2col_special


def _round_up(x: int, multiple: int) -> int:
    return x + (multiple - x % multiple) % multiple

@dataclass
class TritonWarmupConfig:
    cache_dir: str
    num_groups: int
    int_dim: int
    hid_dim: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    batch_sizes: list[int]
    seq_lens: list[int]
    rope_num_heads: int
    dense_int_dim: int
    dense_hid_dim: int
    tokens_small: list[int]
    tokens_large: list[int]

@dataclass(frozen=True)
class DeepGemmWarmupCfg:
    cache_dir: str
    device: str
    dtype: torch.dtype

    hid_dim: int
    int_dim: int
    num_groups: int

    # IMPORTANT: step/min_size/max_size interpretation:
    #   - step: granularity (tokens) you want to cover (e.g., 128)
    #   - min_size: MIN TOTAL (m_sum / k_sum) to warm (not per-expert)
    #   - max_size: MAX PER-EXPERT tokens you want to allow in synthesized group sizes
    #              (buffers are sized as max_size * num_groups)
    step: int
    min_size: int
    max_size: int

    # Cap the token-buffer allocation to avoid OOM when max_size * num_groups
    # is very large.  The sweep is limited to totals that fit within this
    # budget.  None = no cap (default: max_size * num_groups).
    max_buffer_tokens: Optional[int] = field(default=None)
    num_sms: Optional[int] = field(default=None)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_warmup_config(config_path: str) -> TritonWarmupConfig:
    cfg = load_config(config_path)["triton"]
    cfg =  TritonWarmupConfig(
            cache_dir=cfg["cache_dir"],
            num_groups=cfg["num_groups"],
            int_dim=cfg["int_dim"],
            hid_dim=cfg["hid_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            batch_sizes=cfg["batch_sizes"],
            seq_lens=cfg["seq_lens"],
            rope_num_heads=cfg["rope_num_heads"],
            dense_int_dim=cfg["dense_int_dim"],
            dense_hid_dim=cfg["dense_hid_dim"],
            tokens_small=cfg["tokens_small"],
            tokens_large=cfg["tokens_large"],
    )
    return cfg


# Bootstrap sys.path and inject lightweight stubs so this script can be run
# directly without the full project being pip-installed:
#   1. Repo root is added to sys.path so all llmfoundry sub-packages are found.
#   2. Every contrib/* sub-repo root is added (composer, streaming, …).
#   3. Stubs for llmfoundry and llmfoundry.models prevent their heavy
#      __init__.py files from being executed; all modules under
#      llmfoundry/models/ops/float8/ have empty __init__.py files and load fine.
def _setup_standalone_path() -> None:
    # __file__ = <repo>/llmfoundry/models/ops/warmup/utils.py
    # parents: [0]=warmup  [1]=ops  [2]=models  [3]=llmfoundry  [4]=<repo>
    _repo = str(_pathlib.Path(__file__).resolve().parents[4])

    # Add repo root itself.
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    # Add every contrib/<pkg-repo>/ directory (each contains its Python package).
    _contrib = os.path.join(_repo, "contrib")
    if os.path.isdir(_contrib):
        for _entry in sorted(os.listdir(_contrib)):
            _entry_path = os.path.join(_contrib, _entry)
            if os.path.isdir(_entry_path) and _entry_path not in sys.path:
                sys.path.insert(0, _entry_path)

    # Stub out llmfoundry and llmfoundry.models so their __init__.py files
    # (which import composer, transformers, torchmetrics, …) are never run.
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