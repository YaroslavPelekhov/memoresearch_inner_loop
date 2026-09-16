"""Python wrapper for the fused row→col grouped FP8 CUDA kernel (v5).

Two-phase API:
  1. ``row2col_prepare``  – all blocking work (float8 view, group_offsets,
     CUDA buffer allocation).  Call **before** other GPU work.
  2. ``row2col_execute_cached`` – non-blocking: as_strided metadata ops,
     Python attr swaps, async kernel launch.  No CUDA malloc.

The CUDA extension is compiled JIT on first import.  To avoid the JIT
overhead, pre-build with::

    cd llmfoundry/models/ops/fp8/triton_kernels/cuda_row2col
    pip install -e .
"""

from __future__ import annotations

import os
import typing as tp

import torch

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockwiseQTensor,
    )
import transformer_engine_torch as tex

# ---------------------------------------------------------------------------
# Extension loading (JIT fallback)
# ---------------------------------------------------------------------------

_ext = None

def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    try:
        import row2col_cuda_ext as _mod
        _ext = _mod
        return _ext
    except ImportError:
        pass

    import warnings
    from torch.utils.cpp_extension import load as _load
    _DIR = os.path.dirname(os.path.abspath(__file__))

    _src = os.path.join(_DIR, "row2col_kernel.cu")
    assert os.path.isfile(_src), f"CUDA source not found: {_src}"

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        _arch = f"-arch=sm_{major}{minor}"
    else:
        _arch = "-arch=sm_90"

    warnings.warn(
        f"row2col_cuda_ext: JIT-compiling CUDA extension ({_arch}). "
        "This takes 30-60 s on first import; subsequent imports use cache.",
        stacklevel=3,
    )

    _ext = _load(
        name="row2col_cuda_ext",
        sources=[_src],
        extra_cuda_cflags=[
            "-O3", _arch, "--use_fast_math",
            "-lineinfo", "-std=c++17",
        ],
        verbose=True,
    )
    return _ext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128

_kFloat8E4M3 = tex.DType.kFloat8E4M3
_bfloat16 = torch.bfloat16


def _maybe_float8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 or float8_e4m3fn, got {t.dtype}")
    return t


def _precompute_group_offsets(
    list_groups_sizes: tp.List[int],
    device: torch.device,
) -> torch.Tensor:
    """Prefix-sum of group sizes → int64 tensor of length G+1."""
    offsets = [0]
    for gs in list_groups_sizes:
        offsets.append(offsets[-1] + gs)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


# ---------------------------------------------------------------------------
# Two-phase API (v5)
# ---------------------------------------------------------------------------

_single_cache: tp.Dict[str, tp.Any] = {}


def row2col_prepare(
    data: torch.Tensor,
    scales: torch.Tensor,
    list_groups_sizes: tp.List[int],
    m_indices: torch.Tensor,
) -> tp.Tuple:
    """Phase 1: all blocking work (float8 view, group_offsets, buffer alloc).

    Call **before** launching other GPU work (e.g. DGRAD) so the CUDA
    mallocs don't create a pipeline bubble afterwards.
    """
    data = _maybe_float8(data).contiguous()
    group_offsets = _precompute_group_offsets(
        list_groups_sizes, data.device)
    m_idx = m_indices.contiguous().to(torch.int32)
    Y, S = _get_ext().row2col_alloc(data)
    return (data, scales, group_offsets, m_idx, list_groups_sizes, Y, S)


def row2col_execute_cached(
    prep: tp.Tuple,
    *,
    cache_slot: str = "default",
) -> tp.Tuple[Float8BlockwiseQTensor, ...]:
    """Phase 2: non-blocking — only as_strided metadata ops, Python attr
    swaps, and async kernel launch.  No CUDA malloc.
    """
    global _single_cache
    data, scales, group_offsets, m_idx, gs_list, Y, S = prep
    key = tuple(gs_list)

    prev = _single_cache.get(cache_slot)
    if prev is not None and prev[0] == key:
        cached = prev[1]
        result = _get_ext().row2col_cached_prealloc(
            data, scales, group_offsets, m_idx,
            gs_list, cached, Y, S,
        )
        return tuple(result)

    result = _get_ext().row2col_wrapped_prealloc(
        data, scales, group_offsets, m_idx,
        gs_list,
        Float8BlockwiseQTensor, _kFloat8E4M3, _bfloat16, Y, S,
    )
    _single_cache[cache_slot] = (key, result)
    return tuple(result)
