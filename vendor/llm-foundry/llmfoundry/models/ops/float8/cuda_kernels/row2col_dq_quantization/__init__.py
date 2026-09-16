"""Python wrapper for the fused row→col grouped FP8 CUDA kernel.

No TransformerEngine dependency.  Returns raw ``(col_data_flat, col_scale_inv)``
tensors that can be sliced per-group by the caller.

The CUDA extension is compiled JIT on first import.  To avoid the JIT
overhead, pre-build with::

    cd llmfoundry/models/ops/float8/cuda_kernels/row2col_dq_quantization
    pip install -e .

Public API
----------
- ``row2col_requantize``  – alloc + launch + return raw tensors.
"""

from __future__ import annotations

import os
import typing as tp

import torch

# ---------------------------------------------------------------------------
# Extension loading (pre-built → JIT fallback)
# ---------------------------------------------------------------------------

_ext = None
_ext_warmed_up = False


def warmup_ext() -> None:
    """Eagerly compile/load the CUDA extension and barrier-sync all ranks.

    Must be called once during model initialisation, before any DeepEP
    dispatch/combine operations start.  Without this, the first backward pass
    that hits ``row2col_requantize`` triggers a 30-60 s JIT compilation that
    blocks only the ranks with active tokens (e.g. rank 0 under
    FirstExpertGroupGate), causing DeepEP timeouts on all other ranks.
    """
    global _ext_warmed_up
    if _ext_warmed_up:
        return
    _get_ext()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    _ext_warmed_up = True


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    try:
        import row2col_dq_cuda_ext as _mod
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
        f"row2col_dq_cuda_ext: JIT-compiling CUDA extension ({_arch}). "
        + "This takes 30-60 s on first import; subsequent imports use cache.",
        stacklevel=3,
    )

    _ext = _load(
        name="row2col_dq_cuda_ext",
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


def _maybe_float8(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is uint8 (the in-memory representation of float8)."""
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 or float8_e4m3fn, got {t.dtype}")
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def row2col_requantize(
    tensor: torch.Tensor,
    m_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    scales_inv: torch.Tensor,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Fused rowwise-FP8 → grouped-columnwise-FP8 re-quantization.

    Drop-in replacement for ``row2col_requantization_deepgemm_layout_fn``
    with no TransformerEngine dependency.

    Group-offset prefix-sum is computed inside the CUDA kernel (thread 0,
    overlapped with Phase 1 global loads) — no host-side unfused ops.

    Parameters
    ----------
    tensor : (T, H) float8_e4m3fn or uint8
        Input FP8 data in row-major layout.
    m_indices : (T,) int32
        Group index for each token row.
    group_sizes : (G,) int tensor
        Token count per group (each a multiple of 128).
    scales_inv : (T, H // 128) float32
        Row-wise dequantization scales (``dequant = fp8 * scale_inv``).

    Returns
    -------
    col_data_flat : (H * T,) uint8
        Flat FP8 buffer.  Groups packed contiguously; group *g* occupies
        ``[offset_g .. offset_g + H * gs_g)`` stored as ``(H, gs_g)`` row-major
        (= columnwise layout of the original ``(gs_g, H)`` matrix).
    col_scale_inv : (T // 128, H) float32
        Block column-wise scales.  Groups contiguous along dim 0.
    """
    assert tensor.is_cuda and scales_inv.is_cuda
    assert tensor.ndim == 2 and scales_inv.ndim == 2

    T, H = tensor.shape
    assert T % BLOCK_SIZE == 0, f"T={T} not divisible by {BLOCK_SIZE}"
    assert H % BLOCK_SIZE == 0, f"H={H} not divisible by {BLOCK_SIZE}"

    tensor = _maybe_float8(tensor).contiguous()

    # Column-major scale layout for coalesced row-wise loads:
    # stride(0)=1 so consecutive rows sit adjacent in memory.
    if scales_inv.stride(0) != 1:
        scales_inv = scales_inv.mT.contiguous().mT

    m_idx = m_indices.contiguous().to(torch.int32)

    Y = torch.empty(H * T, device=tensor.device, dtype=torch.float8_e4m3fn)
    S = torch.empty(T // BLOCK_SIZE, H, device=tensor.device, dtype=torch.float32)

    _get_ext().row2col_dq(tensor, scales_inv, group_sizes, m_idx, Y, S)
    return Y, S
