"""Python wrapper for the scaling-aware FP8 transpose CUDA kernel.

Integer-domain replacement for the Triton reference
``blockwise_scaling_aware_fp8_transpose`` in
``llmfoundry/models/ops/float8/triton_kernels/row2col_dense_kernels.py``.

No FP8 decode/encode: operates entirely via bitwise exponent-shift on uint8
payload bytes.  Requires scale_inv entries to be power-of-2 floats (as
produced by any quantizer that calls ``pow2_ceil`` on the scale — the default
in this repo).

Public API
----------
``blockwise_scaling_aware_fp8_transpose(rowwise_data, rowwise_scale_inv,
                                        block_size=128)``
    Returns ``(columnwise_data, columnwise_scale_inv)`` — bit-for-bit
    identical to the Triton reference on all valid inputs.

Pre-build the extension to avoid the 30-60 s JIT compilation on first import::

    cd llmfoundry/models/ops/float8/cuda_kernels/scaling_aware_transpose
    pip install -e .
"""

from __future__ import annotations

import os
import typing as tp

import torch

# ---------------------------------------------------------------------------
# Extension loading  (pre-built → JIT fallback)
# ---------------------------------------------------------------------------

_ext = None
_ext_warmed_up = False


def warmup_ext() -> None:
    """Eagerly compile/load the CUDA extension and barrier-sync all ranks.

    Call once during model initialisation so the JIT compilation (30-60 s)
    does not block a single rank mid-training and cause distributed timeouts.
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

    # Try a pre-built wheel first (installed via pip install -e .).
    try:
        import scaling_aware_transpose_cuda_ext as _mod  # type: ignore[import]
        _ext = _mod
        return _ext
    except ImportError:
        pass

    # Fall back to JIT compilation via torch.utils.cpp_extension.load().
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
        f"scaling_aware_transpose_cuda_ext: JIT-compiling CUDA extension "
        f"({_arch}).  This takes 30-60 s on first import; "
        "subsequent imports use cache.",
        stacklevel=3,
    )

    _ext = _load(
        name="scaling_aware_transpose_cuda_ext",
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


def _maybe_uint8(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` viewed as uint8 (FP8 in-memory representation)."""
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise TypeError(
            f"rowwise_data must be uint8 or float8_e4m3fn, got {t.dtype}"
        )
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blockwise_scaling_aware_fp8_transpose(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """FP8-domain transpose: row-wise quantization → column-wise quantization.

    Bit-for-bit replacement for the Triton reference kernel
    ``blockwise_scaling_aware_fp8_transpose`` from row2col_dense_kernels.py.

    Parameters
    ----------
    rowwise_data : (rows, cols) float8_e4m3fn or uint8, CUDA
        FP8 payload in row-major order.  Arbitrary strides are supported;
        a vectorised fast path is used when ``stride(1) == 1``.
    rowwise_scale_inv : (rows, rsi_cols) float32, CUDA
        Row-wise scale_inv in compact-column layout: column ``c_block``
        covers ``rowwise_data[:, c_block*128 : (c_block+1)*128]``.
        Typically ``rsi_cols == cols // 128``.
        Entries must be exact powers of 2 for bit-exact results.
    block_size : int
        Tile size.  Must be 128 (the only value this kernel supports).

    Returns
    -------
    columnwise_data : (cols, rows) uint8 (or float8_e4m3fn if input was)
        Transposed FP8 payload with exponent-shifted bytes.
    columnwise_scale_inv : (nbrows_mult_4, cols) float32
        Column-wise scale_inv.  For every tile (pid_row, pid_col):
        ``scale[pid_row, pid_col*128 : (pid_col+1)*128]`` is uniformly set to
        ``max(rowwise_scale_inv[pid_row*128 : (pid_row+1)*128, pid_col])``.
        The first dimension is padded to the next multiple of 4 (required by
        downstream GEMM kernels that use 4-element scale vectorisation).

    Raises
    ------
    ValueError  if block_size != 128.
    AssertionError  if rows or cols are not multiples of 128.

    Notes
    -----
    Approximation: only the IEEE exponent of scale_inv is used for the
    shift; the mantissa is discarded.  Results are exact when scale_inv
    entries are exact powers of 2, which is guaranteed by the upstream
    ``pow2_ceil`` quantizer used throughout this repo.
    """
    if block_size != BLOCK_SIZE:
        raise ValueError(
            f"block_size must be {BLOCK_SIZE}, got {block_size}. "
            "Use the Triton kernel for non-128 block sizes."
        )

    assert rowwise_data.is_cuda, "rowwise_data must be a CUDA tensor"
    assert rowwise_scale_inv.is_cuda, "rowwise_scale_inv must be a CUDA tensor"
    assert rowwise_data.ndim == 2, \
        f"rowwise_data must be 2-D, got shape {rowwise_data.shape}"
    assert rowwise_scale_inv.ndim == 2, \
        f"rowwise_scale_inv must be 2-D, got shape {rowwise_scale_inv.shape}"

    rows, cols = rowwise_data.shape
    rsi_cols   = rowwise_scale_inv.shape[1]

    assert rows % BLOCK_SIZE == 0, \
        f"rows={rows} must be divisible by block_size={BLOCK_SIZE}"
    assert cols % BLOCK_SIZE == 0, \
        f"cols={cols} must be divisible by block_size={BLOCK_SIZE}"

    # Normalise dtype to uint8; remember original for output view.
    data_dtype = rowwise_data.dtype
    X = _maybe_uint8(rowwise_data)

    # scale_inv must be float32; upcast silently if needed.
    S = rowwise_scale_inv
    if S.dtype != torch.float32:
        S = S.to(torch.float32)

    device = X.device

    # Allocate outputs.
    nbrows        = rows // BLOCK_SIZE
    nbrows_mult_4 = (nbrows + 3) // 4 * 4

    # columnwise_data is [cols, rows] contiguous; kernel writes stride=(rows,1).
    Y     = torch.empty((cols, rows), dtype=torch.uint8, device=device)
    # columnwise_scale_inv is [nbrows_mult_4, cols] contiguous.
    S_out = torch.empty((nbrows_mult_4, cols), dtype=torch.float32, device=device)

    _get_ext().scaling_aware_fp8_transpose(X, S, Y, S_out, rows, cols, rsi_cols)

    # Return columnwise_data with the caller's original dtype.
    columnwise_data: torch.Tensor = (
        Y.view(data_dtype) if data_dtype != torch.uint8 else Y
    )

    return columnwise_data, S_out
