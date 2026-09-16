"""CUDA kernel wrappers for FP8 ops.

Currently exposes:

    blockwise_scaling_aware_fp8_transpose(rowwise_data, rowwise_scale_inv, block_size)
        -> (columnwise_data, columnwise_scale_inv)
"""

from __future__ import annotations

import typing as tp

import torch


def blockwise_scaling_aware_fp8_transpose(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    block_size: int = 128,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """CUDA replacement for ``blockwise_scaling_aware_fp8_transpose`` from
    ``triton_kernels/row2col_dense_kernels.py``.

    Converts a row-wise quantized FP8 tensor to column-wise layout using an
    integer exponent-shift approximation — no FP8 decode/encode.
    Returns ``(columnwise_data, columnwise_scale_inv)`` bit-for-bit identical
    to the Triton reference on all valid inputs.

    See ``cuda_kernels/scaling_aware_transpose/__init__.py`` for full docs.
    """
    # Lazy import: extension compiled only on first call.
    from llmfoundry.models.ops.float8.cuda_kernels.scaling_aware_transpose import (
        blockwise_scaling_aware_fp8_transpose as _fn,
    )
    return _fn(rowwise_data, rowwise_scale_inv, block_size)
