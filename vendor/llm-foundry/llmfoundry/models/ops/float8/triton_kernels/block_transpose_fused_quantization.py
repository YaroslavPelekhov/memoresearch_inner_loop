import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs


_BLOCK_SIZE: int = 128

STATIC_TRITON_PARAMS = {
    7168: {
        "num_warps": 4,
        "maxnreg": 192,
    },
    4096: {
        "num_warps": 4,
        "maxnreg": 192
    },
    2048: {
        "num_warps": 4,
        "maxnreg": 192,
    },
    1536: {
        "num_warps": 16,
        "maxnreg": 64,
    },
    1280: {
        "num_warps": 16,
        "maxnreg": 255
    },
}

@triton.jit
def _block_quantize_kernel_transpose(
    # tensor (weight flattened as seq_len, hid_size)
    tensor_ptr,
    tensor_stride_row, tensor_stride_col,
    tensor_size_row, tensor_size_col,
    # out (num_groups, hid_size, int_dim)
    out_ptr,
    out_stride_g, out_stride_row, out_stride_col,
    out_size_row, out_size_col,
    # scale (num_groups, hid_blocks, seq_blocks_per_group)
    scale_ptr,
    scale_stride_g, scale_stride_row, scale_stride_col,
    scale_size_row, scale_size_col,
    # constexpr
    BLOCK_SIZE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
    int_dim: tl.constexpr,
):
    pid_group = tl.program_id(axis=0)
    pid_token = tl.program_id(axis=1)
    pid_hid = tl.program_id(axis=2)

    # Row in weight for this group + block: global row = group * int_dim + token_block * BLOCK_SIZE
    offset_row = pid_group * int_dim + pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = pid_hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_ptr = tensor_ptr + offset_row[:, None] * tensor_stride_row + offset_col[None, :] * tensor_stride_col
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask, other=0.0)
    tensor_block = tensor_block.to(tl.float32)

    scale = tl.maximum(tl.max(tl.abs(tensor_block)) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    scale_inv = 1.0 / scale
    tensor_block = tensor_block * scale_inv
    tensor_block = tl.trans(tensor_block)

    # save out: output[pid_group, offset_col, pid_token*BLOCK_SIZE + arange]
    out_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_ptr = (
        out_ptr
        + pid_group * out_stride_g
        + offset_col[:, None] * out_stride_row
        + out_row[None, :] * out_stride_col
    )
    out_mask = (offset_col[:, None] < out_size_row) & (out_row[None, :] < out_size_col)
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=out_mask)

    # save scale inv: scale[pid_group, pid_hid, pid_token]
    scale_ptr = (
        scale_ptr
        + pid_group * scale_stride_g
        + pid_hid * scale_stride_row
        + pid_token * scale_stride_col
    )
    scale_mask = (pid_hid < scale_size_row) & (pid_token < scale_size_col)
    tl.store(scale_ptr, scale.to(scale_ptr.dtype.element_ty), mask=scale_mask)


def block_transpose_fused_quantization_128x128_fn(
    weights: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    global _BLOCK_SIZE

    num_groups, int_dim, hid_dim = weights.shape

    weights = weights.reshape(-1, hid_dim)

    assert weights.is_contiguous()
    assert int_dim % 128 == 0 and hid_dim % 128 == 0

    weights_quantized = torch.empty(
        (num_groups, hid_dim, int_dim),
        device=weights.device,
        dtype=torch.float8_e4m3fn,
    )

    seq_blocks_per_group = int_dim // _BLOCK_SIZE
    hid_blocks = hid_dim // _BLOCK_SIZE
    scale_inv = torch.empty(
        (num_groups, hid_blocks, seq_blocks_per_group),
        device=weights.device,
        dtype=torch.float32,
    )

    grid = (num_groups, seq_blocks_per_group, hid_blocks)

    _block_quantize_kernel_transpose[grid](
        *tensor_to_kernel_args(weights, 2),
        weights_quantized, *weights_quantized.stride(), hid_dim, int_dim,
        scale_inv, *scale_inv.stride(), hid_blocks, seq_blocks_per_group,
        int_dim=int_dim,
        BLOCK_SIZE=_BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(STATIC_TRITON_PARAMS, hid_dim,
                                  num_warps=4, maxnreg=192)
    )
    return weights_quantized, scale_inv
