import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs


_ROWQ_BUCKET_SIZE = 2048

STATIC_TRITON_PARAMS = {
    7168: {
        "num_warps": 8,
        "maxnreg": 128
    },
    4096: {
        "num_warps": 8,
        "maxnreg": 192
    },
    3072: {
        "num_warps": 8,
        "maxnreg": 64
    },
    2560: {
        "num_warps": 8,
        "maxnreg": 64
    },
    2048: {
        "num_warps": 8,
        "maxnreg": 64
    },
    1536: {
        "num_warps": 8,
        "maxnreg": 64
    },
    1280: {
        "num_warps": 8,
        "maxnreg": 64
    },
}

@triton.jit
def _rowwise_1d_quantize_kernel(
    # tensor
    tensor_ptr,
    tensor_stride_row, tensor_stride_col,
    tensor_size_row, tensor_size_col,
    # out
    out_ptr,
    out_stride_row, out_stride_col,
    out_size_row, out_size_col,
    # scale
    scale_ptr,
    scale_stride_row, scale_stride_col,
    scale_size_row, scale_size_col,
    # Bucketed token count for the autotune key: total_tokens // _TOKEN_BUCKET_SIZE.
    # Not used in the kernel body; only drives autotuner config selection.
    total_token_bucket,
    # constexpr — injected by the autotuner from the winning Config
    BLOCK_SIZE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    offset_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_row = offset_row.to(tl.int64)
    offset_col = pid_hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = offset_col.to(tl.int64)

    tensor_ptr = tensor_ptr + offset_row[:, None] * tensor_stride_row + offset_col[None, :] * tensor_stride_col
    mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=mask)
    tensor_block = tensor_block.to(tl.float32)

    scale = tl.maximum(tl.max(tl.abs(tensor_block), 1) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    scale_inv = 1 / scale
    tensor_block = tensor_block * scale_inv[:, None]

    # save out
    out_ptr = out_ptr + offset_row[:, None] * out_stride_row + offset_col[None, :] * out_stride_col
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=mask)

    # save scale inv
    scale = scale[:, None]
    scale_ptr = scale_ptr + offset_row[:, None] * scale_stride_row + pid_hid * scale_stride_col
    scale_mask = (offset_row[:, None] < scale_size_row) & (pid_hid < scale_size_col)
    tl.store(scale_ptr, scale.to(scale_ptr.dtype.element_ty), mask=scale_mask)


def row_quantization_1x128_fn(tensor: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    total_tokens, hid_dim = tensor.shape
    assert tensor.is_contiguous()
    assert hid_dim % 128 == 0

    BLOCK_SIZE: int = 128

    if total_tokens % 128 == 0:
        out = torch.empty(
            (total_tokens, hid_dim),
            device=tensor.device,
            dtype=torch.float8_e4m3fn)
        scale_inv = torch.empty(
            (total_tokens, hid_dim // BLOCK_SIZE),
            device=tensor.device,
            dtype=torch.float32)
    else:
        out = torch.zeros(
            (total_tokens, hid_dim),
            device=tensor.device,
            dtype=torch.float8_e4m3fn)
        scale_inv = torch.zeros(
            (total_tokens, hid_dim // BLOCK_SIZE),
            device=tensor.device,
            dtype=torch.float32)

    grid =  (triton.cdiv(total_tokens, BLOCK_SIZE), triton.cdiv(hid_dim, BLOCK_SIZE))
    total_token_bucket = total_tokens // _ROWQ_BUCKET_SIZE
    _rowwise_1d_quantize_kernel[grid](
        *tensor_to_kernel_args(tensor, 2),
        *tensor_to_kernel_args(out, 2),
        *tensor_to_kernel_args(scale_inv, 2),
        total_token_bucket,
        BLOCK_SIZE=BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(STATIC_TRITON_PARAMS, hid_dim,
                                  num_warps=8, maxnreg=128)
    )
    return out, scale_inv
