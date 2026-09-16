import os
import typing as tp

import torch

import triton
import triton.language as tl

from llmfoundry.models.ops.fp8.triton_kernels.utils import tensor_to_kernel_args


if os.getenv("TRITON_AUTOTUNE_ENABLED", "0") == "1":
    CONFIGS: tp.List[triton.Config] = [
        triton.Config({"BLOCK_SIZE": 128, "POWER_TWO_MAX_ROUND": True}, num_warps=4, maxnreg=128),
        triton.Config({"BLOCK_SIZE": 128, "POWER_TWO_MAX_ROUND": True}, num_warps=8, maxnreg=128),
        triton.Config({"BLOCK_SIZE": 128, "POWER_TWO_MAX_ROUND": True}, num_warps=8, maxnreg=192),
    ]
    autotune_fn = triton.autotune(
        configs=CONFIGS,
        key=["total_token_bucket", "tensor_size_col"],
        cache_results=True,
    )
else:
    def null_autotune_fn(fn: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
        return fn

    autotune_fn = null_autotune_fn

_ROW2COL_BUCKET_SIZE = 2048

@triton.jit
def _rowwise_to_columnwise_kernel(
        # input tensor (rowwise, seq_len x hid_dim)
        tensor_ptr,
        tensor_stride_row, tensor_stride_col,
        tensor_size_row, tensor_size_col,
        # rowwise scale_inv for dequant (seq_len x hid_dim//128); dequant = cast(x_fp8) * scale_inv (TE convention)
        scale_inv_ptr,
        scale_inv_stride_row, scale_inv_stride_col,
        scale_inv_size_row, scale_inv_size_col,
        # output tensor (columnwise, hid_dim x seq_len)
        out_ptr,
        out_stride_row, out_stride_col,
        out_size_row, out_size_col,
        # output columnwise scale_inv (seq_len//128 x hid_dim)
        scale_col_ptr,
        scale_col_stride_row, scale_col_stride_col,
        scale_col_size_row, scale_col_size_col,
        # key for autotune
        total_token_bucket,
        # block
        BLOCK_SIZE: tl.constexpr,
        POWER_TWO_MAX_ROUND: tl.constexpr):

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = (pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    offset_col = (pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    # load tensor block (seq_len block x hid_dim block)
    tensor_ptr = tensor_ptr + offset_row[:, None] * tensor_stride_row + offset_col[None, :] * tensor_stride_col
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask)

    # load rowwise scale_inv; dequantize: data_f32 = cast(x_fp8) * scale_inv
    scale_inv_ptr = scale_inv_ptr + offset_row[:, None] * scale_inv_stride_row + pid_col * scale_inv_stride_col
    scale_inv_mask = (offset_row[:, None] < scale_inv_size_row) & (pid_col < scale_inv_size_col)
    scale_inv = tl.load(scale_inv_ptr, mask=scale_inv_mask, other=1.0)

    # dequantize
    tensor_block = tensor_block.to(tl.float32) * scale_inv.to(tl.float32)

    # columnwise quantization
    scale_col = tl.maximum(tl.max(tl.abs(tensor_block), 0) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale_col = tl.exp2(tl.ceil(tl.log2(scale_col)))
    scale_inv_col = 1.0 / scale_col
    tensor_block = tensor_block * scale_inv_col[None, :]

    out_offset_row = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # runs over c
    out_offset_col = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # runs over r
    out_ptr = out_ptr + out_offset_row[None, :] * out_stride_row + out_offset_col[:, None] * out_stride_col
    out_mask = (out_offset_row[None, :] < out_size_row) & (out_offset_col[:, None] < out_size_col)
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=out_mask)

    scale_col_ptr = scale_col_ptr + pid_row * scale_col_stride_row + offset_col[None, :] * scale_col_stride_col
    scale_col_mask = (pid_row < scale_col_size_row) & (offset_col[None, :] < scale_col_size_col)
    tl.store(scale_col_ptr, scale_col[None, :].to(scale_col_ptr.dtype.element_ty), mask=scale_col_mask)


def maybe_float8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    assert tensor.dtype == torch.float8_e4m3fn
    return tensor


def row2col(
    tensor: torch.Tensor,
    scales_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    BLOCK_SIZE: int = 128

    assert tensor.is_cuda and scales_inv.is_cuda
    assert tensor.ndim == 2 and scales_inv.ndim == 2

    seq_len, hidden_size = tensor.shape
    assert hidden_size % BLOCK_SIZE == 0
    assert seq_len % BLOCK_SIZE == 0

    blocks_per_row = hidden_size // BLOCK_SIZE
    assert scales_inv.shape == (seq_len, blocks_per_row), (
        f"{scales_inv.shape} != {(seq_len, blocks_per_row)}"
    )

    tensor_dtype_input = tensor.dtype
    tensor = maybe_float8(tensor)

    columnwise_data = torch.empty(
        hidden_size, seq_len,
        device=tensor.device,
        dtype=torch.float8_e4m3fn,
    )
    columnwise_scale_inv = torch.empty(
        seq_len // BLOCK_SIZE, hidden_size,
        device=tensor.device,
        dtype=torch.float32,
    )

    total_token_bucket = seq_len // _ROW2COL_BUCKET_SIZE

    grid = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE, (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _rowwise_to_columnwise_kernel[grid](
        *tensor_to_kernel_args(tensor, 2),
        *tensor_to_kernel_args(scales_inv, 2),
        *tensor_to_kernel_args(columnwise_data, 2),
        *tensor_to_kernel_args(columnwise_scale_inv, 2),
        total_token_bucket
    )

    tensor = tensor.view(tensor_dtype_input)
    return columnwise_data, columnwise_scale_inv
