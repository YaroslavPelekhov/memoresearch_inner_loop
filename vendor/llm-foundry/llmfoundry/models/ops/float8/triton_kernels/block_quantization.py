import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs

_BLOCK_SIZE: int = 128

STATIC_TRITON_PARAMS = {
    7168: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    4096: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    2048: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    1536: {
        "num_warps": 8,
        "maxnreg": 128
    },
    1280: {
        "num_warps": 8,
        "maxnreg": 255
    },
}

@triton.jit
def _block_quantization_kernel(
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
    # constexpr
    BLOCK_SIZE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    offset_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

    # save out
    out_ptr = out_ptr + offset_row[:, None] * out_stride_row + offset_col[None, :] * out_stride_col
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=tensor_mask)

    # save scale inv
    scale_ptr = scale_ptr + pid_token * scale_stride_row + pid_hid * scale_stride_col
    scale_mask = (pid_token < scale_size_row) & (pid_hid < scale_size_col)
    tl.store(scale_ptr, scale.to(scale_ptr.dtype.element_ty), mask=scale_mask)


def block_quantization_128x128_fn(weight: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    global _BLOCK_SIZE

    weight_shape = weight.shape

    num_groups, int_dim, hid_size = weight.shape
    assert int_dim % 128 == 0 and hid_size % 128 == 0

    weight = weight.view(-1, hid_size)

    assert weight.is_contiguous()

    weight_quantized = torch.empty(
        (num_groups * int_dim, hid_size),
        device=weight.device,
        dtype=torch.float8_e4m3fn)

    scale_shape = (num_groups * int_dim // _BLOCK_SIZE, hid_size // _BLOCK_SIZE)
    scale_inv = torch.empty(scale_shape, device=weight.device, dtype=torch.float32)


    _block_quantization_kernel[scale_shape](
        *tensor_to_kernel_args(weight, 2),
        *tensor_to_kernel_args(weight_quantized, 2),
        *tensor_to_kernel_args(scale_inv, 2),
        BLOCK_SIZE=_BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(STATIC_TRITON_PARAMS, hid_size,
                                  num_warps=8, maxnreg=255)
    )
    weight_quantized = weight_quantized.reshape(weight_shape)
    scale_inv = scale_inv.reshape(num_groups, -1, scale_shape[-1])
    return weight_quantized, scale_inv
