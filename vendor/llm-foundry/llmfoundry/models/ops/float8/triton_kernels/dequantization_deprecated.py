import torch

import triton
import triton.language as tl


@triton.jit
def _dequantize_kernel(
    # q
    q_ptr,
    stride_q_sl,
    stride_q_hs,
    size_q_sl,
    size_q_hs,
    # scale
    scale_ptr,
    stride_scale_sl,
    stride_scale_hs,
    size_scale_sl,
    size_scale_hs,
    # out
    out_ptr,
    stride_out_sl,
    stride_out_hs,
    size_out_sl,
    size_out_hs,
    # block
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    row_offset = row_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offset_q = col_block_id * BLOCK_SIZE * stride_q_hs + tl.arange(0, BLOCK_SIZE)
    q_block_ptr = q_ptr + row_offset[:, None] * stride_q_sl + col_offset_q[None, :]
    mask = (row_offset[:, None] < size_q_sl) & (col_offset_q[None, :] < size_q_hs)
    q_block = tl.load(q_block_ptr, mask=mask, other=0.0)

    scale_ptr = scale_ptr + row_offset[:, None] * stride_scale_sl + col_block_id * stride_scale_hs
    mask = (row_offset[:, None] < size_scale_sl) & (col_block_id < size_scale_hs)
    scale = tl.load(scale_ptr, mask=mask, other=0.0)

    col_offset_out = col_block_id * BLOCK_SIZE * stride_out_hs + tl.arange(0, BLOCK_SIZE)
    out_block_ptr = out_ptr + row_offset[:, None] * stride_out_sl + col_offset_out[None, :]
    q_block = q_block.to(tl.float32) * scale.to(tl.float32)
    mask = (row_offset[:, None] < size_out_sl) & (col_offset_out[None, :] < size_out_hs)
    tl.store(out_block_ptr, q_block.to(out_ptr.dtype.element_ty), mask=mask)


def dequantize_kernel(tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    assert tensor.is_cuda and scales.is_cuda
    assert tensor.ndim == 2 and scales.ndim == 2
    assert tensor.size(-1) % 128 == 0
    seq_len, hs = tensor.shape
    block_size_quant = 128
    blocks_per_row = hs // block_size_quant
    assert scales.shape == (seq_len, blocks_per_row)

    out = torch.empty((seq_len, hs), dtype=torch.bfloat16, device=tensor.device)

    n_blocks_per_row = (seq_len + block_size_quant - 1) // block_size_quant
    n_blocks_per_col = (hs + block_size_quant - 1) // block_size_quant

    tensor = tensor.view(torch.float8_e4m3fn)

    args = [
        tensor, *tensor.stride(), *tensor.size(),
        scales, *scales.stride(), *scales.size(),
        out, *out.stride(), *out.size(),
    ]

    grid = (n_blocks_per_row, n_blocks_per_col)
    _dequantize_kernel[grid](
        *args,
        BLOCK_SIZE=block_size_quant,
        num_warps=4,
        num_stages=3,
    )
    return out
