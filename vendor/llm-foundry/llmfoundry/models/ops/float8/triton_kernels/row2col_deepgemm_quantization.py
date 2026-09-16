import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import (
    maybe_convert_to_float8_e4m3fn,
    tensor_to_kernel_args,
)
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs


BLOCK_SIZE: int = 128
MAX_NUM_GROUPS: int = 128
_ROW2COL_BUCKET_SIZE: int = 2048

STATIC_TRITON_PARAMS = {
    7168: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    4096: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    3072: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    2560: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    2048: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    1536: {
        "num_warps": 8,
        "maxnreg": 128,
    },
    1280: {
        "num_warps": 8,
        "maxnreg": 128,
    }
}

@triton.jit
def _rowwise_to_columnwise_deepgemm_layout_kernel(
        # input tensor rowwise (seq_len x hid_dim)
        tensor_ptr,
        tensor_stride_row, tensor_stride_col,
        tensor_size_row, tensor_size_col,
        # rowwise scale_inv rowwise (seq_len x hid_dim // 128)
        scale_inv_ptr,
        scale_inv_stride_row, scale_inv_stride_col,
        scale_inv_size_row, scale_inv_size_col,
        # output tensor columnwise (hid_dim x seq_len, )
        out_ptr,
        # out_stride_row, out_stride_col,
        out_stride_row,
        # out_size_row, out_size_col,
        out_size_row,
        # output columnwise scale_inv (seq_len//128 x hid_dim)
        scale_col_ptr,
        scale_col_stride_row, scale_col_stride_col,
        scale_col_size_row, scale_col_size_col,
        # m_indices (seq_len, )
        m_indices_ptr,
        # group_sizes (seq_len, )
        group_sizes_ptr,
        # key for autotune
        total_token_bucket,
        # block
        BLOCK_SIZE: tl.constexpr,
        POWER_TWO_MAX_ROUND: tl.constexpr,
        MAX_NUM_GROUPS: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_row = offset_row.to(tl.int64)
    offset_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = offset_col.to(tl.int64)

    # load tensor block (seq_len block x hid_dim block)
    tensor_ptr = (
        tensor_ptr
        + offset_row[:, None] * tensor_stride_row
        + offset_col[None, :] * tensor_stride_col)
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask, other=0.0)

    # load rowwise scale_inv; dequantize: data_f32 = cast(x_fp8) * scale_inv
    scale_inv_ptr = (
        scale_inv_ptr
        + offset_row[:, None] * scale_inv_stride_row
        + pid_col * scale_inv_stride_col)
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

    tensor_block = tl.trans(tensor_block)
    tensor_mask  = tl.trans(tensor_mask)

    # block offset in
    tensor_offset_row = pid_row * BLOCK_SIZE
    index_of_current_group = tl.load(m_indices_ptr + tensor_offset_row).to(tl.int64)

    # сalculate offset
    group_sizes_mask = tl.arange(0, MAX_NUM_GROUPS) < index_of_current_group
    group_sizes = tl.load(
        group_sizes_ptr + tl.arange(0, MAX_NUM_GROUPS),
        mask=group_sizes_mask,
        other=0,
    ).to(tl.int64)
    current_group_offset = tl.sum(group_sizes, axis=0).to(tl.int64)
    current_group_size = tl.load(group_sizes_ptr + index_of_current_group)

    # move pointer to the ossfet of current group
    out_ptr = out_ptr + (current_group_offset * tensor_size_col) * out_stride_row

    out_offset_row = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offset_row = out_offset_row.to(tl.int64)
    out_offset_col = (tensor_offset_row - current_group_offset) + tl.arange(0, BLOCK_SIZE)
    out_offset_col = out_offset_col.to(tl.int64)

    out_ptrs = (
        out_ptr
        + (out_offset_row[:, None] * current_group_size
        + out_offset_col[None, :]) * out_stride_row)
    out_mask = tensor_mask & (out_offset_col[None, :] < current_group_size)
    tl.store(out_ptrs, tensor_block.to(out_ptr.dtype.element_ty), mask=out_mask)

    scale_col_ptr = (
        scale_col_ptr
        + pid_row * scale_col_stride_row
        + offset_col[None, :] * scale_col_stride_col)
    scale_col_mask = (pid_row < scale_col_size_row) & (offset_col[None, :] < scale_col_size_col)
    tl.store(scale_col_ptr, scale_col[None, :].to(scale_col_ptr.dtype.element_ty), mask=scale_col_mask)


def row2col_requantization_deepgemm_layout_fn(
    tensor: torch.Tensor,
    m_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    scales_inv: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    global BLOCK_SIZE

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
    tensor = maybe_convert_to_float8_e4m3fn(tensor)

    # Ensure column-major layout for coalesced row-wise scale loads in the kernel:
    # row-major stride_row = hidden_size//128, giving ~256-byte gaps between rows;
    # column-major stride_row = 1, so consecutive rows sit adjacent in memory.
    if scales_inv.stride(0) != 1:
        scales_inv = scales_inv.mT.contiguous().mT

    columnwise_data = torch.empty(
        (hidden_size * seq_len,),
        device=tensor.device,
        dtype=torch.float8_e4m3fn)

    columnwise_scale_inv = torch.empty(
        seq_len // BLOCK_SIZE, hidden_size,
        device=tensor.device,
        dtype=torch.float32,)

    total_token_bucket = seq_len // _ROW2COL_BUCKET_SIZE

    grid = (triton.cdiv(seq_len, BLOCK_SIZE), triton.cdiv(hidden_size, BLOCK_SIZE))

    _rowwise_to_columnwise_deepgemm_layout_kernel[grid](
        *tensor_to_kernel_args(tensor, 2),
        *tensor_to_kernel_args(scales_inv, 2),
        *tensor_to_kernel_args(columnwise_data, 1),
        *tensor_to_kernel_args(columnwise_scale_inv, 2),
        m_indices,
        group_sizes,
        total_token_bucket,
        BLOCK_SIZE=BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        MAX_NUM_GROUPS=MAX_NUM_GROUPS,
        **params_to_kernel_kwargs(STATIC_TRITON_PARAMS, hidden_size,
                                  num_warps=8, maxnreg=128)
    )

    columnwise_data = columnwise_data.view(tensor_dtype_input)
    return columnwise_data, columnwise_scale_inv
