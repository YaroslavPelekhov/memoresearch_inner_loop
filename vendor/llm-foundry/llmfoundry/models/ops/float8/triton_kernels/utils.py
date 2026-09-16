import typing as tp

import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
import transformer_engine_torch as tex


def maybe_contiguous(tensor: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
    if tensor is not None and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def make_float8_blockwise_qtensor_fn(
    rowwise_data: tp.Optional[torch.Tensor] = None,
    rowwise_scale_inv: tp.Optional[torch.Tensor] = None,
    columnwise_data: tp.Optional[torch.Tensor] = None,
    columnwise_scale_inv: tp.Optional[torch.Tensor] = None,
    check_correctness_for_te_fmt: bool = False,
    is_2D_scaled: bool = False) -> Float8BlockwiseQTensor:
    """
    check_correctness_for_te_fmt: bool - flag, to check shapes for te correct working.
    """

    if check_correctness_for_te_fmt:
        assert not ((rowwise_data is not None) ^ (rowwise_scale_inv is not None))
        if rowwise_data is not None:
            seq_len_data, hid_dim_data = rowwise_data.shape
            hid_dim_sfs, seq_len_sfs = rowwise_scale_inv.shape
            assert seq_len_data == seq_len_sfs
            assert hid_dim_data // 128 == hid_dim_sfs
        assert not ((columnwise_data is not None) ^ (columnwise_scale_inv is not None))
        if columnwise_data is not None:
            hid_dim_data, seq_len_data = columnwise_data.shape
            seq_len_sfs, hid_dim_sfs = columnwise_scale_inv.shape
            assert seq_len_data == seq_len_sfs
            assert hid_dim_data // 128 == hid_dim_sfs
        assert not (rowwise_data is None and columnwise_data is None)
    else:
        assert not ((rowwise_data is not None) ^ (rowwise_scale_inv is not None))
        if rowwise_data is not None:
            seq_len_data, hid_dim_data = rowwise_data.shape
            seq_len_sfs, hid_dim_sfs = rowwise_scale_inv.shape
            assert seq_len_data == seq_len_sfs
            assert hid_dim_data // 128 == hid_dim_sfs
        assert not ((columnwise_data is not None) ^ (columnwise_scale_inv is not None))
        if columnwise_data is not None:
            seq_len_data, hid_dim_data = columnwise_data.shape
            seq_len_sfs, hid_dim_sfs = columnwise_scale_inv.shape
            assert seq_len_data == seq_len_sfs
            assert hid_dim_data // 128 == hid_dim_sfs
        assert not (rowwise_data is None and columnwise_data is None)

    data_shape = rowwise_data.shape if rowwise_data is not None else columnwise_data.shape

    rowwise_data = maybe_contiguous(rowwise_data)
    rowwise_scale_inv = maybe_contiguous(rowwise_scale_inv)

    columnwise_data = maybe_contiguous(columnwise_data)
    columnwise_scale_inv = maybe_contiguous(columnwise_scale_inv)

    tensor = Float8BlockwiseQTensor(
        shape=data_shape,
        dtype=torch.bfloat16,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=None,
        is_2D_scaled=is_2D_scaled,
    )
    return tensor


def maybe_convert_to_float8_e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    assert tensor.dtype == torch.float8_e4m3fn
    return tensor


def tensor_to_kernel_args(tensor: tp.Optional[torch.Tensor], dims: int):
    return (
        [tensor, *tensor.stride(), *tensor.size()]
        if tensor is not None
        else [None] * (1 + 2 * dims)
    )


@triton.jit
def _build_m_indices_kernel(
    # group_sizes
    group_sizes_ptr,
    group_sizes_stride_row,
    group_sizes_size_row,
    # output
    output_ptr,
    output_stride_row,
    output_size_row,
    # constants
    NUM_EXPERTS: tl.constexpr,
    ALIGN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_expert = tl.program_id(0).to(tl.int32)
    pid_block  = tl.program_id(1).to(tl.int32)

    start = tl.full((), 0, tl.int32)
    previously_seen = tl.full((), 0, tl.int32)

    for expert_num in tl.range(0, NUM_EXPERTS):
        expert_size = tl.load(group_sizes_ptr + expert_num * group_sizes_stride_row).to(tl.int32)
        expert_size_padded = ((expert_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE

        start += tl.where((previously_seen == 0) & (expert_num < pid_expert), expert_size_padded, 0)
        previously_seen = tl.where(expert_num == pid_expert, 1, previously_seen)

    group_size = tl.load(group_sizes_ptr + pid_expert * group_sizes_stride_row).to(tl.int32)
    group_size_padded = ((group_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE

    output_offset = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int32)
    output_mask = output_offset < group_size_padded

    output_values = tl.where(output_offset < group_size, pid_expert, -1).to(tl.int32)

    out_ptrs = output_ptr + (start + output_offset) * output_stride_row
    tl.store(out_ptrs, output_values, mask=output_mask)


def build_m_indices(
    group_sizes_list: tp.List[int],
    group_sizes_tensor: torch.Tensor,
    BLOCK_SIZE: int = 128,
):
    group_sizes_padded = [(m + 127) // 128 * 128 for m in group_sizes_list]
    total = sum(group_sizes_padded)
    m_indices = torch.empty((total,), device=group_sizes_tensor.device, dtype=torch.int32)

    grid = (len(group_sizes_list), triton.cdiv(max(group_sizes_padded), BLOCK_SIZE))

    _build_m_indices_kernel[grid](
        *tensor_to_kernel_args(group_sizes_tensor, dims=1),
        *tensor_to_kernel_args(m_indices, dims=1),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_EXPERTS=len(group_sizes_list),
        ALIGN_SIZE=128,)
    return m_indices


