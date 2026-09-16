import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
)
import transformer_engine_torch as tex


_HAS_TE = True

BLOCK_SIZE: int = 128
_ROW2COL_BUCKET_SIZE: int = 2048
MAX_NUM_GROUPS: int = 128

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
def _rowwise_to_grouped_columnwise_kernel(
        # input FP8 tensor: (total_tokens, hidden_size), rowwise layout
        tensor_ptr,
        tensor_stride_row, tensor_stride_col,
        tensor_size_row, tensor_size_col,
        # rowwise scale_inv: logical (total_tokens, hidden_size // 128)
        # accessed via strides, so column-major layout from .mT.contiguous().mT is fine
        scale_inv_ptr,
        scale_inv_stride_row, scale_inv_stride_col,
        scale_inv_size_row, scale_inv_size_col,
        # output: flat 1D FP8 buffer, groups packed contiguously
        # group g occupies [offset_g .. offset_g + hs*gs) stored as (hs, gs) row-major
        out_ptr,
        out_stride_el,
        out_size_el,
        # output: columnwise scale_inv (total_tokens // 128, hidden_size)
        scale_col_ptr,
        scale_col_stride_row, scale_col_stride_col,
        scale_col_size_row, scale_col_size_col,
        # m_indices: (total_tokens,) int32, maps each token to its group index
        m_indices_ptr,
        # group_sizes: (num_groups,) padded group size per expert (multiple of 128)
        group_sizes_ptr,
        # autotune key
        total_token_bucket,
        # constexpr
        BLOCK_SIZE: tl.constexpr,
        POWER_TWO_MAX_ROUND: tl.constexpr,
        MAX_NUM_GROUPS: tl.constexpr,
):
    pid_row = tl.program_id(0)  # token-block index
    pid_col = tl.program_id(1)  # hidden-dim-block index

    row_off = (pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    col_off = (pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    # ---- load FP8 input tile [BLOCK_SIZE x BLOCK_SIZE] ----
    in_ptrs = (
        tensor_ptr
        + row_off[:, None] * tensor_stride_row
        + col_off[None, :] * tensor_stride_col)
    in_mask = (row_off[:, None] < tensor_size_row) & (col_off[None, :] < tensor_size_col)
    block = tl.load(in_ptrs, mask=in_mask, other=0.0)

    # ---- load rowwise scale_inv (one per 128-element hidden chunk per token row) ----
    sc_ptrs = (
        scale_inv_ptr
        + row_off[:, None] * scale_inv_stride_row
        + pid_col * scale_inv_stride_col)
    sc_mask = (row_off[:, None] < scale_inv_size_row) & (pid_col < scale_inv_size_col)
    row_sc = tl.load(sc_ptrs, mask=sc_mask, other=1.0)

    # ---- dequantize: f32 = fp8 * scale_inv ----
    block = block.to(tl.float32) * row_sc.to(tl.float32)

    # ---- columnwise quantization ----
    # reduce over axis 0 (tokens) -> one scale per hidden element
    col_sc = tl.maximum(tl.max(tl.abs(block), 0) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        col_sc = tl.exp2(tl.ceil(tl.log2(col_sc)))
    block = block * (1.0 / col_sc)[None, :]

    # ---- transpose: [tokens, hidden] -> [hidden, tokens] ----
    block = tl.trans(block)
    in_mask = tl.trans(in_mask)

    # ---- determine group, compute flat-buffer offset ----
    tok_start = (pid_row * BLOCK_SIZE).to(tl.int64)
    gid = tl.load(m_indices_ptr + tok_start).to(tl.int64)

    g_range = tl.arange(0, MAX_NUM_GROUPS)
    g_mask = g_range < gid
    g_sizes = tl.load(group_sizes_ptr + g_range, mask=g_mask, other=0).to(tl.int64)
    g_offset = tl.sum(g_sizes, axis=0).to(tl.int64)
    g_size = tl.load(group_sizes_ptr + gid).to(tl.int64)

    # flat index = group_start_flat + h * group_size + t_within_group
    flat_base = g_offset * tensor_size_col
    out_h = (pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    out_t = (tok_start - g_offset + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    out_ptrs = (
        out_ptr
        + (flat_base + out_h[:, None] * g_size + out_t[None, :]) * out_stride_el)
    out_mask = in_mask & (out_t[None, :] < g_size)
    tl.store(out_ptrs, block.to(out_ptr.dtype.element_ty), mask=out_mask)

    # ---- write columnwise scale_inv at [pid_row, col_off] ----
    sc_out_ptrs = (
        scale_col_ptr
        + pid_row * scale_col_stride_row
        + col_off[None, :] * scale_col_stride_col)
    sc_out_mask = (pid_row < scale_col_size_row) & (col_off[None, :] < scale_col_size_col)
    tl.store(sc_out_ptrs, col_sc[None, :].to(scale_col_ptr.dtype.element_ty), mask=sc_out_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _maybe_float8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    assert tensor.dtype == torch.float8_e4m3fn, (
        f"Expected float8_e4m3fn, got {tensor.dtype}")
    return tensor


def row2col_grouped_raw(
    tensor: torch.Tensor,
    scales_inv: torch.Tensor,
    list_groups_sizes: tp.List[int], # is not used in the kernel (remove this??)
    m_indices: torch.Tensor,
    group_sizes_tensor: torch.Tensor,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Low-level fused row->col conversion; returns raw flat buffers.

    Returns
    -------
    col_data_flat : torch.Tensor
        1-D FP8 buffer of length ``hidden_size * total_tokens``.
        Groups are packed contiguously; group *g* occupies
        ``[offset_g .. offset_g + hs * gs)`` stored as ``(hs, gs)`` row-major
        (= columnwise layout of the original ``(gs, hs)`` matrix).
    col_scale_inv : torch.Tensor
        Shape ``(total_tokens // 128, hidden_size)`` float32.
        Groups are contiguous along the row dimension.
    """
    BLOCK_SIZE = 128

    assert tensor.is_cuda and scales_inv.is_cuda
    assert tensor.ndim == 2 and scales_inv.ndim == 2

    total_tokens, hidden_size = tensor.shape
    assert hidden_size % BLOCK_SIZE == 0, (
        f"hidden_size={hidden_size} not divisible by {BLOCK_SIZE}")
    assert total_tokens % BLOCK_SIZE == 0, (
        f"total_tokens={total_tokens} not divisible by {BLOCK_SIZE}")
    assert sum(list_groups_sizes) == total_tokens, (
        f"sum(groups)={sum(list_groups_sizes)} != total_tokens={total_tokens}")
    assert len(list_groups_sizes) <= 128, (
        "MAX_NUM_GROUPS constexpr is 128; increase if more experts needed")

    tensor = _maybe_float8(tensor)

    # Ensure column-major layout for coalesced row-wise scale loads in the kernel:
    # row-major stride_row = hidden_size//128, giving ~256-byte gaps between rows;
    # column-major stride_row = 1, so consecutive rows sit adjacent in memory.
    if scales_inv.stride(0) != 1:
        scales_inv = scales_inv.mT.contiguous().mT

    col_data_flat = torch.empty(
        (hidden_size * total_tokens,),
        device=tensor.device,
        dtype=torch.float8_e4m3fn,
    )
    col_scale_inv = torch.empty(
        (total_tokens // BLOCK_SIZE, hidden_size),
        device=tensor.device,
        dtype=torch.float32,
    )

    total_token_bucket = total_tokens // _ROW2COL_BUCKET_SIZE

    grid = (total_tokens // BLOCK_SIZE, hidden_size // BLOCK_SIZE)

    _rowwise_to_grouped_columnwise_kernel[grid](
        *tensor_to_kernel_args(tensor, 2),
        *tensor_to_kernel_args(scales_inv, 2),
        *tensor_to_kernel_args(col_data_flat, 1),
        *tensor_to_kernel_args(col_scale_inv, 2),
        m_indices,
        group_sizes_tensor,
        total_token_bucket,
        BLOCK_SIZE=BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        MAX_NUM_GROUPS=MAX_NUM_GROUPS,
        **params_to_kernel_kwargs(STATIC_TRITON_PARAMS, hidden_size,
                                  num_warps=8, maxnreg=128)
    )
    return col_data_flat, col_scale_inv


# TODO (fedorovgv): write tests here
def row2col_grouped(
    tensor: torch.Tensor,
    scales_inv: torch.Tensor,
    list_groups_sizes: tp.List[int],
    m_indices: torch.Tensor,
    group_sizes_tensor: torch.Tensor,
) -> tp.Tuple:
    """Fused rowwise FP8 -> grouped columnwise Float8BlockwiseQTensor.

    Drop-in replacement for::

        dequantized = dequantize_kernel(tensor, scales)
        result = tex.split_quantize(dequantized, group_sizes, quantizers)

    Returns a tuple of ``Float8BlockwiseQTensor`` (one per group),
    directly compatible with ``general_grouped_gemm``.

    Parameters
    ----------
    tensor : (total_tokens, hidden_size)  float8_e4m3fn / uint8
    scales_inv : (total_tokens, hidden_size // 128)  float32
        Rowwise scale_inv. Strides are used, so column-major layout is fine.
    list_groups_sizes : list[int]
        Padded group sizes (each a multiple of 128).
    m_indices : (total_tokens,) int32
        Maps each token position to its group index.
    group_sizes_tensor : (num_groups,) int32/int64
        Padded group sizes on device.
    """
    assert _HAS_TE, "transformer_engine is required for row2col_grouped"
    BLOCK_SIZE = 128

    _, hidden_size = tensor.shape

    col_data_flat, col_scale_inv = row2col_grouped_raw(
        tensor, scales_inv, list_groups_sizes, m_indices, group_sizes_tensor,
    )

    result: tp.List[Float8BlockwiseQTensor] = []
    data_offset = 0
    scale_row = 0

    for gs in list_groups_sizes:
        group_col_data = (
            col_data_flat[data_offset : data_offset + hidden_size * gs]
            .view(hidden_size, gs)
        )
        group_col_scale = col_scale_inv[scale_row : scale_row + gs // BLOCK_SIZE]

        result.append(Float8BlockwiseQTensor(
            shape=(gs, hidden_size),
            dtype=torch.bfloat16,
            columnwise_data=group_col_data,
            columnwise_scale_inv=group_col_scale,
            rowwise_data=None,
            rowwise_scale_inv=None,
            is_2D_scaled=False,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=None,
        ))

        data_offset += hidden_size * gs
        scale_row += gs // BLOCK_SIZE

    return tuple(result)
