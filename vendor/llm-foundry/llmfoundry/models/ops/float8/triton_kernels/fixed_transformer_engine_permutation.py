# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Permutation kernels written with OpenAI Triton."""

import warnings
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from triton.language import core
from triton.language.standard import _log2

__all__ = [
    "moe_permute",
    "moe_permute_with_probs",
    "moe_unpermute",
    "moe_sort_chunks_by_index",
    "moe_sort_chunks_by_index_with_probs",
]


# The following three argsort related kernels are adapted from
# the issue https://github.com/triton-lang/triton/issues/3698


@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]

    l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )
    r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )

    l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
    r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)

    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    il_value = l_value.to(idtype, bitcast=True)
    ir_value = r_value.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    flag1 = tl.where(((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix))
    ret = ix ^ flag1
    flag2 = tl.where(((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix))
    ind = indices ^ flag2

    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def _argsort(x, indices, n_dims: tl.constexpr):
    for i in tl.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(x, indices, i, 2 if i < n_dims else 1, n_dims)
    return x, indices


@triton.jit(do_not_specialize=["num_tokens"])
def _row_id_map_pass_1_kernel(
    # pointers
    routing_map_ptr,
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int32)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)


@triton.jit(do_not_specialize=["num_tokens"])
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        mask=(offset < num_tokens),
        other=0,
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id,
        mask=(offset < num_tokens),
    )


@triton.jit
def _row_id_map_pass_3_kernel(
    # pointers
    row_id_map_ptr,
    # sizes
    num_experts: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    LOAD_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dims: tl.constexpr = _log2(LOAD_SIZE)
    off = tl.arange(0, LOAD_SIZE)
    row_id_map = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + stride_row_id_map_expert * off,
        mask=off < num_experts,
        other=-1,
    )
    n_routed = tl.sum(tl.where(row_id_map != -1, 1, 0))
    indices = off
    sorted_map, indices = _argsort(row_id_map, indices, n_dims=n_dims)
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + off * stride_row_id_map_expert,
        sorted_map,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr
        + pid * stride_row_id_map_token
        + (num_experts + off) * stride_row_id_map_expert,
        indices,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert,
        n_routed,
    )


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    """
    Prepare the row_id_map for the permutation.

    Parameters
    ----------
    routing_map: torch.Tensor
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map: torch.Tensor
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    row_id_map = torch.empty((num_tokens, num_experts * 2 + 1), dtype=torch.int32, device="cuda")
    block_size = 1024
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int32, device="cuda")

    # supposing num_tokens == 5, num_experts == 3, block_size == 3
    # and we have a routing_map like this:
    # [[1, 1, 0],
    #  [1, 0, 1],
    #  [0, 0, 1],
    #  [1, 1, 0],
    #  [0, 0, 0]]

    # pass 1: block cumsum
    # for each expert, compute the cumsum of every block_size tokens
    # the row_id_map will be like this after pass 1 (r means useless values):
    # [[1, 1, 0, r, r, r, r],
    #  [2, 0, 1, r, r, r, r],
    #  [0, 0, 2, r, r, r, r],
    #  [1, 1, 0, r, r, r, r],
    #  [0, 0, 0, r, r, r, r]]
    _row_id_map_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        row_id_map.stride(0),
        row_id_map.stride(1),
        block_size,
    )

    # pass 2: cumsum all and process the mask
    # process the block cumsum into the global cumsum and then into the dst row indices
    # the row_id_map will be like this after pass 2 (r means useless value):
    # [[ 0,  3, -1, r, r, r, r],
    #  [ 1, -1,  5, r, r, r, r],
    #  [-1, -1,  6, r, r, r, r],
    #  [ 2,  4, -1, r, r, r, r],
    #  [-1, -1, -1, r, r, r, r]]
    _row_id_map_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        row_id_map.stride(0),
        row_id_map.stride(1),
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )

    # pass 3: make the row_id_map from the sparse structure to the dense structure
    # the row_id_map will be like this after pass 3 (r means useless value):
    # [[3, 0, r, 1, 0, r, 2],
    #  [5, 1, r, 2, 0, r, 2],
    #  [6, r, r, 2, r, r, 1],
    #  [4, 2, r, 1, 0, r, 2],
    #  [r, r, r, r, r, r, 0]]
    grid = (num_tokens,)
    _row_id_map_pass_3_kernel[grid](
        row_id_map,
        num_experts,
        row_id_map.stride(0),
        row_id_map.stride(1),
        triton.next_power_of_2(num_experts),
    )
    return row_id_map


@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    scale_ptr,
    permuted_probs_ptr,
    permuted_scale_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    scale_hidden_dim,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_scale_token,
    stride_scale_hidden,
    stride_permuted_probs_token,
    stride_permuted_scale_token,
    stride_permuted_scale_hidden,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    PERMUTE_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    cur_off = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cur_off < hidden_size
    input_off = pid_t * stride_input_token + cur_off * stride_input_hidden
    inp = tl.load(input_ptr + input_off, mask=mask)
    if PERMUTE_SCALE:
        mask_scale = cur_off < scale_hidden_dim
        scale_off = pid_t * stride_scale_token + cur_off * stride_scale_hidden
        scale = tl.load(scale_ptr + scale_off, mask=mask_scale)
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
        )
        output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
        if PERMUTE_SCALE:
            permuted_scale_off = (
                dst_row * stride_permuted_scale_token + cur_off * stride_permuted_scale_hidden
            )
            tl.store(permuted_scale_ptr + permuted_scale_off, scale, mask=mask_scale)
        if PERMUTE_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
            prob = tl.load(probs_ptr + prob_off)
            if pid_h == 0:
                permuted_prob_off = dst_row * stride_permuted_probs_token
                tl.store(permuted_probs_ptr + permuted_prob_off, prob)
            if prob == 0.0:
                # for routing_map padding
                # dst_row != -1 and prob == 0.0 means that this slot is padded
                tl.store(output_ptr + output_off, 0.0, mask=mask)
            else:
                tl.store(output_ptr + output_off, inp, mask=mask)
        else:
            tl.store(output_ptr + output_off, inp, mask=mask)


try:
    _permute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_permute_kernel)
except RuntimeError:
    pass


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    scale: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
):
    """
    Permute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    scale: torch.Tensor
        The scale of the input tensor. If it is not None, it will be permuted.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.
    num_out_tokens: int
        Number of tokens in the permuted tensor.
    hidden_size: int
        Hidden size of the input tensor.
    scale_hidden_dim: int
        Hidden size of the scale tensor.
    """
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None

    if scale is not None:
        permuted_scale = torch.empty(
            (num_out_tokens, scale_hidden_dim), dtype=scale.dtype, device="cuda"
        )
    else:
        permuted_scale = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        scale,
        permuted_probs,
        permuted_scale,
        num_experts,
        hidden_size,
        scale_hidden_dim,
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        scale.stride(0) if scale is not None else None,
        scale.stride(1) if scale is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        permuted_scale.stride(0) if permuted_scale is not None else None,
        permuted_scale.stride(1) if permuted_scale is not None else None,
        PERMUTE_PROBS=probs is not None,
        PERMUTE_SCALE=scale is not None,
    )
    return output, permuted_scale, permuted_probs


@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    unpermuted_probs_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size
    if PERMUTE_PROBS:
        # write 0.0 to probs_grad that are not routed
        if pid_h == 0:
            map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
            unpermuted_prob_off = (
                pid_t * stride_unpermuted_probs_token
                + stride_unpermuted_probs_expert * map_load_off
            )
            tl.store(
                unpermuted_probs_ptr + unpermuted_prob_off, 0.0, mask=map_load_off < num_experts
            )
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        src_row = tl.load(
            row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
        )
        input_off = src_row * stride_input_token + current_offset * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        inp = inp.to(compute_type)
        if WITH_MERGING_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            merging_prob_off = (
                pid_t * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
            inp *= merging_prob
        accumulator += inp
        if PERMUTE_PROBS:
            if pid_h == 0:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_t * stride_row_id_map_token
                    + (num_experts + idx) * stride_row_id_map_expert
                )
                unpermuted_prob_off = (
                    pid_t * stride_unpermuted_probs_token
                    + expert_idx * stride_unpermuted_probs_expert
                )
                permuted_prob_off = src_row * stride_permuted_probs_token
                prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
    accumulator = accumulator.to(data_type)
    output_off = pid_t * stride_output_token + current_offset * stride_output_hidden
    tl.store(output_ptr + output_off, accumulator, mask=mask)


try:
    _unpermute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_kernel)
except RuntimeError:
    pass


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Union[torch.Tensor, None],
    permuted_probs: Union[torch.Tensor, None],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    """
    Unpermute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs: torch.Tensor
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs: torch.Tensor
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    num_tokens: int
        Number of tokens in the permuted tensor.
    num_experts: int
        Number of experts in the permuted tensor.
    hidden_size: int
        Hidden size of the permuted tensor.
    """
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if permuted_probs is not None:
        unpermuted_probs = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs.dtype, device="cuda"
        )
    else:
        unpermuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _unpermute_kernel[grid](
        inp,
        output,
        row_id_map,
        merging_probs,
        permuted_probs,
        unpermuted_probs,
        num_experts,
        hidden_size,
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
    )
    return output, unpermuted_probs


@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    merging_probs_grad_ptr,
    row_id_map_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
    token_probs_grad_off = (
        pid * stride_merging_probs_grad_token + stride_merging_probs_grad_expert * map_load_off
    )
    tl.store(merging_probs_grad_ptr + token_probs_grad_off, 0.0, mask=map_load_off < num_experts)
    n_routed = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid * stride_row_id_map_token + idx * stride_row_id_map_expert
        )
        expert_idx = tl.load(
            row_id_map_ptr
            + pid * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )
        prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        current_start = 0
        while current_start < hidden_size:
            current_offset = current_start + tl.arange(0, BLOCK_SIZE)
            mask = current_offset < hidden_size
            input_off = (
                pid * stride_fwd_output_grad_token + current_offset * stride_fwd_output_grad_hidden
            )
            inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
            inp = inp.to(compute_type)
            merging_prob_off = (
                pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
            output = inp * merging_prob
            output = output.to(data_type)
            output_off = (
                dst_row * stride_fwd_input_grad_token
                + current_offset * stride_fwd_input_grad_hidden
            )
            tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

            fwd_input_off = (
                dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
            )
            fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
            prob_grad_accum += fwd_input.to(compute_type) * inp
            current_start += BLOCK_SIZE
        probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
        probs_grad_off = (
            pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
        )
        tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)


try:
    _unpermute_bwd_with_merging_probs_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_bwd_with_merging_probs_kernel)
except RuntimeError:
    pass


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    """
    Unpermute backward pass kernel with merging probs.

    Parameters
    ----------
    fwd_output_grad: torch.Tensor
        The gradient of the output tensor of shape `[num_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    fwd_input: torch.Tensor
        The input tensor of the forward pass of shape `[num_out_tokens, hidden_size]`.
    merging_probs: torch.Tensor
        The merging probabilities of the input tensor of shape `[num_tokens, num_experts]`.
    num_tokens: int
        Number of tokens in the permuted tensor.
    num_experts: int
        Number of experts in the permuted tensor.
    num_out_tokens: int
        Number of tokens in the output tensor.
    hidden_size: int
        Hidden size of the output tensor.
    """
    act_grad = torch.empty(
        (num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device="cuda"
    )
    merging_probs_grad = torch.empty(
        (num_tokens, num_experts), dtype=merging_probs.dtype, device="cuda"
    )
    grid = (num_tokens,)
    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_experts,
        hidden_size,
        row_id_map.stride(0),
        row_id_map.stride(1),
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        act_grad.stride(0),
        act_grad.stride(1),
        fwd_input.stride(0),
        fwd_input.stride(1),
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
    )
    return act_grad, merging_probs_grad


@triton.jit
def _make_chunk_sort_map_kernel(
    # pointers
    split_sizes_ptr,
    sorted_indices_ptr,
    dst_rows_ptr,
    # sizes
    num_splits: tl.constexpr,
    # metas
    IDX_LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    sorted_indices = tl.load(
        sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits
    )

    # get chunk idx of the current token in the input tensor
    input_split_sizes = tl.load(
        split_sizes_ptr + load_split_offset, mask=load_split_offset < num_splits, other=0
    ).to(tl.int32)
    input_split_sizes_cumsum = tl.cumsum(input_split_sizes)
    input_split_sizes_mask = tl.where(input_split_sizes_cumsum <= pid, 1, 0)
    input_chunk_idx = tl.sum(input_split_sizes_mask)
    input_split_sizes_presum = tl.sum(input_split_sizes * input_split_sizes_mask)
    in_chunk_offset = pid - input_split_sizes_presum

    # get chunk idx of the current token in the output tensor
    output_chunk_mask = tl.where(sorted_indices == input_chunk_idx, 1, 0)
    output_chunk_idx = tl.argmax(output_chunk_mask, axis=-1)

    # make row_id_map
    output_split_sizes = tl.load(
        split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits
    ).to(tl.int32)
    output_pre_split_sizes = tl.where(load_split_offset < output_chunk_idx, output_split_sizes, 0)
    dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset
    tl.store(dst_rows_ptr + pid, dst_row)


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
):
    """
    Make a row_id_map for chunk sort.

    Parameters
    ----------
    split_sizes: torch.Tensor
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices: torch.Tensor
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens: int
        Number of tokens in the input tensor.
    num_splits: int
        Number of splits of split_sizes and sorted_indices.
    """
    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device="cuda")
    grid = (num_tokens,)
    _make_chunk_sort_map_kernel[grid](
        split_sizes,
        sorted_indices,
        row_id_map,
        num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return row_id_map


@triton.jit
def _sort_chunks_by_map_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    hidden_size: tl.constexpr,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORWARD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if FORWARD:
        src_row = pid_t
        dst_row = tl.load(row_id_map_ptr + pid_t)
    else:
        src_row = tl.load(row_id_map_ptr + pid_t)
        dst_row = pid_t
    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size
    input_offsets = src_row * stride_input_token + current_offset * stride_input_hidden
    output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
    inp = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, inp, mask=mask)
    if PERMUTE_PROBS:
        if pid_h == 0:
            prob_off = src_row * stride_probs_token
            prob = tl.load(probs_ptr + prob_off)
            permuted_prob_off = dst_row * stride_permuted_probs_token
            tl.store(permuted_probs_ptr + permuted_prob_off, prob)


try:
    _sort_chunks_by_map_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_sort_chunks_by_map_kernel)
except RuntimeError:
    pass


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
):
    """
    Sort chunks with row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens: int
        Number of tokens in the input tensor.
    hidden_size: int
        Hidden size of the input tensor.
    is_forward: bool
        Whether the sort is for forward or backward.
    """
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _sort_chunks_by_map_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
        FORWARD=is_forward,
    )
    return output, permuted_probs

class _moe_permute_mask_map(torch.autograd.Function):
    """functional Permute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.probs = probs
            return inp, torch.tensor([], device=inp.device), torch.tensor([], device=inp.device)

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert routing_map.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert (
            num_out_tokens is not None
        ), "num_out_tokens must be provided to the fused permute function."

        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

        fp8 = isinstance(inp, QuantizedTensor)
        per_tensor_recipe = isinstance(inp, Float8Tensor)
        blockwise_recipe = isinstance(inp, Float8BlockwiseQTensor)
        mxfp8_recipe = isinstance(inp, MXFP8Tensor)

        if fp8:
            fp8_dtype = inp._fp8_dtype
            fake_dtype = inp.dtype
            # blockwise scaling
            if blockwise_recipe:
                fp8_scale = inp._rowwise_scale_inv.T.contiguous()
                scale_hidden_dim = fp8_scale.shape[1]
                assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                inp = inp._rowwise_data
            # mxfp8 scaling
            elif mxfp8_recipe:
                fp8_scale = inp._rowwise_scale_inv.contiguous()
                scale_hidden_dim = fp8_scale.shape[1]
                assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                inp = inp._rowwise_data
            # per-tensor scaling
            elif per_tensor_recipe:
                # Kernel does not need scale in per-tensor scaling
                fp8_scale = None
                scale_hidden_dim = None
                fp8_scale_inv = inp._scale_inv
                inp = inp._data
            else:
                raise ValueError("Unsupported FP8 recipe")
        else:
            fp8_scale = None
            fp8_dtype = None
            scale_hidden_dim = None

        output, permuted_scale, permuted_probs = permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            fp8_scale,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            scale_hidden_dim,
        )

        if fp8:
            if per_tensor_recipe:
                output = Float8Tensor(
                    data=output,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=fp8_scale_inv,
                    shape=output.shape,
                    dtype=fake_dtype,
                )
            elif blockwise_recipe:
                output = Float8BlockwiseQTensor(
                    shape=output.shape,
                    dtype=fake_dtype,
                    rowwise_data=output,
                    rowwise_scale_inv=permuted_scale.T.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=fp8_dtype,
                    quantizer=None,
                    is_2D_scaled=False,
                    requires_grad=output.requires_grad,
                )
            elif mxfp8_recipe:
                output = MXFP8Tensor(
                    shape=output.shape,
                    dtype=fake_dtype,
                    fp8_dtype=fp8_dtype,
                    rowwise_data=output,
                    rowwise_scale_inv=permuted_scale.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    quantizer=None,
                    requires_grad=output.requires_grad,
                )

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, ctx.probs

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            assert not isinstance(
                permuted_act_grad, QuantizedTensor
            ), "The backward of moe_permute does not support FP8."
            act_grad, probs_grad = unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


class _moe_unpermute_mask_map(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: Optional[torch.Tensor],
        restore_shape: Optional[torch.Size],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.merging_probs = merging_probs
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = (row_id_map.size(1) - 1) // 2

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        assert not isinstance(
            inp, QuantizedTensor
        ), "The forward of moe_unpermute does not support FP8."
        unpermuted_output, _ = unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )

        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            fp8 = isinstance(unpermuted_act_grad, QuantizedTensor)
            per_tensor_recipe = isinstance(unpermuted_act_grad, Float8Tensor)
            blockwise_recipe = isinstance(unpermuted_act_grad, Float8BlockwiseQTensor)
            mxfp8_recipe = isinstance(unpermuted_act_grad, MXFP8Tensor)

            if fp8:
                fp8_dtype = unpermuted_act_grad._fp8_dtype
                fake_dtype = unpermuted_act_grad.dtype
                # per-tensor scaling
                if per_tensor_recipe:
                    # Kernel does not need scale in per-tensor scaling
                    fp8_scale = None
                    scale_hidden_dim = None
                    fp8_scale_inv = unpermuted_act_grad._scale_inv
                    unpermuted_act_grad = unpermuted_act_grad._data
                # blockwise scaling
                elif blockwise_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.T.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                # mxfp8 scaling
                elif mxfp8_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                else:
                    raise ValueError("Unsupported FP8 recipe")
            else:
                scale_hidden_dim = None
                fp8_dtype = None
                fp8_scale = None

            if ctx.with_probs:
                assert (
                    not fp8
                ), "The backward of moe_unpermute with merging probs does not support FP8."
                act_grad, probs_grad = (
                    unpermute_with_mask_map_bwd_with_merging_probs(
                        unpermuted_act_grad,
                        row_id_map,
                        fwd_input,
                        merging_probs,
                        ctx.num_tokens,
                        ctx.num_experts,
                        ctx.num_permuted_tokens,
                        ctx.hidden_size,
                    )
                )
            else:
                act_grad, permuted_scale, _ = permute_with_mask_map(
                    unpermuted_act_grad,
                    row_id_map,
                    None,
                    fp8_scale,
                    ctx.num_tokens,
                    ctx.num_experts,
                    ctx.num_permuted_tokens,
                    ctx.hidden_size,
                    scale_hidden_dim,
                )

            if fp8:
                if per_tensor_recipe:
                    act_grad = Float8Tensor(
                        data=act_grad,
                        fp8_dtype=fp8_dtype,
                        fp8_scale_inv=fp8_scale_inv,
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                    )
                elif blockwise_recipe:
                    act_grad = Float8BlockwiseQTensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.T.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        fp8_dtype=fp8_dtype,
                        quantizer=None,
                        is_2D_scaled=False,
                        requires_grad=act_grad.requires_grad,
                    )
                elif mxfp8_recipe:
                    act_grad = MXFP8Tensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        fp8_dtype=fp8_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        quantizer=None,
                        requires_grad=act_grad.requires_grad,
                    )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None


def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num: int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    map_type: str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if map_type == "index":
        return _moe_permute_index_map.apply(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = _moe_permute_mask_map.apply(inp, routing_map, num_out_tokens, None)
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_permute_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map: torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    """
    output, row_id_map, permuted_probs = _moe_permute_mask_map.apply(
        inp, routing_map, num_out_tokens, probs
    )
    return output, permuted_probs, row_id_map


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by
        accumulation.
    restore_shape: torch.Size, default = None
        The output shape after the unpermute operation.
    map_type: str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs: torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        return _moe_unpermute_index_map.apply(inp, row_id_map, merging_probs)
    if map_type == "mask":
        return _moe_unpermute_mask_map.apply(inp, row_id_map, merging_probs, restore_shape)
    raise ValueError("map_type should be one of 'mask' or 'index'")


class _moe_chunk_sort(torch.autograd.Function):
    """functional MoE chunk permute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            return inp, probs

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert split_sizes.is_cuda, "TransformerEngine needs CUDA."
        assert sorted_idxs.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        num_tokens, hidden_size = inp.shape
        num_splits = split_sizes.size(0)
        assert num_splits == sorted_idxs.size(0)

        fp8 = isinstance(inp, Float8Tensor)
        if fp8:
            fp8_dtype = inp._fp8_dtype
            fp8_scale_inv = inp._scale_inv
            fake_dtype = inp.dtype
            inp = inp._data

        row_id_map = make_chunk_sort_map(
            split_sizes,
            sorted_idxs,
            num_tokens,
            num_splits,
        )
        output, permuted_probs = sort_chunks_by_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            hidden_size,
            is_forward=True,
        )
        if fp8:
            output = Float8Tensor(
                data=output,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=fp8_scale_inv,
                shape=output.shape,
                dtype=fake_dtype,
            )

        ctx.save_for_backward(row_id_map)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, permuted_probs_grad

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            fp8 = isinstance(permuted_act_grad, Float8Tensor)
            if fp8:
                fp8_dtype = permuted_act_grad._fp8_dtype
                fp8_scale_inv = permuted_act_grad._scale_inv
                fake_dtype = permuted_act_grad.dtype
                permuted_act_grad = permuted_act_grad._data
            act_grad, probs_grad = sort_chunks_by_map(
                permuted_act_grad,
                row_id_map,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.hidden_size,
                is_forward=False,
            )
            if fp8:
                act_grad = Float8Tensor(
                    data=act_grad,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=fp8_scale_inv,
                    shape=act_grad.shape,
                    dtype=fake_dtype,
                )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is split along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.
    """
    output, _ = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, None)
    return output


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor and probs based on the split_sizes and sorted indices.
    The inp tensor is split along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.
    """
    output, permuted_probs = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, probs)
    return output, permuted_probs
