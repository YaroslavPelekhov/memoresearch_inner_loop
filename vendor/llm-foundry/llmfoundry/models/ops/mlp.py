# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py
from functools import partial
from typing import Optional, List
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

import megablocks.ops as ops
from megablocks.backend import kernels

from megablocks.layers.activation_fn import act_fn
# from megablocks.layers.all_to_all import all_to_all  # not in the docker image and is unused

import stk
from composer.utils.dist import get_group_size, get_group_rank
from composer.utils import global_actlog_state

###############################################
# FLASH ATTENTION SWIGLU
###############################################
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)


###############################################
# XFORMERS SWIGLU
###############################################
#@torch.jit.script
@torch.compile
def _swiglu_backward(dy, x):
    # https://github.com/pytorch/pytorch/blob/563b065f5a4b4055fa6b025c2514b566d5fd9439/aten/src/ATen/native/Activation.cpp#L483
    sigm = 1 / (1 + torch.exp(-x.float()))
    return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)
###############################################

class FusedGatedMLPFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        weight1,
        bias1,
        weight2,
        bias2,
        weight3,
        bias3,
        activation="swiglu",
        checkpoint_lvl=0,
        heuristic=-1,
    ):
        """
        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute swiglu inputs in the bwd
        2: recompute swiglu inputs and output in the bwd
        3: recompute all in the bwd

        heuristic:
        -1: use pytorch dense modules
        """
        assert not (bias1 or bias2 or bias3), \
            'all biases should be None for this implementation'
        assert -1 <= heuristic <= 4
        assert activation in ["swiglu"]
        if activation == "swiglu":
            assert heuristic == -1
        assert checkpoint_lvl in [0, 1, 2, 3]
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.activation = activation
        ctx.heuristic = heuristic

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        total_x = x

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2, weight3 = [a.to(dtype=dtype) for a in [weight1, weight2, weight3]]
        weight1 = weight1.contiguous()
        weight2 = weight2.contiguous()
        weight3 = weight3.contiguous()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight1.shape, *weight2.shape, *weight3.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")

        # biases are not supported from here
        if heuristic != -1:
            raise NotImplementedError(f'{activation=} {heuristic=}')
    
        global_actlog_state.use_monitor_variable(total_x, "model.model.layers.{}.mlp.up_proj", "_input.0")

        xw1 = F.linear(total_x, weight1)
        xw2 = F.linear(total_x, weight2)

        if checkpoint_lvl <= 1:
            hidden = F.silu(xw1, inplace=False) * xw2
        else:
            hidden = F.silu(xw1, inplace=True).mul_(xw2) #hidden -> xw1
        
        global_actlog_state.use_monitor_variable(hidden, "model.model.layers.{}.mlp.down_proj", "_input.0")

        output = F.linear(hidden, weight3)

        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, weight2, weight3, hidden, xw1, xw2)
        elif checkpoint_lvl == 1:
            del hidden
            ctx.save_for_backward(x, weight1, weight2, weight3, xw1, xw2)
        elif checkpoint_lvl == 2:
            del xw2
            ctx.save_for_backward(x, weight1, weight2, weight3, hidden)
        elif checkpoint_lvl == 3:
            del xw1, xw2
            ctx.save_for_backward(x, weight1, weight2, weight3)
        output = output.reshape(*batch_shape, output.shape[-1])
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        activation = ctx.activation
        if activation == "swiglu":
            activation_fn = lambda x, y: F.silu(x) * y
        else:
            raise NotImplementedError(f"{activation=}")

        total_x, weight1, weight2, weight3, *rest = ctx.saved_tensors
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()

        if not ctx.heuristic == -1:
            raise NotImplementedError()

        if checkpoint_lvl == 0:
            hidden, xw1, xw2 = rest
        elif checkpoint_lvl == 1:
            xw1, xw2 = rest
            hidden = activation_fn(xw1, xw2)
        elif checkpoint_lvl == 2:
            (hidden,) = rest
            xw1 = F.linear(total_x, weight1)
            xw2 = F.linear(total_x, weight2)
        elif checkpoint_lvl == 3:
            xw1 = F.linear(total_x, weight1)
            xw2 = F.linear(total_x, weight2)
            hidden = F.silu(xw1) * xw2
        else:
            raise NotImplementedError(f"{checkpoint_lvl=}")
        
        global_actlog_state.use_monitor_variable(grad_output, 
                                                 "model.model.layers.{}.mlp.down_proj", 
                                                 "_grad_output.0", 
                                                 is_forward=False)

        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        hidden = hidden.reshape(batch_dim, hidden.shape[-1])
        grad_weight3, _ = fused_dense_cuda.linear_bias_wgrad(
            hidden, grad_output, False
        )
        if activation == "swiglu":
            activation_grad_fn = swiglu_bwd
        else:
            raise NotImplementedError()

        grad_hidden = grad_output @ weight3
        with torch.jit.fuser("fuser2"):
            grad_xw1, grad_xw2 = activation_grad_fn(xw1, xw2, grad_hidden)
        
        global_actlog_state.use_monitor_variable(grad_xw1, 
                                                 "model.model.layers.{}.mlp.gate_proj", 
                                                 "_grad_output.0", 
                                                 is_forward=False)
    
        global_actlog_state.use_monitor_variable(grad_xw2, 
                                                 "model.model.layers.{}.mlp.up_proj", 
                                                 "_grad_output.0", 
                                                 is_forward=False)

        grad_weight1, _ = fused_dense_cuda.linear_bias_wgrad(
            total_x, grad_xw1, False
        )
        grad_weight2, _ = fused_dense_cuda.linear_bias_wgrad(
            total_x, grad_xw2, False
        )
        grad_input = grad_xw2 @ weight2
        torch.addmm(
            grad_input, grad_xw1, weight1, beta=1, alpha=1, out=grad_input
        )  # grad_input = grad_xw2 @ weight2 + grad_xw1 @ weight1

        grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        del total_x, hidden, xw1, xw2
        return (
            grad_input,
            grad_weight1,
            None,
            grad_weight2,
            None,
            grad_weight3,
            None,
            None,
            None,
            None,
            None,
            None,
        )



def stk_swiglu_backward(grad: stk.Matrix, x: stk.Matrix):
    # NOTE: The two sparse matrices must have the same topology.
    if isinstance(grad, stk.Matrix) and isinstance(x, stk.Matrix):
        return stk.Matrix(
            x.size(),
            _swiglu_backward(grad.data, x.data),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t)
    return _swiglu_backward(grad, x)


def _save_stk_matrix(ctx, name, matrix) -> tuple:
    matrix_tensors = (
        matrix.data,
        matrix.row_indices,
        matrix.column_indices,
        matrix.offsets,
        matrix.column_indices_t,
        matrix.offsets_t,
        matrix.block_offsets_t
    )
    setattr(ctx, f"{name}_shape", matrix.shape)
    return matrix_tensors


def _load_stk_matrix(ctx, name, saved4back):
    assert len(saved4back) == 7, f"{len(saved4back)=}"
    matrix = stk.Matrix(getattr(ctx, f"{name}_shape"), *saved4back)
    return matrix

def _load_all_stk_matrix(ctx, names, save_for_backward) -> tuple:
    # 7 elements to load stk.Matrix
    assert len(save_for_backward) % 7 == 0
    assert len(save_for_backward) == 7*len(names)
    result = ()
    for name, i in zip(names, range(len(names))):
        result += (_load_stk_matrix(ctx, name, save_for_backward[7*i: 7*(i+1)]),)
    return result


def _split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    assert (
        tensor.size()[last_dim] % num_partitions == 0
    ), f"last_dim_size = {tensor.size()[last_dim]} is not divisible by num_partitions = {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# From https://github.com/stanford-futuredata/megablocks/blob/7c25169ce87c32c31e8845ef34785d3095b1a2cb/megablocks/layers/dmoe.py#L31
def sparse_transpose(size, row_indices, column_indices, blocking: int, transpose_sort_end_bit: int):
    block_columns = size[1] // blocking

    # Sort row indices by column indices to get the transposed matrix's
    # column indices.
    #
    # NOTE: Our sort operation uses the same width indices as the input values.
    # To avoid overflow when we have large activation matrices we cast to
    # 32-bit before sorting.
    _, gather_indices = ops.sort(column_indices.int(), transpose_sort_end_bit)

    # There are a constant number of blocks in every row of the sparse matrix.
    # A blocks offset is:
    #
    # row_index * blocks_per_row + column_index % blocks_per_row
    #
    # Once we have the block offsets ordered for transposition we can divide
    # by blocks_per_row to get the transposed column indices.
    column_indices_t = row_indices.gather(0, gather_indices.long())
    block_offsets_t = gather_indices.int()

    zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
    nnz_per_column = ops.histogram(column_indices, block_columns)
    nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
    offsets_t = torch.cat([zero, nnz_per_column])
    return column_indices_t, offsets_t, block_offsets_t

# From https://github.com/stanford-futuredata/megablocks/blob/7c25169ce87c32c31e8845ef34785d3095b1a2cb/megablocks/layers/dmoe.py#L59
def topology(x: torch.Tensor, padded_bins: torch.Tensor, num_experts: int, ffn_dim: int, transpose_sort_end_bit: int, blocking: int):
    padded_tokens, _ = x.size()
    assert padded_tokens %  blocking == 0
    assert  ffn_dim %  blocking == 0

    # Offsets for the sparse matrix. All rows have the
    # same number of nonzero blocks dictated by the
    # dimensionality of a single expert.
    block_rows = padded_tokens //  blocking
    blocks_per_row =  ffn_dim //  blocking
    offsets = torch.arange(
        0,
        block_rows * blocks_per_row + 1,
        blocks_per_row,
        dtype=torch.int32,
        device=x.device,
    )

    # Indices for the sparse matrix. The indices for
    # the intermediate matrix are dynamic depending
    # on the mapping of tokens to experts.
    column_indices = ops.topology(
        padded_bins,  blocking, block_rows, blocks_per_row
    )

    # TODO(tgale): This is unused. Remove the need for this in stk.
    # For now, use meta init to save the device memory.
    data = torch.empty(
        column_indices.numel(),
         blocking,
         blocking,
        dtype=x.dtype,
        device="meta",
    )
    shape = (padded_tokens,  ffn_dim *  num_experts)
    row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
    column_indices_t, offsets_t, block_offsets_t = sparse_transpose(
        shape, row_indices, column_indices, blocking, transpose_sort_end_bit
    )
    return stk.Matrix(
        shape,
        data,
        row_indices,
        column_indices,
        offsets,
        column_indices_t,
        offsets_t,
        block_offsets_t,
    )


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x

# From https://github.com/stanford-futuredata/megablocks/blob/7c25169ce87c32c31e8845ef34785d3095b1a2cb/megablocks/layers/dmoe.py#L103
def indices_and_padded_bins(top_experts: torch.Tensor, num_experts: int, blocking :int, sort_end_bit: int):
    # Sort the expert ids to produce the scatter/gather
    # indices for the permutation.
    top_experts = top_experts.int()
    bin_ids, indices = ops.sort(top_experts, sort_end_bit)

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    tokens_per_expert = ops.histogram(top_experts, num_experts)

    # Round the token counts up to the block size used in
    # the matrix muliplications. Caculate the starting
    # position of each bin.
    padded_tokens_per_expert = ops.round_up(tokens_per_expert, blocking)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
    padded_bins = promote_scalar(padded_bins)

    # Calculate the bin bounds for the sorted tokens.
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)
    bins = promote_scalar(bins)
    return indices, bin_ids, bins, padded_bins, tokens_per_expert, padded_tokens_per_expert


class StkFusedGatedMLPFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        weight1,
        bias1,
        weight2,
        bias2,
        weight3,
        bias3,
        expert_weights,
        top_experts,
        ffn_dim,
        top_k,
        num_experts,
        blocking,
        sort_end_bit,
        transpose_sort_end_bit,
        activation="swiglu",
        checkpoint_lvl=0,
        heuristic=-1,
        ep_group=None,
        tp_group=None,
    ):
        """
        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute swiglu inputs in the bwd
        2: recompute swiglu inputs and output in the bwd
        3: recompute swiglu inputs, output, in the bwd
        4: recompute all in the bwd

        heuristic:
        -1: use stk sparse computation
        """
        assert not (bias1 or bias2 or bias3), \
            'all biases should be None for this implementation'
        assert -1 <= heuristic < 0
        assert activation in ["swiglu"]
        assert checkpoint_lvl in [0, 1, 2, 3, 4]
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.activation = activation
        ctx.heuristic = heuristic
        ctx.top_k = top_k
        ctx.ep_group = ep_group
        ctx.tp_group = tp_group

        ep_size = get_group_size(ep_group) or 1
        tp_size = get_group_size(tp_group) or 1

        ## x padding
        top_experts = top_experts.flatten()
        (
            indices,
            bin_ids,
            bins,
            padded_bins,
            tokens_per_expert,
            padded_tokens_per_expert,
        ) = indices_and_padded_bins(
            top_experts,
            num_experts,
            blocking,
            sort_end_bit
        )
        x = x.view(-1, x.shape[-1])
        padded_x = kernels.padded_gather(
            x, indices, bin_ids, None, bins, padded_bins, top_k)
        ############################################################
        global_topo_backward = ()

        if ep_size > 1:
            with torch.no_grad():
                assert  tp_size == 1, 'Not implemented'
                # todo: expert parallel
                repeated_tokens_per_expert = ops.repeat(
                    padded_tokens_per_expert, (1,))
                # Split along last dimension.
                tokens_per_expert_grouped = _split_tensor_along_last_dim(repeated_tokens_per_expert, ep_size)
                tokens_per_expert_grouped_sum = [t.sum().cpu().item() for t in tokens_per_expert_grouped]
                ctx.tokens_per_expert_grouped_sum = tokens_per_expert_grouped_sum

                # Note: torch.split does not create contiguous tensors by default.
                ep_rank = get_group_rank(ep_group)
                parallel_tokens_per_expert = tokens_per_expert_grouped[ep_rank].contiguous()
                tokens_received = tokens_per_expert_grouped_sum[ep_rank]

            parallel_padded_x = torch.split(padded_x, tokens_per_expert_grouped_sum, dim=0)[ep_rank].contiguous()

            with torch.no_grad():
                replicate_bins = ops.inclusive_cumsum(
                    parallel_tokens_per_expert.flatten(), 0)
                replicate_bins = (
                    replicate_bins.view(1)
                    if not len(replicate_bins.size())
                    else replicate_bins
                )
                # Construct the expert indices for the permuted tokens.
                assert tp_size == 1, 'Not implemented'
                parallel_top_expert = torch.arange(
                    num_experts // ep_size,
                    dtype=torch.int32,
                    device=indices.device
                )
                parallel_top_expert = ops.replicate(
                    parallel_top_expert.unsqueeze(dim=0),
                    replicate_bins, tokens_received).flatten()

                parallel_bin_ids, parallel_indices = ops.sort(
                    parallel_top_expert, sort_end_bit)

                parallel_padded_tokens_per_expert = ops.round_up(parallel_tokens_per_expert, blocking)
                parallel_padded_bins = ops.inclusive_cumsum(parallel_padded_tokens_per_expert, 0)
                parallel_padded_bins = promote_scalar(parallel_padded_bins)

                # Calculate the bins boundaries from the token counts.
                parallel_bins = ops.inclusive_cumsum(
                    parallel_tokens_per_expert, 0)
                parallel_bins = (
                    parallel_bins.view(1)
                    if not len(parallel_bins.size())
                    else parallel_bins
                )

            global_topo_backward = (indices, bin_ids, bins, padded_bins)

            padded_x = parallel_padded_x
            tokens_per_expert = parallel_tokens_per_expert
            indices = parallel_indices
            bin_ids = parallel_bin_ids
            bins = parallel_bins
            padded_bins = parallel_padded_bins

        ############################################################
        padded_gather_backward = (indices, bin_ids, bins, padded_bins)
        if torch.is_autocast_enabled():
            padded_x = padded_x.to(dtype=torch.get_autocast_gpu_dtype())
        total_x = padded_x.contiguous()
        ## init topo
        topo = topology(
            total_x,
            padded_bins,
            num_experts // ep_size,
            ffn_dim,
            transpose_sort_end_bit,
            blocking
        )
        ##

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2, weight3 = [a.to(dtype=dtype) for a in [weight1, weight2, weight3]]

        ctx.ep_size = ep_size
        ctx.tp_size = tp_size

        weight1 = weight1.contiguous()
        weight2 = weight2.contiguous()
        weight3 = weight3.contiguous()

        # biases are not supported from here
        if heuristic == -1:
            xw1 = stk.ops.sdd(total_x, weight1, topo)
            xw2 = stk.ops.sdd(total_x, weight2, topo)
            activation_fn_out = act_fn(xw1, partial(F.silu, inplace=checkpoint_lvl >= 2))

            hidden = stk.ops.mul(activation_fn_out, xw2)
            final_glu = stk.ops.dsd(hidden, weight3)
        else:
            raise NotImplementedError(f'{activation=} {heuristic=}')

        expert_weights = expert_weights.flatten().to(total_x.dtype)

        if ep_size > 1:
            tensor_list = [
                torch.empty(
                    (tokens_per_expert_grouped_sum[ep_r], final_glu.shape[-1]),
                    dtype=final_glu.dtype,
                    device=final_glu.device,
                )
                for ep_r in range(get_group_size(ep_group))
            ]
            tensor_list[ep_rank] = final_glu
            handle = torch.distributed.all_gather(tensor_list, final_glu, group=ep_group, async_op=True)
            handle.wait()
            final_glu = torch.cat(tensor_list, dim=0)

        if ep_size > 1:
            (ep_indices, ep_bin_ids, ep_bins, ep_padded_bins) = global_topo_backward
            output = kernels.padded_scatter(
                final_glu, ep_indices, ep_bin_ids, expert_weights, ep_bins, ep_padded_bins, top_k)
        else:
            output = kernels.padded_scatter(
                final_glu, indices, bin_ids, expert_weights, bins, padded_bins, top_k)

        output = output.to(total_x.dtype)
        topo_tensors = ()
        if topo:
            ctx.topo_shape = topo.shape
            topo_tensors = (
                topo.data,
                topo.row_indices,
                topo.column_indices,
                topo.offsets,
                topo.column_indices_t,
                topo.offsets_t,
                topo.block_offsets_t
            )
        save_activations = ()
        if heuristic == -1:
            if checkpoint_lvl == 0:
                save_activations += (final_glu,)
                save_activations += _save_stk_matrix(ctx, 'hidden', hidden)
                save_activations += _save_stk_matrix(ctx, 'xw1', xw1)
                save_activations += _save_stk_matrix(ctx, 'xw2', xw2)
            elif checkpoint_lvl == 1:
                save_activations += (final_glu,)
                save_activations += _save_stk_matrix(ctx, 'xw1', xw1)
                save_activations += _save_stk_matrix(ctx, 'xw2', xw2)
                del hidden
            elif checkpoint_lvl == 2:
                save_activations += (final_glu,)
                save_activations += _save_stk_matrix(ctx, 'hidden', hidden)
                del xw1, xw2
            elif checkpoint_lvl == 3:
                save_activations += (final_glu,)
                del xw1, xw2, hidden
            elif checkpoint_lvl == 4:
                del xw1, xw2, hidden, final_glu
        else:
            raise NotImplementedError()

        ctx.save_for_backward(
            x, weight1, weight2, weight3, expert_weights,
            *padded_gather_backward,
            *global_topo_backward,
            *save_activations,
            *topo_tensors
        )
        return output#.clone()

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        activation = ctx.activation
        heuristic = ctx.heuristic

        if activation != "swiglu":
            raise NotImplementedError(f"{activation=}")

        (
            x,
            weight1,
            weight2,
            weight3,
            # padded_scatter_backward
            expert_weights,
            # padded_gather_backward
            indices,
            bin_ids,
            bins,
            padded_bins,
            #
            *rest
        ) = ctx.saved_tensors

        ep_size = ctx.ep_size
        ep_group = ctx.ep_group
        # global_topo_backward
        if ep_size > 1:
            (
                ep_indices,
                ep_bin_ids,
                ep_bins,
                ep_padded_bins,
            ) = rest[:4]
            rest = rest[4:]

            padded_x = kernels.padded_gather(
                x, ep_indices, ep_bin_ids, None, ep_bins, ep_padded_bins, ctx.top_k)
            ep_rank = get_group_rank(ep_group)
            padded_x = torch.split(padded_x, ctx.tokens_per_expert_grouped_sum, dim=0)[ep_rank].contiguous()
        else:
            padded_x = kernels.padded_gather(
                x, indices, bin_ids, None, bins, padded_bins, ctx.top_k
            )
        if torch.is_autocast_enabled():
            padded_x = padded_x.to(dtype=torch.get_autocast_gpu_dtype())
        total_x = padded_x.contiguous()

        if heuristic == -1:
            topo_tensors = rest[-7:]
            rest = rest[:-7]
            topo = stk.Matrix(ctx.topo_shape, *topo_tensors)
            if checkpoint_lvl == 0:
                final_glu = rest[0]
                hidden, xw1, xw2 = _load_all_stk_matrix(ctx, ('hidden', 'xw1', 'xw2'), rest[1:])

                activation_fn_out = act_fn(xw1, F.silu)
            elif checkpoint_lvl == 1:
                final_glu = rest[0]
                xw1, xw2 = _load_all_stk_matrix(ctx, ('xw1', 'xw2'), rest[1:])

                activation_fn_out = act_fn(xw1, F.silu)
                hidden = stk.ops.mul(activation_fn_out, xw2)
            elif checkpoint_lvl == 2:
                final_glu = rest[0]
                (hidden,) = _load_all_stk_matrix(ctx, ('hidden', ), rest[1:])
                xw1 = stk.ops.sdd(total_x, weight1, topo)
                xw2 = stk.ops.sdd(total_x, weight2, topo)

                activation_fn_out = act_fn(xw1, F.silu)
            elif checkpoint_lvl == 3:
                final_glu = rest[0]
                xw1 = stk.ops.sdd(total_x, weight1, topo)
                xw2 = stk.ops.sdd(total_x, weight2, topo)
                activation_fn_out = act_fn(xw1, F.silu)

                hidden = stk.ops.mul(activation_fn_out, xw2)
            elif checkpoint_lvl == 4:
                xw1 = stk.ops.sdd(total_x, weight1, topo)
                xw2 = stk.ops.sdd(total_x, weight2, topo)
                activation_fn_out = act_fn(xw1, F.silu)

                hidden = stk.ops.mul(activation_fn_out, xw2)
                final_glu = stk.ops.dsd(hidden, weight3)
                if ep_size > 1:
                    tensor_list = [
                        torch.empty(
                            (ctx.tokens_per_expert_grouped_sum[ep_r], final_glu.shape[-1]),
                            dtype=final_glu.dtype,
                            device=final_glu.device,
                        )
                        for ep_r in range(get_group_size(ep_group))
                    ]
                    tensor_list[ep_rank] = final_glu
                    handle = torch.distributed.all_gather(tensor_list, final_glu, group=ep_group, async_op=True)
                    handle.wait()
                    final_glu = torch.cat(tensor_list, dim=0)
            else:
                raise NotImplementedError(f"{checkpoint_lvl=}")
        else:
            raise NotImplementedError(f'{heuristic=}')

        if heuristic == -1:
            # todo: padded tokens we can pass from forward sync
            # [padded tokens , hidden dim ]
            if ep_size > 1:
                grad_final_glu = kernels.padded_gather(
                    grad_output,
                    ep_indices,
                    ep_bin_ids,
                    expert_weights,
                    ep_bins,
                    ep_padded_bins,
                    ctx.top_k)
                ep_rank = get_group_rank(ep_group)
                grad_final_glu = torch.split(grad_final_glu, ctx.tokens_per_expert_grouped_sum, dim=0)[ep_rank].contiguous()
            else:
                grad_final_glu = kernels.padded_gather(
                    grad_output,
                    indices,
                    bin_ids,
                    expert_weights,
                    bins,
                    padded_bins,
                    ctx.top_k)

            grad_weight3 = stk.ops.dsd(hidden.t(), grad_final_glu)
            # NOTE: This reuses the hidden allocation.
            stk.backend.triton_kernels.sdd(
                grad_final_glu, weight3.t(),
                hidden.shape,
                hidden.data,
                hidden.offsets,
                hidden.row_indices,
                hidden.column_indices)
            grad_hidden = hidden
            del grad_final_glu
            # todo: can reuse final_glu allocation
            if ep_size > 1:
                expert_weights_grad = kernels.padded_scatter_wgrad(
                    final_glu,
                    grad_output,
                    ep_indices,
                    ep_bin_ids,
                    ep_bins,
                    ep_padded_bins,
                    ctx.top_k).view(-1, ctx.top_k)
            else:
                expert_weights_grad = kernels.padded_scatter_wgrad(
                    final_glu,
                    grad_output,
                    indices,
                    bin_ids,
                    bins,
                    padded_bins,
                    ctx.top_k).view(-1, ctx.top_k)

            grad_silu = stk.ops.mul(grad_hidden, xw2)
            grad_xw1 = stk_swiglu_backward(grad_silu, xw1)
            del grad_silu

            grad_xw2 = stk.ops.mul(grad_hidden, activation_fn_out)
            del activation_fn_out, grad_hidden

            grad_weight1 = stk.ops.dds(total_x.t(), grad_xw1)
            grad_weight2 = stk.ops.dds(total_x.t(), grad_xw2)

            grad_input = stk.ops.dsd(grad_xw1, weight1.t())
            grad_input.add_(stk.ops.dsd(grad_xw2, weight2.t()))
            del grad_xw1, grad_xw2
        else:
            raise NotImplementedError(f"{heuristic=}")
        del total_x, hidden, xw1, xw2, final_glu

        grad_input = grad_input.contiguous()
        if ep_size > 1:
            tensor_list = [
                torch.empty(
                    (ctx.tokens_per_expert_grouped_sum[ep_r], grad_input.shape[-1]),
                    dtype=grad_input.dtype,
                    device=grad_input.device,
                )
                for ep_r in range(get_group_size(ep_group))
            ]
            tensor_list[ep_rank] = grad_input
            handle = torch.distributed.all_gather(tensor_list, grad_input, group=ep_group, async_op=True)
            handle.wait()
            grad_input = torch.cat(tensor_list, dim=0)
            grad_input = kernels.padded_scatter(
                grad_input, ep_indices, ep_bin_ids, None, ep_bins, ep_padded_bins, ctx.top_k)
        else:
            grad_input = kernels.padded_scatter(
                grad_input, indices, bin_ids, None, bins, padded_bins, ctx.top_k)
        del indices, bin_ids, bins, padded_bins
        return (
            grad_input,
            grad_weight1,
            None,
            grad_weight2,
            None,
            grad_weight3,
            None,
            expert_weights_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_mlp_func(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    weight3: Tensor,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    bias3: Optional[Tensor] = None,
    activation: str = "swiglu",
    checkpoint_lvl: int = 0,
    heuristic: int = -1,
):
    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])
    assert activation in ["swiglu"]
    return FusedGatedMLPFunc.apply(
        x,
        weight1,
        bias1,
        weight2,
        bias2,
        weight3,
        bias3,
        activation,
        checkpoint_lvl,
        heuristic,
    ).reshape([*batch_shape, -1])



def xformers_style_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    checkpoint_lvl: int = 0,
    heuristic: int = -1,
):
    return FusedGatedMLPFunc.apply(
        x, w1, None, w2, None, w3, None, "swiglu", checkpoint_lvl, -1
    )

