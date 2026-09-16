# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Iterable, Tuple

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from torch.amp import custom_bwd, custom_fwd
from composer.utils.dist import get_tp_group, get_tp_group_rank, get_tp_group_size

from llmfoundry.models.parallel.utils import (
    split_along_dim,
    gather_along_dim,
    reduce,
)
from llmfoundry.models.parallel.sequence import SeqAllToAll

@torch._dynamo.allow_in_graph
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return reduce(grad_output, get_tp_group())

@torch._dynamo.allow_in_graph
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, reduce_operation):
        return reduce(input_, get_tp_group(), reduce_operation)

    @staticmethod
    def forward(ctx, input_, reduce_operation):
        return reduce(input_, get_tp_group(), reduce_operation)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

@torch._dynamo.allow_in_graph
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return split_along_dim(input_, -1, get_tp_group())

    @staticmethod
    def forward(ctx, input_):
        return split_along_dim(input_, -1, get_tp_group())

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_dim(grad_output, -1, get_tp_group())


@torch._dynamo.allow_in_graph
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = get_tp_group() if group is None else group
        return gather_along_dim(input_, -1, ctx.group)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return split_along_dim(grad_output, -1, group), None


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        group: ProcessGroup,
        input: torch.Tensor,
        output_split_sizes: Iterable[int] | None,
        input_split_sizes: Iterable[int] | None,
    ) -> torch.Tensor:
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple[None, torch.Tensor, None, None]:
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


class _AllGatherLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.Function,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        process_group: ProcessGroup,
        gather_dim: int,
        no_recompute: bool,
    ):
        ctx.save_for_backward(input_, weight, bias)
        ctx.gather_dim = gather_dim
        ctx.process_group = process_group
        ctx.use_bias = bias is not None

        output = torch.ops.symm_mem.fused_all_gather_matmul(
            input_, 
            [weight.T],
            gather_dim=gather_dim,
            group_name=process_group.group_name,
        )[1][0]

        if bias is not None:
            output += bias

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
    ):
        input_, weight, bias = ctx.saved_tensors

        full_input_shape = list(input_.size())
        full_input_shape[ctx.gather_dim] *= ctx.process_group.size()
        full_input = torch.empty(
            full_input_shape,
            dtype=input_.dtype,
            device=input_.device,
            requires_grad=False,
        )
        handle = torch.distributed.all_gather_into_tensor(
            full_input, input_, group=ctx.process_group, async_op=True
        )

        grad_input = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            grad_output,
            weight,
            "sum",
            scatter_dim=ctx.gather_dim,
            group_name=ctx.process_group.group_name,
        )

        # only keep hidden dim
        grad_output = grad_output.flatten(0, -2)
        handle.wait()
        full_input = full_input.flatten(0, -2)
        grad_weight = grad_output.T @ full_input

        grad_bias = grad_output.sum(dim=0) if ctx.use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None


class _LinearReduceScatter(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.Function,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        process_group: ProcessGroup,
        scatter_dim: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_, weight)
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.use_bias = bias is not None
        with torch.autocast("cuda", enabled=False):
            output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                input_,
                weight.T,
                "sum",
                scatter_dim=scatter_dim,
                group_name=process_group.group_name,
            )
        if bias is not None:
            output += bias
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
    ):
        input_, weight = ctx.saved_tensors
        full_grad_output, grad_input = torch.ops.symm_mem.fused_all_gather_matmul(
            grad_output,
            [weight],
            gather_dim=ctx.scatter_dim,
            group_name=ctx.process_group.group_name,
        )

        full_grad_output = full_grad_output.flatten(0, -2)
        input_ = input_.flatten(0, -2)
        grad_weight = full_grad_output.T @ input_

        grad_bias = grad_output.sum(dim=0) if ctx.use_bias else None
        return grad_input[0], grad_weight, grad_bias, None, None


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_, reduce_operation='SUM'):
    return _ReduceFromModelParallelRegion.apply(input_, reduce_operation)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(
    input_: torch.Tensor, group: ProcessGroup | None = None
) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, group)


def narrow_to_tensor_model_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    tp_rank = get_tp_group_rank()
    tp_size = get_tp_group_size()
    hidden_size = input_.size(dim)
    assert hidden_size % tp_size == 0, "last dimension must be divisible by tp group size"
    local_hidden_size = hidden_size // tp_size
    return input_.narrow(dim, tp_rank * local_hidden_size, local_hidden_size)


def all_gather_linear(
    input_: torch.Tensor,
    weight: torch.Tensor,
    logical_batch_size: int,
    bias: torch.Tensor | None = None,
    no_recompute: bool = False,
) -> torch.Tensor:
    group = get_tp_group()
    in_feat = input_.size(-1)
    input_ = input_.view(-1, in_feat)
    out = _AllGatherLinear.apply(input_, weight, bias, group, 0, no_recompute)
    return out.view(logical_batch_size, -1, out.size(-1))


def linear_reduce_scatter(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    group = get_tp_group()

    in_feat = input_.size(-1)
    input_ = input_.view(-1, in_feat)
    out = _LinearReduceScatter.apply(input_, weight, bias, group, 0)
    return out.unsqueeze(0)


def distribute_async_tp_embeddings(input_: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, local_hidden_size = input_.size()
    # With input_embeds + SP - input_ will be splitted by seq_len and won't be contiguous
    # So we can't use .view()
    input_ = input_.reshape(1, batch_size * seq_len, local_hidden_size)
    out = SeqAllToAll.apply(input_, 2, 1, get_tp_group())
    tp_size = get_tp_group_size()
    out = out.view(1, -1, local_hidden_size * tp_size)
    return out


def all_to_all(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes_: Iterable[int] | None = None,
    input_split_sizes: Iterable[int] | None = None,
) -> torch.Tensor:
    """Wrapper for autograd function"""
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)
