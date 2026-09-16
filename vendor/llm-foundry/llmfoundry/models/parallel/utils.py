import math

import torch
import torch.distributed
from torch.distributed import ReduceOp

from composer.utils.dist import is_trivial_process_group, all_reduce

class _AllReduceWithGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        ctx.group = group
        output_tensor = input_tensor.clone()
        all_reduce(output_tensor, reduce_operation="SUM", group=ctx.group, async_op=False)
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        all_reduce(grad_input, reduce_operation="SUM", group=ctx.group, async_op=False)
        return grad_input, None


def all_reduce_with_grads(input_, group=None):
    return _AllReduceWithGrads.apply(input_, group)

def gather_along_dim(
    input_: torch.Tensor,
    dim: int,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    if is_trivial_process_group(group):
        return input_

    world_size = group.size()

    rank = group.rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    handle = torch.distributed.all_gather(tensor_list, input_, group=group, async_op=True)
    handle.wait()

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _slice_tensor_at_dim(
    input_: torch.Tensor, dim: int, start: int, end: int
) -> torch.Tensor:
    split_index = [slice(None) for _ in range(input_.dim())]
    split_index[dim] = slice(start, end)
    output = input_[split_index].contiguous()
    return output


def split_along_dim(
    input_: torch.Tensor,
    dim: int,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    if is_trivial_process_group(group):
        return input_

    world_size = group.size()
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, "Provided dimension must be divisible by group size"

    local_dim_size = dim_size // world_size
    rank = group.rank()
    dim_offset = rank * local_dim_size
    output = _slice_tensor_at_dim(
        input_, dim, dim_offset, dim_offset + local_dim_size
    )

    return output


def reduce_scatter_along_dim(
    input_: torch.Tensor,
    dim: int,
    group: torch.distributed.ProcessGroup,
):
    """Reduce-scatter the input tensor across model parallel group."""
    if is_trivial_process_group(group):
        return input_

    world_size = group.size()

    dim_size = list(input_.size())
    assert (
        dim_size[dim] % world_size == 0
    ), "Second dimension of the tensor should be divisible by sequence parallel size"

    dim_size[dim] = dim_size[dim] // world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=group, async_op=True
    )
    handle.wait()

    return output


def reduce(input_: torch.Tensor, group: torch.distributed.ProcessGroup, reduce_operation: str="SUM") -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if is_trivial_process_group(group):
        return input_

    # All-reduce.
    all_reduce(input_, group=group, reduce_operation=reduce_operation)

    return input_
