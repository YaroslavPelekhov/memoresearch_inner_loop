from typing import List

import torch
import torch.distributed
from composer.utils.dist import get_sp_group, get_tp_sp_group
from llmfoundry.models.parallel.utils import split_along_dim, gather_along_dim, reduce_scatter_along_dim, all_reduce_with_grads


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return gather_along_dim(input_, 1, get_sp_group())

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return gather_along_dim(input_, 1, get_sp_group())

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return reduce_scatter_along_dim(grad_output, 1, get_sp_group()), None
        else:
            return split_along_dim(grad_output, 1, get_sp_group()), None # HOTFIX


class _GatherFromSequenceParallelRegionOnBatchSizeDimension(torch.autograd.Function):
    """Gather the input from sequence parallel region on batch size dimension and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return gather_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def forward(ctx, input_):
        return gather_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def backward(ctx, grad_output):
        return split_along_dim(grad_output, 0, get_sp_group())


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return reduce_scatter_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def forward(ctx, input_):
        return reduce_scatter_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_dim(grad_output, 0, get_sp_group())


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return split_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def forward(ctx, input_):
        return split_along_dim(input_, 0, get_sp_group())

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_dim(grad_output, 0, get_sp_group())


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def gather_from_sequence_parallel_region_on_batch_size_dimension(input_):
    return _GatherFromSequenceParallelRegionOnBatchSizeDimension.apply(input_)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)

def all_reduce_from_sequence_parallel_region(input_):
    return all_reduce_with_grads(input_, get_sp_group())

def all_reduce_from_tensor_sequence_parallel_region(input_):
    return all_reduce_with_grads(input_, get_tp_sp_group())