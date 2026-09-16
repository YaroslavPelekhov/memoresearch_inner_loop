from typing import Any, Tuple

import torch
import torch.distributed as torch_dist
from torch.distributed.distributed_c10d import ProcessGroup

__all__ = ["SeqAllToAll"]


# DeepSpeed inspired func
def _single_all_to_all(
    input: torch.Tensor, gather_idx: int, scatter_idx: int, group: ProcessGroup
) -> torch.Tensor:
    """All-to-all collective operation modification that applies scattering and
    gathering for the provided dimensions and changes their size by the process
    group size.

    Input is gathered using all-to-all operation for `gather_idx` dimension and
    the `scatter_idx` dimension is reduced by provided `group` size.

    It assumes [bs, seqlen, hidden] input order and reserves 0 dimension for batch
    dimension. Setting either scatter or gather indexes to 0 will raise an error.

    Args:
        input (torch.Tensor): Input tensor to perform operation for.
        gather_idx (int): Dimension that will be gathered. Should be sequence dimension.
        scatter_idx (int): Dimension that will be scattered. Should be a hidden dimension.
        group (ProcessGroup): Process group to perform operation for.

    Returns:
        torch.Tensor: Resulting tensor after all-to-all operation with modified shape.

    Raises:
        ValueError: If either `scatter_idx` or `gather_idx` is set 0. 0 dimension is assumed
        and reserved to be batch size dimension.
    """
    if scatter_idx == gather_idx:
        raise RuntimeError(
            f"`scatter_idx` and `gather_idx` cannot be the same but got both {scatter_idx}."
        )

    if scatter_idx == 0 or gather_idx == 0:
        raise ValueError(
            "Setting `scatter_idx` or `gather_idx` to 0 is not supported as it is reserved for batch dimension."
        )

    group_world_size = torch_dist.get_world_size(group)

    if scatter_idx == 0 or gather_idx == 0:
        raise ValueError(
            "Setting `scatter_idx` or `gather_idx` to 0 is not supported as it is reserved for batch dimension."
        )

    # Permute dimension to perform all-to-all.
    # We need scatter dim at position 0 as all-to-all is performed along this dimension.
    # Put batch size dim (0) as the last one to preserve it.
    # ---
    # Set first 3 dims as [sp, gather, sp_dim], then add intermediate and end dims, batch dim last.
    # Intermediate -- dims between scatter and gather dims, end -- dims afte the last dim between scatter/gather.
    inp_dims = torch.arange(len(input.shape))
    if scatter_idx > gather_idx:
        interm_size = scatter_idx - gather_idx - 1
        end_size = len(inp_dims) - scatter_idx - 1
        dims_a2a = (
            [scatter_idx, gather_idx, scatter_idx + 1]
            + inp_dims[gather_idx + 1 : scatter_idx].tolist()
            + (inp_dims[scatter_idx + 1 :] + 1).tolist()
            + [0]
        )
    else:
        interm_size = gather_idx - scatter_idx - 1
        end_size = len(inp_dims) - gather_idx - 1
        dims_a2a = (
            [scatter_idx, gather_idx + 1, scatter_idx + 1]
            + (inp_dims[scatter_idx + 1 : gather_idx] + 1).tolist()
            + (inp_dims[gather_idx + 1 :] + 1).tolist()
            + [0]
        )

    if len(dims_a2a) != len(inp_dims) + 1:
        raise RuntimeError(
            f"Got incorrect permutations dimensions list! Expected it to have {len(inp_dims)+1} dimensions but got {len(dims_a2a)}."
        )

    # Unflatten scatter_idx splitting it into (SP size, dim / SP size), then perform permutation.
    # ---
    # [bs, gather, _interm_, scatter, _end_] --> [sp, gather, scatter dim / sp, _iterm_, _end_, bs]
    input_t = torch.permute(
        torch.unflatten(input, scatter_idx, (group_world_size, -1)), dims=dims_a2a
    ).contiguous()

    output = torch.empty_like(input_t)
    handle = torch_dist.all_to_all_single(output, input_t, group=group, async_op=True)
    handle.wait()

    # Permute dimensions back to original order.
    # Flatten first 2 dims (scatter and gather) to get appropriate values and sizes,
    # set appropriate dims for scatter and gather dims. Init with the last element as this
    # is the batch size dim, fill other approprietly.
    # ---
    # [sp, gather, scatter dim / sp, _iterm_, _end_, bs] --> [gathered, scatter dim /sp, _interm_, _end_, bs] --> [bs, gather, _interm_, scatter, _end_]
    dims_out = torch.zeros(len(inp_dims), dtype=torch.int32) + len(inp_dims) - 1
    dims_out[gather_idx] = 0
    dims_out[scatter_idx] = 1
    if scatter_idx > gather_idx:
        dims_out[gather_idx + 1 : scatter_idx] = inp_dims[2 : 2 + interm_size]
        dims_out[scatter_idx + 1 :] = inp_dims[
            2 + interm_size : 2 + interm_size + end_size
        ]
    else:
        dims_out[scatter_idx + 1 : gather_idx] = inp_dims[2 : 2 + interm_size]
        dims_out[gather_idx + 1 :] = inp_dims[
            2 + interm_size : 2 + interm_size + end_size
        ]

    dims_out = dims_out.tolist()

    output = torch.permute(output.flatten(0, 1), dims=tuple(dims_out)).contiguous()

    # Normalize singleton-dim stride metadata as .contiguous() ignores it.
    # https://discuss.pytorch.org/t/tensor-stride-not-updated-by-contiguous/4549/2
    # FA3 kernels require correct stride metadata.
    def get_reference_stride(shape: Tuple[int, ...]):
        ref_stride = []
        stride = 1
        for size in reversed(shape):
            ref_stride.append(stride)
            stride *= size
        
        return tuple(reversed(ref_stride))

    ref_stride = get_reference_stride(tuple(output.shape))
    cur_stride = tuple(output.stride())

    if cur_stride != ref_stride:
        assert output.is_contiguous()
        # strides may mismatch for singleton-dims only
        assert all(
            size == 1 or cur == ref
            for size, cur, ref in zip(output.shape, cur_stride, ref_stride)
        ), f"{output.shape=}, {cur_stride=}, {ref_stride=}"

        output = output.as_strided(
            size=output.shape,
            stride=ref_stride,
            storage_offset=output.storage_offset(),
        )

    return output


class SeqAllToAll(torch.autograd.Function):
    """Sequence parallel all-to-all autograd fuction class.

    Remembers `scatter_idx` and `gather_idx` on forward and applies reverse
    call on backward.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        gather_idx: int,
        scatter_idx: int,
        group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _single_all_to_all(
            input, gather_idx=gather_idx, scatter_idx=scatter_idx, group=group
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        return (
            SeqAllToAll.apply(
                *grad_output,
                ctx.scatter_idx,
                ctx.gather_idx,
                ctx.group,
            ),
            None,
            None,
            None,
        )
