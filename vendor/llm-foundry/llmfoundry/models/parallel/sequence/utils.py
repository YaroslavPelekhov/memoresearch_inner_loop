"""
Sequence parallel utils.
"""

from typing import List, Optional, Union, Iterable, Any, Tuple

import torch
import torch.distributed as torch_dist
from composer.utils.dist import get_sp_group_size, get_sp_group_rank
from torch.distributed.distributed_c10d import ProcessGroup


def _split_tensor_equally(tensor: Optional[torch.Tensor], dim: int = -1) -> Optional[List[torch.Tensor]]:
    """Split sequence into chunks of equal size based on sequence parallel group size. It
    is assumed that the last dimension is the default one for sequence dimension.

    `chunk_size = seq_len // sp_group_size`

    Args:
        sequence (torch.Tensor): Sequence tensor to be splitted.
        dim (int, optional): Sequence dimension. Defaults to the last dim, which is -1.

    Returns:
        List[torch.Tensor]: List of chunks that are views of original tensor.
    """
    if tensor is None:
        return None

    assert isinstance(
        tensor, torch.Tensor
    ), f"Expected torch.Tensor as an input, got {type(tensor)}."

    sp_size = get_sp_group_size()
    assert (
        sp_size is not None
    ), "Sequence parallel is not active, global SP group is None!"

    assert tensor.shape[dim] % (sp_size) == 0, (
        "Tensor split dimension must be divisible by SP size, but got "
        f"size {tensor.shape[dim]} at provided dimension {dim} which is not "
        f"divisible by SP size {sp_size}."
    )

    return torch.chunk(tensor, sp_size, dim=dim)


def _split_tensor_zigzag(tensor: Optional[torch.Tensor], dim: int = -1) -> Optional[List[torch.Tensor]]:
    """Split sequence by zigzag pattern between sequence parallel group.
    It is assumed that the last dimension is the default one for sequence dimension.

    Sequence split into 2 * sp_group_size parts and i-th rank works with cat([parts[i], parts[sp_size - i - 1]])

    Args:
        sequence (torch.Tensor): Sequence tensor to be splitted.
        dim (int, optional): Sequence dimension. Defaults to the last dim, which is -1.

    Returns:
        List[torch.Tensor]: List of chunks that are views of original tensor.
    """
    if tensor is None:
        return None

    assert isinstance(
        tensor, torch.Tensor
    ), f"Expected torch.Tensor as an input, got {type(tensor)}."

    sp_size = get_sp_group_size()
    assert (
        sp_size is not None
    ), "Sequence parallel is not active, global SP group is None!"

    assert tensor.shape[dim] % (2 * sp_size) == 0, (
        "Tensor split dimension must be divisible by double SP size, but got "
        f"size {tensor.shape[dim]} at provided dimension {dim} which is not "
        f"divisible by double SP size {2*sp_size}."
    )

    tensor_chunks = tensor.chunk(2 * sp_size, dim=dim)
    tensor_chunks = [
        torch.cat([tensor_chunks[rank], tensor_chunks[2 * sp_size - rank - 1]], dim=dim)
        for rank in range(sp_size)
    ]

    return tensor_chunks


def get_tensors_sp_part(
    tensors: Union[Optional[Union[torch.Tensor, Union[torch.LongTensor, torch.FloatTensor]]], Iterable[torch.Tensor]],
    split_type: Optional[str] = "equal",
    dim: int = -1,
) -> List[Optional[torch.Tensor]]:
    """Get appropriate sequence parallel part of the provided tensor or iterable of tensors
    based on the `split_type` and device's SP rank. Supported split methods are:

    - "equal": split sequence into chunks of equal size based on the sequence parallel
      group size (`chunk_size = seq_len // sp_group_size`).

    Args:
        tensors (Union[torch.Tensor, Iterable[torch.Tensor]]): Sequence tensor to be splitted.
        split_type (str, optional): Split type method. Supported values are: "equal". Defaults to "equal".
        dim (int, optional): Sequence dimension. Defaults to the last dim, which is -1.

    Returns:
        List[torch.Tensor]: List of chunks that are views of original tensor.
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    if split_type == "equal" or split_type == "llama3":
        split_func = _split_tensor_equally
    elif split_type == "zigzag":
        split_func = _split_tensor_zigzag
    else:
        raise ValueError(
            f'Unknown sequence spilt type! Got {split_type}, supported values are: "equal".'
        )

    def processing_func(tensor):
        if tensor is None:
            return None
        else:
            return split_func(tensor, dim=dim)[get_sp_group_rank()]

    return list(map(processing_func, tensors))


def pad_tensor(
    tensor: torch.Tensor, pad_dims: Tuple[int], pad_value: int = 0
) -> torch.Tensor:
    return torch.nn.functional.pad(tensor, pad_dims, mode="constant", value=pad_value)
