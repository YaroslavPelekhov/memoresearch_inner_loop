# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Sequence, Optional, Union, Callable
import torch

from composer.utils.dist import get_tp_group_rank, get_tp_group_size, get_tp_group, get_ep_group

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
}


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim):
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        assert (
            global_vocab_size % world_size == 0
        ), f"global_vocab_size = {global_vocab_size} is not divisible by world_size = {world_size}"
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


# Previously called `_initialize_affine_weight_cpu`,
# func from Megatron
def _initialize_parallel_weight(
    weight: torch.Tensor,
    input_size: int,
    output_size: int,
    partition_dim: int,
    per_partition_size: int,
    init_method: Callable,
    process_group: torch.distributed.ProcessGroup,
    gain: Optional[float] = None,
    return_master_weight: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_master_weight: bool = True,
    init_type: str = "giga",
):
    """Initialize affine weight for model parallel.

    When ``use_master_weight=False`` (default) each rank initialises only its
    own partition directly — no full master weight is ever allocated.  This is
    the memory-efficient path required for very large models.

    When ``use_master_weight=True`` the original behaviour is preserved: the
    full ``[output_size, input_size]`` master weight is materialised on every
    rank, the ``init_method`` is called on it, and the relevant shard is copied
    into ``weight``.  Use this only when you need a deterministic, globally
    consistent initialization (e.g. for debugging or small models).
    ``return_master_weight`` is only meaningful in this mode; in the
    memory-efficient path it always returns ``None``.
    """

    assert weight.shape[partition_dim] == per_partition_size, (
        f"Shape mismatch: weight.shape: {weight.shape}, " +
        f"partition_dim: {partition_dim}, per_partition_size: {per_partition_size}"
    )
    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, dim=partition_dim)

    if use_master_weight:
        # Allocate the full master weight and initialise it, then copy the
        # shard that belongs to this rank.  Memory cost: output_size * input_size
        # elements on *every* rank — avoid for large models.
        master_weight = torch.empty(
            output_size,
            input_size,
            dtype=None,
            requires_grad=False,
            device=device,
        )
        if gain is not None:
            init_method(master_weight, gain=gain)
        else:
            init_method(master_weight)

        weight_list = torch.split(master_weight, per_partition_size, dim=partition_dim)
        process_group_rank = process_group.rank()
        process_group_size = process_group.size()
        rank_weight_list = weight_list[process_group_rank::process_group_size]

        with torch.no_grad():
            torch.cat(rank_weight_list, dim=partition_dim, out=weight)

        if return_master_weight:
            return master_weight

        return None

    # Memory-efficient path: initialise only the local partition.
    # The init_method receives a partition-sized tensor; callers must ensure
    # that the method accounts for the global fan-in / fan-out if scaling
    # matters (e.g. pass a pre-configured partial that uses input_size /
    # output_size rather than the shard dimensions).
    #
    # Seed strategy: all ranks share the same global RNG state, so we must
    # differentiate them.  We draw a base_seed from the global RNG (identical
    # on every rank, but different between successive calls → distinct layers)
    # and offset it by the process-group rank so each rank gets a unique
    # stream.  The init itself runs inside fork_rng to avoid perturbing the
    # global RNG beyond the single randint we consumed.
    if init_type != "deepseek":
        raise ValueError(f"use_master_weight = False is supported only for init_type = deepseek, but got: {init_type}")
    base_seed = torch.randint(0, 2**63 - 1, (1,), device='cpu').item()
    rank_seed = base_seed + process_group.rank()

    devices = [weight.device] if weight.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(rank_seed)
        with torch.no_grad():
            if gain is not None:
                init_method(weight, gain=gain)
            else:
                init_method(weight)

    return None


def _initialize_tp_weight(
    weight: torch.Tensor,
    input_size: int,
    output_size: int,
    partition_dim: int,
    per_partition_size: int,
    init_method: Callable,
    return_master_weight: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_master_weight: bool = True,
    init_type: str = "giga",
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, dim=partition_dim)

    return _initialize_parallel_weight(
        weight,
        input_size,
        output_size,
        partition_dim,
        per_partition_size,
        init_method,
        get_tp_group(),
        return_master_weight=return_master_weight,
        device=device,
        use_master_weight=use_master_weight,
        init_type=init_type,
    )


def _initialize_ep_weight(
    weight: torch.Tensor,
    input_size: int,
    output_size: int,
    partition_dim: int,
    per_partition_size: int,
    init_method: Callable,
    gain: Optional[float] = None,
    return_master_weight: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_master_weight: bool = True,
    init_type: str = "giga",
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    return _initialize_parallel_weight(
        weight,
        input_size,
        output_size,
        partition_dim,
        per_partition_size,
        init_method,
        get_ep_group(),
        gain=gain,
        return_master_weight=return_master_weight,
        device=device,
        use_master_weight=use_master_weight,
        init_type=init_type,
    )
