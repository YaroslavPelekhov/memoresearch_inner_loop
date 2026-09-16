import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.distributed._symmetric_memory import  _pipelined_all_gather_and_consume
from typing import Callable, Optional, Union


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


@triton.jit
def liger_cross_entropy_fwd_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    lse_ptr,
    n_cols,
    ignore_index,
    class_start_idx,
    lse_square_scale: tl.constexpr,
    RETURN_Z_LOSS: tl.constexpr,
    SPLIT:tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes both cross entropy loss and the gradient of the input.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    lse_ptr: Pointer to tensor to store the log-sum-exp (LSE) values for backprop.
    n_cols (int): The number of columns in the input tensor.
    ignore_index (int): The index to ignore in the target.
    class_start_idx (int): Starting index for valid class range. Useful for tensor parallel when each rank only has a subset of classes.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    RETURN_Z_LOSS (int): The boolean value to decide whether storing z loss to z_loss_ptr or not. It must be 0 or 1.
    SPLIT: (bool): Flag for correct loss computation.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride


    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)
    tl.store(lse_ptr + program_id, lse)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    # tl.debug_barrier()

    # 4. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    #      = X_y - m - log d = X_y - lse
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    # loss = lse - ori_X_y
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if y == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        y -= class_start_idx
        if y >= 0 and y < n_cols:
            ori_X_y = tl.load(X_ptr + y)
            loss = (lse if not SPLIT else 0.0) - ori_X_y
        else:
            # If label is out of bounds, we set the CE loss to 0.0.
            loss = 0.0

        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0

    tl.store(loss_ptr, loss)
    if not SPLIT and RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)


@triton.jit
def liger_cross_entropy_grad_logits_bwd_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    lse_ptr,
    precomputed_grad_output_ptr,
    n_cols,
    ignore_index,
    n_non_ignore,
    class_start_idx,
    lse_square_scale: tl.constexpr,
    reduction: tl.constexpr,
    HAS_PRECOMPUTED_GRAD_OUTPUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes gradients for cross-entropy loss using precomputed LSE values.

    Implements efficient backprop through softmax using online computation.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    lse_ptr: Pointer to tensor to load the log-sum-exp (LSE) values for backprop.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    class_start_idx (int): Starting index for valid class range. Useful for tensor parallel when each rank only has a subset of classes.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (tl.constexpr): Reduction mode ('mean' or 'sum')
    BLOCK_SIZE (int): The block size for Triton operations.
    """
    program_id = tl.program_id(0).to(tl.int64)
    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        return

    grad_scale = 1.0
    if HAS_PRECOMPUTED_GRAD_OUTPUT:
        grad_scale = tl.load(precomputed_grad_output_ptr + program_id).to(tl.float32)

    # 3. load log-sum-exp
    lse = tl.load(lse_ptr + program_id)

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For 'sum' reduction, no normalization is applied:
    # dx_y = softmax(x_y) - 1
    # dx_i = softmax(x_i), for i ≠ y
    y -= class_start_idx
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)

        # softmax(x_i)
        # X_block = tl.exp(X_block - m) / d
        X_block =  tl.exp(X_block - lse)
        # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
        X_block += 2.0 * lse_square_scale * lse * X_block

        # special handle dx_y
        X_block = tl.where(X_offsets != y, X_block, X_block - 1)
        # reduction scale
        if reduction == "mean":
            X_block = X_block / n_non_ignore
            
        if HAS_PRECOMPUTED_GRAD_OUTPUT:
            X_block = X_block * grad_scale
            
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def async_tp_fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Union[torch.Tensor, None] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    reduction: str = "mean",
    return_z_loss: bool = False,
    process_group: Optional[ProcessGroup] = None,
    precomputed_grad_output: Optional[torch.Tensor] = None
):
    if target.dtype == torch.long and target.data_ptr() % 16 != 0:
        target = F.pad(target, (0, 1))[..., :-1]
        assert target.data_ptr() % 16 == 0

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank *  V

    assert target.shape == (BT * world_size,)
    if precomputed_grad_output is not None:
        assert precomputed_grad_output.shape == (BT,)

    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    # chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    chunk_size = min(triton.next_power_of_2(triton.cdiv(BT * world_size, inc_factor)), BT)  # (BT + inc_factor - 1) // inc_factor # NOTE: Empirically gives better performance
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device) if _input.requires_grad else None
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT * world_size, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT * world_size, dtype=torch.float32, device=device) if return_z_loss else None

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item() if reduction == "mean" else None

    lse = torch.empty(BT * world_size, dtype=torch.float, device=device)

    mm_out_op: torch._ops.OpOverload = torch.ops.aten.mm.out # type: ignore
    addmm_out_op: torch._ops.OpOverload = torch.ops.aten.addmm.out # type: ignore

    def _make_forward_pass_shard_consumer(
        logits_chunks_shard: tuple[torch.Tensor, ...],
        target_chunks: torch.Tensor,
        loss_1d_slice: torch.Tensor,
        z_loss_1d_slice: Union[torch.Tensor, None],
        lse_slice: torch.Tensor,
        slice_indices: torch.Tensor
    ) -> Callable[[torch.Tensor, int], None]:
        def _shard_consumer(shard: torch.Tensor, rank: int) -> None:
            if bias is not None:
                addmm_out_op(
                    bias,
                    shard, weight.T,
                    out=logits_chunks_shard[rank]
                )
            else:
                mm_out_op(shard, weight.T, out=logits_chunks_shard[rank])

            n_rows = logits_chunks_shard[rank].shape[0]

            _target_chunk = target_chunks.chunk(world_size, dim=0)[rank]
            start_slice_idx, end_slice_idx = slice_indices.chunk(world_size, dim=0)[rank]

            # Here we calculate the gradient of logits_chunk in place so we can save memory.
            liger_cross_entropy_fwd_kernel[(n_rows,)](
                X_ptr=logits_chunks_shard[rank],
                X_stride=logits_chunks_shard[rank].stride(-2),
                Y_ptr=_target_chunk,
                Y_stride=_target_chunk.stride(-1),  # always 1
                loss_ptr=loss_1d_slice[start_slice_idx: end_slice_idx],
                z_loss_ptr=z_loss_1d_slice[start_slice_idx: end_slice_idx] if return_z_loss else None,
                loss_stride=loss_1d_slice[start_slice_idx: end_slice_idx].stride(-1),  # always 1
                lse_ptr=lse_slice[start_slice_idx: end_slice_idx],
                n_cols=V,
                ignore_index=ignore_index,
                class_start_idx=class_start_idx,
                lse_square_scale=lse_square_scale,
                RETURN_Z_LOSS=return_z_loss,
                SPLIT=world_size>1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 # type: ignore
            )

        return _shard_consumer

    def _make_grad_weight_shard_consumer(
        grad_logits_chunks: tuple[torch.Tensor, ...]
    ):
        def _shard_consumer(shard: torch.Tensor, rank: int) -> None:
            # torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) → Tensor
            addmm_out_op(
                grad_weight,
                grad_logits_chunks[rank].T,
                shard, # input_chunk
                out=grad_weight
            )

        return _shard_consumer

    def _asynctp_mm_malloc(chunk_size: int):
        starts = torch.arange(0, world_size * chunk_size, chunk_size)
        ends = starts + chunk_size
        slice_indices = torch.stack([starts, ends], dim=1).flatten()

        # allocate mem for asynctp mm and reuse it on each interation
        _input_chunks_flat = torch.empty(chunk_size * world_size, H, device=device, dtype=_input.dtype)
        logits_chunks = _input_chunks_flat.new_empty(_input_chunks_flat.shape[0], weight.shape[0], dtype=weight.dtype) # [tp_size * chunk_size, V//tp_size]
        logits_chunks_shard = logits_chunks.chunk(world_size)

        return slice_indices, _input_chunks_flat, logits_chunks, logits_chunks_shard

    slice_indices, _input_chunks_flat, logits_chunks, logits_chunks_shard = _asynctp_mm_malloc(chunk_size)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        curr_chunk_size = end_idx - start_idx

        # reallocate mem and recompute slice_indices in the last iteration for asynctp mm if BT % chunk_size != 0
        if curr_chunk_size != chunk_size:
            slice_indices, _input_chunks_flat, logits_chunks, logits_chunks_shard = _asynctp_mm_malloc(curr_chunk_size)

        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        target_chunks = target.view(world_size, BT)[:, start_idx:end_idx].flatten()
        loss_1d_slice = loss_1d.view(world_size, BT)[:, start_idx:end_idx].flatten()
        z_loss_1d_slice = z_loss_1d.view(world_size, BT)[:, start_idx:end_idx].flatten() if return_z_loss else None
        lse_slice = lse.view(world_size, BT)[:, start_idx:end_idx].flatten()

        target_chunks = target_chunks.contiguous()

        consumer = _make_forward_pass_shard_consumer(
            logits_chunks_shard,
            target_chunks,
            loss_1d_slice,
            z_loss_1d_slice,
            lse_slice,
            slice_indices
        )

        _pipelined_all_gather_and_consume(
            shard=_input_chunk,
            shard_consumer=consumer,
            ag_out=_input_chunks_flat,
            group_name= process_group.group_name,
        )

        if world_size > 1:
            lse_slice_allgather = torch.empty(world_size, curr_chunk_size * world_size, dtype=lse_slice.dtype, device=device)
            handle = torch.distributed.all_gather_into_tensor(lse_slice_allgather, lse_slice, group=process_group, async_op=True)
            handle.wait()
            lse_slice = torch.logsumexp(lse_slice_allgather, dim=0)

        loss_1d.view(world_size, BT)[:, start_idx:end_idx] = loss_1d_slice.view(world_size, curr_chunk_size)
        lse.view(world_size, BT)[:, start_idx:end_idx] = lse_slice.view(world_size, curr_chunk_size)
        if return_z_loss:
            z_loss_1d.view(world_size, BT)[:, start_idx:end_idx] = z_loss_1d_slice.view(world_size, curr_chunk_size)

        if grad_input is None:
            continue

        n_rows = curr_chunk_size * world_size
        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        has_precomputed_grad_output = precomputed_grad_output is not None
        grad_output = (
            precomputed_grad_output.view(world_size, BT)[:, start_idx:end_idx] 
            if has_precomputed_grad_output 
            else None
        )
        liger_cross_entropy_grad_logits_bwd_kernel[(n_rows, )](
            X_ptr=logits_chunks,
            X_stride=logits_chunks.stride(-2),
            Y_ptr=target_chunks,
            Y_stride=target_chunks.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            lse_ptr=lse_slice,
            ignore_index=ignore_index,
            class_start_idx=class_start_idx,
            lse_square_scale=lse_square_scale,
            reduction=reduction,
            precomputed_grad_output_ptr=grad_output,
            HAS_PRECOMPUTED_GRAD_OUTPUT=has_precomputed_grad_output,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 # type: ignore
        )

        # if precomputed_grad_output is not None: # NOTE: We moved this part in liger_cross_entropy_grad_logits_bwd_kernel to fix numerical instability issues.
        #     logits_chunks *= precomputed_grad_output.view(world_size, BT)[:, start_idx:end_idx].view(-1, 1) # [chunk_size * world_size x V] * [chunk_size * world_size]

        grad_input[start_idx :end_idx] = torch.ops.symm_mem.fused_matmul_reduce_scatter( # type: ignore
            logits_chunks,
            weight,
            "sum",
            scatter_dim=0,
            group_name=process_group.group_name,
        )

        if grad_weight is not None:
            consumer = _make_grad_weight_shard_consumer(
                grad_logits_chunks = logits_chunks_shard,
            )
            _pipelined_all_gather_and_consume(
                shard=_input_chunk,
                shard_consumer=consumer,
                ag_out=_input_chunks_flat,
                group_name=process_group.group_name,
            )

        if bias is not None:
            torch.add(
                input=grad_bias, # type: ignore
                other=logits_chunks.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    if world_size > 1:
        # If there's no smoothing, if labels are in the vocab of this partition, losses contains
        # - predicted logit, and 0 otherwise.
        # If there's smoothing=0.1, for labels in the vocab of this partition, losses contains
        # -0.9 * predicted logit - 0.1 * sum logit / total_classes.
        # For labels not in the vocab of this partition, losses contains
        # -0.1 * sum logit / total_classes.

        handle_loss = torch.distributed.all_reduce(
            loss_1d, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
        )
        handle_loss.wait() # type: ignore

        # After the allreduce, if there's no smoothing, the total losses are - predicted_logit,
        # we just have to add the (global) lse.
        # If there's smoothing=0.1, the total losses are
        # -0.9 * predicted_logit - 0.1 * sum logit / total_classes.
        # Again, we just have to add the (global) lse.

        loss_1d += lse

        if lse_square_scale != 0.0:
            z_loss_1d = lse_square_scale * lse.square()
            z_loss_1d.masked_fill_(target == ignore_index, 0.0)
            loss_1d += z_loss_1d

        loss_1d.masked_fill_(target == ignore_index, 0.0)

    if reduction == "mean":
        loss = loss_1d.sum() / (target != ignore_index).sum()
        z_loss = z_loss_1d.sum() / (target != ignore_index).sum() if return_z_loss else None
    elif reduction == "sum":
        loss = loss_1d.sum()
        z_loss = z_loss_1d.sum() if return_z_loss else None
    else:
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None

    return loss, z_loss, grad_input, grad_weight, grad_bias

def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Union[torch.Tensor, None] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    reduction: str = "mean",
    return_z_loss: bool = False,
    process_group: Optional[ProcessGroup] = None,
    precomputed_grad_output: Optional[torch.Tensor] = None
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"

    if target.dtype == torch.long and target.data_ptr() % 16 != 0:
        target = F.pad(target, (0, 1))[..., :-1]
        assert target.data_ptr() % 16 == 0

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    assert target.shape == (BT,)
    if precomputed_grad_output is not None:
        assert precomputed_grad_output.shape == (BT,)

    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank *  V

    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device) if _input.requires_grad else None
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_z_loss else None

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item() if reduction == "mean" else None

    lse = torch.empty(BT, dtype=torch.float, device=device)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = F.linear(_input_chunk, weight, bias) # chunk_size x V

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        lse_slice = lse[start_idx:end_idx]

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_cross_entropy_fwd_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            lse_ptr=lse_slice,
            n_cols=V,
            ignore_index=ignore_index,
            class_start_idx=class_start_idx,
            lse_square_scale=lse_square_scale,
            RETURN_Z_LOSS=return_z_loss,
            SPLIT=world_size>1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 # type: ignore
        )

        if world_size > 1:
            curr_chunk_size = end_idx - start_idx
            lse_slice_allgather = torch.empty(world_size, curr_chunk_size, dtype=lse_slice.dtype, device=device)
            handle = torch.distributed.all_gather_into_tensor(lse_slice_allgather, lse_slice, group=process_group, async_op=True)
            handle.wait()
            lse_slice = torch.logsumexp(lse_slice_allgather, dim=0)

        loss_1d[start_idx:end_idx] = loss_1d_slice
        lse[start_idx:end_idx] = lse_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice

        if grad_input is None:
            continue

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        has_precomputed_grad_output = precomputed_grad_output is not None
        grad_output = (
            precomputed_grad_output[start_idx:end_idx] 
            if has_precomputed_grad_output 
            else None
        )
        liger_cross_entropy_grad_logits_bwd_kernel[(n_rows, )](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            lse_ptr=lse_slice,
            ignore_index=ignore_index,
            class_start_idx=class_start_idx,
            lse_square_scale=lse_square_scale,
            reduction=reduction,
            precomputed_grad_output_ptr=grad_output,
            HAS_PRECOMPUTED_GRAD_OUTPUT=has_precomputed_grad_output,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 # type: ignore
        )

        grad_logits_chunk = logits_chunk  # chunk_size x V
        # if precomputed_grad_output is not None: # NOTE: We moved this part in liger_cross_entropy_grad_logits_bwd_kernel to fix numerical instability issues.
        #     grad_logits_chunk *= precomputed_grad_output[start_idx: end_idx].unsqueeze(1) # [chunk_size x V] * [chunk_size]

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=grad_logits_chunk.t().to(
                    _input_chunk.dtype
                ),  # In an autocast scenario without bias, differing logits_chunk data types will cause an addmm operation error.
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=grad_bias, # type: ignore
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    if world_size > 1:
        # If there's no smoothing, if labels are in the vocab of this partition, losses contains
        # - predicted logit, and 0 otherwise.
        # If there's smoothing=0.1, for labels in the vocab of this partition, losses contains
        # -0.9 * predicted logit - 0.1 * sum logit / total_classes.
        # For labels not in the vocab of this partition, losses contains
        # -0.1 * sum logit / total_classes.

        handle_loss = torch.distributed.all_reduce(
            loss_1d, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
        )
        handle_loss.wait()

        if grad_input is not None:
            handle_grad_input = torch.distributed.all_reduce(
                grad_input, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            handle_grad_input.wait()

        # After the allreduce, if there's no smoothing, the total losses are - predicted_logit,
        # we just have to add the (global) lse.
        # If there's smoothing=0.1, the total losses are
        # -0.9 * predicted_logit - 0.1 * sum logit / total_classes.
        # Again, we just have to add the (global) lse.

        loss_1d += lse

        if lse_square_scale != 0.0:
            z_loss_1d = lse_square_scale * lse.square()
            z_loss_1d.masked_fill_(target == ignore_index, 0.0)
            loss_1d += z_loss_1d

        loss_1d.masked_fill_(target == ignore_index, 0.0)

    if reduction == "mean":
        loss = loss_1d.sum() / (target != ignore_index).sum()
        z_loss = z_loss_1d.sum() / (target != ignore_index).sum() if return_z_loss else None
    elif reduction == "sum":
        loss = loss_1d.sum()
        z_loss = z_loss_1d.sum() if return_z_loss else None
    else:
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None

    return loss, z_loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias, precomputed_grad_output):
    """
    WARNING: The current backward implementation for fused_linear_cross_entropy has a fundamental limitation:
    - We multiply ALL gradients (grad_input, grad_weight, grad_bias) ONLY by the ZERO-TH element of grad_output
    - This produces correct results ONLY when:
        1) grad_output is a scalar (single element) → reduction="mean" or "sum"
        2) All elements in grad_output are identical → reduction="none" (but requires identical values)

        For example in this scenario with reduction == "none" gradrients computations will OK
        because here uniform identical scaling for all elements:
        loss = loss.view(labels.shape[0], -1)  # [BS, ...]
        loss = loss.sum(-1)                    # [BS]
        loss = sp_group_size * loss / num_tokens  # [BS] (all elements scaled equally)
        loss = loss.mean()                     # scalar

        Backpropagation through this graph produces grad_output with:
         - All elements identical (from uniform scaling and mean reduction)
         - Grad Output Values = (sp_group_size / num_tokens) * (1/BS)

    Why this limitation?
    - Gradient shape incompatibility:
        grad_output: [BS] (batch size x seq len)
        grad_input: [BS, H] (batch size x seq len, hidden size)
        grad_weight: [V, H] (vocab size, hidden size)
        grad_bias:   [V] (vocab size)
    - Cannot directly multiply grad_output [BS] tensor with grad_weight [V, H] and grad_bias [V]
      because Liger computes gradients in forward pass

    Critical issue:
    If grad_output contains varying values (from custom loss functions
    or complex non-linear computational graphs), this will cause:
        1) Incorrect gradients for weight/bias
        2) Silent errors that are difficult to debug

    Related discussions:
        https://github.com/linkedin/Liger-Kernel/issues/678
        https://github.com/linkedin/Liger-Kernel/pull/680
        https://github.com/linkedin/Liger-Kernel/issues/678#issuecomment-2826754937

    REQUIREMENT: grad_output must be either a scalar tensor or have all identical values!
    """
    if precomputed_grad_output is not None:
        return grad_input, grad_weight, grad_bias

    if grad_output.numel() > 1:
        if not torch.all(grad_output == grad_output.reshape(-1)[0]):
            raise AssertionError("grad_output must be a single-element tensor or have all elements equal")

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        if grad_input is None:
            return grad_input, grad_weight, grad_bias

        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 # type: ignore
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 # type: ignore
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 # type: ignore
            )
    return grad_input, grad_weight, grad_bias


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int =-100,
        reduction: str ="mean",
        lse_square_scale: float = 0.0,
        process_group: Optional[ProcessGroup] = None,
        enable_async_tp: bool = False,
        return_z_loss: bool = False,
        precomputed_grad_output: Optional[torch.Tensor] = None
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ignore_index: the index to ignore in the target
        reduction: reduction to apply
        precomputed_grad_output : precomputed grad_output that can be passed through forward instead of backward,
        see fused_linear_cross_entropy_backward docstring warning. Used by MTP
        """
        if enable_async_tp:
            assert process_group is not None, "Parameter process_group cannot be None when using asynctp."
            forward_method = async_tp_fused_linear_cross_entropy_forward
        else:
            forward_method = fused_linear_cross_entropy_forward

        loss, z_loss, grad_input, grad_weight, grad_bias = forward_method(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=ignore_index,
            reduction=reduction,
            lse_square_scale=lse_square_scale,
            process_group=process_group,
            return_z_loss=return_z_loss,
            precomputed_grad_output=precomputed_grad_output
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input,
            grad_weight,
            grad_bias,
            precomputed_grad_output
        )
        ctx.return_z_loss = return_z_loss

        return loss, z_loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging

        (grad_input, grad_weight, grad_bias, precomputed_grad_output) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias, precomputed_grad_output
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        lse_square_scale: float = 0.0,
        enable_async_tp: bool = False,
        process_group: Optional[ProcessGroup] = None,
        return_z_loss: bool = False,
        precomputed_grad_output: Optional[torch.Tensor] = None
    ):
        super().__init__()
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.lse_square_scale = lse_square_scale
        self.enable_async_tp = enable_async_tp
        self.process_group = process_group
        self.return_z_loss = return_z_loss
        self.precomputed_grad_output = precomputed_grad_output

        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        assert (precomputed_grad_output is None
            or world_size == 1), \
            "precomputed_grad_output was not tested with TP and AsyncTP!"

    def forward(self,
            _input: torch.Tensor,
            lin_weight: torch.Tensor,
            target: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ):

        loss, z_loss = FusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.reduction,
            self.lse_square_scale,
            self.process_group,
            self.enable_async_tp,
            self.return_z_loss,
            self.precomputed_grad_output
        )

        if not self.return_z_loss:
            return loss

        return loss, z_loss