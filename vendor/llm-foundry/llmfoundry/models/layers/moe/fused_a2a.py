import typing as tp

from llmfoundry.models.ops.float8.triton_kernels import row_quant_fn, make_float8_blockwise_qtensor_fn
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True

except ImportError:
    HAVE_DEEP_EP = False

from llmfoundry.models.ops.sum_kernel import sum_2shape_bf16_grads

import os
import torch

_buffer = []

from composer.utils.dist import get_ep_group

# Reserve `idx=0` for the baseline `fused_dispatch` / `fused_combine`
# The overlap path uses a dedicated buffer per chunk to avoid interfering with
# other DeepEP callers in the same process.
_SCMOE_BUFFER_BASE = int(os.environ.get("SCMOE_BUFFER_BASE", "2"))
_SCMOE_DIFFERENT_BUFFERS = False


def _scmoe_buffer_idx(local_idx: int) -> int:
    if _SCMOE_DIFFERENT_BUFFERS:
        return _SCMOE_BUFFER_BASE + local_idx
    return _SCMOE_BUFFER_BASE


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int, idx: int = 0):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    while len(_buffer) <= idx:
        _buffer.append(None)
    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer[idx] is None
        or _buffer[idx].group != group
        or _buffer[idx].num_nvl_bytes < num_nvl_bytes
        or _buffer[idx].num_rdma_bytes < num_rdma_bytes
    ):
        _buffer[idx] = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer[idx]


def _quantize_expert_weights(
    weight_quantizer_handles: tp.Optional[tp.Tuple],
) -> tp.Optional[tp.Tuple]:
    if weight_quantizer_handles is None:
        return None
    with torch.no_grad():
        fn1, fn2, exp = weight_quantizer_handles
        w1 = exp.weight1
        w2 = exp.weight2
        shape_weight1 = (exp.num_local_experts, exp.intermediate_size * 2, exp.hidden_size)
        w1_q = fn1(w1.view(shape_weight1))
        shape_weight2 = (exp.num_local_experts, exp.hidden_size, exp.intermediate_size)
        w2_q = fn2(w2.view(shape_weight2))
        weights_quantized = (w1_q, w2_q)
    return weights_quantized



def _prepare_topk_weights(topk_weights: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
    if topk_weights is None:
        return None
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        token_indices,
        token_probs,
        num_experts,
        group,
        is_float8_dispatch: bool = False,
        weights_quantize_handles: tp.Optional[tp.Tuple] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        """Forward pass of fused dispatch."""

        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())

        if is_float8_dispatch:
            if isinstance(x, Float8BlockwiseQTensor):
                # Already quantized at checkpoint boundary — extract raw data tuple
                x = (x._rowwise_data, x._rowwise_scale_inv)
            else:
                # Standard path: quantize bf16 → float8
                x = row_quant_fn(x)
        else:
            x = x.contiguous()

        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # async weight quantization overlapped with dispatch
        weights_quantized = _quantize_expert_weights(weights_quantize_handles)

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        if is_float8_dispatch:
            recv_x = make_float8_blockwise_qtensor_fn(
                rowwise_data=recv_x[0], rowwise_scale_inv=recv_x[1])

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert_cpu = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x,
                recv_token_indices,
                recv_token_probs,
                tokens_per_expert_cpu,
                num_recv_tokens_per_expert_list,
                handle,
                weights_quantized)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_tokens_per_expert_list,
        grad_handle,
        grad_weights_quantized,
    ):
        """Backward pass of fused dispatch."""
        assert grad_output.dtype in [torch.bfloat16, torch.float32], (
            "Only bfloat16 and float32 are supported for MoE training!"
        )

        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        grad_output = grad_output.contiguous()
        topk_weights = _prepare_topk_weights(grad_token_probs)
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output,
            handle,
            topk_weights=topk_weights,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False, is_float8_training: bool = False):
        """Forward pass of fused combine."""
        x = x.contiguous()
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.is_float8_training = is_float8_training
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        # NOTE (fedorovgv): IMPORTRANT - make all manipulations before making event
        if ctx.is_float8_training:
            grad_output = row_quant_fn(grad_output.contiguous())
            grad_output = (grad_output[0].contiguous().view(torch.uint8), grad_output[1].contiguous())
        else:
            grad_output = grad_output.contiguous()

        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())

        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output,
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()

        if ctx.is_float8_training:
            grad_x = make_float8_blockwise_qtensor_fn(
                rowwise_data=grad_x[0], rowwise_scale_inv=grad_x[1])

        return grad_x, None, None, None, None, None

class DispatchSc:
    @staticmethod
    def forward(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        buffer_idx,
        allocate_on_comm_stream=False,
        is_float8_dispatch: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
        previous_event: tp.Optional[EventOverlap] = None,
    ):
        async_finish = True
        if is_float8_dispatch:
            x = row_quant_fn(x)
        else:
            x = x.contiguous()
        token_indices = token_indices.contiguous()
        token_probs = _prepare_topk_weights(token_probs)
        if previous_event is None and async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x), buffer_idx)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        weights_quantized = _quantize_expert_weights(weight_quantizer_handles)
        if is_float8_dispatch:
            recv_x = make_float8_blockwise_qtensor_fn(
                rowwise_data=recv_x[0],
                rowwise_scale_inv=recv_x[1],
            )

        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            handle,
            after_event_overlap,
            weights_quantized,
        )

    @staticmethod
    def backward(grad_output, grad_token_probs, handle, group, buffer_idx, allocate_on_comm_stream=False):
        async_finish = True
        grad_output = grad_output.contiguous()
        topk_weights = _prepare_topk_weights(grad_token_probs)
        previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(grad_output), buffer_idx)

        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output,
            handle,
            topk_weights=topk_weights,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return grad_x, grad_token_probs, after_event


class CombineSc:
    @staticmethod
    def forward(x, group, handle, buffer_idx, allocate_on_comm_stream=False):
        async_finish = True
        x = x.contiguous()
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x), buffer_idx)
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return combined_x, after_event

    @staticmethod
    def backward(
        grad_output,
        group,
        handle,
        buffer_idx,
        allocate_on_comm_stream,
        is_float8_training: bool = False,
        previous_event: tp.Optional[EventOverlap] = None,
    ):
        async_finish = True
        if is_float8_training:
            grad_output = row_quant_fn(grad_output.contiguous())
            grad_output = (
                grad_output[0].contiguous().view(torch.uint8),
                grad_output[1].contiguous(),
            )
        else:
            grad_output = grad_output.contiguous()
        if previous_event is None and async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(grad_output), buffer_idx)
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output,
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if is_float8_training:
            grad_x = make_float8_blockwise_qtensor_fn(
                rowwise_data=grad_x[0],
                rowwise_scale_inv=grad_x[1],
            )
        return grad_x, after_event


def _merge_chunk_weight_grads(
    grad0: tp.Optional[torch.Tensor],
    grad1: tp.Optional[torch.Tensor],
    ref_weight: torch.Tensor,
) -> torch.Tensor:
    if grad0 is None and grad1 is None:
        return torch.zeros_like(ref_weight)
    if grad0 is None:
        return grad1
    if grad1 is None:
        return grad0
    if grad0.dtype == torch.bfloat16 and grad1.dtype == torch.bfloat16:
        sum_2shape_bf16_grads(grad0, grad1)
        return grad0
    return grad0 + grad1


def _launch_scmoe_recompute_dispatch(
    hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group,
    buffer_idx: int,
    allocate_on_comm_stream: bool,
    is_float8_training: bool,
    previous_event: tp.Optional[EventOverlap] = None,
):
    return DispatchSc.forward(
        hidden_states.detach(),
        token_indices.detach(),
        token_probs.detach(),
        num_experts=num_experts,
        group=group,
        buffer_idx=buffer_idx,
        allocate_on_comm_stream=allocate_on_comm_stream or previous_event is not None,
        is_float8_dispatch=is_float8_training,
        weight_quantizer_handles=None,
        previous_event=previous_event,
    )


def _run_scmoe_recompute_experts(
    experts,
    recv_x,
    recv_token_indices,
    recv_token_probs,
    tokens_per_expert,
    dispatcher_ctx,
    is_float8_training: bool,
    weights_quantized: tp.Optional[tp.Tuple],
):
    with torch.enable_grad():
        recv_x = recv_x.detach().requires_grad_(True)
        recv_token_probs = recv_token_probs.detach().requires_grad_(True)
        recv_order_hidden = experts(
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            dispatcher_ctx,
            is_float8_training=is_float8_training,
            weights_quantized=weights_quantized,
        )
    return recv_x, recv_token_probs, recv_order_hidden


class ScMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        token_dispatcher,
        experts,
        allocate_on_comm_stream,
        is_float8_training,
        save_dispatch_for_backward,
        weight_quantizer_handles,
        weight1,
        weight2,
        *argv_chunks
    ):
        assert len(argv_chunks) % 5 == 0, "argv_chunks must be 5*Tensors/objs per chunk"
        num_chunks = len(argv_chunks) // 5
        assert num_chunks == 2, "ScMoE overlap currently supports exactly 2 chunks"
        group = get_ep_group()

        chunks = []
        for i in range(num_chunks):
            b = 5 * i
            chunks.append({
                'hidden_states': argv_chunks[b + 0],
                'routing_map':   argv_chunks[b + 1],
                'token_indices': argv_chunks[b + 2],
                'token_probs':   argv_chunks[b + 3],
                'ctx':           argv_chunks[b + 4],
            })
        recv_x              = [None] * num_chunks
        recv_token_indices  = [None] * num_chunks
        recv_token_probs    = [None] * num_chunks
        tokens_per_expert   = [None] * num_chunks
        after_event_dispatch = [None] * num_chunks
        after_event_combine = [None] * num_chunks
        recv_order_hidden   = [None] * num_chunks
        combined_x          = [None] * num_chunks
        handles             = [None] * num_chunks
        weights_quantized   = None

        idx = 0
        (recv_x[idx],
            recv_token_indices[idx],
            recv_token_probs[idx],
            tokens_per_expert[idx],
            chunks[idx]['ctx'].deepep_handle,
            after_event_dispatch[idx],
            weights_quantized) = DispatchSc.forward(
            chunks[idx]['hidden_states'],
            chunks[idx]['token_indices'],
            chunks[idx]['token_probs'],
            num_experts=token_dispatcher.num_experts,
            group=group,
            buffer_idx=_scmoe_buffer_idx(idx),
            allocate_on_comm_stream=allocate_on_comm_stream,
            is_float8_dispatch=is_float8_training,
            weight_quantizer_handles=weight_quantizer_handles,
        )
        chunks[idx]['ctx'].deepep_recv_hidden_shape = recv_x[idx].shape
        handles[idx] = chunks[idx]['ctx'].deepep_handle

        idx = 1
        (recv_x[idx],
            recv_token_indices[idx],
            recv_token_probs[idx],
            tokens_per_expert[idx],
            chunks[idx]['ctx'].deepep_handle,
            after_event_dispatch[idx],
            _) = DispatchSc.forward(
            chunks[idx]['hidden_states'],
            chunks[idx]['token_indices'],
            chunks[idx]['token_probs'],
            num_experts=token_dispatcher.num_experts,
            group=group,
            buffer_idx=_scmoe_buffer_idx(idx),
            allocate_on_comm_stream=allocate_on_comm_stream,
            is_float8_dispatch=is_float8_training,
        )
        chunks[idx]['ctx'].deepep_recv_hidden_shape = recv_x[idx].shape
        handles[idx] = chunks[idx]['ctx'].deepep_handle

        after_event_dispatch[0].current_stream_wait()
        idx = 0
        recv_order_hidden[idx] = experts(
            recv_x[idx],
            recv_token_indices[idx],
            recv_token_probs[idx],
            tokens_per_expert[idx],
            chunks[idx]['ctx'],
            is_float8_training=is_float8_training,
            weights_quantized=weights_quantized,
        )
        idx = 1
        after_event_dispatch[idx].current_stream_wait()

        idx = 0
        combined_x[idx], after_event_combine[idx] = CombineSc.forward(
            recv_order_hidden[idx],
            group,
            chunks[idx]['ctx'].deepep_handle,
            buffer_idx=_scmoe_buffer_idx(idx),
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        idx = 1
        recv_order_hidden[idx] = experts(
            recv_x[idx],
            recv_token_indices[idx],
            recv_token_probs[idx],
            tokens_per_expert[idx],
            chunks[idx]['ctx'],
            is_float8_training=is_float8_training,
            weights_quantized=weights_quantized,
        )
        after_event_combine[0].current_stream_wait()
        combined_x[idx], after_event_combine[idx] = CombineSc.forward(
            recv_order_hidden[idx],
            group,
            chunks[idx]['ctx'].deepep_handle,
            buffer_idx=_scmoe_buffer_idx(idx),
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        after_event_combine[1].current_stream_wait()
        ctx.group = group
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.is_float8_training = is_float8_training
        ctx.num_chunks = num_chunks
        ctx.handle_list = handles
        ctx.experts = experts
        ctx.num_experts = token_dispatcher.num_experts
        ctx.weight_quantizer_handles = weight_quantizer_handles
        ctx.weight1 = weight1
        ctx.weight2 = weight2
        ctx.dispatcher_class = type(chunks[idx]['ctx'])
        ctx.save_dispatch_for_backward = save_dispatch_for_backward

        flat_saved = []
        if save_dispatch_for_backward:
            flat_saved.extend([
                recv_x[0],
                recv_token_indices[0],
                recv_token_probs[0],
                tokens_per_expert[0],
                chunks[1]["hidden_states"],
                chunks[1]["token_indices"],
                chunks[1]["token_probs"],
            ])
        else:
            for i in range(num_chunks):
                flat_saved.extend([
                    chunks[i]["hidden_states"],
                    chunks[i]["token_indices"],
                    chunks[i]["token_probs"],
                ])
        ctx.save_for_backward(*flat_saved)

        return tuple(combined_x)

    @staticmethod
    def backward(ctx, *grad_out_chunks):
        num_chunks = ctx.num_chunks
        assert len(grad_out_chunks) == num_chunks

        group = ctx.group
        allocate_on_comm_stream = ctx.allocate_on_comm_stream
        handles = ctx.handle_list
        is_float8_training = ctx.is_float8_training
        save_dispatch_for_backward = ctx.save_dispatch_for_backward

        weight1 = ctx.weight1
        weight2 = ctx.weight2
        experts = ctx.experts
        num_experts = ctx.num_experts

        saved = ctx.saved_tensors

        if save_dispatch_for_backward:
            assert len(saved) == 7
            saved_recv_x0 = saved[0]
            saved_recv_token_indices0 = saved[1]
            saved_recv_token_probs0 = saved[2]
            saved_tokens_per_expert0 = saved[3]
            hidden_states_1 = saved[4]
            token_indices_1 = saved[5]
            token_probs_1 = saved[6]
        else:
            assert len(saved) == 3 * num_chunks
            hidden_states, token_indices, token_probs = [], [], []
            for i in range(num_chunks):
                b = 3 * i
                hidden_states.append(saved[b + 0])
                token_indices.append(saved[b + 1])
                token_probs.append(saved[b + 2])

        grads_out = [None, None, None, None, None, None, None, None]
        idx_w1, idx_w2 = 6, 7
        after_event_dispatch = [None] * num_chunks
        grads_weight1 = [None] * num_chunks
        grads_weight2 = [None] * num_chunks
        grad_recv_order_hidden_0, after_event_combine_0 = CombineSc.backward(
            grad_out_chunks[0], group, handles[0],
            buffer_idx=_scmoe_buffer_idx(0),
            allocate_on_comm_stream=allocate_on_comm_stream,
            is_float8_training=is_float8_training,
        )
        weights_quantized = _quantize_expert_weights(ctx.weight_quantizer_handles)

        if save_dispatch_for_backward:
            recompute_recv_x0 = saved_recv_x0
            recompute_token_indices0 = saved_recv_token_indices0
            recompute_recv_probs0 = saved_recv_token_probs0
            recompute_tokens_per_expert0 = saved_tokens_per_expert0
            recompute_event0 = after_event_combine_0
        else:
            rd0 = _launch_scmoe_recompute_dispatch(
                hidden_states[0], token_indices[0], token_probs[0],
                num_experts=num_experts, group=group,
                buffer_idx=_scmoe_buffer_idx(0),
                allocate_on_comm_stream=allocate_on_comm_stream,
                is_float8_training=is_float8_training,
            )
            (recompute_recv_x0, recompute_token_indices0,
             recompute_recv_probs0, recompute_tokens_per_expert0,
             _, recompute_event0, _) = rd0

        hs1 = hidden_states_1 if save_dispatch_for_backward else hidden_states[1]
        ti1 = token_indices_1 if save_dispatch_for_backward else token_indices[1]
        tp1 = token_probs_1 if save_dispatch_for_backward else token_probs[1]
        rd1 = _launch_scmoe_recompute_dispatch(
            hs1, ti1, tp1,
            num_experts=num_experts, group=group,
            buffer_idx=_scmoe_buffer_idx(1),
            allocate_on_comm_stream=allocate_on_comm_stream,
            is_float8_training=is_float8_training,
        )
        (recompute_recv_x1, recompute_token_indices1,
         recompute_recv_probs1, recompute_tokens_per_expert1,
         _, recompute_event1, _) = rd1

        grad_recv_order_hidden_1, after_event_combine_1 = CombineSc.backward(
            grad_out_chunks[1], group, handles[1],
            buffer_idx=_scmoe_buffer_idx(1),
            allocate_on_comm_stream=allocate_on_comm_stream,
            is_float8_training=is_float8_training,
        )

        after_event_combine_0.current_stream_wait()
        recompute_event0.current_stream_wait()

        recompute_ctx0 = ctx.dispatcher_class()
        recompute_ctx0.deepep_recv_hidden_shape = recompute_recv_x0.shape
        (
            recompute_recv_x0,
            recompute_recv_probs0,
            recompute_recv_order_hidden0,
        ) = _run_scmoe_recompute_experts(
            experts, recompute_recv_x0, recompute_token_indices0,
            recompute_recv_probs0, recompute_tokens_per_expert0,
            recompute_ctx0,
            is_float8_training=is_float8_training,
            weights_quantized=weights_quantized,
        )
        grad_recv_x0, grad_recv_probs0, grads_weight1[0], grads_weight2[0] = (
            torch.autograd.grad(
                outputs=recompute_recv_order_hidden0,
                inputs=(recompute_recv_x0, recompute_recv_probs0, weight1, weight2),
                grad_outputs=grad_recv_order_hidden_0,
                allow_unused=True, retain_graph=False,
                create_graph=False, materialize_grads=True,
            )
        )

        dx0, dtoken_probs0, after_event_dispatch[0] = DispatchSc.backward(
            grad_recv_x0, grad_recv_probs0, handles[0], group,
            buffer_idx=_scmoe_buffer_idx(0),
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        grads_out.extend([dx0, None, None, dtoken_probs0, None])

        after_event_combine_1.current_stream_wait()
        recompute_event1.current_stream_wait()

        recompute_ctx1 = ctx.dispatcher_class()
        recompute_ctx1.deepep_recv_hidden_shape = recompute_recv_x1.shape
        (
            recompute_recv_x1,
            recompute_recv_probs1,
            recompute_recv_order_hidden1,
        ) = _run_scmoe_recompute_experts(
            experts, recompute_recv_x1, recompute_token_indices1,
            recompute_recv_probs1, recompute_tokens_per_expert1,
            recompute_ctx1,
            is_float8_training=is_float8_training,
            weights_quantized=weights_quantized,
        )
        grad_recv_x1, grad_recv_probs1, grads_weight1[1], grads_weight2[1] = (
            torch.autograd.grad(
                outputs=recompute_recv_order_hidden1,
                inputs=(recompute_recv_x1, recompute_recv_probs1, weight1, weight2),
                grad_outputs=grad_recv_order_hidden_1,
                allow_unused=True, retain_graph=False,
                create_graph=False, materialize_grads=True,
            )
        )
        dx1, dtoken_probs1, after_event_dispatch[1] = DispatchSc.backward(
            grad_recv_x1, grad_recv_probs1, handles[1], group,
            buffer_idx=_scmoe_buffer_idx(1),
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        grads_out.extend([dx1, None, None, dtoken_probs1, None])

        grads_out[idx_w1] = _merge_chunk_weight_grads(
            grads_weight1[0], grads_weight1[1], weight1,
        )
        grads_out[idx_w2] = _merge_chunk_weight_grads(
            grads_weight2[0], grads_weight2[1], weight2,
        )
        for event in after_event_dispatch:
            if event is not None:
                event.current_stream_wait()
        return tuple(grads_out)


if HAVE_DEEP_EP:

    def fused_dispatch(
        x: torch.Tensor,
        token_indices,
        token_probs,
        num_experts,
        group,
        is_float8_dispatch: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            is_float8_dispatch,
            weight_quantizer_handles,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False, is_float8_training: bool = False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream, is_float8_training)

    def scmoe_layer(
        token_dispatcher,
        experts,
        weight1,
        weight2,
        *argv_chunks,
        allocate_on_comm_stream: bool = False,
        is_float8_training: bool = False,
        save_dispatch_for_backward: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
    ):
        """Run the overlapped ScMoE DeepEP path.

        Args:
            token_dispatcher: TokenDispatcher instance
            experts: expert module callable used after dispatch
            weight1: expert weight1 parameter (see `GroupedLlamaMLP.weight1`)
            weight2: expert weight2 parameter (see `GroupedLlamaMLP.weight2`)
            allocate_on_comm_stream: match the baseline DeepEP setting if True
            save_dispatch_for_backward: if True, cache dispatch results from
                forward and skip the recompute dispatch in backward (trades
                memory for time).
            argv_chunks: flattened per-chunk inputs
                `[hidden_states, routing_map, token_indices, token_probs, ctx] * num_chunks`

        Returns:
            Tuple of per-chunk combined outputs.
        """
        return ScMoE.apply(
            token_dispatcher,
            experts,
            allocate_on_comm_stream,
            is_float8_training,
            save_dispatch_for_backward,
            weight_quantizer_handles,
            weight1,
            weight2,
            *argv_chunks,
        )

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None
    scmoe_layer = None
