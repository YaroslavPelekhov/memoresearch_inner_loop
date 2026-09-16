from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from composer.utils.dist import get_ep_group_size
from composer.utils.profiler_annotation import decorator_forward_backward

from llmfoundry.models.layers.moe.dispatchers import (
    TokenDispatcher,
    DispatcherContext,
    scmoe_layer,
)
from llmfoundry.models.layers.moe.experts import GroupedLlamaMLP, GroupedSonicMLP, ScMoEBlockGeMM
from llmfoundry.models.layers.moe.sonic_cuda import require_sonic_moe_cuda_device

# TODO: move to utils?
def get_dense_statistics(routing_map: torch.Tensor, routing_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to transform routing map to token indices before permutation.
    Returns (expert_indices, expert_probs, token_indices).
    """
    token_indices, expert_indices = torch.where(routing_map == 1)
    token_indices = token_indices.to(torch.int32)
    expert_indices = expert_indices.to(torch.int32)
    expert_probs = routing_probs[routing_map].view(-1)

    return expert_indices, expert_probs, token_indices


# Type annotation for in-place Tensor initialization function.
InitFn = Callable[[torch.Tensor], None]


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss



class AbstractGMMMoeBlock(nn.Module, ABC):
    """Abstract MoE block with shared init and add_aux_loss. Subclasses set self.experts and implement forward."""

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.token_dispatcher = TokenDispatcher(config)

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        routing_probs: torch.Tensor,
        aux_loss: torch.Tensor | None = None,
        activation_checkpointing_on_layer: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Subclasses implement the MoE forward (permute/dispatch -> experts -> unpermute/combine)."""
        ...

    def add_aux_loss(
        self,
        expert_output: torch.Tensor,
        aux_loss: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply auxiliary loss gradient in backward when training."""
        if self.training and aux_loss is not None:
            expert_output = AddAuxiliaryLoss.apply(expert_output, aux_loss)
        return expert_output


class DeepseekGMMMoeBlock(AbstractGMMMoeBlock):
    """MoE block with separate permute from experts: permute -> experts -> unpermute (GroupedLlamaMLP)."""

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__(config, layer_idx)
        if getattr(config, 'moe_kernel', None) == 'sonic':
            raise ValueError(
                "DeepseekGMMMoeBlock no longer supports moe_kernel='sonic'. "
                "Use moe_impl='SonicGMMMoeBlock' instead."
            )
        ep_size = get_ep_group_size() or 1
        self.experts = GroupedLlamaMLP(config, ep_size)


    @decorator_forward_backward()
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        routing_probs: torch.Tensor,
        aux_loss: torch.Tensor | None = None,
        activation_checkpointing_on_layer: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.size()

        weight_quantizer_handles = None
        if (
            self.config.use_float8_grouped_gemm
            and not self.config.float8_blocking_weight_quant
        ):
            assert self.experts.handle1 is not None
            assert self.experts.handle2 is not None
            weight_quantizer_handles = (
                self.experts.handle1.async_weight_quantizer,
                self.experts.handle2.async_weight_quantizer,
                self.experts,
            )

        is_float8_training = self.config.use_float8_grouped_gemm and self.training

        (
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
            dispatch_ctx,
            weights_quantized,
        ) = self.token_dispatcher.preprocess_experts_input(
            hidden_states=hidden_states,
            probs=routing_probs,
            routing_map=routing_map,
            training=self.training,
            is_float8_training=is_float8_training,
            padding_mask=padding_mask,
            weight_quantizer_handles=weight_quantizer_handles,
        )
        expert_output = self.experts(
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
            weights_quantized,
        )
        expert_output = self.add_aux_loss(expert_output, aux_loss)
        output = self.token_dispatcher.postprocess_experts_output(
            expert_output, dispatch_ctx,
            training=self.training,
            is_float8_training=is_float8_training,
        )
        return output.view(hidden_states_shape)


try:
    from llmfoundry.models.layers.moe.sonicmoe.functional import moe_general_routing_inputs as _sonic_check
    _sonic_available = _sonic_check is not None
except Exception:
    _sonic_available = False

if not _sonic_available:
    class SonicGMMMoeBlock(AbstractGMMMoeBlock):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "SonicGMMMoeBlock requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "SonicGMMMoeBlock requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )
else:
    class SonicGMMMoeBlock(AbstractGMMMoeBlock):
        """MoE block with fused permute and experts: dispatch-only path (GroupedSonicMLP)."""

        def __init__(self, config, layer_idx: int | None = None) -> None:
            require_sonic_moe_cuda_device()
            super().__init__(config, layer_idx)
            assert not config.use_float8_grouped_gemm, (
                "SonicGMMMoeBlock does not support float8 grouped GEMM. "
                "Float8 is only supported for dispatch via DeepEP, not for SonicMoE experts."
            )
            ep_size = get_ep_group_size() or 1
            self.experts = GroupedSonicMLP(config, ep_size)
            assert self.experts.fused_permute, "SonicGMMMoeBlock requires experts with fused_permute"

        def forward(
            self,
            hidden_states: torch.Tensor,
            routing_map: torch.Tensor,
            routing_probs: torch.Tensor,
            aux_loss: torch.Tensor | None = None,
            activation_checkpointing_on_layer: bool = False,
            padding_mask: Optional[torch.Tensor] = None,
            prefix_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

            hidden_states_shape = hidden_states.size()
            (
                dispatched_hidden_states,
                dispatched_routing_map,
                dispatched_routing_probs,
                dispatch_ctx,
            ) = self.token_dispatcher.preprocess_experts_input_dispatch_only(
                hidden_states,
                routing_probs,
                routing_map,
                training=self.training,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
            )
            expert_indices, expert_probs, token_indices = get_dense_statistics(
                dispatched_routing_map,
                dispatched_routing_probs,
            )
            expert_output = self.experts(
                local_hidden_states=dispatched_hidden_states,
                token_idxs=token_indices,
                expert_idxs=expert_indices,
                expert_scores=expert_probs,
            )
            expert_output = self.add_aux_loss(expert_output, aux_loss)
            output = self.token_dispatcher.postprocess_experts_output_dispatch_only(
                expert_output, dispatch_ctx
            )
            return output.view(hidden_states_shape)


class ScMoEBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        ep_size = get_ep_group_size()
        if ep_size is None:
            ep_size = 1

        self.token_dispatcher = TokenDispatcher(self.config)
        self.gemm = ScMoEBlockGeMM(config, self.token_dispatcher)
        self.save_dispatch_for_backward = getattr(
            config, "scmoe_save_dispatch_for_backward", False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        routing_probs: torch.Tensor,
        aux_loss: torch.Tensor | None = None,
        activation_checkpointing_on_layer: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not getattr(self.config, "float8_quantize_at_boundary", False), (
            "ScMoEBlock does not support float8_quantize_at_boundary. "
            "Disable FP8 activation boundary quantization for ScMoE."
        )
        hidden_states_shape = hidden_states.size()
        hidden_states = hidden_states.flatten(0, -2)
        num_tokens = hidden_states.size(0)

        weight_quantizer_handles = None
        if (
            self.config.use_float8_grouped_gemm
            and not getattr(self.config, "float8_blocking_weight_quant", False)
        ):
            assert self.gemm.experts.handle1 is not None
            assert self.gemm.experts.handle2 is not None
            weight_quantizer_handles = (
                self.gemm.experts.handle1.async_weight_quantizer,
                self.gemm.experts.handle2.async_weight_quantizer,
                self.gemm.experts,
            )

        if not self.token_dispatcher.deepep_available(self.training) or num_tokens < 2:
            (
                permuted_hidden_states,
                tokens_per_expert_tensor_gpu,
                tokens_per_expert_tensor_cpu,
                tokens_per_expert_list,
                permuted_probs,
                dispatch_ctx,
                weights_quantized,
            ) = self.token_dispatcher.preprocess_experts_input(
                hidden_states=hidden_states,
                probs=routing_probs,
                routing_map=routing_map,
                training=self.training,
                is_float8_training=self.config.use_float8_grouped_gemm and self.training,
                padding_mask=padding_mask,
                weight_quantizer_handles=weight_quantizer_handles,
            )
            expert_output = self.gemm.experts(
                permuted_hidden_states,
                tokens_per_expert_tensor_gpu,
                tokens_per_expert_tensor_cpu,
                tokens_per_expert_list,
                permuted_probs,
                weights_quantized,
            )
            expert_output = self.add_aux_loss(expert_output, aux_loss)
            output = self.token_dispatcher.postprocess_experts_output(
                expert_output,
                dispatch_ctx,
                training=self.training,
                is_float8_training=self.config.use_float8_grouped_gemm and self.training,
            )
            return output.view(hidden_states_shape)

        probs, routing_map, token_indices, token_probs = self.token_dispatcher.preprocess(
            hidden_states, routing_probs, routing_map, training=self.training, padding_mask=padding_mask, prefix_mask=prefix_mask
        )
        if scmoe_layer is None:
            raise RuntimeError(
                "ScMoE DeepEP path requires `deep_ep` and `llmfoundry.models.layers.moe.fused_a2a.scmoe_layer`. "
                "Install deep_ep or disable DeepEP / use the non-DeepEP MoE path."
            )
        num_chunks = 2
        chunk_size = (hidden_states.size(0) + num_chunks - 1) // num_chunks
        chunks = []
        for i in range(num_chunks):
            l, r = chunk_size * i, chunk_size * (i + 1)
            hd = hidden_states[l:r]
            ctx = DispatcherContext()
            ctx.hidden_shape_before_permute = hd.size()
            chunks.extend([hd, routing_map[l:r], token_indices[l:r], token_probs[l:r], ctx])
        outpt = scmoe_layer(
            self.token_dispatcher,
            self.gemm,
            self.gemm.experts.weight1,
            self.gemm.experts.weight2,
            *chunks,
            allocate_on_comm_stream=True,
            is_float8_training=self.config.use_float8_grouped_gemm,
            save_dispatch_for_backward=self.save_dispatch_for_backward,
            weight_quantizer_handles=weight_quantizer_handles,
        )
        output = torch.cat(outpt).view(hidden_states_shape)
        if self.training and aux_loss is not None:
            output = AddAuxiliaryLoss.apply(output, aux_loss)
        return output

    def add_aux_loss(
        self,
        expert_output: torch.Tensor,
        aux_loss: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.training and aux_loss is not None:
            expert_output = AddAuxiliaryLoss.apply(expert_output, aux_loss)
        return expert_output


MOE_CLASS_REGISTRY = {
    "DeepseekGMMMoeBlock": DeepseekGMMMoeBlock,
    "ScMoEBlock": ScMoEBlock,
    "SonicGMMMoeBlock": SonicGMMMoeBlock,
}
