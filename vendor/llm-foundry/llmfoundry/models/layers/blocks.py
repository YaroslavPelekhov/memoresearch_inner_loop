# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import composer.trainer.sac_utils as sac_utils
from composer.utils import dist, global_actlog_state
from composer.utils.profiler_annotation import decorator_forward_backward

from llmfoundry.models.layers.attention import (
    ATTN_CLASS_REGISTRY,
    _apply_sigmoid_gate_fp32,
)
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, build_ffn
from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.ops.hidden_z_loss_hook import apply_hidden_z_loss_hook

try:
    from llmfoundry.models.layers.moe import MOE_CLASS_REGISTRY
    from llmfoundry.models.layers.moe.dispatchers import DispatcherContext
    from llmfoundry.models.layers.moe.experts import (
        SharedExpertLlamaMLP,
        SharedExpertParallelLlamaMLP,
    )
    from llmfoundry.models.layers.moe.quantize_boundary import quantize_to_fp8_at_boundary
    from llmfoundry.models.layers.moe.routers import GATE_CLASS_REGISTRY
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
except ImportError:
    # MoE/FP8 is outside autoresearch v1; keep dense decoder imports usable.
    MOE_CLASS_REGISTRY = {}
    GATE_CLASS_REGISTRY = {}
    DispatcherContext = object
    SharedExpertLlamaMLP = SharedExpertParallelLlamaMLP = object
    Float8BlockwiseQTensor = ()

    def quantize_to_fp8_at_boundary(value):
        return value


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        attention_layer_idx: int | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_layer_idx = (
            layer_idx if attention_layer_idx is None else attention_layer_idx
        )

        if config.linear_attention_type is not None:
            attn_cls = (
                ATTN_CLASS_REGISTRY[config.attention_type]
                if self.attention_layer_idx in config.full_attention_layers
                else ATTN_CLASS_REGISTRY[config.linear_attention_type]
            )
            self.self_attn = attn_cls(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = ATTN_CLASS_REGISTRY[config.attention_type](
                config=config, layer_idx=layer_idx
            )

        self.layer_idx = layer_idx

        norm_class = resolve_norm_class(config.norm_type)
        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_pre_layernorm = layernorm_type in {"pre", "pre_post"}
        self._use_post_layernorm = layernorm_type in {"post", "pre_post"}

        if self._use_pre_layernorm:
            self.input_layernorm = norm_class.from_config(config)
            self.post_attention_layernorm = norm_class.from_config(config)

        if self._use_post_layernorm:
            self.post_self_attn_layernorm = norm_class.from_config(config)
            self.post_feedforward_layernorm = norm_class.from_config(config)

        if config.tp_size == 1:
            self.mlp = FFN_CLASS_REGISTRY["LlamaMLP"](config, layer_idx=layer_idx)
        else:
            self.mlp = FFN_CLASS_REGISTRY["ParallelMLP"](config, layer_idx=layer_idx)

        if config.enable_async_tp and config.fused_mlp:
            raise NotImplementedError("Not yet")

        self._is_fused_mlp = config.fused_mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
        mtp_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cu_seqlens (`torch.Tensor`, *optional*) cumulative sequence len when packing
            logical_batch_size (`int`, *optional*) used for tensor paralellism to reconstruct output tensor shapes
        """

        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )

        if mtp_inputs is not None and hasattr(
            self.self_attn, "do_mtp_before_layernorm"
        ):
            hidden_states = self.self_attn.do_mtp_before_layernorm(
                mtp_inputs, logical_batch_size
            )

        residual = hidden_states
        if self._use_pre_layernorm:
            global_actlog_state.use_monitor_variable(
                hidden_states,
                "model.model.layers.{}.self_attn.input_layernorm",
                "_input.0",
            )
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            is_left_padded_eval=is_left_padded_eval,
            **tp_kwargs,
        )
        global_actlog_state.use_monitor_variable(
            hidden_states, "model.model.layers.{}.self_attn._attn", "_output.0"
        )
        if self._use_post_layernorm:
            hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self._use_pre_layernorm:
            global_actlog_state.use_monitor_variable(
                hidden_states,
                "model.model.layers.{}.self_attn.post_attention_layernorm",
                "_input.0",
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
        if self._is_fused_mlp:
            hidden_states = self.mlp(
                hidden_states,
                activation_checkpointing_on_layer=(
                    self._activation_checkpointing
                    if hasattr(self, "_activation_checkpointing")
                    else False
                ),
            )
        else:
            hidden_states = self.mlp(hidden_states, **tp_kwargs)

        if self._use_post_layernorm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class AttnGate(nn.Module):
    def __init__(
        self,
        config,
        is_gate: bool = False,
        n_shared_experts: int | None = None,
        layer_idx: int | None = None,
        attention_layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.is_gate = is_gate
        self.attention_layer_idx = (
            layer_idx if attention_layer_idx is None else attention_layer_idx
        )

        if config.linear_attention_type is not None:
            attn_cls = (
                ATTN_CLASS_REGISTRY[config.attention_type]
                if self.attention_layer_idx in config.full_attention_layers
                else ATTN_CLASS_REGISTRY[config.linear_attention_type]
            )
            self._attn = attn_cls(config=config, layer_idx=layer_idx)
        else:
            self._attn = ATTN_CLASS_REGISTRY[config.attention_type](
                config=config, layer_idx=layer_idx
            )

        tp_size = config.tp_size or 1
        if bool(getattr(config, "use_shared_expert_sigmoid", False)) and tp_size > 1:
            raise AssertionError(
                "`use_shared_expert_sigmoid=True` is not supported for `tp_size > 1` "
                "because this path has not been tested."
            )

        assert is_gate == (n_shared_experts is not None)

        if is_gate:
            # gating
            if config.gating_type in GATE_CLASS_REGISTRY:
                self._gate = GATE_CLASS_REGISTRY[config.gating_type](
                    config, layer_idx=layer_idx
                )
            else:
                raise ValueError(f"Invalid gating type: {config.gating_type}")

        else:
            self.mlp = build_ffn(config.tp_size, config)

        norm_class = resolve_norm_class(config.norm_type)
        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_pre_layernorm = layernorm_type in {"pre", "pre_post"}
        self._use_post_layernorm = layernorm_type in {"post", "pre_post"}

        self._hidden_z_loss_attn_coef = float(
            getattr(config, "hidden_z_loss_attn_coef", 0.0)
        )
        # Dense-only: regularize MLP output before its post_feedforward_layernorm
        # (which lives inside `AttnGate` for the dense path — see comment above).
        self._hidden_z_loss_dense_mlp_coef = (
            float(getattr(config, "hidden_z_loss_dense_mlp_coef", 0.0))
            if not is_gate
            else 0.0
        )

        # The hidden_z_loss hooks attach right before `post_self_attn_layernorm`
        # (attn) and `post_feedforward_layernorm` (dense MLP). Those modules only
        # exist when `layernorm_type` includes a post-norm, so non-zero coefs are
        # only meaningful in that case.
        assert (
            self._hidden_z_loss_attn_coef == 0.0 or self._use_post_layernorm
        ), (
            f"hidden_z_loss_attn_coef={self._hidden_z_loss_attn_coef} requires "
            f"layernorm_type in {{'post', 'pre_post'}}, got "
            f"layernorm_type='{layernorm_type}'."
        )
        assert (
            self._hidden_z_loss_dense_mlp_coef == 0.0 or self._use_post_layernorm
        ), (
            f"hidden_z_loss_dense_mlp_coef={self._hidden_z_loss_dense_mlp_coef} "
            f"requires layernorm_type in {{'post', 'pre_post'}}, got "
            f"layernorm_type='{layernorm_type}'."
        )

        if self._use_pre_layernorm:
            self.input_layernorm = norm_class.from_config(config)
            self.post_attention_layernorm = norm_class.from_config(config)

        if self._use_post_layernorm:
            self.post_self_attn_layernorm = norm_class.from_config(config)

            # For dense layers (is_gate=False), post_feedforward_layernorm lives
            # here so it falls inside the activation checkpoint boundary of
            # `AttnGate` (`self_attn` of `LlamaMixDecoderLayer`)
            if not is_gate:
                self.post_feedforward_layernorm = norm_class.from_config(config)

        if n_shared_experts is not None and n_shared_experts > 0:
            if config.tp_size == 1:
                self.shared_experts = SharedExpertLlamaMLP(config)
            else:
                self.shared_experts = SharedExpertParallelLlamaMLP(config)

            self.use_shared_expert_sigmoid = bool(
                getattr(config, "use_shared_expert_sigmoid", False)
            )
            if self.use_shared_expert_sigmoid:
                self.shared_expert_sigmoid = nn.Linear(
                    config.hidden_size,
                    1,
                    bias=False,
                    device=config.init_device,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
        **extra_kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        DispatcherContext | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # Self attention
        mtp_inputs = extra_kwargs.pop("mtp_inputs", None)
        is_left_padded_eval = extra_kwargs.pop("is_left_padded_eval", False)

        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )

        residual = hidden_states
        if self._use_pre_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            is_left_padded_eval=is_left_padded_eval,
            **tp_kwargs,
            **extra_kwargs,
        )
        if mtp_inputs is not None and hasattr(self._attn, "_exec_inside_self_attn"):
            attn_kwargs["mtp_inputs"] = mtp_inputs

        hidden_states, self_attn_weights, present_key_value = self._attn(**attn_kwargs)

        hidden_z_loss_attn = None
        if self._use_post_layernorm:
            hidden_states, hidden_z_loss_attn = apply_hidden_z_loss_hook(
                hidden_states,
                self._hidden_z_loss_attn_coef,
                training=self.training,
            )
            hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if self._use_pre_layernorm:
            hidden_states = self.post_attention_layernorm(hidden_states)

        # Initialize variables for MoE processing
        aux_loss = None
        shared_states = None
        routing_map = None
        routing_probs = None
        hidden_z_loss_dense_mlp = None

        if self.is_gate:
            routing_map, routing_probs, aux_loss = self._gate(
                hidden_states,
                return_sparse_outputs=True,
                logical_batch_size=logical_batch_size,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
            )

            if hasattr(self, "shared_experts"):
                if self.config.tp_size > 1:
                    mlp_kwargs = dict(logical_batch_size=logical_batch_size)
                else:
                    mlp_kwargs = dict()
                shared_states = self.shared_experts(hidden_states, **mlp_kwargs)
                if hasattr(self, "shared_expert_sigmoid"):
                    shared_states = _apply_sigmoid_gate_fp32(
                        shared_states, self.shared_expert_sigmoid(hidden_states)
                    )
                global_actlog_state.use_monitor_variable(
                    shared_states,
                    "model.model.layers.{}.self_attn.shared_experts",
                    "_output.0",
                )
        else:
            if self.config.tp_size > 1:
                mlp_kwargs = dict(logical_batch_size=logical_batch_size)
            else:
                mlp_kwargs = dict()
            hidden_states = self.mlp(hidden_states, **mlp_kwargs)
            global_actlog_state.use_monitor_variable(
                hidden_states, "model.model.layers.{}.self_attn.mlp", "_output.0"
            )
            if self._use_post_layernorm:
                hidden_states, hidden_z_loss_dense_mlp = apply_hidden_z_loss_hook(
                    hidden_states,
                    self._hidden_z_loss_dense_mlp_coef,
                    training=self.training,
                )
                hidden_states = self.post_feedforward_layernorm(hidden_states)

        return (
            hidden_states,
            shared_states,
            residual,
            self_attn_weights,
            present_key_value,
            aux_loss,
            routing_map,
            routing_probs,
            hidden_z_loss_attn,
            hidden_z_loss_dense_mlp,
        )


class LlamaMixDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        attention_layer_idx: int | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = config.tp_size or 1
        self.tp_size = tp_size
        self.layer_idx = layer_idx
        self.attention_layer_idx = (
            layer_idx if attention_layer_idx is None else attention_layer_idx
        )

        is_sparse_layer = (
            layer_idx == -1
            or config.first_k_dense_replace is None
            or layer_idx >= config.first_k_dense_replace
        )

        if config.fused_mlp:
            assert not is_sparse_layer, "Fused MLP is not supported for MoE"

        self.self_attn = AttnGate(
            config,
            is_gate=is_sparse_layer,
            n_shared_experts=(config.n_shared_experts if is_sparse_layer else None),
            layer_idx=layer_idx,
            attention_layer_idx=self.attention_layer_idx,
        )
        self.self_attn._activation_cpu_offload_inputs = True
        
        # NOTE:
        # Only for sparse layers, `post_feedforward_layernorm` lives here (not inside
        # `block_sparse_moe`) so that the MoE combine op is not recomputed during
        # activation checkpointing of `block_sparse_moe`.
        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_post_layernorm = is_sparse_layer and layernorm_type in {"post", "pre_post"}
        self._hidden_z_loss_moe_coef = float(
            getattr(config, "hidden_z_loss_moe_coef", 0.0)
        )

        # The MoE hidden_z_loss hook attaches right before
        # `post_feedforward_layernorm`, which exists only for sparse layers with
        # a post-norm. For dense layers there is no MoE block, so the coef is a
        # no-op there; we still guard sparse layers explicitly.
        assert (
            self._hidden_z_loss_moe_coef == 0.0
            or not is_sparse_layer
            or self._use_post_layernorm
        ), (
            f"hidden_z_loss_moe_coef={self._hidden_z_loss_moe_coef} requires "
            f"layernorm_type in {{'post', 'pre_post'}}, got "
            f"layernorm_type='{layernorm_type}'."
        )

        if is_sparse_layer:
            copy_config = copy.deepcopy(config)
            copy_config.n_shared_experts = None
            self.block_sparse_moe = MOE_CLASS_REGISTRY[config.moe_impl](copy_config)
            self.block_sparse_moe._activation_cpu_offload_inputs = True
            assert MOE_CLASS_REGISTRY[config.moe_impl] is not None, (
                f"Current hardware isn't compitable with moe kernel {config.moe_impl}"
            )

            if self._use_post_layernorm:
                norm_class = resolve_norm_class(config.norm_type)
                self.post_feedforward_layernorm = norm_class.from_config(config)
                self.post_feedforward_layernorm._activation_cpu_offload_inputs = False

    @decorator_forward_backward()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
        mtp_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cu_seqlens (`torch.Tensor`, *optional*) cumulative sequence len when packing
        """

        # Self Attention — отдельный offload-слот (AttnGate vs MoE для sparse).
        with sac_utils.cpu_offload_context:
            (
                hidden_states,
                shared_states,
                residual,
                self_attn_weights,
                present_key_value,
                aux_loss,
                routing_map,
                routing_probs,
                hidden_z_loss_attn,
                hidden_z_loss_dense_mlp,
            ) = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                logical_batch_size=logical_batch_size,
                is_left_padded_eval=is_left_padded_eval,
                mtp_inputs=mtp_inputs,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
            )

        # Fully Connected
        do_activation_checkpointing = (
            self._activation_checkpointing
            if hasattr(self, "_activation_checkpointing")
            else False
        )
        if hasattr(self, "block_sparse_moe"):
            original_hidden_states_shape = hidden_states.size()

            # Pre-quantize OUTSIDE the AC region wrapping `block_sparse_moe` so
            # the tensor AC stashes across forward→backward is FP8
            # (uint8 data + rowwise scales) instead of bf16. Inside the MoE
            # block, FusedDispatch detects the Float8BlockwiseQTensor and skips
            # its own row_quant_fn call (the "Already quantized at checkpoint
            # boundary" branch in fused_a2a).
            cfg = self.block_sparse_moe.config
            dispatcher = self.block_sparse_moe.token_dispatcher
            if (
                self.training
                and getattr(cfg, "float8_quantize_at_boundary", False)
                and getattr(cfg, "use_float8_grouped_gemm", False)
                and getattr(dispatcher, "enable_deepep", False)
                and getattr(dispatcher, "ep_size", 1) > 1
                and not isinstance(hidden_states, Float8BlockwiseQTensor)
            ):
                hidden_states = quantize_to_fp8_at_boundary(hidden_states)

            hidden_states = sac_utils.cpu_offload_sync_function(hidden_states)
            with sac_utils.cpu_offload_context:
                hidden_states = self.block_sparse_moe(
                    hidden_states,
                    routing_map,
                    routing_probs,
                    aux_loss=aux_loss,
                    activation_checkpointing_on_layer=do_activation_checkpointing,
                    padding_mask=padding_mask,
                    prefix_mask=prefix_mask,
                )
                global_actlog_state.use_monitor_variable(
                    hidden_states, "model.model.layers.{}.block_sparse_moe", "_output.0"
                )
                hidden_states = hidden_states.view(original_hidden_states_shape)

            if shared_states is not None:
                hidden_states = hidden_states + shared_states

            hidden_z_loss_moe = None
            if self._use_post_layernorm:
                hidden_states, hidden_z_loss_moe = apply_hidden_z_loss_hook(
                    hidden_states,
                    self._hidden_z_loss_moe_coef,
                    training=self.training,
                )
                hidden_states = sac_utils.cpu_offload_sync_function(hidden_states)
                with sac_utils.cpu_offload_context:
                    hidden_states = self.post_feedforward_layernorm(hidden_states)
        else:
            hidden_z_loss_moe = None

        hidden_states = residual + hidden_states
        hidden_states = sac_utils.cpu_offload_sync_function(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (hidden_z_loss_attn, hidden_z_loss_moe, hidden_z_loss_dense_mlp)

        return outputs


BLOCK_CLASS_REGISTRY = {
    "LlamaDecoderLayer": LlamaDecoderLayer,
    "LlamaMixDecoderLayer": LlamaMixDecoderLayer,
}
