import copy
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from llmfoundry.models.layers import (
    BLOCK_CLASS_REGISTRY,
    FC_CLASS_REGISTRY,
)
from llmfoundry.models.layers.ffn import build_ffn
from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.parallel.tensor import distribute_async_tp_embeddings


def resolve_mtp_block_type(config) -> str:
    mtp_block_type = getattr(config, "mtp_block_type", None)
    if mtp_block_type is None:
        return "moe" if hasattr(config, "moe_impl") else "dense"
    if mtp_block_type not in {"dense", "moe"}:
        raise ValueError(
            f"Unsupported mtp_block_type '{mtp_block_type}'. Expected 'dense' or 'moe'."
        )
    return mtp_block_type


def resolve_mtp_attention_anchor_layer_idx(config) -> int:
    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    return max(num_hidden_layers - 1, 0)


def build_mtp_dense_ffn(config, layer_idx: int | None = None) -> nn.Module:
    mtp_ffn_config = copy.copy(config)
    if getattr(mtp_ffn_config, "dense_intermediate_size", None) is None:
        mtp_ffn_config.dense_intermediate_size = config.intermediate_size
    if getattr(mtp_ffn_config, "dense_fused_mlp_checkpoint_lvl", None) is None:
        mtp_ffn_config.dense_fused_mlp_checkpoint_lvl = getattr(
            config, "fused_mlp_checkpoint_lvl", None
        )
    return build_ffn(config.tp_size, mtp_ffn_config, layer_idx=layer_idx)


def _get_mtp_layer_block_type(mtp_layer: nn.Module) -> str:
    return getattr(
        mtp_layer, "mtp_block_type", resolve_mtp_block_type(mtp_layer.config)
    )


def _get_activation_checkpoint_target(module: nn.Module) -> nn.Module:
    return next(module.children(), module)


def get_mtp_fsdp_modules_with_names(
    mtp_layers: nn.ModuleList,
    prefix: str,
) -> list[tuple[nn.Module, str]]:
    modules_with_names = []
    for layer_idx, mtp_layer in enumerate(mtp_layers):
        decoder_layer = mtp_layer.decoder_layer
        module_prefix = f"{prefix}.mtp_layers.{layer_idx}.decoder_layer"
        if _get_mtp_layer_block_type(mtp_layer) == "dense":
            modules_with_names.append((decoder_layer, module_prefix))
            continue

        if hasattr(decoder_layer, "self_attn"):
            modules_with_names.append(
                (decoder_layer.self_attn, f"{module_prefix}.self_attn")
            )
        if hasattr(decoder_layer, "block_sparse_moe"):
            modules_with_names.append(
                (decoder_layer.block_sparse_moe, f"{module_prefix}.block_sparse_moe")
            )

    return modules_with_names


def get_mtp_activation_checkpointing_modules(
    mtp_layers: nn.ModuleList,
    moe_target_fn: Optional[Callable[[nn.Module], Optional[nn.Module]]] = None,
) -> list[nn.Module]:
    if moe_target_fn is None:
        moe_target_fn = _get_activation_checkpoint_target

    checkpointing_modules = []
    for mtp_layer in mtp_layers:
        decoder_layer = mtp_layer.decoder_layer
        if _get_mtp_layer_block_type(mtp_layer) == "dense":
            checkpointing_modules.append(
                _get_activation_checkpoint_target(decoder_layer)
            )
            continue

        if hasattr(decoder_layer, "self_attn"):
            checkpointing_modules.append(
                _get_activation_checkpoint_target(decoder_layer.self_attn)
            )
        if hasattr(decoder_layer, "block_sparse_moe"):
            target = moe_target_fn(decoder_layer.block_sparse_moe)
            if target is not None:
                checkpointing_modules.append(target)

        if hasattr(decoder_layer, "post_feedforward_layernorm"):
            checkpointing_modules.append(
                _get_activation_checkpoint_target(
                    decoder_layer.post_feedforward_layernorm
                )
            )

    return checkpointing_modules


class MTPAttnAdapter(nn.Module):
    def __init__(
        self,
        base_attn: nn.Module,
        config,
        token_ln: nn.Module,
        hidden_ln: nn.Module,
        input_proj: nn.Module,
        exec_inside_self_attn: bool,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.mtp_token_layernorm = token_ln
        self.mtp_hidden_layernorm = hidden_ln
        self.mtp_input_proj = input_proj
        self.config = config
        self._exec_inside_self_attn = exec_inside_self_attn

    def _run_mtp(
        self,
        prev_h_raw: torch.Tensor,
        inp_emb_raw: torch.Tensor,
        logical_batch_size: Optional[int],
    ) -> torch.Tensor:
        inp_emb = self.mtp_token_layernorm(inp_emb_raw)
        prev_h = self.mtp_hidden_layernorm(prev_h_raw)
        proj_in = torch.cat([prev_h, inp_emb], dim=-1)

        if getattr(self.config, "tp_size", 1) > 1:
            if getattr(self.config, "enable_async_tp", False):
                out = self.mtp_input_proj(
                    proj_in, logical_batch_size=logical_batch_size
                )
                out = distribute_async_tp_embeddings(out)
            else:
                out = self.mtp_input_proj(proj_in, gather_output=True)
        else:
            out = self.mtp_input_proj(proj_in)

        return out

    def do_mtp_before_layernorm(
        self,
        mtp_inputs: Tuple[torch.Tensor, torch.Tensor],
        logical_batch_size: Optional[int],
    ) -> torch.Tensor:
        prev_h_raw, inp_emb_raw = mtp_inputs
        return self._run_mtp(prev_h_raw, inp_emb_raw, logical_batch_size)

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
        **tp_kwargs,
    ):
        mtp_inputs = tp_kwargs.pop("mtp_inputs", None)
        if self._exec_inside_self_attn and mtp_inputs is not None:
            prev_h_raw, inp_emb_raw = mtp_inputs
            hidden_states = self._run_mtp(prev_h_raw, inp_emb_raw, logical_batch_size)

        return self.base_attn(
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
            **tp_kwargs,
        )


class LlamaDecoderMTPLayer(nn.Module):
    def __init__(self, config, attention_layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.mtp_block_type = resolve_mtp_block_type(config)
        self.attention_layer_idx = (
            resolve_mtp_attention_anchor_layer_idx(config)
            if attention_layer_idx is None
            else attention_layer_idx
        )

        norm_class = resolve_norm_class(config.norm_type)
        token_layernorm: nn.Module = norm_class.from_config(config)
        hidden_layernorm: nn.Module = norm_class.from_config(config)

        if config.tp_size == 1:
            input_proj = nn.Linear(
                config.hidden_size * 2, config.hidden_size, bias=False
            )
        else:
            input_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                config=config,
                bias=False,
            )

        if self.mtp_block_type == "moe":
            self.decoder_layer = BLOCK_CLASS_REGISTRY["LlamaMixDecoderLayer"](
                config,
                layer_idx=-1,
                attention_layer_idx=self.attention_layer_idx,
            )
            exec_inside_self_attn = True  # MoE: делаем MTP внутри self_attn.forward
        else:
            self.decoder_layer = BLOCK_CLASS_REGISTRY["LlamaDecoderLayer"](
                config,
                layer_idx=-1,
                attention_layer_idx=self.attention_layer_idx,
            )
            mlp_layer_idx = getattr(self.decoder_layer.mlp, "layer_idx", None)
            self.decoder_layer.mlp = build_mtp_dense_ffn(
                config,
                layer_idx=mlp_layer_idx,
            )
            exec_inside_self_attn = False  # non-MoE: MTP до layer.input_layernorm

        base_attn = self.decoder_layer.self_attn
        self.decoder_layer.self_attn = MTPAttnAdapter(
            base_attn=base_attn,
            config=config,
            token_ln=token_layernorm,
            hidden_ln=hidden_layernorm,
            input_proj=input_proj,
            exec_inside_self_attn=exec_inside_self_attn,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
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
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mtp_inputs = (hidden_states, input_embeds)

        layer_outputs = self.decoder_layer(
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
        hidden_states = layer_outputs[0]
        return hidden_states


class LlamaDecoderMTPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_mtp = config.use_mtp

        if self.use_mtp:
            norm_class = resolve_norm_class(config.norm_type)
            attention_layer_idx = resolve_mtp_attention_anchor_layer_idx(config)
            self.mtp_layers = nn.ModuleList(
                [
                    MTP_CLASS_REGISTRY["LlamaDecoderMTPLayer"](
                        config,
                        attention_layer_idx=attention_layer_idx,
                    )
                    for _ in range(config.mtp_predictor_num)
                ]
            )
            self.mtp_norms = nn.ModuleList(
                [
                    norm_class.from_config(config)
                    for _ in range(config.mtp_predictor_num)
                ]
            )

        else:
            self.mtp_layers = nn.ModuleList()
            self.mtp_norms = nn.ModuleList()

    def forward(
        self,
        input_embeds: torch.Tensor,
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
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ):
        if not self.use_mtp:
            return []

        assert not is_left_padded_eval, "MTP is not supported in inference mode"

        mtp_hidden_states = []
        mtp_h = hidden_states
        for mtp_layer in self.mtp_layers:
            input_embeds = torch.roll(input_embeds, shifts=-1, dims=1)
            input_embeds[:, -1, :] = 0
            mtp_h = mtp_layer(
                input_embeds=input_embeds,
                hidden_states=mtp_h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                logical_batch_size=logical_batch_size,
                is_left_padded_eval=is_left_padded_eval,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
            )
            mtp_hidden_states.append(mtp_h)

        mtp_hidden_states = tuple(
            norm_module(h) for norm_module, h in zip(self.mtp_norms, mtp_hidden_states)
        )

        return mtp_hidden_states


MTP_CLASS_REGISTRY = {
    "LlamaDecoderMTPLayer": LlamaDecoderMTPLayer,
    "LlamaDecoderMTPBlock": LlamaDecoderMTPBlock,
}
