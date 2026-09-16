import functools
from typing import Any, Callable, Dict, List, Optional

import torch
from composer.utils import dist
from llmfoundry.models.layers.moe import AbstractGMMMoeBlock
from llmfoundry.models.layers.norm import resolve_norm_class

from .modelling_gigavision import GigaVisionForCausalLM


def _activation_checkpointing_auto_wrap_policy_gigar(
    fsdp_config: Dict[str, Any], model: GigaVisionForCausalLM
):
    def _checkpoint_policy_fn(m: torch.nn.Module) -> bool:
        num_layers_to_checkpoint = fsdp_config["num_layers_to_checkpoint"]
        if num_layers_to_checkpoint is None:
            num_layers_to_checkpoint = len(model._language_model.model.layers)

        checkpointed_modules = set(
            model._language_model.model.layers[:num_layers_to_checkpoint]
        )
        if model.config.llm_config.config.use_mtp:
            checkpointed_modules |= {model._language_model.model.mtp_block}
        checkpointed_modules |= set(
            model._mm_vision_tower.vision_tower.get_giga_fsdp_modules_for_activation_checkpointing()
        )

        return m in (next(layer.children()) for layer in checkpointed_modules)

    return functools.partial(
        torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
        lambda_fn=_checkpoint_policy_fn,
    )


def _giga_fsdp_modules_to_wrap_with_names_gigar(
    model: GigaVisionForCausalLM, separate_layer_norm_disabled: bool
):
    vision_modules = []
    modules_with_names = [
        (model._language_model.model.embed_tokens, "_language_model.model.embed_tokens")
    ]

    for layer_idx, module in enumerate(model._language_model.model.layers):
        modules_with_names.append((module, f"_language_model.model.layers.{layer_idx}"))

    if separate_layer_norm_disabled:
        modules_with_names.append(
            (model._language_model.model.norm, "_language_model.model.norm")
        )

    modules_with_names.append(
        (model._language_model.lm_head, "_language_model.lm_head")
    )

    if model.config.llm_config.config.use_mtp:
        modules_with_names.append(
            (model._language_model.model.mtp_block, "_language_model.model.mtp_block")
        )

    tower_modules = (
        model._mm_vision_tower.vision_tower.get_giga_fsdp_modules_to_wrap_with_names(
            prefix="_mm_vision_tower.vision_tower."
        )
    )
    modules_with_names.extend(tower_modules)
    vision_modules.extend([m[0] for m in tower_modules])

    modules_with_names.append((model._mm_projector, "_mm_projector"))
    vision_modules.append(model._mm_projector)
    return (modules_with_names, vision_modules)


def _giga_fsdp_rogue_layer_norm_modules_with_names_gigar(model: GigaVisionForCausalLM):
    """
    LayerNorm is rouge only in case it is not a part of STM (SuperTensorModule)
    STM wrapping is declared in giga_fsdp_modules_to_wrap_with_names
    """
    rouge_layer_norms = {model._language_model.model.norm: "_language_model.model.norm"}
    if hasattr(model._language_model.model, "mtp_block"):
        rouge_layer_norms |= {
            m: f"_language_model.model.mtp_block.mtp_norms.{i}"
            for i, m in enumerate(model._language_model.model.mtp_block.mtp_norms)
        }
    return rouge_layer_norms


def _activation_checkpointing_auto_wrap_policy_giga_mix(
    fsdp_config: Dict[str, Any], model: GigaVisionForCausalLM
):
    def _checkpoint_policy_fn(m: torch.nn.Module) -> bool:
        num_layers_to_checkpoint = fsdp_config["num_layers_to_checkpoint"]
        if num_layers_to_checkpoint is None:
            num_layers_to_checkpoint = len(model._language_model.model.layers)

        layers = model._language_model.model.layers[:num_layers_to_checkpoint]
        checkpointed_modules = {module.self_attn for module in layers}
        checkpointed_modules |= {
            module.block_sparse_moe
            for module in layers
            if hasattr(module, "block_sparse_moe")
        }

        # `post_feedforward_layernorm` (sparse layers only) is checkpointed as
        # its own region — it lives on the decoder layer (outside `block_sparse_moe`
        # so the MoE combine op is not recomputed) and the gated-norm forward is
        # expensive enough in activations that leaving it without AC is costly.
        post_norms = {
            module.post_feedforward_layernorm
            for module in layers
            if hasattr(module, "post_feedforward_layernorm")
        }

        if hasattr(model._language_model.model, "mtp_block"):
            for mtp_layer in model._language_model.model.mtp_block.mtp_layers:
                dec = mtp_layer.decoder_layer
                if hasattr(dec, "self_attn"):
                    checkpointed_modules.add(dec.self_attn)
                if hasattr(dec, "block_sparse_moe"):
                    checkpointed_modules.add(dec.block_sparse_moe)
                if hasattr(dec, "post_feedforward_layernorm"):
                    post_norms.add(dec.post_feedforward_layernorm)

        checkpointed_modules |= set(
            model._mm_vision_tower.vision_tower.get_giga_fsdp_modules_for_activation_checkpointing()
        )

        # STMs are wrapped at their first child; post-norms as the whole module
        # (their children would be the internal gate projections of a gated norm).
        return m in post_norms or m in (
            next(layer.children()) for layer in checkpointed_modules
        )

    return functools.partial(
        torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
        lambda_fn=_checkpoint_policy_fn,
    )


def _giga_fsdp_modules_to_wrap_with_names_giga_mix(model: GigaVisionForCausalLM):
    vision_modules = []
    modules_with_names = [
        (model._language_model.model.embed_tokens, "_language_model.model.embed_tokens")
    ]
    for layer_idx, module in enumerate(model._language_model.model.layers):
        modules_with_names.append(
            (module.self_attn, f"_language_model.model.layers.{layer_idx}.self_attn")
        )
        if hasattr(module, "block_sparse_moe"):
            module_name = f"_language_model.model.layers.{layer_idx}.block_sparse_moe"
            mlp_submodule = module.block_sparse_moe
            modules_with_names.append((mlp_submodule, module_name))

    modules_with_names.append(
        (model._language_model.lm_head, "_language_model.lm_head")
    )
    if hasattr(model._language_model.model, "mtp_block"):
        for i, mtp_layer in enumerate(model._language_model.model.mtp_block.mtp_layers):
            dec = mtp_layer.decoder_layer
            if hasattr(dec, "self_attn"):
                modules_with_names.append(
                    (
                        dec.self_attn,
                        f"_language_model.model.mtp_block.mtp_layers.{i}.decoder_layer.self_attn",
                    )
                )
            if hasattr(dec, "block_sparse_moe"):
                modules_with_names.append(
                    (
                        dec.block_sparse_moe,
                        f"_language_model.model.mtp_block.mtp_layers.{i}.decoder_layer.block_sparse_moe",
                    )
                )
    tower_modules = (
        model._mm_vision_tower.vision_tower.get_giga_fsdp_modules_to_wrap_with_names(
            prefix="_mm_vision_tower.vision_tower."
        )
    )
    modules_with_names.extend(tower_modules)
    vision_modules.extend([m[0] for m in tower_modules])

    modules_with_names.append((model._mm_projector, "_mm_projector"))
    vision_modules.append(model._mm_projector)

    return (modules_with_names, vision_modules)


def _giga_fsdp_rogue_layer_norm_modules_with_names_giga_mix(
    model: GigaVisionForCausalLM,
):
    """LayerNorm is rogue only in case it is not a part of STM (SuperTensorModule)
    STM wrapping is declared in giga_fsdp_modules_to_wrap_with_names.

    Note: for sparse layers, `post_feedforward_layernorm` lives on the decoder
    layer itself (see `LlamaMixDecoderLayer.__init__`) so it IS rogue — it is
    not part of any STM. It is additionally wrapped as a standalone activation
    checkpoint target in `_activation_checkpointing_auto_wrap_policy_giga_mix`.
    For dense layers the post-norm lives inside `AttnGate` (the `self_attn`
    STM), so it is not rogue for those layers.
    """
    rogue_layer_norms = {model._language_model.model.norm: "_language_model.model.norm"}
    if hasattr(model._language_model.model, "embed_ln"):
        rogue_layer_norms[model._language_model.model.embed_ln] = (
            "_language_model.model.embed_ln"
        )
    for i, layer in enumerate(model._language_model.model.layers):
        if hasattr(layer, "post_feedforward_layernorm"):
            rogue_layer_norms[layer.post_feedforward_layernorm] = (
                f"_language_model.model.layers.{i}.post_feedforward_layernorm"
            )
    if hasattr(model._language_model.model, "mtp_block"):
        rogue_layer_norms |= {
            m: f"_language_model.model.mtp_block.mtp_norms.{i}"
            for i, m in enumerate(model._language_model.model.mtp_block.mtp_norms)
        }
        for i, mtp_layer in enumerate(model._language_model.model.mtp_block.mtp_layers):
            dec = mtp_layer.decoder_layer
            if hasattr(dec, "post_feedforward_layernorm"):
                rogue_layer_norms[dec.post_feedforward_layernorm] = (
                    f"_language_model.model.mtp_block.mtp_layers.{i}.decoder_layer.post_feedforward_layernorm"
                )
    return rogue_layer_norms


def _special_process_group_fn_giga_mix(module: torch.nn.Module):
    ep_size = dist.get_ep_group_size()
    if isinstance(module, AbstractGMMMoeBlock):
        if ep_size is None or ep_size == 1:
            return (torch.distributed.distributed_c10d._get_default_group(), None)
        else:
            return (dist.get_ep_fsdp_group(), dist.get_ep_group())
    else:
        return None


def _wrap_vision_process_group_fn(
    vision_modules: List[torch.nn.Module], fn: Optional[Callable] = None
) -> Callable:
    def wrapped_fn(module: torch.nn.Module):
        if module in vision_modules:
            return (torch.distributed.distributed_c10d._get_default_group(), None)
        else:
            return fn(module)

    return wrapped_fn


def build_gigavision_fsdp_wrapper_dict(
    target_model: GigaVisionForCausalLM, fsdp_config: dict
) -> dict:
    """Build GigaFSDP wrapper config dict for a GigaVision model.

    Used by both ComposerGigaVisionCausalLM and ComposerTeacherStudentModel (DPO).
    """
    separate_layer_norm_disabled = fsdp_config.get("separate_layer_norm_disabled", True)
    if target_model.config.llm_config.config.enable_async_tp:
        separate_layer_norm_disabled = False

    fsdp_config["num_layers_to_checkpoint"] = (
        target_model._language_model.config.activation_checkpoint_layers_num
    )

    llm_type = target_model.config.llm_config.type

    if llm_type == "gigar":
        all_modules, vision_modules = _giga_fsdp_modules_to_wrap_with_names_gigar(
            target_model, separate_layer_norm_disabled
        )
        norm_type = getattr(
            getattr(target_model._language_model, "config", None),
            "norm_type",
            "LlamaRMSNorm",
        )
        return {
            "activation_checkpointing_auto_wrap_policy": _activation_checkpointing_auto_wrap_policy_gigar(
                fsdp_config, target_model
            ),
            "giga_fsdp_modules_to_wrap_with_names": all_modules,
            "giga_fsdp_rogue_layer_norm_modules_with_names": None
            if separate_layer_norm_disabled
            else _giga_fsdp_rogue_layer_norm_modules_with_names_gigar(target_model),
            "giga_fsdp_layer_norm_module_cls": None
            if separate_layer_norm_disabled
            else resolve_norm_class(norm_type),
            "special_process_group_fn": _wrap_vision_process_group_fn(
                vision_modules, lambda x: None
            ),
        }
    elif llm_type == "giga_mix_causal_lm":
        all_modules, vision_modules = _giga_fsdp_modules_to_wrap_with_names_giga_mix(
            target_model
        )
        norm_type = getattr(
            getattr(target_model.config.llm_config, "config", None),
            "norm_type",
            "LlamaRMSNorm",
        )
        return {
            "activation_checkpointing_auto_wrap_policy": _activation_checkpointing_auto_wrap_policy_giga_mix(
                fsdp_config, target_model
            ),
            "giga_fsdp_modules_to_wrap_with_names": all_modules,
            "giga_fsdp_rogue_layer_norm_modules_with_names": _giga_fsdp_rogue_layer_norm_modules_with_names_giga_mix(
                target_model
            ),
            "giga_fsdp_layer_norm_module_cls": resolve_norm_class(norm_type),
            "special_process_group_fn": _wrap_vision_process_group_fn(
                vision_modules, _special_process_group_fn_giga_mix
            ),
        }
    else:
        raise ValueError(
            f"Gigafsdp wrapping for model type: {llm_type} is not implemented!"
        )
