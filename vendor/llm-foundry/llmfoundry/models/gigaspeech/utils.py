import typing as tp
from functools import partial

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel

from llmfoundry.models.layers.norm import resolve_norm_class


def convert_to_dict(value : tp.Optional[DictConfig]) -> dict:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value or {}


def prepare_speech_model_for_fsdp(model: PreTrainedModel) -> None:
    """FSDP wrap a decoder-only speech model.
    """
    decoder = model.get_decoder()
    decoder_block = decoder.model.layers[0]
    decoder_norm_type = getattr(
        getattr(decoder.model, "config", None),
        "norm_type",
        "LlamaRMSNorm",
    )
    decoder_norm_class = resolve_norm_class(decoder_norm_type)
    modality_adapter = model.get_modality_adapter()
    encoder = model.get_encoder()
    encoder_block = encoder.layers[0]
    lm_head = decoder.lm_head
    embeddings = decoder.model.embed_tokens

    modules = {
        'decoder': decoder,
        'decoder_block': decoder_block,
        'modality_adapter': modality_adapter,
        'encoder': encoder,
        'encoder_block': encoder_block,
        'lm_head': lm_head,
        "embeddings": embeddings,
    }

    for mod_name, module in modules.items():
        if module is None:
            raise ValueError(
                f'Unable to FSDP-wrap this model! `{mod_name}` does not ' +
                'follow common layer/weight naming conventions.')

    decoder_block_type = type(decoder_block)
    encoder_block_type = type(encoder_block)
    encoder_type = type(encoder)
    modality_adapter_type = type(modality_adapter)
    embeddings_type = type(embeddings)

    def is_encoder_or_decoder_block(
        module: torch.nn.Module,
        encoder_block_type: tp.Type,
        decoder_block_type: tp.Type,
        encoder_type: tp.Type,
        modality_adapter_type: tp.Type,
        embeddings_type: tp.Type,
        lm_head_obj: torch.nn.Module,
        decoder_norm_cls: tp.Type,
    ) -> bool:
        is_wrap_block = isinstance(module, encoder_block_type)
        is_wrap_block = is_wrap_block or isinstance(module, decoder_block_type)
        is_wrap_block = is_wrap_block or isinstance(module, encoder_type)
        is_wrap_block = is_wrap_block or isinstance(module, modality_adapter_type)
        is_wrap_block = is_wrap_block or isinstance(module, embeddings_type)
        is_wrap_block = is_wrap_block or (module == lm_head_obj)
        is_wrap_block = is_wrap_block or isinstance(module, decoder_norm_cls)
        return is_wrap_block

    def is_encoder_or_decoder_block_act_wrapper(
        module: torch.nn.Module,
        encoder_block_type: tp.Type,
        decoder_block_type: tp.Type,
        encoder_type: tp.Type,
        modality_adapter_type: tp.Type,
        embeddings_type: tp.Type,
        lm_head_obj: torch.nn.Module,
        decoder_norm_cls: tp.Type,
    ) -> bool:
        is_wrap_block = isinstance(module, encoder_block_type)
        is_wrap_block = is_wrap_block or isinstance(module, decoder_block_type)
        is_wrap_block = is_wrap_block or isinstance(module, encoder_type)
        is_wrap_block = is_wrap_block or isinstance(module, modality_adapter_type)
        is_wrap_block = is_wrap_block or isinstance(module, embeddings_type)
        is_wrap_block = is_wrap_block or (module == lm_head_obj)
        is_wrap_block = is_wrap_block or isinstance(module, decoder_norm_cls)
        return is_wrap_block

    is_encoder_or_decoder_block_wrap = partial(
        is_encoder_or_decoder_block,
        encoder_block_type=encoder_block_type,
        decoder_block_type=decoder_block_type,
        encoder_type=encoder_type,
        modality_adapter_type=modality_adapter_type,
        embeddings_type=embeddings_type,
        lm_head_obj=lm_head,
        decoder_norm_cls=decoder_norm_class,
    )
    model.fsdp_wrap_fn = is_encoder_or_decoder_block_wrap

    is_encoder_or_decoder_block_act_ckpt = partial(
        is_encoder_or_decoder_block_act_wrapper,
        encoder_block_type=encoder_block_type,
        decoder_block_type=int,
        embeddings_type=int,
        encoder_type=int,
        modality_adapter_type=int,
        lm_head_obj=1,
        decoder_norm_cls=int,
    )
    model.activation_checkpointing_fn = is_encoder_or_decoder_block_act_ckpt

    def is_cpu_offload_activation_checkpointing_fn_block(
        module: torch.nn.Module,
        encoder_type: tp.Optional[torch.nn.Module] = None,
    ):
        is_wrap_block = isinstance(module, encoder_type) if encoder_type else False
        return is_wrap_block

    is_cpu_offload_activation_checkpointing_fn_block_partial = partial(
        is_cpu_offload_activation_checkpointing_fn_block,
        encoder_type=None,
    )
    model.cpu_offload_activation_checkpointing_fn = is_cpu_offload_activation_checkpointing_fn_block_partial


def unfrozen_parameter_name_checker(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            should_be_unfreezed = (
                True if "encoder" in name
                or "modality" in name
                or "lora" in name else False
            )
            if not should_be_unfreezed:
                raise RuntimeError(f"Layer with name {name} should be freezed!")
