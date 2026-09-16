# Copyright 2022 MosaicML LLM Foundry authors
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
import copy
import inspect
import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import requests
from packaging import version
from pprint import pformat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# pytorch profiler triggers a lot of logs from "kineto"
#     https://stackoverflow.com/questions/74027680/pytorch-profiler-with-scheduler-prints-unwanted-message-at-step
os.environ["KINETO_LOG_LEVEL"] = "3"
os.environ["LOCAL_WORLD_SIZE"] = str(
    int(os.environ["WORLD_SIZE"]) // int(os.environ.get("NNODES", "1"))
)
os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
sys.path.insert(
    0, str(Path(os.path.realpath(__file__)).parent.parent.parent / "contrib")
)

# noqa: E402
import warnings
from typing import Dict, Optional

import torch

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# The supplied repository omitted its private Composer git submodule. Install
# the narrow public-Composer compatibility bridge before importing Composer or
# llmfoundry. Real private-fork symbols, when available, are never replaced.
from autoresearch.lmfoundry_compat import install as install_lmfoundry_compat

install_lmfoundry_compat()

from clearml import Task
from composer import Trainer
from composer.core import Evaluator
from composer.models import HuggingFaceModel
from composer.utils.timers import gigatimer
from composer.utils import (dist, format_name_with_dist, get_device,
                            get_torch_version, reproducibility)
from dotenv import dotenv_values, find_dotenv, load_dotenv
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry import (
    COMPOSER_MODEL_REGISTRY,
    build_finetuning_dataloader,
    build_text_denoising_dataloader,
)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.models.gigar.modelling_gigar import LlamaForCausalLM as GigarForCausalLM
from llmfoundry.utils.artifact_utils import copy_artifacts
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_icl_data_and_gauntlet,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    log_config,
    process_init_device,
    resolve_epoch_intervals,
    resolve_interval,
    update_batch_size_info,
    to_container,
)
from llmfoundry.utils.validators import validate_job_dispatcher_callback_config
from scripts.train.src.config_loader import (
    prepare_train_config,
    register_relative_config_loader,
    sync_peak_capacity_microbatchsize,
)


def _unsupported_v1_feature(*_args, **_kwargs):
    raise RuntimeError("This non-text feature is outside autoresearch v1")


class GigaMixForCausalLM(torch.nn.Module):
    """Type placeholder used only by multimodal padding inspection."""


build_audio_dataloader = _unsupported_v1_feature
build_evaluator = _unsupported_v1_feature
build_multimodal_dataloader = _unsupported_v1_feature
build_tts_dataloader = _unsupported_v1_feature
build_vision_dataloader = _unsupported_v1_feature
create_curriculum_dataset = _unsupported_v1_feature
wrap_dataloader = _unsupported_v1_feature
GigaLoraModel = _unsupported_v1_feature
_deepgemm_cache_manager = _unsupported_v1_feature


def validate_config(cfg: DictConfig):
    if "eval_first" in cfg:
        warnings.warn(
            "`eval_first` is deprecated, use `eval_before_train` instead. Forcing `eval_before_train` to `skip`."
        )
        cfg["eval_before_train"] = "skip"
    assert "eval_before_train" in cfg, "`eval_before_train` is required"
    assert cfg["eval_before_train"] in ["skip", "force", "regular"], (
        "`eval_before_train` must be one of [skip, force, regular]"
    )

    validate_config_model_dataloader(cfg)
    validate_job_dispatcher_callback_config(cfg)


def validate_config_model_dataloader(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        loaders.append(cfg.eval_loader)
    for loader in loaders:
        if loader.name == "text":
            if cfg.model.name in ["hf_prefix_lm", "hf_t5"]:
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " '
                    + 'dataloader. Please use the "text_denoising" dataloader to pre-train that model type.'
                )
        elif loader.name == "text_denoising":
            if cfg.model.name == "hf_causal_lm":
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text_denoising" '
                    + 'dataloader. Please use the "text" dataloader to pre-train that model type.'
                )
            if (
                loader.mixture_of_denoisers.decoder_only_format
                and cfg.model.name == "hf_t5"
            ):
                warnings.warn(
                    'Model type "hf_t5" requires `decoder_only_format` to be ``False``. '
                    + "Overriding `decoder_only_format` from ``True`` to ``False``."
                )
                loader.mixture_of_denoisers.decoder_only_format = False
            if (
                not loader.mixture_of_denoisers.decoder_only_format
            ) and cfg.model.name == "hf_prefix_lm":
                warnings.warn(
                    'Model type "hf_prefix_lm" requires `decoder_only_format` to be ``True``. '
                    + "Overriding `decoder_only_format` from ``False`` to ``True``."
                )
                loader.mixture_of_denoisers.decoder_only_format = True

    if "icl_tasks" in cfg:
        # we first look if the cache directory is specified
        if "cached_datasets_directory_template" in cfg:
            # by default, we take it from config
            pass
        elif "CACHED_DATASETS_DIRECTORY_TEMPLATE" in os.environ:
            # if the value in config is not specified, we take it from the ENV variable
            cfg["cached_datasets_directory_template"] = os.environ[
                "CACHED_DATASETS_DIRECTORY_TEMPLATE"
            ]
        else:
            # if neither is specified, we do not use caching
            cfg["cached_datasets_directory_template"] = None

        if cfg.model.name == "hf_t5":
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (
        cfg.model.get("fc_type", "torch") != "te"
        and "te" not in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
        and "fp8" in cfg.precision
    ):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            + "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if cfg.get("fsdp_config", None) and cfg.get("gigafsdp_config", None):
        raise RuntimeError(
            "Passed both 'fsdp_config' and 'gigafsdp_config'. Choose one setup."
        )

    if cfg.get("gigafsdp_config", None):
        sac_strategy = cfg.gigafsdp_config.get("sac_strategy", 0)
        assert sac_strategy == 0, (
            f"{sac_strategy=} is currently not supported. Please run with sac_strategy = 0"
        )

    if cfg.get("gigafsdp_config", None):
        cfg.fsdp_config = cfg.gigafsdp_config
        cfg.fsdp_config.gigafsdp_enabled = True

    if cfg.model.get("fc_type", "torch") == "te" or "te" in cfg.model.get(
        "ffn_config", {}
    ).get("ffn_type", "mptmlp"):
        fsdp_config = cfg.get("fsdp_config", None)
        act_ckpt = fsdp_config.get("activation_checkpointing", False)
        act_ckpt_reentrant = fsdp_config.get("activation_checkpointing_reentrant", True)
        if fsdp_config is not None and act_ckpt and not act_ckpt_reentrant:
            warnings.warn(
                "`te.Linear` layers do not support activation_checkpointing with "
                + "`activation_checkpointing_reentrant = False`. "
                + "Setting cfg.fsdp_config.activation_checkpointing_reentrant=True."
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = True

    def get_llm_params(cfg: DictConfig):
        if cfg.get("vision_mode", False):
            llm_params = cfg.model.llm.params
        elif cfg.get("speech_mode", False):
            llm_params = cfg.model.decoder_config
        else:
            llm_params = cfg.model.config_overrides
        return llm_params

    if cfg.get("fsdp_config", False) and cfg.fsdp_config.get(
        "activation_checkpointing", False
    ):
        if cfg.model.get("teacher_model", False) or cfg.model.get(
            "student_model", False
        ):
            assert (
                not cfg.model.teacher_model.get("config_overrides", False)
                or cfg.model.teacher_model.config_overrides.activation_checkpoint_layers_num
                == 0
            )
            activation_checkpoint_layers_num = (
                cfg.model.student_model.config_overrides.get(
                    "activation_checkpoint_layers_num", None
                )
            )
            num_hidden_layers = cfg.model.student_model.config_overrides.get(
                "num_hidden_layers", None
            )

            cfg.model.student_model.config_overrides.activation_checkpoint_layers_num = (
                activation_checkpoint_layers_num or num_hidden_layers
            )

            assert cfg.model.student_model.config_overrides.activation_checkpoint_layers_num, (
                "Pass 'activation_checkpoint_layers_num' or 'num_hidden_layers' for student_model with activation_checkpointing=true"
            )
        else:
            llm_params = get_llm_params(cfg)

            if llm_params.get("activation_checkpoint_layers_num", None) is None:
                warnings.warn(
                    "FSDP activation_checkpointing=true and activation_checkpoint_layers_num is None -> will ckpt all layers"
                )
                if llm_params.get("num_hidden_layers", False):
                    llm_params.activation_checkpoint_layers_num = (
                        llm_params.num_hidden_layers
                    )
                else:
                    llm_params.activation_checkpoint_layers_num = (
                        cfg.model.num_hidden_layers
                    )

    if "te" in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp"):
        warnings.warn(
            "`te.LayerNormMLP` requires has issues with torch._dynamo. "
            + "Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager."
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore

    if cfg.get("fsdp_config", False) and cfg.fsdp_config.get("gigafsdp_enabled", False):
        warnings.warn(
            "GigaFSDP supports model output only in the form of tensors or a tuple of tensors"
        )
        if cfg.model.get("teacher_model", False) or cfg.model.get(
            "student_model", False
        ):
            if cfg.model.teacher_model.get("config_overrides", False):
                cfg.model.teacher_model.config_overrides.return_dict = False
            cfg.model.student_model.config_overrides.return_dict = False
        else:
            llm_params = get_llm_params(cfg)
            llm_params.return_dict = False

        if cfg.get("enable_async_tp", False):
            cfg.fsdp_config.all_reduce_grads_across_model_parallel_group = True

        if (
            cfg.fsdp_config.get("state_dict_type", False)
            and cfg.fsdp_config.state_dict_type != "local"
        ):
            raise ValueError("GigaFSDP supports only state_dict_type=local")

        if (
            cfg.fsdp_config.get("load_monolith_rank0_only", False)
            and cfg.fsdp_config.load_monolith_rank0_only
        ):
            raise ValueError("GigaFSDP supports only load_monolith_rank0_only=false")

    if (
        cfg.get("enable_async_tp", False)
        and cfg.get("fsdp_config", False)
        and not cfg.fsdp_config.get("gigafsdp_enabled", False)
    ):
        raise NotImplementedError(
            "AsyncTP works only with GigaFSDP now. Set `enable_async_tp=false` or enable gigafsdp."
        )

    if cfg.get("model_type", False) == "giga_tts" and (
        cfg.get("enable_async_tp", False) or cfg.get("sp_size", 1) > 1
    ):
        raise ValueError(
            f"Invalid tp_size={cfg.get('tp_size', 1)} (enable_async_tp={cfg.get('enable_async_tp', False)})"
            f" and sp_size={cfg.get('sp_size', 1)} for model_type={cfg.model_type}"
        )

    if cfg.get("enable_async_tp", False) and cfg.model.get("lora", False):
        raise ValueError(
            "Lora not works enable_async_tp now. Set `enable_async_tp=false`"
        )

    model_overrides_cfg = cfg.model.get("config_overrides") or {}

    if model_overrides_cfg.get("use_float8_grouped_gemm", False):
        if not model_overrides_cfg.get("use_new_expert_weight_layout", False):
            raise RuntimeError(
                "Only new weight layout is supported for float8 MoE training."
            )


def validate_batch_sizes(cfg: DictConfig):
    gbs = cfg.get("global_train_batch_size", None)
    mbs = cfg.get("device_train_microbatch_size", "auto")

    if gbs is not None and mbs != "auto":
        tp_sp_group_size = (
            dist.get_tp_sp_group_size() or 1
        )  # TODO, check correctness of dist.get_tp_sp_group_size()
        tp_sp_group_size = max(
            tp_sp_group_size,
            (dist.get_tp_group_size() or 1) * (dist.get_sp_group_size() or 1),
        )
        if dist.get_world_size() % tp_sp_group_size:
            raise ValueError(
                "Total number of gpus should be divisible by tp_sp_group_size."
            )
        accum_batch_size = mbs * dist.get_world_size() // tp_sp_group_size
        if gbs % accum_batch_size:
            raise ValueError(
                f"global_train_batch_size should be divisible by (n_gpus // tp_sp_group_size) * device_train_microbatch_size. gbs {gbs} % accum_batch_size {accum_batch_size}"
            )


def validate_sp_size(cfg: DictConfig):
    sp_size = cfg.get("sp_size", 1)
    # duplicate sp_size to fsdp_config
    if isinstance(sp_size, list):
        fsdp_config = cfg.get("fsdp_config", None)
        min_sp_size = min(sp_size, key=lambda x: x["sp_size"])["sp_size"]
        grad_accum = fsdp_config["gradient_accumulation_steps"]
        gbs = cfg.get("global_train_batch_size")
        if gbs is None:
            raise ValueError(
                "Configuration 'global_train_batch_size' is required but not provided."
            )

        mbs = cfg.get("device_train_microbatch_size")
        if mbs is None:
            raise ValueError(
                "Configuration 'device_train_microbatch_size' is required but not provided."
            )

        ngpus = dist.get_world_size()
        if ngpus is None:
            raise ValueError("World size from 'dist' is required but not provided.")

        tp_size = cfg.get("tp_size")
        if tp_size is None:
            raise ValueError("Configuration 'tp_size' is required but not provided.")

        if gbs * min_sp_size * tp_size / (mbs * ngpus) > grad_accum:
            raise ValueError(
                f"Provided sp_size {min_sp_size=} is not availiable with current grad_accum {grad_accum=} {gbs * min_sp_size * tp_size / (mbs * ngpus)}"
            )


def get_padding_scale(model: PreTrainedModel) -> float:
    r"""Get padding scale for dataloader based on the model's SP split type.

    If configured `sp_split_type` is `zigzag`, the scale is `2`, otherwise defaults to `1`.

    For `Gigar` and `GigaMix` we get the attribute value directly, for other multi-modal
    architectures it is expected for them to have `Gigar` or `GigaMix` instance as
    one of the attributes.

    Args:
        model (PreTrainedModel): Initialized model object.

    Returns:
        float: Determined padding scale.
    """
    padding_scale = 1
    if hasattr(model, "config") and hasattr(model.config, "sp_split_type"):
        # Gigar or GigaMix model instance case when we have `sp_spit_type` attribute directly
        if model.config.sp_split_type == "zigzag":
            padding_scale = 2
    else:
        # Non Gigar or GigaMix model instalce that has Gigar or GigaMix as one of the attributes
        for attr_name, attr_val in model.__dict__.items():
            if isinstance(attr_val, GigarForCausalLM) or isinstance(
                attr_val, GigaMixForCausalLM
            ):
                if getattr(model, attr_name).config.sp_split_type == "zigzag":
                    padding_scale = 2

    return padding_scale


def build_composer_model(
    model_cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, **kwargs: Dict
):
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_cfg.name}")
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer, **kwargs)


def build_composer_peft_model(
    lora_cfg: DictConfig,
    model_cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs: Dict,
) -> HuggingFaceModel:
    """
    Строит и настраивает модель c использованием конфигурации модели, токенизатора и параметров LoRA.
    В зависимости от значения `cfg.tp_size` функция применяет либо стандартную модель LoRA, либо модель LoRA
    с поддержкой tensor parallelism (TP). Также функция помечает определенные модули для обертывания в FSDP
    (Fully Sharded Data Parallel), если они соответствуют заданным условиям.
    Параметры:
    ----------
    model_cfg : dict
        Конфигурация модели, используемая для построения модели.
    tokenizer : PreTrainedTokenizer
        Токенизатор, используемый для обработки входных данных.
    Возвращает:
    -----------
    model : torch.nn.Module
        Модель, настроенная c учетом параметров LoRA и FSDP.
    -----------
    - Если `cfg.tp_size == 1`, используется стандартная модель LoRA.
    - Если `cfg.tp_size > 1`, используется модель LoRA c поддержкой tensor parallelism.
    - Модули, которые не имеют дочерних модулей, имеют атрибут `weight`, помечаются для обертывания в FSDP.
    """
    unfreeze_token_embeds = lora_cfg.pop("unfreeze_token_embeds", False)

    lora_config = LoraConfig(**lora_cfg)
    model = build_composer_model(model_cfg, tokenizer, **kwargs)

    if not hasattr(model.config, "lora_config"):
        model.config = lora_config

    if hasattr(model.model, "_language_model"):  # multimodal
        model.model._language_model = GigaLoraModel(
            model.model._language_model,
            lora_config,
            adapter_name="default",
            model_cfg=model.config.llm_config.config,
        ).model
    else:
        model.model = GigaLoraModel(
            model.model, lora_config, adapter_name="default", model_cfg=model.config
        ).model

    for _, module in model.named_modules():
        if (
            len(list(module.named_children())) == 0
            and hasattr(module, "weight")
            and module.weight.requires_grad
        ):
            module._fsdp_wrap = True

    if not hasattr(model, "gigafsdp_model_normalize_fqns"):
        model.gigafsdp_model_normalize_fqns = lambda param_name: param_name.replace(
            ".base_layer", ""
        )
    else:
        available_function = model.gigafsdp_model_normalize_fqns
        model.gigafsdp_model_normalize_fqns = lambda param_name: available_function(
            param_name
        ).replace(".base_layer", "")

    parts_to_unfreeze = []
    if model_cfg.name == "gigaspeech_causal_lm":
        parts_to_unfreeze.extend(["model.encoder", "model.modality_adapter"])
    if model_cfg.name == "gigavision_causal_lm" and unfreeze_token_embeds:
        parts_to_unfreeze.append("model._language_model.model.embed_tokens")
    if (
        model_cfg.name == "gigar_causal_lm" or model_cfg.name == "giga_mix_causal_lm"
    ) and unfreeze_token_embeds:
        parts_to_unfreeze.append("model.model.embed_tokens")

    def get_list_attr(obj, attr_list):
        if len(attr_list) == 1:
            return getattr(obj, attr_list[0])
        attr, attr_list = attr_list[0], attr_list[1:]
        return get_list_attr(getattr(obj, attr), attr_list)

    for part in parts_to_unfreeze:
        logger.info(f"Unfreezing {part} ...")
        for param in get_list_attr(model, part.split(".")).parameters():
            param.requires_grad = True

    tp_size = cfg.get("tp_size")

    if tp_size > 1 or lora_cfg["peft_type"] == "LoRA-FA":
        logger.info("freezed lora_A")
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False

    return model


def print_trainable_parameters(model: torch.nn.Module) -> None:
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    B = 1e9
    logger.info(
        f"trainable params: {trainable_params / B:.2}B ({trainable_params}) || all params: {all_param / B:.2}B ({all_param}) || trainable: {trainable_params / all_param:.2%}"
    )


def build_dataloader(
    cfg: DictConfig,
    tokenizer: Optional[PreTrainedTokenizerBase],
    device_batch_size: int,
) -> torch.utils.data.DataLoader:
    if cfg.name == "text":
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "text_denoising":
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "finetuning":
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "vision":
        return build_vision_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "multimodal":
        return build_multimodal_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "speech":
        return build_audio_dataloader(
            cfg,
            device_batch_size,
            use_wav_filepaths=cfg.get("use_wav_filepaths", True),
        )
    elif cfg.name == "tts":
        return build_tts_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f"Not sure how to build dataloader with config: {cfg}")


def build_profiler(profiler_cfg: DictConfig):
    """
    Profiler config example:

    ```yaml
    profiler:
        use_profiler: true
        annotation:
          enable: true
          backend: record_function
        trace_handlers:
        - name: json
          params:
            # composer.profiler.JSONTraceHandler kwargs
            composer_trace_dir: ...

        cyclic_schedule:
            # composer.profiler.cyclic_schedule kwargs
            wait: 0
            warmup: 1
            active: 4
            repeat: 1

        profiler_overrides:
            # composer.profiler.Profiler kwargs
            ...
    ```
    """
    from composer.profiler import JSONTraceHandler, Profiler, cyclic_schedule
    from composer.utils.profiler_annotation import profiler_annotation, BackendType

    if not profiler_cfg.get("use_profiler", False):
        profiler_annotation.config_enabled = False
        return None

    annotation_cfg = profiler_cfg.get("annotation") or {}
    annotation_enabled = bool(annotation_cfg.get("enable", False))
    annotation_backend_cfg = annotation_cfg.get(
        "backend", BackendType.RECORD_FUNCTION.value
    )

    try:
        annotation_backend = BackendType(annotation_backend_cfg)
    except ValueError as exc:
        available_backends = ", ".join([backend.value for backend in BackendType])
        raise ValueError(
            f"Unknown profiler annotation backend `{annotation_backend_cfg}`.\nAvailable backends: {available_backends}"
        ) from exc

    profiler_annotation.configure(
        backend=annotation_backend,
        config_enabled=annotation_enabled,
    )

    for requied_value in ["cyclic_schedule", "trace_handlers"]:
        assert requied_value in profiler_cfg, (
            f'"{requied_value}" not specified in config "profiler" section, but it is requred'
        )

    assert len(profiler_cfg.trace_handlers) == 1, (
        "only 1 trace handler is available now"
    )
    trace_handler_cfg = profiler_cfg.trace_handlers[0]
    assert trace_handler_cfg.name == "json", (
        'only "json" trace handler is available now'
    )

    ranks_to_profile = profiler_cfg.get("ranks_to_profile", None)

    if ranks_to_profile is None or str(ranks_to_profile).strip().lower() == "all":
        ranks_to_profile = None
    elif isinstance(ranks_to_profile, (list, ListConfig)) and all(
        isinstance(rank, int) and rank >= 0 for rank in ranks_to_profile
    ):
        ranks_to_profile = list(set(to_container(ranks_to_profile)))
    else:
        raise ValueError(
            f'Invalid value for "ranks_to_profile": {ranks_to_profile}. It should be either "all" or a list of ranks.'
        )

    trace_handlers = [
        JSONTraceHandler(**trace_handler_cfg.params, ranks_to_profile=ranks_to_profile)
    ]

    profiler = Profiler(
        trace_handlers=trace_handlers,
        ranks_to_profile=ranks_to_profile,
        schedule=cyclic_schedule(**profiler_cfg.cyclic_schedule),
        **profiler_cfg.profiler_overrides,
    )
    return profiler


def get_run_meta_dir(cfg: DictConfig) -> Path:
    meta_dir = None
    log_cfg = cfg.get("loggers") or {}
    if log_cfg:
        log_dir = log_cfg["tensorboard"]["log_dir"]
        meta_dir = Path(log_dir) / cfg.run_name / "_signal_files"
        meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def recreate_signal_file(signal_file: Path) -> None:
    """create or replace startup signal file"""
    signal_file.unlink(missing_ok=True)
    signal_file.touch(exist_ok=False)


def apply_fp8_config(model, model_cfg, fp8_config):
    import re

    from llmfoundry.models.ops.float8.gemm import FP8Linear as TEFP8Linear
    from llmfoundry.models.ops.float8.utils.fp8_utils import (
        add_shared_quantizers,
        swap_linear_layers,
    )

    def _is_sm89_or_later():
        # Float8 is only supported on SM89 or later (H100+ GPUs)
        return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
            8,
            9,
        )

    if not _is_sm89_or_later():
        raise RuntimeError(
            "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
        )

    def module_filter_fn(mod: torch.nn.Module, fqn: str, include_layers: str):
        if re.search(include_layers, fqn):
            return True
        return False

    # To flexibly configure linear layers wrapped in fp8 and define activation quantizations
    # using pre-forward hooks that can be shared between multiple projections,
    # we use the corresponding regular expressions `include_linear_layers` and
    # `include_shared_quantizers`.
    #
    # example config:
    # fp8_config:
    #     include_linear_layers: ".*\\.(down_proj|gate_proj|up_proj|q_proj|k_proj|v_proj|o_proj)$"
    #     include_shared_quantizers: ".*\\.(mlp)$"
    #
    include_linear_layers = fp8_config.get("include_linear_layers", None)
    include_shared_quantizers = fp8_config.get("include_shared_quantizers", None)
    assert include_linear_layers is not None and len(include_linear_layers) > 0, (
        "include_linear_layers is required"
    )

    if fp8_config.get("float8_dense_fused_swiglu_quant", False):
        assert not model_cfg.get("config_overrides", {}).get("fused_mlp", False)
        assert not model_cfg.get("fused_mlp", False)
        assert not model_cfg.get("config_overrides", {}).get(
            "fused_mlp_checkpoint_lvl", 0
        )
        assert not model_cfg.get("fused_mlp_checkpoint_lvl", 0)
        # SepProductSiluQuant.apply expects gate/up/down projections to be FP8-wrapped.
        # If any of the three is missing from include_linear_layers, the kernel will get
        # bf16 inputs and silently produce wrong (or NaN) results.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert re.search(include_linear_layers, f"layer.mlp.{proj}"), (
                f"float8_dense_fused_swiglu_quant=True requires '{proj}' to be matched by "
                f"fp8_config.include_linear_layers (got: {include_linear_layers!r})"
            )

    assert _is_sm89_or_later()

    def from_float_func(m: torch.nn.Module) -> torch.nn.Module:
        return TEFP8Linear.swap_module(m)

    swap_linear_layers(
        model,
        from_float_func=from_float_func,
        module_filter_fn=partial(
            module_filter_fn, include_layers=include_linear_layers
        ),
    )

    if include_shared_quantizers is not None and len(include_shared_quantizers) > 0:
        add_shared_quantizers(
            model,
            module_filter_fn=partial(
                module_filter_fn, include_layers=include_shared_quantizers
            ),
        )

    logger.info(f"FP8 Model after wrapping: {model}")


def setup_loggers(cfg: DictConfig) -> list:
    loggers = []
    loggers_cfg = cfg.get("loggers", {})

    console_logger_cfg = dict(loggers_cfg.get("console") or {})
    cfg.log_to_console = bool(console_logger_cfg.get("enabled", cfg.get("log_to_console", True)))
    cfg.console_log_interval = console_logger_cfg.get("log_interval", cfg.get("console_log_interval", "1ba"))
    cfg.console_stream = console_logger_cfg.get("stream", cfg.get("console_stream", "stderr"))
    cfg.log_traces = bool(console_logger_cfg.get("log_traces", cfg.get("log_traces", False)))

    for name, logger_cfg in loggers_cfg.items():
        if name == "console":
            continue

        logger_kwargs = dict(logger_cfg) if logger_cfg is not None else {}
        enabled = bool(logger_kwargs.pop("enabled", True))

        if not enabled:
            continue

        loggers.append(build_logger(name, logger_kwargs))
    return loggers


def setup_gigatimer(cfg: DictConfig) -> None:
    timers_cfg = cfg.get("timers", {})
    if timers_cfg:
        if "timers_threshold" in timers_cfg:
            gigatimer.set_timers_threshold(int(timers_cfg["timers_threshold"]))

        interval_cfg = timers_cfg.get("interval", gigatimer.interval)
        if isinstance(interval_cfg, str):
            raise ValueError(
                f"`timers.interval` must be an integer, got: {interval_cfg!r}"
            )
        interval_steps = int(interval_cfg)
        gigatimer.set_interval(interval_steps)


def main(cfg: DictConfig):
    # Copy fields of existing config
    whole_config = copy.copy(cfg)
    # Check for incompatibilities between the model and data loaders
    # Check for compatibility of options
    validate_config(cfg)

    if (cfg.model.get("config_overrides") or {}).get("use_float8_grouped_gemm", False):
        # Check for compatibility of DeepGEMM version, used only in FP8 training.
        from llmfoundry.models.ops.float8.utils.fp8_utils import _sm90_and_deepgemm_stable_version_check
        _sm90_and_deepgemm_stable_version_check()

    if cfg["varlen_input"]:
        if "train_loader" in cfg:
            cfg.train_loader.batch_type = "flash_attn"
        if "eval_loader" in cfg:
            cfg.eval_loader.batch_type = "flash_attn"

    use_dummy_dataset = (
        cfg.get("train_loader", {}).get("dataset", {}).get("use_dummy", False)
    )

    cfg.model["varlen_input"] = cfg["varlen_input"]

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )

    cfg.dist_timeout = cfg.get("dist_timeout", 600.0)

    tp_size = cfg.get("tp_size", 1)
    sp_size = cfg.get("sp_size", 1)
    ep_size = cfg.get("ep_size", 1)

    assert tp_size >= 1, f"tp_size size should be >= 1, provided tp_size = {tp_size}"
    assert sp_size >= 1 if isinstance(sp_size, int) else len(sp_size) >= 1, (
        f"sp_size size should be >= 1, provided sp_size = {sp_size}"
    )
    assert ep_size >= 1, f"ep_size size should be >= 1, provided ep_size = {ep_size}"

    dist.initialize_dist(
        get_device(None),
        tp_size=tp_size,
        sp_size=sp_size,
        ep_size=ep_size,
        timeout=cfg.dist_timeout,
        enable_async_tp=cfg.get("enable_async_tp", False),
    )
    overrides_cfg = cfg.get("model", {})
    if hasattr(overrides_cfg, "student_model"):
        overrides_cfg = overrides_cfg.get("student_model", {}).get(
            "config_overrides", {}
        )
    else:
        overrides_cfg = overrides_cfg.get("config_overrides", {})
    if overrides_cfg.get("use_float8_grouped_gemm", False) or cfg.model.get("use_float8_grouped_gemm", False):
        from llmfoundry.models.ops.float8.utils.fp8_utils import _sm90_and_deepgemm_stable_version_check
        # Check for compatibility of DeepGEMM version, used only in FP8 training.
        _sm90_and_deepgemm_stable_version_check()
        # Initialize DeepGEMM cache manager on each node on zero rank
        local_rank = dist.get_local_rank()
        _deepgemm_cache_manager(local_rank)

    reproducibility.seed_all(cfg.seed)

    # Run Name
    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("RUN_NAME", "llm")

    # Get batch size info
    cfg = update_batch_size_info(cfg)
    cfg = sync_peak_capacity_microbatchsize(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get("fsdp_config", None)
    if fsdp_config is not None:
        fsdp_config.sp_size = cfg.get("sp_size", 1)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None
    assert isinstance(fsdp_config, Dict) or fsdp_config is None
    if dist.get_world_size() == 1 and fsdp_config is not None:
        raise RuntimeError("Single-GPU mode is not supported.")

    if (
        tp_size > 1
        and not fsdp_config.get("gigafsdp_enabled", False)
        and get_torch_version() >= version.parse("2.4")
    ):
        raise NotImplementedError(
            "Tensor parallel not supported for torch 2.4 FSDP yet"
        )

    init_context = process_init_device(cfg.model, fsdp_config)

    if cfg.get("enable_async_tp", False):
        cfg.model.enable_async_tp = True

    # startup signal
    signal_files_dir = get_run_meta_dir(cfg) if os.environ["RANK"] == "0" else None
    if signal_files_dir:
        recreate_signal_file(signal_files_dir / "startup")

    # build tokenizer
    tokenizer = None
    if cfg.get("tokenizer", None):
        tokenizer = build_tokenizer(cfg.tokenizer)
    else:
        assert use_dummy_dataset, "Tokenizer must be provided"
        logger.warning("Used `use_dummy_dataset=True`. Tokenizer is not initialized.")

    # Build Model
    logger.info("Initializing model...")
    with init_context:
        lora_cfg = None
        if cfg.model.get("lora", None) is not None:
            lora_cfg = cfg.model.lora.args
        elif cfg.get("lora_cfg", None) is not None:
            lora_cfg = cfg.lora_cfg

        composer_model_kwargs = {}
        if cfg.model.name == "gigaspeech_causal_lm":
            composer_model_kwargs.update(
                {
                    "om_generation_config": cfg.generation_config,
                }
            )

        if lora_cfg is not None:
            logger.info("Start building model with LORA...")
            model = build_composer_peft_model(
                lora_cfg=lora_cfg,
                model_cfg=cfg.model,
                tokenizer=tokenizer,
                **composer_model_kwargs,
            )
        else:
            model = build_composer_model(cfg.model, tokenizer, **composer_model_kwargs)

        if cfg.get("fp8_config", None):
            assert tp_size == 1, "FP8 supports only tp_size=1 now."
            apply_fp8_config(model, cfg.model, cfg.fp8_config)
            if cfg.model.get("config_overrides", {}).get(
                "float8_dense_fused_swiglu_quant", False
            ):
                logger.warning(
                    "When float8_dense_fused_swiglu_quant=True, all linear layers included in SwiGLU"
                    + " (gate_proj, up_proj, and down_proj) are expected to be supported in fp8."
                )

        if fsdp_config and fsdp_config.get("gigafsdp_enabled", False):
            model.gigafsdp_model_setup_with_check(fsdp_config)
        print_trainable_parameters(model)
        if os.environ["RANK"] == "0":  # print model for debug
            print("Model", model)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    parameter_count_bounds = cfg.get("parameter_count_bounds")
    if parameter_count_bounds is not None:
        if not (
            isinstance(parameter_count_bounds, (list, tuple, ListConfig))
            and len(parameter_count_bounds) == 2
        ):
            raise ValueError("parameter_count_bounds must be [minimum, maximum]")
        minimum, maximum = map(int, parameter_count_bounds)
        if not minimum <= cfg.n_params <= maximum:
            raise ValueError(
                f"Model has {cfg.n_params:,} parameters; expected "
                f"{minimum:,}..{maximum:,}"
            )
    if os.environ["RANK"] == "0":
        print(f"[autoresearch] n_params={cfg.n_params}", flush=True)

    # NOTE(m1kol): Add padding scale to account for increased `zigzag` split chunking.
    padding_scale = get_padding_scale(model)

    cfg.train_loader["padding_scale"] = padding_scale
    if "eval_loader" in cfg:
        cfg.eval_loader["padding_scale"] = padding_scale

    save_folder = cfg.get("save_folder", None)
    # Copying necessary files to artifact folder
    if os.environ["RANK"] == "0" and save_folder:
        # Creating artifact path near save_folder
        filled_save_directory = format_name_with_dist(save_folder, cfg.run_name)
        artifact_directory = os.path.join(filled_save_directory, "artifacts")
        copy_artifacts(whole_config, artifact_directory, use_dummy_dataset)

    if cfg.train_loader.get("curriculum", False):
        logger.info("Create curriculum dataset, stream proportions will be ignored...")
        create_curriculum_dataset(cfg)
        dist.barrier()

    # Dataloaders
    logger.info("Building train loader...")
    train_loader = build_dataloader(
        cfg.train_loader,
        tokenizer,
        cfg.device_train_batch_size,
    )

    if cfg.model.name == "gigaspeech_causal_lm":
        train_loader = wrap_dataloader(train_loader, cfg.audio_token_id)

    evaluators = None
    if not use_dummy_dataset:
        evaluators = []
        logger.info("Building eval loader...")
        if "eval_loader" in cfg:
            assert model.val_metrics is not None
            dataloader = build_dataloader(
                cfg.eval_loader, tokenizer, cfg.device_eval_batch_size
            )
            eval_loader = Evaluator(
                label="eval",
                dataloader=dataloader,
                metric_names=list(
                    cfg.get("eval_loader_metric_names", model.val_metrics.keys())
                ),
            )
            evaluators.append(eval_loader)

        logger.info("Building eval loaders...")
        if "eval_loaders" in cfg:
            assert model.train_metrics is not None
            for k in cfg.eval_loaders.keys():
                eval_loader = build_evaluator(
                    cfg, k, tokenizer, model, cfg.audio_token_id
                )
                evaluators.append(eval_loader)

        eval_gauntlet_config = cfg.get("eval_gauntlet", None)

        eval_gauntlet_callback = None
        if "icl_tasks" in cfg:
            icl_evaluators, _, eval_gauntlet_callback = build_icl_data_and_gauntlet(
                cfg.icl_tasks,
                eval_gauntlet_config,
                tokenizer,
                cfg.max_seq_len,
                cfg.device_eval_batch_size,
                cached_datasets_directory_template=cfg[
                    "cached_datasets_directory_template"
                ],
            )
            evaluators.extend(icl_evaluators)

    profiler = None
    # DEPRECATED "use_profiler"
    if "profiler" in cfg and "use_profiler" in cfg:
        cfg.profiler.use_profiler = cfg.use_profiler
        logger.warning(
            "'use_profiler' is deprecated. It will be removed in next versions. Use 'profiler.use_profiler' instead."
        )

    assert cfg.get("callbacks", {}).get("online_profiler") is None or (
        "profiler" in cfg and cfg.profiler.get("use_profiler", False)
    ), (
        "`callbacks.online_profiler` requires enabled profiler. Set profiler.use_profiler: true before enabling `callbacks.online_profiler`."
    )

    if "profiler" in cfg:
        profiler = build_profiler(cfg.profiler)

    # Optimizer param group patterns (for scheduler `param_groups` / `param_str_match`)
    opt_param_groups = cfg.optimizer.get("param_groups")
    optimizer_param_str_matches = (
        [str(p.param_str_match) for p in opt_param_groups] if opt_param_groups else None
    )

    # Optimizer
    optimizer_name: str = cfg.optimizer.pop("name")
    optimizer_cfg = cfg.optimizer
    optimizer = build_optimizer(model, optimizer_name, optimizer_cfg)

    # Convert epoch-based intervals to batches before building scheduler
    resolve_epoch_intervals(cfg, logger=logger)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler, optimizer_param_str_matches)

    # Loggers
    loggers = setup_loggers(cfg)

    # Timers
    setup_gigatimer(cfg)

    # Callbacks
    callbacks_cfg = cfg.get("callbacks") or {}
    callbacks = []
    for name, callback_cfg in callbacks_cfg.items():
        callback_kwargs = dict(callback_cfg) if callback_cfg is not None else {}
        callback_obj = build_callback(name, callback_kwargs)
        callbacks.append(callback_obj)
    if not use_dummy_dataset:
        if eval_gauntlet_callback is not None:
            callbacks.append(eval_gauntlet_callback)

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get("algorithms") or {}).items()
    ]

    inner_model_cfg = cfg.get("model") or {}
    model_overrides_cfg = inner_model_cfg.get("config_overrides") or {}
    if inner_model_cfg.get("student_model"):
        model_overrides_cfg = (
            inner_model_cfg.get("student_model").get("config_overrides") or {}
        )

    if model_overrides_cfg.get("aux_loss_free", False):
        algorithms.append(
            build_algorithm(
                "update_bias",
                {
                    "moe_scale_log_interval": cfg.get("moe_log_interval", None),
                },
            )
        )

    if (
        model_overrides_cfg.get("gating_type") is not None
        and model_overrides_cfg.get("gating_type") == "dynamic"
    ):
        algorithms.append(build_algorithm("update_dynamic_router", {}))

    dummy_uniform_with_sequence_warmup_assert_message = (
        "gating_type: dummy_uniform is usually used for performance testing, "
        "which doesn't go well with sequence_warmup. To bypass the assert, "
        "please provide ignore_dummy_uniform_with_sequence_warmup: true in "
        "model_overrides_cfg."
    )
    assert (
        model_overrides_cfg.get("gating_type") != "dummy_uniform"
        or "sequence_warmup" not in cfg.get("algorithms", {})
    ) or model_overrides_cfg.get("ignore_dummy_uniform_with_sequence_warmup", False), (
        dummy_uniform_with_sequence_warmup_assert_message
    )

    # Set autoresume default on if possible
    save_latest_filename = cfg.get("save_latest_filename", "latest-rank{rank}.pt")
    save_filename = cfg.get("save_filename", "ep{epoch}-ba{batch}-rank{rank}.pt")
    load_path = cfg.get("load_path", None)
    teacher_load_path = cfg.get("teacher_load_path", None)
    if tp_size == 1:
        python_log_level = cfg.get(
            "python_log_level",
            "debug" if (os.environ["RANK"] == "0") else "warn",
        )

        if ep_size > 1:
            if load_path is not None:
                is_single_path = isinstance(load_path, str)
                is_safetensors = (
                    is_single_path and (Path(load_path) / "config.json").is_file()
                )

                if not is_safetensors:
                    assert isinstance(load_path, ListConfig), (
                        f"EP size is set to {ep_size} and load_path is provided. Got single entry, "
                        + "but expected to be list!"
                    )
                    assert len(load_path) == ep_size, (
                        f"Got {len(load_path)} checkpoints, but expected to be the same as EP size {ep_size}."
                    )
                    load_path = load_path[dist.get_ep_group_rank()]
                else:
                    assert isinstance(load_path, str), (
                        "safetensors ckpt should be one string in ep>1 case"
                    )

    else:
        python_log_level = cfg.get(
            "python_log_level",
            "debug"
            if (os.environ["RANK"] == "0" or dist.get_fsdp_group_rank() == 0)
            else "warn",
        )

        if ep_size == 1:
            if load_path is not None:
                is_single_path = isinstance(load_path, str)
                is_safetensors = (
                    is_single_path and (Path(load_path) / "config.json").is_file()
                )
                if is_safetensors:
                    assert isinstance(load_path, str), (
                        "safetensors ckpt should be one string in  tp>1 case"
                    )
                else:
                    assert isinstance(load_path, ListConfig), (
                        f"TP size is set to {tp_size} and load_path is provided. Got single entry, but expected to be list!"
                    )
                    assert len(load_path) == tp_size, (
                        f"Got {len(load_path)} checkpoints, but expected to be the same as TP size {tp_size}."
                    )
                    load_path = load_path[dist.get_tp_group_rank()]
            if teacher_load_path is not None:
                assert isinstance(teacher_load_path, ListConfig), (
                    f"TP size is set to {tp_size} and teacher_load_path is provided. Got single entry, but expected to be list!"
                )
                assert len(teacher_load_path) == tp_size, (
                    f"Got {len(teacher_load_path)} checkpoints, but expected to be the same as TP size {tp_size}."
                )
                teacher_load_path = teacher_load_path[dist.get_tp_group_rank()]
        else:
            raise NotImplementedError("EP size > 1 is not supported for TP size > 1")

    save_overwrite = cfg.get("save_overwrite", False)
    save_weights_only = cfg.get("save_weights_only", False)
    autoresume_default = False
    if (
        cfg.run_name is not None
        and save_folder is not None
        and save_latest_filename is not None
        and not save_overwrite
        and not save_weights_only
    ):
        logger.info(
            "As run_name, save_folder, and save_latest_filename are set, changing autoresume default to True..."
        )
        autoresume_default = True

    validate_batch_sizes(cfg)

    # Build the Trainer. The supplied private Composer fork has a few extra
    # keyword arguments. Public Composer receives the common subset.
    logger.info("Building trainer...")
    trainer_kwargs = dict(
        run_name=cfg.run_name,
        seed=cfg.seed,
        deterministic_mode=cfg.get("deterministic_mode", False),
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        global_train_batch_size=cfg.get("global_train_batch_size", None),
        max_duration=cfg.max_duration,
        eval_interval=resolve_interval(cfg, "eval_interval", 1, logger=logger)
        if not use_dummy_dataset
        else 0,
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        progress_bar=cfg.get("progress_bar", False),
        log_to_console=bool(cfg.get("log_to_console", True)) and (os.environ["RANK"] == "0"),
        console_log_interval=cfg.get("console_log_interval", "1ba"),
        console_stream=cfg.get("console_stream", "stderr"),
        log_traces=bool(cfg.get("log_traces", False)),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=resolve_interval(cfg, "save_interval", "1000ba", logger=logger),
        stable_save_interval=resolve_interval(
            cfg, "stable_save_interval", None, logger=logger
        ),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        compute_checkpoint_hashes=cfg.get("compute_checkpoint_hashes", True),
        load_path=load_path,
        teacher_load_path=teacher_load_path,
        load_weights_only=cfg.get("load_weights_only", False),
        load_ignore_keys=cfg.get("load_ignore_keys", None),
        load_strict_model_weights=cfg.get("load_strict_model_weights", True),
        autoresume=cfg.get("autoresume", autoresume_default),
        python_log_level=python_log_level,
        dist_timeout=cfg.dist_timeout,
        auto_log_hparams=cfg.get("auto_log_hparams", True),
        profiler=profiler,
        zero_nan_grads=cfg.get("zero_nan_grads", False),
        save_checkpoint_async=cfg.get("save_checkpoint_async", False),
        compile_config=cfg.get("compile_config", None),
    )
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    unsupported_trainer_kwargs = sorted(
        key for key in trainer_kwargs if key not in trainer_parameters
    )
    if unsupported_trainer_kwargs:
        logger.info(
            "Public Composer ignores private-fork Trainer arguments: %s",
            ", ".join(unsupported_trainer_kwargs),
        )
    trainer = Trainer(
        **{
            key: value
            for key, value in trainer_kwargs.items()
            if key in trainer_parameters
        }
    )
    # we need to validate sp_size before training
    # but we can't do it here because we need to have fsdp_config with updated batch size info
    validate_sp_size(cfg)

    if cfg.get("sft_mode", False):
        # Adding HF saver to trainer
        if cfg.get("save_hf", False):
            for callback_idx, callback_obj in enumerate(trainer.state.callbacks):
                if callback_obj.__class__.__name__ == "CheckpointSaver":
                    trainer.state.callbacks[
                        callback_idx
                    ]._save_checkpoint = save_hf_model_decorator(  # noqa: F821
                        trainer,
                        trainer.state.callbacks[callback_idx]._save_checkpoint,
                        cfg["model_path"],
                        cfg["tokenizer"]["name"],
                        model_config=cfg.model,
                        model_params_to_copy=cfg.get("model_params_to_copy", None),
                    )

        logger.info("Decorated CheckpointSaver")

    logger.info("Logging config...")
    if os.environ["RANK"] == "0":
        log_config(whole_config)

    eval_before_train = cfg["eval_before_train"]
    if eval_before_train == "force":
        trainer.eval()
    elif eval_before_train == "regular":
        if isinstance(cfg.eval_interval, int):
            eval_interval_batches = cfg.eval_interval
        elif isinstance(cfg.eval_interval, str) and cfg.eval_interval[-2:] == "ba":
            eval_interval_batches = int(cfg.eval_interval[:-2])
        else:
            eval_interval_batches = None

        if eval_interval_batches is None or eval_interval_batches == 0:
            logger.info("Skipping eval before train")
        elif trainer.state.timestamp.batch.value % eval_interval_batches == 0:
            trainer.eval()

    logger.info("Starting training...")
    trainer.fit()

    # finish signal
    # it could be only on master node
    if signal_files_dir:
        recreate_signal_file(signal_files_dir / "finish")
    logger.info("Done.")

    dist.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    yaml_resolved_path = Path(yaml_path).resolve()
    yaml_cfg = om.load(yaml_path)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(cfg, DictConfig)

    register_relative_config_loader(yaml_resolved_path.parent)
    om.register_new_resolver("sum", lambda x, y: int(x) + int(y))
    om.register_new_resolver("mult", lambda x, y: int(x) * int(y))
    cfg = prepare_train_config(cfg)

    os.environ["SAVE_LOGITS_FOLDER"] = cfg.get("save_logits_folder", "")

    if cfg.get("DEBUG_MOE_GATE"):
        os.environ["DEBUG_MOE_GATE"] = str(cfg.DEBUG_MOE_GATE)

    if "wandb" in cfg.get("loggers", []):
        _wandb_env_filepath = os.environ.get("ENV_FILEPATH_WANDB")
        if _wandb_env_filepath is None:
            _wandb_env_filepath = (
                Path(__file__).absolute().parent.parent.parent
                / "sber_configs/.wandb.env"
            )
        wandb_env_path = find_dotenv(filename=_wandb_env_filepath)
        assert wandb_env_path, "need to specify .wandb.env"
        config = dotenv_values(wandb_env_path)
        if "WANDB_BASE_URL" in os.environ:
            config["WANDB_BASE_URL"] = os.environ["WANDB_BASE_URL"]

        assert config["WANDB_BASE_URL"] != "..."
        assert config["WANDB_API_KEY"] != "..."
        load_dotenv(wandb_env_path)

    if cfg.use_clearml:
        if os.environ.get("RANK") == "0":
            load_dotenv(find_dotenv())
            if cfg.get("sft_mode", False):
                project_name = "GigaChat/SFT"
            else:
                project_name = cfg.get(
                    "clearml_project_name", "GigaChat/GigaChat_pretrain"
                )
            task = Task.init(
                project_name=project_name,
                task_name=f"{cfg.run_name}",
                tags=cfg.get("clearml_tags", []),
            )
            task.connect(om.to_container(cfg, resolve=True))
            config_preview = pformat(om.to_container(cfg, resolve=True))
            task.upload_artifact(
                name="llmf_config", artifact_object=yaml_path, preview=config_preview
            )
            task.connect(
                {"world_size": dist.get_world_size(), "region": "SR008"},
                name="Conveyor_info",
            )

    if "job_dispatcher_callback_web" in cfg.get("callbacks", []):
        job_dispatcher_config = om.to_container(
            cfg["callbacks"]["job_dispatcher_callback_web"], resolve=True
        )

        if "custom_pipeline" not in job_dispatcher_config:
            job_dispatcher_url = os.environ.get("JOB_DISPATCHER_BASE_URL")
            response = requests.post(
                url=f"{job_dispatcher_url}/jobs/check-config",
                json=job_dispatcher_config,
                verify=False,
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error in config job_dispatcher_callback_web: {response.json()}"
                )
    cfg.date = datetime.now().strftime("%d.%m.%y-%H'%M")
    main(cfg)
