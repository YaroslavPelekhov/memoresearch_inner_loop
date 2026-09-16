import typing as tp
from collections import UserDict
from typing import Mapping, List


import torch
from composer.models import HuggingFaceModel
from composer.utils import dist
from composer.utils.dist import (
    get_tp_group_size,
)
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.gigavision.configuration_gigavision import GigaVisionConfig

from llmfoundry.models.hf.hf_fsdp import hf_get_init_device, prepare_hf_model_for_fsdp

from torchmetrics import Metric

from llmfoundry.models.layers.norm import resolve_norm_class

from llmfoundry.models.gigavision.gigafsdp_wappers_gigavision import (
    build_gigavision_fsdp_wrapper_dict,
)

from llmfoundry.models.base_model_mapping import (
    get_composer_model_class,
    get_inner_model,
    get_model_config,
)
from llmfoundry.models.utils.configuration_utils import get_sp_split_type


def build_model_config_gigavision(
    model_config: DictConfig,
    trust_remote_code: bool = True,
    use_auth_token: bool = False,
):
    config = GigaVisionConfig(
        model_config.mm_vision_tower,
        model_config.mm_projector,
        model_config.llm,
        model_config.image_token,
        model_config.get("video_token", -1),
        model_config.get("enable_async_tp", False),
    )

    if hasattr(config.llm_config.config, "sp_split_type"):
        config.llm_config.config.sp_split_type = get_sp_split_type(
            split_type=config.llm_config.config.sp_split_type,
            attention_type=config.llm_config.config.attention_type,
        )
        config.sp_split_type = config.llm_config.config.sp_split_type

    return config


def build_model_config_text(
    model_config: DictConfig,
    trust_remote_code: bool,
    use_auth_token: bool,
):
    config_name = model_config.name

    init_device = hf_get_init_device(model_config.get("init_device", "cpu"))

    tp_size = model_config.get("tp_size", 1)
    enable_async_tp = model_config.get("enable_async_tp", False) or False

    assert tp_size >= 1, f"tp_size is expected to be >=1, got {tp_size}"
    if tp_size > 1:
        tp_group_size = get_tp_group_size()
        assert tp_size == tp_group_size, (
            f"Wrong tensor parallel group size. {tp_group_size} instead {tp_size}"
        )

        assert enable_async_tp, "Tensor parallelism is supported only with async TP"
    else:
        assert not enable_async_tp, "async TP is supported only with Tensor parallelism"

    config = get_model_config(config_name).from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
        tp_size=tp_size,
        init_device=init_device,
        enable_async_tp=enable_async_tp,
    )

    config.varlen_input = model_config.get("varlen_input", False)

    config.always_gather_output = model_config.get("always_gather_output", False)

    for k, v in model_config.get("config_overrides", {}).items():
        if not hasattr(config, k):
            raise ValueError(
                f'config does not have attribute "{k}" to override ({k}: {v}).'
            )

        attr = getattr(config, k)
        if isinstance(attr, Mapping):
            extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
            if extra_keys:
                raise ValueError(
                    "Config dict override got unknown keys. "
                    + f"Extra keys: {extra_keys}. "
                    + f"Expected (a subset of) keys: {list(attr.keys())}."
                )
            getattr(config, k).update(v)
        elif attr is None and isinstance(v, Mapping):
            setattr(config, k, {})
            getattr(config, k).update(v)
        else:
            setattr(config, k, v)

    return config


CONFIG_BUILDER_REGISTRY = {
    "gigavision_causal_lm": build_model_config_gigavision,
    "gigar_causal_lm": build_model_config_text,
    "giga_mix_causal_lm": build_model_config_text,
}


def build_model_config(
    model_config: DictConfig,
    trust_remote_code: bool,
    use_auth_token: bool,
):
    builder = CONFIG_BUILDER_REGISTRY.get(model_config.name)
    if builder is None:
        raise ValueError(
            f"Unknown model name: {model_config.name}. "
            f"Available: {list(CONFIG_BUILDER_REGISTRY.keys())}"
        )
    return builder(model_config, trust_remote_code, use_auth_token)


def build_inner_model(model_name: str, model_cfg: DictConfig):
    inner_model = get_inner_model(model_name)
    return inner_model(model_cfg)


class ComposerTeacherStudentModel(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        train_metrics: List[Metric],
        eval_metrics: List[Metric],
    ):
        assert "teacher_model" in om_model_config
        assert "student_model" in om_model_config

        trust_remote_code = om_model_config.get("trust_remote_code", True)
        use_auth_token = om_model_config.get("use_auth_token", False)

        teacher_resolved_init_device = hf_get_init_device(
            om_model_config.teacher_model.get("init_device", "cpu")
        )

        student_resolved_init_device = hf_get_init_device(
            om_model_config.student_model.get("init_device", "cpu")
        )

        self.vision = om_model_config.student_model.name == "gigavision_causal_lm"

        self.teacher_config = build_model_config(
            om_model_config.teacher_model,
            trust_remote_code,
            use_auth_token,
        )

        self.student_config = build_model_config(
            om_model_config.student_model,
            trust_remote_code,
            use_auth_token,
        )

        teacher = build_inner_model(
            om_model_config.teacher_model.name, self.teacher_config
        )
        for param in teacher.parameters():
            param.requires_grad = False

        student = build_inner_model(
            om_model_config.student_model.name, self.student_config
        )

        if self.vision:
            teacher.prepare4training(tokenizer, om_model_config.teacher_model)
            student.prepare4training(tokenizer, om_model_config.student_model)
        else:
            prepare_hf_model_for_fsdp(teacher, teacher_resolved_init_device)
            prepare_hf_model_for_fsdp(student, student_resolved_init_device)

        super().__init__(
            model=student,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

        self.teacher = teacher

        if self.vision:
            self.config.loss_inplace_backward = (
                self.student_config.llm_config.config.loss_inplace_backward
            )

    def loss(self, outputs: Mapping, batch: Mapping):
        raise NotImplementedError(
            "Loss is not implemented for the abstract two models class."
        )

    def forward(self, batch: Mapping):
        if not (isinstance(batch, dict) or isinstance(batch, UserDict)):
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model"
            )
        # Further input validation is left to the huggingface forward call
        batch = {k: v for k, v in batch.items() if k in self.model_forward_args}

        with torch.no_grad():
            if self.training:
                teacher_out = self.teacher(**batch)
                # Без barries на самом первом батче
                # модель student возвращает неверные логиты
                dist.barrier()
            else:
                teacher_out = None
        student_out = self.model(**batch)

        if self.student_config.use_return_dict:
            student_out_dict = student_out
        else:  # tuple case
            student_out_dict = dict()
            if len(student_out) != 1:
                student_out_dict["loss"], student_out_dict["logits"] = (
                    student_out[0],
                    student_out[1],
                )

                if hasattr(self.model, "z_loss_eps") and self.model.z_loss_eps != 0.0:
                    student_out_dict["z_loss"] = student_out[-1]
            else:
                student_out_dict["logits"] = student_out[0]

        student_out_dict["teacher_out"] = teacher_out

        return student_out_dict

    def gigafsdp_model_setup(self, fsdp_config: dict):
        if self.vision:
            self._gigafsdp_model_setup_vision(fsdp_config)
        else:
            self._gigafsdp_model_setup_text(fsdp_config)

    def _gigafsdp_model_setup_text(self, fsdp_config: dict):
        fsdp_config["num_layers_to_checkpoint"] = (
            self.config.activation_checkpoint_layers_num
        )
        fsdp_config.setdefault("separate_layer_norm_disabled", False)
        if self.model.config.enable_async_tp:
            fsdp_config["separate_layer_norm_disabled"] = False

        COMPOSER_MODEL_CLASS = get_composer_model_class(type(self.model))
        model_norm_type = getattr(self.model.config, "norm_type", "LlamaRMSNorm")

        fsdp_config["gigafsdp_wrappers"]["model"] = dict(
            activation_checkpointing_auto_wrap_policy=COMPOSER_MODEL_CLASS._activation_checkpointing_auto_wrap_policy(
                self.model, fsdp_config
            ),
            giga_fsdp_modules_to_wrap_with_names=COMPOSER_MODEL_CLASS._giga_fsdp_modules_to_wrap_with_names(
                self.model, fsdp_config
            ),
            giga_fsdp_rogue_layer_norm_modules_with_names=COMPOSER_MODEL_CLASS._giga_fsdp_rogue_layer_norm_modules_with_names(
                self.model
            ),
            # giga_fsdp_layer_norm_module_cls - allows gigafsdp to sync gradients of cls weights
            # works only in case if all_reduce_grads_across_model_parallel_group = true
            giga_fsdp_layer_norm_module_cls=None
            if fsdp_config["separate_layer_norm_disabled"]
            else resolve_norm_class(model_norm_type),
            special_process_group_fn=COMPOSER_MODEL_CLASS._special_process_group_fn
            if hasattr(COMPOSER_MODEL_CLASS, "_special_process_group_fn")
            else None,
        )

        COMPOSER_TEACHER_CLASS = get_composer_model_class(type(self.teacher))
        teacher_norm_type = getattr(self.teacher.config, "norm_type", "LlamaRMSNorm")
        fsdp_config["gigafsdp_wrappers"]["teacher"] = dict(
            activation_checkpointing_auto_wrap_policy=COMPOSER_TEACHER_CLASS._activation_checkpointing_auto_wrap_policy(
                self.teacher, fsdp_config
            ),
            giga_fsdp_modules_to_wrap_with_names=COMPOSER_TEACHER_CLASS._giga_fsdp_modules_to_wrap_with_names(
                self.teacher, fsdp_config
            ),
            giga_fsdp_rogue_layer_norm_modules_with_names=COMPOSER_TEACHER_CLASS._giga_fsdp_rogue_layer_norm_modules_with_names(
                self.teacher
            ),
            # giga_fsdp_layer_norm_module_cls - allows gigafsdp to sync gradients of cls weights
            # works only in case if all_reduce_grads_across_model_parallel_group = true
            giga_fsdp_layer_norm_module_cls=None
            if fsdp_config["separate_layer_norm_disabled"]
            else resolve_norm_class(teacher_norm_type),
            special_process_group_fn=COMPOSER_TEACHER_CLASS._special_process_group_fn
            if hasattr(COMPOSER_TEACHER_CLASS, "_special_process_group_fn")
            else None,
        )

    def _gigafsdp_model_setup_vision(self, fsdp_config: dict):
        for target_model, dict_key in [
            (self.model, "model"),
            (self.teacher, "teacher"),
        ]:
            fsdp_config["gigafsdp_wrappers"][dict_key] = (
                build_gigavision_fsdp_wrapper_dict(target_model, fsdp_config)
            )

    def state_dict(self, *args: tp.Any, **kwargs: tp.Any):
        state_dict = super().state_dict(*args, **kwargs)
        for key in list(state_dict):
            if key.startswith("teacher"):
                state_dict.pop(key)
        return state_dict
