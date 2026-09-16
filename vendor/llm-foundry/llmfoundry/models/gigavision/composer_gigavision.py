import inspect
from collections import UserDict
from typing import Mapping, Optional

from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
)
from composer.models import HuggingFaceModel
from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device, prepare_hf_model_for_fsdp
from llmfoundry.models.utils.configuration_utils import get_sp_split_type
from llmfoundry.utils.config_utils import write_lora_config
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput

from .configuration_gigavision import GigaVisionConfig
from .modelling_gigavision import GigaVisionForCausalLM
from .gigafsdp_wappers_gigavision import build_gigavision_fsdp_wrapper_dict


class ComposerGigaVisionCausalLM(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        config = GigaVisionConfig(
            om_model_config.mm_vision_tower,
            om_model_config.mm_projector,
            om_model_config.llm,
            om_model_config.image_token,
            om_model_config.get("video_token", -1),
            om_model_config.get("enable_async_tp", False),
        )

        # write lora params if exist
        config = write_lora_config(config, om_model_config)

        if hasattr(config.llm_config.config, "sp_split_type"):
            config.llm_config.config.sp_split_type = get_sp_split_type(
                split_type=config.llm_config.config.sp_split_type,
                attention_type=config.llm_config.config.attention_type,
            )
            config.sp_split_type = config.llm_config.config.sp_split_type

        model = GigaVisionForCausalLM(config)

        train_metrics = []
        eval_metrics = [
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        model.prepare4training(tokenizer, om_model_config)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
        )

        self.model_forward_args = inspect.getfullargspec(self.model.forward).args

        resolved_init_device = hf_get_init_device(
            om_model_config.get("init_device", "cpu")
        )
        if isinstance(config.llm_config, GigaMixConfig):
            self.router_aux_loss_coef = config.llm_config.router_aux_loss_coef
            self.num_experts = config.llm_config.num_routed_experts
            self.num_experts_per_tok = config.llm_config.num_experts_per_tok
        # assert resolved_init_device == "cpu", "meta init_device is not implemented"
        prepare_hf_model_for_fsdp(self.model.language_model, resolved_init_device)

    def loss(self, outputs: ModelOutput, batch: Mapping):
        if self.config.llm_config.config.return_dict:
            loss, _ = outputs["loss"], outputs["logits"]
            assert self.config.llm_config.config.z_loss_eps == 0, (
                "CausalLMOutputWithPast does not support z_loss"
            )
            assert not self.config.llm_config.config.use_mtp, (
                "CausalLMOutputWithPast does not support use_mtp"
            )
            return {"total": loss}

        has_hidden_z_loss = (
            hasattr(self.config.llm_config.config, "hidden_z_loss_eps")
            and self.config.llm_config.config.hidden_z_loss_eps > 0
        )
        has_hidden_z_loss_attn = (
            getattr(self.config.llm_config.config, "hidden_z_loss_attn_coef", 0.0) > 0
        )
        has_hidden_z_loss_moe = (
            getattr(self.config.llm_config.config, "hidden_z_loss_moe_coef", 0.0) > 0
        )
        has_hidden_z_loss_dense_mlp = (
            getattr(
                self.config.llm_config.config, "hidden_z_loss_dense_mlp_coef", 0.0
            )
            > 0
        )
        has_z_loss = self.config.llm_config.config.z_loss_eps > 0
        has_mtp_loss = self.config.llm_config.config.use_mtp

        losses_dict = {"total": outputs[0]}

        # Order appended in `GigaMixForCausalLM.forward`:
        # ..., hidden_z_loss, hidden_z_loss_attn, hidden_z_loss_moe,
        # hidden_z_loss_dense_mlp. Strip in reverse.
        if has_hidden_z_loss_dense_mlp:
            if outputs[-1] is not None:
                losses_dict["hidden_z_loss_dense_mlp"] = outputs[-1]
            outputs = outputs[:-1]

        if has_hidden_z_loss_moe:
            if outputs[-1] is not None:
                losses_dict["hidden_z_loss_moe"] = outputs[-1]
            outputs = outputs[:-1]

        if has_hidden_z_loss_attn:
            if outputs[-1] is not None:
                losses_dict["hidden_z_loss_attn"] = outputs[-1]
            outputs = outputs[:-1]

        if has_hidden_z_loss:
            if outputs[-1] is not None:
                losses_dict["hidden_z_loss"] = outputs[-1]
            outputs = outputs[:-1]

        if has_mtp_loss:
            if outputs[-1] is not None:
                losses_dict["loss_mtp"] = outputs[-1]
            outputs = outputs[:-1]

        if has_z_loss:
            if outputs[-1] is not None:
                losses_dict["z_loss"] = outputs[-1]
            outputs = outputs[:-1]

        return losses_dict

    def forward(self, batch: Mapping):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                (
                    "Unexpected batch type. Expected a dictionary with keys corresponding "
                    "to the inputs to the forward function of the Huggingface model"
                )
            )
        return output

    def gigafsdp_model_setup(self, fsdp_config: dict):
        fsdp_config["gigafsdp_wrappers"]["model"] = build_gigavision_fsdp_wrapper_dict(
            self.model, fsdp_config
        )
