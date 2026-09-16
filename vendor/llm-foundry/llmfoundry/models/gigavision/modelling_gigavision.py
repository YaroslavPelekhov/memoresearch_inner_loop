import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from omegaconf import DictConfig
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_gigavision import BaseBackBoneLLMConfig, GigaVisionConfig
from .mm_vision_towers import MMVisionTower, VisionTowerConfig
from .mm_projectors import MMProjector, ProjectorConfig
from .registry import MODEL_CLS_REGISTRY

try:
    from composer.utils.dist import get_tp_group_size, get_sp_group_size
    from llmfoundry.models.parallel.tensor import (
        reduce_from_tensor_model_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )
    from llmfoundry.models.parallel.sequence.utils import get_tensors_sp_part
    from llmfoundry.models.parallel.sequence.mappings import (
        gather_from_sequence_parallel_region_on_batch_size_dimension,
    )
except ImportError:
    warnings.warn(
        "Encountered exception while importing functions for training. "
        "It's normal behaviour if you are loading hf model."
    )

    def get_sp_group_size():
        return 1


def get_llm_class(config: BaseBackBoneLLMConfig) -> PreTrainedModel:
    if config.type not in MODEL_CLS_REGISTRY:
        raise ValueError(f"Unknown llm type: {config.type}")

    model_types_without_init = ["deepseek", "qwen3"]

    if config.pretrain_path:
        llm = MODEL_CLS_REGISTRY[config.type].from_pretrained(
            config.pretrain_path, config=config.config, trust_remote_code=True
        )
    elif config.type in model_types_without_init:
        llm = MODEL_CLS_REGISTRY[config.type].from_config(
            config.config, trust_remote_code=True
        )
    else:
        llm = MODEL_CLS_REGISTRY[config.type](config.config)

    llm.requires_grad_(not config.freeze)

    return llm


# Base class
# Base config - только llm-ка
# __init__ + prepare4training (лоудинг весов, ) + _init_weights + forward


class BaseMultiModalForCausalLM(PreTrainedModel):
    """
    Base class for all multimodal models
    """

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def init_llm(self, config: BaseBackBoneLLMConfig) -> PreTrainedModel:
        return get_llm_class(config)

    def prepare4training(
        self, tokenizer: PreTrainedTokenizerBase, general_config: DictConfig
    ):
        # method that called before trainig, that load all weights, resize embedings and so on
        raise NotImplementedError

    @property
    def language_model(self):
        return self._language_model

    # Add init for yours special layers
    def _init_weights(self, module: torch.nn.Module):
        # `param_init_fn` has the conditioning for all modules classes
        if getattr(self.language_model, "param_init_fn", False):
            self.language_model.param_init_fn(module)
        else:
            self.language_model._init_weights(module)

    def resize_llm_embeddings(
        self, tokenizer: PreTrainedTokenizerBase = None, vocab_size=None
    ):
        if tokenizer is not None:
            vocab_size = max(max(tokenizer.get_added_vocab().values()), len(tokenizer))
        embedding_size = self.language_model.get_input_embeddings().weight.shape[0]
        num_new_tokens = 0
        if embedding_size * self.llm_tp_size < vocab_size:
            assert self.llm_tp_size == 1, (
                "Resize llm embeddings is not possible for tp_size > 1"
            )
            new_embedding_size = math.ceil(vocab_size / 64) * 64
            num_new_tokens = new_embedding_size - embedding_size
            self.language_model.resize_token_embeddings(new_embedding_size)

        if num_new_tokens > 0:
            input_embeddings = self.language_model.get_input_embeddings().weight.data
            output_embeddings = self.language_model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


class GigaVisionForCausalLM(BaseMultiModalForCausalLM):
    config_class = GigaVisionConfig
    base_model_prefix = "gigavision_model"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _tp_plan = {}

    def __init__(self, config: GigaVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._image_token = config.image_token
        self._video_token = config.video_token
        self.config.vocab_size = self.config.llm_config.config.vocab_size

        self.llm_tp_size = 1
        if hasattr(self.config.llm_config.config, "tp_size"):
            self.llm_tp_size = self.config.llm_config.config.tp_size
        self.update_projector_config(config.projector_config)
        self.update_vision_config(config.vision_config)
        self._mm_vision_tower = self.init_vision_tower(config.vision_config)
        self._mm_projector = self.init_projector(config.projector_config)
        self._language_model = self.init_llm(config.llm_config)

    def update_projector_config(self, projector_config: ProjectorConfig):
        if (
            projector_config.params is not None
            and "encoder_num_patches" not in projector_config.params
            and (
                projector_config.type == "ldpnetv2"
                or projector_config.type == "mlp_downsample_custom"
            )
        ):
            projector_config.params["encoder_num_patches"] = (
                self._mm_vision_tower.vision_tower.num_patches()
            )
        projector_config.params["tp_size"] = self.llm_tp_size

    def update_vision_config(self, vision_config: VisionTowerConfig):
        if hasattr(self.config.llm_config.config, "init_device"):
            vision_config.params["llm_init_device"] = (
                self.config.llm_config.config.init_device
            )

    def init_vision_tower(self, config: VisionTowerConfig):
        return MMVisionTower(config)

    def init_projector(self, config: ProjectorConfig):
        return MMProjector(config)

    def param_init_fn(self, module: torch.nn.Module):
        if getattr(module, "reset_parameters", False):
            module.reset_parameters()
        else:
            for m in module.children():
                if getattr(module, "reset_parameters", False):
                    m.reset_parameters()

    def _init_weights(self, module: torch.nn.Module):
        # `param_init_fn` has the conditioning for all modules classes
        if isinstance(module, nn.Conv2d):
            module.reset_parameters()
        elif isinstance(module, nn.BatchNorm2d):
            module.reset_parameters()
        else:
            super()._init_weights(module)

    def get_text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)

    def load_mm_projector(self):
        path = self.config.projector_config.pretrain_path
        if path is not None:
            d = torch.load(path)
            self._mm_projector.load_state_dict(d)

    def prepare4training(
        self, tokenizer: PreTrainedTokenizerBase, general_config: DictConfig
    ):
        # self.load_llm()
        self.load_mm_projector()
        self.resize_llm_embeddings(tokenizer)

    # TODO: проставить property
    @property
    def mm_projector(self):
        return self._mm_projector

    @property
    def vision_tower(self):
        return self._mm_vision_tower

    @property
    def image_token(self):
        return self._image_token

    @property
    def video_token(self):
        return self._video_token

    def pad_vision_tensor_to_sp_group_size(self, vision_tensor):
        sp_group_size = get_sp_group_size() or 1
        if vision_tensor.shape[0] % sp_group_size != 0:
            pad_size = sp_group_size - (vision_tensor.shape[0] % sp_group_size)
            padding = torch.zeros(
                (
                    pad_size,
                    vision_tensor.shape[1],
                    vision_tensor.shape[2],
                    vision_tensor.shape[3],
                ),
                device=vision_tensor.device,
                dtype=vision_tensor.dtype,
            )
            vision_tensor = torch.cat((vision_tensor, padding), dim=0)
        return vision_tensor

    def encode_vision_inputs(self, images, videos):
        n_images = images.shape[0]
        n_frames = videos.shape[0]
        vision_tensor = torch.cat([images, videos], dim=0)

        sp_group_size = get_sp_group_size() or 1
        if sp_group_size > 1 and self._language_model.training:
            vision_tensor = self.pad_vision_tensor_to_sp_group_size(vision_tensor)
            (vision_tensor,) = get_tensors_sp_part(
                vision_tensor, split_type="equal", dim=0
            )

        vision_features = self.vision_tower(vision_tensor)

        if torch.is_inference_mode_enabled():
            vision_features = vision_features.to(dtype=self.mm_projector.dtype)
        vision_features = self.mm_projector(vision_features)

        if sp_group_size > 1 and self._language_model.training:
            # We need to clone image_features to prevent strange gigafsdp error:
            # `RuntimeError: Output 0 of TyingGateBackward is a view and its base
            # or another view of its base has been modified inplace.`
            vision_features = (
                gather_from_sequence_parallel_region_on_batch_size_dimension(
                    vision_features.clone()
                )[: n_images + n_frames]
            )

        image_features = vision_features[:n_images]
        video_features = vision_features[n_images : n_images + n_frames]
        return image_features, video_features

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            # self.parameters() is empty when using gigafsdp
            return f"cuda:{torch.cuda.current_device()}"

    @staticmethod
    def _check_num_mm_tokens_in_ids(embeds, input_ids, token, labels):
        assert (labels == token).sum() == 0, "labels can not contain image tokens"
        assert (input_ids == token).sum() == (embeds.shape[0]), (
            f"Number of multimodal tokens `{token}` in input_ids is smaller than number of output tokens from vision encoder and vision adapter. "
            f"Ensure each {token} is encoded as several {token} tokens in input_ids. {(input_ids == token).sum()} vs {embeds.shape[0]}"
        )

    def _filter_empty_vision_tensors(self, vision_tensor):
        is_empty = vision_tensor.view(vision_tensor.size(0), -1).eq(0).all(dim=1)
        non_empty_indices = ~is_empty
        filtered_vision_tensors = vision_tensor[non_empty_indices]
        return filtered_vision_tensors

    def _prepare_inputs(self, input_ids, labels, images, videos):
        if labels is None:
            labels = torch.full_like(input_ids, -100)

        image_size = self.vision_tower.vision_tower.vision_tower.config.image_size

        if images is not None:
            images = torch.vstack(torch.unbind(images, dim=0))
            images = self._filter_empty_vision_tensors(images)

        if images is None or len(images) == 0:
            images = torch.zeros(1, 3, image_size, image_size).to(self.device)

        if videos is not None:
            videos = torch.vstack(torch.unbind(videos, dim=0))
            videos = self._filter_empty_vision_tensors(videos)

        if videos is None or len(videos) == 0:
            videos = torch.zeros(1, 3, image_size, image_size).to(self.device)

        return input_ids, labels, images, videos

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        videos: Optional[torch.Tensor],
    ):
        input_ids, labels, images, videos = self._prepare_inputs(
            input_ids, labels, images, videos
        )
        image_embeds, video_embeds = self.encode_vision_inputs(images, videos)

        if getattr(self.mm_projector.projector, "use_reduction", False):
            video_embeds, input_ids, attention_mask, labels = (
                self.mm_projector.projector.apply_compression_video_token(
                    input_ids, video_embeds, attention_mask, labels
                )
            )

        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        video_embeds = video_embeds.reshape(-1, image_embeds.shape[-1])
        if (input_ids == self.image_token).any():
            self._check_num_mm_tokens_in_ids(
                image_embeds, input_ids, self.image_token, labels
            )

        if (input_ids == self.video_token).any():
            self._check_num_mm_tokens_in_ids(
                video_embeds, input_ids, self.video_token, labels
            )

        input_embeds = self.get_text_embeds(input_ids)
        new_input_embeds = self._get_new_embeds(
            input_ids,
            input_embeds,
            image_embeds,
            video_embeds,
        )

        return (
            None,
            None,
            attention_mask,
            past_key_values,
            new_input_embeds,
            labels,
        )

    def _replace_vision_token_placeholders_with_embeds(
        self,
        flat_input_ids: torch.Tensor,
        flat_input_embeds: torch.Tensor,
        embeds: torch.Tensor,
        placeholder_token: int,
    ):
        flat_embeds = embeds.to(flat_input_embeds.device)
        enable_async_tp = (
            self.config.llm_config.config.enable_async_tp
            if hasattr(self.config.llm_config.config, "enable_async_tp")
            else False
        )

        if enable_async_tp:
            # hidden_size is splitted on tp_size by EmbeddingParallelEmbedding class
            # flat_input_embeds shape: [bs, seq_len, hidden_size // tp_size]
            assert flat_input_embeds.size(-1) * get_tp_group_size() == flat_embeds.size(
                -1
            ), (
                f"flat_input_embeds hidden_size={flat_input_embeds.size(-1) * get_tp_group_size()} "
                f"is not equal to flat_video_embeds hidden_size={flat_embeds.size(-1)}"
            )

        if self.llm_tp_size > 1:
            # if the embedders were guaranteed to be deterministic across ranks, we could simply chunk the outputs here
            # unfortunately, they are not, thus reduction is performed to stabilize vision embedder outputs across different TP ranks
            flat_embeds = reduce_from_tensor_model_parallel_region(flat_embeds)
            # output of all-reduce is a view, "/" can't be done inplace
            flat_embeds = flat_embeds / get_tp_group_size()

        # flat_embeds shape: [bs, seq_len, hidden_size]
        if enable_async_tp:
            flat_embeds = scatter_to_tensor_model_parallel_region(flat_embeds)
            # flat_embeds shape: [bs, seq_len, hidden_size // tp_size]

        assert flat_embeds.size(-1) == flat_input_embeds.size(-1), (
            f"Modality with placeholder_token=`{placeholder_token}` embeds hidden_size={flat_embeds.size(-1)} "
            f"!= input_embeds hidden_size={flat_input_embeds.size(-1)}"
        )

        # Replace embeddings of text image tokens with image embeddings from vision encoder and adapter
        flat_mask = flat_input_ids == placeholder_token

        if flat_mask.any():
            flat_mask_sum = flat_mask.sum()
            flat_embeds_count = flat_embeds.shape[0]
            assert flat_mask_sum == flat_embeds_count, (
                f"Mismatch in the number of places to replace ({flat_mask_sum}) "
                + f"and available tokens ({flat_embeds_count})."
            )

        new_input_embeds = torch.masked_scatter(
            flat_input_embeds, flat_mask[:, None], flat_embeds
        )

        return new_input_embeds

    def _get_new_embeds(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        video_embeds: torch.Tensor,
    ):
        flat_input_ids = input_ids.reshape(-1).to(input_embeds.device)
        flat_input_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
        flat_input_embeds = self._replace_vision_token_placeholders_with_embeds(
            flat_input_ids, flat_input_embeds, image_embeds, self.image_token
        )
        flat_input_embeds = self._replace_vision_token_placeholders_with_embeds(
            flat_input_ids, flat_input_embeds, video_embeds, self.video_token
        )

        new_input_embeds = flat_input_embeds.reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        return new_input_embeds

    def _try_set_eval_for_non_trainable_modules(self):
        if self.config.vision_config.freeze:
            self.vision_tower.eval()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.llm_tp_size > 1:
            self._try_set_eval_for_non_trainable_modules()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                videos,
            )

        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None or videos is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None, images, videos
                )
            )
        else:
            inputs_embeds = self.get_text_embeds(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        videos = kwargs.pop("videos", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images

        if videos is not None:
            inputs["videos"] = videos
        return inputs
