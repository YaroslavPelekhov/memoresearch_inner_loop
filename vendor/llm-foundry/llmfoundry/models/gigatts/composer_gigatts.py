import torch
import inspect
import functools
from collections import UserDict
from typing import Mapping, Optional, Tuple

from composer.metrics.nlp import LanguageCrossEntropy
from composer.metrics.tts import (
    TextCrossEntropy,
    SpeechCrossEntropy,
    TextNextTokenAccuracy,
    SpeechNextTokenAccuracy,
)

from composer.models import HuggingFaceModel
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput
from llmfoundry.utils.config_utils import write_lora_config

from .configuration_gigatts import GigaTTSConfig
from .modelling_gigatts import GigaTTSForCausalLM


class ComposerGigaTTSCausalLM(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.loss_weights = om_model_config.loss_weights

        config = GigaTTSConfig(
            speech_embeddings_config=om_model_config.speech_embeddings,
            audio_adapter_config=om_model_config.audio_adapter,
            llm_projector_config=om_model_config.llm_projector,
            audio_llm_projector_config=om_model_config.audio_llm_projector,
            llm_config=om_model_config.llm,
        )
        # write lora params if exist
        config = write_lora_config(config, om_model_config)

        model = GigaTTSForCausalLM(config)

        train_metrics = [] # for optimization

        eval_metrics = [
            TextCrossEntropy(),   # TODO: memory optimization
            SpeechCrossEntropy(), # TODO: memory optimization
            TextNextTokenAccuracy(),
            SpeechNextTokenAccuracy(),
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

    def loss(self, outputs: ModelOutput, batch: Mapping):
        loss_text, loss_speech = outputs[0], outputs[1]
        return {
            'total': self.loss_weights['text'] * loss_text + self.loss_weights['speech'] * loss_speech,
            'loss_text': self.loss_weights['text'] * loss_text.detach().cpu(),
            'loss_speech': self.loss_weights['speech'] * loss_speech.detach().cpu(),
        }

    def forward(self, batch: Mapping, inner_microbatch_size: Optional[int]=None):
        # TODO: add inner microbatching logic from HuggingFaceModel
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


    def eval_forward(self, batch: Mapping, outputs: Optional[Tuple[torch.Tensor]] = None, inner_microbatch_size: Optional[int]=None):
        # TODO: add inner microbatching logic from HuggingFaceModel
        text_labels = batch.pop('labels')
        speech_labels = batch.pop('speech_labels')

        # shift
        with torch.no_grad():
            text_labels = text_labels.roll(shifts=(-1,), dims=(-1,)) # BS x SeqLen
            text_labels = text_labels.contiguous() # BS x SeqLen
            text_labels[:, -1] = -100

        with torch.no_grad():
            speech_labels = speech_labels.roll(shifts=(-1,), dims=(-1,)) # BS x N_q x SeqLen
            speech_labels = speech_labels.permute((0, 2, 1)).contiguous() # BS x SeqLen x N_q
            speech_labels[:, -1, :] = -100

        assert text_labels.ndim == 2 and speech_labels.ndim == 3
        assert text_labels.shape[:] == speech_labels.shape[:-1]

        self.labels = {
            'text_labels': text_labels,
            'speech_labels': speech_labels,
        }

        if outputs is None:
            outputs = self(batch)
            assert outputs[4] is None and outputs[5] is None # No KV-cache
            assert len(outputs) == 6

        return {
            'text_logits': outputs[2].contiguous(),
            'speech_logits': outputs[3].contiguous()
        }


    def gigafsdp_model_setup(self, fsdp_config : dict):
        activation_checkpointing_auto_wrap_policy=lambda model: functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=lambda m: (
                m in (
                    next(layer.children()) for layer in model._language_model.model.layers[:(
                        len(model._language_model.model.layers)
                        if fsdp_config['num_layers_to_checkpoint'] is None
                        else fsdp_config['num_layers_to_checkpoint']
                    )]
                ) or m in (
                    layer for layer in model._mm_audio_adapter.audio_adapter.audio_llm.layers
                )
            )
        )

        giga_fsdp_modules_to_wrap_with_names=lambda model: (
            [
                (
                    m,
                    f"_mm_speech_embeddings_layers.{i}",
                )
                for i, m in enumerate(model._mm_speech_embeddings_layers)
            ]
            + [(model._language_model.model.embed_tokens, "_language_model.model.embed_tokens")]
            + [(model._mm_llm_projector, "_mm_llm_projector")]
            + [
                (
                    m,
                    f"_language_model.model.layers.{i}",
                )
                for i, m in enumerate(model._language_model.model.layers)
            ]
            + [(model._language_model.model.norm, "_language_model.model.norm")]
            + [(model._language_model.lm_head, "_language_model.lm_head")]
            + [(model._mm_audio_llm_projector, "_mm_audio_llm_projector")]
            + [(model._mm_audio_adapter, "_mm_audio_adapter")]
        )

        fsdp_config["num_layers_to_checkpoint"] = self.model._language_model.config.activation_checkpoint_layers_num
        fsdp_config["gigafsdp_wrappers"]["model"] = {
            "activation_checkpointing_auto_wrap_policy": activation_checkpointing_auto_wrap_policy(self.model),
            "giga_fsdp_modules_to_wrap_with_names": giga_fsdp_modules_to_wrap_with_names(self.model),
            "giga_fsdp_rogue_layer_norm_modules_with_names": None,
            "giga_fsdp_layer_norm_module_cls": None,
        }
