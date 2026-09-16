import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from omegaconf import DictConfig
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from composer.utils import dist
except ImportError as err:
    warnings.warn(
        "Encountered exception while importing functions for training. "
        "It's normal behaviour if you are loading hf model."
    )

from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_gigatts import BaseBackBoneLLMConfig, GigaTTSConfig
from .mm_audio_adapters import MMAudioAdapter, AudioAdapterConfig
from .mm_projectors import MMProjector, ProjectorConfig
from .registry import MODEL_CLS_REGISTRY

#####
##### TODO: refactor
#####
from llmfoundry.models.gigavision.modelling_gigavision import BaseMultiModalForCausalLM
#####

class GigaTTSForCausalLM(BaseMultiModalForCausalLM):
    config_class = GigaTTSConfig
    base_model_prefix = "gigatts_model"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config: GigaTTSConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # speech vocabulary
        self._mm_speech_embeddings_layers = nn.ModuleList([
            nn.Embedding(
                **config.speech_embeddings_config['codebook_params']
            ) for _ in range(config.speech_embeddings_config['n_codebooks'])
        ])

        # llm
        self._language_model = self.init_llm(config.llm_config)

        self.llm_tp_size = 1
        if hasattr(self.config.llm_config.config, "tp_size"):
            self.llm_tp_size = self.config.llm_config.config.tp_size

        # llm projector
        self.update_projector_config(config.llm_projector_config)
        self._mm_llm_projector = self.init_projector(config.llm_projector_config)

        # audio llm projector
        self.update_projector_config(config.audio_llm_projector_config)
        self._mm_audio_llm_projector = self.init_projector(config.audio_llm_projector_config)

        # audio adapter
        self._mm_audio_adapter = self.init_audio_adapter(config.audio_adapter_config)
        self.config.vocab_size = self.language_model.config.vocab_size

    @property
    def llm_projector(self):
        return self._mm_llm_projector

    @property
    def language_model(self):
        return self._language_model

    @property
    def audio_llm_projector(self):
        return self._mm_audio_llm_projector

    @property
    def audio_adapter(self):
        return self._mm_audio_adapter

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            # self.parameters() is empty when using gigafsdp
            return f"cuda:{torch.cuda.current_device()}"

    def prepare4training(self, tokenizer: PreTrainedTokenizerBase, general_config: DictConfig):
        self.resize_llm_embeddings(tokenizer)

    def update_projector_config(self, projector_config: ProjectorConfig):
        if projector_config.params is not None:
            projector_config.params["tp_size"] = self.tp_size

    def init_audio_adapter(self, config: AudioAdapterConfig):
        return MMAudioAdapter(config)

    def set_fsdp_wrap(self):
        for speech_embeds in self._mm_speech_embeddings_layers:
            speech_embeds._fsdp_wrap = True
        self._mm_llm_projector._fsdp_wrap = True
        self._mm_audio_llm_projector._fsdp_wrap = True
        self._mm_audio_adapter._fsdp_wrap = True

        for key, model_param in self.language_model.named_modules():
            if (
                isinstance(model_param, nn.Embedding)
            ):
                model_param._fsdp_wrap = True
            # print(key, getattr(model_param, "_fsdp_wrap", False))
        # TODO: проставить варнинги, сделать зеркалирование параметров
        self.fsdp_wrap_fn = self.language_model.fsdp_wrap_fn
        self.activation_checkpointing_fn = self.language_model.activation_checkpointing_fn

    def init_projector(self, config: ProjectorConfig):
        return MMProjector(config)

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

    def get_speech_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.ndim == 3
        n_codebooks = input_ids.shape[1]
        embeds = 0
        for i in range(n_codebooks):
            embeds += self._mm_speech_embeddings_layers[i](input_ids[:, i, :])
        embeds /= n_codebooks
        return embeds

    def _cross_entropy_loss(self, logits, labels):
        '''
            Input:
                - logits: torch.Tensor[BatchSize x HiddenSize x Nq x Len]
                - labels: torch.Tensor[BatchSize x Nq x Len]
            Output:
                - Loss Value
        '''
        loss_fct = FusedCrossEntropyLoss(
            inplace_backward=self.config.llm_config.config.loss_inplace_backward, # TODO: refactoring
            reduction='none',
            # Note: disable it because audio loss is computed replicated between tp ranks
            # process_group=dist.get_tp_group(),
        )

        num_tokens = (labels != -100).sum(dim=(1, 2))

        with torch.no_grad():
            shift_labels = labels.roll(shifts=(-1,), dims=(-1,)) # BS x N_q x SeqLen
            shift_labels = shift_labels.permute((0, 2, 1)).contiguous() # BS x SeqLen x N_q
            shift_labels[:, -1, :] = -100

        assert shift_labels.shape[:] == logits.shape[:-1], (
            f"Logits and labels shape mismatch, got labels {shift_labels.shape} and {logits.shape[:2]} logits."
        )
        shift_labels = shift_labels.to(logits.device)
        shift_labels = shift_labels.view(-1) # (BS x SeqLen x N_q)
        logits = logits.view(-1, logits.shape[-1]) # (BS x SeqLen x N_q) x Logits

        loss = loss_fct(logits, shift_labels)
        loss = loss.view(labels.shape[0], -1)
        loss = loss.sum(-1)
        loss = loss / (num_tokens + 1e-8)
        return loss.mean()

    def audio_loss(self, logits, labels: Optional[torch.Tensor] = None):
        '''
            Input:
                - logits: torch.Tensor[BatchSize x Len x Nq x HiddenSize]
                - labels: torch.Tensor[BatchSize x Len x Nq]
            Output:
                - None when labels is None
                - Loss Value
        '''
        if labels is not None:
            return self._cross_entropy_loss(
                logits=logits, # BS Len Nq Logits
                labels=labels
            )
        else:
            return None

    def prepare_input_embeds(
        self,
        *,
        input_ids,
        speech_input_ids,
    ):
        speech_embeds = self.get_speech_embeds(speech_input_ids)
        text_embeds = self.get_text_embeds(input_ids)
        assert text_embeds.shape == speech_embeds.shape, (text_embeds.shape, speech_embeds.shape)
        llm_embeds = self.llm_projector(
            text_embeds=text_embeds,
            speech_embeds=speech_embeds
        )
        return llm_embeds, speech_embeds

    def prepare_llm_attention_mask(self, attention_mask, speech_attention_mask):
        '''
            optimization: FlashAttn without mask
            - llm_attention_mask == None for train/val steps
            - llm_attention_mask != None for gen (we use left padding)
        '''
        assert not ((attention_mask is None) ^ (speech_attention_mask is None)), "attention masks are used simultaneously"
        if attention_mask is not None:
            assert not self.training, "empty mask for gen step"
            assert attention_mask.shape == speech_attention_mask.shape, "invalid masks shapes"
            return attention_mask | speech_attention_mask
        else:
            # for train and valid steps in composer model
            return None

    def _prepare_llm_output_fsdp(self, out):
        '''
            Input:  out[CausalLMOutputWithPast]
            Output: tuple[llm_loss, llm_logits, llm_hiddens, llm_kv_cache]
        '''
        assert isinstance(out, CausalLMOutputWithPast)
        llm_loss = out.get('loss', None)
        llm_logits = out.logits
        llm_hiddens = out.hidden_states
        llm_kv_cache = out.get('past_key_values', None)
        return llm_loss, llm_logits, llm_hiddens, llm_kv_cache

    def _prepare_llm_output_gigafsdp(self, out):
        '''
            Input:  out[tuple]
            Output: tuple[llm_loss, llm_logits, llm_hiddens, llm_kv_cache]
        '''
        assert isinstance(out, tuple)
        if len(out) == 3:
            # train mode
            llm_loss = None
            llm_logits, llm_hiddens = out[0], out[1]
        else:
            # eval mode
            assert len(out) == 4
            llm_loss, llm_logits, llm_hiddens = out[0], out[1], out[2]
        return llm_loss, llm_logits, llm_hiddens, None # gigafsdp does not support kv_cache

    def prepare_llm_output(self, out):
        '''
            Output: tuple[llm_loss, llm_logits, llm_hiddens, llm_kv_cache]
        '''
        if isinstance(out, CausalLMOutputWithPast): # TorchFSDP
            return self._prepare_llm_output_fsdp(out)
        elif isinstance(out, tuple): # GigaFSDP
            return self._prepare_llm_output_gigafsdp(out)
        else:
            raise NotImplementedError(f"Unknown LLM output type {type(out)}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        speech_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,          # None for train/valid step
        speech_attention_mask: Optional[torch.Tensor] = None,   # None for train/valid step
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        speech_past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        speech_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        '''
            Input:
                Text: input_ids, labels, attention_mask
                Speech: speech_input_ids, speech_labels, speech_attention_mask

            Output:
            Tuple[
                1. llm_loss (text)
                2. audio_llm_loss (audio)
                3. llm_logits (text) : BS x SeqLen x Hidden
                4. audio_llm_logits (audio) : BS x SeqLen x N_q x Logits
                5. past_key_values (text)
                6. speech_past_key_values
            ]
        '''

        if inputs_embeds is None:
            llm_input_embeds, speech_embeds = self.prepare_input_embeds(
                input_ids=input_ids,
                speech_input_ids=speech_input_ids,
            )

        llm_attention_mask = self.prepare_llm_attention_mask(
            attention_mask=attention_mask,
            speech_attention_mask=speech_attention_mask
        )

        llm_out = self.language_model.forward(
            input_ids=None,
            attention_mask=llm_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=llm_input_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        llm_loss, llm_logits, llm_hiddens, llm_kv_cache = self.prepare_llm_output(llm_out)

        # audio projector
        audio_llm_input_embeds = self.audio_llm_projector(
            llm_hiddens_embeds=llm_hiddens,
            speech_embeds=speech_embeds
        )

        # audio adapter
        audio_llm_logits, audio_llm_kv_cache = self.audio_adapter(
            embeds=audio_llm_input_embeds,
            attention_mask=llm_attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=speech_past_key_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )

        # calculate audio loss
        audio_llm_loss = self.audio_loss(
            logits=audio_llm_logits,
            labels=speech_labels
        )

        return (
            llm_loss,
            audio_llm_loss,
            llm_logits,
            audio_llm_logits,
            llm_kv_cache,
            audio_llm_kv_cache
        )
