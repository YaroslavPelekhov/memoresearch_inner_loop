"""PyTorch Gigar model."""

import inspect
import functools
from importlib import import_module
import math
from collections import UserDict
from typing import List, Mapping, Optional, Tuple, Union

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torch.utils.checkpoint

from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss

import transformers
from transformers import GenerationMixin
from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
    LanguageCrossEntropy,
    LanguagePerplexity,
)
from composer.models import HuggingFaceModel
from composer.utils import dist, global_actlog_state
import composer.trainer.sac_utils as sac_utils

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch import nn

from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils.generic import ModelOutput

from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device, prepare_hf_model_for_fsdp
from llmfoundry.models.layers import (
    BLOCK_CLASS_REGISTRY,
    FC_CLASS_REGISTRY,
    RING_ATTN_CLASSES,
    PARALLEL_EMBEDDING_REGISTRY,
    MTP_CLASS_REGISTRY,
    VocabParallelEmbedding,
    EmbeddingParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    LlamaRMSNorm,
)
from llmfoundry.models.layers.norm import resolve_norm_class

from llmfoundry.models.layers.ffn import ParallelMLP
from llmfoundry.models.parallel.tensor import distribute_async_tp_embeddings
from llmfoundry.models.utils.configuration_utils import get_sp_split_type
from llmfoundry.models.utils.param_init_fns import (
    MODEL_INIT_REGISTRY,
    set_module_custom_std,
)
from llmfoundry.models.utils.flash_seqlens import (
    get_cu_seqlens_from_pos_ids,
    get_llama3_cu_seqlens_from_pos_ids,
)
from llmfoundry.models.parallel.sequence.utils import get_tensors_sp_part, pad_tensor
from llmfoundry.utils.config_utils import write_lora_config, to_container
from llmfoundry.models.ops.zloss import z_loss_fn

_CONFIG_FOR_DOC = "GigarConfig"


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GigarConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = GigarConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        # `param_init_fn` has the conditioning for all modules classes
        self.param_init_fn(module)

    # Params initialization function needed for meta devie parameters initialization
    def param_init_fn(self, module: nn.Module):
        init_type = self.config.init_type
        if init_type == "giga":
            gain = 1.0
            init_config = {
                "name": "xavier_normal_",
                "init_gain": gain,
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

            if hasattr(module, "custom_init_std"):
                fan_in, fan_out = module.custom_init_std
                init_config = set_module_custom_std(init_config, fan_in, fan_out, gain)

        elif init_type == "deepseek":
            std = 0.006
            init_config = {
                "name": "baseline_",
                "init_std": std,
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

        elif init_type == "dclm":
            init_config = {
                "name": "dclm_init_",
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

        elif init_type == "olmo3":
            init_config = {
                "name": "olmo3_init_",
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

        else:
            raise ValueError(
                f'Expected `init_type` to be one of "giga", "deepseek", "dclm" or "olmo3", got {init_type}.'
            )

        init_fn_name = init_config["name"]
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.num_hidden_layers,
            **init_config,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: GigarConfig
    """

    def __init__(self, config: GigarConfig):
        super().__init__(config)

        if config.enable_async_tp:
            assert config.parallel_embedding_type == "EmbeddingParallelEmbedding", (
                "Sharded tensor parallelism only works with EmbeddingParallelEmbedding"
            )

        if config.tp_size == 1:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                device=config.init_device,
            )
        else:
            if config.parallel_embedding_type == "EmbeddingParallelEmbedding":
                # TODO: should probably move this logic to GigarConfig?
                parallel_output_style = (
                    "shard_hidden" if config.enable_async_tp else "replicate"
                )
                emb_kwargs = dict(parallel_output_style=parallel_output_style)
            else:
                emb_kwargs = dict()

            self.embed_tokens = PARALLEL_EMBEDDING_REGISTRY[
                config.parallel_embedding_type
            ](
                config.vocab_size,
                config.hidden_size,
                config=config,
                padding_idx=config.pad_token_id,
                **emb_kwargs,
            )

        if config.init_type in {"dclm", "olmo3"}:
            self.embed_tokens._init_std = 1 / math.sqrt(config.hidden_size)

        if config.decoder_layer_module:
            import_module(config.decoder_layer_module)
        try:
            decoder_layer_cls = BLOCK_CLASS_REGISTRY[config.decoder_layer_type]
        except KeyError as exc:
            available = ", ".join(sorted(BLOCK_CLASS_REGISTRY))
            raise ValueError(
                f"Unknown decoder_layer_type {config.decoder_layer_type!r}. "
                f"Registered decoder layers: {available}"
            ) from exc
        decoder_layer_kwargs = dict(config.decoder_layer_kwargs or {})
        self.layers = nn.ModuleList(
            [
                decoder_layer_cls(config, layer_idx, **decoder_layer_kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.use_mtp = config.use_mtp
        self.mtp_block = MTP_CLASS_REGISTRY["LlamaDecoderMTPBlock"](config)

        if config.activation_checkpoint_layers_num:
            for i in range(config.num_hidden_layers):
                if i < config.activation_checkpoint_layers_num:
                    self.layers[i]._activation_checkpointing = True
                else:
                    self.layers[i]._activation_checkpointing = False

        norm_class = resolve_norm_class(config.norm_type)
        self.norm = norm_class.from_config(
            config
        )


        self.gradient_checkpointing = False

        # Initialize weights using Transformers' `post_init` method that uses
        # `_init_weights` method defined in LlamaPreTrainedModel class
        # ---
        # Do only in case of a non meta init (cpu, cuda (possibly)) as the meta
        # device initialization is handled by the FSDP module and the `param_init_fn`.
        # ---
        # NOTE (Sbr): comment out as we use LlamaForCausalLM model class that applies
        # its `post_init` function recursivly so this would be the second time we do
        # init here (third if we cound layers default init). Uncomment if you're going to
        # use this class for some other custom purposes.
        #
        # if config.init_device != "meta":
        #     self.post_init()

        self.enable_async_tp = config.enable_async_tp

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):  # pylint: disable=unused-argument
        """Return ``attention_mask`` as is."""
        # [bsz, seq_len]
        return attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Force set `use_cache=False` as it might leak through config.json. However, `use_cache=True`
        # is supported for audio modality when using packed_attn + GQA, hence we provide a separate
        # parameter `use_cache_force` to explicitly enable caching in that case.
        use_cache = (
            False if not self.config.use_cache_force else self.config.use_cache_force
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder input_ids or decoder inputs_embeds"
            )
        assert (input_ids is None) != (inputs_embeds is None), (
            "`input_ids` and `inputs_embeds` are mutually exclusive arguments"
        )

        if self.enable_async_tp and logical_batch_size is None:
            logical_batch_size = batch_size

        # Adjst `seq_length` in SP case to construct proper `input_ids` and `attention_mask`
        # for the whole sequence.
        sp_group_size = dist.get_sp_group_size() or 1
        if sp_group_size > 1:
            # NOTE (Sbr): Do not adjust `seq_length` in eval mode. See LlamaForCausalLM
            # forward note for explanation.
            # ---
            # TODO (Sbr): Remove if when the issue is fixed.
            if self.training:
                seq_length = seq_length * sp_group_size

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            # NOTE: with async_tp enabled the seq_length witll be only / tp of the full seq_len
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Note: make unified all2all for all types of input (input_ids or inputs_embeds)
        if self.enable_async_tp:
            # inputs_embeds shape: [bs, seq_len, hidden_size // tp_size]
            inputs_embeds = distribute_async_tp_embeddings(inputs_embeds)
            # inputs_embeds shape: [1, bs * seq_len // tp_size, hidden_size]

        # NOTE (Sbr): Disable attention mask formation as it is used and formed inside
        # flash attention forward funcs.
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past),
        #         dtype=torch.bool,
        #         device=inputs_embeds.device,
        #     )

        attention_mask = self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                transformers.logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = global_actlog_state.apply_layer_counter(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            with sac_utils.cpu_offload_context:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                        output_attentions,
                        None,
                        cu_seqlens,
                        max_seqlen,
                        logical_batch_size=logical_batch_size,
                    )

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        logical_batch_size=logical_batch_size,
                        is_left_padded_eval=is_left_padded_eval,
                    )

            hidden_states = layer_outputs[0]
            hidden_states = sac_utils.cpu_offload_sync_function(hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.use_mtp:
            mtp_hidden_states = self.mtp_block(
                input_embeds=inputs_embeds,
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
            )
        else:
            mtp_hidden_states = None

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            elts = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if self.use_mtp and self.training:
                elts.append(mtp_hidden_states)

            return tuple(v for v in elts if v is not None)

        if self.use_mtp:
            raise NotImplementedError("Returning 'dict' is not supported with mtp")

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GigarConfig):
        super().__init__(config)

        assert not (
            config.pretraining_tp > 1
            and (config.tp_size > 1 or config.enable_async_tp or config.use_liger)
        ), "pretraining_tp cannot be used with tp, asynctp and use_liger options."

        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.z_loss_eps = config.z_loss_eps

        self.lm_head: ColumnParallelLinear = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            config.vocab_size,
            config=config,
            bias=False,
            lm_head=True,
            gather_output=False,
        )

        if config.init_type in {"dclm", "olmo3"}:
            self.lm_head._init_std = 1 / math.sqrt(config.hidden_size)

        # Initialize layers and apply other related processing
        # ---
        # NOTE (Sbr): only done here, the LlamaModel call is commented to avoid
        # unnecessary re-initialization calls.
        if config.init_device != "meta":
            self.post_init()

        self.use_mtp = config.use_mtp
        self.ignore_index = config.ignore_index

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _build_shifted_labels(self, shift_labels: torch.Tensor) -> List[torch.Tensor]:
        labels = [shift_labels]
        if self.use_mtp and self.training:
            cur = shift_labels.clone()
            for _ in range(self.config.mtp_predictor_num):
                with torch.no_grad():
                    cur = cur.roll(shifts=(-1,), dims=(1,)).contiguous()
                    cur[:, -1] = self.ignore_index
                labels.append(cur.clone())
        return labels

    def _compute_mtp_loss(self, losses, num_tokens_global):
        if not self.use_mtp:
            return torch.zeros((), device=losses.device)

        mtp_losses = losses[1:]  ## 0 is for main block
        steps = torch.arange(
            1, self.config.mtp_predictor_num + 1, device=num_tokens_global.device
        ).view(-1, 1)
        num_tokens = (
            (num_tokens_global.view(1, -1) - steps).clamp_min_(1).to(torch.float32)
        )
        per_step = (mtp_losses.sum(dim=-1) / num_tokens).sum(dim=0)
        sp_size = dist.get_sp_group_size() or 1
        return sp_size * per_step.mean() / self.config.mtp_predictor_num

    def _compute_loss(self, inputs, shift_labels, loss_fn, num_tokens, **kwargs):
        return_z_loss = kwargs.pop("return_z_loss", False)
        if kwargs:
            kwargs["return_z_loss"] = return_z_loss

        labels = self._build_shifted_labels(
            shift_labels
        )  ## labels for main and mtp blocks
        labels_stack = torch.stack(labels, dim=0)  ## [num_mtp + 1, bs, seq_len]
        flat_inputs = inputs.view(
            -1, inputs.size(dim=-1)
        )  ## [(num_mtp + 1) * bs * seq_len, hidden_dim]
        flat_labels = labels_stack.view(-1).to(
            flat_inputs.device
        )  ## [(num_mtp + 1) * bs * seq_len]
        loss_flat = loss_fn(flat_inputs, flat_labels, **kwargs)

        z_loss = None
        if (
            self.z_loss_eps != 0.0 and self.config.use_liger
        ):  ## same as if return_z_loss
            loss_flat, z_loss = loss_flat
        # # NOTE(m1kol): Might be necessary, need to remove in the future if FA
        # # and Liger versions are better (see `lse_square_scale` in `loss_fct` above).
        if self.z_loss_eps != 0.0 and not self.config.use_liger:
            z_loss = z_loss_fn(flat_inputs, self.z_loss_eps)
            z_loss.masked_fill_(flat_labels == self.ignore_index, 0.0)
            loss_flat += z_loss

        bs, seq_len = shift_labels.size()
        losses = loss_flat.view(len(labels), bs, seq_len)  # len(labels) = num_mtp + 1

        # calculate cross entropy loss for main block
        # ---
        # Norm CE loss by SP size (`num_tokens = num_tokens / sp_size`) as `num_tokens`
        # are provided from forward and calculated before splitting `labels` to SP chunks.
        sp_size = dist.get_sp_group_size() or 1
        loss = losses[0]
        loss_ce = loss.view(bs, -1)
        loss_ce = loss_ce.sum(-1)
        loss_ce = sp_size * loss_ce / num_tokens
        if getattr(self.config, "_data_profiler_enabled", False):
            self.config._data_profiler_per_sample_losses = loss_ce.detach()
        loss_ce = loss_ce.mean()

        # calculate avg cross entropy loss for mtp blocks
        loss_mtp = self._compute_mtp_loss(losses, num_tokens)

        # calculate avg z-loss for main and mtp blocks
        if z_loss is not None:
            z_loss = z_loss.view(len(labels), bs, seq_len)
            steps = torch.arange(
                0, self.config.mtp_predictor_num + 1, device=num_tokens.device
            ).view(-1, 1)
            zloss_num_tokens = (
                (num_tokens.view(1, -1) - steps).clamp_min_(1).to(torch.float32)
            )
            z_loss = (z_loss.sum(dim=-1) / zloss_num_tokens).sum(dim=0)
            z_loss = sp_size * z_loss.mean() / len(labels)

        return loss_ce, loss_mtp, z_loss

    def calculate_liger_loss(
        self, hidden_states, outputs, shift_labels, num_tokens, return_z_loss
    ):
        grad_output = None
        if self.use_mtp:
            if isinstance(outputs, tuple):
                mtp_hidden_states = outputs[-1]
                # note (sbr):
                #  it is important to remove the last element from the tuple
                #  because the number of elements in the final outputs
                outputs = outputs[:-1]
            else:
                raise NotImplementedError("Returning 'dict' is not supported with mtp")

            def compute_static_grad_output_for_mtp() -> torch.Tensor:
                """
                Statically сomputes gradient for loss_flat tensor of this operations:
                ```python
                losses = loss_flat.view(len(labels_list), bs, seq_len) # (num_mtp + 1, bs, seq_len)
                loss = losses[0]
                loss_mtp = (losses[1:].sum(-1) / tokens_stack).sum(0) # (bs,)
                loss_mtp = loss_mtp.mean() / self.config.mtp_predictor_num

                loss_ce = loss.view(labels.shape[0], -1)
                loss_ce = loss_ce.sum(-1)
                loss_ce = sp_group_size * loss_ce / num_tokens
                loss_ce = loss_ce.mean()
                loss_mtp = sp_group_size * loss_mtp
                loss = loss_ce + self.config.mtp_loss_weight * loss_mtp
                ```

                Liger does not support computational graphs where gradient losses contain unequal values in the backward pass.
                For this reason we compute grad_output and pass it trough Liger forward.
                """
                bs, seq_len = shift_labels.size()
                sp_size = dist.get_sp_group_size() or 1

                # fp32 computing for numerical stability inside Liger kernel
                with torch.amp.autocast("cuda", enabled=False):
                    steps = torch.arange(
                        1, self.config.mtp_predictor_num + 1, device=num_tokens.device
                    ).view(-1, 1)
                    tokens_stack = (
                        (num_tokens.view(1, -1) - steps).clamp_min_(1).to(torch.float32)
                    )

                    num_mtp = self.config.mtp_predictor_num
                    mtp_loss_weight = self.config.mtp_loss_weight
                    total_elements = (num_mtp + 1) * bs * seq_len
                    loss_flat_grad = torch.zeros(
                        total_elements, device=num_tokens.device, dtype=torch.float32
                    )  # ((num_mtp+1)*bs*seq_len,)

                    # Main loss gradients
                    grad_main = sp_size / (num_tokens * bs)  # (bs,)
                    grad_main = grad_main.repeat_interleave(seq_len)  # (bs * seq_len,)
                    main_elements = bs * seq_len
                    loss_flat_grad[:main_elements] = grad_main

                    # MTP loss gradients
                    grad_mtp = (
                        sp_size * mtp_loss_weight / (num_mtp * bs * tokens_stack)
                    )  # (num_mtp, bs)
                    grad_mtp = grad_mtp.unsqueeze(-1)  # (num_mtp, bs, 1)
                    grad_mtp = grad_mtp.expand(
                        -1, -1, seq_len
                    )  # (num_mtp, bs, seq_len)
                    grad_mtp = grad_mtp.reshape(-1)  # (num_mtp * bs * seq_len,)

                    loss_flat_grad[main_elements:] = grad_mtp

                    if (
                        torch.isnan(loss_flat_grad).any()
                        or torch.isinf(loss_flat_grad).any()
                    ):
                        raise ValueError("Invalid values in grad_output computation")

                return loss_flat_grad

            grad_output = compute_static_grad_output_for_mtp()
            assert isinstance(mtp_hidden_states, tuple)
            combined_hidden = (hidden_states,) + mtp_hidden_states
        else:

            def compute_static_grad_output_for_sft() -> torch.Tensor:
                """
                Statically сomputes gradient for loss_flat tensor of this operations:
                ```python
                loss_ce = loss.view(labels.shape[0], -1) # (bs, seq_len)
                loss_ce = loss_ce.sum(-1)
                loss_ce = sp_group_size * loss_ce / num_tokens
                loss_ce = loss_ce.mean()
                ```

                Liger does not support computational graphs where gradient losses contain unequal values in the backward pass.
                For this reason we compute grad_output and pass it trough Liger forward.
                """
                bs, seq_len = shift_labels.size()
                sp_size = dist.get_sp_group_size() or 1

                # fp32 computing for numerical stability inside Liger kernel
                with torch.amp.autocast("cuda", enabled=False):
                    # CE loss gradients
                    grad_loss = sp_size / (num_tokens * bs)  # (bs,)
                    grad_loss = grad_loss.repeat_interleave(seq_len)  # (bs * seq_len,)

                    if torch.isnan(grad_loss).any() or torch.isinf(grad_loss).any():
                        raise ValueError("Invalid values in grad_output computation")

                return grad_loss

            # If training is performed on the SFT dataset
            if num_tokens.numel() > 1 and not torch.all(
                num_tokens == num_tokens.reshape(-1)[0]
            ):
                grad_output = compute_static_grad_output_for_sft()

            combined_hidden = (hidden_states,)

        loss, loss_mtp, z_loss = self._compute_loss(
            torch.cat(combined_hidden, dim=0)
            if len(combined_hidden) > 1
            else combined_hidden[0],
            shift_labels,
            self.lm_head,
            num_tokens=num_tokens,
            use_liger=self.config.use_liger,
            process_group=dist.get_tp_group(),
            lse_square_scale=self.z_loss_eps,
            return_z_loss=return_z_loss,
            precomputed_grad_output=grad_output,
        )
        return loss, loss_mtp, z_loss, outputs

    def calculate_standard_loss(self, logits, shift_labels, num_tokens, return_z_loss):
        loss_fn = FusedCrossEntropyLoss(
            inplace_backward=self.config.loss_inplace_backward,
            reduction="none",
            process_group=dist.get_tp_group(),
            # NOTE(m1kol): Disable Z-loss calculation via FA for now as
            # DCLM tests show it's broken somehow. Z-loss is calculated
            # later using Python (torch).
            # lse_square_scale=self.z_loss_eps,
            # return_z_loss=return_z_loss,
            ignore_index=self.ignore_index,
        )

        loss, loss_mtp, z_loss = self._compute_loss(
            logits.contiguous(),
            shift_labels,
            loss_fn,
            num_tokens=num_tokens,
            return_z_loss=return_z_loss,
        )
        return loss, loss_mtp, z_loss

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            logical_batch_size (`int`, *optional*):
                The batch size, shared across all TP ranks in a single group. During AsyncTP the original
                tensor may be sharded across the batch or sequence dimension. This arguments is used
                to restore the original (logical) tensor's shape after unsharding. For the terminology
                refer tor PyTorch's [documentation](https://pytorch.org/docs/stable/distributed.tensor.html)

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if input_ids is not None:
            bs, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            assert len(inputs_embeds.shape) == 3, (
                "inputs_embeds must be a 3D tensor: [bs, seq_len, hidden_size]"
            )
            bs, seq_len, _ = inputs_embeds.shape
        else:
            raise RuntimeError(
                f"Either `input_ids` or `inputs_embeds` must be provided, but both are None."
            )

        # NOTE: [bs, seq_len] here always must be full (without division by tp_size / sp_size, reshaping to [1, bs * seq_len] and etc)

        if self.model.config.enable_async_tp and logical_batch_size is None:
            logical_batch_size = bs

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            num_tokens = (labels != self.ignore_index).sum(-1)

        # Split sequences for sequence parallel case
        tp_sp_group_size = dist.get_tp_sp_group_size() or 1
        tp_group_size = dist.get_tp_group_size() or 1
        sp_group_size = dist.get_sp_group_size() or 1
        is_sequence_parallel_active: bool = sp_group_size > 1
        is_ring_attn_type: bool = self.config.attention_type in RING_ATTN_CLASSES

        # NOTE (Sbr): Disable sequence parallel processing during eval mode as there's
        # an error on generative eval (gsm8k).
        # ---
        # TODO (Sbr): Make SP available during eval mode as well. Now there's a problem
        # that the gsm8k gets stuck in the end and fails due to NCCL timeout error.
        if not self.training:
            is_sequence_parallel_active = False

        # multimodal models might omit this argument
        if input_ids is not None and is_sequence_parallel_active:
            assert input_ids.shape[-1] % sp_group_size == 0, (
                f"input_ids length should be divisible by sp_group_size {sp_group_size}"
            )
            inputs_device = input_ids.device

        if inputs_embeds is not None and is_sequence_parallel_active:
            assert inputs_embeds.shape[1] % sp_group_size == 0, (
                f"inputs_embeds length should be divisible by sp_group_size {sp_group_size}"
            )
            inputs_device = inputs_embeds.device

        cu_seqlens = max_seqlen = None
        # In the llama3-ring-attention case we need to compute cu_seqlens before SP,
        # because this ring-attention only works with global cu_seqlens
        if (
            self.config.sp_split_type == "llama3"
            and position_ids is not None
            and (getattr(self.config, "varlen_input", False) or not self.training)
        ):
            cu_seqlens, max_seqlen = get_llama3_cu_seqlens_from_pos_ids(position_ids)

        if is_sequence_parallel_active:
            if position_ids is None and is_ring_attn_type:
                position_ids = torch.arange(
                    0,
                    seq_len,
                    dtype=torch.long,
                    device=inputs_device,
                ).view(1, -1)

            # NOTE (Sbr): Would probably need to add `past_key_values` split if they will be used. They are not
            # used as we use `use_cache=Fasle`. Would probably also need to change code in other places too.

            # We caculate attention for the whole sequence so we don't split `attention_mask` and `position_ids`.
            tesnors_to_split = (labels, input_ids, inputs_embeds)
            sp_output = get_tensors_sp_part(
                tesnors_to_split,
                split_type=self.config.sp_split_type if self.training else "equal",
                dim=1,
            )
            labels, input_ids, inputs_embeds = sp_output

            if is_ring_attn_type:
                (position_ids,) = get_tensors_sp_part(
                    position_ids, split_type=self.config.sp_split_type, dim=-1
                )

        # In the case of ring-attention types, except for llama3 we need to compute cu_seqlens after SP,
        # because these ring-attention types only work with locally distributed cu_seqlens
        if (
            self.config.sp_split_type != "llama3"
            and position_ids is not None
            and (getattr(self.config, "varlen_input", False) or not self.training)
        ):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logical_batch_size=logical_batch_size,
            is_left_padded_eval=is_left_padded_eval,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = outputs[0]

        per_gpu_vocab_size = self.config.vocab_size
        if self.training and dist.get_tp_group_size() is not None:
            per_gpu_vocab_size //= dist.get_tp_group_size()

        def check_shapes_match(labels_shape: tuple, inputs: torch.Tensor):
            if self.config.use_liger and self.config.enable_async_tp:
                batch_size, seq_len = labels_shape
                labels_shape = (1, batch_size * seq_len // dist.get_tp_group_size())

            assert labels_shape == inputs.shape[:2], (
                f"Inputs and labels (batch_size, seq_len) shapes mismatch, got labels {labels_shape} and {inputs.shape[:2]} inputs."
            )

        # In evaluation mode we need logits, so we use only lm_head without cross-entropy.
        logits = None
        if not (self.config.use_liger and self.training):
            if self.use_mtp and self.training:
                if isinstance(outputs, tuple):
                    mtp_hidden_states = outputs[-1]
                    outputs = outputs[:-1]  # remove MTP hidden states from outputs
                else:
                    raise NotImplementedError(
                        "Returning 'dict' is not supported with mtp"
                    )
                assert isinstance(mtp_hidden_states, tuple)
                combined_hidden = (hidden_states,) + mtp_hidden_states
            else:
                combined_hidden = (hidden_states,)

            if self.config.tp_size > 1 and self.config.enable_async_tp:
                logit_parts = [
                    self.lm_head(
                        hidden_state,
                        gather_output=not self.training,
                        logical_batch_size=logical_batch_size,
                    )
                    for hidden_state in combined_hidden
                ]
                if len(logit_parts) > 1:
                    logits = torch.cat(logit_parts, dim=0).flatten(0, 1)
                else:
                    logits = logit_parts[0]
            else:
                if len(combined_hidden) > 1:
                    cat_hidden = torch.cat(combined_hidden, dim=0)
                else:
                    cat_hidden = combined_hidden[0]

                logits_cat = self.lm_head(
                    cat_hidden,
                    gather_output=not self.training,
                    logical_batch_size=logical_batch_size * len(combined_hidden)
                    if logical_batch_size
                    else logical_batch_size,
                )

                if len(combined_hidden) > 1:
                    logits = logits_cat.flatten(0, 1)
                else:
                    logits = logits_cat

        loss = None
        z_loss = None
        loss_ce = None
        loss_mtp = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # ---
            # NOTE (Sbr): Add labels length slice to `shift_logits` to account for sequence parallel case
            # when the input is padded (strip last padded tokens).

            # cyclic shift labels and replace last element with mask (-100)
            with torch.no_grad():
                shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
                shift_labels[:, -1] = self.ignore_index

            shift_labels_shape = shift_labels.shape

            return_z_loss = (self.z_loss_eps != 0.0) and self.config.use_liger

            # The liger is only used during training because it fuses the lm_head and cross-entropy operations.
            # For this reason, it's not possible to get logits from lm_head, so it's incompatible with evaluation mode.
            if self.config.use_liger and self.training:
                check_shapes_match(shift_labels_shape, hidden_states)
                loss_ce, loss_mtp, z_loss, outputs = self.calculate_liger_loss(
                    hidden_states,
                    outputs,
                    shift_labels,
                    num_tokens,
                    return_z_loss,
                )
            else:
                loss_ce, loss_mtp, z_loss = self.calculate_standard_loss(
                    logits,
                    shift_labels,
                    num_tokens,
                    return_z_loss,
                )

            loss = loss_ce + self.config.mtp_loss_weight * loss_mtp

        # Gather tensors for the whole sequence in evaluation mode
        # ---
        # TODO (Sbr): Add padding and slicing to original seq_len here not only for
        # `input_ids` but for other outputs too.
        if not self.training and is_sequence_parallel_active:
            sp_group = dist.get_sp_group()
            logits = torch.concatenate(dist.all_gather(logits, group=sp_group), dim=1)[
                :, :seq_len
            ]
            if outputs.hidden_states is not None:
                outputs.hidden_states = [
                    torch.concatenate(dist.all_gather(hidden, group=sp_group), dim=1)[
                        :, :seq_len
                    ]
                    for hidden in outputs.hidden_states
                ]

        if self.config.delete_logits and self.training:
            del logits
            # NOTE: we can't set logits to None, because ModelOutput ignores elements with None
            logits = torch.empty(1)

        if not return_dict:
            main_output = tuple() if loss is None else (loss,)
            # kv cache, hidden states and etc
            decoder_outputs = outputs[1:]
            addition_outputs = (loss_ce,)

            main_output += (logits,)
            if z_loss is not None:
                addition_outputs += (z_loss,)

            if self.use_mtp and self.training:
                addition_outputs += (loss_mtp,)

            return main_output + decoder_outputs + addition_outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class ComposerGigarCausalLM(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        trust_remote_code = om_model_config.get("trust_remote_code", True)
        use_auth_token = om_model_config.get("use_auth_token", False)

        # Get and use resolved device in case of "mixed" value init_device value
        resolved_init_device = hf_get_init_device(
            om_model_config.get("init_device", "cpu")
        )
        # TP config
        tp_size = om_model_config.get("tp_size", 1)
        assert tp_size >= 1, f"tp_size is expected to be >=1, got {tp_size}"
        if tp_size > 1:
            tp_group_size = dist.get_tp_group_size()
            assert tp_size == tp_group_size, (
                f"Wrong tensor parallel group size. {tp_group_size} instead {tp_size}"
            )

        config = GigarConfig.from_pretrained(
            om_model_config.get("pretrained_model_name_or_path"),
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            init_device=resolved_init_device,
            tp_size=tp_size,
            enable_async_tp=om_model_config.get("enable_async_tp", False),
        )

        # write lora params if exist
        config = write_lora_config(config, om_model_config)

        config.varlen_input = om_model_config.get("varlen_input", False)
        config.delete_logits = True

        # Extract config_overrides and handle sp_split_type and tied_modules_groups separately
        config_overrides = om_model_config.get("config_overrides", {})
        sp_split_override = config_overrides.pop("sp_split_type", None)
        tied_modules_groups = config_overrides.pop("tied_modules_groups", [[]])

        # set config overrides
        for k, v in config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            if isinstance(v, (ListConfig, DictConfig)):
                v = om.to_container(v, resolve=True)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f"Config dict override got unknown keys. "
                        + f"Extra keys: {extra_keys}. "
                        + f"Expected (a subset of) keys: {list(attr.keys())}."
                    )
                getattr(config, k).update(v)
            # necessary case to allow for rope_scaling to be overriden in llama config
            elif attr is None and isinstance(v, Mapping):
                setattr(config, k, {})
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        setattr(config, "sp_split_type", sp_split_override)
        config.sp_split_type = get_sp_split_type(
            split_type=config.sp_split_type,
            attention_type=config.attention_type,
            varlen_input=config.varlen_input,
        )

        setattr(config, "tied_modules_groups", tied_modules_groups)
        config.tied_modules_groups = [to_container(modules_group) for modules_group in tied_modules_groups]

        if config.tp_size > 1:
            assert config.pretraining_tp == 1, (
                "pretraining_tp is not supported with Parallel Loss"
            )

        # TODO (Sbr): check if such initialization is correct or can be done other way
        # (or it's more appropriate to do so other way)
        # Some context initialization is crucial for processes syncronization
        with dist.run_local_rank_zero_first():
            model = LlamaForCausalLM(config)

            if config.freeze_non_embed:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.model.embed_tokens.parameters():
                    param.requires_grad = True

                for param in model.lm_head.parameters():
                    param.requires_grad = True

            if config.non_freeze_layers_idxs is not None:
                for param in model.parameters():
                    param.requires_grad = False

                for i, layer in enumerate(model.model.layers):
                    if i in config.non_freeze_layers_idxs:
                        for param in layer.parameters():
                            param.requires_grad = True

        prepare_hf_model_for_fsdp(model, resolved_init_device)

        # It utilies a lot of memory and has no real point
        # train_metrics = [LanguageCrossEntropy(), LanguagePerplexity()]
        train_metrics = []
        eval_metrics = [
            # Required by the fixed held-out DCLM screen evaluator. ICL
            # evaluators select their own metric names explicitly.
            LanguageCrossEntropy(),
            # LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
        )
        self.model_forward_args = inspect.getfullargspec(self.model.forward).args
        self.z_loss_eps = config.z_loss_eps
        self.use_mtp = config.use_mtp

    def loss(self, outputs: ModelOutput, batch: Mapping):
        z_loss = None
        loss_ce = None
        loss_mtp = None
        if self.config.use_return_dict:
            loss = outputs["loss"]
            logits = outputs["logits"]
            assert self.z_loss_eps == 0, (
                "CausalLMOutputWithPast does not support z_loss"
            )
            assert not self.use_mtp, "CausalLMOutputWithPast does not support use_mtp"
        else:
            loss = outputs[0]
            logits = outputs[1]

            values = outputs[2:]
            if self.use_mtp and self.training:
                loss_mtp = values[-1]
                values = values[:-1]

            if self.z_loss_eps > 0:
                z_loss = values[-1]
                values = values[:-1]

            loss_ce = values[-1]
            values = values[:-1]

        return_d = {"total": loss}

        if z_loss is not None:
            return_d["z_loss"] = z_loss

        if loss_mtp is not None:
            return_d["loss_mtp"] = loss_mtp

        if loss_ce is not None:
            return_d["loss_ce"] = loss_ce

        return return_d

    def forward(self, batch: Mapping):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)
        else:
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model"
            )

        return output

    @staticmethod
    def _activation_checkpointing_auto_wrap_policy(
        model: LlamaForCausalLM, fsdp_config: dict
    ):
        def _checkpoint_policy_fn(m: nn.Module) -> bool:
            num_layers_to_checkpoint = fsdp_config["num_layers_to_checkpoint"]
            if num_layers_to_checkpoint is None:
                num_layers_to_checkpoint = len(model.model.layers)

            layers = model.model.layers[:num_layers_to_checkpoint]
            checkpointed_modules = {layer for layer in layers}

            if hasattr(model.model, "mtp_block"):
                checkpointed_modules |= {
                    mtp_layer.decoder_layer
                    for mtp_layer in model.model.mtp_block.mtp_layers
                }

            return m in (next(layer.children()) for layer in checkpointed_modules)

        return functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=_checkpoint_policy_fn,
        )

    @staticmethod
    def _giga_fsdp_modules_to_wrap_with_names(
        model: LlamaForCausalLM, fsdp_config: dict
    ):
        modules_with_names = [(model.model.embed_tokens, "model.embed_tokens")]

        for i, m in enumerate(model.model.layers):
            modules_with_names.append((m, f"model.layers.{i}"))

        if hasattr(model.model, "mtp_block"):
            for i, m in enumerate(model.model.mtp_block.mtp_layers):
                modules_with_names.append(
                    (m.decoder_layer, f"model.mtp_block.mtp_layers.{i}.decoder_layer")
                )

        modules_with_names.append((model.lm_head, "lm_head"))

        if fsdp_config["separate_layer_norm_disabled"]:
            modules_with_names.append((model.model.norm, "model.norm"))

        return modules_with_names

    @staticmethod
    def _giga_fsdp_rogue_layer_norm_modules_with_names(model: LlamaForCausalLM):
        """LayerNorm is rouge only in case it is not a part of STM (SuperTensorModule)
        STM wrapping is declared in giga_fsdp_modules_to_wrap_with_names
        """
        rouge_layer_norms = {model.model.norm: "model.norm"}
        if hasattr(model.model, "mtp_block"):
            rouge_layer_norms |= {
                m: f"model.mtp_block.mtp_norms.{i}"
                for i, m in enumerate(model.model.mtp_block.mtp_norms)
            }
        return rouge_layer_norms

    def gigafsdp_model_setup(self, fsdp_config: dict):
        fsdp_config["num_layers_to_checkpoint"] = (
            self.config.activation_checkpoint_layers_num
        )
        fsdp_config.setdefault("separate_layer_norm_disabled", False)
        if self.model.config.enable_async_tp:
            fsdp_config["separate_layer_norm_disabled"] = False

        fsdp_config["gigafsdp_wrappers"]["model"] = dict(
            activation_checkpointing_auto_wrap_policy=self._activation_checkpointing_auto_wrap_policy(
                self.model, fsdp_config
            ),
            giga_fsdp_modules_to_wrap_with_names=self._giga_fsdp_modules_to_wrap_with_names(
                self.model, fsdp_config
            ),
            giga_fsdp_rogue_layer_norm_modules_with_names=None
            if fsdp_config["separate_layer_norm_disabled"]
            else self._giga_fsdp_rogue_layer_norm_modules_with_names(self.model),
            # giga_fsdp_layer_norm_module_cls - allows gigafsdp to sync gradients of cls weights
            # works only in case if all_reduce_grads_across_model_parallel_group = true
            giga_fsdp_layer_norm_module_cls=None
            if fsdp_config["separate_layer_norm_disabled"]
            else resolve_norm_class(self.config.norm_type),
            special_process_group_fn=None,
        )

    def compile_model_wrap(self, compile_config):
        from giga_fsdp import GigaFSDP

        # ComposerModel -> LLaMaForCausalLM -> LLaMaModel -> its layers
        # TO DO: enable compile to wrap the entire layer
        for i in range(len(self.model.model.layers)):
            layer_module = self.model.model.layers[i]
            if isinstance(self.model, GigaFSDP):
                layer_module.root_module.self_attn = torch.compile(
                    layer_module.root_module.self_attn, **compile_config
                )
                layer_module.root_module.mlp.swiglu_fn = torch.compile(
                    layer_module.root_module.mlp.swiglu_fn, **compile_config
                )
            else:
                layer_module.self_attn = torch.compile(
                    layer_module.self_attn, **compile_config
                )
                layer_module.mlp.swiglu_fn = torch.compile(
                    layer_module.mlp.swiglu_fn, **compile_config
                )
