import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast
from llmfoundry.models.gigatts.registry import MODEL_CLS_REGISTRY, CONFIG_CLS_REGISTRY
from .base_audio_adapter import BaseAudioAdapter


class LlamaAdapter(BaseAudioAdapter):
    def __init__(self, type, params: dict, **kwargs):
        super().__init__(**kwargs)

        self.model_params = params['model_params']
        self.custom_params = params['custom_params']

        # output projection
        # TODO: there could be more complex head
        assert 'head_size' in self.custom_params and len(self.custom_params['head_size']) == 2
        head_total_size = self.custom_params['head_size'][0] * self.custom_params['head_size'][1]

        # Note: can't use vocab_size=0 for tp model
        if self.model_params.get("tp_size", 1) > 1:
            self.model_params['vocab_size'] = head_total_size

        model_cls = MODEL_CLS_REGISTRY[type]
        config_cls = CONFIG_CLS_REGISTRY[type]
        self.audio_llm = model_cls(config_cls(**self.model_params)).model

        delattr(self.audio_llm, "embed_tokens")

        # input projection (replicated on tp ranks)
        if 'in_channels' in self.custom_params and self.model_params['hidden_size'] != self.custom_params['in_channels']:
            self.in_proj = nn.Linear(self.custom_params['in_channels'], self.model_params['hidden_size'], bias=False)
        else:
            self.in_proj = nn.Identity()

        # lm_head (replicated on tp ranks)
        self.lm_head = nn.Linear(self.model_params['hidden_size'], head_total_size, bias=False)

        self.stop_grad = self.custom_params.get('stop_grad', False)
        
    def forward(self, embeds: torch.Tensor, **kwargs):
        '''
            input: projected embeddings (acoustic + speech)
            output: [
                logits tensor with shape [BatchSize, SequenceLength, N_Q, HiddenSize]
                Optional[kv_cache]
            ]
        '''
        if self.stop_grad:
            embeds = embeds.detach()
        proj_embeds = self.in_proj(embeds)

        # tp splitted region starts here
        out = self.audio_llm(inputs_embeds=proj_embeds, **kwargs)
        # tp splitted region ends here

        if isinstance(out, BaseModelOutputWithPast):
            last_hiddens = out.last_hidden_state
            kv_cache = out.get('past_key_values', None)
        elif isinstance(out, tuple):
            last_hiddens = out[0]
            kv_cache = None
        else:
            raise NotImplementedError(f"Unknown LLM output type {type(llm_out)}")

        last_hiddens = self.lm_head(last_hiddens)
        last_hiddens = last_hiddens.view(last_hiddens.shape[0], last_hiddens.shape[1], *self.custom_params['head_size']) # BatchSize SeqLen N_q HiddenSize

        return last_hiddens, kv_cache
