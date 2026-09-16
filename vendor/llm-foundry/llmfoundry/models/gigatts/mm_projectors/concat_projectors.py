import torch
import torch.nn as nn

from .base_projector import BaseProjector


class ConcatProjectorLLM(BaseProjector):
    def __init__(
        self,
        text_embedding_size: int,
        speech_embedding_size: int,
        out_embedding_size: int,
        bias: bool = False,
        init_device: str = 'cpu',
        stop_grad_speech: bool = False,
        stop_grad_text: bool = False,
        use_speech: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_embedding_size = text_embedding_size
        self.speech_embedding_size = speech_embedding_size
        self.out_embedding_size = out_embedding_size
        self.bias = bias
        self.stop_grad_text, self.stop_grad_speech = stop_grad_text, stop_grad_speech
        self.use_speech = use_speech

        self.linear = nn.Linear(
            self.text_embedding_size + self.speech_embedding_size,
            self.out_embedding_size,
            bias=self.bias,
            device=init_device
        )

        if init_device == "cpu":
            self.reset_parameters()
        self._init_rules()

    def forward(self, *, text_embeds: torch.Tensor, speech_embeds: torch.Tensor) -> torch.Tensor:
        assert isinstance(text_embeds, torch.Tensor) and isinstance(speech_embeds, torch.Tensor)
        assert text_embeds.shape == speech_embeds.shape

        if self.stop_grad_text:
            text_embeds = text_embeds.detach()
        if self.stop_grad_speech:
            speech_embeds = speech_embeds.detach()

        C = 1 if self.use_speech else 0
        embeds = torch.cat([text_embeds, speech_embeds], dim=-1)
        projected_embeds = self.linear(embeds)
        return text_embeds + C * projected_embeds

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def _init_rules(self):
        def raise_init_error(*args, **kwargs):
            raise RuntimeError("This module must be inited by it's parent reset_parameters")
        self.linear.skip_init = True
        self.linear.reset_parameters = raise_init_error


class ConcatProjectorAudioLLM(BaseProjector):
    def __init__(
        self,
        llm_embedding_size: int,
        speech_embedding_size: int,
        out_embedding_size: int,
        hidden_layer_id: int,
        bias: bool = False,
        init_device: str = 'cpu',
        stop_grad: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_embedding_size = llm_embedding_size
        self.speech_embedding_size = speech_embedding_size
        self.out_embedding_size = out_embedding_size
        self.bias = bias
        self.hidden_layer_id = hidden_layer_id
        self.stop_grad = stop_grad

        self.linear = nn.Linear(
            self.llm_embedding_size + self.speech_embedding_size,
            self.out_embedding_size,
            bias=self.bias,
            device=init_device
        )

    def forward(self, *, llm_hiddens_embeds: list[torch.Tensor], speech_embeds: torch.Tensor) -> torch.Tensor:
        assert isinstance(llm_hiddens_embeds, tuple)
        assert len(llm_hiddens_embeds) > self.hidden_layer_id
        llm_embeds = llm_hiddens_embeds[self.hidden_layer_id]
        assert llm_embeds.shape == speech_embeds.shape
        if self.stop_grad:
            llm_embeds, speech_embeds = llm_embeds.detach(), speech_embeds.detach()
        embeds = torch.cat([llm_embeds, speech_embeds], dim=-1)
        projected_embeds = self.linear(embeds)
        return projected_embeds
