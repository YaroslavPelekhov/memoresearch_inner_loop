import torch

from typing import Optional, Union, Dict, Any, List
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig
import os

from .base_audio_adapter import BaseAudioAdapter
from .llama_adapter import LlamaAdapter


class AudioAdapterConfig(PretrainedConfig):
    def __init__(
        self,
        type: Optional[str] = None,
        freeze: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.freeze = freeze
        self.params = params if params is not None else None


class MMAudioAdapter(PreTrainedModel):
    config_class = AudioAdapterConfig
    _no_split_modules = []

    def __init__(self, config: AudioAdapterConfig):
        super().__init__(config)
        self.audio_adapter = self._build_audio_adapter()

    def _build_audio_adapter(self) -> BaseAudioAdapter:
        if self.config.type in ("llama", "gigar"):
            return LlamaAdapter(
                type=self.config.type,
                freeze=self.config.freeze,
                params=self.config.params
            )
        else:
            raise ValueError(f"Adapter with type `{type}` is not implemented!")

    def forward(self, embeds: torch.Tensor, **kwargs):
        return self.audio_adapter(embeds, **kwargs)
