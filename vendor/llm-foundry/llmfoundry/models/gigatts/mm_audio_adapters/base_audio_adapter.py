import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class BaseAudioAdapter(nn.Module):
    def __init__(self, freeze=False,):
        super().__init__()
        self._freeze_model(freeze)

    def _freeze_model(self, freeze):
        self.requires_grad_(not freeze)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
