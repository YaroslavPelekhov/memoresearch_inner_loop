import typing as tp

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class DummyEncoderConfig(PretrainedConfig): pass


class DummyEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        return input, lengths
