import typing as tp

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class LinearProjectorConfig(PretrainedConfig):
    model_type = "linear_projector"

    def __init__(
        self,
        encoder_out_size: int = 512,
        hidden_size: int = 4096,
        **kwargs: tp.Dict,
    ):
        super().__init__(**kwargs)

        self.encoder_out_size = encoder_out_size
        self.hidden_size = hidden_size


class LinearProjector(nn.Module):
    def __init__(self, config: PretrainedConfig, **kwargs: tp.Dict):
        super().__init__(**kwargs)

        input_size = config.encoder_out_size
        self.linear = nn.Linear(input_size, config.hidden_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        return self.linear(x), lengths


class MLPProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"

    def __init__(
        self,
        encoder_out_size: int = 512,
        hidden_size: int = 4096,
        **kwargs: tp.Dict,
    ):
        super().__init__(**kwargs)

        self.encoder_out_size = encoder_out_size
        self.hidden_size = hidden_size


class MLPProjector(nn.Module):
    def __init__(self, config: MLPProjectorConfig, **kwargs: tp.Dict):
        super().__init__(**kwargs)

        input_size = config.encoder_out_size
        self.model = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        return self.model(x), lengths


class AttentionProjectorConfig(PretrainedConfig):
    model_type = "attention_projector"

    def __init__(
        self,
        encoder_out_size: int = 512,
        n_heads: int = 4,
        ff_expansion_factor: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        dropout_att: float = 0.0,
        hidden_size: int = 4096,
        **kwargs: tp.Dict,
    ):
        super().__init__(**kwargs)

        self.encoder_out_size = encoder_out_size
        self.n_heads = n_heads
        self.ff_expansion_factor = ff_expansion_factor
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_att = dropout_att
        self.hidden_size = hidden_size
