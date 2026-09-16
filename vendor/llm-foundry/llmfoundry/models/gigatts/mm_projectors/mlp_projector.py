import torch
import torch.nn as nn

from .base_projector import BaseProjector


class MLPProjector(BaseProjector):
    def __init__(
        self,
        num_layers: int = 2,
        mm_hidden_size: int = 1024,
        hidden_size: int = 4096,
        **kwargs,
    ):
        super().__init__(mm_hidden_size, hidden_size)

        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(num_layers - 1):
            modules += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        self.module = nn.Sequential(*modules)

    def forward(self, x) -> torch.Tensor:
        x = self.module(x)
        return x
