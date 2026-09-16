import torch
import torch.nn as nn

from typing import Any

from .base_projector import BaseProjector


class MLPProjector(BaseProjector):
    def __init__(
        self,
        num_layers: int = 2,
        mm_hidden_size: int = 1024,
        hidden_size: int = 4096,
        **kwargs: Any,
    ):
        """
        MLP projector.

        Args:
            num_layers: Number of layers.
            mm_hidden_size: Hidden size of the input tensor.
            hidden_size: Hidden size of the output tensor.
        """
        super().__init__(mm_hidden_size, hidden_size)

        layers = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(num_layers - 1):
            layers += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InternViT output: (batch_size, seq_len, mm_hidden_size) == (n, 1024, 1024)
        # FastViTHD output: (batch_size, seq_len, mm_hidden_size) == (n, 256, 3072)
        return self.ffn(x)
