import torch
import torch.nn as nn


class BaseProjector(nn.Module):
    """
    Base class for image projectors.
    """

    def __init__(self, mm_hidden_size: int, hidden_size: int):
        super().__init__()
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
