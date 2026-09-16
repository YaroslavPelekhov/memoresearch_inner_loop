import torch
import torch.nn as nn


class BaseProjector(nn.Module):
    """
    Base class for projectors.
    """
    def __init__(self, freeze, *args, **kwargs):
        super().__init__()
        self._freeze_model(freeze)

    def _freeze_model(self, freeze):
        self.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

