import typing as tp

import torch
import torch.nn as nn
from transformers import  PretrainedConfig

from .builder import build_acoustic_subsampler, build_acoustic_projector


class ModalityAdapterConfig(PretrainedConfig):
    def __init__(
            self,
            subsampler_cfg: PretrainedConfig,
            projector_cfg: PretrainedConfig,
            *args: tp.List,
            **kwargs: tp.Dict,
        ) -> None:
        super().__init__(*args, **kwargs)
        self.projector_cfg = projector_cfg
        self.subsampler_cfg = subsampler_cfg


class ModalityAdapter(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.subsampler = build_acoustic_subsampler(config.subsampler_cfg)
        self.projector = build_acoustic_projector(config.projector_cfg)

    def forward(self, specs: torch.Tensor, spec_lenghts: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        """Subsampled specs and then project into decoder prefix space.
        """
        specs, spec_lenghts = self.subsampler(specs, spec_lenghts)
        specs, spec_lenghts = self.projector(specs, spec_lenghts)
        return specs, spec_lenghts


def build_modality_adapter(config: PretrainedConfig, **kwargs: tp.Dict) -> ModalityAdapter:
    return ModalityAdapter(config, **kwargs)
