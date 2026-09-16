import typing as tp

import torch
import torch.nn as nn

from transformers import PretrainedConfig


class IdentitySubsamplerConfig(PretrainedConfig):
    model_type = "identity_subsampler"

    def __init__(self, **kwargs: tp.Dict) -> None:
        super().__init__(**kwargs)


class IdentitySubsampler(torch.nn.Identity):
    def __init__(self, config: IdentitySubsamplerConfig, **kwargs: tp.Dict) -> None:
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        return x, lengths


class StackingSubsamplerConfig(PretrainedConfig):
    model_type = "stacking_subsampler"

    def __init__(
        self,
        subsampling_factor: int = 4,
        **kwargs: tp.Dict,
    ) -> None:
        super().__init__(**kwargs)

        self.subsampling_factor = subsampling_factor


class StackingSubsampler(nn.Module):
    def __init__(self, config: StackingSubsamplerConfig, **kwargs: tp.Dict) -> None:
        super().__init__(**kwargs)

        if isinstance(config, dict):
            config = StackingSubsamplerConfig(**config)

        self.subsampling_factor = config.subsampling_factor

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py#L25
        b, t, h = x.size()

        mask = torch.arange(t, device=x.device).unsqueeze(0).unsqueeze(-1).expand(b, t, h)
        mask = mask >= lengths.view(b, 1, 1)
        x = x.clone()
        x[mask] = 0

        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths
