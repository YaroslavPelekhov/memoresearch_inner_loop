import math
import torch
import typing as tp
import torch.nn as nn

from .base_projector import BaseProjector
from .mlp_projector import MLPProjector


class LDPNetV2Projector(BaseProjector):
    def __init__(
        self,
        num_layers: int = 2,
        mm_hidden_size: int = 1024,
        hidden_size: int = 4096,
        num_image_tokens: int = 144,
        encoder_num_patches: int = 576,
        realization: str = "default", # averge
        **kwargs,
    ):
        super().__init__(mm_hidden_size, hidden_size)
        self.mlp = MLPProjector(num_layers, mm_hidden_size, hidden_size)

        downsample_size = int(math.sqrt(num_image_tokens))
        assert downsample_size * downsample_size == num_image_tokens, "`num_image_tokens` parameter must be square number!"
        if realization == "default":
            self.dwn = DownsampleLayer((downsample_size, downsample_size), encoder_num_patches)
        elif realization == "no_adaptive_avg":
            self.dwn = DownsampleLayerOnnx((downsample_size, downsample_size), encoder_num_patches)
        else:
            raise ValueError(f"Realization type = {realization} is not supported!")
        self.peg = PosInjectLayer(hidden_size, hidden_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, downsample_shape: tp.Tuple[int, int], encoder_num_patches: int):
        super().__init__()

        self.downsample_shape = downsample_shape
        self.pool_layer = nn.AdaptiveAvgPool2d(downsample_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_images, num_image_tokens, mm_hidden_size = x.shape
        h = int(math.sqrt(num_image_tokens))
        assert h * h == num_image_tokens

        x = x.permute(0, 2, 1).reshape(num_images, mm_hidden_size, h, h)
        x = self.pool_layer(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DownsampleLayerOnnx(nn.Module):
    def __init__(self, downsample_shape: tp.Tuple[int, int], encoder_num_patches: int):
        super().__init__()

        self.num_patches_per_side = int(math.sqrt(encoder_num_patches))
        assert self.num_patches_per_side * self.num_patches_per_side == encoder_num_patches

        # Formula: https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        stride = self.num_patches_per_side // downsample_shape[0]
        kernel_size = self.num_patches_per_side - (downsample_shape[0] - 1) * stride
        self.pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_images, _, mm_hidden_size = x.shape

        x = x.permute(0, 2, 1).reshape(
            num_images, mm_hidden_size, self.num_patches_per_side, self.num_patches_per_side
        )
        x = self.pool_layer(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosInjectLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=True, groups=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_images, num_image_tokens, mm_hidden_size = x.shape
        h = int(math.sqrt(num_image_tokens))
        assert h * h == num_image_tokens

        conv_features = x.transpose(1, 2).view(num_images, mm_hidden_size, h, h)
        x = self.conv(conv_features) + conv_features
        x = x.flatten(2).transpose(1, 2)
        return x

