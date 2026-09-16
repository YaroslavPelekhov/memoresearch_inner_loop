import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig
from typing import Optional, List, Tuple, Dict, Any


class BaseVisionTower(nn.Module):
    def __init__(
        self,
        pretrain_path: Optional[str] = None,
        ve_config: Optional[PretrainedConfig] = None,
        freeze: bool = False,
        select_layer: int = -2,
        interp_size: Optional[int] = None,
        to_square: bool = False,
        **kwargs: Any,
    ):
        """
        Initializes the vistion tower module.

        Args:
            pretrain_path (str): Name or path to the CLIP vision model to be loaded.
            select_layer (int): Index of layer to select features from.
        """
        super().__init__()

        self.is_loaded = False
        self.pretrain_path = pretrain_path
        self.select_layer = select_layer
        self._interp_size = interp_size
        self.to_square = to_square
        self.freeze = freeze
        self.ve_config = ve_config
        self.load_model()
        if self.freeze:
            self.freeze_model()

    def freeze_model(self) -> None:
        """
        Freezes the model parameters if the `freeze` flag is set.
        """
        self.freeze = True
        self.vision_tower.requires_grad_(False)

    def interpolate(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the image features to a square shape or to a target size.

        Args:
            image_features (torch.Tensor): Input image features with shape [b, num_tokens, dim].

        Returns:
            torch.Tensor: Interpolated image features.
        """

        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        # TODO experiment with linear layer
        if self.to_square:
            # b, dim, num_tokens
            image_features = image_features.permute(0, 2, 1).contiguous()
            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(h * w),
                mode="linear",
                align_corners=False,
            )
            # b, num_tokens, dim
            image_features = image_features.permute(0, 2, 1)

        if self._interp_size is None:
            return image_features

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def load_model(self, device_map: Optional[Dict[str, Any]] = None) -> None:
        """
        Loads the vision tower model.

        Args:
            device_map (Optional[Dict], optional): Device map for model sharding. Defaults to None.
        """
        raise NotImplementedError()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Processes images through the vision tower and extracts features.

        Args:
            images: A batch of images with shape [num_images, num_channels, h, w]

        Returns:
            torch.Tensor: Extracted image features with shape [num_images, num_image_tokens, num_channels].
        """
        raise NotImplementedError()

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, add_prefix: str = ""
    ) -> List[Tuple[nn.Module, str]]:
        raise NotImplementedError()

    def get_giga_fsdp_modules_for_activation_checkpointing(self) -> List[nn.Module]:
        raise NotImplementedError()

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size
