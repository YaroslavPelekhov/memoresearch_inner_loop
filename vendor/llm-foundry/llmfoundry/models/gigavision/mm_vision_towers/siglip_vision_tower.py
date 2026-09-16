# SIGLIP FROM https://huggingface.co/google/siglip-so400m-patch14-384
import torch

from transformers import SiglipVisionModel
from transformers import SiglipConfig
from contextlib import nullcontext

from .base_vision_tower import BaseVisionTower


class SiglipVisionTower(BaseVisionTower):
    def __init__(
        self,
        ve_config: SiglipConfig,
        pretrain_path: str = None,
        select_layer: int = -2,
        select_feature: str = "patch",
        **kwargs
    ):
        if ve_config is not None:
            ve_config = ve_config.vision_config
        super().__init__(pretrain_path, ve_config=ve_config, select_layer=select_layer, **kwargs)
        self.select_feature = select_feature

    def load_model(self, device_map = None) -> None:
        if self.pretrain_path is not None:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.pretrain_path)
        else:
            self.vision_tower = SiglipVisionModel(self.ve_config)

    def feature_select(self, image_forward_outs: torch.Tensor) -> torch.Tensor:
        """
        Selects and extracts features from the model outputs.

        Args:
            image_forward_outs (torch.Tensor): Output from the vision tower forward pass.

        Returns:
            torch.Tensor: Selected image features.
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        conditional_no_grad = torch.no_grad() if self.freeze else nullcontext()
        with conditional_no_grad:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = self.interpolate(image_features)
        return image_features

    def num_patches_per_side(self):
        return self.vision_tower.config.image_size // self.vision_tower.config.patch_size

    def num_patches(self):
        return (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2
