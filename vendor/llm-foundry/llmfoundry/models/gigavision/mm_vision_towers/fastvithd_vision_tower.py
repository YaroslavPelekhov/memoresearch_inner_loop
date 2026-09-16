import torch

from contextlib import nullcontext
from typing import Optional, List, Tuple, Dict, Any
from transformers import PretrainedConfig
from transformers.utils import logging
from accelerate import load_checkpoint_in_model

from .base_vision_tower import BaseVisionTower
from .fastvithd.configuration_fastvithd import FastViTHDVisionConfig
from .fastvithd.modeling_fastvithd import FastViTHDVisionModel


logger = logging.get_logger(__name__)


class FastViTHDVisionTower(BaseVisionTower):
    def __init__(
        self,
        ve_config: PretrainedConfig,  # This is FastViTHDVisionConfig
        pretrain_path: Optional[str] = None,
        llm_init_device: str = "cpu",  # Overridden by init_device LLM config
        select_layer: int = -1,
        drop_path_rate: float = 0.0,
        freeze: bool = False,
        **kwargs: Any,
    ):
        self.llm_init_device = llm_init_device
        self.drop_path_rate = drop_path_rate

        super().__init__(
            pretrain_path=pretrain_path,
            ve_config=ve_config,
            freeze=freeze,
            select_layer=select_layer,
            **kwargs,
        )

    def load_model(self, device_map: Optional[Dict[str, Any]] = None) -> None:
        if self.pretrain_path is None:
            self.ve_config.drop_path_rate = self.drop_path_rate
            self.vision_tower = FastViTHDVisionModel(self.ve_config)
        elif self.llm_init_device == "cpu":
            self.vision_tower = FastViTHDVisionModel.from_pretrained(
                self.pretrain_path,
                device_map="cpu",
                trust_remote_code=True,
                drop_path_rate=self.drop_path_rate,
            )
        elif self.llm_init_device == "meta":
            config = FastViTHDVisionConfig.from_pretrained(self.pretrain_path)
            config.drop_path_rate = self.drop_path_rate
            self.vision_tower = FastViTHDVisionModel(config).to_empty(
                device="cpu", recurse=True
            )
            load_checkpoint_in_model(
                self.vision_tower, self.pretrain_path, strict=False
            )
        self.vision_tower = self.vision_tower.to(torch.bfloat16)

    def feature_select(self, image_forward_outs: torch.Tensor) -> torch.Tensor:
        """
        Selects and extracts features from the model outputs.

        Args:
            image_forward_outs (torch.Tensor): Output from the vision tower forward pass.

        Returns:
            torch.Tensor: Selected image features.
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        B, C, H, W = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return image_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        conditional_no_grad = torch.no_grad() if self.freeze else nullcontext()

        with conditional_no_grad:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def num_patches_per_side(self):
        return (
            self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        )

    def num_patches(self):
        patches_per_side = self.num_patches_per_side()
        return patches_per_side**2

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, prefix: str = ""
    ) -> List[Tuple[torch.nn.Module, str]]:
        modules_to_wrap_with_names = []

        modules_to_wrap_with_names.append(
            (self.vision_tower.patch_embed, "vision_tower.patch_embed")
        )

        for stage_idx, stage_module in enumerate(self.vision_tower.network):
            if isinstance(stage_module, torch.nn.Sequential):
                for block_idx, block in enumerate(stage_module):
                    modules_to_wrap_with_names.append(
                        (block, f"vision_tower.network.{stage_idx}.{block_idx}")
                    )
            else:
                modules_to_wrap_with_names.append(
                    (stage_module, f"vision_tower.network.{stage_idx}")
                )

        modules_to_wrap_with_names.append(
            (self.vision_tower.conv_exp, "vision_tower.conv_exp")
        )

        if prefix and not prefix.endswith("."):
            prefix += "."

        return [(module, prefix + name) for module, name in modules_to_wrap_with_names]

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        modules = []

        for module in self.vision_tower.network:
            if isinstance(module, torch.nn.Sequential):
                modules.extend(list(module))
            else:
                modules.append(module)

        return modules
