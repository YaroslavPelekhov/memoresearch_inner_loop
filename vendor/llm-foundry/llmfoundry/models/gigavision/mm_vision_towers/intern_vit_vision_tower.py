import torch

from contextlib import nullcontext
from typing import Optional, List, Tuple, Dict, Any
from transformers import PretrainedConfig
from transformers.utils import logging
from accelerate import load_checkpoint_in_model

from .base_vision_tower import BaseVisionTower
from .intern_vit.configuration_intern_vit import InternVisionConfig
from .intern_vit.modeling_intern_vit import InternVisionModel


logger = logging.get_logger(__name__)


class InternViTVisionTower(BaseVisionTower):
    def __init__(
        self,
        ve_config: PretrainedConfig,  # This is InternVisionConfig
        pretrain_path: Optional[str] = None,
        deterministic_attention: bool = False,
        llm_init_device: str = "cpu",  # Overridden by init_device LLM config
        select_layer: int = -1,
        drop_path_rate: float = 0.0,
        freeze: bool = False,
        **kwargs: Any,
    ):
        self.deterministic_attention = deterministic_attention
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
            self.ve_config.deterministic_attention = self.deterministic_attention
            self.ve_config.drop_path_rate = self.drop_path_rate
            self.vision_tower = InternVisionModel(self.ve_config)
        elif self.llm_init_device == "cpu":
            self.vision_tower = InternVisionModel.from_pretrained(
                self.pretrain_path,
                device_map="cpu",
                trust_remote_code=True,
                drop_path_rate=self.drop_path_rate,
                deterministic_attention=self.deterministic_attention,
            )
        elif self.llm_init_device == "meta":
            config = InternVisionConfig.from_pretrained(self.pretrain_path)
            config.deterministic_attention = self.deterministic_attention
            config.drop_path_rate = self.drop_path_rate
            self.vision_tower = InternVisionModel(config).to_empty(
                device="cpu", recurse=True
            )
            load_checkpoint_in_model(self.vision_tower, self.pretrain_path, strict=True)

    def feature_select(self, image_forward_outs: torch.Tensor) -> torch.Tensor:
        """
        Selects and extracts features from the model outputs.

        Args:
            image_forward_outs (torch.Tensor): Output from the vision tower forward pass.

        Returns:
            torch.Tensor: Selected image features.
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
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
        modules_to_wrap_with_names = [
            (self.vision_tower.embeddings, "vision_tower.embeddings"),
        ] + [
            (module, f"vision_tower.encoder.layers.{idx}")
            for idx, module in enumerate(self.vision_tower.encoder.layers)
        ]

        if prefix and not prefix.endswith("."):
            prefix += "."

        modules_to_wrap_with_names = [
            (item[0], prefix + item[1]) for item in modules_to_wrap_with_names
        ]
        return modules_to_wrap_with_names

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        return self.vision_tower.encoder.layers
