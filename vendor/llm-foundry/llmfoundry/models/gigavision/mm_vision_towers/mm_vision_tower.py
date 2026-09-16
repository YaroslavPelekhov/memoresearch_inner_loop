import os
import torch

from typing import Optional, Union, Any, List
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig
from omegaconf import DictConfig

from .base_vision_tower import BaseVisionTower
from .intern_vit_vision_tower import InternViTVisionTower
from .fastvithd_vision_tower import FastViTHDVisionTower
from .siglip_vision_tower import SiglipVisionTower


_VISION_TOWER_REGISTRY: dict[str, type[BaseVisionTower]] = {
    "internvit": InternViTVisionTower,
    "siglip": SiglipVisionTower,
    "fastvit": FastViTHDVisionTower,
}


class VisionTowerConfig(PretrainedConfig):
    vision_config_subfolder = "ve_config"

    def __init__(
        self,
        type: Optional[str] = None,
        pretrain_path: Optional[str] = None,
        freeze: Optional[bool] = None,
        params: Optional[DictConfig] = None,  # Contains drop_path_rate, etc.
        gradient_checkpointing: Optional[bool] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.type = type
        self.pretrain_path = pretrain_path
        self.freeze = freeze
        self.params = params  # Contains drop_path_rate, etc.
        self.gradient_checkpointing = gradient_checkpointing
        if self.pretrain_path is not None:
            self.ve_config = AutoConfig.from_pretrained(
                self.pretrain_path, trust_remote_code=True
            )
        else:
            self.ve_config: PretrainedConfig = None

    def set_ve_config(self):
        if self.pretrain_path is None and len(self.name_or_path) > 0:
            pp = os.path.join(self.name_or_path, self.vision_config_subfolder)
            self.ve_config = AutoConfig.from_pretrained(
                pp, trust_remote_code=True, local_files_only=True
            )


class MMVisionTower(PreTrainedModel):
    config_class = VisionTowerConfig
    _no_split_modules = [
        "InternVisionEncoderLayer", # For InternViT
        "InternVisionEmbeddings", # For InternViT
        "PatchEmbed", # For FastViTHD
        "MobileOneBlock", # For FastViTHD
        "RepMixerBlock", # For FastViTHD
        "AttentionBlock", # For FastViTHD
    ]

    def __init__(self, config: VisionTowerConfig):
        super().__init__(config)
        self.vision_tower = self._build_vision_tower(
            self.config.type, self.config.params
        )

    def forward(
        self, x: Union[torch.Tensor, List[torch.Tensor]], **kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(x, list):
            output = []
            for image in x:
                output.append(
                    self.vision_tower(
                        image.to(dtype=self.dtype).unsqueeze(0),
                        **kwargs,
                    )
                )
        else:
            output = self.vision_tower(x.to(dtype=self.dtype), **kwargs)
        return output

    def param_init_fn(self, module: torch.nn.Module):
        pass

    def _build_vision_tower(self, type: str, params: dict) -> BaseVisionTower:
        type = type.replace("_", "").lower()
        if type[-2:] == "hd":
            type = type[:-2]
        try:
            cls = _VISION_TOWER_REGISTRY[type]
        except KeyError:
            valid = ", ".join(sorted(_VISION_TOWER_REGISTRY))
            raise ValueError(
                f"Unknown vision tower type {type}. Valid types are: {valid}"
            )
        init_kwargs = dict(
            ve_config=self.config.ve_config,
            pretrain_path=self.config.pretrain_path,
            freeze=self.config.freeze,
            **params,
        )
        return cls(**init_kwargs)
