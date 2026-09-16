import torch

from typing import Optional, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel

try:
    from composer.utils.reproducibility import seed_context
except ImportError:
    print("Can not import modules: seed_context. Probably running in inference mode.")

from .base_projector import BaseProjector
from .mlp_projector import MLPProjector
from .ldpnetv2_projector import LDPNetV2Projector
from .pixel_shuffle_projector import PixelShuffleProjector


_PROJECTOR_REGISTRY: dict[str, type[BaseProjector]] = {
    "mlp": MLPProjector,
    "pixel_shuffle": PixelShuffleProjector,
    "mlp_downsample": PixelShuffleProjector,
    "ldpnetv2": LDPNetV2Projector,
}


class ProjectorConfig(PretrainedConfig):
    model_type: str = "projector"

    def __init__(
        self,
        type: Optional[str] = None,
        pretrain_path: Optional[str] = None,
        freeze: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.type = type
        self.pretrain_path = pretrain_path
        self.freeze = freeze
        self.params = params if params is not None else None


class MMProjector(PreTrainedModel):
    config_class = ProjectorConfig
    _no_split_modules = ["BaseProjector"]

    def __init__(self, config: ProjectorConfig):
        super().__init__(config)
        if self.config.params.get("tp_size", 1) > 1:
            self.projector = self._build_mm_projector_with_seed(
                self.config.type, self.config.params
            )
        else:
            self.projector = self._build_mm_projector(
                self.config.type, self.config.params
            )

        if self.config.freeze:
            self.projector.requires_grad_(False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Projector input tensor with shape [num_images, num_image_tokens, mm_hidden_size]

        Returns:
            torch.Tensor: Output tensor with shape [num_images, num_image_tokens, embedding_dim]

        Raises:
            ValueError: If an unknown projector type is specified in the configuration.
        """
        x = self.projector(x, **kwargs)
        return x

    def param_init_fn(self, module: torch.nn.Module):
        pass

    def _build_mm_projector(self, projector_type: str, params: dict) -> BaseProjector:
        try:
            cls = _PROJECTOR_REGISTRY[projector_type]
        except KeyError:
            valid = ", ".join(sorted(_PROJECTOR_REGISTRY))
            raise ValueError(
                f"Unknown projector type {projector_type}. Valid types are: {valid}"
            )
        return cls(**params)

    def _build_mm_projector_with_seed(
        self, projector_type: str, params: dict
    ) -> BaseProjector:
        with seed_context(self.config.params.get("seed", 42)):
            return self._build_mm_projector(projector_type, params)
