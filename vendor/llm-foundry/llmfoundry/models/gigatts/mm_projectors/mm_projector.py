import torch

from typing import Optional, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel

try:
    from composer.utils.reproducibility import seed_context
except ImportError as err:
    print(f"Can not import modules: seed_context. Probably running in inference mode.")

from .base_projector import BaseProjector
from .mlp_projector import MLPProjector
from .concat_projectors import ConcatProjectorAudioLLM, ConcatProjectorLLM

PROJECTOR_MODEL_REGISTRY_CLS = {
    'mlp': MLPProjector,
    'concat_projector_audio_llm': ConcatProjectorAudioLLM,
    'concat_projector_llm': ConcatProjectorLLM
}

class ProjectorConfig(PretrainedConfig):
    model_type: str = "projector"

    def __init__(
        self,
        type: Optional[str] = None,
        pretrain_path: Optional[str] = None,
        freeze: Optional[bool] = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
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
        self.projector = self._build_mm_projector(self.config)

    def forward(self, **kwargs) -> torch.Tensor:
        return self.projector(**kwargs)

    def _build_mm_projector(self, config: ProjectorConfig) -> BaseProjector:
        
        if config.type in PROJECTOR_MODEL_REGISTRY_CLS:
            return PROJECTOR_MODEL_REGISTRY_CLS[config.type](
                freeze=config.freeze, pretrain_path=config.pretrain_path, **config.params
            )
        else:
            raise ValueError(f"Projector with type `{config.type}` is not implemented!")
