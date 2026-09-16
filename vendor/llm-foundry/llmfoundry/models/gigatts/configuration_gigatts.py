from pathlib import Path
from typing import Optional, Union, Tuple
import os
import json
import warnings
import shutil
import glob
try:
    from composer.utils.dist import get_tp_group_size
except ImportError:
    warnings.warn(f"Can not import modules for training. Probably running in inference mode.")

from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

from .mm_audio_adapters import AudioAdapterConfig
from .mm_projectors import ProjectorConfig
from .registry import CONFIG_CLS_REGISTRY
from llmfoundry.utils.config_utils import to_dict_container


#####
##### TODO: refactor
from llmfoundry.models.gigavision.configuration_gigavision import BaseBackBoneLLMConfig
#####

class GigaTTSConfig(PretrainedConfig):
    model_type = "giga_tts"
    audio_adapter_config_cls = AudioAdapterConfig
    projector_config_cls = ProjectorConfig
    llm_config_cls = BaseBackBoneLLMConfig

    def __init__(
        self,
        audio_adapter_config: Optional[DictConfig] = None,
        llm_projector_config: Optional[DictConfig] = None,
        audio_llm_projector_config: Optional[DictConfig] = None,
        llm_config: Optional[DictConfig] = None,
        speech_embeddings_config: Optional[DictConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_adapter_config = self.audio_adapter_config_cls(**to_dict_container(audio_adapter_config or {}))
        self.llm_projector_config = self.projector_config_cls(**to_dict_container(llm_projector_config or {}))
        self.audio_llm_projector_config = self.projector_config_cls(**to_dict_container(audio_llm_projector_config or {}))
        self.llm_config = self.llm_config_cls(**to_dict_container(llm_config or {}))
        self.speech_embeddings_config = to_dict_container(speech_embeddings_config or {})
        self.return_dict = self.llm_config.config.return_dict

        # TODO: fix sp_split_type in train.py
        self.sp_split_type = 'equal'

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> Union["GigaTTSConfig", Tuple["GigaTTSConfig", dict]]:
        """
        Override method from PretrainedConfig
        Method is called in AutoConfig.from_pretrained and AutoModelForCasualLM.from_pretrained
        """
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Override from PretrainedConfig to pass _name_or_path in final dict. Need for working `GigaTTSConfig.from_pretrained`
        In vanilla method _name_or_path is not set in config_dict to use it `from_dict`
        """
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["_name_or_path"] = str(pretrained_model_name_or_path)
        return config_dict, kwargs
