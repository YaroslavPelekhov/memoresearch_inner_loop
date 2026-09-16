from pathlib import Path
from typing import Optional, Union, Tuple
import os
import json
import logging
import shutil
import glob


from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

from .mm_vision_towers import VisionTowerConfig
from .mm_projectors import ProjectorConfig
from .registry import CONFIG_CLS_REGISTRY


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


try:
    from composer.utils.dist import get_tp_group_size
except ImportError:
    logger.warning("Can not import modules for training. Probably running in inference mode.")


class BaseBackBoneLLMConfig(PretrainedConfig):
    model_type: str = "backbone_llm"
    llm_config_subfolder: str = "llm_config"
    def __init__(
        self,
        type: str = "llama",
        config_path: Optional[str] = None,
        pretrain_path: Optional[str] = None,
        freeze: bool = False,
        params: Optional[DictConfig] = None,
        **kwargs,
    ):
        """

        kwargs - kwargs for from_pretrained method for Config

        """
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        super().__init__(**kwargs)
        self.type = type
        self.config_path = config_path or pretrain_path
        self.pretrain_path = pretrain_path
        self.freeze = freeze
        self.config = self.get_config_and_override(params, trust_remote_code=trust_remote_code, **kwargs)

    def set_config(self):
        if self.config_path is None:
            self.config_path = str(Path(self.name_or_path) / self.llm_config_subfolder)
            self.config = self.get_config_and_override(None)

    def get_config_and_override(self, params, **kwargs):
        if self.type not in CONFIG_CLS_REGISTRY:
            raise ValueError(f"Can not init config with type {self.type}")

        dict_params = convert_to_dict(params) if params else kwargs.pop("config", {})
        dict_params["tp_size"] = dict_params.get("tp_size", 1)
        dict_params.update(kwargs)

        if "trust_remote_code" not in dict_params:
            dict_params["trust_remote_code"] = True

        if self.config_path:
            if "pretrained_model_name_or_path" in dict_params:
                del dict_params["pretrained_model_name_or_path"]
            config = CONFIG_CLS_REGISTRY[self.type].from_pretrained(self.config_path, **dict_params)
        else:
            config = CONFIG_CLS_REGISTRY[self.type](**dict_params)

        if not hasattr(config, "tp_size"):
            # If config does not have tp_size parameter, you are loading non-sber model 
            # and don't need following checks
            return config

        assert config.tp_size >= 1
        if config.tp_size > 1:
            tp_group_size = get_tp_group_size()
            assert config.tp_size == tp_group_size, f"Wrong tensor parallel group size. {tp_group_size} instead of {config.tp_size}"
        return config


def convert_to_dict(value : Optional[DictConfig]) -> dict:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value or {}


class GigaVisionConfig(PretrainedConfig):
    model_type = "giga_vision"
    vision_config_cls = VisionTowerConfig
    projector_config_cls = ProjectorConfig
    llm_config_cls = BaseBackBoneLLMConfig

    def __init__(
            self,
            vision_config: DictConfig = None,
            projector_config: DictConfig = None,
            llm_config: DictConfig = None,
            image_token: int = -1,
            video_token: int = -1,
            enable_async_tp: bool = False,
            **kwargs
        ):
        super().__init__(**kwargs)
        if enable_async_tp:
            llm_config.params.enable_async_tp = enable_async_tp

        self.vision_config = self.vision_config_cls(**convert_to_dict(vision_config))
        self.projector_config = self.projector_config_cls(**convert_to_dict(projector_config))
        self.llm_config = self.llm_config_cls(**convert_to_dict(llm_config))

        self.image_token = image_token
        if self.image_token == -1:
            logger.warning("Setting default value for `image_token = -1`. Probably it is not set in model config.")

        self.video_token = video_token
        if self.video_token == -1:
            logger.warning("Setting default value for `video_token = -1`. Probably it is not set in model config.")

        self.return_dict = self.llm_config.config.return_dict
        self.sp_split_type = getattr(self.llm_config.config, "sp_split_type", None)

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> Union["GigaVisionConfig", Tuple["GigaVisionConfig", dict]]:
        """
        Override method from PretrainedConfig
        Method is called in AutoConfig.from_pretrained and AutoModelForCasualLM.from_pretrained
        so we set name_or_path in nested config to correctly init model afterwards
        """
        return_kwargs = kwargs.get("return_unused_kwargs", False)
        cls.setup_name_or_path_dict(config_dict)
        out = super().from_dict(config_dict, **kwargs)
        if return_kwargs:
            cl, kwargs = out
        else:
            cl: "GigaVisionConfig" = out
        cl.setup_name_or_path()
        if return_kwargs:
            return cl, kwargs
        else:
            return cl

    def to_json_string(self, use_diff: bool = True) -> str:
        # Force json serialized config to be full, not diff
        return super().to_json_string(use_diff=False)

    @classmethod
    def setup_name_or_path_dict(cls, config_dict):
        np = config_dict.get("_name_or_path", "")
        if np != "":
            config_dict['llm_config']['config_path'] = os.path.join(np, cls.llm_config_cls.llm_config_subfolder)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Override from PretrainedConfig to pass _name_or_path in final dict. Need for working `GigaVisionConfig.from_pretrained`
        In vanilla method _name_or_path is not set in config_dict to use it `from_dict`
        """
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["_name_or_path"] = str(pretrained_model_name_or_path)
        return config_dict, kwargs
    
    def setup_name_or_path(self):
        "On inference time set name_or_path from main config to inner configs, to load model correctly"
        if self.name_or_path != "":
            self.llm_config.name_or_path = str(self.name_or_path)
            self.vision_config.name_or_path = str(self.name_or_path)
        self.vision_config.set_ve_config()
        self.llm_config.set_config()

    def _cp_modeling_files(self, src_path: str, dst_path: str):
        d = glob.glob(os.path.join(src_path, "*.py"))
        if d is not None:
            for p in d:
                shutil.copy(src=p, dst=os.path.join(dst_path, Path(p).name))

    def _save_full_config(self, config: PretrainedConfig, save_directory: str):
        # config._name_or_path = None
        c = config.to_dict()
        c["_name_or_path"] = ""
        c["pretrain_path"] = None
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(c, f, indent=2)
        return c

    def _save_partial_config(self, config: PretrainedConfig, save_directory: str, pretrain_path: Optional[str]):
        Path(save_directory).mkdir(exist_ok=True, parents=True)
        c = self._save_full_config(config, save_directory)
        if "auto_map" in c:
            scr_path = pretrain_path if pretrain_path is not None else config.name_or_path
            self._cp_modeling_files(scr_path, save_directory)
        config.name_or_path = ""
        if getattr(config, "pretrained_model_name_or_path", None) is not None:
            config.pretrained_model_name_or_path = None
        if getattr(config, "config_path", None) is not None:
            config.config_path = ""

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        vcs = os.path.join(save_directory, self.vision_config.vision_config_subfolder)
        lcs = os.path.join(save_directory, self.llm_config.llm_config_subfolder)
        self._save_partial_config(self.vision_config.ve_config, vcs, self.vision_config.pretrain_path)
        self._save_partial_config(self.llm_config.config, lcs, self.llm_config.pretrain_path)
        self.vision_config.pretrain_path = None
        self.llm_config.config_path = None
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)
