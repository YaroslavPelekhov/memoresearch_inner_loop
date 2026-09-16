import os

from typing import Union, List, Optional, Any
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig


logger = logging.get_logger(__name__)


class FastViTHDVisionConfig(PretrainedConfig):
    model_type = "fastvithd"

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 64,
        num_channels: int = 3,
        # Params from nn.Module from original code
        layers: Optional[List[int]] = None,
        token_mixers: Optional[List[str]] = None,
        embed_dims: Optional[List[int]] = None,
        mlp_ratios: Optional[List[int]] = None,
        downsamples: Optional[List[bool]] = None,
        se_downsamples: Optional[List[bool]] = None,
        repmixer_kernel_size: int = 3,
        norm_layer: str = "normlayerchannel",
        act_layer: str = "gelu",
        pos_embs: Optional[List[str]] = None,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        cls_ratio: float = 2.0,
        inference_mode: bool = True,
        stem_scale_branch: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # pos_embs, norm_layer, act_layer are parsed from str to actual values in FastViTHDVisionModel.__init__()

        if layers is None:
            layers = [2, 12, 24, 4, 2]
        if token_mixers is None:
            token_mixers = [
                "repmixer",
                "repmixer",
                "repmixer",
                "attention",
                "attention",
            ]
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768, 1536]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4, 4]
        if downsamples is None:
            downsamples = [True, True, True, True, True]
        if se_downsamples is None:
            se_downsamples = [False, False, False, False, False]
        if pos_embs is None:
            pos_embs = ["none", "none", "none", "repcpe", "repcpe"]

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.layers = layers
        self.token_mixers = token_mixers
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.downsamples = downsamples
        self.se_downsamples = se_downsamples
        self.repmixer_kernel_size = repmixer_kernel_size
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.pos_embs = pos_embs
        self.down_patch_size = down_patch_size
        self.down_stride = down_stride
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.cls_ratio = cls_ratio
        self.inference_mode = inference_mode
        self.stem_scale_branch = stem_scale_branch

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs: Any
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


AutoConfig.register("fastvithd", FastViTHDVisionConfig)
