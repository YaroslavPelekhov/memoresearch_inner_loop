from omegaconf import DictConfig

from .builder import build_acoustic_encoder, build_acoustic_projector, build_acoustic_subsampler
from .dummy_modules import DummyEncoderConfig
from .conformer import ConformerConfig
from .transformer import TransformerEncoderConfig
from .projectors import LinearProjectorConfig, MLPProjectorConfig, AttentionProjectorConfig
from .subsamplers import IdentitySubsamplerConfig, StackingSubsamplerConfig

ENCODER_CFG_RESOLVER = {
    "dummy": DummyEncoderConfig,
    "conformer": ConformerConfig,
    "transformer": TransformerEncoderConfig,
}

PROJECTOR_CFG_RESOLVER = {
    "linear_projector": LinearProjectorConfig,
    "mlp_projector": MLPProjectorConfig,
    "attention_projector": AttentionProjectorConfig,
}

SUBSAMPLER_CFG_RESOLVER = {
    "identity_subsampler": IdentitySubsamplerConfig,
    "stacking_subsampler": StackingSubsamplerConfig,
}


def build_encoder_config(encoder_cfg: DictConfig):
    assert encoder_cfg.get("model_type", "default") in ENCODER_CFG_RESOLVER, (
        f"Found model type {encoder_cfg.get('model_type', 'default')}, but " +
        f"allowed only {ENCODER_CFG_RESOLVER.keys()} !"
    )
    cfg_cls = ENCODER_CFG_RESOLVER[encoder_cfg.model_type](**encoder_cfg)
    return cfg_cls


def build_projector_config(projector_config: DictConfig):
    assert projector_config.get("model_type", "default") in PROJECTOR_CFG_RESOLVER
    cfg_cls = PROJECTOR_CFG_RESOLVER[projector_config.model_type](**projector_config)
    return cfg_cls


def build_subsampler_config(subsampler_config: DictConfig):
    assert subsampler_config.get("model_type", "default") in SUBSAMPLER_CFG_RESOLVER
    cfg_cls = SUBSAMPLER_CFG_RESOLVER[subsampler_config.model_type](**subsampler_config)
    return cfg_cls
