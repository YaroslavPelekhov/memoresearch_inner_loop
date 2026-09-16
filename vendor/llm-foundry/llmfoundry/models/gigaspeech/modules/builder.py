import typing as tp

from transformers import PretrainedConfig

from .dummy_modules import DummyEncoder
from .conformer import ConformerEncoder
from .transformer import TransformerEncoder
from .projectors import LinearProjector, MLPProjector
from .subsamplers import IdentitySubsampler, StackingSubsampler


def build_acoustic_encoder(
        config: PretrainedConfig, **kwargs: tp.Dict
    ) -> tp.Union[DummyEncoder, ConformerEncoder]:
    """Build acoustic encoder by provided config.
    """
    if config.model_type == "dummy":
        return DummyEncoder(config, **kwargs)
    elif config.model_type == "conformer":
        return ConformerEncoder(config, **kwargs)
    elif config.model_type == "transformer":
        return TransformerEncoder(config, **kwargs)
    else:
        raise NotImplementedError()


def build_acoustic_projector(
        config: PretrainedConfig, **kwargs: tp.Dict,
    ) -> tp.Union[LinearProjector, MLPProjector]:
    """Build acoustic projector by config.
    """
    if config.model_type == "linear_projector":
        return LinearProjector(config, **kwargs)
    elif config.model_type == "mlp_projector":
        return MLPProjector(config, **kwargs)
    else:
        raise NotImplementedError()


def build_acoustic_subsampler(
        config: PretrainedConfig, **kwargs: tp.Dict,
    ) -> tp.Union[LinearProjector, MLPProjector]:
    """Build acoustic subsampler by config.
    """
    if config.model_type == "identity_subsampler":
        return IdentitySubsampler(config, **kwargs)
    elif config.model_type == "stacking_subsampler":
        return StackingSubsampler(config, **kwargs)
    else:
        raise NotImplementedError()
