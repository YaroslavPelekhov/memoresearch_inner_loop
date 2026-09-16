import typing as tp

from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from llmfoundry.models.gigar.modelling_gigar import LlamaForCausalLM
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

DECODER_CONFIG_CLASS_RESOLVER: tp.Dict[str, PreTrainedModel] = {
    "gigar": GigarConfig,
    "giga_mix": GigaMixConfig,
}

DECODER_MODEL_CLASS_RESOLVER: tp.Dict[str, PreTrainedModel] = {
    "gigar": LlamaForCausalLM,
    "giga_mix": GigaMixForCausalLM,
}


def build_decoder(
        config: PretrainedConfig, **kwargs: tp.Dict,
    ) -> tp.Union[LlamaForCausalLM, GigaMixForCausalLM]:
    """Build encoder by config.
    """
    return DECODER_MODEL_CLASS_RESOLVER[config.model_type](config, **kwargs)
