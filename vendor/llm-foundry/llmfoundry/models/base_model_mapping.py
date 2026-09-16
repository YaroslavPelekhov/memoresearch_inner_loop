from llmfoundry.models.gigar import ComposerGigarCausalLM, LlamaForCausalLM, GigarConfig
from llmfoundry.models.giga_mix import (
    ComposerGigaMixCausalLM,
    GigaMixForCausalLM,
    GigaMixConfig,
)
from llmfoundry.models.gigavision import ComposerGigaVisionCausalLM
from llmfoundry.models.gigavision.modelling_gigavision import GigaVisionForCausalLM
from llmfoundry.models.gigavision.configuration_gigavision import GigaVisionConfig


def get_composer_model_class(model_type):
    HF_TO_COMPOSER_REGISTRY = {
        GigaMixForCausalLM: ComposerGigaMixCausalLM,
        LlamaForCausalLM: ComposerGigarCausalLM,
        GigaVisionForCausalLM: ComposerGigaVisionCausalLM,
    }
    if model_type not in HF_TO_COMPOSER_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_type}")

    return HF_TO_COMPOSER_REGISTRY[model_type]


def get_inner_model(model_name: str):
    MODEL_REGISTRY = {
        "gigar_causal_lm": LlamaForCausalLM,
        "giga_mix_causal_lm": GigaMixForCausalLM,
        "gigavision_causal_lm": GigaVisionForCausalLM,
    }
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_name}")
    return MODEL_REGISTRY[model_name]


def get_model_config(config_name: str):
    CONFIG_REGISTRY = {
        "gigar_causal_lm": GigarConfig,
        "giga_mix_causal_lm": GigaMixConfig,
        "gigavision_causal_lm": GigaVisionConfig,
    }

    if config_name is None:
        raise ValueError("Please specify the model name in the yaml config")
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={config_name}")
    return CONFIG_REGISTRY[config_name]
