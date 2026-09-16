import warnings

from transformers import LlamaConfig, LlamaForCausalLM
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.gigar.modelling_gigar import LlamaForCausalLM as GigarForCausalLM


MODEL_CLS_REGISTRY = {
    "llama": LlamaForCausalLM,
    "gigar": GigarForCausalLM
}

CONFIG_CLS_REGISTRY = {
    "llama": LlamaConfig,
    "gigar": GigarConfig
}
