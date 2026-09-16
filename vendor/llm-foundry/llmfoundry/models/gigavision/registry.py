import warnings

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoConfig,
    LlamaConfig,
)

MODEL_CLS_REGISTRY = {
    "deepseek": AutoModelForCausalLM,
    "llama": LlamaForCausalLM,
    "qwen3": AutoModelForCausalLM,
}

CONFIG_CLS_REGISTRY = {
    "llama": LlamaConfig,
    "deepseek": AutoConfig,
    "qwen3": AutoConfig,
}

not_imported_modules = []

try:
    from llmfoundry.models.gigar.modelling_gigar import LlamaForCausalLM as GigarForCausalLM
    from llmfoundry.models.gigar import GigarConfig

    MODEL_CLS_REGISTRY.update({"gigar": GigarForCausalLM})
    CONFIG_CLS_REGISTRY.update({"gigar": GigarConfig})
except ImportError:
    not_imported_modules.extend(["GigarForCausalLM", "GigarConfig"])


try:
    from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
    from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

    MODEL_CLS_REGISTRY.update({"giga_mix_causal_lm": GigaMixForCausalLM})
    CONFIG_CLS_REGISTRY.update({"giga_mix_causal_lm": GigaMixConfig})
except ImportError:
    not_imported_modules.extend(["GigaMixForCausalLM", "GigaMixConfig"])


if not_imported_modules:
    warnings.warn(f"Can not import modules: {not_imported_modules}. Probably running in inference mode.")
