"""Focused public import surface for the vendored GigaChat LLM Foundry.

The upstream package eagerly imports every speech, vision, MoE, and internal
callback extension. The corresponding private git submodules were absent from
the supplied snapshot. Keep all source files in the vendor tree, but load only
the ordinary text-training components used by autoresearch v1.
"""

from llmfoundry.models.gigar import ComposerGigarCausalLM


def _unavailable_loader(*_args, **_kwargs):
    raise RuntimeError("Autoresearch v1 supports the LLM Foundry text loader only")


build_finetuning_dataloader = _unavailable_loader
build_text_denoising_dataloader = _unavailable_loader

COMPOSER_MODEL_REGISTRY = {"gigar_causal_lm": ComposerGigarCausalLM}

__all__ = [
    "COMPOSER_MODEL_REGISTRY",
    "build_finetuning_dataloader",
    "build_text_denoising_dataloader",
]

__version__ = "0.2.0"
