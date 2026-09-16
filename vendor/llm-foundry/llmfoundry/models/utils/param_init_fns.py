# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn

from llmfoundry.models.layers.custom_embedding import (
    EmbeddingParallelEmbedding,
    VocabParallelEmbedding,
)
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
try:
    from llmfoundry.models.layers.lora_layer import LoraLinear, LoraGroupedMLP
    from llmfoundry.models.layers.moe import GroupedAbstractMLP, MoEGate
except ImportError:
    class _UnavailablePrivateLayer(nn.Module):
        pass

    LoraLinear = LoraGroupedMLP = _UnavailablePrivateLayer
    GroupedAbstractMLP = MoEGate = _UnavailablePrivateLayer
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY, ZERO_CENTERED_NORM_CLASSES, GATED_NORM_CLASSES
from llmfoundry.models.layers.attention import Qwen3NextGatedDeltaNet, KimiDeltaAttention
from fla.modules import FusedRMSNormGated

def renorm_keep_std(weight: torch.Tensor, dim: int = 0):
    with torch.no_grad():
        std = weight.std()
        weight.div_(weight.norm(dim=dim, keepdim=True))
        weight.mul_(std / weight.std())


def set_module_custom_std(
    init_config: dict, fan_in: int, fan_out: int, gain: float = 1.0
):
    if init_config["name"] == "xavier_normal_":
        init_std = gain * math.sqrt(2 / (fan_in + fan_out))  # xavier normal std
        init_config["name"] = "baseline_"
        init_config["init_std"] = init_std
    else:
        raise NotImplementedError(
            "Custom init std is not supported for non xavier normal method yet!"
        )

    return init_config


def torch_default_param_init_fn_(
    module: nn.Module,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(f"Initializing network using module's reset_parameters attribute")

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()  # type: ignore


def fused_init_helper_(module: nn.Module, init_fn_: Callable):
    # parameter initialization is often based on the parameters shape.
    # If a layer is fused, initialization should be based on the shapes
    # of the original tensor instead of the shape of the fused tensor.
    # Layers which are fused should have the _fused attribute defined.
    # The first element of _fused is the dimension along which the tensor is fused.
    # This is followed by an iterable of split indices."

    _fused = getattr(module, "_fused", None)

    if _fused is None:
        raise RuntimeError(f"Internal logic error")

    dim, splits = _fused
    splits = (0, *splits, module.weight.size(dim))  # type: ignore
    for s, e in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim  # type: ignore
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])  # type: ignore


def generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    """Generic parameter init that applies `init_fn_` per module type."""
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(f"If model has bias parameters they are initialized to 0.")

    if hasattr(module, "skip_init") and module.skip_init:
        return

    # enable user to divide _is_residual weights by
    # a value which defaults to math.sqrt(2 * cfg.n_layers)
    init_div_is_residual = init_div_is_residual

    if init_div_is_residual is False:
        # not used, for pyright
        div_is_residual = 1.0
    elif init_div_is_residual is True:
        div_is_residual = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual, float) or isinstance(
        init_div_is_residual, int
    ):
        div_is_residual = init_div_is_residual
    elif (
        isinstance(
            init_div_is_residual,  # type: ignore
            str,
        )
        and init_div_is_residual.isnumeric()
    ):
        # do not trust YAML parsing to always convert numbers to numbers
        div_is_residual = float(init_div_is_residual)
    else:
        # not used, for pyright
        div_is_residual = 1.0
        raise ValueError(
            f"Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}"
        )

    if init_div_is_residual is not False:
        if verbose > 1:
            warnings.warn(
                f"Initializing _is_residual layers then dividing them by {div_is_residual:.3f}. "
                + f"Set `init_div_is_residual: false` in init config to disable this."
            )

    if isinstance(module, LoraLinear):
        module.reset_parameters()

    if isinstance(module, LoraGroupedMLP):
        module.reset_parameters()

    if isinstance(module, tuple(set(FC_CLASS_REGISTRY.values()))):
        # Linear

        # Special case for Tensor Parallel classes
        if isinstance(
            module,
            (
                FC_CLASS_REGISTRY["RowParallelLinear"],
                FC_CLASS_REGISTRY["ColumnParallelLinear"],
            ),
        ):
            module._init_params["init_method"] = init_fn_
            module.reset_parameters()

            # Handle gate renormalization for TP layers
            # This version is naive. For correct ColumnParallel we need communication
            if getattr(module, '_is_moe_gate', False) and getattr(module, '_renorm_router_weights', False):
                raise AssertionError("Doesn't work correctly for tp > 1. TODO add weights statistics communication")
                # weight = None
                # if hasattr(module, 'weight') and module.weight is not None:
                #     weight = module.weight
                # elif hasattr(module, '_tp_linear_submodule'):
                #     weight = module._tp_linear_submodule.weight

                # if weight is not None:
                #     renorm_keep_std(weight, dim=1)

            return

        if hasattr(module, "_fused"):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
            if getattr(module, '_is_moe_gate', False) and getattr(module, '_renorm_router_weights', False):
                # Normalize across the input dimension (dim=1) for each expert
                renorm_keep_std(module.weight, dim=1)

        if module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)
            torch.nn.init.zeros_(module.bias)

        if init_div_is_residual is not False and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(div_is_residual)  # type: ignore

    # legacy router init
    elif isinstance(module, MoEGate) and isinstance(getattr(module, "weight", None), nn.Parameter):
        # NOTE: we don't enter this block as MoEGate has no weights^ only MoEGate.gate
        # MoE
        module.reset_parameters()

    elif isinstance(module, GroupedAbstractMLP):
        module._init_params["init_method"] = init_fn_
        module.reset_parameters()

    elif isinstance(module, nn.Embedding):
        # Embedding
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f"Embedding layer initialized to 0.")
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
            if verbose > 1:
                warnings.warn(
                    f"Embedding layer initialized using normal distribution with mean=0 and {std=}."
                )

        elif emb_init_uniform_lim is not None:
            lim = emb_init_uniform_lim
            if isinstance(lim, Sequence):
                if len(lim) > 2:
                    raise ValueError(
                        f"Uniform init requires a min and a max limit. User input: {lim}."
                    )
                if lim[0] == lim[1]:
                    warnings.warn(f"Embedding layer initialized to {lim[0]}.")
            else:
                if lim == 0:
                    warnings.warn(f"Embedding layer initialized to 0.")
                lim = [-lim, lim]
            a, b = lim
            emb_init_fn_ = partial(torch.nn.init.uniform_, a=a, b=b)
            if verbose > 1:
                warnings.warn(
                    f"Embedding layer initialized using uniform distribution in range {lim}."
                )
        else:
            emb_init_fn_ = init_fn_

        emb_init_fn_(module.weight)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)

    elif isinstance(module, (VocabParallelEmbedding, EmbeddingParallelEmbedding)):
        # Embeddings for Tensor Parallel
        init_method = partial(
            module._init_embeddings,
            init_method=init_fn_,
            padding_idx=module.padding_idx,
        )
        module._init_params["init_method"] = init_method
        module.reset_parameters()

    elif isinstance(module, tuple(set(NORM_CLASS_REGISTRY.values()))):  # type: ignore
        # Norm
        if verbose > 1:
            warnings.warn(
                f"Norm weights are set to 1. If norm layer has a bias it is initialized to 0."
            )
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, ZERO_CENTERED_NORM_CLASSES):
                # Keep neutral start for `1 + weight` parameterization.
                torch.nn.init.zeros_(module.weight)  # type: ignore
            else:
                torch.nn.init.ones_(module.weight)  # type: ignore

        
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)  # type: ignore

    elif isinstance(module, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)  # type: ignore

    elif isinstance(module, (Qwen3NextGatedDeltaNet, KimiDeltaAttention, FusedRMSNormGated)):
        module.reset_parameters()

    elif isinstance(module, LoraGroupedMLP):
        # LoRA weights already initialized above via _init_lora_weights()
        pass

    else:
        for _ in module.parameters(recurse=False):
            # raise error if uninitialized module has any parameters
            raise NotImplementedError(
                f"{module.__class__.__name__} parameters are not initialized by param_init_fn."
            )


def _normal_init_(std: float, mean: float = 0.0):
    return partial(torch.nn.init.normal_, mean=mean, std=std)


def _normal_param_init_fn_(
    module: nn.Module,
    std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    init_fn_ = _normal_init_(std=std)

    if verbose > 1:
        warnings.warn(f"Using torch.nn.init.normal_ init fn mean=0.0, std={std}")

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def baseline_param_init_fn_(
    module: nn.Module,
    init_std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    if init_std is None:
        raise ValueError(
            "You must set model.init_config['init_std'] to a float value to use the default initialization scheme."
        )
    _normal_param_init_fn_(
        module=module,
        std=init_std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def small_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    # very close to kaiming normal
    # from Transformers without Tears (2019) - Nguyen & Salazar
    std = math.sqrt(2 / (5 * d_model))
    _normal_param_init_fn_(
        module=module,
        std=std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def neox_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs  # unused, just to capture any extra args from the config
    residual_div = n_layers / math.sqrt(10)  # small std / wang std

    if verbose > 1:
        warnings.warn(f"setting init_div_is_residual to {residual_div}")

    small_param_init_fn_(
        module=module,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=residual_div,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def kaiming_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = "fan_in",
    init_nonlinearity: str = "leaky_relu",
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config

    if verbose > 1:
        warnings.warn(
            f"Using nn.init.kaiming_uniform_ init fn with parameters: "
            + f"a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}"
        )

    kaiming_uniform_ = partial(
        nn.init.kaiming_uniform_,
        a=init_gain,
        mode=fan_mode,
        nonlinearity=init_nonlinearity,
    )

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def kaiming_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = "fan_in",
    init_nonlinearity: str = "leaky_relu",
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config

    if verbose > 1:
        warnings.warn(
            f"Using nn.init.kaiming_normal_ init fn with parameters: "
            + f"a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}"
        )

    kaiming_normal_ = partial(
        torch.nn.init.kaiming_normal_,
        a=init_gain,
        mode=fan_mode,
        nonlinearity=init_nonlinearity,
    )

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def xavier_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    if verbose > 1:
        warnings.warn(
            f"Using torch.nn.init.xavier_uniform_ init fn with parameters: "
            + f"gain={init_gain}"
        )

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def xavier_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    verbose: int = 0,
    **kwargs: Any,
):
    del kwargs  # unused, just to capture any extra args from the config
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    if verbose > 1:
        warnings.warn(
            f"Using torch.nn.init.xavier_normal_ init fn with parameters: "
            + f"gain={init_gain}"
        )

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def dclm_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = False,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    """DCLM style initialization."""
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(
            "Using DCLM init: trunc_normal with std=1/sqrt(fan_in), depth scaling for output projection layers."
        )

    INIT_LAYERS_POOL = tuple(
        list(FC_CLASS_REGISTRY.values())
        + [nn.Embedding, EmbeddingParallelEmbedding, VocabParallelEmbedding]
    )
    if isinstance(module, INIT_LAYERS_POOL):
        init_std = getattr(module, "_init_std", None)
        if init_std is None:
            raise RuntimeError(
                f"DCLM style init expects FC and Embedding layers to have `_init_std` attribute, but got `None` for module {module}."
            )

        init_fn_ = partial(
            torch.nn.init.trunc_normal_,
            mean=0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
    else:
        # A fallback init fn.
        init_fn_ = partial(torch.nn.init.xavier_normal_, gain=1)

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        n_layers=n_layers,
        d_model=d_model,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def olmo3_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = False,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs: Any,
):
    """OLMo3 / nGPT-style \"normalized\" initialization.

    Embeddings: normal_(mean=0, std=d_model**-0.5).
    Linears: trunc_normal_(mean=0, std=_olmo3_init_std, a=-3*std, b=3*std).
    Per-linear std is set by set_olmo3_init_std_on_module_tree() from
    (d_model, n_layers, intermediate_size); no init logic in attention/ffn.
    """
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(
            "Using OLMo3 init: normal_ for embeddings (std=d_model**-0.5), "
            + "trunc_normal_ for linears with depth scaling (same as DCLM)."
        )

    INIT_LAYERS_POOL = tuple(
        list(FC_CLASS_REGISTRY.values())
        + [nn.Embedding, EmbeddingParallelEmbedding, VocabParallelEmbedding]
    )
    if isinstance(module, INIT_LAYERS_POOL):
        init_std = getattr(module, "_init_std", None)
        if init_std is None:
            raise RuntimeError(
                f"Olmo3 style init expects FC and Embedding layers to have `_init_std` attribute, but got `None` for module {module}."
            )

        if isinstance(
            module, (nn.Embedding, EmbeddingParallelEmbedding, VocabParallelEmbedding)
        ):
            init_fn_ = partial(
                torch.nn.init.normal_,
                mean=0,
                std=init_std,
            )
        else:
            init_fn_ = partial(
                torch.nn.init.trunc_normal_,
                mean=0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
    else:
        # A fallback init fn.
        init_fn_ = partial(torch.nn.init.xavier_normal_, gain=1)

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        n_layers=n_layers,
        d_model=d_model,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


MODEL_INIT_REGISTRY = {
    "default_": torch_default_param_init_fn_,
    "baseline_": baseline_param_init_fn_,
    "kaiming_uniform_": kaiming_uniform_param_init_fn_,
    "kaiming_normal_": kaiming_normal_param_init_fn_,
    "neox_init_": neox_param_init_fn_,
    "small_init_": small_param_init_fn_,
    "xavier_uniform_": xavier_uniform_param_init_fn_,
    "xavier_normal_": xavier_normal_param_init_fn_,
    "dclm_init_": dclm_param_init_fn_,
    "olmo3_init_": olmo3_param_init_fn_,
}
