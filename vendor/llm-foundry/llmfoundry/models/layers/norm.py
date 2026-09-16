# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Type, Union
try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
except ImportError:
    def dropout_add_rms_norm(
        x0,
        residual,
        weight,
        bias,
        dropout_p,
        epsilon,
        rowscale,
        layerscale,
    ):
        del residual, bias, dropout_p, rowscale, layerscale
        output = x0.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + epsilon)
        return (output * weight.float()).to(dtype=x0.dtype)

import torch._inductor.config as config

config.max_autotune_gemm_backends = "ATEN,TRITON,NVGEMM"
config.max_autotune_conv_backends = "ATEN,TRITON,NVGEMM"

# avoid recompiles
torch.fx.experimental._config.use_duck_shape = False

def _cast_if_autocast_enabled(tensor: torch.Tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_zero_centered_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_zero_centered_gated_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    eps: float,
    layernorm_gating_weight: float,
):
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    output = (output * (1.0 + weight.float())).to(dtype=x.dtype)
    gate_hidden = F.linear(output, gate_up_weight.to(dtype=x.dtype))
    gate_hidden = F.silu(gate_hidden.float()).to(dtype=x.dtype)
    gate_hidden = F.linear(gate_hidden, gate_down_weight.to(dtype=x.dtype))
    gate = torch.sigmoid(gate_hidden.float()).to(dtype=x.dtype)
    return (output * float(layernorm_gating_weight) * gate).to(dtype=x.dtype)


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-5):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        weight: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.empty(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

        if device != "meta":
            torch.nn.init.ones_(self.weight)

    @classmethod
    def from_config(cls, config, input_dim: Optional[int] = None):
        """Alternative constructor that takes a config object."""
        return cls(
            normalized_shape=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    def forward(self, x: torch.Tensor):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


class LlamaRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device))
        self.variance_epsilon = eps

        if hidden_size <= 8192:
            self.rmsnorm_fn = dropout_add_rms_norm
        else:
            self.rmsnorm_fn = torch.compile(self.compute_rmsnorm, fullgraph=True)

        if device != "meta":
            torch.nn.init.ones_(self.weight)

    @classmethod
    def from_config(cls, config, input_dim: Optional[int] = None):
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, hidden_states: torch.Tensor):
        downcast_hidden_states = _cast_if_autocast_enabled(hidden_states)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )

        if self.rmsnorm_fn is dropout_add_rms_norm:
            return self.rmsnorm_fn(
                x0=downcast_hidden_states,
                residual=None,
                weight=downcast_weight,
                bias=None,
                dropout_p=0.0,
                epsilon=self.variance_epsilon,
                rowscale=None,
                layerscale=None,
            )
        else:
            return self.rmsnorm_fn(
                x=downcast_hidden_states,
                weight=downcast_weight,
                eps=self.variance_epsilon,
            )


# NOTE: Original from Qwen3NextRMSNorm
class ZeroCenteredRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, device=device))

    @classmethod
    def from_config(cls, config, input_dim: Optional[int] = None):
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @staticmethod
    def compute_zero_centered_rmsnorm(
        x: torch.Tensor, weight: torch.Tensor, eps: float
    ):
        return _compiled_zero_centered_rmsnorm(x=x, weight=weight, eps=eps)

    def forward(self, x):
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.float32,
        ):
            output = self.compute_zero_centered_rmsnorm(
                x=x,
                weight=self.weight,
                eps=self.eps,
            )
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class ZeroCenteredGatedNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.r = 16  # TODO: fix magic constant
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, device=device))
        self.gate_up_projection = torch.nn.Linear(
            hidden_size, self.r, bias=False, device=device
        )
        self.gate_down_projection = torch.nn.Linear(
            self.r, hidden_size, bias=False, device=device
        )
        self.variance_epsilon = eps
        self.layernorm_gating_weight = layernorm_gating_weight

    @classmethod
    def from_config(cls, config, input_dim: Optional[int] = None):
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            layernorm_gating_weight=getattr(config, "layernorm_gating_weight", 2.0),
            device=getattr(config, "init_device", None),
        )

    @staticmethod
    def compute_zero_centered_gated_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_down_weight: torch.Tensor,
        eps: float,
        layernorm_gating_weight: float,
    ):
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(x, 2)
        return _compiled_zero_centered_gated_norm(
            x=x,
            weight=weight,
            gate_up_weight=gate_up_weight,
            gate_down_weight=gate_down_weight,
            eps=eps,
            layernorm_gating_weight=layernorm_gating_weight
        )

    def forward(self, x: torch.Tensor):
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.float32,
        ):
            output = self.compute_zero_centered_gated_norm(
                x=x,
                weight=self.weight,
                gate_up_weight=self.gate_up_projection.weight,
                gate_down_weight=self.gate_down_projection.weight,
                eps=self.variance_epsilon,
                layernorm_gating_weight=self.layernorm_gating_weight,
            )

        return output


NORM_CLASS_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "rmsnorm": RMSNorm,
    "LlamaRMSNorm": LlamaRMSNorm,
    "ZeroCenteredRMSNorm": ZeroCenteredRMSNorm,
    "ZeroCenteredGatedNorm": ZeroCenteredGatedNorm,
}

ZERO_CENTERED_NORM_CLASSES = (ZeroCenteredRMSNorm, ZeroCenteredGatedNorm)

GATED_NORM_CLASSES = (ZeroCenteredGatedNorm,)


def resolve_norm_class(norm_type: str) -> Type[torch.nn.Module]:
    if not isinstance(norm_type, str):
        raise TypeError(
            f"`norm_type` must be a string, got {type(norm_type).__name__}."
        )

    raw_norm_type = norm_type
    norm_type = norm_type.strip()
    if not norm_type:
        raise ValueError("`norm_type` must be a non-empty string.")

    norm_class = NORM_CLASS_REGISTRY.get(norm_type)
    if norm_class is None:
        available_norms = ", ".join(sorted(NORM_CLASS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown norm type `{raw_norm_type}`. Available values: {available_norms}."
        )
    return norm_class
