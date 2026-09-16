import math
import warnings
from typing import Callable, Optional

import torch
import transformer_engine_torch as tex
from torch import nn
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
)
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorBase,
)


def check_dim_for_fp8(tensor: torch.Tensor) -> bool:
    """Assert that tensor or tensors dimensions are supported for FP8 TN GEMM."""
    return math.prod(tensor.size()[:-1]) % 128 == 0 and tensor.size()[-1] % 128 == 0


def pad_x_to_y_divisible(x: int, y: int) -> int:
    return (y - x % y) % y


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    return isinstance(tensor, (QuantizedTensorBase, QuantizedTensor))


def pad_tensor_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    pad_tuple = ()

    if not check_dim_for_fp8(tensor):
        pad_last_dim = pad_x_to_y_divisible(tensor.size()[-1], 128)
        pad_tuple += (0, pad_last_dim)
        if len(tensor.size()) > 1:
            if len(tensor.size()) == 2 or math.prod(tensor.size()[:-1]) % 128 != 0:
                pad_pen_dim = pad_x_to_y_divisible(tensor.size()[-2], 128)
                pad_tuple += (0, pad_pen_dim)
        padded_tensor = nn.functional.pad(tensor, pad_tuple, value=0.0)
        return padded_tensor
    return tensor


def unpad_tensor_if_needed(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if tensor.size() != shape:
        if len(tensor.size()) > 1:
            h, w = shape[-2:]
            return tensor[..., :h, :w].contiguous()
        else:
            return tensor[:shape[-1]].contiguous()
    return tensor


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """Generic function to swap linear layers in a module with a new type of linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if isinstance(module, nn.Linear) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        return from_float_func(module)

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if isinstance(module, nn.Linear) and (
            module_filter_fn is None or module_filter_fn(module, cur_fqn)
        ):
            new_linear_module = from_float_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_linear_module)

    post_order_traversal(root_module)
    return root_module


def shared_quantization_hook(module: nn.Module, inputs):
    if _is_quantized_tensor(inputs[0]):
        return inputs[0], *inputs[1:]
    else:
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=1,
        )
        prev_internal = quantizer.internal
        quantizer.internal = False
        pad_inp = pad_tensor_if_needed(inputs[0])
        quantized_inp = quantizer(pad_inp)
        quantizer.internal = prev_internal
        quantized_inp.base_shape = inputs[0].shape
        return quantized_inp, *inputs[1:]


def add_shared_quantizers(
    module: nn.Module,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """Generic function to add shared quantizers to a module."""

    if module_filter_fn(module, ""):
        module.register_forward_pre_hook(shared_quantization_hook)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):

        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            if module_filter_fn(child_module, new_fqn):
                child_module.register_forward_pre_hook(shared_quantization_hook)

            post_order_traversal(child_module, new_fqn, module)

    post_order_traversal(root_module)
    return root_module


def _sm90_and_deepgemm_stable_version_check():
        import deep_gemm
        assert torch.cuda.is_available(), "CUDA is not available, training with FP8 is not supported"
        device_capability = torch.cuda.get_device_capability()
        gigagemm_stable_commit = "e57ff37"
        gigagemm_version = deep_gemm.__commit__
        if device_capability != (9, 0):
            raise RuntimeError(
                f"For DeepGEMM expected SM90, but got {device_capability}."
            )
        if gigagemm_version != gigagemm_stable_commit:
            warnings.warn(
                f"DeepGEMM stable commit is {gigagemm_stable_commit}, but got {gigagemm_version}."
                " Rewarmup may be required. Check llmfoundry/models/ops/warmup/README.md (DeepGEMM section) for more details."
            )
