from typing import Dict, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from torch import nn
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)
from transformer_engine.pytorch.tensor.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
)

from .utils.fp8_utils import (
    _is_quantized_tensor,
    pad_tensor_if_needed,
    unpad_tensor_if_needed,
)
from .triton_kernels.row2col_dense_kernels import (
    block_quantization_transpose,
)
from .cuda_kernels import (
    blockwise_scaling_aware_fp8_transpose,
)


def clear_tensor_data(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    for t in tensors:
        if t is not None:
            if isinstance(t, Float8BlockwiseQTensor):
                t.clear()
            else:
                t.data = torch.Tensor()
            del t

def get_block_quant_transposed_tensor(weight: torch.Tensor) -> Float8BlockwiseQTensor:
    """Get columnwise 2D scaling weights on backward

    Args:
        weight (torch.Tensor): weight to be transposed
    Returns:
        Float8BlockwiseQTensor: columnwise 2D scaling weights
    """

    columnwise_data, columnwise_scale_inv = block_quantization_transpose(weight)
    q_tensor = Float8BlockwiseQTensor(
        shape=columnwise_data.shape,
        dtype=torch.bfloat16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise_data=None,
        rowwise_scale_inv=None,
        columnwise_data=columnwise_data.view(torch.uint8),
        columnwise_scale_inv=columnwise_scale_inv,
        is_2D_scaled=True,
        quantizer=None
    )
    return q_tensor


def get_columnwise_quantized_tensor(inp_quantized: Float8BlockwiseQTensor) -> Float8BlockwiseQTensor:
    """Get columnwise quantized tensor from rowwise quantized tensor

    Args:
        inp_quantized (Float8BlockwiseQTensor): rowwise quantized tensor
    Returns:
        Float8BlockwiseQTensor: columnwise quantized tensor
    """

    if inp_quantized._columnwise_data is not None:
        return inp_quantized

    rowwise_data = inp_quantized._rowwise_data
    h_size = rowwise_data.shape[-1]
    rowwise_data = rowwise_data.reshape(-1, h_size)
    rowwise_scale_inv = inp_quantized._rowwise_scale_inv
    if rowwise_scale_inv.shape[0] != rowwise_data.shape[0]:
        if not rowwise_scale_inv.is_contiguous():
            rowwise_scale_inv = rowwise_scale_inv.T.contiguous()
        else:
            rowwise_scale_inv = rowwise_scale_inv.T

    (
        columnwise_data,
        columnwise_scale_inv,
    ) = blockwise_scaling_aware_fp8_transpose(rowwise_data, rowwise_scale_inv)
    inp_quantized = Float8BlockwiseQTensor(
        shape=columnwise_data.shape,
        dtype=torch.bfloat16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise_data=None,
        rowwise_scale_inv=None,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        is_2D_scaled=False,
        quantizer=None
    )
    clear_tensor_data(rowwise_data, rowwise_scale_inv)
    return inp_quantized


class FP8Gemm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor | Float8BlockwiseQTensor,
        weight: torch.Tensor,
        dtype: torch.dtype,
        act_requires_grad: bool,
        input_quantizer: Optional[Float8BlockQuantizer] = None,
        weight_quantizer: Optional[Float8BlockQuantizer] = None,
        grad_quantizer: Optional[Float8BlockQuantizer] = None,
        residual: Optional[torch.Tensor] = None,
        is_recompute: bool = False,
        do_unpad_out: bool = True
    ) -> torch.Tensor:
        """Forward autograd function

        Args:
            inp (torch.Tensor | Float8BlockwiseQTensor): input tensor
            weight (torch.Tensor): weight tensor
            dtype (torch.dtype): data type for GEMM output
            act_requires_grad (bool): requires grad for input tensor
            input_quantizer (Optional[Float8BlockQuantizer]): input quantizer for input tensor
            weight_quantizer (Optional[Float8BlockQuantizer]): weight quantizer for weight tensor
            grad_quantizer (Optional[Float8BlockQuantizer]): gradient quantizer for gradients
                                                  on backward pass
            residual (Optional[torch.Tensor]): residual connection tensor
            is_recompute (bool): requantize fp8 weights on backward
            do_unpad_out (bool): do unpad gemm output if needed
        Returns:
            Torch.Tensor: forward output tensor
        """

        ctx.inp_shape = getattr(inp, "base_shape", inp.size())
        ctx.inp_cur_shape = inp.size()
        ctx.w_shape = weight.shape
        out_features, _in_features = ctx.w_shape
        ctx.out_shape = (*list(ctx.inp_shape[:-1]), out_features)

        inp_device = inp.device
        ctx.do_unpad_out = do_unpad_out
        ctx.has_residual = residual is not None

        # get quantized inputs
        inp_is_quantized = _is_quantized_tensor(inp)
        if inp_is_quantized:
            inp_quantized = inp
        else:
            inp = pad_tensor_if_needed(inp)
            inp_quantized = tex.quantize(inp, input_quantizer)

        # get quantized weights
        weight = pad_tensor_if_needed(weight)
        ctx.padded_out_shape = (*inp.size()[:-1], weight.size(0))

        if not inp.requires_grad:
            del inp

        quantizer_internal = weight_quantizer.internal
        weight_quantizer.internal = False
        weight_quantized = weight_quantizer.quantize(weight)
        weight_quantizer.internal = quantizer_internal

        # TODO (veveselov): diverge if fuse residual, skip now
        if ctx.has_residual:
            gemm_out = residual.contiguous() if not residual.is_contiguous() \
                                                    else residual
            if is_recompute:
                gemm_out = gemm_out.clone()
            gemm_out = pad_tensor_if_needed(gemm_out)
        else:
            gemm_out = torch.empty(ctx.padded_out_shape, dtype=dtype, device=inp_device)

        # Forward GEMM
        # Note: y = x * w^T
        general_gemm(
            weight_quantized,
            inp_quantized,
            get_workspace(),
            out=gemm_out,
            out_dtype=dtype,
            bias=None,
            grad=False,
            use_split_accumulator=True,
            accumulate=ctx.has_residual
        )
        if ctx.do_unpad_out:
            gemm_out = unpad_tensor_if_needed(gemm_out, ctx.out_shape)

        ctx.needs_wgrad = weight.requires_grad
        if ctx.needs_wgrad:
            args = (inp_quantized, weight)
        else:
            args = (weight,)
        tensors_to_save, tensor_objects = prepare_for_saving(*args)
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_quantizer = grad_quantizer
        ctx.dtype = dtype
        ctx.act_requires_grad = act_requires_grad
        return gemm_out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor | Float8BlockwiseQTensor
    ) -> Tuple[torch.Tensor | None, ...]:
        """Backward autograd function

        Args:
            grad_output (torch.Tensor | Float8BlockwiseQTensor): gradients tensor
        Returns:
            Tuple[Union[torch.Tensor, None], ...]: gradients tensors for input,
                                                   weights and bias
        """

        needs_wgrad = ctx.needs_wgrad
        if needs_wgrad:
            inp_quantized, weight = (
                restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            )
        else:
            weight, = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        ctx.tensor_objects = None

        if _is_quantized_tensor(weight):
            if not ctx.weight_quantizer.columnwise_usage:
                weight.update_usage(columnwise_usage=True)
                weight.update_usage(rowwise_usage=False)
        else:
            weight = pad_tensor_if_needed(weight)
            weight = get_block_quant_transposed_tensor(weight)

        grad_is_quantized = _is_quantized_tensor(grad_output)
        if grad_is_quantized:
            grad_output_quantized = grad_output
        else:
            grad_output = pad_tensor_if_needed(grad_output)
            grad_output_quantized = tex.quantize(grad_output,
                                                 ctx.grad_quantizer)
        # dgrad GEMM
        # Note: dx = dy * w
        dgrad = None
        if ctx.act_requires_grad:
            dgrad, *_ = general_gemm(
                weight,
                grad_output_quantized,
                get_workspace(),
                layout="NN",
                grad=True,
                out_dtype=ctx.dtype,
                use_split_accumulator=True
            )
            dgrad = unpad_tensor_if_needed(dgrad, ctx.inp_cur_shape)

        # wgrad GEMM (skipped entirely when weights are frozen)
        # Note: dw = dy^T * x
        wgrad = None
        if needs_wgrad:
            if inp_quantized._columnwise_data is None:
                inp_quantized = get_columnwise_quantized_tensor(inp_quantized)
            if grad_output_quantized._columnwise_data is None:
                grad_output_quantized = get_columnwise_quantized_tensor(grad_output_quantized)

            wgrad, *_ = general_gemm(
                inp_quantized,
                grad_output_quantized,
                get_workspace(),
                layout="NT",
                grad=True,
                bias=None,
                out_dtype=ctx.dtype,
                use_split_accumulator=True
            )
            wgrad = unpad_tensor_if_needed(wgrad, ctx.w_shape)
            clear_tensor_data(inp_quantized, grad_output_quantized)
        else:
            clear_tensor_data(grad_output_quantized)

        return (
            dgrad,
            wgrad,
            None,  # dtype
            None,  # act_requires_grad
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # grad_quantizer
            grad_output if ctx.has_residual else None,
            None,  # is_recompute
            None   # do_unpad_out
        )


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Linear layer implementation

        Args:
            in_features (int): input features dim
            out_features (int): output features dim
            dtype (torch.dtype): weights data type
            device (Optional[Union[str, torch.device]]): device to use
        """
        self.quantizers = self._init_quantizers()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self._is_recompute = False
        super().__init__(
            self.in_features,
            self.out_features,
            False, # bias
            device=device,
            dtype=self.dtype)

    def set_is_recompute_state(self, v: bool):
        self._is_recompute = v

    def _init_quantizers(self) -> Dict[str, Float8BlockQuantizer]:
        """Initialize quantizers

        Returns:
            Dict[str, Float8BlockQuantizer]: initialized blockwise quantizers
        """
        keys_to_quantizer_kwargs = {
            "input": {
                      "fp8_dtype": tex.DType.kFloat8E4M3,
                      "rowwise": True,
                      "columnwise": False,
                      "block_scaling_dim": 1
                      },
            "weight": {
                       "fp8_dtype": tex.DType.kFloat8E4M3,
                       "rowwise": True,
                       "columnwise": False,
                       "block_scaling_dim": 2,
                       },
            "grad": {
                     "fp8_dtype": tex.DType.kFloat8E4M3,
                     "rowwise": True,
                     "columnwise": False,
                     "block_scaling_dim": 1
                     },
        }
        quantizers = {
            k: Float8BlockQuantizer(**keys_to_quantizer_kwargs[k])
            for k in keys_to_quantizer_kwargs.keys()
        }
        for k in keys_to_quantizer_kwargs.keys():
            quantizers[k].internal = True
        return quantizers

    def extra_repr(self):
        return f'in_features={self.in_features}, '\
               f'out_features={self.out_features}'

    def _apply_gemm(self,
                    inp: torch.Tensor | Float8BlockwiseQTensor,
                    weight: torch.Tensor,
                    residual: Optional[torch.Tensor],
                    is_recompute: bool,
                    do_unpad_out: bool):
        """Apply GEMM operation

        Args:
            inp (torch.Tensor | Float8BlockwiseQTensor): input tensor
            weight (torch.Tensor): weight tensor
            residual (Optional[torch.Tensor]): residual connection tensor
            is_recompute (bool): requantize fp8 weights on backward
            do_unpad_out (bool): do unpad gemm out if needed
        """
        input_quantizer = self.quantizers["input"]
        weight_quantizer = self.quantizers["weight"]
        grad_quantizer = self.quantizers["grad"]
        act_dtype = inp.dtype if isinstance(inp, torch.Tensor) \
                              else torch.bfloat16
        act_requires_grad = inp.requires_grad if isinstance(inp, torch.Tensor) \
                                              else True

        return FP8Gemm.apply(
            inp,
            weight,
            act_dtype,
            act_requires_grad,
            input_quantizer,
            weight_quantizer,
            grad_quantizer,
            residual,
            is_recompute,
            do_unpad_out
        )

    def forward(self, x: torch.Tensor | Float8BlockwiseQTensor,
                residual: Optional[torch.Tensor] = None,
                do_unpad_out: bool = True):
        assert self.weight is not None, "weights is not initialized."
        return self._apply_gemm(x, self.weight, residual, self._is_recompute, do_unpad_out)

    def quantize_input(
        self,
        inp: torch.Tensor
    ) -> Float8BlockwiseQTensor:
        """Quantize input outside the GEMM

        Args:
            inp (torch.Tensor): input tensor

        Returns:
            Float8BlockwiseQTensor: quantized input tensor
        """
        padded_inp = pad_tensor_if_needed(inp)
        quantizer = self.quantizers["input"]
        prev_internal = quantizer.internal
        quantizer.internal = False
        quantized_inp = quantizer(padded_inp)
        quantizer.internal = prev_internal
        quantized_inp.base_shape = inp.shape
        return quantized_inp

    @classmethod
    def swap_module(
        cls,
        module: nn.Linear
    ):
        """Create an FP8 Linear with fp8 compute from a regular nn.Linear

        Args:
            module (nn.Linear): nn.Linear module

        Returns:
            new_module (FP8Linear): FP8 Linear module
        """
        assert module.bias is None, "bias is not supported for FP8Linear"
        new_module = cls(
            module.in_features,
            module.out_features,
            module.weight.dtype,
            device=module.weight.device,
        )
        # TODO (veveselov): need to remove this attributes in the future
        for attr in ("_fused", "_is_residual", "custom_init_std", "skip_init"):
            if hasattr(module, attr):
                setattr(new_module, attr, getattr(module, attr))

        # Reuse existing weight buffer to avoid extra allocation and copy
        if module.weight.device.type != "meta":
            new_module._parameters["weight"] = module.weight
        return new_module
