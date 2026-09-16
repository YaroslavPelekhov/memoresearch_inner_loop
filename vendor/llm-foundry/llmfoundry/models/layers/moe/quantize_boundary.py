"""Activation quantization at the MoE checkpoint boundary.

`quantize_to_fp8_at_boundary(x)` row-quantizes a bf16 activation into a
`Float8BlockwiseQTensor` with an autograd-aware identity backward. It is
designed to be called just outside the activation-checkpoint region around
`block_sparse_moe`, so the tensor stashed across forward→backward is the
FP8 (uint8 data + rowwise scales) pair rather than the bf16 activation.
Inside the MoE block, `FusedDispatch` detects the already-quantized input
and takes its `isinstance(Float8BlockwiseQTensor)` branch, skipping a second
`row_quant_fn` call.
"""

import torch
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
)

from llmfoundry.models.ops.float8.triton_kernels import (
    make_float8_blockwise_qtensor_fn,
    row_quant_fn,
)


class _QuantizeToFp8AtBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Float8BlockwiseQTensor:
        assert x.dtype == torch.bfloat16, f"expected bf16, got {x.dtype}"
        assert x.is_cuda, "activation must be on CUDA"
        assert x.shape[-1] % 128 == 0, (
            f"hidden size must be 128-aligned for rowwise quant; got {x.shape[-1]}"
        )
        ctx.orig_shape = x.shape
        flat = x.contiguous().view(-1, x.shape[-1])
        data, scale = row_quant_fn(flat)
        return make_float8_blockwise_qtensor_fn(
            rowwise_data=data,
            rowwise_scale_inv=scale,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # grad_output is bf16 from FusedDispatch.backward -> buffer.combine.
        # Reshape back to the original pre-flatten shape.
        return grad_output.reshape(ctx.orig_shape)


def quantize_to_fp8_at_boundary(x: torch.Tensor) -> Float8BlockwiseQTensor:
    """Row-quantize a bf16 activation into a Float8BlockwiseQTensor.

    Call this immediately before entering an activation-checkpointed MoE
    block when `use_float8_grouped_gemm=True` and `ep_size > 1`. The returned
    tensor is what AC will stash across the forward→backward window.
    """
    return _QuantizeToFp8AtBoundary.apply(x)
