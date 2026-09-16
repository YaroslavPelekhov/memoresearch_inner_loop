import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex

from llmfoundry.models.ops.float8.triton_kernels.block_transpose_fused_quantization import (
    block_transpose_fused_quantization_128x128_fn
)


@pytest.mark.parametrize("num_groups, int_dim, hid_dim", [  # pyright: ignore[reportUntypedFunctionDecorator]
    (4, 1280, 1536),
    (1, 128, 128),
    (2, 256, 512),
    (8, 640, 768),
])
def test_block_quantization(num_groups: int, int_dim: int, hid_dim: int) -> None:
    device = torch.cuda.current_device()

    weights = torch.randn(
        (num_groups, int_dim, hid_dim),
        device=device,
        dtype=torch.bfloat16)

    kwargs = {
        "fp8_dtype": tex.DType.kFloat8E4M3,
        "rowwise": True,
        "columnwise": True,
        "block_scaling_dim": 2
    }
    weights_quantizer_te = Float8BlockQuantizer(**kwargs)

    for group in range(num_groups):
        wq_te = weights_quantizer_te.quantize(weights[group])
        wq_te_bfp16 = wq_te._columnwise_data.view(torch.float8_e4m3fn).to(torch.bfloat16)

        wq_triton = block_transpose_fused_quantization_128x128_fn(weights[group].unsqueeze(0))
        wq_triton_bfp16 = wq_triton[0].view(torch.float8_e4m3fn).to(torch.bfloat16)

        assert torch.allclose(wq_te_bfp16, wq_triton_bfp16), (
            f"Mismatch at group {group}: "
            f"max diff = {(wq_te_bfp16 - wq_triton_bfp16).abs().max().item()}"
        )
