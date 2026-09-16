import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex

from llmfoundry.models.ops.float8.triton_kernels.row_quantization import row_quantization_1x128_fn


@pytest.mark.parametrize("hid_dim", [  # pyright: ignore[reportUntypedFunctionDecorator]
    1280, 128, 256, 640,
])
def test_row_quantization(hid_dim: int) -> None:
    device = torch.cuda.current_device()

    token_group_sizes = [128*17, 128*5, 128*9, 128]

    tensor_groups = torch.randn(
        (sum(token_group_sizes), hid_dim),
        device=device,
        dtype=torch.bfloat16)

    kwargs = {
        "fp8_dtype": tex.DType.kFloat8E4M3,
        "rowwise": True,
        "columnwise": False,
        "block_scaling_dim": 1
    }
    tensor_group_row_quantizer_te = Float8BlockQuantizer(**kwargs)

    tq_triton = row_quantization_1x128_fn(tensor_groups)
    tq_triton_bfp16 = tq_triton[0].to(torch.bfloat16)

    tq_te = tensor_group_row_quantizer_te.quantize(tensor_groups)
    tq_te = tq_te._rowwise_data.view(torch.float8_e4m3fn)
    tq_te_bfp16 = tq_te.to(torch.bfloat16)

    assert torch.allclose(tq_te_bfp16, tq_triton_bfp16)
