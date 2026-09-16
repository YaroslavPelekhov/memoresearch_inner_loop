import random
import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex

from llmfoundry.models.ops.float8.triton_kernels.row2col_deepgemm_quantization import (
    row2col_requantization_deepgemm_layout_fn
)
from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices

BLOCK = 128


# writed by opus
def _build_reference(
    row_fp8: torch.Tensor,
    row_scale_inv: torch.Tensor,
    group_sizes_list: list[int],
    hid_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ PyTorch reference: dequant row-wise, requant column-wise
        per 128-row block, transpose, per-group layout.
    """
    total_tokens = row_fp8.shape[0]
    device = row_fp8.device
    n_row_blocks = total_tokens // BLOCK
    n_col_blocks = hid_dim // BLOCK

    ref_col_scale = torch.empty(n_row_blocks, hid_dim, device=device, dtype=torch.float32)
    transposed_blocks: dict[tuple[int, int], torch.Tensor] = {}

    for rb in range(n_row_blocks):
        for cb in range(n_col_blocks):
            r_s, r_e = rb * BLOCK, (rb + 1) * BLOCK
            c_s, c_e = cb * BLOCK, (cb + 1) * BLOCK

            block_f32 = row_fp8[r_s:r_e, c_s:c_e].to(torch.float32) * row_scale_inv[r_s:r_e, cb:cb + 1]

            col_max = block_f32.abs().max(dim=0).values / 448.0
            col_max = col_max.clamp(min=1e-30)
            col_scale = torch.exp2(torch.ceil(torch.log2(col_max)))

            quantized = (block_f32 / col_scale[None, :]).to(torch.float8_e4m3fn)
            transposed_blocks[(rb, cb)] = quantized.T

            ref_col_scale[rb, c_s:c_e] = col_scale

    ref_parts: list[torch.Tensor] = []
    token_offset = 0
    for gs in group_sizes_list:
        group_out = torch.empty(hid_dim, gs, device=device, dtype=torch.float8_e4m3fn)
        first_rb = token_offset // BLOCK
        for local_rb in range(gs // BLOCK):
            rb = first_rb + local_rb
            for cb in range(n_col_blocks):
                c_s, c_e = cb * BLOCK, (cb + 1) * BLOCK
                t_s, t_e = local_rb * BLOCK, (local_rb + 1) * BLOCK
                group_out[c_s:c_e, t_s:t_e] = transposed_blocks[(rb, cb)]
        ref_parts.append(group_out.contiguous().view(-1))
        token_offset += gs

    return torch.cat(ref_parts), ref_col_scale


def _run_row2col_test(
    hid_dim: int,
    group_sizes_list: list[int],
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.cuda.current_device()
    group_sizes_tensor = torch.tensor(group_sizes_list, device=device, dtype=torch.int32)
    total_tokens = sum(group_sizes_list)

    m_indices = build_m_indices(group_sizes_list, group_sizes_tensor)
    tensor_groups = torch.randn(
        (total_tokens, hid_dim),
        device=device,
        dtype=torch.bfloat16)

    row_kwargs = {
        "fp8_dtype": tex.DType.kFloat8E4M3,
        "rowwise": True,
        "columnwise": False,
        "block_scaling_dim": 1,
    }
    row_quantizer = Float8BlockQuantizer(**row_kwargs)
    tq_te = row_quantizer.quantize(tensor_groups)

    row_data = tq_te._rowwise_data
    row_scale_inv = tq_te._rowwise_scale_inv.mT.contiguous()

    triton_col_data, triton_col_scale = row2col_requantization_deepgemm_layout_fn(
        tensor=row_data,
        m_indices=m_indices,
        group_sizes=group_sizes_tensor,
        scales_inv=row_scale_inv)

    row_fp8 = row_data.view(torch.float8_e4m3fn)
    ref_col_data, ref_col_scale = _build_reference(
        row_fp8, row_scale_inv, group_sizes_list, hid_dim)

    triton_fp8 = triton_col_data.view(torch.float8_e4m3fn)

    diff = (ref_col_data.to(torch.float32) - triton_fp8.to(torch.float32)).abs().max().item()
    assert torch.allclose(
        ref_col_data.to(torch.bfloat16),
        triton_fp8.to(torch.bfloat16),
    ), f"Data mismatch: max diff = {diff}"

    diff = (ref_col_scale - triton_col_scale).abs().max().item()
    assert torch.allclose(ref_col_scale, triton_col_scale), (
        f"Scale mismatch: max diff = {diff}"
    )


@pytest.mark.parametrize(  # pyright: ignore[reportUntypedFunctionDecorator]
    "hid_dim, group_sizes_list",
    [
        (128, [128, 128]),
        (256, [128 * 4, 128 * 2]),
        (512, [128 * 17, 128 * 5, 128 * 9, 128]),
        (768, [128 * 8, 128 * 12]),
        (1280, [128 * 17, 128 * 5, 128 * 9, 128]),
        (1536, [128 * 6, 128 * 10, 128 * 4]),
    ],
    ids=["hid128-g2", "hid256-g2", "hid512-g4", "hid768-g2", "hid1280-g4", "hid1536-g3"],
)
def test_row2col_requantization_deepgemm_layout_fn(
    hid_dim: int,
    group_sizes_list: list[int],
) -> None:
    _run_row2col_test(hid_dim, group_sizes_list)


def test_row2col_requantization_random_group_sizes() -> None:
    """Random group sizes (128-aligned), multiple seeds."""
    for seed in range(5):
        num_groups = random.randint(2, 8)
        group_sizes_list = [128 * random.randint(1, 20) for _ in range(num_groups)]
        hid_dim = 128 * random.choice([2, 4, 6, 8, 10, 12])
        _run_row2col_test(hid_dim, group_sizes_list, seed=seed)
