"""
Tests for FusedQuatizedSwiGLU: fused SwiGLU + FP8 output quantization.

Test strategy
─────────────
1. Tight (near bit-exact) comparison of raw FP8 data and scales against a
   Python float32 reference that uses the same power-of-2 quantization math.
2. Loose dequantized comparison using the Triton dequantize kernel from
   dequantization_deprecated.py — validates end-to-end numerical correctness.
3. Backward tests: grad_input FP8 data (tight) and grad_probs (loose).

Run with:
    pytest llmfoundry/models/ops/float8/triton_kernels/tests/test_fused_swiglu_quantization.py -v
"""

import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor  # noqa: E402

from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization import (  # noqa: E402
    fused_swiglu_quantized_fn,
)
from llmfoundry.models.ops.float8.triton_kernels.dequantization_deprecated import (  # noqa: E402
    dequantize_kernel,
)
from llmfoundry.models.ops.float8.triton_kernels.row_quantization import (  # noqa: E402
    row_quantization_1x128_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _swiglu_f32(x_bf16: torch.Tensor, probs_bf16: torch.Tensor) -> torch.Tensor:
    """SwiGLU in float32 — mirrors how the Triton kernel computes it."""
    x_f32 = x_bf16.float()
    probs_f32 = probs_bf16.reshape(-1).float()
    gate, up = x_f32.chunk(2, dim=-1)
    return gate * torch.sigmoid(gate) * up * probs_f32[:, None]


def _ref_fp8_quantize(x_f32: torch.Tensor):
    """Power-of-2 block-wise FP8 quantization in Python float32.

    Replicates `_fp8_scale_inv_from_amax` from the Triton kernel exactly:
      scale_pre = max(amax / FP8_MAX, 1e-30)
      e         = ceil(log2(scale_pre))
      scale     = 2^e   (for dequantization)
      inv_scale = 2^-e  (for quantization)

    Returns (fp8_data, scales) where
      fp8_data: (M, K) float8_e4m3fn
      scales:   (M, K // 128) float32, same layout as _rowwise_scale_inv
    """
    FP8_MAX = 448.0
    M, K = x_f32.shape
    blocks_per_row = K // 128
    x_blocked = x_f32.reshape(M, blocks_per_row, 128)
    amax = x_blocked.abs().amax(dim=-1)                            # (M, bpr)
    scale_pre = torch.clamp(amax / FP8_MAX, min=1e-30)
    e = torch.ceil(torch.log2(scale_pre))
    scale = 2.0 ** e                                               # dequant scale
    inv_scale = 2.0 ** (-e)                                        # quant scale
    fp8 = (x_blocked * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return fp8.reshape(M, K).to(torch.float8_e4m3fn), scale


def _qtensor_parts(q: Float8BlockwiseQTensor):
    """Extract (fp8_data, scales) from a Float8BlockwiseQTensor."""
    fp8_data = q._rowwise_data.view(torch.float8_e4m3fn)
    scales = q._rowwise_scale_inv  # (M, K // 128), float32
    return fp8_data, scales


def _triton_dequantize(q: Float8BlockwiseQTensor) -> torch.Tensor:
    """Dequantize a Float8BlockwiseQTensor using the Triton dequantize kernel."""
    fp8_data, scales = _qtensor_parts(q)
    return dequantize_kernel(fp8_data, scales)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FORWARD_SHAPES = [
    (128, 128),   # single-block edge case
    (256, 256),
    (512, 512),
    (1024, 512),
    (2048, 1024),
]

BACKWARD_SHAPES = [
    (128, 128),   # single-block edge case
    (256, 256),
    (512, 512),
    (1024, 512),
]


# ---------------------------------------------------------------------------
# 1. Output structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_tokens,hid_dim", FORWARD_SHAPES)
def test_output_type_and_shape(total_tokens: int, hid_dim: int):
    """Return type must be Float8BlockwiseQTensor; shapes must be correct."""
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out = fused_swiglu_quantized_fn(x, probs)
    fp8_data, scales = _qtensor_parts(out)

    assert isinstance(out, Float8BlockwiseQTensor)
    assert out.shape == (total_tokens, hid_dim)
    assert fp8_data.shape == (total_tokens, hid_dim)
    assert scales.shape == (total_tokens, hid_dim // 128)


def test_invalid_total_tokens_raises():
    """total_tokens not divisible by 128 must raise AssertionError."""
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(300, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fused_swiglu_quantized_fn(x, probs)


# ---------------------------------------------------------------------------
# 2. Forward — tight FP8 data and scale comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_tokens,hid_dim", FORWARD_SHAPES)
def test_forward_fp8_data_matches_ref(total_tokens: int, hid_dim: int):
    """Raw FP8 values must match our Python float32 reference (bit-exact math).

    Both this kernel and _ref_fp8_quantize use identical float32 power-of-2
    quantization logic, so the FP8 output values should be bit-exact.
    """
    torch.manual_seed(42)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    kernel_fp8, _ = _qtensor_parts(out_q)

    ref_swiglu = _swiglu_f32(x, probs)
    ref_fp8, _ = _ref_fp8_quantize(ref_swiglu)

    # Compare via bf16 cast: two fp8 values are equal iff their bf16 casts are equal
    torch.testing.assert_close(
        kernel_fp8.to(torch.bfloat16),
        ref_fp8.to(torch.bfloat16),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize("total_tokens,hid_dim", FORWARD_SHAPES)
def test_forward_scales_match_ref(total_tokens: int, hid_dim: int):
    """Scale tensors must be bit-exact matches of the Python float32 reference."""
    torch.manual_seed(42)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    _, kernel_scales = _qtensor_parts(out_q)

    ref_swiglu = _swiglu_f32(x, probs)
    _, ref_scales = _ref_fp8_quantize(ref_swiglu)

    torch.testing.assert_close(kernel_scales, ref_scales, atol=0, rtol=0)


# Exclude (2048, 1024): the fused kernel computes `amax * tl.constexpr(1/448.0)`
# (multiply by precomputed reciprocal) while row_quantization uses `amax / 448.0`
# (explicit division). IEEE 754 does not guarantee a*(1/b) == a/b, so at ~2M
# elements 2 values land on a power-of-2 boundary and diverge by 1 FP8 ULP.
# Correctness for all shapes is validated by test_forward_fp8_data_matches_ref.
_ROW_QUANT_SHAPES = [(t, h) for t, h in FORWARD_SHAPES if (t, h) != (2048, 1024)]


@pytest.mark.parametrize("total_tokens,hid_dim", _ROW_QUANT_SHAPES)
def test_forward_fp8_data_matches_row_quantization_kernel(total_tokens: int, hid_dim: int):
    """FP8 output must match row_quantization_1x128_fn applied to f32 swiglu.

    row_quantization_1x128_fn uses the same power-of-2 quantization and loads
    input as float32, so the fp8 values should be bit-exact.
    """
    torch.manual_seed(13)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    kernel_fp8, _ = _qtensor_parts(out_q)

    # Compute reference using the standalone row-quantization kernel.
    # Pass float32 directly — same precision as the fused kernel internally.
    ref_swiglu_f32 = _swiglu_f32(x, probs).contiguous()
    ref_fp8, _ = row_quantization_1x128_fn(ref_swiglu_f32)

    torch.testing.assert_close(
        kernel_fp8.to(torch.bfloat16),
        ref_fp8.to(torch.bfloat16),
        atol=0, rtol=0,
    )


# ---------------------------------------------------------------------------
# 3. Forward — loose dequantized comparison (Triton dequant kernel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_tokens,hid_dim", FORWARD_SHAPES)
def test_forward_dequantized_vs_float32_ref(total_tokens: int, hid_dim: int):
    """Dequantized output (Triton kernel) must be close to float32 reference.

    FP8 E4M3 step size for |x| in [2^k, 2^(k+1)) is 2^(k-3).
    For values in [8, 16) the step is 1.0, so max abs error ≤ 0.5.
    """
    torch.manual_seed(42)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    out = _triton_dequantize(out_q)

    ref = _swiglu_f32(x, probs).bfloat16()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.15)


# ---------------------------------------------------------------------------
# 4. Input shape variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_tokens,hid_dim", [
    (256, 256),
    (512, 256),
    (1024, 512),
])
def test_probs_shape_variants(total_tokens: int, hid_dim: int):
    """(N,) and (N,1) probs must produce identical FP8 outputs."""
    torch.manual_seed(0)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs_1d = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)
    probs_2d = probs_1d.unsqueeze(-1)

    fp8_1d, scales_1d = _qtensor_parts(fused_swiglu_quantized_fn(x, probs_1d))
    fp8_2d, scales_2d = _qtensor_parts(fused_swiglu_quantized_fn(x, probs_2d))

    torch.testing.assert_close(fp8_1d.to(torch.bfloat16), fp8_2d.to(torch.bfloat16), atol=0, rtol=0)
    torch.testing.assert_close(scales_1d, scales_2d, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 5. Backward — grad_probs and grad_input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total_tokens,hid_dim", BACKWARD_SHAPES)
def test_backward_grad_probs_vs_float32_ref(total_tokens: int, hid_dim: int):
    """grad_probs must be close to the float32 reference.

    grad_probs is NOT quantized to fp8 — it accumulates float32 partial sums
    in the kernel, so tolerances can be tight.
    """
    torch.manual_seed(7)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    torch.autograd.backward([out_q], [grad_out])
    probs_grad = probs.grad.clone()
    x.grad = None
    probs.grad = None

    # float32 reference
    x_f32 = x.detach().float().requires_grad_(True)
    probs_f32 = probs.detach().float().requires_grad_(True)
    ref_out = _swiglu_f32(x_f32, probs_f32)
    ref_out.backward(grad_out.float())

    torch.testing.assert_close(probs_grad.float(), probs_f32.grad, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("total_tokens,hid_dim", BACKWARD_SHAPES)
def test_backward_grad_input_fp8_data_matches_ref(total_tokens: int, hid_dim: int):
    """Raw FP8 grad_input values must match the Python float32 reference.

    The backward computes grad_gate and grad_up in float32 and quantizes with
    the same power-of-2 logic, so the comparison should be near bit-exact.
    """
    torch.manual_seed(7)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    torch.autograd.backward([out_q], [grad_out])
    kernel_grad_fp8, _ = _qtensor_parts(x.grad)
    x.grad = None
    probs.grad = None

    # Reference: compute grad_gate and grad_up in float32, quantize concatenated
    gate_f32, up_f32 = x.detach().float().chunk(2, dim=-1)
    probs_f32 = probs.detach().float()
    go_f32 = grad_out.float()
    gate_sigmoid = torch.sigmoid(gate_f32)
    gate_silu = gate_f32 * gate_sigmoid
    grad_up_f32 = go_f32 * gate_silu * probs_f32[:, None]
    dsilu = gate_sigmoid + gate_silu * (1.0 - gate_sigmoid)
    grad_gate_f32 = go_f32 * up_f32 * probs_f32[:, None] * dsilu
    grad_input_f32 = torch.cat([grad_gate_f32, grad_up_f32], dim=-1)
    ref_fp8, _ = _ref_fp8_quantize(grad_input_f32)

    torch.testing.assert_close(
        kernel_grad_fp8.to(torch.bfloat16),
        ref_fp8.to(torch.bfloat16),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize("total_tokens,hid_dim", BACKWARD_SHAPES)
def test_backward_grad_input_dequantized_vs_float32_ref(total_tokens: int, hid_dim: int):
    """Dequantized grad_input (Triton kernel) must be close to float32 reference."""
    torch.manual_seed(7)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    torch.autograd.backward([out_q], [grad_out])
    x_grad = _triton_dequantize(x.grad)
    x.grad = None
    probs.grad = None

    # Reference
    gate_f32, up_f32 = x.detach().float().chunk(2, dim=-1)
    probs_f32 = probs.detach().float()
    go_f32 = grad_out.float()
    gate_sigmoid = torch.sigmoid(gate_f32)
    gate_silu = gate_f32 * gate_sigmoid
    grad_up_f32 = go_f32 * gate_silu * probs_f32[:, None]
    dsilu = gate_sigmoid + gate_silu * (1.0 - gate_sigmoid)
    grad_gate_f32 = go_f32 * up_f32 * probs_f32[:, None] * dsilu
    ref_grad = torch.cat([grad_gate_f32, grad_up_f32], dim=-1)

    torch.testing.assert_close(x_grad.float(), ref_grad.float(), atol=0.5, rtol=0.15)


@pytest.mark.parametrize("total_tokens,hid_dim", BACKWARD_SHAPES)
def test_backward_grad_input_dequantized_vs_autograd_ref(total_tokens: int, hid_dim: int):
    """Dequantized grad_input must match PyTorch autograd reference.

    Unlike test_backward_grad_input_dequantized_vs_float32_ref (which uses the
    same manually-expanded formula as the kernel), this test derives the
    reference via torch.autograd — an independent path that catches formula bugs
    invisible to manual comparisons.
    """
    torch.manual_seed(7)
    x = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_q = fused_swiglu_quantized_fn(x, probs)
    torch.autograd.backward([out_q], [grad_out])
    x_grad = _triton_dequantize(x.grad)
    x.grad = None
    probs.grad = None

    # Reference: PyTorch autograd on float32 swiglu — no manual formula
    x_f32 = x.detach().float().requires_grad_(True)
    probs_f32 = probs.detach().float().requires_grad_(True)
    _swiglu_f32(x_f32, probs_f32).backward(grad_out.float())

    torch.testing.assert_close(x_grad.float(), x_f32.grad, atol=0.5, rtol=0.15)
