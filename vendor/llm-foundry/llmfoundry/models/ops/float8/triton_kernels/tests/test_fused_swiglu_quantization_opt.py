"""
Correctness tests for fused SwiGLU + FP8 quantization kernel.
Covers forward and backward with and without swiglu_limit clipping.

Reference chain:
    fused_swiglu_f32 (+ optional clamp) → _ref_fp8_quantize
        compared against
    fused_swiglu_quantized_fn

Run with:
    pytest llmfoundry/models/ops/float8/triton_kernels/tests/test_fused_swiglu_quantization_opt.py -v
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


# ── helpers ────────────────────────────────────────────────────────────────────

def _qtensor_parts(q: Float8BlockwiseQTensor):
    """Return (fp8_data, scale_inv) from a Float8BlockwiseQTensor."""
    return q._rowwise_data.view(torch.float8_e4m3fn), q._rowwise_scale_inv


def _dequant(q: Float8BlockwiseQTensor) -> torch.Tensor:
    fp8, scales = _qtensor_parts(q)
    return dequantize_kernel(fp8, scales)


def _ref_swiglu_f32(x: torch.Tensor, probs: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """Float32 SwiGLU with optional DeepSeek-style activation clipping.

    Matches the kernel formula exactly:
        gate = clamp(gate, max=limit)           [upper-only]
        up   = clamp(up, -limit, limit)         [symmetric]
        out  = silu(gate) * up * probs
    """
    gate, up = x.float().chunk(2, dim=-1)
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up   = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return gate * torch.sigmoid(gate) * up * probs.float().reshape(-1)[:, None]


def _ref_fp8_quantize(x_f32: torch.Tensor):
    """Power-of-2 block-wise FP8 quantisation matching the kernel's arithmetic.

    Replicates _fp8_scale_inv_from_amax:
        e         = ceil(log2(max(amax/448, 1e-30)))
        scale     = 2^e   (dequant scale stored as scale_inv)
        inv_scale = 2^-e  (used to quantise)
    """
    FP8_MAX = 448.0
    M, K = x_f32.shape
    bpr = K // 128
    x_b = x_f32.reshape(M, bpr, 128)
    amax = x_b.abs().amax(dim=-1)
    scale_pre = amax.clamp(min=1e-30) / FP8_MAX
    e = torch.ceil(torch.log2(scale_pre))
    scale     = 2.0 ** e
    inv_scale = 2.0 ** (-e)
    fp8 = (x_b * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return fp8.reshape(M, K).to(torch.float8_e4m3fn), scale


# ── parametrisation ────────────────────────────────────────────────────────────

SHAPES = [
    (128,  128),
    (256,  256),
    (512,  512),
    (1024, 512),
    (2048, 1024),
    # Production hid_dim values — must hit the static tuning tables.
    (128,  1280),
    (128,  1536),
    (128,  2048),
    (128,  4096),
]

# 0.0 → no clamp (branch compiled away)
# 8.0 → DeepSeek-style aggressive clamp
# 448.0 → FP8_MAX, clips only values that would saturate fp8 anyway
SWIGLU_LIMITS = [0.0, 8.0, 448.0]


# ── forward: kernel vs f32 reference ──────────────────────────────────────────

@pytest.mark.parametrize("total_tokens,hid_dim", SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_fwd_fp8_data(total_tokens: int, hid_dim: int, swiglu_limit: float):
    """FP8 output must match the f32 reference pipeline.

    The kernel computes amax * constexpr(1/448) while _ref_fp8_quantize uses
    amax / 448; IEEE 754 does not guarantee a*(1/b)==a/b, so we allow 1 ULP
    (atol=0 on bf16 cast is strict; if it flickers on the boundary use atol=1).
    """
    torch.manual_seed(42)
    x     = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_k    = fused_swiglu_quantized_fn(x, probs, swiglu_limit=swiglu_limit)
    fp8_k, _ = _qtensor_parts(out_k)

    ref_f32    = _ref_swiglu_f32(x, probs, swiglu_limit).contiguous()
    ref_fp8, _ = _ref_fp8_quantize(ref_f32)

    torch.testing.assert_close(
        fp8_k.to(torch.bfloat16), ref_fp8.to(torch.bfloat16), atol=0, rtol=0,
    )


@pytest.mark.parametrize("total_tokens,hid_dim", SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_fwd_dequantized(total_tokens: int, hid_dim: int, swiglu_limit: float):
    """Dequantised output must be numerically close to f32 reference.

    FP8 E4M3 step for values in [8, 16) is 1.0, so max abs error ≤ 0.5.
    """
    torch.manual_seed(42)
    x     = torch.randn(total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16)

    out_k   = fused_swiglu_quantized_fn(x, probs, swiglu_limit=swiglu_limit)
    deq_k   = _dequant(out_k)
    ref_f32 = _ref_swiglu_f32(x, probs, swiglu_limit)

    torch.testing.assert_close(deq_k.float(), ref_f32, atol=0.5, rtol=0.15)


# ── backward: kernel vs f32 autograd reference ────────────────────────────────

@pytest.mark.parametrize("total_tokens,hid_dim", SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_bwd_vs_ref(total_tokens: int, hid_dim: int, swiglu_limit: float):
    """Dequantised grad_input and grad_probs must match f32 autograd reference.

    The f32 reference uses PyTorch autograd through clamp+chunk+silu, which
    provides an independent gradient path.  Tolerance is loose (atol=0.5)
    because fp8 quantisation of the gradient introduces a dequant error of
    up to 0.5 for values in [8, 16).
    """
    torch.manual_seed(7)
    x = torch.randn(
        total_tokens, 2 * hid_dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    probs = torch.rand(total_tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    # ── kernel backward ──
    out_k = fused_swiglu_quantized_fn(x, probs, swiglu_limit=swiglu_limit)
    torch.autograd.backward([out_k], [grad_out])
    grad_x_deq = _dequant(x.grad)
    grad_p_k   = probs.grad.clone()
    x.grad     = None
    probs.grad = None

    # ── f32 autograd reference ──
    x_f32 = x.detach().float().requires_grad_(True)
    p_f32 = probs.detach().float().requires_grad_(True)
    gate, up = x_f32.chunk(2, dim=-1)
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up   = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    swiglu_f32 = gate * torch.sigmoid(gate) * up * p_f32[:, None]
    swiglu_f32.backward(grad_out.float())

    torch.testing.assert_close(
        grad_x_deq.float(), x_f32.grad,
        atol=0.5, rtol=0.15,
        msg="dequantised grad_input must be close to f32 autograd",
    )
    torch.testing.assert_close(
        grad_p_k.float(), p_f32.grad,
        atol=5e-2, rtol=2e-2,
        msg="grad_probs must be close to f32 autograd",
    )


# ── clamp-boundary gradient correctness ───────────────────────────────────────

def test_clamp_boundary_gradients():
    """Verify clamp-boundary gradient semantics with a controlled input.

    Construction:
      - Rows [:64]:  gate and up exceed the clamp limit → grad_gate, grad_up must be ~0
      - Rows [64:]:  gate and up within range           → grad_gate, grad_up must flow

    grad_probs must be non-zero on all rows: probs scales the clamped output,
    so d/d_probs is non-zero regardless of whether activations were clamped.
    """
    limit        = 4.0
    total_tokens = 128   # divisible by FWD_ROW_BLOCK=64 and BWD_ROW_BLOCK=32
    hid_dim      = 128
    device       = "cuda"

    x = torch.zeros(total_tokens, 2 * hid_dim, device=device, dtype=torch.bfloat16)
    x[:64, :hid_dim] = limit * 2    # gate saturated
    x[:64, hid_dim:] = limit * 2    # up saturated
    x[64:, :hid_dim] = limit * 0.5  # gate in range
    x[64:, hid_dim:] = limit * 0.5  # up in range

    probs    = torch.ones(total_tokens, device=device, dtype=torch.bfloat16)
    grad_out = torch.ones(total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    x.requires_grad_(True)
    probs.requires_grad_(True)

    out = fused_swiglu_quantized_fn(x, probs, swiglu_limit=limit)
    torch.autograd.backward([out], [grad_out])

    grad_x_deq    = _dequant(x.grad)
    grad_gate_deq = grad_x_deq[:, :hid_dim]
    grad_up_deq   = grad_x_deq[:, hid_dim:]
    grad_p        = probs.grad

    assert grad_gate_deq[:64].abs().max().item() < 1e-2, \
        "grad_gate must be ~0 where gate was clamped"
    assert grad_up_deq[:64].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped"
    assert grad_gate_deq[64:].abs().max().item() > 1e-3, \
        "grad_gate must be non-zero where gate was not clamped"
    assert grad_up_deq[64:].abs().max().item() > 1e-3, \
        "grad_up must be non-zero where up was not clamped"
    assert grad_p.abs().max().item() > 1e-3, \
        "grad_probs must be non-zero even when activations are clamped"


# ── clip_counts monitoring correctness ────────────────────────────────────────

def test_clip_counts():
    """clip_counts must accumulate the exact number of clamped elements.

    Construction (total_tokens=128, hid_dim=128, limit=4.0):
      - Rows   0:32  gate = limit*2  → 32*128 = 4096 gate clips
      - Rows  32:64  up  = +limit*2  → 32*128 = 4096 up clips (positive side)
      - Rows  64:96  up  = -limit*2  → 32*128 = 4096 up clips (negative side)
      - Rows  96:128 all zeros       → 0 clips

    Expected: clip_counts[0] = 4096, clip_counts[1] = 8192.
    """
    limit        = 4.0
    total_tokens = 128
    hid_dim      = 128
    device       = "cuda"

    x = torch.zeros(total_tokens, 2 * hid_dim, device=device, dtype=torch.bfloat16)
    x[:32,  :hid_dim]  = limit * 2   # gate saturated (rows 0:32)
    x[32:64, hid_dim:] = limit * 2   # up saturated positive (rows 32:64)
    x[64:96, hid_dim:] = -limit * 2  # up saturated negative (rows 64:96)

    probs       = torch.ones(total_tokens, device=device, dtype=torch.bfloat16)
    clip_counts = torch.zeros(2, dtype=torch.int32, device=device)

    fused_swiglu_quantized_fn(x, probs, swiglu_limit=limit, clip_counts=clip_counts)

    expected_gate = 32 * hid_dim   # 4096
    expected_up   = 64 * hid_dim   # 8192

    assert clip_counts[0].item() == expected_gate, \
        f"gate clip count: expected {expected_gate}, got {clip_counts[0].item()}"
    assert clip_counts[1].item() == expected_up, \
        f"up clip count: expected {expected_up}, got {clip_counts[1].item()}"


def test_clip_counts_disabled():
    """clip_counts must remain zero when monitoring is off.

    Two cases:
      (a) swiglu_limit=0.0 — ACTIVATION_MONITOR_ON compiled away even if buffer provided
      (b) clip_counts=None  — monitoring disabled regardless of swiglu_limit
    """
    total_tokens = 128
    hid_dim      = 128
    device       = "cuda"

    x = torch.randn(total_tokens, 2 * hid_dim, device=device, dtype=torch.bfloat16) * 100
    probs = torch.ones(total_tokens, device=device, dtype=torch.bfloat16)

    # (a) limit=0 → monitor_on=False, counts must stay at zero
    clip_counts = torch.zeros(2, dtype=torch.int32, device=device)
    fused_swiglu_quantized_fn(x, probs, swiglu_limit=0.0, clip_counts=clip_counts)
    assert clip_counts[0].item() == 0 and clip_counts[1].item() == 0, \
        "counts must stay zero when swiglu_limit=0.0"

    # (b) clip_counts=None → no error, monitoring simply skipped
    fused_swiglu_quantized_fn(x, probs, swiglu_limit=8.0, clip_counts=None)


# ── cross-saturation gradient correctness ─────────────────────────────────────

def test_clamp_cross_saturation():
    """Verify that gate and up clamp masks are independent.

    Row groups (32 rows each, total=128 divisible by FWD_ROW_BLOCK=64, BWD_ROW_BLOCK=32):
      - Rows   0:32  gate saturated, up saturated   → grad_gate~0, grad_up~0
      - Rows  32:64  gate saturated, up in range    → grad_gate~0, grad_up nonzero
      - Rows  64:96  gate in range,  up saturated   → grad_gate nonzero, grad_up~0
      - Rows  96:128 both in range                  → grad_gate nonzero, grad_up nonzero

    grad_probs must be nonzero on all rows because d(out)/d(probs) = silu(gate_c)*up_c,
    which is nonzero even for clamped values unless the clamped output itself is zero.
    """
    limit        = 4.0
    total_tokens = 128
    hid_dim      = 128
    device       = "cuda"

    x = torch.zeros(total_tokens, 2 * hid_dim, device=device, dtype=torch.bfloat16)
    # gate column (first half)
    x[:64,  :hid_dim] = limit * 2    # groups 0 and 1: gate saturated
    x[64:,  :hid_dim] = limit * 0.5  # groups 2 and 3: gate in range
    # up column (second half)
    x[:32,  hid_dim:] = limit * 2    # group 0: up saturated
    x[64:96, hid_dim:] = limit * 2   # group 2: up saturated
    x[32:64, hid_dim:] = limit * 0.5 # group 1: up in range
    x[96:,  hid_dim:] = limit * 0.5  # group 3: up in range

    probs    = torch.ones(total_tokens, device=device, dtype=torch.bfloat16)
    grad_out = torch.ones(total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    x.requires_grad_(True)
    probs.requires_grad_(True)

    out = fused_swiglu_quantized_fn(x, probs, swiglu_limit=limit)
    torch.autograd.backward([out], [grad_out])

    grad_x_deq    = _dequant(x.grad)
    grad_gate_deq = grad_x_deq[:, :hid_dim]
    grad_up_deq   = grad_x_deq[:, hid_dim:]

    # gate clamped in rows 0:64
    assert grad_gate_deq[:64].abs().max().item() < 1e-2, \
        "grad_gate must be ~0 where gate was clamped (rows 0:64)"
    # gate not clamped in rows 64:128
    assert grad_gate_deq[64:].abs().max().item() > 1e-3, \
        "grad_gate must be nonzero where gate was not clamped (rows 64:128)"

    # up clamped in rows 0:32 and 64:96
    assert grad_up_deq[:32].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped (rows 0:32)"
    assert grad_up_deq[64:96].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped (rows 64:96)"
    # up not clamped in rows 32:64 and 96:128
    assert grad_up_deq[32:64].abs().max().item() > 1e-3, \
        "grad_up must be nonzero where up was not clamped (rows 32:64)"
    assert grad_up_deq[96:].abs().max().item() > 1e-3, \
        "grad_up must be nonzero where up was not clamped (rows 96:128)"

    # grad_probs: nonzero everywhere (output of clamped rows is nonzero)
    assert probs.grad.abs().max().item() > 1e-3, \
        "grad_probs must be nonzero"
