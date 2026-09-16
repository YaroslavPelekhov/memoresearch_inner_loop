"""
Correctness tests for dense fused SwiGLU + FP8 quantization kernel.
Covers forward and backward with and without swiglu_limit clipping.

Reference chain:
    _ref_swiglu_f32(gate, up, swiglu_limit) → _ref_fp8_quantize
        compared against
    SepProductSiluQuant.apply(gate, up, ..., swiglu_limit)

Run with:
    pytest llmfoundry/models/ops/float8/triton_kernels/tests/test_fused_swiglu_quantization_dense_opt.py -v
"""

import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor  # noqa: E402

from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization_dense import (  # noqa: E402
    SepProductSiluQuant,
)
from llmfoundry.models.ops.float8.triton_kernels.dequantization_deprecated import (  # noqa: E402
    dequantize_kernel,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _qtensor_to_fp8_2d(q: Float8BlockwiseQTensor) -> torch.Tensor:
    """Return fp8 data as a 2D (n_rows, h_dim) tensor."""
    return q._rowwise_data.view(torch.float8_e4m3fn).reshape(-1, q.shape[-1])


def _dequant_2d(q: Float8BlockwiseQTensor) -> torch.Tensor:
    """Dequantize a Float8BlockwiseQTensor, returning a 2D (n_rows, h_dim) tensor.

    Scale is stored in transposed layout (blocks_per_row_mult4, n_rows) by the kernel.
    dequantize_kernel expects (n_rows, blocks_per_row), so transpose + slice.
    """
    h_dim  = q.shape[-1]
    bpr    = h_dim // 128
    fp8    = _qtensor_to_fp8_2d(q)              # (n_rows, h_dim)
    # scale_inv: (bpr_mult4, n_rows) → T → (n_rows, bpr_mult4) → [:, :bpr]
    scale  = q._rowwise_scale_inv.T[:, :bpr].contiguous()  # (n_rows, bpr)
    return dequantize_kernel(fp8, scale)


def _ref_swiglu_f32(gate: torch.Tensor, up: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """Float32 SwiGLU with optional DeepSeek-style activation clipping.

    Matches the kernel formula exactly:
        gate = clamp(gate, max=limit)          [upper-only]
        up   = clamp(up, -limit, limit)        [symmetric]
        out  = silu(gate) * up
    Returns a 2D (n_rows, h_dim) tensor.
    """
    h_dim = gate.shape[-1]
    g = gate.float().reshape(-1, h_dim)
    u = up.float().reshape(-1, h_dim)
    if swiglu_limit > 0:
        g = g.clamp(max=swiglu_limit)
        u = u.clamp(min=-swiglu_limit, max=swiglu_limit)
    return g * torch.sigmoid(g) * u


def _ref_fp8_quantize(x_f32: torch.Tensor):
    """Power-of-2 block-wise FP8 quantisation matching the kernel's arithmetic.

    Replicates _fp8_scale_inv_from_amax with POWER_TWO_MAX_ROUND=True:
        e         = ceil(log2(max(amax/448, 1e-30)))
        scale     = 2^e   (dequant scale stored as scale_inv)
        inv_scale = 2^-e  (used to quantise)
    Input must be 2D (M, K).
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

# (total_tokens, hid_dim) — covers small shapes + all static tuning table entries
SHAPES = [
    (128,   128),
    (256,   256),
    (512,   512),
    (1024,  512),
    (2048, 1024),
    # Production hid_dim values — must hit the static tuning tables.
    (128,  1280),
    (128,  1536),
    (128,  2048),
    (128,  8960),
    (128, 11008),
    (128, 18432),
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
    gate = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)
    up   = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_k = SepProductSiluQuant.apply(gate, up,
                                      torch.float8_e4m3fn, True,
                                      swiglu_limit, None)
    fp8_k = _qtensor_to_fp8_2d(out_k)  # (total_tokens, hid_dim)

    ref_f32    = _ref_swiglu_f32(gate, up, swiglu_limit)  # (total_tokens, hid_dim)
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
    gate = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)
    up   = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_k   = SepProductSiluQuant.apply(gate, up,
                                        torch.float8_e4m3fn, True,
                                        swiglu_limit, None)
    deq_k   = _dequant_2d(out_k)                           # (total_tokens, hid_dim)
    ref_f32 = _ref_swiglu_f32(gate, up, swiglu_limit)      # (total_tokens, hid_dim)

    torch.testing.assert_close(deq_k.float(), ref_f32, atol=0.5, rtol=0.15)


# ── backward: kernel vs f32 autograd reference ────────────────────────────────

@pytest.mark.parametrize("total_tokens,hid_dim", SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_bwd_vs_ref(total_tokens: int, hid_dim: int, swiglu_limit: float):
    """Dequantised grad_gate and grad_up must match f32 autograd reference.

    The f32 reference uses PyTorch autograd through clamp+silu, which provides
    an independent gradient path. Tolerance is loose (atol=0.5) because fp8
    quantisation of the gradient introduces a dequant error of up to 0.5 for
    values in [8, 16).
    """
    torch.manual_seed(7)
    gate = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    up   = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    # ── kernel backward ──
    out_k = SepProductSiluQuant.apply(gate, up,
                                      torch.float8_e4m3fn, True,
                                      swiglu_limit, None)
    torch.autograd.backward([out_k], [grad_out])
    grad_gate_deq = _dequant_2d(gate.grad)   # (total_tokens, hid_dim)
    grad_up_deq   = _dequant_2d(up.grad)     # (total_tokens, hid_dim)
    gate.grad = None
    up.grad   = None

    # ── f32 autograd reference ──
    g_f32 = gate.detach().float().requires_grad_(True)
    u_f32 = up.detach().float().requires_grad_(True)
    g = g_f32.clamp(max=swiglu_limit)           if swiglu_limit > 0 else g_f32
    u = u_f32.clamp(-swiglu_limit, swiglu_limit) if swiglu_limit > 0 else u_f32
    ref = g * torch.sigmoid(g) * u
    ref.backward(grad_out.float())
    # Reference grads are 3D (1, total_tokens, hid_dim) → flatten to 2D for comparison
    ref_grad_gate = g_f32.grad.reshape(total_tokens, hid_dim)
    ref_grad_up   = u_f32.grad.reshape(total_tokens, hid_dim)

    torch.testing.assert_close(
        grad_gate_deq.float(), ref_grad_gate,
        atol=0.5, rtol=0.15,
        msg="dequantised grad_gate must be close to f32 autograd",
    )
    torch.testing.assert_close(
        grad_up_deq.float(), ref_grad_up,
        atol=0.5, rtol=0.15,
        msg="dequantised grad_up must be close to f32 autograd",
    )


# ── clamp-boundary gradient correctness ───────────────────────────────────────

def test_clamp_boundary_gradients():
    """Verify clamp-boundary gradient semantics with a controlled input.

    Construction:
      - Rows [:64]:  gate and up exceed the clamp limit → grad_gate, grad_up must be ~0
      - Rows [64:]:  gate and up within range           → grad_gate, grad_up must flow
    """
    limit        = 4.0
    total_tokens = 128   # divisible by kernel tile sizes
    hid_dim      = 128
    device       = "cuda"

    gate = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)
    up   = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    gate[0, :64, :] = limit * 2    # gate saturated
    up[0, :64, :]   = limit * 2    # up saturated
    gate[0, 64:, :] = limit * 0.5  # gate in range
    up[0, 64:, :]   = limit * 0.5  # up in range

    grad_out = torch.ones(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    gate.requires_grad_(True)
    up.requires_grad_(True)

    out = SepProductSiluQuant.apply(gate, up,
                                    torch.float8_e4m3fn, True,
                                    limit, None)
    torch.autograd.backward([out], [grad_out])

    grad_gate_deq = _dequant_2d(gate.grad)   # (total_tokens, hid_dim)
    grad_up_deq   = _dequant_2d(up.grad)     # (total_tokens, hid_dim)

    assert grad_gate_deq[:64].abs().max().item() < 1e-2, \
        "grad_gate must be ~0 where gate was clamped"
    assert grad_up_deq[:64].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped"
    assert grad_gate_deq[64:].abs().max().item() > 1e-3, \
        "grad_gate must be non-zero where gate was not clamped"
    assert grad_up_deq[64:].abs().max().item() > 1e-3, \
        "grad_up must be non-zero where up was not clamped"


# ── clip_counts monitoring correctness ────────────────────────────────────────

def test_clip_counts():
    """clip_counts must accumulate the exact number of clamped elements.

    Construction (shape (1,128,128), limit=4.0):
      - Rows   0:32  gate = limit*2  → 32*128 = 4096 gate clips
      - Rows  32:64  up  = +limit*2  → 32*128 = 4096 up clips (positive side)
      - Rows  64:96  up  = -limit*2  → 32*128 = 4096 up clips (negative side)
      - Rows  96:128 zeros           → 0 clips

    Expected: clip_counts[0] = 4096, clip_counts[1] = 8192.
    """
    limit        = 4.0
    total_tokens = 128
    hid_dim      = 128
    device       = "cuda"

    gate = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)
    up   = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    gate[0, :32,  :] = limit * 2   # gate saturated
    up[0,  32:64, :] = limit * 2   # up saturated positive
    up[0,  64:96, :] = -limit * 2  # up saturated negative

    clip_counts = torch.zeros(2, dtype=torch.int32, device=device)

    SepProductSiluQuant.apply(gate, up,
                              torch.float8_e4m3fn, True,
                              limit, clip_counts)

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

    gate = torch.randn(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16) * 100
    up   = torch.randn(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16) * 100

    # (a) limit=0 → monitor_on=False, counts must stay at zero
    clip_counts = torch.zeros(2, dtype=torch.int32, device=device)
    SepProductSiluQuant.apply(gate, up,
                              torch.float8_e4m3fn, True,
                              0.0, clip_counts)
    assert clip_counts[0].item() == 0 and clip_counts[1].item() == 0, \
        "counts must stay zero when swiglu_limit=0.0"

    # (b) clip_counts=None → no error, monitoring simply skipped
    SepProductSiluQuant.apply(gate, up,
                              torch.float8_e4m3fn, True,
                              8.0, None)


# ── cross-saturation gradient correctness ─────────────────────────────────────

def test_clamp_cross_saturation():
    """Verify that gate and up clamp masks are independent.

    Row groups (32 rows each, total=128):
      - Rows   0:32  gate saturated, up saturated   → grad_gate~0, grad_up~0
      - Rows  32:64  gate saturated, up in range    → grad_gate~0, grad_up nonzero
      - Rows  64:96  gate in range,  up saturated   → grad_gate nonzero, grad_up~0
      - Rows  96:128 both in range                  → grad_gate nonzero, grad_up nonzero
    """
    limit        = 4.0
    total_tokens = 128
    hid_dim      = 128
    device       = "cuda"

    gate = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)
    up   = torch.zeros(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    gate[0, :64,  :] = limit * 2    # groups 0 and 1: gate saturated
    gate[0, 64:,  :] = limit * 0.5  # groups 2 and 3: gate in range
    up[0,  :32,   :] = limit * 2    # group 0: up saturated
    up[0,  64:96, :] = limit * 2    # group 2: up saturated
    up[0,  32:64, :] = limit * 0.5  # group 1: up in range
    up[0,  96:,   :] = limit * 0.5  # group 3: up in range

    grad_out = torch.ones(1, total_tokens, hid_dim, device=device, dtype=torch.bfloat16)

    gate.requires_grad_(True)
    up.requires_grad_(True)

    out = SepProductSiluQuant.apply(gate, up,
                                    torch.float8_e4m3fn, True,
                                    limit, None)
    torch.autograd.backward([out], [grad_out])

    grad_gate_deq = _dequant_2d(gate.grad)  # (total_tokens, hid_dim)
    grad_up_deq   = _dequant_2d(up.grad)

    assert grad_gate_deq[:64].abs().max().item() < 1e-2, \
        "grad_gate must be ~0 where gate was clamped (rows 0:64)"
    assert grad_gate_deq[64:].abs().max().item() > 1e-3, \
        "grad_gate must be nonzero where gate was not clamped (rows 64:128)"

    assert grad_up_deq[:32].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped (rows 0:32)"
    assert grad_up_deq[64:96].abs().max().item() < 1e-2, \
        "grad_up must be ~0 where up was clamped (rows 64:96)"
    assert grad_up_deq[32:64].abs().max().item() > 1e-3, \
        "grad_up must be nonzero where up was not clamped (rows 32:64)"
    assert grad_up_deq[96:].abs().max().item() > 1e-3, \
        "grad_up must be nonzero where up was not clamped (rows 96:128)"


# ── power_two_max_round=False scaling path ────────────────────────────────────

def _ref_fp8_quantize_no_pow2(x_f32: torch.Tensor):
    """Block-wise FP8 quantisation without power-of-two rounding.

    scale     = amax / FP8_MAX
    inv_scale = FP8_MAX / amax
    """
    FP8_MAX = 448.0
    M, K = x_f32.shape
    bpr = K // 128
    x_b = x_f32.reshape(M, bpr, 128)
    amax = x_b.abs().amax(dim=-1).clamp(min=FP8_MAX * 1e-30)
    inv_scale = FP8_MAX / amax
    fp8 = (x_b * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return fp8.reshape(M, K).to(torch.float8_e4m3fn)


POW2_FALSE_SHAPES = [(128, 128), (256, 512)]

@pytest.mark.parametrize("total_tokens,hid_dim", POW2_FALSE_SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_power_two_max_round_false(total_tokens: int, hid_dim: int, swiglu_limit: float):
    """power_two_max_round=False must produce valid FP8 output and correct gradients.

    FP8 bytes must match the non-power-of-two reference formula.
    Dequantized output and gradients must be close to f32 reference.
    """
    torch.manual_seed(42)
    gate = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    up   = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(1, total_tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_k = SepProductSiluQuant.apply(gate, up,
                                      torch.float8_e4m3fn, False,
                                      swiglu_limit, None)

    # FP8 bytes match non-pow2 reference
    fp8_k   = _qtensor_to_fp8_2d(out_k)
    ref_f32 = _ref_swiglu_f32(gate, up, swiglu_limit)
    ref_fp8 = _ref_fp8_quantize_no_pow2(ref_f32)
    torch.testing.assert_close(
        fp8_k.to(torch.bfloat16), ref_fp8.to(torch.bfloat16), atol=0, rtol=0,
    )

    # Backward matches f32 autograd
    torch.autograd.backward([out_k], [grad_out])
    grad_gate_deq = _dequant_2d(gate.grad)
    grad_up_deq   = _dequant_2d(up.grad)
    gate.grad = None
    up.grad   = None

    g_f32 = gate.detach().float().requires_grad_(True)
    u_f32 = up.detach().float().requires_grad_(True)
    g = g_f32.clamp(max=swiglu_limit)            if swiglu_limit > 0 else g_f32
    u = u_f32.clamp(-swiglu_limit, swiglu_limit) if swiglu_limit > 0 else u_f32
    (g * torch.sigmoid(g) * u).backward(grad_out.float())

    torch.testing.assert_close(
        grad_gate_deq.float(), g_f32.grad.reshape(total_tokens, hid_dim),
        atol=0.5, rtol=0.15,
        msg="power_two_max_round=False: grad_gate must match f32 autograd",
    )
    torch.testing.assert_close(
        grad_up_deq.float(), u_f32.grad.reshape(total_tokens, hid_dim),
        atol=0.5, rtol=0.15,
        msg="power_two_max_round=False: grad_up must match f32 autograd",
    )


# ── batch > 1 correctness ─────────────────────────────────────────────────────

BATCH_GT1_SHAPES = [(2, 64, 256), (4, 32, 128)]

@pytest.mark.parametrize("batch,tokens,hid_dim", BATCH_GT1_SHAPES)
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
def test_batch_gt_1(batch: int, tokens: int, hid_dim: int, swiglu_limit: float):
    """3D inputs with batch > 1 must produce correct forward output and gradients.

    The kernel flattens (batch, tokens, hid_dim) → (batch*tokens, hid_dim) internally.
    This test verifies the reshaping and scale layout are consistent.
    """
    torch.manual_seed(42)
    gate = torch.randn(batch, tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    up   = torch.randn(batch, tokens, hid_dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    grad_out = torch.randn(batch, tokens, hid_dim, device="cuda", dtype=torch.bfloat16)

    out_k = SepProductSiluQuant.apply(gate, up,
                                      torch.float8_e4m3fn, True,
                                      swiglu_limit, None)

    n_rows  = batch * tokens
    deq_k   = _dequant_2d(out_k)                          # (n_rows, hid_dim)
    ref_f32 = _ref_swiglu_f32(gate, up, swiglu_limit)     # (n_rows, hid_dim)
    torch.testing.assert_close(deq_k.float(), ref_f32, atol=0.5, rtol=0.15,
                                msg="batch>1: forward dequantised must match f32 ref")

    torch.autograd.backward([out_k], [grad_out])
    grad_gate_deq = _dequant_2d(gate.grad)   # (n_rows, hid_dim)
    grad_up_deq   = _dequant_2d(up.grad)
    gate.grad = None
    up.grad   = None

    g_f32 = gate.detach().float().requires_grad_(True)
    u_f32 = up.detach().float().requires_grad_(True)
    g = g_f32.clamp(max=swiglu_limit)            if swiglu_limit > 0 else g_f32
    u = u_f32.clamp(-swiglu_limit, swiglu_limit) if swiglu_limit > 0 else u_f32
    (g * torch.sigmoid(g) * u).backward(grad_out.float())

    torch.testing.assert_close(
        grad_gate_deq.float(), g_f32.grad.reshape(n_rows, hid_dim),
        atol=0.5, rtol=0.15,
        msg="batch>1: grad_gate must match f32 autograd",
    )
    torch.testing.assert_close(
        grad_up_deq.float(), u_f32.grad.reshape(n_rows, hid_dim),
        atol=0.5, rtol=0.15,
        msg="batch>1: grad_up must match f32 autograd",
    )
