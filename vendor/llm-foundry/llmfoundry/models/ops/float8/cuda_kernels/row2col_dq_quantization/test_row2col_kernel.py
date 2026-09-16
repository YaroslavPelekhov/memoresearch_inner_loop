"""Correctness tests for the fused row→col grouped FP8 CUDA kernel (no TE).

Verifies:
  1. Output shapes.
  2. Output scales are positive powers of 2.
  3. Dequantized roundtrip: output_fp8 * col_scale ≈ input_fp8 * row_scale
     (within FP8 quantization tolerance).
  4. Determinism (same inputs → identical outputs).
  5. Match against pure-PyTorch reference implementation.

The correctness check uses a dequantized-roundtrip invariant rather than
bit-exact reference matching, because the CUDA kernel is compiled with
``--use_fast_math`` (``--prec-div=false``), which can cause the division
``amax / 448`` to differ from IEEE 754 by ~1 ULP.  Near a power-of-2
boundary this shifts ``pow2_ceil`` by one step — a valid quantization
choice that does not affect downstream accuracy.

Run::

    python -m pytest test_row2col_kernel.py -v --tb=short
    # or standalone:
    python test_row2col_kernel.py
"""

from __future__ import annotations

import sys
import os
import types as _types
import pathlib as _pathlib


def _setup_standalone_path() -> None:
    """Add repo root to sys.path so we can import the cuda_kernels
    sub-package without pulling in composer and other heavy deps."""
    _repo = str(_pathlib.Path(__file__).resolve().parents[6])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    _llmf = os.path.join(_repo, "llmfoundry")
    for _pkg, _dir in [
        ("llmfoundry",        _llmf),
        ("llmfoundry.models", os.path.join(_llmf, "models")),
    ]:
        if _pkg not in sys.modules:
            _stub = _types.ModuleType(_pkg)
            _stub.__path__ = [_dir]
            _stub.__package__ = _pkg
            _stub.__file__ = os.path.join(_dir, "__init__.py")
            sys.modules[_pkg] = _stub


_setup_standalone_path()

import pytest
import torch

TILE = 128

# ---------------------------------------------------------------------------
# Test shapes: (T, H, group_sizes)
# All group sizes are multiples of 128; sum(group_sizes) == T; T,H % 128 == 0
# ---------------------------------------------------------------------------

SHAPES = [
    # --- minimal ---
    (128, 128, [128]),
    # --- small ---
    (256, 128, [128, 128]),
    (256, 256, [256]),
    (384, 256, [128, 256]),
    (384, 384, [384]),
    # --- medium, equal groups ---
    (512, 256, [256, 256]),
    (512, 512, [128, 128, 128, 128]),
    (512, 512, [256, 256]),
    (768, 384, [256, 256, 256]),
    # --- medium, unequal groups ---
    (512, 256, [128, 384]),
    (512, 512, [128, 128, 256]),
    (768, 512, [128, 256, 384]),
    # --- large, equal groups ---
    (1024, 512, [256, 256, 256, 256]),
    (1024, 1024, [256, 256, 256, 256]),
    (1024, 1024, [512, 512]),
    # --- large, unequal groups ---
    (1024, 512, [128, 256, 384, 256]),
    (1024, 1024, [128, 128, 256, 512]),
    # --- large, single group ---
    (1024, 1024, [1024]),
    # --- xlarge ---
    (2048, 1024, [256] * 8),
    (2048, 2048, [256] * 8),
    (4096, 2048, [512] * 8),
    (4096, 4096, [512] * 8),
    # --- xlarge, single group ---
    (4096, 4096, [4096]),
    # --- xlarge, many small groups ---
    (4096, 4096, [128] * 32),
    # --- xxlarge ---
    (8192, 4096, [1024] * 8),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_m_indices(group_sizes, device):
    T = sum(group_sizes)
    m = torch.zeros(T, dtype=torch.int32, device=device)
    off = 0
    for gid, gs in enumerate(group_sizes):
        m[off:off + gs] = gid
        off += gs
    return m


def make_inputs(T, H, group_sizes, device="cuda"):
    x = torch.randn(T, H, device=device)
    data = x.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = (
        torch.rand(T, H // TILE, dtype=torch.float32, device=device) * 0.1
        + 0.01
    )
    m_indices = make_m_indices(group_sizes, device)
    return data, scales, m_indices


def _get_api():
    from llmfoundry.models.ops.float8.cuda_kernels.row2col_dq_quantization import (
        row2col_requantize,
    )
    return row2col_requantize


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------

def _reference_pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(x)))


def reference_row2col(data_uint8, scales_inv, group_sizes, m_indices):
    """Pure-PyTorch reference implementation (slow but correct)."""
    T, H = data_uint8.shape
    dev = data_uint8.device

    data_f32 = data_uint8.view(torch.float8_e4m3fn).float()

    col_flat = torch.empty(H * T, dtype=torch.uint8, device=dev)
    col_scale = torch.empty(T // TILE, H, dtype=torch.float32, device=dev)

    offsets = [0]
    for g in group_sizes:
        offsets.append(offsets[-1] + g)

    for rb in range(T // TILE):
        for cb in range(H // TILE):
            r0, c0 = rb * TILE, cb * TILE

            tile = data_f32[r0:r0 + TILE, c0:c0 + TILE]
            rsc = scales_inv[r0:r0 + TILE, cb:cb + 1]

            dq = tile * rsc

            amax = dq.abs().amax(dim=0)
            raw_sc = (amax / 448.0).clamp(min=1e-30)
            sc = _reference_pow2_ceil(raw_sc)

            rq = (dq / sc.unsqueeze(0)).to(torch.float8_e4m3fn)
            rq_T = rq.T.contiguous().view(torch.uint8)   # [TILE_h, TILE_t]

            gid = m_indices[r0].item()
            g_off = offsets[gid]
            g_sz = group_sizes[gid]
            tok_in = r0 - g_off
            flat_base = g_off * H

            h_range = torch.arange(TILE, device=dev)
            t_range = torch.arange(TILE, device=dev)
            addrs = (
                flat_base
                + (c0 + h_range).unsqueeze(1) * g_sz
                + tok_in
                + t_range.unsqueeze(0)
            ).flatten()
            col_flat.scatter_(0, addrs, rq_T.flatten())

            col_scale[rb, c0:c0 + TILE] = sc

    return col_flat, col_scale


# ---------------------------------------------------------------------------
# Correctness: dequantized roundtrip invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", SHAPES, ids=[
    f"{T}x{H}_g{'x'.join(str(g) for g in gs)}" for T, H, gs in SHAPES
])
def test_correctness(T, H, gs):
    """Verify dequant(output) ≈ dequant(input) within FP8 precision."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)
    data_f32 = data.view(torch.float8_e4m3fn).float()

    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)
    Y, S = row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    assert Y.shape == (H * T,), f"Y shape: {Y.shape}, expected ({H * T},)"
    assert S.shape == (T // TILE, H), \
        f"S shape: {S.shape}, expected ({T // TILE}, {H})"

    offsets = [0]
    for g in gs:
        offsets.append(offsets[-1] + g)

    for g_idx, g_size in enumerate(gs):
        g_off = offsets[g_idx]
        flat_base = g_off * H

        out_block = Y[flat_base:flat_base + H * g_size].view(H, g_size)

        for tr in range(g_size // TILE):
            r0 = g_off + tr * TILE
            seq_block = r0 // TILE

            tile = data_f32[r0:r0 + TILE, :]
            row_sc = scales[r0:r0 + TILE, :]
            input_dq = tile * row_sc.repeat_interleave(TILE, dim=1)[:, :H]

            out_tile = out_block[:, tr * TILE:(tr + 1) * TILE]
            out_f32 = out_tile.view(torch.float8_e4m3fn).float()
            col_sc = S[seq_block, :]
            output_dq = out_f32 * col_sc.unsqueeze(1)

            diff = (input_dq.T - output_dq).abs()
            peak = input_dq.abs().max().clamp(min=1e-10)
            norm_err = diff.max() / peak

            # FP8 E4M3: 3 mantissa bits → max quant error ~6.25%. Use 8% margin.
            assert norm_err < 0.08, (
                f"G{g_idx} tile {tr}: normalised roundtrip error "
                f"{norm_err.item():.4f} exceeds 8%"
            )


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 512, [128, 256, 384, 256]),
    (2048, 1024, [256] * 8),
])
def test_output_shapes(T, H, gs):
    """Verify output tensor shapes."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)
    Y, S = row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    assert Y.shape == (H * T,)
    assert S.shape == (T // TILE, H)
    assert Y.dtype == torch.uint8
    assert S.dtype == torch.float32


# ---------------------------------------------------------------------------
# Scales are positive powers of 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", SHAPES[:10], ids=[
    f"{T}x{H}_g{'x'.join(str(g) for g in gs)}" for T, H, gs in SHAPES[:10]
])
def test_scales_pow2(T, H, gs):
    """Verify output scales are positive exact powers of 2."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)
    _, S = row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    s_bits = S.contiguous().view(torch.int32)
    assert (s_bits & 0x007FFFFF == 0).all(), \
        "Some scales are not exact powers of 2"
    assert (S > 0).all(), "Some scales are non-positive"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 1024, [256, 256, 256, 256]),
])
def test_deterministic(T, H, gs):
    """Same inputs produce identical outputs across calls."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)

    Y1, S1 = row2col_requantize(data, m_idx, gs_tensor, scales)
    Y2, S2 = row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    assert torch.equal(Y1, Y2), "data not deterministic"
    assert torch.equal(S1, S2), "scales not deterministic"


# ---------------------------------------------------------------------------
# Match against pure-PyTorch reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", SHAPES[:8], ids=[
    f"{T}x{H}_g{'x'.join(str(g) for g in gs)}" for T, H, gs in SHAPES[:8]
])
def test_vs_reference(T, H, gs):
    """Compare CUDA kernel against pure-PyTorch reference (dequant level)."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)

    Y_cuda, S_cuda = row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    Y_ref, S_ref = reference_row2col(data, scales, gs, m_idx)

    offsets = [0]
    for g in gs:
        offsets.append(offsets[-1] + g)

    for g_idx, g_size in enumerate(gs):
        g_off = offsets[g_idx]
        flat_base = g_off * H

        cuda_block = Y_cuda[flat_base:flat_base + H * g_size].view(H, g_size)
        ref_block = Y_ref[flat_base:flat_base + H * g_size].view(H, g_size)

        for tr in range(g_size // TILE):
            sb = (g_off + tr * TILE) // TILE

            cuda_dq = (
                cuda_block[:, tr * TILE:(tr + 1) * TILE]
                .view(torch.float8_e4m3fn).float()
                * S_cuda[sb].unsqueeze(1)
            )
            ref_dq = (
                ref_block[:, tr * TILE:(tr + 1) * TILE]
                .view(torch.float8_e4m3fn).float()
                * S_ref[sb].unsqueeze(1)
            )

            peak = ref_dq.abs().max().clamp(min=1e-10)
            norm_err = (cuda_dq - ref_dq).abs().max() / peak

            assert norm_err < 0.08, (
                f"G{g_idx} tile {tr}: CUDA vs ref error "
                f"{norm_err.item():.4f} exceeds 8%"
            )


# ---------------------------------------------------------------------------
# group_sizes as Python list (not just tensor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (256, 256, [256]),
    (512, 256, [128, 384]),
])
def test_group_sizes_as_list(T, H, gs):
    """Verify the kernel accepts group_sizes as a Python list."""
    row2col_requantize = _get_api()

    data, scales, m_idx = make_inputs(T, H, gs)

    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)
    Y_t, S_t = row2col_requantize(data, m_idx, gs_tensor, scales)

    Y_l, S_l = row2col_requantize(data, m_idx, gs, scales)
    torch.cuda.synchronize()

    assert torch.equal(Y_t, Y_l), "List vs tensor group_sizes differ (data)"
    assert torch.equal(S_t, S_l), "List vs tensor group_sizes differ (scales)"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
