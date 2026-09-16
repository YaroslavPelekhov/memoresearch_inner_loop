"""Correctness tests for the v5 two-phase row→col grouped FP8 CUDA kernel.

Verifies:
  1. Output shapes are correct.
  2. Output scales are valid (positive powers of 2).
  3. Dequantized roundtrip: output_fp8 * col_scale ≈ input_fp8 * row_scale
     (within FP8 quantization tolerance).
  4. Caching behaviour (QTensor shell reuse, slot independence).
  5. Determinism (same inputs → identical outputs).

The correctness check uses a dequantized-roundtrip invariant rather than
bit-exact reference matching, because the CUDA kernel is compiled with
``--use_fast_math`` (``--prec-div=false``), which can cause the division
``amax / 448`` to differ from IEEE 754 by ~1 ULP.  Near a power-of-2
boundary this shifts ``pow2_ceil`` by one step — a valid quantization
choice that does not affect downstream accuracy.

Run::

    python -m pytest test_row2col.py -v --tb=short
    # or standalone:
    python test_row2col.py
"""

from __future__ import annotations
import sys, os, types as _types, pathlib as _pathlib

import transformer_engine as te  # noqa: F401  (must precede transformer_engine_torch)


def _setup_standalone_path() -> None:
    """Stub llmfoundry/llmfoundry.models so we can import the cuda_row2col
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
    # Generate via randn → float8 to guarantee no NaN bit patterns (0x7F/0xFF).
    x = torch.randn(T, H, device=device)
    data = x.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(T, H // TILE, dtype=torch.float32, device=device) * 0.1 + 0.01
    m_indices = make_m_indices(group_sizes, device)
    return data, scales, m_indices


def _get_api():
    from llmfoundry.models.ops.float8.cuda_kernels.row2col_te_quantization import (
        row2col_prepare, row2col_execute_cached, _single_cache,
    )
    return row2col_prepare, row2col_execute_cached, _single_cache


# ---------------------------------------------------------------------------
# Correctness: dequantized roundtrip invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", SHAPES, ids=[
    f"{T}x{H}_g{'x'.join(str(g) for g in gs)}" for T, H, gs in SHAPES
])
def test_correctness(T, H, gs):
    """Verify dequant(output) ≈ dequant(input) within FP8 precision."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    _single_cache.clear()

    data, scales, m_idx = make_inputs(T, H, gs)
    data_f32 = data.view(torch.float8_e4m3fn).float()

    prep = row2col_prepare(data, scales, gs, m_idx)
    result = row2col_execute_cached(prep, cache_slot=f"test_{T}_{H}_{len(gs)}")
    torch.cuda.synchronize()

    assert len(result) == len(gs)

    offsets = [0]
    for g in gs:
        offsets.append(offsets[-1] + g)

    for g_idx, g_size in enumerate(gs):
        qt = result[g_idx]
        g_off = offsets[g_idx]

        cuda_data = qt._columnwise_data          # [H, g_size]
        cuda_scale = qt._columnwise_scale_inv     # [g_size//128, H]

        # --- shape check ---
        assert cuda_data.shape == (H, g_size), \
            f"G{g_idx}: data shape {cuda_data.shape}, expected ({H}, {g_size})"
        assert cuda_scale.shape == (g_size // TILE, H), \
            f"G{g_idx}: scale shape {cuda_scale.shape}, expected ({g_size // TILE}, {H})"

        # --- scales are positive powers of 2 ---
        s_bits = cuda_scale.contiguous().view(torch.int32)
        assert (s_bits & 0x007FFFFF == 0).all(), \
            f"G{g_idx}: some scales are not exact powers of 2"
        assert (cuda_scale > 0).all(), \
            f"G{g_idx}: some scales are non-positive"

        # --- dequantized roundtrip per row-tile ---
        out_f32 = cuda_data.view(torch.float8_e4m3fn).float()  # [H, g_size]

        for tr in range(g_size // TILE):
            r0 = g_off + tr * TILE

            # Input dequant: [128, H]
            tile = data_f32[r0:r0 + TILE, :]
            row_sc = scales[r0:r0 + TILE, :]                  # [128, H//128]
            input_dq = tile * row_sc.repeat_interleave(TILE, dim=1)[:, :H]

            # Output dequant: [H, 128] — transposed relative to input
            out_tile = out_f32[:, tr * TILE:(tr + 1) * TILE]   # [H, 128]
            col_sc = cuda_scale[tr, :]                         # [H]
            output_dq = out_tile * col_sc.unsqueeze(1)         # [H, 128]

            # Compare: output_dq[h, t] should ≈ input_dq[t, h]
            diff = (input_dq.T - output_dq).abs()
            peak = input_dq.abs().max().clamp(min=1e-10)
            norm_err = diff.max() / peak

            # FP8 E4M3 has 3 mantissa bits → max quantization error ≈ 6.25%
            # of peak.  Using 8% threshold for margin.
            assert norm_err < 0.08, (
                f"G{g_idx} tile {tr}: normalised roundtrip error "
                f"{norm_err.item():.4f} exceeds 8%"
            )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 512, [256, 256, 256, 256]),
    (2048, 1024, [256] * 8),
])
def test_cache_reuse(T, H, gs):
    """Second call with same group_sizes reuses QTensor object identity."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    slot = f"reuse_{T}_{H}"
    _single_cache.pop(slot, None)

    data1, scales1, m_idx = make_inputs(T, H, gs)
    data2, scales2, _ = make_inputs(T, H, gs)

    prep1 = row2col_prepare(data1, scales1, gs, m_idx)
    r1 = row2col_execute_cached(prep1, cache_slot=slot)

    prep2 = row2col_prepare(data2, scales2, gs, m_idx)
    r2 = row2col_execute_cached(prep2, cache_slot=slot)
    torch.cuda.synchronize()

    for i in range(len(gs)):
        assert r1[i] is r2[i], \
            f"Group {i}: QTensor shell not reused on cache hit"


@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 512, [256, 256, 256, 256]),
])
def test_cache_miss_on_new_sizes(T, H, gs):
    """Different group_sizes invalidate the cache."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    slot = "miss_test"
    _single_cache.pop(slot, None)

    data1, scales1, m_idx1 = make_inputs(T, H, gs)
    prep1 = row2col_prepare(data1, scales1, gs, m_idx1)
    r1 = row2col_execute_cached(prep1, cache_slot=slot)

    gs2 = [gs[0] + 128] + [g for g in gs[1:]]
    gs2[-1] -= 128
    assert sum(gs2) == T
    data2, scales2, m_idx2 = make_inputs(T, H, gs2)
    prep2 = row2col_prepare(data2, scales2, gs2, m_idx2)
    r2 = row2col_execute_cached(prep2, cache_slot=slot)
    torch.cuda.synchronize()

    assert len(r1) != len(r2) or any(
        r1[i] is not r2[i] for i in range(min(len(r1), len(r2)))
    ), "Cache should have been invalidated for different group_sizes"


@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 512, [256, 256, 256, 256]),
])
def test_separate_cache_slots(T, H, gs):
    """Different cache_slots maintain independent caches."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    _single_cache.pop("slot_a", None)
    _single_cache.pop("slot_b", None)

    data, scales, m_idx = make_inputs(T, H, gs)

    prep = row2col_prepare(data, scales, gs, m_idx)
    ra = row2col_execute_cached(prep, cache_slot="slot_a")

    prep = row2col_prepare(data, scales, gs, m_idx)
    rb = row2col_execute_cached(prep, cache_slot="slot_b")
    torch.cuda.synchronize()

    for i in range(len(gs)):
        assert ra[i] is not rb[i], \
            f"Group {i}: different slots should have different QTensor objects"


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 512, [128, 256, 384, 256]),
    (2048, 1024, [256] * 8),
])
def test_output_shapes(T, H, gs):
    """Verify output QTensor shapes are correct."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    _single_cache.clear()

    data, scales, m_idx = make_inputs(T, H, gs)
    prep = row2col_prepare(data, scales, gs, m_idx)
    result = row2col_execute_cached(prep, cache_slot=f"shape_{T}_{H}")
    torch.cuda.synchronize()

    assert len(result) == len(gs)
    for i, g in enumerate(gs):
        qt = result[i]
        cd = qt._columnwise_data
        cs = qt._columnwise_scale_inv
        assert cd.shape == (H, g), \
            f"G{i}: data shape {cd.shape}, expected ({H}, {g})"
        assert cs.shape == (g // TILE, H), \
            f"G{i}: scale shape {cs.shape}, expected ({g // TILE}, {H})"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,H,gs", [
    (512, 256, [256, 256]),
    (1024, 1024, [256, 256, 256, 256]),
])
def test_deterministic(T, H, gs):
    """Same inputs produce identical outputs across calls."""
    row2col_prepare, row2col_execute_cached, _single_cache = _get_api()
    _single_cache.clear()

    data, scales, m_idx = make_inputs(T, H, gs)

    prep1 = row2col_prepare(data, scales, gs, m_idx)
    r1 = row2col_execute_cached(prep1, cache_slot="det_1")

    _single_cache.clear()
    prep2 = row2col_prepare(data, scales, gs, m_idx)
    r2 = row2col_execute_cached(prep2, cache_slot="det_2")
    torch.cuda.synchronize()

    for i in range(len(gs)):
        d1 = r1[i]._columnwise_data
        d2 = r2[i]._columnwise_data
        s1 = r1[i]._columnwise_scale_inv
        s2 = r2[i]._columnwise_scale_inv
        if d1.dtype == torch.float8_e4m3fn:
            d1 = d1.view(torch.uint8)
        if d2.dtype == torch.float8_e4m3fn:
            d2 = d2.view(torch.uint8)
        assert torch.equal(d1, d2), f"G{i}: data not deterministic"
        assert torch.equal(s1, s2), f"G{i}: scale not deterministic"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
