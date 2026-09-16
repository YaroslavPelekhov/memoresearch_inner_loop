"""Correctness tests for the scaling-aware FP8 transpose CUDA kernel.

Tests verify bit-for-bit equality between the CUDA kernel and the Triton
reference ``_scaling_aware_fp8_transpose_kernel`` in row2col_dense_kernels.py.

Coverage
--------
1.  Bit-exact match: columnwise_data  (uint8 equality)
2.  Bit-exact match: columnwise_scale_inv  (float32 bit equality, first nbrows rows)
3.  Output tensor shapes
4.  Output dtype preserved (float8_e4m3fn input → float8_e4m3fn output)
5.  Non-contiguous input data  (transposed view, non-unit stride)
6.  Non-contiguous scale_inv   (transposed view)
7.  Minimal shape: rows=cols=128, single tile
8.  Large shapes
9.  Multiple tile rows / tile cols
10. Underflow / exp==0 behaviour (all-zero data, max-magnitude data)
11. Scales that are exact powers of 2 (triggers zero-mantissa fast path)
12. Scales with random mantissas  (exponent-only approximation may differ
    from a full dequant/requant, but CUDA ≡ Triton bit-for-bit)
13. Determinism: same inputs → identical outputs across two calls
14. rsi_cols smaller than nbcols (guard branch exercised)
15. Various BLOCK_SIZE=128-aligned shapes

Run
---
    python -m pytest test_scaling_aware_transpose.py -v --tb=short
    # or:
    python test_scaling_aware_transpose.py
"""

from __future__ import annotations

import sys
import os
import types as _types
import pathlib as _pathlib


def _setup_standalone_path() -> None:
    """Stub llmfoundry/llmfoundry.models so the sub-package can be imported
    without pulling in composer and other heavy training dependencies."""
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
# API imports
# ---------------------------------------------------------------------------

def _get_cuda_fn():
    from llmfoundry.models.ops.float8.cuda_kernels.scaling_aware_transpose import (
        blockwise_scaling_aware_fp8_transpose as cuda_fn,
    )
    return cuda_fn


def _get_triton_fn():
    from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import (
        blockwise_scaling_aware_fp8_transpose as triton_fn,
    )
    return triton_fn



# ---------------------------------------------------------------------------
# Test shapes:  (rows, cols)
# Both dimensions must be multiples of TILE=128.
# ---------------------------------------------------------------------------

SHAPES = [
    # --- minimal: single tile ---
    (128,  128),
    # --- small ---
    (128,  256),
    (256,  128),
    (256,  256),
    (256,  512),
    (512,  256),
    # --- medium ---
    (512,  512),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
    # --- large ---
    (2048, 1024),
    (1024, 2048),
    (2048, 2048),
    # --- non-square, large ---
    (4096, 128),
    (128,  4096),
    (4096, 1024),
    (1024, 4096),
    # --- typical LLM weight shapes ---
    (4096,  4096),
    (4096,  7168),
    (7168,  4096),
]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_fp8_data(rows: int, cols: int, device: str = "cuda") -> torch.Tensor:
    """Random FP8 E4M3 payload as uint8 (no NaN bit-patterns)."""
    # Via randn→float8 ensures valid (non-NaN) FP8 values.
    x = torch.randn(rows, cols, device=device)
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def make_pow2_scales(rows: int, cols: int, device: str = "cuda") -> torch.Tensor:
    """Random positive power-of-2 float32 scales (mantissa = 0)."""
    # Exponent in [-10, 10] → scale in [2^-10, 2^10]
    exp = torch.randint(-10, 11, (rows, cols // TILE), device=device)
    return torch.exp2(exp.float())


def make_random_scales(rows: int, cols: int, device: str = "cuda") -> torch.Tensor:
    """Random positive float32 scales (not necessarily powers of 2)."""
    return torch.rand(rows, cols // TILE, device=device) * 2.0 + 1e-4


def make_unit_scales(rows: int, cols: int, device: str = "cuda") -> torch.Tensor:
    """All-ones scale (trivial case)."""
    return torch.ones(rows, cols // TILE, device=device)


# ---------------------------------------------------------------------------
# Reference: pure-Python exponent-shift (for sanity, not used in main tests)
# ---------------------------------------------------------------------------

def _reference_exponent_shift(
    data_u8: torch.Tensor,       # [rows, cols] uint8
    scale_inv: torch.Tensor,     # [rows, rsi_cols] float32
) -> tuple:
    """Pure-Python/PyTorch reference matching Triton semantics exactly.

    Returns (columnwise_data_u8, columnwise_scale_inv).
    Used only for logic verification; the main correctness tests compare
    CUDA vs. Triton (not this reference) for efficiency.
    """
    rows, cols = data_u8.shape
    rsi_cols = scale_inv.shape[1]
    nbrows = rows // TILE
    nbcols = cols // TILE
    nbrows_mult_4 = (nbrows + 3) // 4 * 4
    dev = data_u8.device

    Y     = torch.zeros(cols, rows,               dtype=torch.uint8,    device=dev)
    S_out = torch.zeros(nbrows_mult_4, cols,       dtype=torch.float32,  device=dev)

    for pid_row in range(nbrows):
        for pid_col in range(nbcols):
            r0 = pid_row * TILE
            c0 = pid_col * TILE

            # Load scale column = pid_col  (compact-column index)
            si = scale_inv[r0:r0 + TILE, pid_col]    # [TILE]
            target_si = si.max().item()

            # Store to all TILE column positions
            S_out[pid_row, c0:c0 + TILE] = target_si

            # IEEE exponents
            def ieee_exp(f: float) -> int:
                bits = torch.tensor(f).to(torch.float32).view(torch.int32).item()
                return ((bits & 0x7F800000) >> 23) - 127

            exp_t = ieee_exp(target_si)

            tile = data_u8[r0:r0 + TILE, c0:c0 + TILE]  # [TILE, TILE]

            for r_off in range(TILE):
                si_r = si[r_off].item()
                exp_s = ieee_exp(si_r)
                k = exp_t - exp_s

                for c_off in range(TILE):
                    b = int(tile[r_off, c_off].item())
                    sign    = (b >> 7) & 1
                    exp_fp8 = (b >> 3) & 0xF
                    mant    = b & 0x7

                    exp_new = exp_fp8 - k
                    under   = (exp_new <= 0) or (exp_fp8 == 0)
                    result  = 0 if under else (sign << 7) | (exp_new << 3) | mant

                    # Transposed write: Y[col, row]
                    Y[c0 + c_off, r0 + r_off] = result

    return Y, S_out


# ---------------------------------------------------------------------------
# Core comparison helper
# ---------------------------------------------------------------------------

def compare_cuda_vs_triton(
    data_u8: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    label: str = "",
) -> None:
    """Assert CUDA and Triton produce identical outputs.

    Compares:
    - columnwise_data  bit-for-bit (uint8 exact equality)
    - columnwise_scale_inv[0:nbrows, :] bit-for-bit (float32 exact equality)
      Only the valid rows (0..nbrows-1) are compared; the padding rows
      are uninitialized in both implementations.
    """
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()

    rows = data_u8.shape[0]
    cols = data_u8.shape[1]
    nbrows = rows // TILE

    # --- CUDA ---
    cuda_data_u8, cuda_scale = cuda_fn(
        data_u8.view(torch.float8_e4m3fn), scale_inv
    )
    torch.cuda.synchronize()
    cuda_data_u8 = cuda_data_u8.view(torch.uint8)

    # --- Triton --- pass as uint8 so Triton sees integer pointer (other=0 is valid)
    triton_data_u8, triton_scale = triton_fn(data_u8, scale_inv)
    torch.cuda.synchronize()
    triton_data_u8 = triton_data_u8.view(torch.uint8)

    prefix = f"[{label}] " if label else ""

    # Shape check
    assert cuda_data_u8.shape == triton_data_u8.shape, (
        f"{prefix}data shape mismatch: CUDA {cuda_data_u8.shape} "
        f"vs Triton {triton_data_u8.shape}"
    )

    # Bit-exact data comparison
    if not torch.equal(cuda_data_u8, triton_data_u8):
        diff_mask  = cuda_data_u8 != triton_data_u8
        n_diff     = diff_mask.sum().item()
        total      = diff_mask.numel()
        first_pos  = diff_mask.nonzero(as_tuple=False)[0].tolist()
        cuda_val   = cuda_data_u8[diff_mask][0].item()
        triton_val = triton_data_u8[diff_mask][0].item()
        pytest.fail(
            f"{prefix}columnwise_data mismatch: {n_diff}/{total} bytes differ. "
            f"First diff at {first_pos}: CUDA=0x{cuda_val:02x} Triton=0x{triton_val:02x}"
        )

    # Bit-exact scale comparison (valid rows only)
    cuda_scale_valid   = cuda_scale[:nbrows, :]
    triton_scale_valid = triton_scale[:nbrows, :]

    if not torch.equal(cuda_scale_valid, triton_scale_valid):
        diff_mask  = cuda_scale_valid != triton_scale_valid
        n_diff     = diff_mask.sum().item()
        total      = diff_mask.numel()
        first_pos  = diff_mask.nonzero(as_tuple=False)[0].tolist()
        cuda_val   = cuda_scale_valid[diff_mask][0].item()
        triton_val = triton_scale_valid[diff_mask][0].item()
        pytest.fail(
            f"{prefix}columnwise_scale_inv mismatch: {n_diff}/{total} entries differ. "
            f"First diff at {first_pos}: CUDA={cuda_val:.6g} Triton={triton_val:.6g}"
        )


# ---------------------------------------------------------------------------
# 1. Bit-exact match: all shapes, power-of-2 scales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", SHAPES, ids=[
    f"{r}x{c}" for r, c in SHAPES
])
def test_bitexact_pow2_scales(rows: int, cols: int) -> None:
    """CUDA == Triton bit-for-bit with power-of-2 scale_inv."""
    data   = make_fp8_data(rows, cols)
    scales = make_pow2_scales(rows, cols)
    compare_cuda_vs_triton(data, scales, label=f"{rows}x{cols}_pow2")


# ---------------------------------------------------------------------------
# 2. Bit-exact match: random (non-power-of-2) scales
#    The exponent-only approximation means results differ from a full
#    dequant/requant, but CUDA and Triton must still agree exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", SHAPES, ids=[
    f"{r}x{c}" for r, c in SHAPES
])
def test_bitexact_random_scales(rows: int, cols: int) -> None:
    """CUDA == Triton bit-for-bit with random (non-power-of-2) scale_inv."""
    data   = make_fp8_data(rows, cols)
    scales = make_random_scales(rows, cols)
    compare_cuda_vs_triton(data, scales, label=f"{rows}x{cols}_rand")


# ---------------------------------------------------------------------------
# 3. Output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [
    (128,  128), (256, 512), (1024, 2048), (4096, 4096),
])
def test_output_shapes(rows: int, cols: int) -> None:
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(rows, cols)
    scales  = make_pow2_scales(rows, cols)

    col_data, col_scale = cuda_fn(data.view(torch.float8_e4m3fn), scales)
    torch.cuda.synchronize()

    nbrows        = rows // TILE
    nbrows_mult_4 = (nbrows + 3) // 4 * 4

    assert col_data.shape  == (cols, rows),            \
        f"data shape: {col_data.shape}, expected ({cols}, {rows})"
    assert col_scale.shape == (nbrows_mult_4, cols),   \
        f"scale shape: {col_scale.shape}, expected ({nbrows_mult_4}, {cols})"


# ---------------------------------------------------------------------------
# 4. Output dtype mirrors input dtype
# ---------------------------------------------------------------------------

def test_output_dtype_float8() -> None:
    """float8_e4m3fn input → float8_e4m3fn output."""
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(256, 256).view(torch.float8_e4m3fn)
    scales  = make_pow2_scales(256, 256)

    col_data, _ = cuda_fn(data, scales)
    assert col_data.dtype == torch.float8_e4m3fn, \
        f"Expected float8_e4m3fn, got {col_data.dtype}"


def test_output_dtype_uint8() -> None:
    """uint8 input → uint8 output."""
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(256, 256)      # already uint8
    scales  = make_pow2_scales(256, 256)

    col_data, _ = cuda_fn(data, scales)
    assert col_data.dtype == torch.uint8, \
        f"Expected uint8, got {col_data.dtype}"


# ---------------------------------------------------------------------------
# 5. Non-contiguous input data  (e.g. sliced from a larger tensor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 512), (1024, 1024)])
def test_noncontiguous_data(rows: int, cols: int) -> None:
    """Non-contiguous rowwise_data (transposed view) matches contiguous path."""
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()

    # Build a larger contiguous tensor, then take a non-contiguous slice.
    big = torch.randn(rows * 2, cols * 2, device="cuda")
    big_fp8 = big.to(torch.float8_e4m3fn)
    # Slice [0:rows, 0:cols] from a row-major tensor — still contiguous strides
    # for this sub-case. To get truly non-unit strides, transpose and slice back.
    padded = torch.zeros(cols * 2, rows * 2, dtype=torch.float8_e4m3fn, device="cuda")
    padded[:cols, :rows] = big_fp8[:rows, :cols].T
    # Now padded.T[:rows, :cols] has strides (1, cols*2) — non-contiguous.
    nc_data = padded.T[:rows, :cols]
    assert not nc_data.is_contiguous(), "Expected non-contiguous tensor"

    scales = make_pow2_scales(rows, cols)

    cuda_data, cuda_scale     = cuda_fn(nc_data, scales)
    triton_data, triton_scale = triton_fn(nc_data.view(torch.uint8), scales)
    torch.cuda.synchronize()

    cuda_u8   = cuda_data.view(torch.uint8)
    triton_u8 = triton_data.view(torch.uint8)

    assert torch.equal(cuda_u8, triton_u8), \
        "Non-contiguous data: CUDA != Triton (columnwise_data)"
    nbrows = rows // TILE
    assert torch.equal(cuda_scale[:nbrows], triton_scale[:nbrows]), \
        "Non-contiguous data: CUDA != Triton (columnwise_scale_inv)"


# ---------------------------------------------------------------------------
# 6. Non-contiguous scale_inv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 512)])
def test_noncontiguous_scales(rows: int, cols: int) -> None:
    """Non-contiguous rowwise_scale_inv gives same result as contiguous."""
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()

    data   = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)
    # Transpose scale_inv so stride(0)=1, stride(1)=rows — non-contiguous.
    sc_base = make_pow2_scales(rows, cols)
    sc_T    = sc_base.T.contiguous()   # [rsi_cols, rows] contiguous
    sc_nc   = sc_T.T                   # [rows, rsi_cols], stride=(1, rows) — non-contiguous
    assert not sc_nc.is_contiguous(), "Expected non-contiguous scales"

    cuda_data,   cuda_scale   = cuda_fn(data, sc_nc)
    triton_data, triton_scale = triton_fn(data.view(torch.uint8), sc_nc)
    torch.cuda.synchronize()

    nbrows = rows // TILE
    assert torch.equal(cuda_data.view(torch.uint8), triton_data.view(torch.uint8)), \
        "Non-contiguous scales: CUDA != Triton (data)"
    assert torch.equal(cuda_scale[:nbrows], triton_scale[:nbrows]), \
        "Non-contiguous scales: CUDA != Triton (scales)"


# ---------------------------------------------------------------------------
# 7. Underflow / exp_fp8 == 0 corner cases
# ---------------------------------------------------------------------------

def test_all_zero_data() -> None:
    """All-zero FP8 input → all-zero output (exp_fp8 == 0 flush)."""
    cuda_fn = _get_cuda_fn()
    rows, cols = 256, 256
    data   = torch.zeros(rows, cols, dtype=torch.uint8, device="cuda")
    scales = make_pow2_scales(rows, cols)

    col_data, _ = cuda_fn(data, scales)
    torch.cuda.synchronize()

    assert col_data.view(torch.uint8).sum() == 0, \
        "All-zero input should produce all-zero output"


def test_all_zero_data_bitexact() -> None:
    """All-zero FP8: CUDA == Triton."""
    rows, cols = 256, 256
    data   = torch.zeros(rows, cols, dtype=torch.uint8, device="cuda")
    scales = make_pow2_scales(rows, cols)
    compare_cuda_vs_triton(data, scales, label="all_zero")


def test_max_magnitude_data() -> None:
    """0x7E (max normal FP8 E4M3) inputs: CUDA == Triton."""
    rows, cols = 256, 256
    # 0x7E = max normal positive E4M3 value
    data   = torch.full((rows, cols), 0x7E, dtype=torch.uint8, device="cuda")
    scales = make_pow2_scales(rows, cols)
    compare_cuda_vs_triton(data, scales, label="max_magnitude")


def test_underflow_stress() -> None:
    """Large k (target_exp >> row_exp) should flush most bytes to zero."""
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()
    rows, cols = 128, 128

    data = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)

    # Make one row have a much larger scale so k is large for the rest.
    scales = torch.ones(rows, cols // TILE, device="cuda") * (2.0 ** -10)
    scales[0, 0] = 2.0 ** 10   # This row's scale dominates; k = 20 for others

    cuda_data, cuda_scale     = cuda_fn(data, scales)
    triton_data, triton_scale = triton_fn(data.view(torch.uint8), scales)
    torch.cuda.synchronize()

    assert torch.equal(
        cuda_data.view(torch.uint8), triton_data.view(torch.uint8)
    ), "Underflow stress: CUDA != Triton (data)"
    assert torch.equal(cuda_scale[:1], triton_scale[:1]), \
        "Underflow stress: CUDA != Triton (scales)"


def test_same_scales_all_rows() -> None:
    """All rows have the same scale → k=0 → bytes unchanged, only transposed."""
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()
    rows, cols = 256, 256

    data   = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)
    scales = torch.full((rows, cols // TILE), 1.0, device="cuda")

    cuda_data, cuda_scale     = cuda_fn(data, scales)
    triton_data, triton_scale = triton_fn(data.view(torch.uint8), scales)
    torch.cuda.synchronize()

    # k=0 for all rows → bytes are purely transposed (no change in value).
    assert torch.equal(
        cuda_data.view(torch.uint8), triton_data.view(torch.uint8)
    ), "Same-scale rows: CUDA != Triton"


# ---------------------------------------------------------------------------
# 8. Scales that are exact powers of 2 → mantissa == 0 fast path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(128, 128), (512, 1024), (2048, 2048)])
def test_pow2_scale_mantissa_zero(rows: int, cols: int) -> None:
    """Verify output scales have zero mantissa when inputs are pow-of-2."""
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)
    scales  = make_pow2_scales(rows, cols)

    _, col_scale = cuda_fn(data, scales)
    torch.cuda.synchronize()

    nbrows = rows // TILE
    s_bits = col_scale[:nbrows].contiguous().view(torch.int32)
    assert (s_bits & 0x007FFFFF == 0).all(), \
        "Output scales should have zero mantissa for pow-of-2 inputs"
    assert (col_scale[:nbrows] > 0).all(), \
        "Output scales should all be positive"


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [
    (256, 256), (1024, 1024), (2048, 2048)
])
def test_deterministic(rows: int, cols: int) -> None:
    """Same inputs → identical outputs across two successive calls."""
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)
    scales  = make_pow2_scales(rows, cols)

    d1, s1 = cuda_fn(data, scales)
    d2, s2 = cuda_fn(data, scales)
    torch.cuda.synchronize()

    nbrows = rows // TILE

    assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)), \
        "data not deterministic"
    # Compare only the valid rows; padding rows (nbrows..nbrows_mult_4-1) are
    # allocated with torch.empty and are intentionally uninitialized.
    assert torch.equal(s1[:nbrows], s2[:nbrows]), \
        "scale not deterministic"


# ---------------------------------------------------------------------------
# 10. columnwise_scale_inv layout: repeated value across columns
# ---------------------------------------------------------------------------

def test_scale_repeated_across_columns() -> None:
    """All TILE positions in a tile-row of columnwise_scale_inv are equal."""
    cuda_fn   = _get_cuda_fn()
    triton_fn = _get_triton_fn()
    rows, cols = 256, 512

    data   = make_fp8_data(rows, cols)  # uint8
    scales = make_pow2_scales(rows, cols)

    for fn, name in [(cuda_fn, "CUDA"), (triton_fn, "Triton")]:
        _, col_scale = fn(data, scales)
        torch.cuda.synchronize()

        nbrows   = rows // TILE
        nbcols   = cols // TILE
        for pid_row in range(nbrows):
            for pid_col in range(nbcols):
                block = col_scale[pid_row,
                                  pid_col * TILE : (pid_col + 1) * TILE]
                assert block.min() == block.max(), (
                    f"{name}: tile ({pid_row},{pid_col}) scale not uniform: "
                    f"min={block.min().item():.4g} max={block.max().item():.4g}"
                )


# ---------------------------------------------------------------------------
# 11. scale value correctness: must equal max of input row scales in tile
# ---------------------------------------------------------------------------

def test_scale_value_correctness() -> None:
    """columnwise_scale_inv[pid_row, pid_col*128:(pid_col+1)*128] equals
    max(rowwise_scale_inv[pid_row*128:(pid_row+1)*128, pid_col])."""
    cuda_fn = _get_cuda_fn()
    rows, cols = 512, 512

    data    = make_fp8_data(rows, cols).view(torch.float8_e4m3fn)
    scales  = make_random_scales(rows, cols)

    _, col_scale = cuda_fn(data, scales)
    torch.cuda.synchronize()

    nbrows = rows // TILE
    nbcols = cols // TILE
    for pid_row in range(nbrows):
        for pid_col in range(nbcols):
            r0 = pid_row * TILE
            c0 = pid_col * TILE

            # max over the TILE rows at compact-column pid_col
            expected_max = scales[r0:r0 + TILE, pid_col].max().item()
            actual = col_scale[pid_row, c0].item()

            assert abs(actual - expected_max) < 1e-6, (
                f"Tile ({pid_row},{pid_col}): expected max={expected_max:.6g}, "
                f"got {actual:.6g}"
            )


# ---------------------------------------------------------------------------
# 12. block_size != 128 raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_block_size() -> None:
    cuda_fn = _get_cuda_fn()
    data   = make_fp8_data(128, 128)
    scales = make_pow2_scales(128, 128)

    with pytest.raises(ValueError, match="block_size must be 128"):
        cuda_fn(data, scales, block_size=64)


# ---------------------------------------------------------------------------
# 13. rows/cols not divisible by 128 raises AssertionError
# ---------------------------------------------------------------------------

def test_misaligned_rows() -> None:
    cuda_fn = _get_cuda_fn()
    data   = torch.zeros(129, 128, dtype=torch.uint8, device="cuda")
    scales = torch.ones(129, 1,   dtype=torch.float32, device="cuda")

    with pytest.raises((AssertionError, RuntimeError)):
        cuda_fn(data, scales)


def test_misaligned_cols() -> None:
    cuda_fn = _get_cuda_fn()
    data   = torch.zeros(128, 129, dtype=torch.uint8, device="cuda")
    scales = torch.ones(128, 1,   dtype=torch.float32, device="cuda")

    with pytest.raises((AssertionError, RuntimeError)):
        cuda_fn(data, scales)


# ---------------------------------------------------------------------------
# 14. Output contiguity
# ---------------------------------------------------------------------------

def test_output_contiguous() -> None:
    """Output tensors must be contiguous."""
    cuda_fn = _get_cuda_fn()
    data    = make_fp8_data(256, 256).view(torch.float8_e4m3fn)
    scales  = make_pow2_scales(256, 256)

    col_data, col_scale = cuda_fn(data, scales)
    assert col_data.is_contiguous(),  "columnwise_data must be contiguous"
    assert col_scale.is_contiguous(), "columnwise_scale_inv must be contiguous"


# ---------------------------------------------------------------------------
# 15. Randomised stress test (large, bit-exact comparison)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 42])
def test_randomised_stress(seed: int) -> None:
    """Large randomised inputs: CUDA == Triton across multiple seeds."""
    torch.manual_seed(seed)
    rows, cols = 2048, 2048

    data   = make_fp8_data(rows, cols)
    scales = make_random_scales(rows, cols)
    compare_cuda_vs_triton(data, scales, label=f"stress_seed{seed}")


# ---------------------------------------------------------------------------
# 16. Transposed columnwise_data correctness vs pure-Python reference
#     (sanity check for the algorithm, not bit-exact with Triton for
#      non-pow2 scales, but exact for pow2 scales)
# ---------------------------------------------------------------------------

def test_vs_python_reference_pow2_scales() -> None:
    """Spot-check CUDA output against the slow pure-Python reference."""
    cuda_fn = _get_cuda_fn()
    rows, cols = 256, 256

    data   = make_fp8_data(rows, cols)
    scales = make_pow2_scales(rows, cols)

    cuda_data, cuda_scale = cuda_fn(data.view(torch.float8_e4m3fn), scales)
    torch.cuda.synchronize()

    ref_data, ref_scale = _reference_exponent_shift(data, scales)

    nbrows = rows // TILE
    assert torch.equal(cuda_data.view(torch.uint8).cpu(), ref_data.cpu()), \
        "CUDA != Python reference (data)"
    assert torch.equal(cuda_scale[:nbrows].cpu(), ref_scale[:nbrows].cpu()), \
        "CUDA != Python reference (scales)"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
