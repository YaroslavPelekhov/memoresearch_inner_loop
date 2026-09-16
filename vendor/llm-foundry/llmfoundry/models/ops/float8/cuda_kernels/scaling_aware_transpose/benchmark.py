"""Microbenchmark: CUDA scaling-aware FP8 transpose vs Triton reference.

Reports kernel time (ms), effective bandwidth (GB/s), and speedup.
Also compares against the full dequant/requant CUDA kernel (row2col_dq)
if available, to quantify the benefit of the integer-only fast path.

Run
---
    cd llmfoundry/models/ops/float8/cuda_kernels/scaling_aware_transpose
    python benchmark.py
"""

from __future__ import annotations

import sys
import os
import types as _types
import pathlib as _pathlib


def _setup_standalone_path() -> None:
    """Stub llmfoundry/llmfoundry.models for standalone execution."""
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

import torch

TILE    = 128
WARMUP  = 20
ITERS   = 100


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_inputs(rows: int, cols: int, device: str = "cuda"):
    """Return (fp8_data, scale_inv) for benchmarking."""
    x      = torch.randn(rows, cols, device=device)
    data   = x.to(torch.float8_e4m3fn)
    scales = torch.exp2(
        torch.randint(-10, 11, (rows, cols // TILE),
                      dtype=torch.float32, device=device)
    )
    return data, scales


def bytes_moved(rows: int, cols: int) -> int:
    """Minimum bytes read + written (roofline lower bound)."""
    rsi_cols = cols // TILE
    nbrows   = rows // TILE
    return (
        rows * cols              # read FP8 data
        + rows * rsi_cols * 4    # read float32 row scales
        + cols * rows            # write FP8 data (transposed)
        + nbrows * cols * 4      # write float32 col scales
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median kernel time in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_cuda(data, scales, warmup: int = WARMUP, iters: int = ITERS) -> float:
    from llmfoundry.models.ops.float8.cuda_kernels.scaling_aware_transpose import (
        blockwise_scaling_aware_fp8_transpose,
    )
    return _bench(blockwise_scaling_aware_fp8_transpose,
                  data, scales, warmup=warmup, iters=iters)


def bench_triton(data, scales, warmup: int = WARMUP, iters: int = ITERS) -> float:
    try:
        from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import (
            blockwise_scaling_aware_fp8_transpose,
        )
    except ImportError:
        return None
    # Pass as uint8 so Triton sees an integer-typed pointer; avoids the
    # int→fp8 cast error introduced in newer Triton versions (other=0 in tl.load).
    data_u8 = data.view(torch.uint8)
    return _bench(blockwise_scaling_aware_fp8_transpose,
                  data_u8, scales, warmup=warmup, iters=iters)


def bench_cuda_dq(data, scales, warmup: int = WARMUP, iters: int = ITERS):
    """Benchmark the full dequant/requant CUDA kernel for reference.

    This kernel does exact FP8↔float32 conversion so it has more compute,
    but also produces exact columnwise scales.  Useful to quantify the
    integer-only fast path speedup.
    """
    try:
        from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import (
            rowwise_to_columnwise_inplace,
        )
    except ImportError:
        return None

    # rowwise_to_columnwise_inplace takes (tensor_uint8, scale_inv)
    data_u8    = data.view(torch.uint8)
    rows, cols = data_u8.shape
    rsi_cols   = cols // TILE
    scales_in  = scales if scales.shape == (rows, rsi_cols) else scales

    def _run():
        rowwise_to_columnwise_inplace(data_u8, scales_in)

    return _bench(_run, warmup=warmup, iters=iters)


# ---------------------------------------------------------------------------
# Benchmark shapes  (representative LLM hidden sizes)
# ---------------------------------------------------------------------------

BENCH_SHAPES = [
    # (rows, cols)
    (128,    128),      # minimal
    (256,    256),
    (512,    512),
    (1024,   512),
    (1024,  1024),
    (2048,  1024),
    (1024,  2048),
    (2048,  2048),
    (4096,  2048),
    (2048,  4096),
    (4096,  4096),
    (4096,  7168),
    (7168,  4096),
    (8192,  4096),
    (4096,  8192),
    (4096, 12288),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda"
    print(f"Device : {torch.cuda.get_device_name()}")
    print(f"Warmup : {WARMUP}  Iters : {ITERS}")
    print()

    col_w    = 10
    hdr_parts = [
        f"{'rows':>6}", f"{'cols':>6}",
        f"{'CUDA ms':>{col_w}}", f"{'CUDA GB/s':>{col_w+2}}",
        f"{'Triton ms':>{col_w}}", f"{'vs Triton':>{col_w}}",
        f"{'DQ-inplace ms':>14}",
    ]
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))

    for rows, cols in BENCH_SHAPES:
        data, scales = make_inputs(rows, cols, device)
        nb = bytes_moved(rows, cols)

        t_cuda   = bench_cuda(data, scales)
        t_triton = bench_triton(data, scales)
        t_dq     = bench_cuda_dq(data, scales)

        bw_cuda = nb / (t_cuda * 1e-3) / 1e9

        triton_str   = f"{t_triton:.3f}"  if t_triton is not None else "  N/A"
        speedup_str  = f"{t_triton / t_cuda:.2f}x" if t_triton is not None else "  N/A"
        dq_str       = f"{t_dq:.3f}" if t_dq is not None else "  N/A"

        parts = [
            f"{rows:>6}", f"{cols:>6}",
            f"{t_cuda:>{col_w}.3f}", f"{bw_cuda:>{col_w+2}.1f}",
            f"{triton_str:>{col_w}}", f"{speedup_str:>{col_w}}",
            f"{dq_str:>14}",
        ]
        print("  ".join(parts))


if __name__ == "__main__":
    main()
