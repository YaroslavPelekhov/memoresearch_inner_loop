"""Benchmark: CUDA row→col DQ kernel vs Triton deepgemm baseline.

Reports kernel time (ms), effective bandwidth (GB/s), and speedup.

Run::

    python benchmark.py
"""

from __future__ import annotations

import sys
import os
import types as _types
import pathlib as _pathlib


def _setup_standalone_path() -> None:
    """Add repo root to sys.path for standalone execution."""
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

TILE = 128
WARMUP = 10
ITERS = 50


# ---------------------------------------------------------------------------
# Data generation
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
    data = torch.randint(0, 255, (T, H), dtype=torch.uint8, device=device)
    scales = (
        torch.rand(T, H // TILE, dtype=torch.float32, device=device) * 0.1
        + 0.01
    )
    m_indices = make_m_indices(group_sizes, device)
    return data, scales, m_indices


def bytes_moved(T, H):
    """Total bytes read + written (lower bound for effective bandwidth)."""
    read_data   = T * H                    # FP8 input
    read_scales = T * (H // TILE) * 4      # float32 rowwise scales
    write_data  = T * H                    # FP8 output (transposed)
    write_scales = (T // TILE) * H * 4     # float32 colwise scales
    return read_data + read_scales + write_data + write_scales


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_cuda_dq(data, scales, gs, m_idx, warmup=WARMUP, iters=ITERS):
    """Benchmark the standalone CUDA kernel (this module)."""
    from llmfoundry.models.ops.float8.cuda_kernels.row2col_dq_quantization import (
        row2col_requantize,
    )
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)

    for _ in range(warmup):
        row2col_requantize(data, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        row2col_requantize(data, m_idx, gs_tensor, scales)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_triton_deepgemm(data, scales, gs, m_idx, warmup=WARMUP, iters=ITERS):
    """Benchmark the Triton deepgemm-layout kernel."""
    try:
        from llmfoundry.models.ops.float8.triton_kernels.row2col_deepgemm_quantization import (
            row2col_requantization_deepgemm_layout_fn,
        )
    except ImportError:
        return None

    data_fp8 = data.view(torch.float8_e4m3fn)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)

    for _ in range(warmup):
        row2col_requantization_deepgemm_layout_fn(
            data_fp8, m_idx, gs_tensor, scales)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        row2col_requantization_deepgemm_layout_fn(
            data_fp8, m_idx, gs_tensor, scales)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_cuda_te(data, scales, gs, m_idx, warmup=WARMUP, iters=ITERS):
    """Benchmark the CUDA TE kernel (if available)."""
    try:
        from llmfoundry.models.ops.float8.cuda_kernels.row2col_te_quantization import (
            row2col_prepare, row2col_execute_cached, _single_cache,
        )
    except ImportError:
        return None

    _single_cache.pop("bench_te", None)

    prep = row2col_prepare(data, scales, gs, m_idx)
    _ = row2col_execute_cached(prep, cache_slot="bench_te")
    torch.cuda.synchronize()

    for _ in range(warmup):
        prep = row2col_prepare(data, scales, gs, m_idx)
        _ = row2col_execute_cached(prep, cache_slot="bench_te")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        prep = row2col_prepare(data, scales, gs, m_idx)
        _ = row2col_execute_cached(prep, cache_slot="bench_te")
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Benchmark shapes
# ---------------------------------------------------------------------------

BENCH_SHAPES = [
    (512,   512,  [256, 256]),
    (1024,  512,  [256] * 4),
    (1024,  1024, [256] * 4),
    (2048,  1024, [256] * 8),
    (2048,  2048, [256] * 8),
    (4096,  2048, [512] * 8),
    (4096,  4096, [512] * 8),
    (4096,  4096, [4096]),
    (4096,  4096, [128] * 32),
    (8192,  4096, [1024] * 8),
]


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP}  Iters: {ITERS}")
    print()

    has_triton = True
    has_te = True

    hdr_parts = [f"{'T':>6}", f"{'H':>6}", f"{'G':>3}",
                 f"{'CUDA-DQ ms':>11}", f"{'CUDA-DQ GB/s':>13}"]
    if has_triton:
        hdr_parts += [f"{'Triton ms':>10}", f"{'vs Triton':>10}"]
    if has_te:
        hdr_parts += [f"{'CUDA-TE ms':>11}", f"{'vs TE':>8}"]

    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))

    for T, H, gs in BENCH_SHAPES:
        data, scales, m_idx = make_inputs(T, H, gs, device)
        n_bytes = bytes_moved(T, H)

        t_dq = bench_cuda_dq(data, scales, gs, m_idx)
        bw_dq = n_bytes / (t_dq * 1e-3) / 1e9

        parts = [
            f"{T:>6}", f"{H:>6}", f"{len(gs):>3}",
            f"{t_dq:>11.3f}", f"{bw_dq:>13.1f}",
        ]

        if has_triton:
            t_tr = bench_triton_deepgemm(data, scales, gs, m_idx)
            if t_tr is None:
                has_triton = False
                parts += [f"{'N/A':>10}", f"{'N/A':>10}"]
            else:
                parts += [f"{t_tr:>10.3f}", f"{t_tr / t_dq:>9.2f}x"]

        if has_te:
            t_te = bench_cuda_te(data, scales, gs, m_idx)
            if t_te is None:
                has_te = False
                parts += [f"{'N/A':>11}", f"{'N/A':>8}"]
            else:
                parts += [f"{t_te:>11.3f}", f"{t_te / t_dq:>7.2f}x"]

        print("  ".join(parts))


if __name__ == "__main__":
    main()
