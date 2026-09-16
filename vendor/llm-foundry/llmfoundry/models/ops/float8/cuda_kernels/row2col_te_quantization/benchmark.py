"""Benchmark: CUDA v5 row→col kernel vs Triton baseline.

Reports kernel time (ms), effective bandwidth (GB/s), and speedup.

Run::

    python benchmark_row2col.py
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

import torch

TILE = 128
WARMUP = 10
ITERS = 50


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
    scales = torch.rand(T, H // TILE, dtype=torch.float32, device=device) * 0.1 + 0.01
    m_indices = make_m_indices(group_sizes, device)
    return data, scales, m_indices


def bytes_moved(T, H):
    """Total bytes read + written by the kernel."""
    read_data = T * H                  # FP8 input
    read_scales = T * (H // TILE) * 4  # float32 rowwise scales
    write_data = T * H                 # FP8 output (transposed)
    write_scales = (T // TILE) * H * 4 # float32 colwise scales
    return read_data + read_scales + write_data + write_scales


def bench_cuda_v5(data, scales, gs, m_idx, warmup=WARMUP, iters=ITERS):
    """Benchmark the two-phase CUDA v5 API (prepare + execute_cached)."""
    from llmfoundry.models.ops.fp8.triton_kernels.cuda_row2col import (
        row2col_prepare, row2col_execute_cached, _single_cache,
    )
    _single_cache.pop("bench", None)

    # Cold run to populate cache
    prep = row2col_prepare(data, scales, gs, m_idx)
    _ = row2col_execute_cached(prep, cache_slot="bench")
    torch.cuda.synchronize()

    # Warmup (cached path)
    for _ in range(warmup):
        prep = row2col_prepare(data, scales, gs, m_idx)
        _ = row2col_execute_cached(prep, cache_slot="bench")
    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        prep = row2col_prepare(data, scales, gs, m_idx)
        _ = row2col_execute_cached(prep, cache_slot="bench")
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_triton(data, scales, gs, m_idx, warmup=WARMUP, iters=ITERS):
    """Benchmark the Triton row2col_grouped kernel."""
    try:
        from llmfoundry.models.ops.fp8.triton_kernels.row2col_grouped import (
            row2col_grouped,
        )
    except ImportError:
        return None

    data_fp8 = data.view(torch.float8_e4m3fn)
    gs_tensor = torch.tensor(gs, dtype=torch.int64, device=data.device)

    # Warmup
    for _ in range(warmup):
        _ = row2col_grouped(data_fp8, scales, gs, m_idx, gs_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = row2col_grouped(data_fp8, scales, gs, m_idx, gs_tensor)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


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
    hdr = f"{'T':>6} {'H':>6} {'G':>3}  {'CUDA ms':>8} {'Triton ms':>10} {'CUDA GB/s':>10} {'Triton GB/s':>12} {'Speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    for T, H, gs in BENCH_SHAPES:
        data, scales, m_idx = make_inputs(T, H, gs, device)
        n_bytes = bytes_moved(T, H)

        t_cuda = bench_cuda_v5(data, scales, gs, m_idx)
        t_triton = bench_triton(data, scales, gs, m_idx)

        bw_cuda = n_bytes / (t_cuda * 1e-3) / 1e9
        bw_triton = (n_bytes / (t_triton * 1e-3) / 1e9) if t_triton else 0

        triton_str = f"{t_triton:.3f}" if t_triton else "N/A"
        bw_triton_str = f"{bw_triton:.1f}" if t_triton else "N/A"
        speedup_str = f"{t_triton / t_cuda:.2f}x" if t_triton else "N/A"

        print(
            f"{T:>6} {H:>6} {len(gs):>3}  "
            f"{t_cuda:>8.3f} {triton_str:>10} "
            f"{bw_cuda:>10.1f} {bw_triton_str:>12} "
            f"{speedup_str:>8}"
        )


if __name__ == "__main__":
    main()
