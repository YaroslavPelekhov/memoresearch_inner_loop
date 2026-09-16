# CUDA Kernels

## `row2col_te_quantization/`

### `row2col_kernel.cu`

**Kernel:** `row2col_grouped_kernel`
- **Input:** `X (T, H)` fp8 row-major, `S_in (T, H // 128)` row-wise scales, `group_offsets (G+1,)` int64 prefix sums, `m_indices (T,)` int32 group id per row
- **Output:** `Y_flat (H * T,)` fp8 column-wise grouped, `S_out (T // 128, H)` column-wise scales
- **Grid:** `(T // 128, H // 128)`, **Block:** 256 threads (8 warps), **Tile:** 128x128

Fused row-wise FP8 to grouped column-wise FP8 conversion optimized for Hopper (SM90+). Dequantizes with row-wise scales, computes per-column amax via warp-level `__shfl_xor` reduction, requantizes with power-of-2-ceiled column-wise scales, and transposes — all in a single kernel using vectorized 128-bit loads/stores and bank-conflict-free shared memory (stride=132 padding).

**C++ API (pybind11):**
- `row2col_alloc(X)` — pre-allocates output buffers `(Y, S)` for a given input shape.
- `row2col_cached_prealloc(X, S_in, group_offsets, m_indices, group_sizes, cached, Y, S)` — cache-hit path: updates existing QTensor attributes in-place and launches the kernel asynchronously with no CUDA malloc.
- `row2col_wrapped_prealloc(X, S_in, group_offsets, m_indices, group_sizes, qtensor_cls, fp8_dtype, torch_dtype, Y, S)` — cache-miss path: builds per-group `Float8BlockwiseQTensor` views via `as_strided` and launches the kernel.

### `__init__.py`

Python wrapper exposing a two-phase API for the CUDA kernel:
- `row2col_prepare(data, scales, list_groups_sizes, m_indices)` — Phase 1 (blocking): converts to uint8/float8 view, computes group offset prefix sums, and pre-allocates CUDA output buffers.
- `row2col_execute_cached(prep, cache_slot)` — Phase 2 (non-blocking): performs only `as_strided` metadata ops and async kernel launch with no CUDA malloc; caches QTensor shells per group-size signature.

### `setup.py`

Build script for compiling the CUDA extension via `torch.utils.cpp_extension.CUDAExtension` targeting SM90 (`-arch=sm_90`).

### `benchmark.py` / `test_row2col_kernel.py`

Benchmarking and correctness tests for the CUDA row2col kernel.
