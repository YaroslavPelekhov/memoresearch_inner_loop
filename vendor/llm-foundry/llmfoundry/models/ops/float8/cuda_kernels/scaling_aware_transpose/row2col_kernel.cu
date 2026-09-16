/**
 * Scaling-aware FP8 transpose kernel (integer-only, no FP8 decode/encode).
 *
 * Converts a row-wise quantized FP8 tensor [rows, cols] to a column-wise
 * layout [cols, rows] using an exponent-shift approximation that never
 * decodes or encodes FP8 values.  Semantics match the Triton reference
 * ``_scaling_aware_fp8_transpose_kernel`` in row2col_dense_kernels.py
 * bit-for-bit on all valid inputs.
 *
 * -------------------------------------------------------------------------
 * Algorithm (per 128×128 tile (pid_row, pid_col))
 * -------------------------------------------------------------------------
 *   1. Load FP8 data tile [TILE, TILE] into smem_in.
 *   2. Load per-row scale_inv values (one float32 per row).
 *      IMPORTANT: scale column index is pid_col, NOT pid_col * TILE.
 *      rowwise_scale_inv is in "compact-column" layout where column c_block
 *      covers data columns [c_block*128, (c_block+1)*128).
 *   3. Compute target_si = max(scale_inv[0..127]) via warp shuffle +
 *      cross-warp shared-memory reduction.
 *   4. For each FP8 byte in the tile:
 *        sign    = bit 7
 *        exp_fp8 = bits [6:3]  (E4M3 biased exponent)
 *        mant    = bits [2:0]
 *        k       = ieee_exp(target_si) − ieee_exp(si_row)   [always >= 0]
 *        exp_new = exp_fp8 − k
 *        result  = 0x00 if (exp_new <= 0) OR (exp_fp8 == 0)
 *                  else (sign<<7) | (exp_new<<3) | mant
 *   5. Write result transposed to columnwise_data[col, row].
 *   6. Write target_si to columnwise_scale_inv[pid_row, c] for all c in
 *      the TILE-wide column range [pid_col*TILE, (pid_col+1)*TILE).
 *
 * -------------------------------------------------------------------------
 * Thread / block layout
 * -------------------------------------------------------------------------
 *   Grid  : (rows / TILE,  cols / TILE)
 *   Block : THREADS = 128 threads = 4 warps
 *   Thread tid (0..127) owns row (pid_row * TILE + tid) of the input.
 *
 * -------------------------------------------------------------------------
 * Shared memory
 * -------------------------------------------------------------------------
 *   smem_in / smem_out : TILE × SMEM_STRIDE bytes  (SMEM_STRIDE = 132)
 *   SMEM_STRIDE / 4 = 33 is coprime with 32 → bank-conflict-free for the
 *   32-bit (4-byte) access pattern used in Phase 1, Phase 3 (reads), and
 *   Phase 4.  Phase 3 byte-scatter writes to smem_out incur a 4-way bank
 *   conflict, which is negligible for a DRAM-bandwidth-bound kernel.
 *
 * -------------------------------------------------------------------------
 * Performance
 * -------------------------------------------------------------------------
 *   - Vectorized uint32 global loads (fast path: stride_c == 1).
 *   - Scalar byte loads for non-contiguous layouts (fallback).
 *   - Vectorized uint4 reads from smem_out (bank-conflict-free, proved in
 *     the comment above Phase 4).
 *   - Vectorized uint4 global stores to columnwise_data.
 *   - 128 coalesced float32 stores for columnwise_scale_inv.
 *
 * -------------------------------------------------------------------------
 * Alignment guarantees (required for vectorized paths)
 * -------------------------------------------------------------------------
 *   - X rows: contiguous tensors from PyTorch are aligned to >= 512 bytes;
 *     row start = base + r * cols where cols % TILE == 0 → 4-byte aligned.
 *   - Y columns: addr = (c) * rows + pid_row*TILE + t.  rows % TILE == 0
 *     and t = (lane&7)*16 both divisible by 16 → 16-byte aligned for uint4.
 *   - smem_out[h][t]: __align__(128) + h*132 + t, t divisible by 16 → 16B.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int TILE        = 128;
static constexpr int THREADS     = 128;              // 4 warps; thread tid owns row tid
static constexpr int SMEM_PAD    = 4;
static constexpr int SMEM_STRIDE = TILE + SMEM_PAD;  // 132; SMEM_STRIDE/4=33 coprime with 32

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(THREADS)
scaling_aware_fp8_transpose_kernel(
    const uint8_t * __restrict__ X,       // [rows, cols]  uint8, strides (stride_r, stride_c)
    const float   * __restrict__ S_in,    // [rows, rsi_cols]  float32, strides (stride_sr, stride_sc)
    uint8_t       * __restrict__ Y,       // [cols, rows]  uint8 contiguous (stride = (rows, 1))
    float         * __restrict__ S_out,   // [nbrows_mult_4, cols] float32 contiguous (stride = (cols, 1))
    int     rows,
    int     cols,
    int     rsi_cols,
    int64_t stride_r,     // rowwise_data row stride (elements)
    int64_t stride_c,     // rowwise_data col stride (elements)
    int64_t stride_sr,    // scale_inv row stride (elements)
    int64_t stride_sc     // scale_inv col stride (elements)
)
{
    const int pid_row = blockIdx.x;
    const int pid_col = blockIdx.y;
    const int tid     = threadIdx.x;   // 0 .. THREADS-1 = 127

    const int warp_id = tid >> 5;      // 0..3
    const int lane_id = tid & 31;      // 0..31

    // -------------------------------------------------------------------
    // Shared memory
    // smem_in[row_in_tile][col_in_tile]  - input tile, row-major within tile
    // smem_out[col_in_tile][row_in_tile] - transposed output
    // smem_warp_max[4]                   - one warp-max per warp
    // -------------------------------------------------------------------
    __shared__ __align__(128) uint8_t smem_in [TILE][SMEM_STRIDE];
    __shared__ __align__(128) uint8_t smem_out[TILE][SMEM_STRIDE];
    __shared__ __align__(16)  float   smem_warp_max[4];

    // Global row index owned by this thread
    const int r_global = pid_row * TILE + tid;

    // ====================================================================
    // Phase 1: Global → smem_in  +  load per-row scale_inv
    // ====================================================================

    // Scale column = pid_col (compact-column index), NOT pid_col * TILE.
    // For out-of-bounds threads (r_global >= rows or pid_col >= rsi_cols)
    // use 0.0f; since all valid scales are positive, this doesn't affect max.
    float si = 0.0f;
    if (r_global < rows && pid_col < rsi_cols) {
        si = S_in[(int64_t)r_global * stride_sr
                + (int64_t)pid_col  * stride_sc];
    }

    // Load TILE bytes of the row into smem_in[tid][0..TILE-1].
    if (r_global < rows) {
        if (stride_c == 1) {
            // Fast path: row-contiguous data.
            // Vectorised uint32 loads (32 × 4 bytes = 128 bytes per thread).
            // Alignment: PyTorch guarantees >= 512-byte tensor alignment;
            // row start = base + r * cols, cols % TILE == 0 → 4-byte aligned.
            const uint8_t *row_ptr =
                X + (int64_t)r_global * stride_r + (int64_t)pid_col * TILE;
            const uint32_t *src = reinterpret_cast<const uint32_t *>(row_ptr);
                  uint32_t *dst = reinterpret_cast<      uint32_t *>(smem_in[tid]);
            #pragma unroll
            for (int i = 0; i < TILE / 4; ++i)
                dst[i] = src[i];
        } else {
            // Slow path: arbitrary column stride — scalar byte loads.
            const int c_base = pid_col * TILE;
            for (int c = 0; c < TILE; ++c)
                smem_in[tid][c] = X[(int64_t)r_global * stride_r
                                  + (int64_t)(c_base + c) * stride_c];
        }
    } else {
        // Zero-pad out-of-bounds row (guard; caller asserts rows % TILE == 0
        // so this branch is never taken in practice).
        uint32_t *dst = reinterpret_cast<uint32_t *>(smem_in[tid]);
        #pragma unroll
        for (int i = 0; i < TILE / 4; ++i)
            dst[i] = 0u;
    }

    __syncthreads();  // smem_in fully written before Phase 2 reads

    // ====================================================================
    // Phase 2: Compute target_si = max over all TILE row scales
    //          via warp-level butterfly + cross-warp shared-memory reduce.
    // ====================================================================

    // Step 1: Warp-level butterfly (5 shuffle steps, all 32 lanes contribute).
    float warp_max = si;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, off));

    // Step 2: Lane 0 of each warp writes its warp maximum to smem.
    if (lane_id == 0)
        smem_warp_max[warp_id] = warp_max;

    __syncthreads();  // all 4 warp maxima visible to all threads

    // Step 3: All threads read 4 warp maxima and take the final max.
    //         (3 comparisons — cheaper than another reduction round.)
    float target_si = smem_warp_max[0];
    #pragma unroll
    for (int i = 1; i < 4; ++i)
        target_si = fmaxf(target_si, smem_warp_max[i]);

    // Extract unbiased IEEE 754 exponents via bit-reinterpretation.
    // Using __float_as_uint avoids any --use_fast_math interference.
    // Both target_si and si are normal positive floats (quantization scales),
    // so the exponent field [30:23] is in [1, 254] and the unbiased value
    // is in [-126, 127].
    const int32_t exp_t = static_cast<int32_t>(
        (__float_as_uint(target_si) & 0x7F800000u) >> 23) - 127;
    const int32_t exp_s = static_cast<int32_t>(
        (__float_as_uint(si)        & 0x7F800000u) >> 23) - 127;

    // k >= 0 because target_si = max(si) implies exp_t >= exp_s.
    // Stored as signed int32 so that (exp_fp8 - k) can safely go negative;
    // the underflow check below catches and flushes those cases to zero.
    const int32_t k = exp_t - exp_s;

    // ====================================================================
    // Phase 3: Exponent-shift FP8 bytes + scatter-transpose into smem_out
    //
    // For each byte b in smem_in[tid][0..TILE-1]:
    //   sign    = bit 7
    //   exp_fp8 = bits [6:3]  (FP8 E4M3 biased exponent)
    //   mant    = bits [2:0]
    //   exp_new = exp_fp8 - k    (signed arithmetic; may be <= 0)
    //   under   = (exp_new <= 0) | (exp_fp8 == 0)   [matches Triton exactly]
    //   result  = 0x00 if under, else (sign<<7) | (exp_new<<3) | mant
    //
    // Written transposed: smem_out[col_in_tile][row_in_tile = tid].
    //
    // Bank-conflict note: 32 threads in a warp each write a byte to
    // smem_out[c][warp_base..warp_base+31] (32 consecutive bytes).
    // This is a 4-way conflict (4 threads per 4-byte bank).  Acceptable
    // because the kernel is DRAM-bandwidth bound and smem is ~50x faster.
    // ====================================================================

    #pragma unroll
    for (int ci = 0; ci < TILE / 4; ++ci) {
        // Load 4 consecutive FP8 bytes as uint32.
        // smem_in[tid][ci*4] bank: (tid*33 + ci) % 32 — different for each
        // lane (since 33 is coprime with 32) → zero bank conflicts here.
        const uint32_t in_chunk =
            reinterpret_cast<const uint32_t *>(smem_in[tid])[ci];

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int     c       = ci * 4 + j;
            const uint8_t b       = static_cast<uint8_t>(in_chunk >> (j * 8));

            // Decompose FP8 E4M3 byte fields
            const int32_t sign    = (b >> 7) & 1;
            const int32_t exp_fp8 = (b >> 3) & 0xF;
            const int32_t mant    = b & 0x7;

            // Signed exponent shift (allows negative values for under-check)
            const int32_t exp_new = exp_fp8 - k;

            // Flush-to-zero conditions (identical to Triton):
            //   exp_new <= 0 : underflow after scale adjustment
            //   exp_fp8 == 0 : original byte is zero or subnormal — keep as zero
            const bool under = (exp_new <= 0) | (exp_fp8 == 0);

            smem_out[c][tid] = under
                ? static_cast<uint8_t>(0)
                : static_cast<uint8_t>(
                      (sign    << 7)
                    | (exp_new << 3)
                    | mant);
        }
    }

    // Write target_si to all TILE output-column positions in this tile-row.
    // columnwise_scale_inv[pid_row, pid_col*TILE + tid] — 512-byte coalesced
    // float32 write (128 threads × 4 bytes = 512 bytes in one cache line group).
    // This mirrors the Triton store:
    //   tl.store(ptrs + pid_row * cols + c_offsets, target_si, mask=valid_c)
    // which broadcasts the scalar target_si to all TILE column positions.
    const int c_out = pid_col * TILE + tid;
    if (c_out < cols) {   // guard; always true when cols % TILE == 0
        S_out[(int64_t)pid_row * cols + c_out] = target_si;
    }

    __syncthreads();  // smem_out fully written before Phase 4 reads

    // ====================================================================
    // Phase 4: smem_out → global  (vectorized uint4 stores)
    //
    // Mirrors the Phase 4 pattern of row2col_dq_kernel / row2col_grouped_kernel.
    // 8 iterations × 128 threads × 16 bytes = 16 384 bytes = full 128×128 tile.
    //
    // Thread decomposition (within a warp of 32 lanes):
    //   h = (lane >> 3) + it * 16   (col-in-tile index, = output row, 0..127)
    //   t = (lane  & 7) << 4        (row-in-tile byte offset: 0,16,...,112)
    //
    // Bank conflict analysis for smem_out uint32 reads:
    //   bank of smem_out[h][t] = (h * 33 + t/4) % 32
    //   For lane l: h = (l>>3)+base_h, t/4 = (l&7)*4.
    //   bank = ((l>>3)*33 + (l&7)*4 + base_h*33) % 32
    //        = (l*[33-4] div + (l&7)*[4] + l>>3*33 — simplified below:
    //   For it=0, lane 0..31: banks = {0,4,8,12,16,20,24,28,
    //                                   1,5,9,13,17,21,25,29,
    //                                   2,6,10,14,18,22,26,30,
    //                                   3,7,11,15,19,23,27,31}
    //   All 32 banks are distinct → zero bank conflicts. ✓
    //   For other it values, base_h*33 shifts banks by (it*16*33)%32 — still
    //   a permutation of all 32 banks.
    //
    // Global store address:
    //   Y[(pid_col*TILE + h) * rows + pid_row*TILE + t]
    // where pid_row*TILE and t are both multiples of 16 → 16-byte aligned
    // for uint4 stores (when rows % TILE == 0, since rows % 16 == 0).
    // Consecutive t values within the same h hit adjacent 16-byte chunks
    // of the same output row → near-coalesced (8 threads = 1 cache line).
    // ====================================================================

    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int h = (tid >> 3) + it * 16;  // col-in-tile (0..127 across 8 iters)
        const int t = (tid &  7) << 4;       // row-in-tile byte offset (0,16,..,112)

        uint4 gv;
        gv.x = *reinterpret_cast<const uint32_t *>(&smem_out[h][t     ]);
        gv.y = *reinterpret_cast<const uint32_t *>(&smem_out[h][t +  4]);
        gv.z = *reinterpret_cast<const uint32_t *>(&smem_out[h][t +  8]);
        gv.w = *reinterpret_cast<const uint32_t *>(&smem_out[h][t + 12]);

        // columnwise_data[(pid_col*TILE + h) * rows + pid_row*TILE + t]
        const int64_t addr =
            (int64_t)(pid_col * TILE + h) * rows
            + (int64_t)pid_row * TILE + t;

        *reinterpret_cast<uint4 *>(Y + addr) = gv;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

static void launch_scaling_aware_fp8_transpose(
    torch::Tensor X,
    torch::Tensor S_in,
    torch::Tensor Y,
    torch::Tensor S_out,
    int rows, int cols, int rsi_cols)
{
    TORCH_CHECK(rows % TILE == 0,
        "rows must be a multiple of ", TILE, ", got ", rows);
    TORCH_CHECK(cols % TILE == 0,
        "cols must be a multiple of ", TILE, ", got ", cols);
    TORCH_CHECK(X.dtype()     == torch::kUInt8,   "X must be uint8");
    TORCH_CHECK(S_in.dtype()  == torch::kFloat32, "S_in must be float32");
    TORCH_CHECK(Y.dtype()     == torch::kUInt8,   "Y must be uint8");
    TORCH_CHECK(S_out.dtype() == torch::kFloat32, "S_out must be float32");
    TORCH_CHECK(X.is_cuda() && S_in.is_cuda() && Y.is_cuda() && S_out.is_cuda(),
        "All tensors must be on CUDA");
    TORCH_CHECK(Y.is_contiguous(),    "Y must be contiguous");
    TORCH_CHECK(S_out.is_contiguous(), "S_out must be contiguous");

    const dim3 grid(rows / TILE, cols / TILE);
    const dim3 block(THREADS);

    auto stream = at::cuda::getCurrentCUDAStream();

    scaling_aware_fp8_transpose_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t *>(X.data_ptr()),
        S_in.data_ptr<float>(),
        static_cast<uint8_t *>(Y.data_ptr()),
        S_out.data_ptr<float>(),
        rows, cols, rsi_cols,
        X.stride(0),    X.stride(1),
        S_in.stride(0), S_in.stride(1));

    C10_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// pybind11
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "scaling_aware_fp8_transpose",
        &launch_scaling_aware_fp8_transpose,
        "FP8 exponent-shift transpose: [rows,cols] row-wise → [cols,rows] col-wise "
        "(integer-only, no FP8 decode/encode).",
        py::arg("X"),
        py::arg("S_in"),
        py::arg("Y"),
        py::arg("S_out"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("rsi_cols"));
}
