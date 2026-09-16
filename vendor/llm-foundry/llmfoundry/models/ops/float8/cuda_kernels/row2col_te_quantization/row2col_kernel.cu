/**
 * Fused rowwise-FP8 → grouped-columnwise-FP8 conversion kernel for Hopper (SM90+).
 *
 * Replaces the Triton row2col_grouped kernel with an optimized CUDA C++
 * implementation that uses:
 *   - Vectorized 128-bit global loads/stores
 *   - Bank-conflict-minimized shared-memory transpose (stride=132 padding)
 *   - Warp-level column-amax reduction via __shfl_xor (no cross-warp reduction)
 *   - <cuda_fp8.h> API for FP8↔float conversion (portable across SM89+/SM100+)
 *   - Precomputed group offsets (eliminates per-CTA prefix-sum scan)
 *
 * Tile: 128×128  (matches FP8 block-scale granularity)
 * CTA:  256 threads = 8 warps
 * Each warp owns 16 columns × 128 rows (32 lanes × 4 rows/lane).
 *
 * Scale convention (TE-compatible):
 *   scale_inv  = ceil_pow2(amax / 448)
 *   quant:   fp8  = f32 / scale_inv
 *   dequant: f32  = fp8 * scale_inv
 */

 #include <cuda_runtime.h>
 #include <cuda_fp16.h>
 #include <cuda_fp8.h>
 #include <cstdint>
 #include <cstring>
 #include <cassert>
 #include <torch/extension.h>
 #include <c10/cuda/CUDAException.h>
 #include <c10/cuda/CUDAStream.h>
 
 // ---------------------------------------------------------------------------
 // Constants
 // ---------------------------------------------------------------------------
 
 static constexpr int TILE        = 128;
 static constexpr int THREADS     = 256;
 // Padding so that SMEM_STRIDE / 4 = 33 is coprime with 32 → bank-conflict-
 // free for the vectorised load/store pattern used in Phases 1 & 4.
 static constexpr int SMEM_PAD    = 4;
 static constexpr int SMEM_STRIDE = TILE + SMEM_PAD;   // 132
 static constexpr float FP8_MAX   = 448.0f;
 
 // ---------------------------------------------------------------------------
 // FP8 ↔ float helpers  (cuda_fp8.h API – uses HW on SM100+, emulation on SM90)
 // ---------------------------------------------------------------------------
 
 __device__ __forceinline__ float fp8_to_f32(uint8_t raw) {
     __half_raw hr = __nv_cvt_fp8_to_halfraw(
         static_cast<__nv_fp8_storage_t>(raw), __NV_E4M3);
     __half h;
     memcpy(&h, &hr, sizeof(h));
     return __half2float(h);
 }
 
 __device__ __forceinline__ uint8_t f32_to_fp8(float v) {
     return static_cast<uint8_t>(
         __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
 }
 
 __device__ __forceinline__ void fp8x4_to_f32x4(uint32_t pk,
                                                 float &v0, float &v1,
                                                 float &v2, float &v3) {
     v0 = fp8_to_f32(static_cast<uint8_t>(pk));
     v1 = fp8_to_f32(static_cast<uint8_t>(pk >> 8));
     v2 = fp8_to_f32(static_cast<uint8_t>(pk >> 16));
     v3 = fp8_to_f32(static_cast<uint8_t>(pk >> 24));
 }
 
 __device__ __forceinline__ uint32_t f32x4_to_fp8x4(float v0, float v1,
                                                     float v2, float v3) {
     return static_cast<uint32_t>(f32_to_fp8(v0))
          | (static_cast<uint32_t>(f32_to_fp8(v1)) <<  8)
          | (static_cast<uint32_t>(f32_to_fp8(v2)) << 16)
          | (static_cast<uint32_t>(f32_to_fp8(v3)) << 24);
 }
 
 // Exact pow2_ceil via bit manipulation – immune to --use_fast_math log2 errors.
 __device__ __forceinline__ float pow2_ceil_positive(float v) {
     unsigned int bits = __float_as_uint(v);
     unsigned int mant = bits & 0x007FFFFFu;
     if (mant == 0u) return v;
     unsigned int exp_field = bits & 0x7F800000u;
     return __uint_as_float(exp_field + 0x00800000u);
 }
 
 // ---------------------------------------------------------------------------
 // Kernel
 // ---------------------------------------------------------------------------
 //
 // Grid : (total_tokens / 128,  hidden_size / 128)
 // Block: 256 threads  =  8 warps
 //
 // Warp w  (w = 0..7)  owns columns  [w*16 .. w*16+15].
 // Lane l  (l = 0..31) owns rows     [l*4  .. l*4+3].
 //
 // Phase 1 – Load 128×128 FP8 tile + 128 row-scales into smem.
 // Phase 2 – Dequant + per-column amax  (warp-level reduction, no cross-warp).
 // Phase 3 – Re-quantise + transpose → smem_out.
 // Phase 4 – Coalesced 128-bit stores from smem_out to grouped global buffer.
 // ---------------------------------------------------------------------------
 
 __global__ void __launch_bounds__(THREADS)
 row2col_grouped_kernel(
     const uint8_t * __restrict__ X,               // [T, H]   FP8 row-major
     const float   * __restrict__ S_in,             // rowwise scale_inv
     const int64_t * __restrict__ group_offsets,     // [G+1]  prefix sum
     const int32_t * __restrict__ m_indices,         // [T]    group id per row
     uint8_t       * __restrict__ Y_flat,            // output flat FP8
     float         * __restrict__ S_out,             // [T/128, H]  col scale_inv
     int hidden_size,
     int total_tokens,
     int64_t s_in_stride_row,
     int64_t s_in_stride_col)
 {
     const int tile_row = blockIdx.x;               // token-block  index
     const int tile_col = blockIdx.y;               // hidden-block index
     const int tid      = threadIdx.x;
 
     const int row_start = tile_row * TILE;
     const int col_start = tile_col * TILE;
 
     const int     gid          = m_indices[row_start];
     const int64_t g_offset     = group_offsets[gid];
     const int     g_size       = (int)(group_offsets[gid + 1] - g_offset);
     const int     tok_in_group = row_start - (int)g_offset;
     const int64_t flat_base    = g_offset * (int64_t)hidden_size;
 
     __shared__ __align__(128) uint8_t smem_in [TILE][SMEM_STRIDE];
     __shared__ __align__(128) uint8_t smem_out[TILE][SMEM_STRIDE];
     __shared__ float smem_scales   [TILE];
     __shared__ float smem_col_scale[TILE];
 
     // ==================================================================
     // Phase 1 :  Global → smem  (vectorised 128-bit loads)
     // ==================================================================
     #pragma unroll
     for (int it = 0; it < 4; ++it) {
         const int r = (tid >> 3) + (it << 5);
         const int c = (tid & 7) << 4;
 
         const uint8_t *src = X + (int64_t)(row_start + r) * hidden_size
                                + col_start + c;
         uint4 gv = *reinterpret_cast<const uint4 *>(src);
         *reinterpret_cast<uint32_t *>(&smem_in[r][c     ]) = gv.x;
         *reinterpret_cast<uint32_t *>(&smem_in[r][c +  4]) = gv.y;
         *reinterpret_cast<uint32_t *>(&smem_in[r][c +  8]) = gv.z;
         *reinterpret_cast<uint32_t *>(&smem_in[r][c + 12]) = gv.w;
     }
 
     if (tid < TILE) {
         smem_scales[tid] = S_in[(int64_t)(row_start + tid) * s_in_stride_row
                                + (int64_t)tile_col * s_in_stride_col];
     }
     __syncthreads();
 
     // ==================================================================
     // Phase 2 :  Dequant  +  column amax  (warp-parallel)
     // ==================================================================
     const int warp_id     = tid >> 5;
     const int lane_id     = tid & 31;
     const int my_col_base = warp_id << 4;
     const int my_row_base = lane_id << 2;
 
     float amax[16];
     #pragma unroll
     for (int c = 0; c < 16; ++c) amax[c] = 0.0f;
 
     #pragma unroll
     for (int r = 0; r < 4; ++r) {
         const int   row       = my_row_base + r;
         const float row_scale = smem_scales[row];
 
         #pragma unroll
         for (int c4 = 0; c4 < 4; ++c4) {
             uint32_t raw = *reinterpret_cast<const uint32_t *>(
                                &smem_in[row][my_col_base + (c4 << 2)]);
             float v0, v1, v2, v3;
             fp8x4_to_f32x4(raw, v0, v1, v2, v3);
 
             v0 *= row_scale;  v1 *= row_scale;
             v2 *= row_scale;  v3 *= row_scale;
 
             const int base = c4 << 2;
             amax[base    ] = fmaxf(amax[base    ], fabsf(v0));
             amax[base + 1] = fmaxf(amax[base + 1], fabsf(v1));
             amax[base + 2] = fmaxf(amax[base + 2], fabsf(v2));
             amax[base + 3] = fmaxf(amax[base + 3], fabsf(v3));
         }
     }
 
     #pragma unroll
     for (int off = 16; off >= 1; off >>= 1) {
         #pragma unroll
         for (int c = 0; c < 16; ++c)
             amax[c] = fmaxf(amax[c],
                             __shfl_xor_sync(0xFFFFFFFF, amax[c], off));
     }
 
     if (lane_id == 0) {
         #pragma unroll
         for (int c = 0; c < 16; ++c) {
             float raw = fmaxf(amax[c] / FP8_MAX, 1e-30f);
             smem_col_scale[my_col_base + c] = pow2_ceil_positive(raw);
         }
     }
     __syncthreads();
 
     // ==================================================================
     // Phase 3 :  Re-quantise  +  transpose  →  smem_out
     // ==================================================================
     float inv_col_sc[16];
     #pragma unroll
     for (int c = 0; c < 16; ++c)
         inv_col_sc[c] = 1.0f / smem_col_scale[my_col_base + c];
 
     #pragma unroll
     for (int c4 = 0; c4 < 4; ++c4) {
         float vals[4][4];
 
         #pragma unroll
         for (int r = 0; r < 4; ++r) {
             const int   row       = my_row_base + r;
             const float row_scale = smem_scales[row];
             uint32_t raw = *reinterpret_cast<const uint32_t *>(
                                &smem_in[row][my_col_base + (c4 << 2)]);
             float v0, v1, v2, v3;
             fp8x4_to_f32x4(raw, v0, v1, v2, v3);
 
             const int base = c4 << 2;
             vals[r][0] = v0 * row_scale * inv_col_sc[base    ];
             vals[r][1] = v1 * row_scale * inv_col_sc[base + 1];
             vals[r][2] = v2 * row_scale * inv_col_sc[base + 2];
             vals[r][3] = v3 * row_scale * inv_col_sc[base + 3];
         }
 
         #pragma unroll
         for (int j = 0; j < 4; ++j) {
             uint32_t pk = f32x4_to_fp8x4(vals[0][j], vals[1][j],
                                           vals[2][j], vals[3][j]);
             *reinterpret_cast<uint32_t *>(
                 &smem_out[my_col_base + (c4 << 2) + j][my_row_base]) = pk;
         }
     }
     __syncthreads();
 
     // ==================================================================
     // Phase 4 :  smem_out → global  (vectorised 128-bit stores)
     // ==================================================================
     #pragma unroll
     for (int it = 0; it < 4; ++it) {
         const int h  = (tid >> 3) + (it << 5);
         const int t  = (tid & 7) << 4;
 
         uint4 gv;
         gv.x = *reinterpret_cast<const uint32_t *>(&smem_out[h][t     ]);
         gv.y = *reinterpret_cast<const uint32_t *>(&smem_out[h][t +  4]);
         gv.z = *reinterpret_cast<const uint32_t *>(&smem_out[h][t +  8]);
         gv.w = *reinterpret_cast<const uint32_t *>(&smem_out[h][t + 12]);
 
         const int64_t gaddr = flat_base
                             + (int64_t)(col_start + h) * g_size
                             + tok_in_group + t;
         *reinterpret_cast<uint4 *>(Y_flat + gaddr) = gv;
     }
 
     if (tid < TILE) {
         S_out[(int64_t)tile_row * hidden_size + col_start + tid] =
             smem_col_scale[tid];
     }
 }
 
 // ---------------------------------------------------------------------------
 // Launcher (always uses the optimized kernel)
 // ---------------------------------------------------------------------------
 
 static void launch_row2col_grouped(
     torch::Tensor X,
     torch::Tensor S_in,
     torch::Tensor group_offsets,
     torch::Tensor m_indices,
     torch::Tensor Y_flat,
     torch::Tensor S_out)
 {
     const int total_tokens = X.size(0);
     const int hidden_size  = X.size(1);
 
     TORCH_CHECK(total_tokens % TILE == 0,
                 "total_tokens must be a multiple of 128");
     TORCH_CHECK(hidden_size % TILE == 0,
                 "hidden_size must be a multiple of 128");
 
    dim3 grid(total_tokens / TILE, hidden_size / TILE);
    dim3 block(THREADS);

    auto stream = at::cuda::getCurrentCUDAStream();
    row2col_grouped_kernel<<<grid, block, 0, stream>>>(
         X.data_ptr<uint8_t>(),
         S_in.data_ptr<float>(),
         group_offsets.data_ptr<int64_t>(),
         m_indices.data_ptr<int32_t>(),
         Y_flat.data_ptr<uint8_t>(),
         S_out.data_ptr<float>(),
         hidden_size, total_tokens,
         S_in.stride(0), S_in.stride(1));
 
     C10_CUDA_CHECK(cudaGetLastError());
 }
 
 // ---------------------------------------------------------------------------
 // Helper: build per-group QTensor views on pre-allocated flat buffers using
 // as_strided (single ATen op per tensor, avoids slice+view overhead).
 // ---------------------------------------------------------------------------
 
 static py::tuple build_qtensor_views(
     torch::Tensor Y_flat,
     torch::Tensor S_out,
     const std::vector<int64_t>& group_sizes,
     int hidden_size,
     py::object& qtensor_cls,
     py::object& fp8_dtype,
     py::object& torch_dtype)
 {
     using namespace pybind11::literals;
 
     const int64_t G = (int64_t)group_sizes.size();
     py::tuple result(G);
 
     int64_t data_off  = 0;
     int64_t scale_row = 0;
     const int64_t H  = (int64_t)hidden_size;
     const int64_t scale_cols = S_out.size(1);
 
     for (int64_t g = 0; g < G; ++g) {
         const int64_t gs = group_sizes[g];
 
         torch::Tensor col_data = Y_flat.as_strided(
             {H, gs}, {gs, 1}, data_off);
         torch::Tensor col_scale = S_out.as_strided(
             {gs / TILE, scale_cols}, {scale_cols, 1}, scale_row * scale_cols);
 
         py::tuple shape = py::make_tuple(gs, H);
 
         result[g] = qtensor_cls(
             shape, torch_dtype,
             "columnwise_data"_a = col_data,
             "columnwise_scale_inv"_a = col_scale,
             "rowwise_data"_a = py::none(),
             "rowwise_scale_inv"_a = py::none(),
             "is_2D_scaled"_a = false,
             "fp8_dtype"_a = fp8_dtype,
             "quantizer"_a = py::none());
 
         data_off  += H * gs;
         scale_row += gs / TILE;
     }
     return result;
 }
 
 // ---------------------------------------------------------------------------
 // v5 API: two-phase (prepare + execute) for overlapping alloc with other work.
 // ---------------------------------------------------------------------------
 
 // Phase 1: pre-allocate output buffers (blocking CUDA malloc).
 std::vector<torch::Tensor> row2col_alloc(torch::Tensor X)
 {
     const int H = X.size(1);
     const int T = X.size(0);
     TORCH_CHECK(T % TILE == 0, "total_tokens must be a multiple of 128");
     TORCH_CHECK(H % TILE == 0, "hidden_size must be a multiple of 128");
     auto Y = torch::empty({(int64_t)H * T}, X.options().dtype(torch::kUInt8));
     auto S = torch::empty({T / TILE, H},    X.options().dtype(torch::kFloat32));
     return {Y, S};
 }
 
 // Phase 2 (cache hit): update cached QTensor attrs + async kernel launch.
 py::tuple row2col_cached_prealloc(
     torch::Tensor X,
     torch::Tensor S_in,
     torch::Tensor group_offsets,
     torch::Tensor m_indices,
     std::vector<int64_t> group_sizes,
     py::tuple cached,
     torch::Tensor Y,
     torch::Tensor S)
 {
     const int H = X.size(1);
     const int64_t G  = (int64_t)group_sizes.size();
     const int64_t sc = S.size(1);
     int64_t d_off = 0, s_row = 0;
     for (int64_t g = 0; g < G; ++g) {
         const int64_t gs = group_sizes[g];
         torch::Tensor col_data = Y.as_strided(
             {(int64_t)H, gs}, {gs, 1}, d_off);
         torch::Tensor col_scale = S.as_strided(
             {gs / TILE, sc}, {sc, 1}, s_row * sc);
         py::object qt = cached[g];
         qt.attr("_columnwise_data") = col_data;
         qt.attr("_columnwise_scale_inv") = col_scale;
         d_off += (int64_t)H * gs;
         s_row += gs / TILE;
     }
     launch_row2col_grouped(X, S_in, group_offsets, m_indices, Y, S);
     return cached;
 }
 
 // Phase 2 (cache miss): build QTensor shells + async kernel launch.
 py::tuple row2col_wrapped_prealloc(
     torch::Tensor X,
     torch::Tensor S_in,
     torch::Tensor group_offsets,
     torch::Tensor m_indices,
     std::vector<int64_t> group_sizes,
     py::object qtensor_cls,
     py::object fp8_dtype,
     py::object torch_dtype,
     torch::Tensor Y,
     torch::Tensor S)
 {
     const int H = X.size(1);
     py::tuple result = build_qtensor_views(
         Y, S, group_sizes, H, qtensor_cls, fp8_dtype, torch_dtype);
     launch_row2col_grouped(X, S_in, group_offsets, m_indices, Y, S);
     return result;
 }
 
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("row2col_alloc", &row2col_alloc,
           "Pre-allocate output buffers (CUDA)",
           py::arg("X"));
     m.def("row2col_cached_prealloc", &row2col_cached_prealloc,
           "Cached row→col with pre-allocated buffers (CUDA)",
           py::arg("X"), py::arg("S_in"),
           py::arg("group_offsets"), py::arg("m_indices"),
           py::arg("group_sizes"),
           py::arg("cached"), py::arg("Y"), py::arg("S"));
     m.def("row2col_wrapped_prealloc", &row2col_wrapped_prealloc,
           "First-call QTensor build with pre-allocated buffers (CUDA)",
           py::arg("X"), py::arg("S_in"),
           py::arg("group_offsets"), py::arg("m_indices"),
           py::arg("group_sizes"),
           py::arg("qtensor_cls"), py::arg("fp8_dtype"),
           py::arg("torch_dtype"), py::arg("Y"), py::arg("S"));
 }
 