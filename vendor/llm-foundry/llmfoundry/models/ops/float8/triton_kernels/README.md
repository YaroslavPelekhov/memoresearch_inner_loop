# Triton Kernels

## Quantization Kernels

### `block_quantization.py`
**Kernel:** `_block_quantization_kernel`
- **Input:** `tensor (R, C)` — bf16/fp32 2D matrix where `R % 128 == 0` and `C % 128 == 0`
- **Output:** `out (R, C)` fp8, `scale (R // 128, C // 128)`

Block-wise quantization of a 2D matrix into FP8. Processes 128x128 blocks, computes per-block scale as `max(abs(x)) / 448` (optionally rounded to power of 2), and writes the quantized tensor with per-block scales.

### `row_quantization.py`
**Kernel:** `_rowwise_1d_quantize_kernel`
- **Input:** `tensor (T, H)` — 2D activation tensor
- **Output:** `out (T, H)` fp8, `scale (T, H // 128)`

Row-wise quantization into FP8. Works on 128x128 tiles, computes one scale per row per 128-element column chunk, and outputs FP8 values with row-wise scales.

## Row-to-Column Requantization Kernels

### `row2col_te_quantization.py`
**Kernel:** `_rowwise_to_grouped_columnwise_kernel`
- **Input:** `tensor (T, H)` fp8 row-wise, `scale_inv (T, H // 128)`, `m_indices (T,)`, `group_sizes (num_groups,)`
- **Output:** `out (H * T,)` fp8 column-wise, `scale_col (T // 128, H)`

Converts row-wise FP8 to grouped column-wise layout for MoE. Dequantizes with row-wise scales, re-quantizes with column-wise scales, transposes each block, and packs groups contiguously for `general_grouped_gemm`.

### `row2col_deepgemm_quantization.py`
**Kernel:** `_rowwise_to_columnwise_deepgemm_layout_kernel`
- **Input:** `tensor (seq_len, hid_dim)` fp8 row-wise, `scale_inv (seq_len, hid_dim // 128)`, `m_indices (seq_len,)`, `group_sizes (seq_len,)`
- **Output:** `columnwise_data (hid_dim * seq_len,)` fp8, `columnwise_scale_inv (seq_len // 128, hid_dim)`

Same row-to-column conversion as `row2col_te_quantization` but targets DeepGEMM layout. Dequantizes, requantizes with column-wise scales, transposes, and writes to a group-contiguous output buffer.

### `row2col_dense_kernels.py`
**Kernel 1:** `_rowwise_to_columnwise_inplace_kernel`
- **Input:** `tensor (seq_len, H)` fp8 row-wise, `scales_inv (seq_len, H // 128)`
- **Output:** `output (seq_len, H)` fp8 column-wise, `scales_inv_col (seq_len // 128, H)`

Dense (non-grouped) row-to-column requantization. Dequantizes with row-wise scales, requantizes with column-wise scales, and transposes per block.

**Kernel 2:** `_scaling_aware_fp8_transpose_kernel`
- **Input:** `rowwise_data (R, C)` fp8, `rowwise_scale_inv (R, C // 128)`
- **Output:** `columnwise_data (C, R)` fp8, `columnwise_scale_inv (R // 128, C)`

Transforms row-wise FP8 into column-wise layout using exponent adjustments and bit manipulation directly in the FP8 domain, without converting to FP32 and back.

**Kernel 3:** `_block_quantize_kernel`
- **Input:** `weight (int_dim, hid_size)` fp32
- **Output:** `output (hid_size, int_dim)` fp8, `scale_inv (hid_blocks, seq_blocks)`

Quantizes a weight matrix to FP8 with 128x128 block-wise scales and transposes from `(int_dim, hid_size)` to `(hid_size, int_dim)`.

## Block Transpose + Quantization

### `block_transpose_fused_quantization.py`
**Kernel:** `_block_quantize_kernel_transpose`
- **Input:** `tensor (R, C)` where `R = num_groups * int_dim`
- **Output:** `out (num_groups, C, int_dim)` fp8, `scale (num_groups, C // 128, int_dim // 128)`

Fused block quantization and transpose. Quantizes 128x128 blocks with per-block scaling, transposes each block, and outputs in `(num_groups, hid_dim, int_dim)` layout with matching scales.

## Fused SwiGLU + Quantization Kernels

### `fused_swiglu_quantization.py` (MoE variant)
**Kernel 1:** `_swiglu_quant_fwd_kernel`
- **Input:** `input_tensor (T, 2 * H)` — interleaved gate|up, `probs (T,)`
- **Output:** `prod_silu_output (T, H)` fp8, `input_tensor_quantized (T, 2 * H)` fp8, plus row-wise scales

Computes `silu(gate) * up * probs` and quantizes the result to FP8. Also quantizes gate and up separately to FP8 for the backward pass.

**Kernel 2:** `_swiglu_quant_bwd_kernel`
- **Input:** `grad_output (T, H)`, `input_tensor_quantized (T, 2 * H)` fp8, `probs (T,)`
- **Output:** `grad_input (T, 2 * H)` fp8, `grad_probs_partial (T, H // 128)`

Backward pass through SwiGLU for MoE. Dequantizes saved FP8 gate/up, computes grad_gate and grad_up, quantizes them to FP8, and writes partial gradients for probs.

### `fused_swiglu_quantization_dense.py` (Dense variant)
**Kernel 1:** `swiglu_quant_forward_kernel_opt`
- **Input:** `gate (N, C)`, `up (N, C)` — separate tensors
- **Output:** `output (N, C)` fp8, `output_scale (C // 128, N)` transposed; optionally `gate_fp8`, `up_fp8` with scales

Optimized fused SwiGLU (`silu(gate) * up`) with FP8 output quantization for dense (non-MoE) layers. Gate and up are separate inputs; optionally saves FP8 copies for the backward pass.

**Kernel 2:** `swiglu_quant_backward_kernel_opt`
- **Input:** `gate (N, C)`, `up (N, C)` (fp32 or fp8), `grad_out (N, C)`
- **Output:** `grad_gate (N, C)` fp8, `grad_up (N, C)` fp8, with transposed scales

Backward pass for dense SwiGLU. Optionally dequantizes saved FP8 gate/up, computes and FP8-quantizes grad_gate and grad_up.

## MoE Routing Kernels

### `fused_te_ops.py`
**Kernel 1:** `_row_to_padded_row_id_kernel`
- **Input:** `group_sizes (num_experts,)`
- **Output:** `out_indices (total_tokens,)`

Maps each token index to its padded row index for expert groups by computing cumulative padded expert sizes.

**Kernel 2:** `_permute_and_pad_kernel`
- **Input:** `input (tokens, H)`, `row_id_map (tokens, num_experts)`, `probs (tokens, num_experts)`, `scale (tokens, H // 128)`
- **Output:** `output (total_tokens, H)`, `permuted_probs (total_tokens,)`, `permuted_scale (total_tokens, H // 128)`

Permutes and pads token activations for expert-parallel MoE: routes each token to its expert row, pads to 128-alignment, and optionally permutes probs and scales.

**Kernel 3:** `_unpad_and_unpermute_kernel`
- **Input:** `input (total_tokens, H)`, `row_id_map`, `permuted_probs (total_tokens,)`
- **Output:** `output (tokens, H)`

Reverses permute-and-pad: gathers expert outputs by token, accumulates contributions from multiple experts, and restores original token order.

## Utility

### `utils.py`
**Kernel:** `_build_m_indices_kernel`
- **Input:** `group_sizes (num_experts,)`
- **Output:** `output (total,)` where `total = sum of padded group sizes`

Builds an index vector for grouped GEMM: for each position in the padded concatenation of expert groups, writes the expert index (or -1 for padding rows).

Also provides helpers: `tensor_to_kernel_args`, `make_float8_blockwise_qtensor_fn`, `maybe_convert_to_float8_e4m3fn`.

## Deprecated

### `dequantization_deprecated.py`
**Kernel:** `_dequantize_kernel`
- **Input:** `q (seq_len, H)` fp8, `scale (seq_len, H // 128)`
- **Output:** `out (seq_len, H)` bf16

Dequantizes FP8 blockwise (128 columns per block) to bfloat16 by multiplying each block by its scale.

### `row2col_requantization_deprecated.py`
**Kernel:** `_rowwise_to_columnwise_kernel`
- **Input:** `tensor (seq_len, H)` fp8 row-wise, `scale_inv (seq_len, H // 128)`
- **Output:** `out (H, seq_len)` fp8 column-wise, `scale_col (seq_len // 128, H)`

Converts row-wise FP8 to column-wise by dequantizing, transposing to `(H, seq_len)`, and requantizing with new column-wise scales.
