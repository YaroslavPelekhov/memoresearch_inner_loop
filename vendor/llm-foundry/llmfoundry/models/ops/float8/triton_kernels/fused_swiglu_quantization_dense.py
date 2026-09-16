import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
)
import transformer_engine_torch as tex

# --------------------------------------------------------------------------------------
# Optimized SwiGLU(+quant) kernels
#
# This file rewrites the kernels from `swiglu_quant_kernels.py` in a more GPU-friendly way:
#   * 2D grid launch (row, col_block) to eliminate div/mod per program
#   * use tl.num_programs(axis) to compute (row-major / col-major) scale indices cheaply
#   * reduce divisions in FP8 scaling by computing both (scale, inv_scale) from amax
#   * optional power-of-two rounding uses exp2/log2 but avoids scalar division via exp2(-e)
#   * explicit compiler hints (max_contiguous / multiple_of) to help vectorization
#   * explicit cache modifiers / eviction policy knobs for memory-bound behavior
#   * autotune for BLOCK_SIZE + num_warps
# --------------------------------------------------------------------------------------

# FP8 E4M3 max finite magnitude is 448.
FP8_MAX: tl.constexpr = 448.0
INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX
MIN_SCALE: tl.constexpr = 1e-30
MIN_ABS: tl.constexpr = FP8_MAX * MIN_SCALE  # clamps amax so that scale >= MIN_SCALE
Q_BLOCK: tl.constexpr = 128

STATIC_TRITON_PARAMS_FWD = {
    18432: {
        "num_warps": 4,
        "maxnreg": 64,
        "TILE_SIZE": 1024,
    },
    11008: {
        "num_warps": 2,
        "maxnreg": 32,
        "TILE_SIZE": 512,
    },
    8960: {
        "num_warps": 4,
        "maxnreg": 128,
        "TILE_SIZE": 512,
    },
    2048: {
        "num_warps": 4,
        "maxnreg": 32,
        "TILE_SIZE": 512
    },
    1536: {
        "num_warps": 2,
        "maxnreg": 255,
        "TILE_SIZE": 256
    },
    1280: {
        "num_warps": 2,
        "maxnreg": 128,
        "TILE_SIZE": 256
    }
}

STATIC_TRITON_PARAMS_BWD = {
    18432: {
        "num_warps": 2,
        "maxnreg": 32,
        "TILE_SIZE": 512,
    },
    11008: {
        "num_warps": 1,
        "maxnreg": 255,
        "TILE_SIZE": 1024,
    },
    8960: {
        "num_warps": 1,
        "maxnreg": 64,
        "TILE_SIZE": 256,
    },
    2048: {
        "num_warps": 4,
        "maxnreg": 32,
        "TILE_SIZE": 512
    },
    1536: {
        "num_warps": 2,
        "maxnreg": 32,
        "TILE_SIZE": 256
    },
    1280: {
        "num_warps": 2,
        "maxnreg": 128,
        "TILE_SIZE": 256
    }
}

@triton.jit
def _fp8_scale_inv_from_amax(amax, POWER_TWO_MAX_ROUND: tl.constexpr):
    """Compute (scale, inv_scale) for FP8 E4M3 dynamic scaling.

    We store `scale` (used for dequant) and use `inv_scale` (used for quant):
      scale     = max(|x|)/FP8_MAX   (optionally rounded up to power-of-two)
      inv_scale = 1/scale

    Optimization vs. naive implementation:
      - POWER_TWO_MAX_ROUND=False: compute inv_scale from amax directly (FP8_MAX/amax)
        which avoids an extra division by `scale`.
      - POWER_TWO_MAX_ROUND=True: reuse exponent e = ceil(log2(scale_pre)) and compute
        inv_scale = exp2(-e) (no division).
    """
    amax_f32 = amax.to(tl.float32)

    if POWER_TWO_MAX_ROUND:
        scale_pre = tl.maximum(amax_f32 * INV_FP8_MAX, MIN_SCALE)
        e = tl.ceil(tl.log2(scale_pre))
        scale = tl.exp2(e)
        inv_scale = tl.exp2(-e)
    else:
        # Clamp amax so scale >= MIN_SCALE (and avoid div0)
        amax_f32 = tl.maximum(amax_f32, MIN_ABS)
        scale = amax_f32 * INV_FP8_MAX
        inv_scale = FP8_MAX / amax_f32

    return scale, inv_scale


# -----------------------------
# Forward: y = silu(gate) * up  (+ quantize y)
# -----------------------------
@triton.heuristics(
    values={
        "EVEN_TILE": lambda args: (args["n_cols"] % args["TILE_SIZE"]) == 0
    }
)
@triton.jit
def swiglu_quant_forward_kernel_opt(
    gate_ptr,
    up_ptr,
    output_ptr,
    output_scale_ptr,
    # runtime
    n_cols,
    # meta
    TILE_SIZE: tl.constexpr,
    EVEN_TILE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
    # clamp (compiled away entirely when APPLY_CLAMP=False)
    APPLY_CLAMP: tl.constexpr,
    swiglu_limit: tl.constexpr,
    clip_counts_ptr,
    ACTIVATION_MONITOR_ON: tl.constexpr,
):
    # 2D launch: pid_m = row, pid_t = col_block
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    grid_m = tl.num_programs(0)  # == n_rows
    NUM_Q: tl.constexpr = TILE_SIZE // Q_BLOCK

    offs_q = tl.arange(0, NUM_Q)
    offs_i = tl.arange(0, Q_BLOCK)

    offs = pid_t * TILE_SIZE + offs_q[:, None] * Q_BLOCK + offs_i[None, :]
    row_start = pid_m * n_cols
    idx = row_start + offs

    gate_ptr = tl.multiple_of(gate_ptr, 16)
    up_ptr = tl.multiple_of(up_ptr, 16)
    if EVEN_TILE:
        gate = tl.load(gate_ptr + idx, cache_modifier=".cg")
        up = tl.load(up_ptr + idx, cache_modifier=".cg")
    else:
        mask = offs < n_cols
        gate = tl.load(gate_ptr + idx, mask=mask, cache_modifier=".cg")
        up = tl.load(up_ptr + idx, mask=mask, cache_modifier=".cg")

    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)

    # Clamp (compiled away entirely when APPLY_CLAMP=False)
    if APPLY_CLAMP:
        if ACTIVATION_MONITOR_ON:
            gate_clamped = gate_f32 >= swiglu_limit
            up_clamped = (up_f32 <= -swiglu_limit) | (up_f32 >= swiglu_limit)
        gate_f32 = tl.minimum(gate_f32, swiglu_limit)
        up_f32 = tl.minimum(tl.maximum(up_f32, -swiglu_limit), swiglu_limit)
        if ACTIVATION_MONITOR_ON:
            tl.atomic_add(clip_counts_ptr,     tl.sum(gate_clamped.to(tl.int32)))
            tl.atomic_add(clip_counts_ptr + 1, tl.sum(up_clamped.to(tl.int32)))

    # SwiGLU: silu(gate) * up
    sig = tl.sigmoid(gate_f32)
    act = gate_f32 * sig * up_f32

    amax_act = tl.max(tl.abs(act), axis=1)
    out_scale, out_inv_scale = _fp8_scale_inv_from_amax(amax_act, POWER_TWO_MAX_ROUND)

    act_q = act.to(tl.float32) * out_inv_scale[:, None]
    if EVEN_TILE:
        tl.store(output_ptr + idx,
            act_q.to(output_ptr.dtype.element_ty),
            cache_modifier=".wb")
    else:
        tl.store(output_ptr + idx,
            act_q.to(output_ptr.dtype.element_ty), mask=mask,
            cache_modifier=".wb")

    # Output scale in transposed layout: (blocks_per_row_mult4, n_rows)
    col_block = pid_t * NUM_Q + offs_q
    s_off = col_block * grid_m + pid_m

    blocks_per_row = n_cols // Q_BLOCK
    if EVEN_TILE:
        tl.store(output_scale_ptr + s_off,
                    out_scale.to(output_scale_ptr.dtype.element_ty))
    else:
        s_mask = col_block < blocks_per_row
        tl.store(output_scale_ptr + s_off,
                    out_scale.to(output_scale_ptr.dtype.element_ty), mask=s_mask)


# -----------------------------
# Backward: given grad_out, compute grad_gate and grad_up (+ quantize)
# -----------------------------

@triton.heuristics(
    values={
        "EVEN_TILE": lambda args: (args["n_cols"] % args["TILE_SIZE"]) == 0
    }
)
@triton.jit
def swiglu_quant_backward_kernel_opt(
    gate_ptr,
    up_ptr,
    grad_out_ptr,
    # quantized grad outputs
    grad_gate_ptr,
    grad_up_ptr,
    grad_gate_scale_ptr,
    grad_up_scale_ptr,
    # runtime
    n_cols,
    # meta
    TILE_SIZE: tl.constexpr,
    EVEN_TILE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
    # clamp (compiled away when APPLY_CLAMP=False)
    APPLY_CLAMP: tl.constexpr,
    swiglu_limit: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    grid_m = tl.num_programs(0)  # == n_rows
    NUM_Q: tl.constexpr = TILE_SIZE // Q_BLOCK

    offs_q = tl.arange(0, NUM_Q)
    offs_i = tl.arange(0, Q_BLOCK)

    offs = pid_t * TILE_SIZE + offs_q[:, None] * Q_BLOCK + offs_i[None, :]

    row_start = pid_m * n_cols
    idx = row_start + offs
    if EVEN_TILE:
        gate = tl.load(gate_ptr + idx, cache_modifier=".cg").to(tl.float32)
        up = tl.load(up_ptr + idx, cache_modifier=".cg").to(tl.float32)
        g_out = tl.load(grad_out_ptr + idx, cache_modifier=".cg").to(tl.float32)
    else:
        mask = offs < n_cols
        gate = tl.load(gate_ptr + idx, mask=mask, cache_modifier=".cg").to(tl.float32)
        up = tl.load(up_ptr + idx, mask=mask, cache_modifier=".cg").to(tl.float32)
        g_out = tl.load(grad_out_ptr + idx, mask=mask, cache_modifier=".cg").to(tl.float32)

    # Clamp-aware gradient masking.
    # gate/up are the exact pre-clamp BF16 values saved in ctx.
    # Masks are computed first; then gate/up are clamped to match forward values.
    if APPLY_CLAMP:
        gate_clamp_mask = gate < swiglu_limit         # True where gate was NOT clamped
        up_clamp_mask   = (up > -swiglu_limit) & (up < swiglu_limit)  # True where up was NOT clamped
        gate = tl.minimum(gate, swiglu_limit)
        up   = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)

    # Backprop through silu(gate) * up
    sig = tl.sigmoid(gate)
    silu = gate * sig
    grad_up = g_out * silu

    # d/dx silu(x) = sig(x) * (1 + x * (1 - sig(x)))
    dsilu = sig + silu * (1.0 - sig)
    grad_gate = (g_out * up) * dsilu

    # Zero gradients through clamped elements
    if APPLY_CLAMP:
        grad_up   = tl.where(up_clamp_mask,   grad_up,   0.0)
        grad_gate = tl.where(gate_clamp_mask, grad_gate, 0.0)

    # Blockwise scales in transposed layout: (blocks_per_row_mult4, n_rows)
    col_block = pid_t * NUM_Q + offs_q
    s_off = col_block * grid_m + pid_m

    amax_gg = tl.max(tl.abs(grad_gate), axis=1)
    gg_scale, gg_inv_scale = _fp8_scale_inv_from_amax(amax_gg, POWER_TWO_MAX_ROUND)

    amax_gu = tl.max(tl.abs(grad_up), axis=1)
    gu_scale, gu_inv_scale = _fp8_scale_inv_from_amax(amax_gu, POWER_TWO_MAX_ROUND)

    grad_gate_q = grad_gate * gg_inv_scale[:, None]
    grad_up_q = grad_up * gu_inv_scale[:, None]
    if EVEN_TILE:
        tl.store(grad_gate_ptr + idx,
                 grad_gate_q.to(grad_gate_ptr.dtype.element_ty), cache_modifier=".wb")
        tl.store(grad_up_ptr + idx,
                 grad_up_q.to(grad_up_ptr.dtype.element_ty), cache_modifier=".wb")
    else:
        tl.store(grad_gate_ptr + idx,
                 grad_gate_q.to(grad_gate_ptr.dtype.element_ty), mask=mask, cache_modifier=".wb")
        tl.store(grad_up_ptr + idx,
                 grad_up_q.to(grad_up_ptr.dtype.element_ty), mask=mask, cache_modifier=".wb")

    blocks_per_row = n_cols // Q_BLOCK
    if EVEN_TILE:
        tl.store(grad_gate_scale_ptr + s_off,
                 gg_scale.to(grad_gate_scale_ptr.dtype.element_ty))
        tl.store(grad_up_scale_ptr + s_off,
                 gu_scale.to(grad_up_scale_ptr.dtype.element_ty))
    else:
        s_mask = col_block < blocks_per_row
        tl.store(grad_gate_scale_ptr + s_off,
                 gg_scale.to(grad_gate_scale_ptr.dtype.element_ty), mask=s_mask)
        tl.store(grad_up_scale_ptr + s_off,
                 gu_scale.to(grad_up_scale_ptr.dtype.element_ty), mask=s_mask)


class SepProductSiluQuant(torch.autograd.Function):

    @staticmethod
    def _make_fp8_blockwise_qtensor(
        data: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        to_uint8: bool = False
    ) -> Float8BlockwiseQTensor:
        fake_dtype = torch.bfloat16
        fp8_dtype = tex.DType.kFloat8E4M3 if dtype == torch.float8_e4m3fn \
                                          else tex.DType.kFloat8E5M2
        data_c = data if data.is_contiguous() else data.contiguous()
        scale_c = scale if scale.is_contiguous() else scale.contiguous()
        data_c = data_c.view(torch.uint8) if to_uint8 else data_c
        return Float8BlockwiseQTensor(
            shape=data.shape,
            dtype=fake_dtype,
            rowwise_data=data_c,
            rowwise_scale_inv=scale_c,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=fp8_dtype,
            quantizer=None,
            is_2D_scaled=False
        )

    @staticmethod
    def forward(
        ctx,
        gate_input: torch.Tensor,
        up_input: torch.Tensor,
        dtype: torch.dtype = torch.float8_e4m3fn,
        power_two_max_round: bool = True,
        swiglu_limit: float = 0.0,
        clip_counts: tp.Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        batch, tokens, h_dim = gate_input.shape
        assert h_dim % int(Q_BLOCK) == 0, (
            f"Expected h_dim ({h_dim}) divisible by Q_BLOCK (128) for blockwise scaling."
        )

        ctx.inp_shape = gate_input.shape
        ctx.fp8_dtype = dtype
        ctx.power_two_max_round = power_two_max_round
        ctx.swiglu_limit = swiglu_limit

        blocks_per_row = h_dim // int(Q_BLOCK)
        blocks_per_row_mult_4 = (blocks_per_row + 4 - 1) // 4 * 4
        n_rows = batch * tokens

        prod_silu_out = torch.empty((n_rows, h_dim),
                                    device=gate_input.device,
                                    dtype=dtype)
        output_scale = torch.empty((blocks_per_row_mult_4, n_rows),
                                   device=gate_input.device,
                                   dtype=torch.float32)

        gate_input_c = gate_input if gate_input.is_contiguous() else gate_input.contiguous()
        up_input_c = up_input if up_input.is_contiguous() else up_input.contiguous()
        gate_input_flat = gate_input_c.view(-1, h_dim)
        up_input_flat = up_input_c.view(-1, h_dim)

        monitor_on = clip_counts is not None and swiglu_limit > 0

        grid = lambda meta: (n_rows, triton.cdiv(h_dim, meta["TILE_SIZE"]))
        swiglu_quant_forward_kernel_opt[grid](
            gate_input_flat,
            up_input_flat,
            prod_silu_out,
            output_scale,
            h_dim,
            POWER_TWO_MAX_ROUND=power_two_max_round,
            APPLY_CLAMP=swiglu_limit > 0,
            swiglu_limit=float(swiglu_limit),
            clip_counts_ptr=clip_counts,
            ACTIVATION_MONITOR_ON=monitor_on,
            **params_to_kernel_kwargs(STATIC_TRITON_PARAMS_FWD, h_dim,
                                      num_warps=4, maxnreg=32, TILE_SIZE=1024)
        )

        ctx.save_for_backward(gate_input_flat, up_input_flat)

        output_reshaped = prod_silu_out.view(ctx.inp_shape)
        return SepProductSiluQuant._make_fp8_blockwise_qtensor(
            output_reshaped,
            output_scale,
            dtype,
            to_uint8=True
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> tp.Tuple[Float8BlockwiseQTensor, ...]:

        gate_input, up_input = ctx.saved_tensors
        batch, tokens, h_dim = ctx.inp_shape
        power_two_max_round = ctx.power_two_max_round
        swiglu_limit = ctx.swiglu_limit

        blocks_per_row = h_dim // int(Q_BLOCK)
        blocks_per_row_mult_4 = (blocks_per_row + 4 - 1) // 4 * 4
        n_rows = batch * tokens

        grad_gate = torch.empty((n_rows, h_dim), device=gate_input.device, dtype=ctx.fp8_dtype)
        grad_up = torch.empty((n_rows, h_dim), device=up_input.device, dtype=ctx.fp8_dtype)
        grad_gate_scale = torch.empty((blocks_per_row_mult_4, n_rows),
                            device=gate_input.device,
                            dtype=torch.float32)
        grad_up_scale = torch.empty((blocks_per_row_mult_4, n_rows),
                            device=gate_input.device,
                            dtype=torch.float32)

        grad_output = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        gate_input_flat = gate_input.view(-1, h_dim)
        up_input_flat = up_input.view(-1, h_dim)
        grad_output_flat = grad_output.view(-1, h_dim)

        grid = lambda meta: (n_rows, triton.cdiv(h_dim, meta["TILE_SIZE"]))
        swiglu_quant_backward_kernel_opt[grid](
            gate_input_flat,
            up_input_flat,
            grad_output_flat,
            grad_gate,
            grad_up,
            grad_gate_scale,
            grad_up_scale,
            h_dim,
            POWER_TWO_MAX_ROUND=power_two_max_round,
            APPLY_CLAMP=swiglu_limit > 0,
            swiglu_limit=float(swiglu_limit),
            **params_to_kernel_kwargs(STATIC_TRITON_PARAMS_BWD, h_dim,
                                      num_warps=4, maxnreg=32, TILE_SIZE=1024)
        )

        grad_gate_reshaped = grad_gate.view(ctx.inp_shape)
        grad_up_reshaped = grad_up.view(ctx.inp_shape)

        grad_gate_out = SepProductSiluQuant._make_fp8_blockwise_qtensor(
            grad_gate_reshaped,
            grad_gate_scale,
            ctx.fp8_dtype,
            to_uint8=True
        )
        grad_up_out = SepProductSiluQuant._make_fp8_blockwise_qtensor(
            grad_up_reshaped,
            grad_up_scale,
            ctx.fp8_dtype,
            to_uint8=True
        )
        return (
            grad_gate_out,
            grad_up_out,
            None,  # dtype
            None,  # power_two_max_round
            None,  # swiglu_limit
            None,  # clip_counts
        )
