"""Fused SwiGLU + FP8 quantization kernel.

Forward
    ROW_BLOCK=64  COL_BLOCK=128  num_warps=4  maxnreg=160
    128 threads × 160 regs = 20 480 ≪ 65 536 → ~3 blocks/SM on H100.

Backward
    ROW_BLOCK=32  COL_BLOCK=128  num_warps=4  maxnreg=128
    grad_up is quantised and stored before grad_gate registers are ever
    allocated, cutting simultaneous live fp32 tiles from ~9 to ~7.
    128 threads × 128 regs = 16 384 → 4 blocks/SM on H100.
"""

import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
import transformer_engine_torch as tex

_SWIGLU_TOKEN_BUCKET_SIZE = 2048

FP8_MAX: float = 448.0
INV_FP8_MAX = tl.constexpr(1.0 / FP8_MAX)
MIN_SCALE = tl.constexpr(1e-30)

# ── tuning tables ──────────────────────────────────────────────────────────────
# hid_dim key; col block is always 128 (FP8 blockwise scale granularity).

STATIC_TRITON_PARAMS_FWD = {
    4096: {"num_warps": 4, "maxnreg": 160},
    2048: {"num_warps": 4, "maxnreg": 160},
    1536: {"num_warps": 4, "maxnreg": 160},
    1280: {"num_warps": 4, "maxnreg": 160},
}

STATIC_TRITON_PARAMS_BWD = {
    4096: {"num_warps": 4, "maxnreg": 128},
    2048: {"num_warps": 4, "maxnreg": 128},
    1536: {"num_warps": 4, "maxnreg": 128},
    1280: {"num_warps": 4, "maxnreg": 128},
}

# Row-block sizes (col block must remain 128 for FP8 blockwise scale layout).
FWD_ROW_BLOCK = 64
BWD_ROW_BLOCK = 32
COL_BLOCK     = 128


# ── helpers ────────────────────────────────────────────────────────────────────

@triton.jit
def _fp8_scale_inv_from_amax(amax):
    amax_f32  = amax.to(tl.float32)
    scale_pre = tl.maximum(amax_f32 * INV_FP8_MAX, MIN_SCALE)
    e         = tl.ceil(tl.log2(scale_pre))
    return tl.exp2(e), tl.exp2(-e)   # scale, inv_scale


# ── forward kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _swiglu_quant_fwd_kernel(
    input_tensor_ptr,
    input_tensor_stride_row, input_tensor_stride_col,
    input_tensor_size_row,   input_tensor_size_col,
    probs_ptr,
    probs_stride_row,
    probs_size_row,
    prod_silu_output_ptr,
    prod_silu_output_stride_row, prod_silu_output_stride_col,
    prod_silu_output_size_row,   prod_silu_output_size_col,
    prod_silu_output_sf_ptr,
    prod_silu_output_sf_stride_row, prod_silu_output_sf_stride_col,
    prod_silu_output_sf_size_row,   prod_silu_output_sf_size_col,
    # bucket drives static-param selection only (not used in kernel body)
    input_token_bucket,
    hid_dim:   tl.constexpr,
    ROW_BLOCK: tl.constexpr,   # 64
    COL_BLOCK: tl.constexpr,   # 128  (must match FP8 blockwise scale granularity)
    APPLY_CLAMP: tl.constexpr,
    swiglu_limit: tl.constexpr,
    # monitoring — compiled away entirely when ACTIVATION_MONITOR_ON=False
    clip_counts_ptr,
    ACTIVATION_MONITOR_ON: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    pid_hid   = tl.program_id(axis=1)

    row_offsets      = (pid_token * ROW_BLOCK + tl.arange(0, ROW_BLOCK)).to(tl.int64)
    gate_col_offsets = pid_hid * COL_BLOCK + tl.arange(0, COL_BLOCK)
    up_col_offsets   = hid_dim + gate_col_offsets

    gate_block = tl.load(
        input_tensor_ptr
        + row_offsets[:, None] * input_tensor_stride_row
        + gate_col_offsets[None, :] * input_tensor_stride_col
    ).to(tl.float32)

    up_block = tl.load(
        input_tensor_ptr
        + row_offsets[:, None] * input_tensor_stride_row
        + up_col_offsets[None, :] * input_tensor_stride_col
    ).to(tl.float32)

    probs_vec = tl.load(probs_ptr + row_offsets * probs_stride_row).to(tl.float32)

    if APPLY_CLAMP:
        if ACTIVATION_MONITOR_ON:
            gate_clamped = gate_block >= swiglu_limit
            up_clamped   = (up_block <= -swiglu_limit) | (up_block >= swiglu_limit)
        up_block   = tl.minimum(tl.maximum(up_block, -swiglu_limit), swiglu_limit)
        gate_block = tl.minimum(gate_block, swiglu_limit)
        if ACTIVATION_MONITOR_ON:
            tl.atomic_add(clip_counts_ptr,     tl.sum(gate_clamped.to(tl.int32)))
            tl.atomic_add(clip_counts_ptr + 1, tl.sum(up_clamped.to(tl.int32)))

    # silu(gate) * up * probs
    gate_sigm  = tl.sigmoid(gate_block)
    swiglu     = (gate_block * gate_sigm) * up_block * probs_vec[:, None]

    # per-row block quantisation to FP8
    swiglu_amax          = tl.max(tl.abs(swiglu), axis=1)
    swiglu_sf, swiglu_inv_sf = _fp8_scale_inv_from_amax(swiglu_amax)
    swiglu_q             = swiglu * swiglu_inv_sf[:, None]

    tl.store(
        prod_silu_output_ptr
        + row_offsets[:, None] * prod_silu_output_stride_row
        + gate_col_offsets[None, :] * prod_silu_output_stride_col,
        swiglu_q.to(prod_silu_output_ptr.dtype.element_ty),
    )
    tl.store(
        prod_silu_output_sf_ptr
        + row_offsets * prod_silu_output_sf_stride_row
        + pid_hid * prod_silu_output_sf_stride_col,
        swiglu_sf.to(prod_silu_output_sf_ptr.dtype.element_ty),
    )


# ── backward kernel ────────────────────────────────────────────────────────────

@triton.jit
def _swiglu_quant_bwd_kernel(
    grad_output_ptr,
    grad_output_stride_row, grad_output_stride_col,
    grad_output_size_row,   grad_output_size_col,
    probs_ptr,
    probs_stride_row,
    probs_size_row,
    input_ptr,
    input_stride_row, input_stride_col,
    input_size_row,   input_size_col,
    grad_input_ptr,
    grad_input_stride_row, grad_input_stride_col,
    grad_input_size_row,   grad_input_size_col,
    grad_input_sf_ptr,
    grad_input_sf_stride_row, grad_input_sf_stride_col,
    grad_input_sf_size_row,   grad_input_sf_size_col,
    grad_probs_partial_ptr,
    grad_probs_partial_stride_row, grad_probs_partial_stride_col,
    grad_probs_partial_size_row,   grad_probs_partial_size_col,
    grad_token_bucket,
    hid_dim:   tl.constexpr,
    ROW_BLOCK: tl.constexpr,   # 32
    COL_BLOCK: tl.constexpr,   # 128
    APPLY_CLAMP: tl.constexpr,
    swiglu_limit: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    pid_hid   = tl.program_id(axis=1)

    row_offsets      = (pid_token * ROW_BLOCK + tl.arange(0, ROW_BLOCK)).to(tl.int64)
    gate_col_offsets = pid_hid * COL_BLOCK + tl.arange(0, COL_BLOCK)
    up_col_offsets   = hid_dim + gate_col_offsets

    blocks_per_row = grad_probs_partial_size_col

    # ── loads ────────────────────────────────────────────────────────────────
    gate_block = tl.load(
        input_ptr
        + row_offsets[:, None] * input_stride_row
        + gate_col_offsets[None, :] * input_stride_col
    ).to(tl.float32)

    up_block = tl.load(
        input_ptr
        + row_offsets[:, None] * input_stride_row
        + up_col_offsets[None, :] * input_stride_col
    ).to(tl.float32)

    grad_out = tl.load(
        grad_output_ptr
        + row_offsets[:, None] * grad_output_stride_row
        + gate_col_offsets[None, :] * grad_output_stride_col
    ).to(tl.float32)

    probs_vec = tl.load(probs_ptr + row_offsets * probs_stride_row).to(tl.float32)

    # ── reapply clamp and derive masks (compiled away when APPLY_CLAMP=False) ──
    if APPLY_CLAMP:
        gate_clamp_mask = gate_block < swiglu_limit
        up_clamp_mask   = (up_block > -swiglu_limit) & (up_block < swiglu_limit)
        gate_block = tl.minimum(gate_block, swiglu_limit)
        up_block   = tl.minimum(tl.maximum(up_block, -swiglu_limit), swiglu_limit)

    # ── phase 1: grad_up — quantise and store before grad_gate registers open ─
    # Shrinks peak liveness: grad_up tiles are freed before dsilu/grad_gate exist.
    sig    = tl.sigmoid(gate_block)
    silu_g = gate_block * sig

    grad_up = grad_out * silu_g * probs_vec[:, None]
    if APPLY_CLAMP:
        grad_up = tl.where(up_clamp_mask, grad_up, 0.0)

    up_amax      = tl.max(tl.abs(grad_up), axis=1)
    up_sf, up_inv_sf = _fp8_scale_inv_from_amax(up_amax)

    tl.store(
        grad_input_ptr
        + row_offsets[:, None] * grad_input_stride_row
        + up_col_offsets[None, :] * grad_input_stride_col,
        (grad_up * up_inv_sf[:, None]).to(grad_input_ptr.dtype.element_ty),
    )
    tl.store(
        grad_input_sf_ptr
        + row_offsets * grad_input_sf_stride_row
        + (blocks_per_row + pid_hid) * grad_input_sf_stride_col,
        up_sf.to(grad_input_sf_ptr.dtype.element_ty),
    )
    # grad_up, up_sf, up_inv_sf, up_amax registers are now free

    # ── grad_probs partial (uses grad_out, silu_g, up_block — already live) ──
    partial_grad_probs = tl.sum(grad_out * silu_g * up_block, axis=1)
    tl.store(
        grad_probs_partial_ptr
        + row_offsets * grad_probs_partial_stride_row
        + pid_hid * grad_probs_partial_stride_col,
        partial_grad_probs,
    )

    # ── phase 2: grad_gate ────────────────────────────────────────────────────
    dsilu     = sig + silu_g * (1.0 - sig)

    grad_gate = grad_out * up_block * probs_vec[:, None] * dsilu
    if APPLY_CLAMP:
        grad_gate = tl.where(gate_clamp_mask, grad_gate, 0.0)

    gate_amax = tl.max(tl.abs(grad_gate), axis=1)
    gate_sf, gate_inv_sf = _fp8_scale_inv_from_amax(gate_amax)

    tl.store(
        grad_input_ptr
        + row_offsets[:, None] * grad_input_stride_row
        + gate_col_offsets[None, :] * grad_input_stride_col,
        (grad_gate * gate_inv_sf[:, None]).to(grad_input_ptr.dtype.element_ty),
    )
    tl.store(
        grad_input_sf_ptr
        + row_offsets * grad_input_sf_stride_row
        + pid_hid * grad_input_sf_stride_col,
        gate_sf.to(grad_input_sf_ptr.dtype.element_ty),
    )


# ── autograd Function ──────────────────────────────────────────────────────────

class FusedQuantizedSwiGLU(torch.autograd.Function):

    @staticmethod
    def _make_qtensor(data: torch.Tensor, scale: torch.Tensor) -> Float8BlockwiseQTensor:
        return Float8BlockwiseQTensor(
            shape=data.shape,
            dtype=torch.bfloat16,
            rowwise_data=data.contiguous(),
            rowwise_scale_inv=scale.contiguous(),
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=None,
            is_2D_scaled=False,
        )

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, probs: torch.Tensor,
                swiglu_limit: float = 0.0,
                power_two_max_round: bool = True,
                clip_counts: tp.Optional[torch.Tensor] = None) -> Float8BlockwiseQTensor:
        ctx.input_tensor_shape = input_tensor.shape
        ctx.probs_shape = probs.shape
        ctx.swiglu_limit = swiglu_limit

        total_tokens = input_tensor.shape[0]
        hid_dim      = input_tensor.shape[1] // 2
        assert total_tokens % FWD_ROW_BLOCK == 0 and hid_dim % COL_BLOCK == 0

        blocks_per_row = hid_dim // COL_BLOCK
        out    = torch.empty((total_tokens, hid_dim),
                             device=input_tensor.device, dtype=torch.float8_e4m3fn)
        out_sf = torch.empty((total_tokens, blocks_per_row),
                             device=input_tensor.device, dtype=torch.float32)

        x_c = input_tensor.contiguous()
        p_c = probs.reshape(-1).contiguous()

        # clip_counts is an int32[2] persistent buffer owned by the caller;
        # the caller must zero it before each forward call.
        monitor_on = clip_counts is not None and swiglu_limit > 0

        token_bucket = total_tokens // _SWIGLU_TOKEN_BUCKET_SIZE
        grid = (total_tokens // FWD_ROW_BLOCK, blocks_per_row)

        _swiglu_quant_fwd_kernel[grid](
            *tensor_to_kernel_args(x_c, 2),
            *tensor_to_kernel_args(p_c, 1),
            *tensor_to_kernel_args(out, 2),
            *tensor_to_kernel_args(out_sf, 2),
            token_bucket,
            hid_dim,
            ROW_BLOCK=FWD_ROW_BLOCK,
            COL_BLOCK=COL_BLOCK,
            APPLY_CLAMP=swiglu_limit > 0,
            swiglu_limit=float(swiglu_limit),
            clip_counts_ptr=clip_counts,
            ACTIVATION_MONITOR_ON=monitor_on,
            **params_to_kernel_kwargs(STATIC_TRITON_PARAMS_FWD, hid_dim,
                                      num_warps=4, maxnreg=160),
        )

        ctx.save_for_backward(x_c, p_c)
        return FusedQuantizedSwiGLU._make_qtensor(out, out_sf)

    @staticmethod
    def backward(ctx, grad_output: tp.Union[torch.Tensor, Float8BlockwiseQTensor]):
        x_c, p_c = ctx.saved_tensors
        total_tokens = ctx.input_tensor_shape[0]
        hid_dim      = ctx.input_tensor_shape[1] // 2
        blocks_per_row = hid_dim // COL_BLOCK
        device = x_c.device
        swiglu_limit = ctx.swiglu_limit

        assert isinstance(grad_output, torch.Tensor)
        grad_output = grad_output.contiguous()

        grad_in    = torch.empty((total_tokens, 2 * hid_dim),
                                 device=device, dtype=torch.float8_e4m3fn)
        grad_in_sf = torch.empty((total_tokens, 2 * blocks_per_row),
                                 device=device, dtype=torch.float32)
        grad_probs_partial = torch.empty((total_tokens, blocks_per_row),
                                         device=device, dtype=torch.float32)

        token_bucket = total_tokens // _SWIGLU_TOKEN_BUCKET_SIZE
        grid = (total_tokens // BWD_ROW_BLOCK, blocks_per_row)

        _swiglu_quant_bwd_kernel[grid](
            *tensor_to_kernel_args(grad_output, 2),
            *tensor_to_kernel_args(p_c, 1),
            *tensor_to_kernel_args(x_c, 2),
            *tensor_to_kernel_args(grad_in, 2),
            *tensor_to_kernel_args(grad_in_sf, 2),
            *tensor_to_kernel_args(grad_probs_partial, 2),
            token_bucket,
            hid_dim,
            ROW_BLOCK=BWD_ROW_BLOCK,
            COL_BLOCK=COL_BLOCK,
            APPLY_CLAMP=swiglu_limit > 0,
            swiglu_limit=float(swiglu_limit),
            **params_to_kernel_kwargs(STATIC_TRITON_PARAMS_BWD, hid_dim,
                                      num_warps=4, maxnreg=128),
        )

        grad_probs = grad_probs_partial.sum(dim=-1)
        return FusedQuantizedSwiGLU._make_qtensor(grad_in, grad_in_sf), grad_probs, None, None, None


def fused_swiglu_quantized_fn(
    input_tensor: torch.Tensor,
    probs: torch.Tensor,
    swiglu_limit: float = 0.0,
    power_two_max_round: bool = True,
    clip_counts: tp.Optional[torch.Tensor] = None,
) -> Float8BlockwiseQTensor:
    """Fused SwiGLU + FP8 row-wise quantization.

    swiglu_limit > 0 enables DeepSeek-style activation clipping:
        gate = clamp(gate, -inf, swiglu_limit)
        up   = clamp(up, -swiglu_limit, swiglu_limit)
    The clamp is compiled away (zero cost) when swiglu_limit=0.0.

    clip_counts: optional int32[2] persistent buffer (owned by the caller).
        When provided and swiglu_limit > 0, the kernel atomically accumulates
        [gate_clamped_count, up_clamped_count] into this buffer via ACTIVATION_MONITOR_ON.
        The caller must zero the buffer before each forward call and all-reduce
        it across the EP group afterward to obtain the layer-wide totals.
    """
    return FusedQuantizedSwiGLU.apply(input_tensor, probs, swiglu_limit, power_two_max_round, clip_counts)
