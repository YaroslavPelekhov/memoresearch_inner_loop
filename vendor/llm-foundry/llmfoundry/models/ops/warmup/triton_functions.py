from itertools import product
import logging
import typing as tp

import torch
import triton

from helpers import _build_permute_layout, _deepgemm_layout
from utils import TritonWarmupConfig, _BLOCK, _setup_standalone_path

log = logging.getLogger("triton_warmup")

_setup_standalone_path()

# ---------------------------------------------------------------------------
# Base kernel warmups
# ---------------------------------------------------------------------------

def warmup_blocks(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up block_quantization and block_quantization_transpose."""
    from llmfoundry.models.ops.float8.triton_kernels.block_quantization import \
        block_quantization_128x128_fn
    from llmfoundry.models.ops.float8.triton_kernels.block_transpose_fused_quantization import \
        block_transpose_fused_quantization_128x128_fn
    hid_dim = cfg.hid_dim
    int_dim = cfg.int_dim
    num_experts = cfg.num_groups
    w_in = torch.randn(num_experts, 2 * int_dim, hid_dim, device=device, dtype=torch.bfloat16)
    block_quantization_128x128_fn(w_in)
    block_transpose_fused_quantization_128x128_fn(w_in)
    w_out = torch.randn(num_experts, hid_dim, int_dim, device=device, dtype=torch.bfloat16)
    block_quantization_128x128_fn(w_out)
    block_transpose_fused_quantization_128x128_fn(w_out)
    log.info("block_quantization done")


def warmup_row_quantization(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up row_quantization."""
    from llmfoundry.models.ops.float8.triton_kernels.row_quantization import (
        _ROWQ_BUCKET_SIZE, row_quantization_1x128_fn)
    tokens_large = list(range(_ROWQ_BUCKET_SIZE,
                                   cfg.tokens_large + _ROWQ_BUCKET_SIZE,
                                   _ROWQ_BUCKET_SIZE))
    all_dims = (cfg.hid_dim, cfg.int_dim, 2 * cfg.int_dim)
    for i, tok in enumerate(tokens_large):
        for dim in all_dims:
            t = torch.randn(tok, dim, device=device, dtype=torch.bfloat16)
            row_quantization_1x128_fn(t)
        if i % 20 == 0:
            log.info("  row_quantization: %d / %d", i, len(tokens_large))
    log.info("row_quantization done")


def warmup_row2col_te(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up row2col TE."""
    from llmfoundry.models.ops.float8.triton_kernels.row2col_te_quantization import (
        _ROW2COL_BUCKET_SIZE, row2col_grouped_raw)
    tokens_large = list(range(_ROW2COL_BUCKET_SIZE,
                                   cfg.tokens_large + _ROW2COL_BUCKET_SIZE,
                                   _ROW2COL_BUCKET_SIZE))
    all_dims = (cfg.hid_dim, cfg.int_dim, 2 * cfg.int_dim)
    for i, tok in enumerate(tokens_large):
        for dim in all_dims:
            m_indices, group_sizes = _deepgemm_layout(tok, device)
            t_q = torch.randint(0, 256, (tok, dim), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
            scales_inv = torch.rand(tok, dim // _BLOCK, device=device, dtype=torch.float32)
            row2col_grouped_raw(t_q, scales_inv, [tok], m_indices, group_sizes)
        if i % 20 == 0:
            log.info("  row2col: %d / %d", i, len(tokens_large))
    log.info("row2col done")


def warmup_row2col_deepgemm(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up row2col_deepgemm."""
    from llmfoundry.models.ops.float8.triton_kernels.row2col_deepgemm_quantization import (
        _ROW2COL_BUCKET_SIZE, row2col_requantization_deepgemm_layout_fn)
    tokens_large = list(range(_ROW2COL_BUCKET_SIZE,
                                   cfg.tokens_large + _ROW2COL_BUCKET_SIZE,
                                   _ROW2COL_BUCKET_SIZE))
    all_dims = (cfg.hid_dim, cfg.int_dim, 2 * cfg.int_dim)
    for i, tok in enumerate(tokens_large):
        for dim in all_dims:
            m_indices, group_sizes = _deepgemm_layout(tok, device)
            t_q = torch.randint(0, 256, (tok, dim), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
            scales_inv = torch.rand(tok, dim // _BLOCK, device=device, dtype=torch.float32).T.contiguous().T
            row2col_requantization_deepgemm_layout_fn(t_q, m_indices, group_sizes, scales_inv)
        if i % 20 == 0:
            log.info("  row2col_deepgemm: %d / %d", i, len(tokens_large))
    log.info("row2col_deepgemm done")


def warmup_mla_q_rope(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up LlamaRotaryEmbedding + apply_rotary_emb (MLA Q-RoPE path)."""
    import dataclasses

    from llmfoundry.models.layers.custom_embedding import LlamaRotaryEmbedding
    from llmfoundry.models.layers.triton_rotary_embeddings import apply_rotary_emb_mla

    @dataclasses.dataclass
    class _RoPEConfig:
        max_position_embeddings: int
        head_dim: int
        init_device: object
        rope_theta: float = 100000.0
        dtype: object = torch.float32

    batch_sizes = cfg.batch_sizes
    seq_lens = cfg.seq_lens
    num_heads = cfg.rope_num_heads
    qk_nope_head_dim = cfg.qk_nope_head_dim
    qk_rope_head_dim = cfg.qk_rope_head_dim
    combos = product(batch_sizes, seq_lens)
    for batch_size, seq_len in combos:
        cfg = _RoPEConfig(
            max_position_embeddings=seq_len,
            head_dim=qk_rope_head_dim,
            init_device=device,
        )
        rotary_emb = LlamaRotaryEmbedding(cfg, device=device)
        query = torch.randn(
            batch_size, seq_len, num_heads, 
            qk_nope_head_dim + qk_rope_head_dim,
            device=device, dtype=torch.bfloat16,
            requires_grad=True,
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = rotary_emb(
            query, seq_len=seq_len,
            position_ids=position_ids, cu_seqlens=None,
        )
        apply_rotary_emb_mla(
            query, cos, sin,
            inplace=False, interleaved=True, head_offset=qk_nope_head_dim,
        )
    log.info("mla_q_rope done")


def warmup_permute_fwd(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _permute_and_pad_kernel forward pass."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
        _PERMUTE_TOKEN_BUCKET_SIZE, _permute_and_pad_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.utils import \
        tensor_to_kernel_args

    tokens_large = list(range(_PERMUTE_TOKEN_BUCKET_SIZE,
                                   cfg.tokens_large + _PERMUTE_TOKEN_BUCKET_SIZE,
                                   _PERMUTE_TOKEN_BUCKET_SIZE))
    hid_dim = cfg.hid_dim
    num_experts = cfg.num_groups
    for i, tok in enumerate(tokens_large):
        row_id_map, dst_to_padded, total_padded = _build_permute_layout(tok, num_experts, device, n_routed=1)
        total_token_rank_bucket = tok // _PERMUTE_TOKEN_BUCKET_SIZE
        grid = lambda META: (tok, triton.cdiv(hid_dim, META["BLOCK_SIZE"]))
        probs_in = torch.rand(tok, num_experts, device=device, dtype=torch.float32)
        scale = torch.rand(tok, hid_dim // _BLOCK, device=device, dtype=torch.float32)
        permuted_scale = torch.zeros(total_padded, hid_dim // _BLOCK, device=device, dtype=torch.float32)
        data_in = torch.randint(0, 256, (tok, hid_dim), device=device, dtype=torch.uint8)
        data_out = torch.zeros(total_padded, hid_dim, device=device, dtype=torch.uint8)
        probs_out = torch.zeros(total_padded, device=device, dtype=torch.float32)
        _permute_and_pad_kernel[grid](
            *tensor_to_kernel_args(data_in,        2),
            *tensor_to_kernel_args(data_out,       2),
            *tensor_to_kernel_args(row_id_map,     2),
            *tensor_to_kernel_args(probs_in,       2),
            *tensor_to_kernel_args(scale,          2),
            *tensor_to_kernel_args(probs_out,      1),
            *tensor_to_kernel_args(permuted_scale, 2),
            *tensor_to_kernel_args(dst_to_padded,  1),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            PERMUTE_SCALE=True,
            PERMUTE_PROBS=True,
        )
        data_in = torch.randint(0, 256, (tok, hid_dim), device=device, dtype=torch.bfloat16)
        data_out = torch.zeros(total_padded, hid_dim, device=device, dtype=torch.bfloat16)
        probs_out = torch.zeros(total_padded, device=device, dtype=torch.float32)
        _permute_and_pad_kernel[grid](
            *tensor_to_kernel_args(data_in,        2),
            *tensor_to_kernel_args(data_out,       2),
            *tensor_to_kernel_args(row_id_map,     2),
            *tensor_to_kernel_args(probs_in,       2),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(probs_out,      1),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(dst_to_padded,  1),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            PERMUTE_SCALE=False,
            PERMUTE_PROBS=True,
        )
        if i % 20 == 0:
            log.info("  permute_fwd: %d / %d", i, len(tokens_large))
    log.info("permute_fwd done")


def warmup_permute_bwd(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _permute_and_pad_kernel backward pass."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
        _PERMUTE_TOKEN_BUCKET_SIZE, _permute_and_pad_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.utils import \
        tensor_to_kernel_args

    tokens_large = list(range(_PERMUTE_TOKEN_BUCKET_SIZE, 
                                   cfg.tokens_large + _PERMUTE_TOKEN_BUCKET_SIZE,
                                   _PERMUTE_TOKEN_BUCKET_SIZE))
    hid_dim = cfg.hid_dim
    num_experts = cfg.num_groups
    for i, tok in enumerate(tokens_large):
        row_id_map, dst_to_padded, total_padded = _build_permute_layout(tok, num_experts, device, n_routed=1)
        total_token_rank_bucket = tok // _PERMUTE_TOKEN_BUCKET_SIZE
        grid = lambda META: (tok, triton.cdiv(hid_dim, META["BLOCK_SIZE"]))
        scale = torch.rand(tok, hid_dim // _BLOCK, device=device, dtype=torch.float32)
        permuted_scale = torch.zeros(total_padded, hid_dim // _BLOCK, device=device, dtype=torch.float32)
        data_in = torch.randint(0, 256, (tok, hid_dim), device=device, dtype=torch.uint8)
        data_out = torch.zeros(total_padded, hid_dim, device=device, dtype=torch.uint8)
        _permute_and_pad_kernel[grid](
            *tensor_to_kernel_args(data_in,        2),
            *tensor_to_kernel_args(data_out,       2),
            *tensor_to_kernel_args(row_id_map,     2),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(scale,          2),
            *tensor_to_kernel_args(None,           1),
            *tensor_to_kernel_args(permuted_scale, 2),
            *tensor_to_kernel_args(dst_to_padded,  1),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            PERMUTE_SCALE=True,
            PERMUTE_PROBS=False,
        )
        data_in = torch.randint(0, 256, (tok, hid_dim), device=device, dtype=torch.bfloat16)
        data_out = torch.zeros(total_padded, hid_dim, device=device, dtype=torch.bfloat16)
        _permute_and_pad_kernel[grid](
            *tensor_to_kernel_args(data_in,        2),
            *tensor_to_kernel_args(data_out,       2),
            *tensor_to_kernel_args(row_id_map,     2),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(None,           1),
            *tensor_to_kernel_args(None,           2),
            *tensor_to_kernel_args(dst_to_padded,  1),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            PERMUTE_SCALE=False,
            PERMUTE_PROBS=False,
        )
        if i % 20 == 0:
            log.info("  permute_bwd: %d / %d", i, len(tokens_large))
    log.info("permute_bwd done")


def warmup_unpermute_fwd(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _unpad_and_unpermute_kernel forward pass."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
        _PERMUTE_TOKEN_BUCKET_SIZE, _unpad_and_unpermute_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.utils import \
        tensor_to_kernel_args
    tokens_large = list(range(_PERMUTE_TOKEN_BUCKET_SIZE,
                                   cfg.tokens_large + _PERMUTE_TOKEN_BUCKET_SIZE,
                                   _PERMUTE_TOKEN_BUCKET_SIZE))
    hid_dim = cfg.hid_dim
    num_experts = cfg.num_groups
    for i, tok in enumerate(tokens_large):
        row_id_map, dst_to_padded, total_padded = _build_permute_layout(tok, num_experts, device, n_routed=1)
        total_token_rank_bucket = tok // _PERMUTE_TOKEN_BUCKET_SIZE
        grid = lambda META: (tok, triton.cdiv(hid_dim, META["BLOCK_SIZE"]))
        data_in = torch.randn(total_padded, hid_dim, device=device, dtype=torch.bfloat16)
        output = torch.zeros(tok, hid_dim, device=device, dtype=torch.bfloat16)

        _unpad_and_unpermute_kernel[grid](
            *tensor_to_kernel_args(data_in,       2),
            *tensor_to_kernel_args(output,        2),
            *tensor_to_kernel_args(row_id_map,    2),
            *tensor_to_kernel_args(dst_to_padded, 1),
            *tensor_to_kernel_args(None,          1),
            *tensor_to_kernel_args(None,          2),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            UNPERMUTE_PROBS=False,
        )
        if i % 20 == 0:
            log.info("  unpermute_fwd: %d / %d", i, len(tokens_large))
    log.info("unpermute_fwd done")


def warmup_unpermute_bwd(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _unpad_and_unpermute_kernel backward pass."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
        _PERMUTE_TOKEN_BUCKET_SIZE, _unpad_and_unpermute_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.utils import \
        tensor_to_kernel_args
    tokens_large = list(range(_PERMUTE_TOKEN_BUCKET_SIZE,
                                   cfg.tokens_large + _PERMUTE_TOKEN_BUCKET_SIZE,
                                   _PERMUTE_TOKEN_BUCKET_SIZE))
    hid_dim = cfg.hid_dim
    num_experts = cfg.num_groups
    for i, tok in enumerate(tokens_large):
        row_id_map, dst_to_padded, total_padded = _build_permute_layout(tok, num_experts, device, n_routed=1)
        total_token_rank_bucket = tok // _PERMUTE_TOKEN_BUCKET_SIZE
        grid = lambda META: (tok, triton.cdiv(hid_dim, META["BLOCK_SIZE"]))
        data_in = torch.randn(total_padded, hid_dim, device=device, dtype=torch.bfloat16)
        output = torch.zeros(tok, hid_dim, device=device, dtype=torch.bfloat16)
        grad_permuted_probs = torch.randn(total_padded, device=device, dtype=torch.float32)
        grad_input_probs = torch.zeros(tok, num_experts, device=device, dtype=torch.float32)
        _unpad_and_unpermute_kernel[grid](
            *tensor_to_kernel_args(data_in,             2),
            *tensor_to_kernel_args(output,              2),
            *tensor_to_kernel_args(row_id_map,          2),
            *tensor_to_kernel_args(dst_to_padded,       1),
            *tensor_to_kernel_args(grad_permuted_probs, 1),
            *tensor_to_kernel_args(grad_input_probs,    2),
            num_experts=num_experts,
            total_token_rank_bucket=total_token_rank_bucket,
            UNPERMUTE_PROBS=True,
        )
        if i % 20 == 0:
            log.info("  unpermute_bwd: %d / %d", i, len(tokens_large))
    log.info("unpermute_bwd done")


def warmup_fused_swiglu(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up both fwd and bwd Triton kernels by calling them directly."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization import (
        _SWIGLU_TOKEN_BUCKET_SIZE, _swiglu_quant_bwd_kernel,
        _swiglu_quant_fwd_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.utils import \
        tensor_to_kernel_args
    tokens_large = list(range(_SWIGLU_TOKEN_BUCKET_SIZE,
                                   cfg.tokens_large + _SWIGLU_TOKEN_BUCKET_SIZE,
                                   _SWIGLU_TOKEN_BUCKET_SIZE))
    int_dim = cfg.int_dim
    for i, tok in enumerate(tokens_large):
        bpr = int_dim // _BLOCK  # blocks_per_row
        token_bucket = tok // _SWIGLU_TOKEN_BUCKET_SIZE

        # ---- forward ----
        input_tensor = torch.randn(tok, 2 * int_dim, device=device, dtype=torch.bfloat16).contiguous()
        probs = torch.randn(tok, device=device, dtype=torch.float32).contiguous()
        out_swiglu = torch.empty(tok, int_dim, device=device, dtype=torch.float8_e4m3fn)
        out_sf = torch.empty(tok, bpr, device=device, dtype=torch.float32)
        in_q = torch.empty(tok, 2 * int_dim, device=device, dtype=torch.float8_e4m3fn)
        in_sf = torch.empty(tok, 2 * bpr, device=device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(tok, META["BLOCK_SIZE"]), triton.cdiv(int_dim, META["BLOCK_SIZE"]))

        _swiglu_quant_fwd_kernel[grid](  # type: ignore[index]
            *tensor_to_kernel_args(input_tensor, 2),
            *tensor_to_kernel_args(probs,        1),
            *tensor_to_kernel_args(out_swiglu,   2),
            *tensor_to_kernel_args(out_sf,       2),
            *tensor_to_kernel_args(in_q,         2),
            *tensor_to_kernel_args(in_sf,        2),
            token_bucket,
            int_dim,
        )

        # ---- backward ----
        grad_out = torch.randn(tok, int_dim, device=device, dtype=torch.bfloat16).contiguous()
        grad_in_q = torch.empty(tok, 2 * int_dim, device=device, dtype=torch.float8_e4m3fn)
        grad_in_sf = torch.empty(tok, 2 * bpr, device=device, dtype=torch.float32)
        grad_probs_p = torch.empty(tok, bpr, device=device, dtype=torch.float32)

        _swiglu_quant_bwd_kernel[grid](  # type: ignore[index]
            *tensor_to_kernel_args(grad_out,     2),
            *tensor_to_kernel_args(probs,        1),
            *tensor_to_kernel_args(in_q,         2),
            *tensor_to_kernel_args(in_sf,        2),
            *tensor_to_kernel_args(grad_in_q,    2),
            *tensor_to_kernel_args(grad_in_sf,   2),
            *tensor_to_kernel_args(grad_probs_p, 2),
            token_bucket,
            int_dim,
        )
        if i % 20 == 0:
            log.info("  fused_swiglu: %d / %d", i, len(tokens_large))
    log.info("fused_swiglu done")

# ---------------------------------------------------------------------------
# Dense-layer kernel warmups
# ---------------------------------------------------------------------------

def warmup_dense_block_quantize(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _block_quantize_kernel (dense weight quantize-transpose) from row2col_dense.py."""
    from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import \
        block_quantization_transpose
    hid_dense_dim = cfg.dense_hid_dim
    int_dense_dim = cfg.dense_int_dim
    hid_dim = cfg.hid_dim
    int_dim = cfg.int_dim
    for h_dim, i_dim in zip((hid_dense_dim, hid_dim), (int_dense_dim, int_dim)):
        w_in = torch.randn(2 * i_dim, h_dim, device=device, dtype=torch.bfloat16)
        block_quantization_transpose(w_in)
        block_quantization_transpose(w_in.T.contiguous())
        w_out = torch.randn(h_dim, i_dim, device=device, dtype=torch.bfloat16)
        block_quantization_transpose(w_out)
        block_quantization_transpose(w_out.T.contiguous())
    log.info("dense_block_quantize done")


def warmup_dense_row2col(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up _scaling_aware_fp8_transpose_kernel from row2col_dense.py."""
    from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import \
        blockwise_scaling_aware_fp8_transpose
    tokens_small = list(range(_BLOCK, cfg.tokens_small + _BLOCK, _BLOCK))
    all_dims = (cfg.dense_hid_dim, cfg.dense_int_dim, 2 * cfg.dense_int_dim,
                cfg.hid_dim, cfg.int_dim, 2 * cfg.int_dim)
    for i, rows in enumerate(tokens_small):
        for dim in all_dims:
            bpr = dim // _BLOCK
            rw_data = torch.randint(0, 256, (rows, dim), device=device, dtype=torch.uint8)
            rw_scale = torch.rand(rows, bpr, device=device, dtype=torch.float32)
            blockwise_scaling_aware_fp8_transpose(rw_data, rw_scale)
        if i % 20 == 0:
            log.info("  dense_row2col: %d / %d", i, len(tokens_small))
    log.info("dense_row2col done")


def warmup_dense_swiglu(cfg: TritonWarmupConfig, device: torch.device) -> None:
    """Warms up swiglu_quant_forward_kernel_opt and swiglu_quant_backward_kernel_opt."""
    from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization_dense import (
        swiglu_quant_backward_kernel_opt, swiglu_quant_forward_kernel_opt)

    tokens_small = list(range(_BLOCK, cfg.tokens_small + _BLOCK, _BLOCK))
    int_dense_dim = cfg.dense_int_dim
    int_dim = cfg.int_dim
    for i, tok in enumerate(tokens_small):
        for dim in (int_dense_dim, int_dim):
            bpr   = dim // _BLOCK
            bpr_m4 = (bpr + 3) // 4 * 4
            gate = torch.randn(tok, dim, device=device, dtype=torch.bfloat16)
            up = torch.randn(tok, dim, device=device, dtype=torch.bfloat16)
            out = torch.empty(tok, dim, device=device, dtype=torch.float8_e4m3fn)
            out_scale = torch.empty(bpr_m4, tok, device=device, dtype=torch.float32)

            grid = lambda META: (tok, triton.cdiv(dim, META["TILE_SIZE"]))  # noqa: E731

            swiglu_quant_forward_kernel_opt[grid](  # type: ignore[index]
                gate, up, out,
                None, None,       # gate_fp8, up_fp8 (only needed when FP8_INPUT_STORE=True)
                out_scale,
                None, None,       # gate_scale, up_scale
                dim,
                FP8_INPUT_STORE=False,
                POWER_TWO_MAX_ROUND=True,
            )

            grad_out = torch.randn(tok, dim, device=device, dtype=torch.bfloat16)
            grad_gate = torch.empty(tok, dim, device=device, dtype=torch.float8_e4m3fn)
            grad_up = torch.empty(tok, dim, device=device, dtype=torch.float8_e4m3fn)
            grad_gate_scale = torch.empty(bpr_m4, tok, device=device, dtype=torch.float32)
            grad_up_scale = torch.empty(bpr_m4, tok, device=device, dtype=torch.float32)

            swiglu_quant_backward_kernel_opt[grid](  # type: ignore[index]
                gate, up,
                None, None,      # gate_scale, up_scale (only needed when FP8_INPUT_STORE=True)
                grad_out,
                grad_gate, grad_up,
                grad_gate_scale, grad_up_scale,
                dim,
                FP8_INPUT_STORE=False,
                POWER_TWO_MAX_ROUND=True,
            )

        if i % 20 == 0:
            log.info("  dense_swiglu: %d / %d", i, len(tokens_small))
    log.info("dense_swiglu done")

# ---------------------------------------------------------------------------
# Group and kernel object registry (only kernels decorated with @triton.autotune)
# ---------------------------------------------------------------------------

# Mapping between the kernel name and the keys and ignore keys on save from @triton.autotune.
KERNEL_KEY_METADATA: tp.Dict[str, tp.List[str]] = {
    "block_quantization": {
            "keys":                ["tensor_size_row", "tensor_size_col"],
    },
    "block_quantization_transpose": {
            "keys":                ["tensor_size_row", "tensor_size_col"],
    },
    "row_quantization": {
            "keys":                ["total_token_bucket", "tensor_size_col"],
            "ignore_keys_on_save": ["total_token_bucket"]
    },
    "row2col_te": {
            "keys":                ["total_token_bucket", "tensor_size_col"],
            "ignore_keys_on_save": ["total_token_bucket"]
    },
    "row2col_deepgemm": {
            "keys":                ["total_token_bucket", "tensor_size_col"],
            "ignore_keys_on_save": ["total_token_bucket"]
    },
    "swiglu_fwd": {
            "keys":                ["input_token_bucket", "hid_dim"],
            "ignore_keys_on_save": ["input_token_bucket"]
    },
    "swiglu_bwd": {
            "keys":                ["grad_token_bucket",  "hid_dim"],
            "ignore_keys_on_save": ["grad_token_bucket"]
    },
    "permute_fwd": {
            "keys":                ["total_token_rank_bucket", "input_size_col"],
            "ignore_keys_on_save": ["total_token_rank_bucket"]
    },
    "permute_bwd": {
            "keys":                ["total_token_rank_bucket", "input_size_col"],
            "ignore_keys_on_save": ["total_token_rank_bucket"]
    },
    "unpermute_fwd": {
            "keys":                ["total_token_rank_bucket", "input_size_col"],
            "ignore_keys_on_save": ["total_token_rank_bucket"]
    },
    "unpermute_bwd": {
            "keys":                ["total_token_rank_bucket", "input_size_col"],
            "ignore_keys_on_save": ["total_token_rank_bucket"]
    },
    "mla_q_rope": {
            "keys":                ["seqlen", "rotary_dim", "seqlen_ro", "head_offset", "nheads"],
            "ignore_keys_on_save": ["seqlen", "seqlen_ro"]
    },
    # Dense kernels
    "dense_fp8_transpose": {
            "keys":                ["rows", "cols"],
            "ignore_keys_on_save": ["rows"]
    },
    "dense_block_quantize": {
            "keys":                ["tensor_size_row", "tensor_size_col"],
    },
    "dense_swiglu_fwd": {
            "keys":                ["n_cols"],
    },
    "dense_swiglu_bwd": {
            "keys":                ["n_cols"],
    },
}

# Mapping between the warmup group name from config and the function.
WARMUP_FNS: tp.Dict[str, tp.Callable[[torch.device], None]] = {
    "blocks_quantization":          warmup_blocks,
    "row_quantization":             warmup_row_quantization,
    "row2col_te":                   warmup_row2col_te,
    "row2col_deepgemm":             warmup_row2col_deepgemm,
    "fused_swiglu":                 warmup_fused_swiglu,
    "mla_q_rope":                   warmup_mla_q_rope,
    "permute_fwd":                  warmup_permute_fwd,
    "permute_bwd":                  warmup_permute_bwd,
    "unpermute_fwd":                warmup_unpermute_fwd,
    "unpermute_bwd":                warmup_unpermute_bwd,
    # Dense-layer groups (opt-in via submit_warmup.py --dense)
    "dense_block_quantize":         warmup_dense_block_quantize,
    "dense_row2col":                warmup_dense_row2col,
    "dense_fused_swiglu":           warmup_dense_swiglu,
}

# Maps each group to the autotune kernel cache names it populates.
# Groups with no @triton.autotune only JIT-compile; nothing to persist.
GROUP_TO_AUTOTUNE_KERNELS: tp.Dict[str, tp.List[str]] = {
    "blocks_quantization":          ["block_quantization", "block_quantization_transpose"],
    "row_quantization":             ["row_quantization"],
    "row2col_te":                   ["row2col_te"],
    "row2col_deepgemm":             ["row2col_deepgemm"],
    "fused_swiglu":                 ["swiglu_fwd", "swiglu_bwd"],
    "mla_q_rope":                   ["mla_q_rope"],                                    # JIT-only: no @triton.autotune to persist
    "permute_fwd":                  ["permute_fwd"],  
    "permute_bwd":                  ["permute_bwd"],  
    "unpermute_fwd":                ["unpermute_fwd"],
    "unpermute_bwd":                ["unpermute_bwd"],
    # Dense-layer groups
    "dense_block_quantize":         ["dense_block_quantize"],
    "dense_row2col":                ["dense_fp8_transpose"],
    "dense_fused_swiglu":           ["dense_swiglu_fwd", "dense_swiglu_bwd"],
}

def _get_autotuned_kernels() -> tp.Dict[str, tp.Any]:
    """Return {cache_name: live Triton JITFunction} for every autotuned kernel."""
    from llmfoundry.models.ops.float8.triton_kernels.block_quantization import \
        _block_quantization_kernel
    from llmfoundry.models.ops.float8.triton_kernels.block_transpose_fused_quantization import \
        _block_quantize_kernel_transpose
    from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization import (
        _swiglu_quant_bwd_kernel, _swiglu_quant_fwd_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization_dense import \
        swiglu_quant_backward_kernel_opt as _dense_swiglu_bwd_kernel
    from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization_dense import \
        swiglu_quant_forward_kernel_opt as _dense_swiglu_fwd_kernel
    from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
        _permute_and_pad_kernel, _unpad_and_unpermute_kernel)
    from llmfoundry.models.ops.float8.triton_kernels.row2col_deepgemm_quantization import \
        _rowwise_to_columnwise_deepgemm_layout_kernel as _r2c_deepgemm
    from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import \
        _block_quantize_kernel as _dense_block_quant_kernel
    from llmfoundry.models.ops.float8.triton_kernels.row2col_dense_kernels import \
        _scaling_aware_fp8_transpose_kernel as _dense_fp8_transpose_kernel
    from llmfoundry.models.ops.float8.triton_kernels.row2col_te_quantization import \
        _rowwise_to_grouped_columnwise_kernel as _r2c_te
    from llmfoundry.models.ops.float8.triton_kernels.row_quantization import \
        _rowwise_1d_quantize_kernel
    from llmfoundry.models.layers.triton_rotary_embeddings import \
        rotary_kernel_mla as _rotary_kernel_mla
    return {
        "block_quantization":           _block_quantization_kernel,
        "block_quantization_transpose": _block_quantize_kernel_transpose,
        "row_quantization":             _rowwise_1d_quantize_kernel,
        "row2col_te":                   _r2c_te,
        "row2col_deepgemm":             _r2c_deepgemm,
        "swiglu_fwd":                   _swiglu_quant_fwd_kernel,
        "swiglu_bwd":                   _swiglu_quant_bwd_kernel,
        "permute_fwd":                  _permute_and_pad_kernel,
        "permute_bwd":                  _permute_and_pad_kernel,
        "unpermute_fwd":                _unpad_and_unpermute_kernel,
        "unpermute_bwd":                _unpad_and_unpermute_kernel,
        "dense_fp8_transpose":          _dense_fp8_transpose_kernel,
        "dense_block_quantize":         _dense_block_quant_kernel,
        "dense_swiglu_fwd":             _dense_swiglu_fwd_kernel,
        "dense_swiglu_bwd":             _dense_swiglu_bwd_kernel,
        "mla_q_rope":                   _rotary_kernel_mla,
    }
