# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers."""

import copy
import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import composer.utils.dist as dist
from composer.utils.profiler_annotation import decorator_forward_backward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn.bert_padding import pad_input, unpad_input
from packaging import version
from transformers import PretrainedConfig

from llmfoundry.models.layers.custom_embedding import (
    DeepseekV2RotaryEmbedding,
    DeepseekV2YarnRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_emb,
    apply_rotary_emb_kv_,
    apply_rotary_emb_kv_packed_,
    apply_rotary_emb_qkv_,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek,
    yarn_get_mscale,
)
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.layers.triton_rotary_embeddings import apply_rotary_emb_mla
from llmfoundry.models.parallel.sequence.all_to_all import SeqAllToAll
from llmfoundry.models.parallel.sequence.ring_flash_attn import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_varlen_kvpacked_func,
    llama3_flash_attn_varlen_qkvpacked_func,
    ring_flash_attn_func,
    ring_flash_attn_kvpacked_func,
    ring_flash_attn_qkvpacked_func,
    ring_flash_attn_varlen_func,
    ring_flash_attn_varlen_kvpacked_func,
    ring_flash_attn_varlen_qkvpacked_func,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_kvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_varlen_func,
    zigzag_ring_flash_attn_varlen_kvpacked_func,
    zigzag_ring_flash_attn_varlen_qkvpacked_func,
)
from llmfoundry.models.parallel.tensor.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from llmfoundry.models.utils.flash_attn_import import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

log = logging.getLogger(__name__)

# avoid recompiles
torch.fx.experimental._config.use_duck_shape = False

try:
    from causal_conv1d import causal_conv1d_fn
except:
    raise ImportError("Failed to import causal_conv1d. Please ensure that causal-conv1d package is installed in docker image")

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
except:
    raise ImportError("Failed to import flash-linear-attention.")

import triton
import triton.language as tl


flash_attn_varlen_func = torch.compiler.disable(flash_attn_varlen_func)
flash_attn_kvpacked_func = torch.compiler.disable(flash_attn_kvpacked_func)
flash_attn_varlen_kvpacked_func = torch.compiler.disable(
    flash_attn_varlen_kvpacked_func
)
flash_attn_varlen_qkvpacked_func = torch.compiler.disable(
    flash_attn_varlen_qkvpacked_func
)
flash_attn_qkvpacked_func = torch.compiler.disable(flash_attn_qkvpacked_func)

pad_input = torch.compiler.disable(pad_input)
unpad_input = torch.compiler.disable(unpad_input)
rearrange = torch.compiler.disable(rearrange)


def _reset_is_causal(
    num_query_tokens: int, num_key_tokens: int, original_is_causal: bool
):
    # disable causal when it is not needed
    # necessary for flash & triton for generation with kv_cache
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                "MPT does not support query and key with different number of tokens, unless number of query tokens is 1."
            )
        else:
            return False
    return original_is_causal


@torch.compile(fullgraph=True, dynamic=True)
def _compiled_apply_sigmoid_gate_fp32(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    gate_scale = torch.sigmoid(gate.float()).to(dtype=attn_output.dtype)
    return attn_output * gate_scale


def _apply_sigmoid_gate_fp32(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    torch._dynamo.mark_dynamic(attn_output, 0)
    torch._dynamo.decorators.mark_unbacked(attn_output, 0)
    torch._dynamo.mark_dynamic(attn_output, 1)
    torch._dynamo.mark_dynamic(gate, 0)
    torch._dynamo.decorators.mark_unbacked(gate, 0)
    torch._dynamo.mark_dynamic(gate, 1)
    with torch.autocast(device_type=attn_output.device.type, dtype=torch.float32):
        return _compiled_apply_sigmoid_gate_fp32(attn_output, gate)


def scaled_multihead_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]
]:
    q = rearrange(query, "b s (h d) -> b h s d", h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, "b s (h d) -> b h d s", h=kv_n_heads)
    v = rearrange(value, "b s (h d) -> b h s d", h=kv_n_heads)

    if past_key_value is not None:
        # attn_impl: flash & triton use kernels which expect input shape [b, s, h, d_head].
        # kv_cache is therefore stored using that shape.
        # attn_impl: torch stores the kv_cache in the ordering which is most advantageous
        # for its attn computation ie
        # keys are stored as tensors with shape [b, h, d_head, s] and
        # values are stored as tensors with shape [b, h, s, d_head]
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

        if (attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k) or (
            attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q
        ):
            raise RuntimeError(
                f"attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}."
            )
        attn_weight = attn_weight + attn_bias

    min_val = torch.finfo(q.dtype).min

    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn(
                "Propagating key_padding_mask to the attention module "
                + "and applying it within the attention module can cause "
                + "unnecessary computation/memory usage. Consider integrating "
                + "into attn_bias once and passing that to each attention "
                + "module instead."
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val
        )

    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout_p, training=training, inplace=True
        )

    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, "b h s d -> b s (h d)")

    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value


def check_valid_inputs(
    *tensors: torch.Tensor, valid_dtypes: Optional[List[torch.dtype]] = None
):
    if valid_dtypes is None:
        valid_dtypes = [torch.float16, torch.bfloat16]
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f"{tensor.dtype=} must be in {valid_dtypes=}.")
        if not tensor.is_cuda:
            raise TypeError(f"Inputs must be cuda tensors ({tensor.is_cuda=}).")


def flash_attn_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]
]:
    try:
        from flash_attn import bert_padding, flash_attn_interface  # type: ignore # yapf: disable # isort: skip
    except:
        raise RuntimeError("Please install flash-attn==1.0.3.post0")

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

    if attn_bias is not None:
        raise NotImplementedError(f"attn_bias not implemented for flash attn.")

    batch_size, seqlen = query.shape[:2]

    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1) :]

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = bert_padding.unpad_input(
        query, query_padding_mask
    )
    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=n_heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k, _ = bert_padding.unpad_input(
        key, key_padding_mask
    )
    key_unpad = rearrange(
        key_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else n_heads
    )

    value_unpad, _, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(
        value_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else n_heads
    )

    if multiquery:
        # Expanding a tensor does not allocate new memory, but only creates a new
        # view on the existing tensor where a dimension of size one is expanded
        # to a larger size by setting the stride to 0.
        # - pytorch docs
        #
        # hopefully the kernels can utilize this and we're jot just wasting BW here
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(
            value_unpad.size(0), n_heads, value_unpad.size(-1)
        )

    dropout_p = dropout_p if training else 0.0

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)

    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=reset_is_causal,
        return_attn_probs=needs_weights,
    )

    output = bert_padding.pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, batch_size, seqlen
    )
    return output, None, past_key_value


def triton_flash_attn_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]
]:
    try:
        from llmfoundry.models.layers.flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse("2.0.0"):
            _installed = True
            # if torch1.13.1 revert to using triton flash attn from HazyResearch
            # with flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            # installing triton-pre-mlir works for both torch1.13.1 and torch2.0+
            # default recommendation is to install this variant
            raise RuntimeError(
                "Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU "
                + "and `pip install .[gpu]` if installing from llm-foundry source or "
                + "`pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` "
                + "if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). "
                + "Note: (1) requires you have CMake and PyTorch already installed."
            )

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

    if dropout_p:
        raise NotImplementedError(f"Dropout not implemented for attn_impl: triton.")
    dropout_p = dropout_p if training else 0.0

    if needs_weights:
        raise NotImplementedError(f"attn_impl: triton cannot return attn weights.")

    if key_padding_mask is not None:
        warnings.warn(
            "Propagating key_padding_mask to the attention module "
            + "and applying it within the attention module can cause "
            + "unnecessary computation/memory usage. Consider integrating "
            + "into attn_bias once and passing that to each attention "
            + "module instead."
        )
        b_size, s_k = key_padding_mask.shape[:2]

        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)

        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min
        )

    query = rearrange(query, "b s (h d) -> b s h d", h=n_heads)
    key = rearrange(key, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)
    value = rearrange(value, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)

    if multiquery:
        # necessary to repeat instead of expand tensor because
        # output contains NaN in edge cases such as with head dimension = 8
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(  # type: ignore
        query, key, value, attn_bias, reset_is_causal, softmax_scale
    )

    output = attn_output.view(*attn_output.shape[:2], -1)  # type: ignore

    return output, None, past_key_value


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = "triton",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        fc_kwargs = {}
        if fc_type != "te":
            fc_kwargs["device"] = device
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            3 * self.d_model,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            norm_class = resolve_norm_class(norm_type)
            self.q_ln = norm_class(self.d_model, device=device)
            self.k_ln = norm_class(self.d_model, device=device)

        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    "While `attn_impl: triton` can be faster than `attn_impl: flash` "
                    + "it uses more memory. When training larger models this can trigger "
                    + "alloc retries which hurts performance. If encountered, we recommend "
                    + "using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`."
                )
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    "Using `attn_impl: torch`. If your model does not use `alibi` or "
                    + "`prefix_lm` we recommend using `attn_impl: flash` otherwise "
                    + "we recommend using `attn_impl: triton`."
                )
        else:
            raise ValueError(f"{attn_impl=} is an invalid setting.")

        self.out_proj = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        needs_weights: bool = False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value


class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = "triton",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop

        fc_kwargs = {}
        if fc_type != "te":
            fc_kwargs["device"] = device
        # NOTE: if we ever want to make attn TensorParallel, I'm pretty sure we'll
        # want to split Wqkv into Wq and Wkv where Wq can be TensorParallel but
        # Wkv shouldn't be TensorParallel
        # - vchiley
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](
            d_model,
            d_model + 2 * self.head_dim,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            norm_class = resolve_norm_class(norm_type)
            self.q_ln = norm_class(d_model, device=device)
            self.k_ln = norm_class(
                self.head_dim, device=device
            )

        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    "While `attn_impl: triton` can be faster than `attn_impl: flash` "
                    + "it uses more memory. When training larger models this can trigger "
                    + "alloc retries which hurts performance. If encountered, we recommend "
                    + "using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`."
                )
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    "Using `attn_impl: torch`. If your model does not use `alibi` or "
                    + "`prefix_lm` we recommend using `attn_impl: flash` otherwise "
                    + "we recommend using `attn_impl: triton`."
                )
        else:
            raise ValueError(f"{attn_impl=} is an invalid setting.")

        self.out_proj = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        needs_weights: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2
        )

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value


def attn_bias_shape(
    attn_impl: str,
    n_heads: int,
    seq_len: int,
    alibi: bool,
    prefix_lm: bool,
    causal: bool,
    use_sequence_id: bool,
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f"{attn_impl=} is an invalid setting.")


def build_attn_bias(
    attn_impl: str,
    attn_bias: torch.Tensor,
    n_heads: int,
    seq_len: int,
    causal: bool = False,
    alibi: bool = False,
    alibi_bias_max: int = 8,
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(
                build_alibi_bias(
                    n_heads,
                    seq_len,
                    full=not causal,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                )
            )
        return attn_bias
    else:
        raise ValueError(f"{attn_impl=} is an invalid setting.")


def gen_slopes(
    n_heads: int, alibi_bias_max: int = 8, device: Optional[torch.device] = None
):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)

    if _n_heads != n_heads:
        # if n_heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]

    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    full: bool = False,
    alibi_bias_max: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(
        1, 1, 1, seq_len
    )
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device
        ).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
):  # pylint: disable=invalid-name,unnecessary-lambda-assignment
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, q_d = q.shape
    _, seqlen_k, nheads_k, k_d = k.shape
    _, _, _, v_d = v.shape
    assert k.shape[:-1] == (batch_size, seqlen_k, nheads_k)
    assert v.shape[:-1] == (batch_size, seqlen_k, nheads_k)
    # q/k_head_dim = 192 and v_head_dim = 128 for DeepSeek-like MLA
    assert (q_d == k_d == v_d) or (q_d == k_d == 192 and v_d == 128)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
            q, query_padding_mask
        )

        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )  # noqa: E731

    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        max_seqlen_q = seqlen_q

        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )  # noqa: E731

    if key_padding_mask is not None:
        k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        return (qkv_unpad, cu_seqlens_q, max_seqlen_q, qkv, output_pad_fn)

    if kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        return (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
        )

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
    )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.apply_qk_norm = config.apply_qk_norm

        if config.attention_hidden_size is None:
            self.attention_hidden_size = config.hidden_size
        else:
            self.attention_hidden_size = config.attention_hidden_size
        self.num_heads = config.num_attention_heads
        self.init_device = config.init_device

        assert self.attention_hidden_size % self.num_heads == 0
        if config.head_dim:
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.attention_hidden_size // self.num_heads

        self.tp_size = dist.get_tp_group_size()
        if self.tp_size is None:
            self.tp_size = 1

        sp_size = dist.get_sp_group_size()
        if sp_size is None:
            sp_size = 1

        self.num_key_value_heads = config.num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads

        assert self.num_heads % self.num_key_value_heads == 0
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        assert self.num_heads % (self.tp_size * sp_size) == 0
        assert self.num_key_value_heads % (self.tp_size * sp_size) == 0

        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = self.num_key_value_heads // self.tp_size

        self.deterministic = config.deterministic_attention
        self.gated_attention = getattr(config, "gated_attention", False)
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)
        if self.gated_attention and self.tp_size > 1:
            raise AssertionError("Functional need fixes for tp > 1")
        if self.gated_attention and self.pretraining_tp > 1:
            raise AssertionError(
                "`gated_attention` is not supported with `pretraining_tp > 1`."
            )

        if self.tp_size == 1:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                device=self.init_device,
            )
            if self.gated_attention:
                self.gate_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.attention_bias,
                    device=self.init_device,
                )
        else:
            self.q_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                self.hidden_size,
                self.num_heads * self.head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )
            self.k_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )
            self.v_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )
            self.o_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
                self.num_heads * self.head_dim,
                self.hidden_size,
                config=config,
                bias=False,
                input_is_parallel=True,
            )
            if self.gated_attention:
                self.gate_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    config=config,
                    bias=config.attention_bias,
                    gather_output=False,
                )

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.q_proj._init_std = init_std
            self.k_proj._init_std = init_std
            self.v_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = init_std / math.sqrt(2 * (self.layer_idx + 1))
                else:
                    log.info(
                        "`LlamadAttention` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `o_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.o_proj._init_std = init_std

        # NOTE(m1kol): Qwen3-style QK norms
        if self.apply_qk_norm:
            norm_class = resolve_norm_class(self.config.norm_type)
            self.q_norm = norm_class.from_config(config, input_dim=self.head_dim)
            self.k_norm = norm_class.from_config(config, input_dim=self.head_dim)

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.init_device)
        self.softmax_scale = self.head_dim ** (-0.5)

        self.varlen_input = getattr(self.config, "varlen_input", False)

        warnings.warn(
            "Current attentions implementation is not optimized, try to use `LlamaPackedAttention` class instead."
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel

        attention_mask: [bsz, q_len]
        """
        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )
        assert not use_cache, "custom attention does not support cache"
        # pylint: disable=duplicate-code

        if self.tp_size > 1:
            query_states = self.q_proj(hidden_states, **tp_kwargs)
            key_states = self.k_proj(hidden_states, **tp_kwargs)
            value_states = self.v_proj(hidden_states, **tp_kwargs)
            if self.gated_attention:
                gate = self.gate_proj(hidden_states, **tp_kwargs)

        elif self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if self.gated_attention:
                gate = self.gate_proj(hidden_states)

        # All-to-all gather query, key and value states
        # ---
        # TODO (Sbr; for future): Could change applying all-to-all after computing and applying
        # rotary and repeating KV to save some memory. For this would need to get the
        # whole cos and sin for whole seq len --> apply appropriate part depending on
        # SP rank.
        # ---
        # NOTE: `position_ids` are not used.

        # NOTE (Sbr): Disabled sequence parallel processing in eval model as
        # there's an error on generative eval tasks (gsm8k; hanging with fail
        # in the end of computing.
        # ---
        # TODO (Sbr): Fix SP errors on eval and remove this.
        if self.training:
            sp_size = dist.get_sp_group_size() or 1
            is_sp_active = sp_size > 1
        else:
            sp_size = 1
            is_sp_active = False

        if is_sp_active:
            # Scatter by hidden, gather by sequence.
            # [bs, N/SP, D] --> [bs, N, D/SP]
            # ---
            # params: (input, gather_idx, scatter_idx, group)
            sp_group = dist.get_sp_group()
            query_states = SeqAllToAll.apply(query_states, 1, 2, sp_group)
            key_states = SeqAllToAll.apply(key_states, 1, 2, sp_group)
            value_states = SeqAllToAll.apply(value_states, 1, 2, sp_group)

        bsz, q_len, *_ = query_states.size()

        # Divide `num_heads` and `num_key_value_heads` by TP and SP sizes as they are set during `__init__()`
        # and know nothing about it.
        query_states = query_states.view(
            bsz, q_len, self.num_heads // (self.tp_size * sp_size), self.head_dim
        )
        key_states = key_states.view(
            bsz,
            q_len,
            self.num_key_value_heads // (self.tp_size * sp_size),
            self.head_dim,
        )
        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads // (self.tp_size * sp_size),
            self.head_dim,
        )
        # [bsz, q_len, nh, hd]

        if self.apply_qk_norm:
            query_states = self.q_norm(query_states).to(query_states.dtype)
            key_states = self.k_norm(key_states).to(key_states.dtype)

        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cu_seqlens_for_rope = cu_seqlens
        max_seqlen_for_rope = max_seqlen
        if isinstance(cu_seqlens_for_rope, dict):
            if not "cu_seqlens_q" in cu_seqlens_for_rope:
                err_msg = (
                    f"No `cu_seqlens_q` in provided `cu_seqlens` dict. Check the code "
                    "there must be some error regarding SP Ring implementation work. "
                    f"Present keys are {tuple(cu_seqlens_for_rope.keys())}."
                )
                raise RuntimeError(err_msg)
            cu_seqlens_for_rope = cu_seqlens_for_rope["cu_seqlens_q"]
            max_seqlen_for_rope = max_seqlen_for_rope["max_seqlen_q"]

        # NOTE(makolesov): Disable varlen on eval as `position_ids` and `cu_seqlens` are
        # not properly formed for eval datasets.
        _is_varlen_active = self.training and self.varlen_input

        cos, sin = self.rotary_emb(
            value_states,
            seq_len=kv_seq_len,
            position_ids=None if _is_varlen_active else position_ids,
            cu_seqlens=None if _is_varlen_active else cu_seqlens_for_rope,
        )

        if max_seqlen_for_rope is not None:
            max_seqlen_for_rope = min(max_seqlen_for_rope, cos.shape[0])

        query_states = apply_rotary_emb(
            query_states.view(-1, *query_states.shape[2:])
            if _is_varlen_active
            else query_states,
            cos,
            sin,
            inplace=True,
            cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
            max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
        ).view(*query_states.shape)
        key_states = apply_rotary_emb(
            key_states.view(-1, *key_states.shape[2:])
            if _is_varlen_active
            else key_states,
            cos,
            sin,
            inplace=True,
            cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
            max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
        ).view(*key_states.shape)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if output_attentions:
            warnings.warn(
                "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
            )

        #
        # flash-attn v2 start
        #

        if self.training:
            # during training q,k,v always have same seqlen
            assert key_states.shape == query_states.shape, (
                f"key_states.shape={key_states.shape}, query_states.shape={query_states.shape}"
            )
            is_causal = True
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = key_states.shape == query_states.shape

        if cu_seqlens is not None and max_seqlen is not None:
            # special handling using sample packing
            q_unpad = rearrange(query_states, "b s h d -> (b s) h d")
            k_unpad = rearrange(key_states, "b s h d -> (b s) h d")
            v_unpad = rearrange(value_states, "b s h d -> (b s) h d")
            qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)

            output = flash_attn_varlen_qkvpacked_func(
                qkv_unpad,
                cu_seqlens,
                max_seqlen,
                0.0,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        elif query_states.shape == key_states.shape:
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query_states,
                key_states,
                value_states,
                qkvpacked=False,
                # We have disabled _prepare_decoder_attention_mask in LlamaModel
                # the attention_mask should be the same as the key_padding_mask
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)
        else:
            if attention_mask is None or attention_mask.all().item():
                output = flash_attn_kvpacked_func(
                    query_states,
                    torch.stack([key_states, value_states], 2),
                    causal=is_causal,
                    deterministic=self.deterministic,
                )
            else:
                (  # pylint: disable=unbalanced-tuple-unpacking
                    q_unpad,
                    kv_unpad,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    _,
                    _,
                    output_pad_fn,
                ) = generate_qkv(
                    query_states,
                    key_states,
                    value_states,
                    kvpacked=True,
                    key_padding_mask=attention_mask,
                    query_padding_mask=attention_mask[:, -query_states.size(1) :]
                    if attention_mask is not None
                    else None,
                )

                output_unpad = flash_attn_varlen_kvpacked_func(
                    q_unpad,
                    kv_unpad,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    0.0,
                    softmax_scale=None,
                    causal=is_causal,
                    deterministic=self.deterministic,
                )
                output = output_pad_fn(output_unpad)

        attn_output = output
        if attn_output.size() != (
            bsz,
            q_len,
            self.num_heads // (self.tp_size * sp_size),
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads // (self.tp_size * sp_size), self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        #
        # flash-attn v2 end
        #

        # All-to-all gather output
        if is_sp_active:
            # Scatter by sequence, gather by hidden.
            attn_output = SeqAllToAll.apply(attn_output, 2, 1, dist.get_sp_group())

        if self.gated_attention:
            attn_output = _apply_sigmoid_gate_fp32(attn_output, gate)

        if self.tp_size > 1:
            attn_output = self.o_proj(attn_output)
        elif self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.attention_hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.attention_hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.pretraining_tp)
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class LlamaPackedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.apply_qk_norm = config.apply_qk_norm
        assert not self.apply_qk_norm, (
            "QK norm is not yet fully working in `LlamaPackedAttention`. "
            "Please set `apply_qk_norm: false` in training config."
        )

        if config.attention_hidden_size is None:
            self.attention_hidden_size = config.hidden_size
        else:
            self.attention_hidden_size = config.attention_hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads

        assert self.num_heads % self.num_key_value_heads == 0, (
            "`num_heads` should be divisible by `num_key_value_heads`, but got "
            f"{self.num_heads} heads and {self.num_key_value_heads} kv heads."
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        assert self.attention_hidden_size % self.num_heads == 0, (
            "`attention_hidden_size` should be divisible by `num_heads`, but got "
            f"{self.attention_hidden_size} attention size and {self.num_heads} heads."
        )
        if config.head_dim:
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.attention_hidden_size // self.num_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.init_device = config.init_device

        self.tp_size = dist.get_tp_group_size()
        if self.tp_size is None:
            self.tp_size = 1

        sp_size = dist.get_sp_group_size()
        if sp_size is None:
            sp_size = 1

        self.is_ring_attn = self.config.attention_type in RING_ATTN_CLASSES

        self.num_key_value_heads = config.num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads

        if self.is_ring_attn:
            tp_x_sp_size = self.tp_size
        else:
            tp_x_sp_size = dist.get_tp_sp_group_size()

        tp_x_sp_size = tp_x_sp_size or 1
        assert self.num_heads % self.num_key_value_heads == 0
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.deterministic = config.deterministic_attention
        if getattr(config, "gated_attention", False):
            raise AssertionError(
                "`gated_attention` is not supported for packed or ring attention classes."
            )

        assert self.num_heads % tp_x_sp_size == 0, (
            f"Number of attention heads should be divisible by the TP*SP size for all2all SP and by TP size for ring SP, but got "
            f"{self.num_heads} heads and tp, sp = {self.tp_size}, {sp_size} and divider {tp_x_sp_size}."
        )
        assert self.num_key_value_heads % tp_x_sp_size == 0, (
            f"Number of key-value heads should be divisible by the TP*SP size for all2all SP and by TP size for ring SP, but got "
            f"{self.num_key_value_heads} heads and tp, sp = {self.tp_size}, {sp_size} and divider {tp_x_sp_size}."
        )

        # Divide only by tp_size here as we perform all-to-all sync after/before qkv/ouput rearrange.
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = self.num_key_value_heads // self.tp_size

        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_key_value_heads)

        if self.tp_size == 1:
            self.qkv_proj = nn.Linear(
                self.hidden_size,
                qkv_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )

            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                device=self.init_device,
            )
        else:
            self.qkv_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                self.hidden_size,
                qkv_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )

            self.o_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
                self.num_heads * self.head_dim,
                self.hidden_size,
                config=config,
                bias=False,
                input_is_parallel=True,
            )

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.init_device)

        # NOTE(m1kol): Qwen3-style QK norms
        if self.apply_qk_norm:
            norm_class = resolve_norm_class(self.config.norm_type)
            self.q_norm = norm_class.from_config(config, input_dim=self.head_dim)
            self.k_norm = norm_class.from_config(config, input_dim=self.head_dim)

        # NOTE (Sber): set 'custom_init_std' attr with (fan_in, fan_out) value to apply fixed std normal init.
        # We don't set _fused attr because in gqa setup it leads to incorrect init with xavier
        self.qkv_proj.custom_init_std = (self.hidden_size, self.hidden_size)
        self.o_proj._is_residual = True

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.qkv_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = init_std / math.sqrt(2 * (self.layer_idx + 1))
                else:
                    log.info(
                        "`LlamaPackedAttention` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `o_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.o_proj._init_std = init_std

        self.varlen_input = getattr(self.config, "varlen_input", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        hidden_states: [bsz, seq_len, hidden_size]
        attention_mask: [bsz, seq_len]
        """
        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )

        if self.tp_size > 1:
            qkv_states = self.qkv_proj(hidden_states, **tp_kwargs)
        else:
            qkv_states = self.qkv_proj(hidden_states)

        bsz, seq_len, _ = qkv_states.size()

        # [bsz, seqlen, qkv_dim]
        if self.num_key_value_heads == self.num_heads:
            qkv_states = rearrange(
                qkv_states, "b s (three h d) -> b s three h d", three=3, d=self.head_dim
            )
        else:
            query_states = rearrange(
                qkv_states[..., : self.num_heads_per_rank * self.head_dim],
                "... (h d) -> ... h d",
                d=self.head_dim,
            )
            kv_states = rearrange(
                qkv_states[..., self.num_heads_per_rank * self.head_dim :],
                "... (two hkv d) -> ... two hkv d",
                two=2,
                d=self.head_dim,
            )
        # [bsz, seq_len, nh, hd]

        # NOTE (Sbr): Disabled sequence parallel processing in eval model as
        # there's an error on generative eval tasks (gsm8k; hanging with fail
        # in the end of computing.
        # ---
        # TODO (Sbr): Fix SP errors on eval and remove this.
        if self.training:
            sp_size = dist.get_sp_group_size() or 1
            is_a2a_sp_active = sp_size > 1 and not self.is_ring_attn
        else:
            sp_size = 1
            is_a2a_sp_active = False

        if is_a2a_sp_active:
            # Scatter by hidden, gather by sequence.
            # [bs, N/SP, 3, head_num, head_dim] --> [bs, N, 3, head_num/SP, head_dim]
            sp_group = dist.get_sp_group()

            def gather_scatter(states, scatter_dim=3):
                states = SeqAllToAll.apply(states, 1, scatter_dim, sp_group)
                return states

            if self.num_key_value_heads == self.num_heads:
                qkv_states = gather_scatter(qkv_states, scatter_dim=3)
            else:
                query_states = gather_scatter(query_states, scatter_dim=2)
                kv_states = gather_scatter(kv_states, scatter_dim=3)
            seq_len = seq_len * sp_size

        # Apply QK norm via split/stack to avoid problems with graph breaks when
        # using inplace operations via slices.
        if self.apply_qk_norm:
            if self.num_key_value_heads == self.num_heads:
                query_states, key_states, value_states = torch.split(
                    qkv_states, 1, dim=2
                )
            else:
                key_states, value_states = torch.split(kv_states, 1, dim=2)

            query_states = self.q_norm(query_states).to(query_states.dtype)
            key_states = self.k_norm(key_states).to(key_states.dtype)

            # TODO(m1kol): As split returns a view of the original tensor we could just
            # perform the operations above and thus would change tensors inplace. So no
            # need to do the stacking here, just wasting memory. The same principle is
            # applied for rotary application, for example.
            #
            # Need to test it and remove after if it's ok.
            if self.num_key_value_heads == self.num_heads:
                qkv_states = torch.stack(
                    (query_states, key_states, value_states), dim=2
                )
            else:
                key_states, value_states = torch.stack(
                    (key_states, value_states), dim=2
                )

        seq_len_with_past_key_value = seq_len
        if past_key_value is not None:
            assert use_cache, "past_key_value is none, when using cache.."
            seq_len_with_past_key_value += past_key_value[0].shape[1]

        cu_seqlens_for_rope = cu_seqlens
        max_seqlen_for_rope = max_seqlen
        if isinstance(cu_seqlens_for_rope, dict):
            if not "cu_seqlens_q" in cu_seqlens_for_rope:
                err_msg = (
                    f"No `cu_seqlens_q` in provided `cu_seqlens` dict. Check the code "
                    "there must be some error regarding SP Ring implementation work. "
                    f"Present keys are {tuple(cu_seqlens_for_rope.keys())}."
                )
                raise RuntimeError(err_msg)
            cu_seqlens_for_rope = cu_seqlens_for_rope["cu_seqlens_q"]
            max_seqlen_for_rope = max_seqlen_for_rope["max_seqlen_q"]
        # apply rotary emb

        # NOTE(makolesov): Disable varlen on eval as `position_ids` and `cu_seqlens` are
        # not properly formed for eval datasets.
        _is_varlen_active = self.training and self.varlen_input

        if is_left_padded_eval:
            # due to the triton kernel limitation in `apply_rotary_emb`, we only pass cos/sin for position_ids[0],
            # which might be shifted in case of left-padded batches (e.g., [0,0,...]); however, we actually need
            # [0,1,...], so we ignore position_ids in this case.
            cos, sin = self.rotary_emb(
                qkv_states,
                seq_len=seq_len_with_past_key_value,
                cu_seqlens=None if _is_varlen_active else cu_seqlens_for_rope,
            )
        else:
            cos, sin = self.rotary_emb(
                qkv_states,
                seq_len=seq_len_with_past_key_value,
                position_ids=None if _is_varlen_active else position_ids,
                cu_seqlens=None if _is_varlen_active else cu_seqlens_for_rope,
            )

        if max_seqlen_for_rope is not None:
            max_seqlen_for_rope = min(max_seqlen_for_rope, cos.shape[0])

        if self.num_key_value_heads == self.num_heads:
            assert not use_cache, "we don`t support cache for mha at the moment."
            qkv_states = apply_rotary_emb_qkv_(
                qkv_states,
                cos,
                sin,
                cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
                max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
            )  # inplace
        else:
            kv_cache_enabled = past_key_value is not None
            query_states, kv_states = apply_rotary_emb_kv_packed_(
                query_states,
                kv_states,
                cos,
                sin,
                position_ids,
                is_left_padded_eval=is_left_padded_eval,
                use_cache=use_cache,
                is_training=self.training,
                kv_cache_enabled=kv_cache_enabled,  # second generation step with use_cache
                cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
                max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
            )

        if past_key_value is not None:
            past_kv_states = past_key_value[0]
            kv_states = torch.cat([past_kv_states, kv_states], dim=1)

        past_key_value = None
        if use_cache:
            assert self.num_key_value_heads != self.num_heads, (
                "we don`t support cache for mha at the moment."
            )
            past_key_value = (kv_states,)

        if output_attentions:
            warnings.warn(
                "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
            )

        #
        # flash-attn v2 start
        #

        if self.num_key_value_heads == self.num_heads:  # MHA
            output = self._flash_attention_mha_forward(
                qkv_states,
                attention_mask,
                cu_seqlens,
                max_seqlen,
            )
        else:  # GQA
            output = self._flash_attention_gqa_forward(
                query_states, kv_states, attention_mask, cu_seqlens, max_seqlen
            )

        attn_output = output

        # All-to-all gather output
        if is_a2a_sp_active:
            # Scatter by sequence, gather by hidden.
            attn_output = SeqAllToAll.apply(attn_output, 2, 1, dist.get_sp_group())
            seq_len = seq_len // sp_size

        if attn_output.size() != (bsz, seq_len, self.num_heads_per_rank, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, seq_len, self.num_heads_per_rank, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        #
        # flash-attn v2 end
        #

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _flash_attention_mha_forward(
        self,
        qkv_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ):
        is_causal = True
        bsz = qkv_states.shape[0]

        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            qkv_states = rearrange(qkv_states, "b s three h d -> (b s) three h d")

            output = flash_attn_varlen_qkvpacked_func(
                qkv_states,
                cu_seqlens,
                max_seqlen,
                0.0,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        elif attention_mask is None:
            output = flash_attn_qkvpacked_func(
                qkv_states,
                causal=is_causal,
                deterministic=self.deterministic,
            )
        else:  # need to unpad
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                qkv_states[:, :, 0],
                qkv_states[:, :, 1],
                qkv_states[:, :, 2],
                qkvpacked=False,  # can be packed
                # We have disabled _prepare_decoder_attention_mask in LlamaModel
                # the attention_mask should be the same as the key_padding_mask
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -qkv_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)

        return output

    def _flash_attention_gqa_forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ):
        is_causal = True
        bsz = query_states.shape[0]

        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            query_states = rearrange(query_states, "b s ... -> (b s) ...")
            kv_states = rearrange(kv_states, "b s ... -> (b s) ...")
            output = flash_attn_varlen_kvpacked_func(
                query_states,
                kv_states,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                deterministic=self.deterministic,
            )

            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)

        elif attention_mask is None or attention_mask.size(1) == 1:
            output = flash_attn_kvpacked_func(
                query_states,
                kv_states,
                causal=is_causal,
                deterministic=self.deterministic,
            )
        else:  # need to unpad
            (
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query_states,
                kv_states[:, :, 0],
                kv_states[:, :, 1],
                kvpacked=True,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query_states.size(1) :]
                if attention_mask is not None
                else None,
            )

            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)
        return output


class LlamaPackedRingAttention(LlamaPackedAttention):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx=layer_idx)

        if self.config.sp_split_type == "equal":
            self.flash_attn_qkvpacked_func = ring_flash_attn_qkvpacked_func
            self.flash_attn_kvpacked_func = ring_flash_attn_kvpacked_func
            self.flash_attn_varlen_qkvpacked_func = (
                ring_flash_attn_varlen_qkvpacked_func
            )
            self.flash_attn_varlen_kvpacked_func = ring_flash_attn_varlen_kvpacked_func
            self.flash_attn_varlen_func = ring_flash_attn_varlen_func
        elif self.config.sp_split_type == "zigzag":
            self.flash_attn_qkvpacked_func = zigzag_ring_flash_attn_qkvpacked_func
            self.flash_attn_kvpacked_func = zigzag_ring_flash_attn_kvpacked_func
            self.flash_attn_varlen_qkvpacked_func = (
                zigzag_ring_flash_attn_varlen_qkvpacked_func
            )
            self.flash_attn_varlen_kvpacked_func = (
                zigzag_ring_flash_attn_varlen_kvpacked_func
            )
            self.flash_attn_varlen_func = zigzag_ring_flash_attn_varlen_func
        elif self.config.sp_split_type == "llama3":
            self.flash_attn_varlen_qkvpacked_func = (
                llama3_flash_attn_varlen_qkvpacked_func
            )
            self.flash_attn_varlen_kvpacked_func = (
                llama3_flash_attn_varlen_kvpacked_func
            )
            self.flash_attn_varlen_func = llama3_flash_attn_varlen_func
        else:
            raise NotImplementedError(f"{self.config.sp_split_type=} is not supported!")

    def _flash_attention_mha_forward(
        self,
        qkv_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        max_seqlen: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ):
        sp_size = dist.get_sp_group_size() or 1
        if not self.training or sp_size == 1:  # disable ring with eval mode or sp=1
            return super()._flash_attention_mha_forward(
                qkv_states, attention_mask, cu_seqlens, max_seqlen
            )

        is_causal = True
        bsz = qkv_states.shape[0]

        sp_group = dist.get_sp_group()

        # NOTE: The ring with sp_split_type='equal' and varlen_input=True produces incorrect losses
        # (significant discrepancies are observed when comparing plots against reference)
        assert not self.config.sp_split_type == "equal" or not getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'equal' does not currently support varlen_input."

        assert not self.config.sp_split_type == "zigzag" or not getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'zigzag' does not currently support varlen_input."

        assert not self.config.sp_split_type == "llama3" or getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'llama3' should only be used with varlen_input."

        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            qkv_states = rearrange(qkv_states, "b s three h d -> (b s) three h d")
            if self.config.sp_split_type == "llama3":
                output = self.flash_attn_varlen_qkvpacked_func(
                    qkv_states,
                    cu_seqlens_q=cu_seqlens["cu_seqlens_q"],
                    cu_seqlens_k=cu_seqlens["cu_seqlens_k"],
                    max_seqlen_q=max_seqlen["max_seqlen_q"],
                    max_seqlen_k=max_seqlen["max_seqlen_k"],
                    heads_k_stride=self.config.llama3_ring_heads_k_stride,
                    local_k_slice=cu_seqlens["local_k_slice"],
                    dropout_p=0.0,
                    causal=is_causal,
                    group=sp_group,
                    deterministic=self.deterministic,
                )
            else:
                output = self.flash_attn_varlen_qkvpacked_func(
                    qkv_states,
                    cu_seqlens,
                    max_seqlen,
                    0.0,
                    causal=is_causal,
                    group=sp_group,
                    deterministic=self.deterministic,
                )

            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)

        elif attention_mask is None:
            output = self.flash_attn_qkvpacked_func(
                qkv_states,
                causal=is_causal,
                group=sp_group,
                deterministic=self.deterministic,
            )

        else:  # need to unpad
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                qkv_states[:, :, 0],
                qkv_states[:, :, 1],
                qkv_states[:, :, 2],
                qkvpacked=False,  # can be packed
                # We have disabled _prepare_decoder_attention_mask in LlamaModel
                # the attention_mask should be the same as the key_padding_mask
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -qkv_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = self.flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                max_seqlen_q,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                group=sp_group,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)

        return output

    def _flash_attention_gqa_forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        max_seqlen: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ):
        sp_size = dist.get_sp_group_size() or 1
        if not self.training or sp_size == 1:  # disable ring with eval mode or sp=1
            return super()._flash_attention_gqa_forward(
                query_states, kv_states, attention_mask, cu_seqlens, max_seqlen
            )

        is_causal = True
        bsz = query_states.shape[0]

        sp_group = dist.get_sp_group()

        # NOTE: The ring with sp_split_type='equal' and varlen_input=True produces incorrect losses
        # (significant discrepancies are observed when comparing plots against reference)
        assert not self.config.sp_split_type == "equal" or not getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'equal' does not currently support varlen_input."

        assert not self.config.sp_split_type == "zigzag" or not getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'zigzag' does not currently support varlen_input."

        assert not self.config.sp_split_type == "llama3" or getattr(
            self.config, "varlen_input", False
        ), "sp_split_type 'llama3' should only be used with varlen_input."

        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            query_states = rearrange(query_states, "b s ... -> (b s) ...")
            kv_states = rearrange(kv_states, "b s ... -> (b s) ...")
            if self.config.sp_split_type == "llama3":
                query_states = query_states.view(-1, self.num_heads, self.head_dim)
                kv_states = kv_states.view(
                    -1, 2, self.num_key_value_heads, self.head_dim
                )

                output = self.flash_attn_varlen_kvpacked_func(
                    query_states,
                    kv_states,
                    cu_seqlens_q=cu_seqlens["cu_seqlens_q"],
                    cu_seqlens_k=cu_seqlens["cu_seqlens_k"],
                    max_seqlen_q=max_seqlen["max_seqlen_q"],
                    max_seqlen_k=max_seqlen["max_seqlen_k"],
                    heads_k_stride=self.config.llama3_ring_heads_k_stride,
                    local_k_slice=cu_seqlens["local_k_slice"],
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=is_causal,
                    group=sp_group,
                    deterministic=self.deterministic,
                )
            else:
                output = self.flash_attn_varlen_kvpacked_func(
                    query_states,
                    kv_states,
                    cu_seqlens,
                    max_seqlen,
                    0.0,
                    softmax_scale=None,
                    causal=is_causal,
                    group=sp_group,
                    deterministic=self.deterministic,
                )

            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)

        elif attention_mask is None:
            output = self.flash_attn_kvpacked_func(
                query_states,
                kv_states,
                causal=is_causal,
                group=sp_group,
                deterministic=self.deterministic,
            )
        else:  # need to unpad
            (
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query_states,
                kv_states[:, :, 0],
                kv_states[:, :, 1],
                kvpacked=True,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = self.flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                max_seqlen_q,
                0.0,
                softmax_scale=None,
                causal=is_causal,
                group=sp_group,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)
        return output


@torch.compile
def correct_qk_rope_deepseek(x):
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return x


class LlamaLatentAttention(nn.Module):
    """
    Multi-headed Latent Attention (MLA)

    Check out the original paper: https://arxiv.org/pdf/2405.04434
    and the reference implementation: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
    """

    def __init__(self, config: PretrainedConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.apply_torch_compile = config.apply_torch_compile_to_projections
        self.use_custom_rotary_kernel = config.use_custom_rotary_kernel
        self.apply_qk_norm = config.apply_qk_norm

        if config.attention_hidden_size is None:
            self.attention_hidden_size = config.hidden_size
        else:
            self.attention_hidden_size = config.attention_hidden_size
        self.num_heads = config.num_attention_heads
        self.num_krope_heads = config.num_krope_heads
        if self.num_krope_heads is None:
            self.num_krope_heads = self.num_heads
        assert self.num_krope_heads == 1 or self.num_krope_heads == self.num_heads, (
            "Only num_krope_heads = num_heads or num_krope_heads = 1 is supported, "
            f"num_krope_heads = {self.num_krope_heads}, num_heads = {self.num_heads}"
        )

        self.num_key_value_heads = config.num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads

        assert config.num_attention_heads == self.num_key_value_heads, (
            "GQA for MLA is not supported (does it even make sense?)"
        )

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim  # V has no rope part
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.use_mla_scaling_factor = bool(
            getattr(config, "use_mla_scaling_factor", False)
        )
        # LongCat-Flash style scale correction for low-rank latent states.
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            q_hidden_dim = self.q_lora_rank
        else:
            q_hidden_dim = self.hidden_size
        if self.kv_lora_rank is not None and self.kv_lora_rank > 0:
            kv_hidden_dim = self.kv_lora_rank
        else:
            kv_hidden_dim = self.hidden_size

        if self.use_mla_scaling_factor:
            self.alpha_q = math.sqrt(self.hidden_size / q_hidden_dim)
            self.alpha_kv = math.sqrt(self.hidden_size / kv_hidden_dim)
        else:
            self.alpha_q = 1.0
            self.alpha_kv = 1.0

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.init_device = config.init_device

        self.tp_size = dist.get_tp_group_size()
        if self.tp_size is None:
            self.tp_size = 1

        self.sp_size = dist.get_sp_group_size()
        if self.sp_size is None:
            self.sp_size = 1

        assert (self.tp_size > 1 and not self.apply_qk_norm) or self.tp_size == 1, (
            "QK norm is not yet supported in `LlamaLatentAttention` with `tp_size` > 1. "
            "Please set `apply_qk_norm: false` in training config."
        )

        self.is_ring_attn = self.config.attention_type in RING_ATTN_CLASSES

        if self.is_ring_attn:
            self.tp_x_sp_size = self.tp_size
        else:
            self.tp_x_sp_size = self.tp_size * self.sp_size

        self.deterministic = config.deterministic_attention

        # Divide only by tp_size here as we perform all-to-all sync after/before qkv/ouput rearrange.
        if self.num_heads % self.tp_size != 0:
            raise RuntimeError(
                f"Number of heads is not divisable by TP size. Got {self.num_heads} heads and {self.tp_size} TP size."
            )
        self.num_heads_per_rank = self.num_heads // self.tp_size

        if self.num_krope_heads != 1 and self.num_krope_heads % self.tp_size != 0:
            raise RuntimeError(
                f"Number of k_rope heads is not divisable by TP size. Got {self.num_krope_heads} heads and {self.tp_size} TP size."
            )
        if self.num_krope_heads != 1:
            self.num_krope_heads_per_rank = self.num_krope_heads // self.tp_size
        else:
            self.num_krope_heads_per_rank = 1

        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention and self.tp_size > 1:
            raise AssertionError("Functional need fixes for tp > 1")
        norm_class = resolve_norm_class(config.norm_type)

        if self.tp_size == 1:
            if self.q_lora_rank == 0:
                self.q_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=config.attention_bias,
                    device=self.init_device,
                )
            else:
                self.dq_proj = nn.Linear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=config.attention_bias,
                    device=self.init_device,
                )
                self.q_norm = norm_class.from_config(
                    config, input_dim=self.q_lora_rank
                )
                self.uq_proj = nn.Linear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=config.attention_bias,
                    device=self.init_device,
                )

            self.dkv_proj = nn.Linear(
                self.hidden_size,
                self.kv_lora_rank,
                bias=config.attention_bias,
                device=self.init_device,
            )

            self.kv_norm = norm_class.from_config(
                config, input_dim=self.kv_lora_rank
            )

            self.uk_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )

            self.uv_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_heads * self.v_head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )

            self.kr_proj = nn.Linear(
                self.hidden_size,
                self.num_krope_heads * self.qk_rope_head_dim,
                bias=config.attention_bias,
                device=self.init_device,
            )

            self.o_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                device=self.init_device,
            )
            if self.gated_attention:
                self.gate_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.v_head_dim,
                    bias=False,
                    device=self.init_device,
                )
        else:
            if self.q_lora_rank == 0:
                self.q_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    config=config,
                    bias=config.attention_bias,
                    gather_output=False,
                )
            else:
                self.dq_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
                    self.hidden_size,
                    self.q_lora_rank,
                    config=config,
                    bias=config.attention_bias,
                    input_is_parallel=False,
                )
                self.q_norm = norm_class.from_config(
                    config, input_dim=self.q_lora_rank
                )
                self.uq_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                    self.q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    config=config,
                    bias=config.attention_bias,
                    gather_output=False,
                )

            self.dkv_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
                self.hidden_size,
                self.kv_lora_rank,
                config=config,
                bias=config.attention_bias,
                input_is_parallel=False,
            )

            self.kv_norm = norm_class.from_config(
                config, input_dim=self.kv_lora_rank
            )

            self.uk_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                config.kv_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )

            self.uv_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                config.kv_lora_rank,
                self.num_heads * self.v_head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )

            self.kr_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                self.hidden_size,
                self.num_krope_heads * self.qk_rope_head_dim,
                config=config,
                bias=config.attention_bias,
                gather_output=False,
            )

            self.o_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                config=config,
                bias=False,
                input_is_parallel=True,
            )
            if self.gated_attention:
                self.gate_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
                    self.hidden_size,
                    self.num_heads * self.v_head_dim,
                    config=config,
                    bias=False,
                    gather_output=False,
                )

        if self.apply_qk_norm:
            # NOTE(m1kol): This chage should be ok, tested it sometime.
            self.qk_q_norm = norm_class.from_config(
                config, input_dim=self.num_heads_per_rank * self.qk_head_dim
            )
            self.qk_k_norm = norm_class.from_config(
                config, input_dim=self.num_heads_per_rank * self.qk_head_dim
            )

            # TODO(m1kol): Qwen3-style QK norm. Need to rework this to work properly,
            # then replace the above.
            # self.qk_q_norm = norm_class(self.qk_head_dim)
            # self.qk_k_norm = norm_class(self.qk_head_dim)

        self.o_proj._is_residual = True

        config_for_rope = copy.deepcopy(self.config)
        config_for_rope.head_dim = self.config.qk_rope_head_dim

        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_origin == "deepseek":
            raise AssertionError("This functionality is not yet tested")
            if not self.config.rope_scaling or not self.config.rope_scaling["type"]:
                self.rotary_emb = DeepseekV2RotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            elif self.config.rope_scaling["type"] == "yarn":
                scaling_factor = self.config.rope_scaling["factor"]
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
                self.softmax_scale = self.q_head_dim ** (-0.5)
                if self.config.rope_scaling is not None:
                    mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
                    scaling_factor = self.config.rope_scaling["factor"]
                    if mscale_all_dim:
                        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                        self.softmax_scale = self.softmax_scale * mscale * mscale

            else:
                scaling_type = (
                    self.config.rope_scaling.get("type")
                    if self.config.rope_scaling is not None
                    else None
                )
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        elif self.config.rope_origin == "custom":
            self.rotary_emb = LlamaRotaryEmbedding(
                config_for_rope, device=self.init_device
            )
            self.softmax_scale = (
                self.q_head_dim ** (-0.5) * self.rotary_emb.attention_scaling**2
            )

        else:
            raise ValueError(f"Unknown RoPE origin {self.config.rope_origin}")

        if self.apply_torch_compile:
            error_msg = (
                "Compiling MLA `_compute_qkv` method makes program crash because of recompilation "
                "and reaching `recompile_limit` (activation monitor causes a lot of them especially). "
                "Set `apply_torch_compile_to_projections: false` in training YAML config (MLA section)."
            )
            assert not self.apply_torch_compile, error_msg

            self._compute_qkv = torch.compile(self._compute_qkv)

        self.varlen_input = getattr(self.config, "varlen_input", False)

        if self.use_custom_rotary_kernel and self.varlen_input:
            raise ValueError(
                "Varlen input is not supported for custom rotary kernel in MLA."
            )

        if config.init_type in {"dclm", "olmo3"}:
            raise RuntimeError(f"{config.init_type} init is not supported for MLA.")

    def _get_hidden_states_shapes(
        self, hidden_states: torch.Tensor, logical_batch_size: Optional[int] = None
    ):
        if logical_batch_size is None:
            batch_size, seq_len, _ = hidden_states.size()
        else:
            _, batch_size_x_seq_len_per_rank, _ = hidden_states.size()
            batch_size_x_seq_len = batch_size_x_seq_len_per_rank * self.tp_size
            batch_size = logical_batch_size
            seq_len = batch_size_x_seq_len // batch_size

        return batch_size, seq_len

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        logical_batch_size: Optional[int] = None,
    ):
        """Compute query, key, and value tensors from hidden states"""
        batch_size, seq_len = self._get_hidden_states_shapes(
            hidden_states, logical_batch_size
        )

        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if self.tp_size > 1 and logical_batch_size is not None
            else dict()
        )

        if self.q_lora_rank == 0:
            query = self.q_proj(hidden_states, **tp_kwargs)
            query = self.alpha_q * query
        else:
            q_latent = self.q_norm(self.dq_proj(hidden_states))
            q_latent = self.alpha_q * q_latent
            query = self.uq_proj(q_latent, **tp_kwargs)

        latent = self.dkv_proj(hidden_states)
        k_rope = self.kr_proj(hidden_states, **tp_kwargs)
        if self.tp_size > 1 and self.num_krope_heads == 1:
            k_rope = gather_from_tensor_model_parallel_region(k_rope)

        latent = self.kv_norm(latent)
        latent = self.alpha_kv * latent

        # Project latent to keys and values
        k_nope = self.uk_proj(latent, **tp_kwargs)
        value = self.uv_proj(latent, **tp_kwargs)

        if self.apply_qk_norm:
            query = self.qk_q_norm(query).to(query.dtype)
            # TODO(m1kol): This is  not good, need to change this for keys somehow.
            key = self.qk_k_norm(torch.cat([k_nope, k_rope], dim=-1)).to(k_nope.dtype)
            k_nope, k_rope = torch.split(
                key, [k_nope.shape[-1], k_rope.shape[-1]], dim=-1
            )

        # Reshape tensors
        query = query.view(
            batch_size, seq_len, self.num_heads_per_rank, self.qk_head_dim
        )
        k_nope = k_nope.view(
            batch_size, seq_len, self.num_heads_per_rank, self.qk_nope_head_dim
        )
        # Transform `k_rope` to a proper shape here as we just use it later on. And it is needed
        # for all-to-all SP. `self.num_krope_heads_per_rank` is 1 if num_krope_heads is 1, otherwise
        # it's a real calculated value.
        k_rope = k_rope.view(
            batch_size, seq_len, self.num_krope_heads_per_rank, self.qk_rope_head_dim
        )
        k_rope = k_rope.expand(
            batch_size, seq_len, self.num_heads_per_rank, self.qk_rope_head_dim
        ).contiguous()
        value = value.view(
            batch_size, seq_len, self.num_heads_per_rank, self.v_head_dim
        )

        return query, k_nope, k_rope, value
    
    @decorator_forward_backward()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        hidden_states: [bsz, seq_len, hidden_size]
        attention_mask: [bsz, seq_len]
        """
        assert use_cache == False, "custom attention does not support cache"
        assert past_key_value is None, (
            "custom attention does not support past_key_value"
        )
        batch_size, seq_len = self._get_hidden_states_shapes(
            hidden_states, logical_batch_size
        )

        # Compute q, k, v tensors using the (potentially compiled) function
        query, k_nope, k_rope, value = self._compute_qkv(
            hidden_states, logical_batch_size
        )
        if self.gated_attention:
            gate = self.gate_proj(hidden_states)
            gate = gate.view(
                batch_size,
                seq_len,
                self.num_heads_per_rank * self.v_head_dim,
            )

        if not self.use_custom_rotary_kernel:
            q_nope, q_rope = torch.split(
                query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

        # NOTE (Sbr): Disabled sequence parallel processing in eval model as
        # there's an error on generative eval tasks (gsm8k; hanging with fail
        # in the end of computing.
        # ---
        # TODO (Sbr): Fix SP errors on eval and remove this.
        if self.training:
            is_a2a_sp_active = self.sp_size > 1 and not self.is_ring_attn
        else:
            is_a2a_sp_active = False

        if is_a2a_sp_active:
            # Scatter by hidden, gather by sequence.
            sp_group = dist.get_sp_group()

            def gather_scatter(states, scatter_dim=-2):
                states = SeqAllToAll.apply(states, 1, scatter_dim, sp_group)
                return states

            if self.use_custom_rotary_kernel:
                query = gather_scatter(query, scatter_dim=2)
            else:
                q_nope = gather_scatter(q_nope, scatter_dim=2)
                q_rope = gather_scatter(q_rope, scatter_dim=2)
            k_nope = gather_scatter(k_nope, scatter_dim=2)
            k_rope = gather_scatter(k_rope, scatter_dim=2)
            value = gather_scatter(value, scatter_dim=2)

            seq_len = seq_len * self.sp_size

        # Get `cu_seqlens` for RoPE to determine if `sin` and `cos` should be recomputed.
        cu_seqlens_for_rope = cu_seqlens
        max_seqlen_for_rope = max_seqlen
        if isinstance(cu_seqlens_for_rope, dict):
            if not "cu_seqlens_q" in cu_seqlens_for_rope:
                err_msg = (
                    f"No `cu_seqlens_q` in provided `cu_seqlens` dict. Check the code "
                    "there must be some error regarding SP Ring implementation work. "
                    f"Present keys are {tuple(cu_seqlens_for_rope.keys())}."
                )
                raise RuntimeError(err_msg)
            cu_seqlens_for_rope = cu_seqlens_for_rope["cu_seqlens_q"]
            max_seqlen_for_rope = max_seqlen_for_rope["max_seqlen_q"]

        # NOTE(makolesov): Disable varlen on eval as `position_ids` and `cu_seqlens` are
        # not properly formed for eval datasets.
        _is_varlen_active = self.training and self.varlen_input

        # Apply rotary embeddings
        if self.config.rope_origin == "deepseek":
            assert self.num_krope_heads == 1, (
                f"DeepSeek-style RoPE works only with `num_krope_heads=1`, but got {self.num_krope_heads}"
            )
            assert not self.use_custom_rotary_kernel, (
                "unable to use 'deepseek' RoPE origin_type with self.use_custom_rotary_kernel=True"
            )

            k_rope = k_rope.transpose(1, 2)
            k_nope = k_nope.transpose(1, 2)
            q_rope = q_rope.transpose(1, 2)
            q_nope = q_nope.transpose(1, 2)

            cos, sin = self.rotary_emb(
                value.transpose(1, 2), seq_len=seq_len, cu_seqlens=cu_seqlens_for_rope
            )
            q_rope, k_rope = apply_rotary_pos_emb_deepseek(
                q_rope, k_rope, cos, sin, position_ids
            )
            if self.tp_size > 1:
                k_rope = reduce_from_tensor_model_parallel_region(k_rope, "AVG")
            query = torch.cat([q_nope, q_rope], dim=-1)
            key = torch.cat([k_nope, k_rope], dim=-1)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)

        else:
            if self.use_custom_rotary_kernel:
                cos, sin = self.rotary_emb(
                    query,
                    seq_len=seq_len,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens_for_rope,
                )
                # add interleaved mode
                cos = cos.contiguous()
                sin = sin.contiguous()
                query = apply_rotary_emb_mla(
                    query,
                    cos,
                    sin,
                    inplace=True,
                    interleaved=True,
                    head_offset=self.qk_nope_head_dim,
                )
            else:
                cos, sin = self.rotary_emb(
                    q_rope,
                    seq_len=seq_len,
                    position_ids=None if _is_varlen_active else position_ids,
                    cu_seqlens=None if _is_varlen_active else cu_seqlens_for_rope,
                )

                if max_seqlen_for_rope is not None:
                    max_seqlen_for_rope = min(max_seqlen_for_rope, cos.shape[0])

                ## custom application of transforms for interleaved RoPE (same as interleaved=True in apply_rotary_emb)
                q_rope = apply_rotary_emb(
                    q_rope.view(-1, *q_rope.shape[2:]) if _is_varlen_active else q_rope,
                    cos,
                    sin,
                    inplace=True,
                    interleaved=True,
                    cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
                    max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
                ).view(*q_rope.shape)

                query = torch.cat([q_nope, q_rope], dim=-1)

            k_rope = apply_rotary_emb(
                k_rope.view(-1, *k_rope.shape[2:]) if _is_varlen_active else k_rope,
                cos,
                sin,
                inplace=True,
                interleaved=True,
                cu_seqlens=cu_seqlens_for_rope if _is_varlen_active else None,
                max_seqlen=max_seqlen_for_rope if _is_varlen_active else None,
            ).view(*k_rope.shape)

            if self.tp_size > 1:
                k_rope = reduce_from_tensor_model_parallel_region(k_rope, "AVG")

            key = torch.cat([k_nope, k_rope], dim=-1)

        if output_attentions:
            raise ValueError(
                "Output attentions is not supported for patched `LlamaLatentAttention`"
            )

        #
        # flash-attn v2 start
        #

        value_orig_shape = value.shape[-1]
        is_padded = False
        # q/k_head_dim = 192 and v_head_dim = 128 for DeepSeek-like MLA
        if query.shape[-1] != value.shape[-1] and (query.shape[-1], value.shape[-1]) != (192, 128):
            value = F.pad(value, [0, query.shape[-1] - value.shape[-1]])
            is_padded = True

        attn_output = self._flash_attention_mla_forward(
            query, key, value, attention_mask, cu_seqlens, max_seqlen
        )
        if is_padded:
            attn_output = attn_output.split(value_orig_shape, dim=-1)[0]

        # All-to-all gather output
        if is_a2a_sp_active:
            # Scatter by sequence, gather by hidden.
            attn_output = SeqAllToAll.apply(attn_output, 2, 1, dist.get_sp_group())
            seq_len = seq_len // self.sp_size

        expected_size = (batch_size, seq_len, self.num_heads_per_rank, self.v_head_dim)
        if attn_output.size() != expected_size:
            raise ValueError(
                f"`attn_output` should be of size {expected_size}, but is {attn_output.size()}"
            )

        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        #
        # flash-attn v2 end
        #

        if self.gated_attention:
            attn_output = _apply_sigmoid_gate_fp32(attn_output, gate)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _flash_attention_mla_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ):
        is_causal = True
        bsz = query.shape[0]  # [bsz, seq_len, num_heads, head_dim]

        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            query = rearrange(query, "b s h d -> (b s) h d")
            key = rearrange(key, "b s h d -> (b s) h d")
            value = rearrange(value, "b s h d -> (b s) h d")

            assert query.shape == key.shape, (
                f"packed attention need query.shape == key.shape, but got {query.shape} != {key.shape}"
            )
            # q/k_head_dim = 192 and v_head_dim = 128 for DeepSeek-like MLA
            assert query.shape == value.shape or \
                   (query.shape[:-1] == value.shape[:-1] and (query.shape[-1], value.shape[-1]) == (192, 128)), (
                f"packed attention need query.shape == value.shape, but got {query.shape} != {value.shape}"
            )

            # Use flash_attn_varlen_func instead of the packed version
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,  # cu_seqlens_k = cu_seqlens_q in this case
                max_seqlen,
                max_seqlen,  # max_seqlen_k = max_seqlen_q in this case
                0.0,
                softmax_scale=self.softmax_scale,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = rearrange(output, "(b s) h d -> b s h d", b=bsz)

        elif attention_mask is None:
            # qkv = torch.stack([query, key, value], dim=2)
            # output = flash_attn_qkvpacked_func(
            #     qkv,
            #     causal=is_causal,
            #     deterministic=self.deterministic,
            # )
            # # Use flash_attn_func instead of the packed version
            output = flash_attn_func(
                query,
                key,
                value,
                causal=is_causal,
                deterministic=self.deterministic,
                softmax_scale=self.softmax_scale,
            )

        else:  # need to unpad
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query,
                key,
                value,
                qkvpacked=False,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale=self.softmax_scale,
                causal=is_causal,
                deterministic=self.deterministic,
            )
            output = output_pad_fn(output_unpad)

        return output  # [bsz, seq_len, num_heads, head_dim]


class LlamaLatentRingAttention(LlamaLatentAttention):
    def __init__(self, config: PretrainedConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)

        if self.config.sp_split_type == "equal":
            self.flash_attn_func = ring_flash_attn_func
            self.flash_attn_kvpacked_func = ring_flash_attn_kvpacked_func
            self.flash_attn_qkvpacked_func = ring_flash_attn_qkvpacked_func
            self.flash_attn_varlen_func = ring_flash_attn_varlen_func
            self.flash_attn_varlen_kvpacked_func = ring_flash_attn_varlen_kvpacked_func
            self.flash_attn_varlen_qkvpacked_func = (
                ring_flash_attn_varlen_qkvpacked_func
            )
        elif self.config.sp_split_type == "zigzag":
            self.flash_attn_func = zigzag_ring_flash_attn_func
            self.flash_attn_kvpacked_func = zigzag_ring_flash_attn_kvpacked_func
            self.flash_attn_qkvpacked_func = zigzag_ring_flash_attn_qkvpacked_func
            self.flash_attn_varlen_func = zigzag_ring_flash_attn_varlen_func
            self.flash_attn_varlen_kvpacked_func = (
                zigzag_ring_flash_attn_varlen_kvpacked_func
            )
            self.flash_attn_varlen_qkvpacked_func = (
                zigzag_ring_flash_attn_varlen_qkvpacked_func
            )
        elif self.config.sp_split_type == "llama3":
            self.flash_attn_varlen_func = llama3_flash_attn_varlen_func
            self.flash_attn_varlen_kvpacked_func = (
                llama3_flash_attn_varlen_kvpacked_func
            )
            self.flash_attn_varlen_qkvpacked_func = (
                llama3_flash_attn_varlen_qkvpacked_func
            )
        else:
            raise NotImplementedError(f"{self.config.sp_split_type=} is not supported!")

    def _flash_attention_mla_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ):
        sp_size = dist.get_sp_group_size() or 1
        sp_group = dist.get_sp_group()
        if not self.training or sp_size == 1:  # disable ring with eval mode or sp=1
            return super()._flash_attention_mla_forward(
                query, key, value, attention_mask, cu_seqlens, max_seqlen
            )

        is_causal = True
        bsz = key.shape[0]
        if cu_seqlens is not None and max_seqlen is not None and attention_mask is None:
            query = rearrange(query, "b s h d -> (b s) h d")
            key = rearrange(key, "b s h d -> (b s) h d")
            value = rearrange(value, "b s h d -> (b s) h d")

            assert query.shape == key.shape, (
                f"Queries and keys are expected to be of the same shape, but got {query.shape} != {key.shape}"
            )
            # q/k_head_dim = 192 and v_head_dim = 128 for DeepSeek-like MLA
            assert query.shape == value.shape or \
                   (query.shape[:-1] == value.shape[:-1] and (query.shape[-1], value.shape[-1]) == (192, 128)), (
                f"Queries and values are expected to be of the same shape, but got {query.shape} != {value.shape}"
            )

            if self.config.sp_split_type == "llama3":
                output = self.flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens["cu_seqlens_q"],
                    cu_seqlens[
                        "cu_seqlens_k"
                    ],  # cu_seqlens_k = cu_seqlens_q in this case
                    max_seqlen["max_seqlen_q"],
                    max_seqlen["max_seqlen_k"],
                    heads_k_stride=self.config.llama3_ring_heads_k_stride,
                    local_k_slice=cu_seqlens["local_k_slice"],
                    dropout_p=0.0,
                    softmax_scale=self.softmax_scale,
                    causal=is_causal,
                    deterministic=self.deterministic,
                    group=sp_group,
                )
            else:
                output = self.flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens,
                    max_seqlen,  # max_seqlen_k = max_seqlen_q in this case
                    dropout_p=0.0,
                    softmax_scale=self.softmax_scale,
                    causal=is_causal,
                    deterministic=self.deterministic,
                    group=sp_group,
                )

            output = rearrange(output, "(b s) h d -> b s h d", b=bsz)

        elif attention_mask is None:
            output = self.flash_attn_func(
                query,
                key,
                value,
                dropout_p=0.0,
                causal=is_causal,
                deterministic=self.deterministic,
                softmax_scale=self.softmax_scale,
                group=sp_group,
            )

        else:  # need to unpad
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query,
                key,
                value,
                qkvpacked=False,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query.size(1) :]
                if attention_mask is not None
                else None,
            )
            output_unpad = self.flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=is_causal,
                deterministic=self.deterministic,
                group=sp_group,
            )
            output = output_pad_fn(output_unpad)

        return output # [bsz, seq_len, num_heads, head_dim]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32},  num_warps=1, num_stages=1),
        triton.Config({'BLOCK_D': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8, num_stages=4),
    ],
    key=['B', 'S', 'D'],
)
@triton.jit
def fused_g_forward_kernel(
    a_ptr, A_log_ptr, dt_ptr, out_ptr,
    stride_ab, stride_as, stride_ad,
    B, S, D,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_blk = tl.program_id(1)
    b = row_idx // S
    s = row_idx  % S

    d_off = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    a_off = b * stride_ab + s * stride_as + d_off * stride_ad
    mask  = d_off < D

    A_log = tl.load(A_log_ptr + d_off, mask=mask, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + d_off, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + a_off, mask=mask, other=0.0).to(tl.float32)

    neg_exp_A = -tl.exp(A_log)
    x = a + dt

    exp_neg_x = tl.exp(-tl.abs(x))
    softplus_x = tl.where(
        x >= 0,
        x + tl.log(1.0 + exp_neg_x),
        tl.log(1.0 + tl.exp(x))
    )

    tl.store(out_ptr + a_off, neg_exp_A * softplus_x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32},  num_warps=1, num_stages=1),
        triton.Config({'BLOCK_D': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8, num_stages=4),
    ],
    key=['B', 'S', 'D'],
)
@triton.jit
def fused_g_backward_kernel(
    a_ptr, A_log_ptr, dt_ptr, grad_out_ptr,
    grad_a_ptr, grad_A_log_ptr,
    stride_b, stride_s, stride_d,
    B, S, D,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_blk = tl.program_id(1)
    b = row_idx // S
    s = row_idx  % S

    d_off = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    off = b * stride_b + s * stride_s + d_off * stride_d

    a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + d_off, mask=mask, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + d_off, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + off, mask=mask, other=0.0).to(tl.float32)

    neg_exp_A = -tl.exp(A_log)
    x = a + dt
    pos = x >= 0
    exp_neg_x = tl.exp(-tl.abs(x))
    softplus_x = tl.where(pos, x + tl.log(1.0 + exp_neg_x), tl.log(1.0 + tl.exp(x)))
    sigmoid_x = tl.where(pos, 1.0 / (1.0 + exp_neg_x), tl.exp(x) / (1.0 + tl.exp(x)))

    g = neg_exp_A * softplus_x
    grad_a = grad_out * neg_exp_A * sigmoid_x

    tl.store(grad_a_ptr + off, grad_a, mask=mask)
    tl.store(grad_A_log_ptr + off, grad_out * g, mask=mask)


class FusedG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, A_log, dt) -> torch.Tensor:
        B, S, D = a.shape
        out  = torch.empty(B, S, D, device=a.device, dtype=a.dtype)
        def grid(meta: dict): return (B * S, triton.cdiv(D, meta["BLOCK_D"]))
        fused_g_forward_kernel[grid](
            a, A_log, dt, out,
            a.stride(0), a.stride(1), a.stride(2),
            B, S, D,
        )
        ctx.save_for_backward(a, A_log, dt)
        ctx.shape = (B, S, D)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, A_log, dt = ctx.saved_tensors
        B, S, D = ctx.shape

        grad_out = grad_out.contiguous()
        grad_a = torch.empty(B, S, D, device=a.device, dtype=a.dtype)
        grad_A_log_inter = torch.empty(B, S, D, device=a.device, dtype=A_log.dtype)

        def grid(meta: dict): return (B * S, triton.cdiv(D, meta["BLOCK_D"]))
        fused_g_backward_kernel[grid](
            a, A_log, dt, grad_out,
            grad_a, grad_A_log_inter,
            a.stride(0), a.stride(1), a.stride(2),
            B, S, D,
        )

        grad_dt = grad_a.sum((0, 1))

        return grad_a, grad_A_log_inter.sum((0, 1)), grad_dt

class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        tp_size = dist.get_tp_group_size() or 1
        sp_size = dist.get_sp_group_size() or 1

        if tp_size > 1:
            raise ValueError(
                "Qwen3NextGatedDeltaNet does not support tensor parallel yet! :("
            )
        self.tp_size = tp_size
        self.sp_size = sp_size

        self.head_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads

        self.conv_size = config.linear_conv_kernel_dim
        self.norm_eps = config.rms_norm_eps
        self.o_norm_eps = getattr(config, "linear_attn_o_norm_eps", None)
        if self.o_norm_eps is None:
            self.o_norm_eps = self.norm_eps

        self.linear_gating_type = getattr(config, "linear_gating_type", "gated_rmsnorm")
        self.linear_sigmoid_gate_scale = config.linear_sigmoid_gate_scale
        self.conv_init_std = config.linear_conv_init_std

        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        self.use_legacy_qkvz_layout = getattr(config, "linear_use_legacy_qkvz_layout", False)

        projection_size_ba = 2 * self.num_v_heads
        self.ba_proj = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        if self.use_legacy_qkvz_layout:
            projection_size_qkvz = 2 * self.key_dim + 2 * self.value_dim
            self.qkvz_proj = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        else:
            projection_size_qkv = 2 * self.key_dim + self.value_dim
            projection_size_z = self.value_dim
            self.qkv_proj = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
            self.z_proj = nn.Linear(self.hidden_size, projection_size_z, bias=False)


        # see reset_parameters(), init from flash-linear-attention
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))

        if self.use_legacy_qkvz_layout:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                bias=False,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                bias=False,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_size,
                bias=False,
                activation="silu",
            )
        else:
            self.qkv_conv1d = ShortConvolution(
                hidden_size=projection_size_qkv,
                kernel_size=self.conv_size,
                bias=False,
                activation="silu"
            )

        if self.linear_gating_type == "gated_rmsnorm":
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.o_norm_eps,
                activation="silu",
            )
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid":
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.o_norm_eps,
                activation="sigmoid",
            )
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid_zero_centered":
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.o_norm_eps,
                activation="sigmoid",
                zero_centered=True,
            )
        elif self.linear_gating_type == "sigmoid_gate":
            pass
        else:
            raise ValueError((
                f"Unknown linear_gating_type: {self.linear_gating_type}. " \
                f"Supported: 'gated_rmsnorm', 'gated_rmsnorm_sigmoid', "
                f"'gated_rmsnorm_sigmoid_zero_centered', 'sigmoid_gate'."
            ))
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.reset_parameters()

    @staticmethod
    def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None):
        # NOTE: attention mask is a 2D boolean tensor
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states.masked_fill_(~attention_mask[:, :, None], 0.)

        return hidden_states

    def _build_cp_context(
        self,
        batch_size: int,
        local_q_len: int,
        cu_seqlens: torch.Tensor | None,
        sp_group,
        device: torch.device,
    ):
        if cu_seqlens is None:
            global_seq_len = local_q_len * self.sp_size
            cu_seqlens = torch.arange(
                batch_size + 1,
                device=device,
                dtype=torch.long,
            ) * global_seq_len
        else:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)

        return build_cp_context(
            cu_seqlens=cu_seqlens,
            group=sp_group,
            conv1d_kernel_size=self.conv_size,
        )


    def fix_query_key_value_ordering(self, mixed_qkv, z, mixed_ba):
        new_tensor_shape_qkv = mixed_qkv.size()[:-1] + (
            self.num_heads,
            2 * self.head_dim + self.head_v_dim * self.num_v_heads // self.num_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_heads, 2 * self.num_v_heads // self.num_heads)

        mixed_qkv = mixed_qkv.view(*new_tensor_shape_qkv)
        z = z.view(*z.size()[:-1], self.num_v_heads, self.head_v_dim)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkv = [
            self.head_dim,
            self.head_dim,
            (self.num_v_heads // self.num_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_heads, self.num_v_heads // self.num_heads]
        query, key, value = torch.split(mixed_qkv, split_arg_list_qkv, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z.contiguous(), b.contiguous(), a.contiguous()

    def fix_query_key_value_ordering_legacy(self, mixed_qkvz, mixed_ba):
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_heads,
            2 * self.head_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_heads, 2 * self.num_v_heads // self.num_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_dim,
            self.head_dim,
            (self.num_v_heads // self.num_heads * self.head_v_dim),
            (self.num_v_heads // self.num_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_heads, self.num_v_heads // self.num_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z.contiguous(), b.contiguous(), a.contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ):

        assert not use_cache and past_key_value is None, "Qwen3NextGatedDeltaNet attention does not support cache and past_key_value"
        batch_size, q_len, _ = hidden_states.shape

        mode = "fused_recurrent" if q_len <= 64 and self.sp_size == 1 else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)

        sp_group = dist.get_sp_group()

        is_sp_active = self.sp_size > 1 and self.training

        projected_states_ba = self.ba_proj(hidden_states)

        if self.use_legacy_qkvz_layout:
            projected_states_qkvz = self.qkvz_proj(hidden_states)
        else:
            projected_states_qkv = self.qkv_proj(hidden_states)
            projected_states_z = self.z_proj(hidden_states)

        if is_sp_active:
            cp_context = self._build_cp_context(
                batch_size=batch_size,
                local_q_len=q_len,
                cu_seqlens=cu_seqlens,
                sp_group=sp_group,
                device=hidden_states.device,
            )
        else:
            cp_context = None

        if self.use_legacy_qkvz_layout:
            # query, key, value, z: [bsize, seq_len // sp_size, num_heads, key_dim or value_dim]
            # b, a:                 [bsize, seq_len // sp_size, num_v_heads]
            query, key, value, z, b, a = self.fix_query_key_value_ordering_legacy(
                projected_states_qkvz,
                projected_states_ba,
            )
            del projected_states_qkvz, projected_states_ba

            query = rearrange(query, "... h d -> ... (h d)")
            key = rearrange(key, "... h d -> ... (h d)")
            value = rearrange(value, "... h d -> ... (h d)")

            q, _ = self.q_conv1d(query, cp_context=cp_context, cu_seqlens=cu_seqlens)
            k, _ = self.k_conv1d(key, cp_context=cp_context, cu_seqlens=cu_seqlens)
            v, _ = self.v_conv1d(value, cp_context=cp_context, cu_seqlens=cu_seqlens)

            q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k))
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        else:
            qkv, _ = self.qkv_conv1d(
                projected_states_qkv,
                cp_context=cp_context,
                cu_seqlens=cu_seqlens,
            )
            del projected_states_qkv

            # query, key, value, z: [bsize, seq_len // sp_size, num_heads, key_dim or value_dim]
            # b, a:                 [bsize, seq_len // sp_size, num_v_heads]
            q, k, v, z, b, a = self.fix_query_key_value_ordering(
                qkv,
                projected_states_z,
                projected_states_ba,
            )
            del qkv, projected_states_z, projected_states_ba

        b.sigmoid_()
        g = FusedG.apply(a, self.A_log, self.dt_bias)

        if mode == "chunk":
            if self.num_v_heads // self.num_heads > 1:
                q = q.repeat_interleave(self.num_v_heads // self.num_heads, dim=2)
                k = k.repeat_interleave(self.num_v_heads // self.num_heads, dim=2)
            o, _ = chunk_gated_delta_rule(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g,
                beta=b,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
        elif mode == "fused_recurrent":
            # fused_recurrent supports gva as is
            o, _ = fused_recurrent_gated_delta_rule(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g,
                beta=b,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if is_sp_active:
            expected_size = (batch_size, q_len, self.num_v_heads, self.head_v_dim)
            if o.size() != expected_size:
                raise ValueError(
                    f"`attn_output` should be of size {expected_size}, but is {o.size()}"
                )

        # gating
        if self.linear_gating_type == "gated_rmsnorm":
            o = self.o_norm(o, z)
            o = rearrange(o, "b t h d -> b t (h d)")
        elif self.linear_gating_type in ("gated_rmsnorm_sigmoid", "gated_rmsnorm_sigmoid_zero_centered"):
            o = self.o_norm(o, z)
            if self.linear_sigmoid_gate_scale != 1.0:
                o = o * self.linear_sigmoid_gate_scale
            o = rearrange(o, "b t h d -> b t (h d)")
        elif self.linear_gating_type == "sigmoid_gate":
            o = rearrange(o, "b t h d -> b t (h d)").contiguous()
            z = rearrange(z, "b t h d -> b t (h d)").contiguous()
            o = _apply_sigmoid_gate_fp32(o, z * self.linear_sigmoid_gate_scale)
        o = self.o_proj(o)
        if attention_mask is not None:
            o = self.apply_mask_to_padding_states(o, attention_mask)
        return o, None, past_key_value

    def reset_parameters(self):
        def log_uniform_(tensor, low, high):
            with torch.no_grad():
                log_low = torch.tensor(low, dtype=tensor.dtype).log()
                log_high = torch.tensor(high, dtype=tensor.dtype).log()
                tensor.uniform_(log_low, log_high).exp_()
            return tensor

        A = log_uniform_(torch.empty(self.num_v_heads, dtype=torch.float32), 1, 16)
        self.A_log.data.copy_(torch.log(A))

        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = log_uniform_(torch.empty(self.num_v_heads), dt_min, dt_max)
        dt = torch.clamp(dt, min=dt_init_floor)

        if hasattr(self, "o_norm"):
            self.o_norm.reset_parameters()
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

        if self.use_legacy_qkvz_layout:
            torch.nn.init.normal_(self.q_conv1d.weight, mean=0.0, std=self.conv_init_std)
            torch.nn.init.normal_(self.k_conv1d.weight, mean=0.0, std=self.conv_init_std)
            torch.nn.init.normal_(self.v_conv1d.weight, mean=0.0, std=self.conv_init_std)
        else:
            torch.nn.init.normal_(self.qkv_conv1d.weight, mean=0.0, std=self.conv_init_std)


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config

        tp_size = dist.get_tp_group_size() or 1
        sp_size = dist.get_sp_group_size() or 1

        if tp_size > 1:
            raise ValueError(
                "KimiDeltaAttention does not support tensor parallel yet! :("
            )
        self.tp_size = tp_size
        self.sp_size = sp_size

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_conv_kernel_dim
        self.head_dim = config.linear_key_head_dim
        self.num_heads = config.linear_num_key_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.head_v_dim = config.linear_value_head_dim
        self.num_v_heads = config.linear_num_value_heads

        assert (self.num_v_heads == self.num_heads) and (self.head_v_dim == self.head_k_dim), "KimiDeltaAttention code suppports equal head_k_dim and head_v_dim / num_k_heads and num_v_heads"

        if self.sp_size > 1:
            assert self.num_heads % self.sp_size == 0, "num_heads must be divisible by sp_size"

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        self.A_log = torch.nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, -1, 1)
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    @staticmethod
    def index_first_axis(x, indices):
        other_shape = x.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def pad_input(
        hidden_states: torch.Tensor,
        indices: torch.LongTensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        output = KimiDeltaAttention.index_put_first_axis(hidden_states, indices, batch_size * seq_len)
        return rearrange(output, "(b s) ... -> b s ...", b=batch_size)

    @staticmethod
    @tensor_cache
    def get_unpad_data(
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        lens = prepare_lens_from_mask(attention_mask)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = lens.max().item()
        cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
        return indices, cu_seqlens, max_seqlen_in_batch


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
    ):
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] " + \
                "(0 = padding). 3D masks are not supported here.",
            )

        sp_group = dist.get_sp_group()
        sp_rank = dist.get_sp_group_rank()

        is_sp_active = self.sp_size > 1 and self.training

        assert not use_cache, "KimiAttention is not tested with use_cache=True"
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = KimiDeltaAttention.get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = KimiDeltaAttention.index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        def gather_scatter(states, scatter_dim=2):
            states = SeqAllToAll.apply(states, 1, scatter_dim, sp_group)
            return states

        def _slice_conv1d_weights_for_sp_rank():
            q_weight = self.q_conv1d.weight
            k_weight = self.k_conv1d.weight
            v_weight = self.v_conv1d.weight

            nheads_key_local_sp = self.num_k_heads // self.sp_size
            qkv_size = nheads_key_local_sp * self.head_dim
            qkv_start = sp_rank * qkv_size

            q_weight_sliced = q_weight.narrow(0, qkv_start, qkv_size)
            k_weight_sliced = k_weight.narrow(0, qkv_start, qkv_size)
            v_weight_sliced = v_weight.narrow(0, qkv_start, qkv_size)
            return q_weight_sliced, k_weight_sliced, v_weight_sliced

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if is_sp_active:
            q, k, v = (gather_scatter(x) for x in (q, k, v))

            conv_q_weight, conv_k_weight, conv_v_weight = (
                _slice_conv1d_weights_for_sp_rank()
            )

            # causal_conv1d_fn expects:
            # x: (batch, dim, seqlen)
            # w: (dim, width)
            def _apply_causal_conv1d_fn(x, weight, bias):
                x = rearrange(x, "b s d -> b d s")
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(weight, "d 1 w -> d w"),
                    bias=bias,  # TODO: support slice bias
                    activation="silu",
                    seq_idx=None,
                )
                x = rearrange(x, "b d s -> b s d")
                return x

            q = _apply_causal_conv1d_fn(q, conv_q_weight, self.q_conv1d.bias)
            k = _apply_causal_conv1d_fn(k, conv_k_weight, self.k_conv1d.bias)
            v = _apply_causal_conv1d_fn(v, conv_v_weight, self.v_conv1d.bias)
        else:
            q, _ = self.q_conv1d(
                x=q,
                cu_seqlens=cu_seqlens,
            )
            k, _ = self.k_conv1d(
                x=k,
                cu_seqlens=cu_seqlens,
            )
            v, _ = self.v_conv1d(
                x=v,
                cu_seqlens=cu_seqlens,
            )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        # g = fused_kda_gate(g, self.A_log, dt_bias=self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k, g = map(lambda x: rearrange(
            x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)


        if is_sp_active:
            g = gather_scatter(g)
            beta = gather_scatter(beta)

            # A_log and dt_bias have shapes without seq_len
            # thus requiring special logic here
            A_log_len = self.A_log.shape[2]
            A_log_len_per_rank = A_log_len // self.sp_size
            start = A_log_len_per_rank * sp_rank
            A_log = self.A_log.narrow(2, start, A_log_len_per_rank)

            dt_bias_len = len(self.dt_bias)
            dt_bias_len_per_rank = dt_bias_len // self.sp_size
            start = dt_bias_len_per_rank * sp_rank
            dt_bias = self.dt_bias.narrow(0, start, dt_bias_len_per_rank)
        else:
            A_log = self.A_log
            dt_bias = self.dt_bias

        if mode == "chunk":
            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            g = fused_kda_gate(g=g, A_log=A_log, dt_bias=dt_bias)
            o, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        if is_sp_active:
            o = SeqAllToAll.apply(
                o,
                2,
                1,
                dist.get_sp_group()
            )

            expected_size = (batch_size, q_len, self.num_v_heads, self.head_v_dim)
            if o.size() != expected_size:
                raise ValueError(
                    f"`attn_output` should be of size {expected_size}, but is {o.size()}"
                )

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_value

    def reset_parameters(self):
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16).view(1, 1, -1, 1)
        self.A_log.data.copy_(torch.log(A))
        self.A_log._no_weight_decay = True # not sure here
        dt = torch.zeros(self.head_dim * self.num_heads, dtype=torch.float32)
        self.dt_bias.data.copy_(dt)
        self.dt_bias._no_weight_decay = True # not sure here


RING_ATTN_CLASSES = {
    "LlamaPackedRingAttention",
    "LlamaLatentRingAttention",
}

ATTN_CLASS_REGISTRY = {
    "multihead_attention": MultiheadAttention,
    "multiquery_attention": MultiQueryAttention,
    "LlamaAttention": LlamaAttention,
    "LlamaPackedAttention": LlamaPackedAttention,
    "LlamaPackedRingAttention": LlamaPackedRingAttention,
    "LlamaLatentAttention": LlamaLatentAttention,
    "LlamaLatentRingAttention": LlamaLatentRingAttention,
    "Qwen3NextGatedDeltaNet": Qwen3NextGatedDeltaNet,
    "KimiDeltaAttention": KimiDeltaAttention
}
