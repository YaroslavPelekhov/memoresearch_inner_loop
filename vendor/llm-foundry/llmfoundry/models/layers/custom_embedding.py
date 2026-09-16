# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import math
import typing as tp
from functools import partial
from typing import Any, Callable, Literal, Optional, Tuple

import composer.utils.dist as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from composer.utils.dist import get_tp_group, get_tp_group_rank, get_tp_group_size
from einops import rearrange
from flash_attn.layers.rotary import (
    apply_rotary_emb,
)  # apply_rotary_emb_qkv_,; apply_rotary_emb_kv_,
from flash_attn.ops.triton.rotary import apply_rotary
from torch import Tensor
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from llmfoundry.models.parallel.tensor import (
    VocabUtility,
    _initialize_tp_weight,
    distribute_async_tp_embeddings,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from llmfoundry.models.utils.flash_seqlens import get_cu_seqlens_from_pos_ids

apply_rotary_emb = torch.compiler.disable(apply_rotary_emb)
# apply_rotary_emb_kv_ = torch.compiler.disable(apply_rotary_emb_kv_)
# apply_rotary_emb_qkv_ = torch.compiler.disable(apply_rotary_emb_qkv_)


def _apply_rotary_emb_qkv(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
    seqlen_offsets: tp.Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    apply_rotary_fn = partial(
        apply_rotary,
        interleaved=interleaved,
        inplace=inplace,
        conjugate=conjugate,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert max_seqlen is not None

    if cos_k is None and sin_k is None and qkv.is_contiguous():
        # Call 1 kernel instead of 2 kernels
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
        # dimensions, we get the same tensor
        if is_varlen:
            total_seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            qk = qkv[:, :2].reshape(total_seqlen, -1, headdim)
            qk = apply_rotary_fn(qk, cos, sin)
        else:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            qk = apply_rotary_fn(qk, cos, sin)
        if not inplace:
            if is_varlen:
                qkv = torch.cat(
                    [rearrange(qk, "ts (t h) d -> ts t h d", t=2), qkv[:, 2:]],
                    dim=1,
                )
            else:
                qkv = torch.cat(
                    [
                        rearrange(qk, "b s (t h) d -> b s t h d", t=2),
                        qkv[:, :, 2:],
                    ],
                    dim=2,
                )

    else:
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        if is_varlen:
            total_seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            q, k = qkv[:, 0], qkv[:, 1]
        else:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            q, k = qkv[:, :, 0], qkv[:, :, 1]
        q = apply_rotary_fn(q, cos, sin)
        k = apply_rotary_fn(k, cos_k, sin_k)
        if not inplace:
            if is_varlen:
                qkv = torch.stack([q, k, qkv[:, 2]], dim=1)
            else:
                qkv = torch.stack([q, k, qkv[:, :, 2]], dim=2)
    return qkv


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    This class is required for correct gradient calculation under performance constraints in varlen regime.
    This is a copy of the corresponding class from Flash Attention repository:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L267,
    which allows for additional cu_seqlens and max_seqlen params required for varlen.
    """

    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: tp.Union[int, torch.Tensor] = 0,
        num_heads_q: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # apply_rotary_emb_qkv_inplace(
        qkv = _apply_rotary_emb_qkv(
            qkv,
            cos,
            sin,
            cos_k,
            sin_k,
            interleaved=interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offsets,
            num_heads_q=num_heads_q,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        max_seqlen = ctx.max_seqlen
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k, cu_seqlens = ctx.saved_tensors
        dqkv = _apply_rotary_emb_qkv(
            dqkv,
            cos,
            sin,
            cos_k,
            sin_k,
            interleaved=ctx.interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offsets,
            num_heads_q=ctx.num_heads_q,
            conjugate=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return dqkv, None, None, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: tp.Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
    cu_seqlens=None,
    max_seqlen=None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """

    bs, seq_len, three, nhqkv, hd = qkv.size()
    assert three == 3
    if cu_seqlens is not None:
        qkv = qkv.view(-1, three, nhqkv, hd)
    qkv = ApplyRotaryEmbQKV_.apply(
        qkv,
        cos,
        sin,
        cos_k,
        sin_k,
        interleaved,
        seqlen_offsets,
        num_heads_q,
        cu_seqlens,
        max_seqlen,
    )
    if cu_seqlens is not None:
        qkv = qkv.view(bs, seq_len, three, nhqkv, hd)
    return qkv


class ApplyRotaryEmbKV_(torch.autograd.Function):
    """
    This class is required for correct gradient calculation under performance constraints in varlen regime.
    This is a copy of the corresponding class from Flash Attention repository:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L267,
    which allows for additional cu_seqlens and max_seqlen params required for varlen.
    """

    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets=0,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        if cu_seqlens is not None:
            total_seqlen, two, nheads, headdim = kv.shape
            assert two == 2
            k = kv[:, 0]
        else:
            batch, seqlen, two, nheads, headdim = kv.shape
            assert two == 2
            k = kv[:, :, 0]
        apply_rotary(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets, cu_seqlens)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.max_seqlen = max_seqlen
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        max_seqlen = ctx.max_seqlen
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets, cu_seqlens = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors

        if cu_seqlens is None:
            dk = dkv[:, :, 0]
        else:
            dk = dkv[:, 0]

        apply_rotary(
            dk,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return dkv, None, None, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(
        kv, cos, sin, interleaved, seqlen_offsets, cu_seqlens, max_seqlen
    )


def check_tensor_equality(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    """Check if provided tensors are equal handling None case.

    This function checks tensor equality taking into account that provided
    arguments can be `None`. If some of them is `None`, direct equal operand
    is used. If both are `torch.Tensor`, then `torch.equal()` function is called.
    """
    res = False
    if a is None or b is None:
        res = a == b
    else:
        res = torch.equal(a, b)

    return res


class SharedEmbedding(nn.Embedding):
    def forward(self, input: Tensor, unembed: bool = False) -> Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)

class ScaledEmbedding(nn.Embedding):
    """nn.Embedding with additional output scaling in forward."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.scalar_embed_scale = embed_scale
        self.register_buffer(
            "embed_scale", torch.tensor(embed_scale), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.scalar_embed_scale


ParallelOutputStyleT = Literal["replicate", "shard_sequence", "shard_hidden"]


class EmbeddingParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        config: Any,
        parallel_output_style: ParallelOutputStyleT,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.parallel_output_style = parallel_output_style
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        # Keep the input dimensions.
        self.tensor_model_parallel_size = get_tp_group_size()
        self.global_vocab_size = num_embeddings

        assert embedding_dim % self.tensor_model_parallel_size == 0
        self.embedding_dim_per_partition = (
            embedding_dim // self.tensor_model_parallel_size
        )

        self._init_params = {
            "input_size": self.embedding_dim,
            "output_size": self.global_vocab_size,
            "partition_dim": 1,
            "per_partition_size": self.embedding_dim_per_partition,
            "init_method": partial(self._init_embeddings, padding_idx=self.padding_idx),
            "device": config.init_device,
            "use_master_weight": config.use_master_weight,
            "init_type": config.init_type,
        }

        self.weight = nn.Parameter(
            torch.empty(
                self.global_vocab_size,
                self.embedding_dim_per_partition,
                device=config.init_device,
            )
        )

        if config.init_device != "meta":
            self.reset_parameters()

    @staticmethod
    def _init_embeddings(
        weight: torch.Tensor,
        padding_idx: Optional[int] = None,
        init_method: Optional[Callable] = None,
    ) -> None:
        if init_method:
            init_method(weight)
        else:
            nn.init.xavier_normal_(weight)

        if padding_idx is not None:
            with torch.no_grad():
                weight[padding_idx].fill_(0)

    def reset_parameters(self) -> None:
        # Update device before init to account for tensor being moved
        self._init_params["device"] = self.weight.device
        _initialize_tp_weight(self.weight, **self._init_params)

    def forward(
        self,
        input_ids: torch.Tensor,
        parallel_output_style: ParallelOutputStyleT | None = None,
    ) -> torch.Tensor:
        assert torch.all(input_ids >= 0), "input ids must be positive"

        if parallel_output_style is None:
            parallel_output_style = self.parallel_output_style

        try:
            output_parallel = self.weight[input_ids]
        except IndexError:
            raise IndexError("An input token id greater than vocab size")

        if parallel_output_style == "replicate":
            return gather_from_tensor_model_parallel_region(output_parallel)
        elif parallel_output_style == "shard_sequence":
            return self._all_to_all_output(output_parallel)
        elif parallel_output_style == "shard_hidden":
            return output_parallel
        else:
            raise ValueError(f"Unknown parallel output style: {parallel_output_style}")

    def _all_to_all_output(self, input_: torch.Tensor) -> torch.Tensor:
        return distribute_async_tp_embeddings(input_)


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        config,
        padding_idx: int = 0,
    ):
        super(VocabParallelEmbedding, self).__init__()

        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        # Keep the input dimensions.
        self.tensor_model_parallel_size = get_tp_group_size()
        self.global_vocab_size = num_embeddings
        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.global_vocab_size,
            get_tp_group_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        self._init_params = {
            "input_size": self.embedding_dim,
            "output_size": self.global_vocab_size,
            "partition_dim": 0,
            "per_partition_size": self.num_embeddings_per_partition,
            "init_method": partial(self._init_embeddings, padding_idx=self.padding_idx),
            "device": config.init_device,
            "use_master_weight": config.use_master_weight,
            "init_type": config.init_type,
        }

        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=config.init_device,
            )
        )

        # Initialize weights if on cpu device and skip for meta device.
        # For meta device params are initialized on FSDP model wrapping in Composer
        # using `param_init_fn`.
        if config.init_device != "meta":
            self.reset_parameters()

    @staticmethod
    def _init_embeddings(
        weight: torch.Tensor,
        padding_idx: Optional[int] = None,
        init_method: Optional[Callable] = None,
    ):
        if init_method:
            init_method(weight)
        else:
            nn.init.xavier_normal_(weight)

        if padding_idx is not None:
            with torch.no_grad():
                weight[padding_idx].fill_(0)

    def reset_parameters(self):
        # Update device before init to account for tensor being moved
        self._init_params["device"] = self.weight.device
        _initialize_tp_weight(self.weight, **self._init_params)

    def forward(self, input_):
        assert not torch.any((input_ < 0) | (input_ >= self.global_vocab_size)), (
            "An input token is out of bounds of the embedding table"
        )
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


def apply_rotary_pos_emb(
    q: tp.Union[torch.FloatTensor, torch.Tensor],
    k: tp.Union[torch.FloatTensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: tp.Optional[torch.LongTensor] = None,
    cu_seqlens: tp.Optional[torch.Tensor] = None,
    max_seqlen: tp.Optional[int] = None,
    seqlen_offsets: tp.Union[int, torch.Tensor] = 0,
    is_flash: bool = False,
    inplace: bool = True,
) -> tp.Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Args:
    q (torch.FloatTensor) : bs x seq_len x nh x hd
    k (torch.FloatTensor): bs x seq_len x nh x hd
    position_ids (torch.LongTensor): bs x seq_len
    """
    if not is_flash:
        q = q.transpose(1, 2)  # make it [batch_size, seqlen, nheads, headdim]
        k = k.transpose(1, 2)  # make it [batch_size, seqlen, nheads, headdim]

    q_rot = apply_rotary_emb(
        q,
        cos,
        sin,
        inplace=inplace,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    k_rot = apply_rotary_emb(
        k,
        cos,
        sin,
        inplace=inplace,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )

    if not is_flash:
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)
    return q_rot, k_rot


def apply_rotary_emb_kv_packed_(
    q: tp.Union[torch.FloatTensor, torch.Tensor],
    kv: tp.Union[torch.FloatTensor, torch.Tensor],
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    position_ids: torch.LongTensor,
    is_left_padded_eval: bool = False,
    use_cache: bool = False,
    is_training: bool = False,
    kv_cache_enabled: bool = False,
    cu_seqlens=None,
    max_seqlen=None,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Apply rope for query states and packed kv_states.
    Args:
        q (torch.Tensor) : bs x seq_len x nh x hd
        kv (torch.Tensor): bs x seq_len x 2 x nh x hd
        position_ids (torch.LongTensor): bs x seq_len
        is_left_padded_eval (bool): if false, indicates that evaluation inputs are right-padded.
            this enables the use of `apply_rotary_emb` functions without requiring additional
            padding-related parameters during evaluation.
    """
    # avoid any unnecessary checks during training steps
    if is_training or not is_left_padded_eval:
        q_shape = q.shape
        kv_shape = kv.shape
        if cu_seqlens is not None:
            q = q.view(-1, *q_shape[2:])
            kv = kv.view(-1, *kv_shape[2:])
        q = apply_rotary_emb(q, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        kv = apply_rotary_emb_kv_(
            kv, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

        if cu_seqlens is not None:
            q = q.view(*q_shape)
            kv = kv.view(*kv_shape)

    else:
        # cos/sin for position_ids = [0, 1, ..., seq_len-1]
        # control rope via flash_attn.layers.rotary.apply_rotary_emb parameters.

        if kv_cache_enabled:
            # (*) Second step generation with kv-cache.
            # Uses cached key-value states from previous generation steps, rotating the
            # positional embeddings by seqlen_offset (current generated sequence length)
            # to maintain correct positional information
            seqlen_offsets = position_ids[:, -1]
            cu_seqlens, max_seqlen = None, None
            k = kv[:, :, 0, :, :]  # bs x seq_len x nh x hd

        else:
            # (*) First-step generation (with cache) or (**) arbitrary-step generation
            # (without cache). Assumes left-padded sequences. We avoid rotating padding
            # positions by using cu_seqlens.
            seqlen_offsets = 0
            cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
            cu_seqlens = cu_seqlens.squeeze(0)
            max_seqlen = int(max_seqlen.detach().cpu().max().item())
            bs, seq_len, _, nhkv, hd = kv.size()
            k = kv[:, :, 0, :, :].reshape(-1, nhkv, hd)  # total_seq_len x nh x hd
            bs, seq_len, nhq, hd = q.size()
            q = q.reshape(-1, nhq, hd)  # total_seq_len x nh x hd

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            is_flash=True,
        )

        if not kv_cache_enabled:
            k = k.reshape(bs, seq_len, nhkv, hd)
            q = q.reshape(bs, seq_len, nhq, hd)

        kv[:, :, 0, :, :] = k

    return q, kv


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None
        self._cu_seqlen_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)  # FIX
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x,
        seq_len: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if (
            self.max_seq_len_cached is None
            or seq_len > self.max_seq_len_cached
            or not check_tensor_equality(self._cu_seqlen_cached, cu_seqlens)
        ):
            self._cu_seqlen_cached = cu_seqlens
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


def apply_mscale_correction(attention_factor, mscale):
    attention_factor = (attention_factor - 1) * mscale + 1
    return attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # No need to keep BC with yarn, unreleased when this new pattern was created.
    if len(rope_kwargs) > 0:
        raise ValueError(
            f"Unexpected arguments: `**rope_kwargs` should be unset in `_compute_yarn_parameters`, got {rope_kwargs}"
        )

    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor
        if hasattr(config, "partial_rotary_factor")
        else 1.0
    )
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    ) or (config.hidden_size // config.num_attention_heads)

    dim = int(head_dim * partial_rotary_factor)

    ## change to original_max_position_embeddings same as deepseek
    # max_position_embeddings = config.max_position_embeddings

    original_max_position_embeddings = (
        config.rope_scaling.get("original_max_position_embeddings")
        or config.max_position_embeddings
    )
    factor = config.rope_scaling["factor"]

    # Sets the attention factor as suggested in the paper
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        # mscale = config.rope_scaling.get("mscale_all_dim") or 1
        attention_factor = 0.1 * math.log(factor) + 1.0

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, original_max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (
            dim
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    def find_correction_range(
        low_rot, high_rot, dim, base, original_max_position_embeddings
    ):
        """Find dimension range bounds based on rotations"""
        low = math.floor(
            find_correction_dim(low_rot, dim, base, original_max_position_embeddings)
        )
        high = math.ceil(
            find_correction_dim(high_rot, dim, base, original_max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(
        low, high, dim // 2
    ).float().to(device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_factor


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_deepseek(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig, device=None):
        super().__init__()

        # Initit module on cpu in case of meta init to avoid errors with the buffers not being
        # initialized and avoid erros with operations applied during initialization.
        if device == "meta":
            device = "cpu"

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "yarn":
            # HOTFIX: test this functionality for next iterations
            # raise AssertionError("This functionality is not yet tested")
            self.rope_init_fn = _compute_yarn_parameters
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        if self.rope_type == "yarn":
            # HOTFIX: test this functionality for next iterations
            # raise AssertionError("This functionality is not yet tested")
            mscale = config.rope_scaling.get("mscale", 1)
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 1)
            self.attention_scaling = apply_mscale_correction(
                self.attention_scaling, mscale_all_dim
            )
            self.rope_attention_scaling = float(
                apply_mscale_correction(self.attention_scaling, mscale)
                / apply_mscale_correction(self.attention_scaling, mscale_all_dim)
            )
        else:
            self.rope_attention_scaling = 1
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cached_sp_group_size = None

        # Cache `cu_seqlens` to determine if provided sample have different `position_ids`
        # notations and cached `cos` and `sin` values should be recomputed.
        #
        # No need to `copy` or `deepcopy` as `cu_seqlens` do not change outside and
        # are the same for all the modules / layers.
        self._cached_cu_seqlens = None

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def _update_cos_sin_cache(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)

            # NOTE: Do not concat `freqs` here for `emb` (done in `transformers` version)
            # as FlashAttention rotary kernels later require (seqlen_rotary, rotary_dim / 2) shape.
            emb = freqs
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.rope_attention_scaling
        sin = sin * self.rope_attention_scaling

        # cos, sin - (bs, seqlen, head_dim)
        self._cos_cached = cos.to(dtype=x.dtype)[0]
        self._sin_cached = sin.to(dtype=x.dtype)[0]
        # cos, sin - (seqlen, head_dim)

        self._seq_len_cached = position_ids.shape[-1]
        self._cached_sp_group_size = dist.get_sp_group_size() or 1

    def _need_update(
        self, x: torch.Tensor, seqlen: torch.Tensor, cu_seqlens: torch.Tensor
    ):
        if (
            seqlen != self._seq_len_cached
            or not check_tensor_equality(self._cached_cu_seqlens, cu_seqlens)
            or not self.training
            or (self.training and self._cos_cached.is_inference())
            or self._cos_cached is None
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._cached_cu_seqlens = cu_seqlens
            return True
        else:
            return False

    def _need_update_in_dynamic_sp(self):
        # When reducing sp_size (4 -> 2) during dynamic_sp, we need to update cos/sin for former ranks 2-3,
        # but we couldn't detect this when sequence length (`position_ids.shape[-1]`) remains unchanged.
        # Therefore, we add an explicit check for sp group size changed.
        sp_group_size = dist.get_sp_group_size() or 1
        if (
            self._cached_sp_group_size is None
            or self._cached_sp_group_size != sp_group_size
        ):
            # change _cached_sp_group_size in _update_cos_sin_cache
            return True
        return False

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int = None,
        position_ids: tp.Optional[torch.Tensor] = None,
        cu_seqlens: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, ...]:
        """
        x: (batch, seqlen, nheads, headdim)
        """
        seq_len = seq_len if seq_len is not None else x.shape[1]

        if (
            self._need_update(x, seq_len, cu_seqlens)
            or self._need_update_in_dynamic_sp()
        ):
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, device=x.device, dtype=torch.float32
                ).view(1, -1)
            self._update_cos_sin_cache(x, position_ids)

        return self._cos_cached, self._sin_cached


PARALLEL_EMBEDDING_REGISTRY = {
    "VocabParallelEmbedding": VocabParallelEmbedding,
    "EmbeddingParallelEmbedding": EmbeddingParallelEmbedding,
}
