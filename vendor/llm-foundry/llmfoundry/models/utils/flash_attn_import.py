import torch
import sys
from typing import List, Optional, Tuple

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_qkvpacked_func,
    _flash_attn_forward,
    _flash_attn_backward,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
    flash_attn_func,
)

HAS_FA3 = False

gpu_type = torch.cuda.get_device_capability()
gpu_major_version = gpu_type[0]
if gpu_major_version >= 9:
    try:
        from hopper.flash_attn_interface import (
            _flash_attn_forward as _flash_attn_forward_v3,
            _flash_attn_backward as _flash_attn_backward_v3,
            # flash_attn_qkvpacked_func as flash_attn_qkvpacked_func_v3, # Contains a bug in backward sp-mode MHA
            # flash_attn_func as flash_attn_func_v3, # Contains a bug in forward sp-mode MLA
            flash_attn_varlen_func as flash_attn_varlen_v3_func,
        )

        HAS_FA3 = True
    except ImportError:
        print(
            "Failed to import FlashAttention 3 for Hopper GPU. Fallback to FlashAttention 2.",
            file=sys.stderr,
        )
elif gpu_major_version < 8:
    raise RuntimeError(
        f"Unsupported GPU capability: {gpu_type=}. FlashAttention requires at least compute capability 8.0 (A100) or newer."
    )


if HAS_FA3:
    """
    Autograd methods and wrapper functions for FlashAttention-3 (Hopper GPUs).

    This block redefines (and implements from FA2 if they don't exist) key FA3 methods.
    when FlashAttention-3 is available (Hopper architecture, compute capability >= 9.0). 
    These functions mirror the FlashAttention-2 (v2.8.3) APIs, 
    but internally call FlashAttention-3 CUDA kernels from `hopper.flash_attn_interface`.

    The primary purpose is to ensure backward compatibility for higher-level code 
    expecting FA2-style interfaces while transparently leveraging FA3 performance improvements.

    ---
    Origins:
      Modified from the following sources (FlashAttention 2.8.3):
        - https://github.com/Dao-AILab/flash-attention/blob/c485eea/flash_attn/flash_attn_interface.py
        - https://github.com/Dao-AILab/flash-attention/blob/c485eea/hopper/flash_attn_interface.py

    ---
    Notes:
      • Implemented functions mirror the FlashAttention 2 public API and call the new v3 implementations
      • These definitions are only active if HAS_FA3 == True (Hopper GPU detected).
      • Dropout is currently unused in FA3 kernels but preserved for API consistency.
    """

    def canonize_strides(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.stride(0) == 1:
            empty_tensor = torch.empty(tensor.shape, dtype = tensor.dtype, device = tensor.device)
            empty_tensor.copy_(tensor)
            tensor = empty_tensor

        return tensor

    class FlashAttnFunc(torch.autograd.Function):
        """
        https://github.com/Dao-AILab/flash-attention/blob/c485eeade0c3ec9ce186c3640c52c9f1ce090b81/hopper/flash_attn_interface.py#L254
        """
        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            softmax_scale,
            causal,
            qv=None,
            q_descale=None, k_descale=None, v_descale=None,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            deterministic=False,
            sm_margin=0,
            return_softmax=False,
        ):
            if softmax_scale is None:
                softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)

            # fix MLA in sp-mode
            v = canonize_strides(v)

            # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
            out, softmax_lse, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                qv,  # qv
                None,  # out
                None, None, None,   # cu_seqlens_q/k/k_new
                None, None,   # seqused_q/k
                None, None,   # max_seqlen_q/k
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                q_descale, k_descale, v_descale,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=sm_margin,
            )
            # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.save_for_backward(q, k, v, out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.attention_chunk = attention_chunk
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.sm_margin = sm_margin
            return (out, softmax_lse) if return_softmax else out

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse = ctx.saved_tensors
            assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn_backward_v3(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                None, None, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                None, None, # max_seqlen_q, max_seqlen_k,
                dq,
                dk,
                dv,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                ctx.softcap,
                ctx.deterministic,
                ctx.sm_margin,
            )
            dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
            dk = dk[..., : k.shape[-1]]
            dv = dv[..., : v.shape[-1]]
            return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    class FlashAttnQKVPackedFunc(torch.autograd.Function):
        """
        https://github.com/Dao-AILab/flash-attention/blob/c485eeade0c3ec9ce186c3640c52c9f1ce090b81/hopper/flash_attn_interface.py#L157
        """
        @staticmethod
        def forward(
            ctx,
            qkv,
            softmax_scale,
            causal,
            q_descale=None, k_descale=None, v_descale=None,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            deterministic=False,
            num_heads_q=None,
            sm_margin=0,
            return_softmax=False,
        ):
            if softmax_scale is None:
                softmax_scale = qkv.shape[-1] ** (-0.5)
            if qkv.dim() == 5:
                assert qkv.shape[-3] == 3
                q, k, v = qkv.unbind(dim=-3)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
                q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
            out, softmax_lse, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                None,  # out
                None, None, None,   # cu_seqlens_q/k/k_new
                None, None,   # seqused_q/k
                None, None,   # max_seqlen_q/k
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                q_descale, k_descale, v_descale,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                sm_margin=sm_margin,
            )
            # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.save_for_backward(q, k, v, out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.attention_chunk = attention_chunk
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.ndim = qkv.dim()
            ctx.sm_margin = sm_margin
            return (out, softmax_lse) if return_softmax else out

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse = ctx.saved_tensors
            assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
            if ctx.ndim == 5:
                qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
                dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
                dq, dk, dv = dqkv.unbind(dim=-3)
            else:
                num_heads_q = q.shape[2]
                num_heads_k = k.shape[2]
                qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
                dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
                dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)

            # fix MHA in sp-mode
            dout = canonize_strides(dout)

            _flash_attn_backward_v3(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                None, None, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                None, None, # max_seqlen_q, max_seqlen_k,
                dq,
                dk,
                dv,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                ctx.softcap,
                ctx.deterministic,
                ctx.sm_margin,
            )
            dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
            return dqkv, None, None, None, None, None, None, None, None, None, None, None, None

    class FlashAttnKVPackedFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_softmax,
            is_grad_enabled,
        ):
            is_grad = is_grad_enabled and any(
                x.requires_grad for x in [q, kv]
            )
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()

            # fix GQA in sp-mode
            q = canonize_strides(q)

            head_size_og = q.size(3)
            if head_size_og % 8 != 0:
                q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
                k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
                v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            out_padded, softmax_lse, *rest =  _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                None,  # out
                None, None, None,   # cu_seqlens_q/k/k_new
                None, None,   # seqused_q/k
                None, None,   # max_seqlen_q/k
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                q_descale=None, k_descale=None, v_descale=None,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=0,
                softcap=softcap,
                sm_margin=0,
            )
            if is_grad:
                ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
                ctx.dropout_p = dropout_p
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.softcap = softcap
                ctx.deterministic = deterministic
            out = out_padded[..., :head_size_og]
            return out if not return_softmax else (out, softmax_lse)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse = ctx.saved_tensors
            dq = torch.empty_like(q)
            kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
            dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)

            # fix GQA in sp-mode
            dout = canonize_strides(dout)

            head_size_og = dout.size(3)
            dout_padded = dout
            if head_size_og % 8 != 0:
                dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
            _flash_attn_backward_v3(
                dout_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                None, None, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                None, None, # max_seqlen_q, max_seqlen_k,
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                ctx.softcap,
                ctx.deterministic,
                0, # sm_margin
            )
            dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
            dkv = dkv[..., : dout.shape[-1]]
            return dq, dkv, None, None, None, None, None, None, None, None

    class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_softmax,
            is_grad_enabled,
        ):
            is_grad = is_grad_enabled and any(
                x.requires_grad for x in [q, kv]
            )
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            k, v = kv[:, 0].detach(), kv[:, 1].detach()
            head_size_og = q.size(2)
            if head_size_og % 8 != 0:
                q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
                k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
                v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            out_padded, softmax_lse, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                None,  # out
                cu_seqlens_q,
                cu_seqlens_k,
                None,   # cu_seqlens_k_new
                None, # seqused_q
                None, # seqused_k
                max_seqlen_q,
                max_seqlen_k,
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                q_descale=None, k_descale=None, v_descale=None,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=0,
                softcap=softcap,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )
            if is_grad:
                ctx.save_for_backward(
                    q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k
                )
                ctx.dropout_p = dropout_p
                ctx.max_seqlen_q = max_seqlen_q
                ctx.max_seqlen_k = max_seqlen_k
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.softcap = softcap
                ctx.deterministic = deterministic
            out = out_padded[..., :head_size_og]
            return out if not return_softmax else (out, softmax_lse)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
            dq = torch.empty_like(q)
            kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
            dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
            head_size_og = dout.size(2)
            dout_padded = dout
            if head_size_og % 8 != 0:
                dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
            _flash_attn_backward_v3(
                dout_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                None, # seqused_q
                None, # seqused_k
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                dq,
                dkv[:, 0],
                dkv[:, 1],
                ctx.softmax_scale,
                ctx.causal,
                window_size=ctx.window_size,
                softcap=ctx.softcap,
                deterministic=ctx.deterministic,
                sm_margin=0,
            )
            dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
            dkv = dkv[..., : dout.shape[-1]]
            return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None

    class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_softmax,
            is_grad_enabled,
        ):
            is_grad = is_grad_enabled and qkv.requires_grad
            if softmax_scale is None:
                softmax_scale = qkv.shape[-1] ** (-0.5)
            q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
            head_size_og = q.size(2)
            if head_size_og % 8 != 0:
                q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
                k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
                v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            out_padded, softmax_lse, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                None,  # out
                cu_seqlens,
                cu_seqlens,
                None,   # cu_seqlens_k_new
                None, # seqused_q
                None, # seqused_k
                max_seqlen,
                max_seqlen,
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                q_descale = None, k_descale = None, v_descale = None,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=0,
                softcap=softcap,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )
            if is_grad:
                ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens)
                ctx.dropout_p = dropout_p
                ctx.max_seqlen = max_seqlen
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.softcap = softcap
                ctx.deterministic = deterministic
            out = out_padded[..., :head_size_og]
            return out if not return_softmax else (out, softmax_lse)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            head_size_og = dout.size(2)
            dout_padded = dout
            if head_size_og % 8 != 0:
                dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
            _flash_attn_backward_v3(
                dout_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens,
                cu_seqlens,
                None, # seqused_q
                None, # seqused_k
                ctx.max_seqlen,
                ctx.max_seqlen,
                dqkv[:, 0],
                dqkv[:, 1],
                dqkv[:, 2],
                ctx.softmax_scale,
                ctx.causal,
                window_size=ctx.window_size,
                softcap=ctx.softcap,
                deterministic=ctx.deterministic,
                sm_margin=0,
            )
            dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
            return dqkv, None, None, None, None, None, None, None, None, None, None
        
    def flash_attn_func_v3(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_attn_probs=False,
        ):
        """
        https://github.com/Dao-AILab/flash-attention/blob/c485eeade0c3ec9ce186c3640c52c9f1ce090b81/hopper/flash_attn_interface.py#L507
        """
        return FlashAttnFunc.apply(
            q,
            k,
            v,
            softmax_scale,
            causal,
            qv,
            q_descale, k_descale, v_descale,
            window_size,
            attention_chunk,
            softcap,
            num_splits,
            pack_gqa,
            deterministic,
            sm_margin,
            return_attn_probs,
        )
        
    def flash_attn_qkvpacked_func_v3(
        qkv,
        softmax_scale=None,
        causal=False,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_attn_probs=False,
    ):
        """
        https://github.com/Dao-AILab/flash-attention/blob/c485eeade0c3ec9ce186c3640c52c9f1ce090b81/hopper/flash_attn_interface.py#L445
        """
        return FlashAttnQKVPackedFunc.apply(
            qkv,
            softmax_scale,
            causal,
            q_descale, k_descale, v_descale,
            window_size,
            attention_chunk,
            softcap,
            deterministic,
            num_heads_q,
            sm_margin,
            return_attn_probs,
        )

    def flash_attn_kvpacked_func_v3(
        q,
        kv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        is_grad_enabled=True,
    ):
        return FlashAttnKVPackedFunc.apply(
            q,
            kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_attn_probs,
            is_grad_enabled,
        )

    def flash_attn_varlen_kvpacked_func_v3(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        is_grad_enabled=True,
    ):
        return FlashAttnVarlenKVPackedFunc.apply(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_attn_probs,
            is_grad_enabled,
        )

    def flash_attn_varlen_qkvpacked_func_v3(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        is_grad_enabled=True,
    ):
        return FlashAttnVarlenQKVPackedFunc.apply(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            return_attn_probs,
            is_grad_enabled,
        )

    def _flash_attn_forward_v3_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        return_softmax: bool,
        q_descale=None, k_descale=None, v_descale=None,
        attention_chunk=0,
        sm_margin=0,
    ):
        out, softmax_lse, *rest = _flash_attn_forward_v3(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )
        return out, softmax_lse, *rest

    def _flash_attn_backward_v3_wrapper(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor] = None,
        sm_margin=0,
    ):
        _flash_attn_backward_v3(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            softmax_scale,
            causal,
            (window_size_left, window_size_right),
            softcap,
            deterministic,
            sm_margin,
        )

    def _flash_attn_varlen_forward_v3_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        seqused_q=None,
        seqused_k=None,
        qv = None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        return_softmax: bool = False,
        block_table: Optional[torch.Tensor] = None,
        leftpad_k: Optional[torch.Tensor] = None,
        sm_margin=0,
        num_splits=1,
        attention_chunk=0,
    ):
        out, softmax_lse, *rest = _flash_attn_forward_v3(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=None,
            sm_margin=sm_margin,
        )
        return out, softmax_lse, *rest

    def _flash_attn_varlen_backward_v3_wrapper(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        seqused_q=None,
        seqused_k=None,
        sm_margin=0,
        rng_state: Optional[torch.Tensor] = None,
    ):
        _flash_attn_backward_v3(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            dq,
            dk,
            dv,
            softmax_scale,
            causal,
            window_size=(window_size_left, window_size_right),
            softcap=softcap,
            deterministic=deterministic,
            sm_margin=sm_margin,
        )

    def flash_attn_varlen_func_v3_wrapper(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        res = flash_attn_varlen_v3_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        return res

    flash_attn_varlen_func = flash_attn_varlen_func_v3_wrapper
    flash_attn_kvpacked_func = flash_attn_kvpacked_func_v3
    flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func_v3
    flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func_v3
    flash_attn_qkvpacked_func = flash_attn_qkvpacked_func_v3
    _flash_attn_forward = _flash_attn_forward_v3_wrapper
    _flash_attn_backward = _flash_attn_backward_v3_wrapper
    _flash_attn_varlen_forward = _flash_attn_varlen_forward_v3_wrapper
    _flash_attn_varlen_backward = _flash_attn_varlen_backward_v3_wrapper
    flash_attn_func = flash_attn_func_v3


__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_qkvpacked_func",
    "_flash_attn_forward",
    "_flash_attn_backward",
    "_flash_attn_varlen_forward",
    "_flash_attn_varlen_backward",
    "flash_attn_func",
]
