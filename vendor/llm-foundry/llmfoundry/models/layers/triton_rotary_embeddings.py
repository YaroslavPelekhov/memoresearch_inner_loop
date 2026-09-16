import torch
import triton
import triton.language as tl
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs


STATIC_TRITON_CONFIGS = {
    64: {
        "num_warps": 8,
        "num_stages": 1,
        "maxnreg": 128,
        "BLOCK_M": 16,
        "HEADS_PER_PROG": 1,
    },
}

@triton.jit
def rotary_kernel_mla(
    query, 
    seqlen, nheads, rotary_dim,
    stride_batch, stride_seqlen, stride_heads, stride_head_dim,
    rot_query,
    out_stride_batch, out_stride_seqlen, out_stride_heads, out_stride_head_dim,
    COS,
    seqlen_ro,
    cos_stride_seqlen, cos_stride_k,
    SIN,
    sin_stride_seqlen, sin_stride_k,
    head_offset,
    BLOCK_M: tl.constexpr,
    N_rot_head: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    CONJUGATE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head_block = tl.program_id(2)

    rm = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    rotary_dim_half = rotary_dim // 2
    rk_half = tl.arange(0, N_rot_head // 2)
    rk = tl.arange(0, N_rot_head)

    seq_mask = rm[:, None] < seqlen
    rm_off = rm[:, None] * stride_seqlen
    out_rm_off = rm[:, None] * out_stride_seqlen
    base_batch_ptr = query + pid_batch * stride_batch + head_offset * stride_head_dim
    out_batch_ptr = rot_query + pid_batch * out_stride_batch + head_offset * out_stride_head_dim

    cos_ptr = COS + rm[:, None] * cos_stride_seqlen + rk_half[None, :] * cos_stride_k
    sin_ptr = SIN + rm[:, None] * sin_stride_seqlen + rk_half[None, :] * sin_stride_k
    mask = (rm[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half)
    cos_val = tl.load(cos_ptr, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr, mask=mask, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin_val = -sin_val

    head_start = pid_head_block * HEADS_PER_PROG
    for h in tl.static_range(0, HEADS_PER_PROG):
        pid_head = head_start + h
        head_mask = pid_head < nheads
        head_seq_mask = head_mask & seq_mask
        base_ptr = base_batch_ptr + pid_head * stride_heads
        out_ptr = out_batch_ptr + pid_head * out_stride_heads

        if not INTERLEAVED:
            x0_ptr = base_ptr + rm_off + rk_half[None, :] * stride_head_dim
            x1_ptr = x0_ptr + rotary_dim_half * stride_head_dim
            x0_out_ptr = out_ptr + out_rm_off + rk_half[None, :] * out_stride_head_dim
            x1_out_ptr = x0_out_ptr + rotary_dim_half * out_stride_head_dim
            xmask = head_seq_mask & (rk_half[None, :] < rotary_dim_half)
            x0 = tl.load(x0_ptr, mask=xmask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptr, mask=xmask, other=0.0).to(tl.float32)
            o0 = x0 * cos_val - x1 * sin_val
            o1 = x0 * sin_val + x1 * cos_val
            tl.store(x0_out_ptr, o0.to(x0_out_ptr.dtype.element_ty), mask=xmask)
            tl.store(x1_out_ptr, o1.to(x1_out_ptr.dtype.element_ty), mask=xmask)
        else:
            x_ptr = base_ptr + rm_off + rk[None, :] * stride_head_dim
            x_out_ptr = out_ptr + out_rm_off + rk[None, :] * out_stride_head_dim
            xmask = head_seq_mask & (rk[None, :] < rotary_dim)
            x = tl.load(x_ptr, mask=xmask, other=0.0).to(tl.float32)
            x0, x1 = tl.split(tl.reshape(x, [BLOCK_M, N_rot_head // 2, 2]))
            o0 = x0 * cos_val - x1 * sin_val
            o1 = x0 * sin_val + x1 * cos_val
            o = tl.reshape(tl.join(o0, o1), [BLOCK_M, N_rot_head])
            tl.store(x_out_ptr, o.to(x_out_ptr.dtype.element_ty), mask=xmask)


def apply_rotary_mla(
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_offset: int,
    inplace: bool = True,
    conjugate: bool = False,
    interleaved: bool = False
) -> torch.Tensor:
    """
    Applies rotary embeddings to a slice of the `query` tensor starting at `head_offset`
    along the last dimension. The input `query` has shape (batch, seqlen, nheads, head_dim),
    and the rotary-applied part is `query[..., head_offset:]`.

    Args:
        query: Input tensor of shape (batch, seqlen, nheads, head_dim).
        cos: Cosine table of shape (seqlen_ro, rotary_dim / 2).
        sin: Sine table of shape (seqlen_ro, rotary_dim / 2).
        head_offset: Offset in the last dimension where the rotary part starts.
        inplace: Whether to apply the operation in-place (default: True).
        conjugate: Whether to apply the conjugate variant (negate `sin`) (default: False).

    Returns:
        The `query` tensor with rotary embeddings applied to the specified slice.
    """
    if not inplace:
        rot_query = query.clone()
    else:
        rot_query = query
    batch, seqlen, nheads, head_dim = query.shape
    rotary_dim = head_dim - head_offset
    assert cos.shape[1] * 2 == rotary_dim, "rotary_dim mismatch: expected cos.shape[1] * 2 to equal rotary_dim"
    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    grid = lambda meta: (
        batch,
        triton.cdiv(seqlen, meta["BLOCK_M"]),
        triton.cdiv(nheads, meta["HEADS_PER_PROG"]),
    )
    rotary_kernel_mla[grid](
        query,
        seqlen, nheads, rotary_dim,
        *query.stride(),
        rot_query,
        *rot_query.stride(),
        cos,
        cos.shape[0],
        *cos.stride(),
        sin,
        *sin.stride(),
        head_offset,
        N_rot_head=triton.next_power_of_2(rotary_dim),
        CONJUGATE=conjugate,
        INTERLEAVED=interleaved,
        **params_to_kernel_kwargs(STATIC_TRITON_CONFIGS, rotary_dim,
                                  num_warps=8, num_stages=1, 
                                  maxnreg=128, BLOCK_M=16, HEADS_PER_PROG=1)
    )
    return rot_query


class ApplyRotaryEmbMLA(torch.autograd.Function):
    """
    Logic is similar to:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L38
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_offset: int,
        inplace: bool = True,
        interleaved: bool = True
    ):
        out = apply_rotary_mla(
            query,
            cos,
            sin,
            head_offset=head_offset,
            inplace=inplace,
            interleaved=interleaved
        )
        ctx.save_for_backward(cos, sin)
        ctx.head_offset = head_offset
        ctx.inplace = inplace
        ctx.interleaved = interleaved
        return out if not inplace else query

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        cos, sin = ctx.saved_tensors

        dx = apply_rotary_mla(
            do,
            cos,
            sin,
            head_offset=ctx.head_offset,
            inplace=ctx.inplace,
            interleaved=ctx.interleaved,
            conjugate=True,
        )
        return dx, None, None, None, None, None


def apply_rotary_emb_mla(
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_offset: int,
    inplace: bool = True,
    interleaved: bool = True
) -> torch.Tensor:
    return ApplyRotaryEmbMLA.apply(query, cos, sin, head_offset, inplace, interleaved)
