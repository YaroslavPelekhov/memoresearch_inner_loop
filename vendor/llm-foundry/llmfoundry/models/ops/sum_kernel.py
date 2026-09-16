import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
#         triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
#     ],
#     key=['num_elements'],
# )
@triton.jit
def add_bf16_grad_tensors_kernel(
    A_ptr, 
    B_ptr, 
    num_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    a_fp32 = a.to(tl.float32)
    b_fp32 = b.to(tl.float32)
    result_fp32 = a_fp32 + b_fp32
    result = result_fp32.to(tl.bfloat16)
    tl.store(A_ptr + offsets, result, mask=mask)


def sum_2shape_bf16_grads(A: torch.Tensor, B: torch.Tensor):
    if A.shape != B.shape:
        raise ValueError(f"Expected matching grad shapes, got {A.shape} and {B.shape}")
        
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 tensors, got {A.dtype} and {B.dtype}")
    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    num_elements = A_flat.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    add_bf16_grad_tensors_kernel[grid](
        A_flat, 
        B_flat, 
        num_elements,
        BLOCK_SIZE=512,
        num_warps=8
    )
