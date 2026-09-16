import typing as tp

import torch
from torch.autograd.profiler import record_function as record_fn

from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

from llmfoundry.models.ops.float8.triton_kernels import (
    block_quant_fn,
    block_quant_trans_fn,
    row_quant_fn,
    row2col_deep_gemm_fn,
)
from llmfoundry.models.ops.float8.cuda_kernels.row2col_te_quantization import (
    row2col_prepare,
    row2col_execute_cached,
)

from llmfoundry.models.ops.float8.triton_kernels.row2col_te_quantization import row2col_grouped
from llmfoundry.models.ops.float8.cuda_kernels.row2col_dq_quantization import (
    row2col_requantize as row2col_cuda_deep_gemm_fn,
)
from llmfoundry.models.ops.float8.deep_gemm_handle import _deep_gemm_handle

_cached_handle: tp.Optional[_deep_gemm_handle] = None


def _get_deep_gemm_handle(num_sms: tp.Optional[int] = None) -> _deep_gemm_handle:
    global _cached_handle
    if _cached_handle is None:
        _cached_handle = _deep_gemm_handle(num_sms=num_sms)
    return _cached_handle


def clear_tensor_data(*tensors: tp.Tuple[tp.Optional[torch.Tensor], ...]) -> None:
    for t in tensors:
        if t is not None:
            if isinstance(t, Float8BlockwiseQTensor):
                t.clear()
            else:
                t.data = torch.Tensor()


class _GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                tensor_groups: torch.Tensor,
                list_groups_sizes_padded: tp.List[int],
                weights_quantized: tp.Optional[tp.Tuple[Float8BlockwiseQTensor, ...]],
                wgrad_backend: str,
                weights: torch.Tensor,
                m_indices: torch.Tensor,
                tensor_groups_sizes_padded: torch.Tensor,
                triton_row2col: bool,) -> torch.Tensor:

        ctx.device = tensor_groups.device
        ctx.weight_shape = weights.shape
        ctx.inp_shape = tensor_groups.shape
        ctx.num_groups = len(list_groups_sizes_padded)

        with record_fn("grouped_gemm::fwd"):
            ctx.is_quantized_input = isinstance(tensor_groups, Float8BlockwiseQTensor)
            if not ctx.is_quantized_input:
                with record_fn(f"grouped_gemm::fwd::row_quant_fn"):
                    tensor_groups_quantized = row_quant_fn(tensor_groups)
            else:
                tensor_groups_quantized = (
                    tensor_groups._rowwise_data,
                    tensor_groups._rowwise_scale_inv)

            if weights_quantized is None:
                with record_fn(f"grouped_gemm::fwd::block_quant_fn"):
                    weights_quantized = block_quant_fn(weights)
            assert weights_quantized is not None

            ctx.m_indices = m_indices

            with record_fn(f"grouped_gemm::fwd::m_grouped_fp8_gemm"):
                num_groups, int_dim, _ = weights_quantized[0].shape

                out = torch.empty(
                    [sum(list_groups_sizes_padded), int_dim],
                    dtype=torch.bfloat16,
                    device=ctx.device,
                    requires_grad=True)

                weights_quantized = (
                    weights_quantized[0].reshape(num_groups, -1, weights_quantized[0].size(-1)),
                    weights_quantized[1].reshape(num_groups, -1, weights_quantized[1].size(-1)),)

                tensor_groups_quantized = (
                    tensor_groups_quantized[0].view(torch.float8_e4m3fn).contiguous(),
                    tensor_groups_quantized[1],)
                    # tensor_groups_quantized[1].mT.contiguous().mT,)

                _get_deep_gemm_handle().m_grouped_fp8_gemm_nt_contiguous(
                    tensor_groups_quantized, weights_quantized, out, m_indices)

        ctx.needs_wgrad = weights.requires_grad
        if ctx.needs_wgrad:
            ctx.save_for_backward(*tensor_groups_quantized)
        ctx.weights = weights
        ctx.list_groups_sizes = list_groups_sizes_padded
        ctx.tensor_groups_sizes = tensor_groups_sizes_padded
        ctx.wgrad_backend = wgrad_backend
        ctx.triton_row2col = triton_row2col
        clear_tensor_data(*weights_quantized)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: tp.Union[torch.Tensor, Float8BlockwiseQTensor],
    ) -> tp.Tuple[tp.Optional[torch.Tensor], ...]:

        with record_fn("grouped_gemm::bwd"):
            needs_wgrad = ctx.needs_wgrad
            weights = ctx.weights
            tensor_groups_quantized = ctx.saved_tensors if needs_wgrad else ()

            # DGRAD
            is_quantized_grad_output = isinstance(grad_output, Float8BlockwiseQTensor)
            if is_quantized_grad_output:
                assert isinstance(grad_output, Float8BlockwiseQTensor)
                assert grad_output._rowwise_data.shape[0] == grad_output._rowwise_scale_inv.shape[0]
                assert grad_output._rowwise_data.shape[1] // 128 == grad_output._rowwise_scale_inv.shape[1]
                grad_output_quantized = (
                    grad_output._rowwise_data.view(torch.float8_e4m3fn),
                    grad_output._rowwise_scale_inv)
            else:
                with record_fn(f"grouped_gemm::bwd::row_quant_fn"):
                    grad_output_quantized = row_quant_fn(grad_output)

            assert grad_output_quantized[0].shape[0] == grad_output_quantized[1].shape[0]
            assert grad_output_quantized[0].shape[1] // 128 == grad_output_quantized[1].shape[1]

            # weights quantization for dgrad
            with record_fn(f"grouped_gemm::bwd::block_quant_trans_fn"):
                weights_quantized = block_quant_trans_fn(weights)

            # dgrad = grad_output x weights
            with record_fn(f"grouped_gemm::bwd::dgrad_m_grouped_fp8_gemm"):
                dgrad = torch.empty(
                    [sum(ctx.list_groups_sizes), ctx.inp_shape[-1]],
                    dtype=torch.bfloat16,
                    device=ctx.device,)

                weights_quantized = (
                    weights_quantized[0].reshape(ctx.num_groups, -1, weights_quantized[0].size(-1)),
                    weights_quantized[1].reshape(ctx.num_groups, -1, weights_quantized[1].size(-1)),)

                grad_output_quantized = (
                    grad_output_quantized[0],
                    grad_output_quantized[1],)
                    # grad_output_quantized[1].mT.contiguous().mT,)

                prep_tensor, prep_grad = None, None
                if (needs_wgrad and ctx.wgrad_backend == "transformer_engine" and not ctx.triton_row2col):
                    prep_tensor = row2col_prepare(
                        tensor_groups_quantized[0],
                        tensor_groups_quantized[1],
                        ctx.list_groups_sizes,
                        ctx.m_indices)
                    prep_grad = row2col_prepare(
                        grad_output_quantized[0],
                        grad_output_quantized[1],
                        ctx.list_groups_sizes,
                        ctx.m_indices)

                _get_deep_gemm_handle().m_grouped_fp8_gemm_nt_contiguous(
                    grad_output_quantized, weights_quantized, dgrad, ctx.m_indices)

                clear_tensor_data(*weights_quantized)

            # WGRAD (skipped entirely when weights are frozen)
            wgrad = None
            if needs_wgrad:

                if ctx.wgrad_backend == "deep_gemm":
                    with record_fn(f"grouped_gemm::bwd::row2col_deep_gemm_fn"):

                        if ctx.triton_row2col:
                            tensor_groups_quantized_col = row2col_deep_gemm_fn(
                                tensor_groups_quantized[0],
                                ctx.m_indices,
                                ctx.tensor_groups_sizes,
                                tensor_groups_quantized[1],)
                        else:
                            tensor_groups_quantized_col = row2col_cuda_deep_gemm_fn(
                                tensor_groups_quantized[0],
                                ctx.m_indices,
                                ctx.tensor_groups_sizes,
                                tensor_groups_quantized[1],)

                        clear_tensor_data(*tensor_groups_quantized)

                        if ctx.triton_row2col:
                            grad_output_quantized_col = row2col_deep_gemm_fn(
                                grad_output_quantized[0],
                                ctx.m_indices,
                                ctx.tensor_groups_sizes,
                                grad_output_quantized[1],)
                        else:
                            grad_output_quantized_col = row2col_cuda_deep_gemm_fn(
                                grad_output_quantized[0],
                                ctx.m_indices,
                                ctx.tensor_groups_sizes,
                                grad_output_quantized[1],)

                        clear_tensor_data(*grad_output_quantized)

                    # TODO: replace with torch.empty
                    wgrad = torch.zeros(
                        ctx.weight_shape, dtype=torch.bfloat16, device=ctx.device)

                    # wgrad = torch.zeros(
                    #     ctx.weight_shape, dtype=torch.float32, device=ctx.device,)

                    grad_output_quantized_col = (
                        grad_output_quantized_col[0].view(torch.float8_e4m3fn),
                        grad_output_quantized_col[1].mT)

                    tensor_groups_quantized_col = (
                        tensor_groups_quantized_col[0].view(torch.float8_e4m3fn),
                        tensor_groups_quantized_col[1].mT,)

                    _get_deep_gemm_handle().k_grouped_fp8_gemm_contiguous(
                        a=grad_output_quantized_col,
                        b=tensor_groups_quantized_col,
                        d=wgrad,
                        ks=ctx.list_groups_sizes,
                        ks_tensor=ctx.tensor_groups_sizes,)

                    clear_tensor_data(*grad_output_quantized_col)
                    clear_tensor_data(*tensor_groups_quantized_col)

                else:
                    with record_fn(f"grouped_gemm::bwd::row2col_grouped"):

                        if not ctx.triton_row2col:
                            assert prep_tensor is not None and prep_grad is not None
                            tensor_groups_quantized_col = row2col_execute_cached(
                                prep_tensor, cache_slot="tensor")
                            grad_output_quantized_col = row2col_execute_cached(
                                prep_grad, cache_slot="grad")
                            prep_tensor = None
                            prep_grad = None

                        else:
                            tensor_groups_quantized_col = row2col_grouped(
                                tensor_groups_quantized[0],
                                tensor_groups_quantized[1],
                                ctx.list_groups_sizes,
                                ctx.m_indices,
                                ctx.tensor_groups_sizes)

                            grad_output_quantized_col = row2col_grouped(
                                grad_output_quantized[0],
                                grad_output_quantized[1],
                                ctx.list_groups_sizes,
                                ctx.m_indices,
                                ctx.tensor_groups_sizes)

                        clear_tensor_data(*tensor_groups_quantized)
                        clear_tensor_data(*grad_output_quantized)

                    # wgrad = grad_output^T x input
                    with record_fn(f"grouped_gemm::bwd::te_wgrad"):
                        wgrad = torch.empty(
                            ctx.weight_shape,
                            dtype=torch.bfloat16,
                            device=ctx.device)
                        wgrad_ptrs = wgrad.unbind(0)
                        general_grouped_gemm(
                            tensor_groups_quantized_col,
                            grad_output_quantized_col,
                            wgrad_ptrs,
                            torch.bfloat16,
                            get_multi_stream_cublas_workspace(),
                            layout="NT",
                            grad=True,
                            m_splits=ctx.list_groups_sizes,
                            use_bias=None,
                            bias=None,
                            use_split_accumulator=True,
                            accumulate=False)

                        clear_tensor_data(*grad_output_quantized_col)
                        clear_tensor_data(*tensor_groups_quantized_col)

            else:
                clear_tensor_data(*grad_output_quantized)

        return (dgrad.view(ctx.inp_shape),
                None,
                None,
                None,
                wgrad,
                None,
                None,
                None)


class GroupedGemmFp8Wrapper:

    def __init__(self, num_groups: int, num_sms: tp.Optional[int] = None) -> None:
        super().__init__()
        self.async_quantized_weights: bool = False
        self.weights_quantized: tp.Optional[tp.Tuple[Float8BlockwiseQTensor, ...]] = None
        _get_deep_gemm_handle(num_sms=num_sms)

    def async_weight_quantizer(self, weights: torch.Tensor) -> tp.Tuple[Float8BlockwiseQTensor, ...]:
        self.async_quantized_weights = True
        weights_quantized = block_quant_fn(weights)
        return weights_quantized

    def group_gemm(self,
                   tensor_groups: torch.Tensor,
                   tensor_groups_sizes: torch.Tensor,
                   wgrad_backend: str,
                   weights: torch.Tensor,
                   weights_quantized: tp.Optional[tp.Tuple[Float8BlockwiseQTensor, ...]] = None,
                   m_indices: tp.Optional[torch.Tensor] = None,
                   tensor_groups_sizes_padded: tp.Optional[torch.Tensor] = None,
                   triton_row2col: bool = False) -> torch.Tensor:

        return _GroupedGemm.apply(
            tensor_groups,
            tensor_groups_sizes,
            weights_quantized,
            wgrad_backend,
            weights,
            m_indices,
            tensor_groups_sizes_padded,
            triton_row2col,)
