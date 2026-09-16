from functools import cache

import torch
from torch import nn
from torch.amp import custom_fwd, custom_bwd

import flux
import torch.distributed


from composer.utils import dist


class KernelFactory:
    # all constants set to flux' benchmark defaults
    all_gather_local_copy: bool = True
    all_gather_ring_mode: int = -1
    all_gather_fast_accum: bool = False
    reduce_scatter_fuse_reduction = False  # bench default was False

    @torch.no_grad
    def all_gather_gemm(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
        transpose_weight: bool,
    ):
        assert len(input_.size()) == 2

        local_seq_len, in_features = input_.size()
        out_features = weight.size(1) if transpose_weight else weight.size(0)

        kernel = self.make_all_gather_gemm_kernel(
            local_seq_len, out_features, in_features, input_.dtype, transpose_weight
        )
        
        if self.all_gather_local_copy:
            kernel.copy_local(input_)

        return kernel.forward(
            input_,
            weight,
            bias=None,
            input_scale=None,
            weight_scale=None,
            output_scale=None,
            fast_accum=self.all_gather_fast_accum,
        )

    @torch.no_grad
    def gemm_reduce_scatter(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
        transpose_weight: bool,
    ):
        assert len(input_.size()) == 2

        seq_len = input_.size(0)
        out_features = weight.size(1) if transpose_weight else weight.size(0)

        gemm_rs_kernel = self.make_gemm_reduce_scatter_kernel(
            seq_len, out_features, input_.dtype, transpose_weight
        )

        # for some reason this call returns a fp32 tensor in autocast=true region
        with torch.autocast("cuda", enabled=False):
            return gemm_rs_kernel.forward(
                input_,
                weight,
                bias=None,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False,
            )

    @classmethod
    @cache
    def make_all_gather_gemm_kernel(
        cls,
        local_batch_size: int,
        out_features: int,
        in_features: int,
        input_dtype: torch.dtype,
        transposed_weight: bool,
    ):
        full_batch_size = local_batch_size * dist.get_tp_group_size()

        return flux.AGKernel(
            dist.get_tp_group(),
            nnodes=1,
            full_m=full_batch_size,
            n_dim=out_features,
            k_dim=in_features,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            transpose_weight=transposed_weight,
            local_copy=cls.all_gather_local_copy,
            ring_mode=flux.AgRingMode(cls.all_gather_ring_mode),
        )

    @classmethod
    @cache
    def make_gemm_reduce_scatter_kernel(
        cls,
        batch_size: int,
        in_features: int,
        input_dtype: torch.dtype,
        transpose_weight: bool,
    ):
        return flux.GemmRS(
            tp_group=dist.get_tp_group(),
            nnodes=1,
            max_m=batch_size,
            n_dim=in_features,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            transpose_weight=transpose_weight,
            fuse_reduction=cls.reduce_scatter_fuse_reduction,
        )


class _ColumnParallelLinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.Function,
        weight: torch.Tensor,
        input_: torch.Tensor,
        kernel_factory: KernelFactory,
    ):
        ctx.save_for_backward(weight, input_)
        ctx.kernel_factory = kernel_factory

        return kernel_factory.all_gather_gemm(
            input_, weight, transpose_weight=False
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
    ):
        input_: torch.Tensor
        weight, input_ = ctx.saved_tensors

        batch_size = grad_output.size(0)
        in_features = input_.size(1)

        full_inputs = torch.empty(
            batch_size,
            in_features,
            dtype=input_.dtype,
            device=input_.device,
            requires_grad=False,
        )
        handle = torch.distributed.all_gather_into_tensor(
            full_inputs, input_, group=dist.get_tp_group(), async_op=True
        )

        # NB:
        # weight.grad.T = (grad_output.T @ full_input).T 
        #               = full_input.T @ grad_output
        #               = all_gather_gemm(input_=input_, weight=grad_output, transpose_weight=False)

        kernel_factory: KernelFactory = ctx.kernel_factory
        grad_input = kernel_factory.gemm_reduce_scatter(
            grad_output, weight, transpose_weight=True
        )
        handle.wait()
        grad_weight = grad_output.T @ full_inputs

        return grad_weight, grad_input, None

class _RowParallelLinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.Function,
        weight: torch.Tensor,
        input_: torch.Tensor,
        kernel_factory: KernelFactory,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight, input_)
        ctx.kernel_factory = kernel_factory

        output = kernel_factory.gemm_reduce_scatter(input_, weight, False)
        out_features = weight.size(0)
        return output.view(-1, out_features)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
    ):
        grad_output = grad_output
        input_: torch.Tensor
        weight: torch.Tensor
        weight, input_ = ctx.saved_tensors

        seq_len = input_.size(0)
        out_features = grad_output.size(1)

        full_grad_output = torch.empty(
            (seq_len, out_features),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        handle = torch.distributed.all_gather_into_tensor(
            full_grad_output, grad_output, group=dist.get_tp_group(), async_op=True
        )

        kernel_factory: KernelFactory = ctx.kernel_factory
        grad_input = kernel_factory.all_gather_gemm(
            grad_output, weight, transpose_weight=True
        )

        # NB:
        # weight.grad = full_grad_output.T @ input_
        #             = all_gather_gemm(full_grad_output.T, input_)  // flux does not work well with strided inputs
        handle.wait()
        grad_weight = full_grad_output.T @ input_

        return grad_weight, grad_input, None


class FluxColumnParallelFn:
    def __init__(self):
        self._kernel_factory = KernelFactory()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
        logical_batch_size: int,
    ) -> torch.Tensor:
        in_feat = input_.size(-1)
        input_ = input_.view(-1, in_feat)
        out = _ColumnParallelLinearFunc.apply(
            weight, input_, self._kernel_factory
        )
        return out.view(logical_batch_size, -1, out.size(-1))

    def __call__(
        self, input_: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(input_, weight)


class FluxRowParallelFn:
    def __init__(self):
        self._kernel_factory = KernelFactory()

    def forward(self, input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        in_feat = input_.size(-1)
        input_ = input_.view(-1, in_feat)
        out = _RowParallelLinearFunc.apply(weight, input_, self._kernel_factory)
        return out.unsqueeze(0)

    def __call__(
        self, input_: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(input_, weight)
