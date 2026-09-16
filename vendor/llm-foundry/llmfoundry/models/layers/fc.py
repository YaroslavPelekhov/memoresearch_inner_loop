# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import re
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from composer.utils.dist import get_tp_group_size, get_tp_group_rank, get_tp_group
from llmfoundry.models.ops.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from llmfoundry.models.parallel.sequence.all_to_all import SeqAllToAll
from llmfoundry.models.parallel.tensor import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    _initialize_tp_weight,
    set_tensor_model_parallel_attributes,
    all_gather_linear,
    linear_reduce_scatter,
)


class LinearWithFrozenWeight(torch.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as torch.nn.functional.linear with
    weight.requires_grad==False, but in experiments they are not identical
    mathematically."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        input,
        weight,
        bias,
    ):
        ctx.save_for_backward(weight)
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        return grad_input, None, None


def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    async_grad_allreduce (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """

    args = [
        input,
        weight,
        bias,
    ]

    return LinearWithFrozenWeight.apply(*args)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        input,
        weight,
        bias,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        total_input = input
        grad_input = grad_output.matmul(weight)

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel is True, this must be
        False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """

    args = [input, weight, bias]
    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
       skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed
                                      as a keyword argument `weight` during the forward pass. Note
                                      that this does not affect bias, which will be allocated if
                                      bias is True. Defaults to False.
        is_expert: If True, the layer is treated as an MoE expert layer.
        config: ModelParallelConfig object
        tp_comm_buffer_name: Communication buffer name is not used in
                             non-Transformer-Engine modules.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,  # type: ignore
        bias: bool = True,
        lm_head: bool = False,
        gather_output: bool = False,
        keep_master_weight_for_test: bool = False,
        init_method: Optional[Callable] = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        self.config = config
        self.use_bias: bool = bias
        self.tp_size: int = get_tp_group_size() if get_tp_group_size() else 1

        # TODO (sbr, fedorovgv) : check this
        # assert lm_head or self.tp_size > 1, "Cannot use ColumnParallelLinear in tp1 because it does not convert to fp8 unless it is lm_head."

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.fused_gather_input = False

        if self.tp_size == 1:
            factory_kwargs = {
                "device": config.init_device,
                "dtype": getattr(config, "dtype", None),
            }

            self._init_params = {
                "init_method": nn.init.xavier_normal_,
                "init_method_forced": init_method,
            }

            self.weight = nn.Parameter(
                torch.empty((output_size, input_size), **factory_kwargs)
            )
            if self.use_bias:
                self.bias: Union[torch.Tensor, None] = nn.Parameter(
                    torch.empty(output_size, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)

        else:
            # Divide the weight matrix along the last dimension.
            assert output_size % self.tp_size == 0, (
                f"output_size = {output_size} is not divisible by self.tp_size = {self.tp_size}"
            )

            self.output_size_per_partition = output_size // self.tp_size

            # Parameters.
            # Note: torch.nn.functional.linear performs XA^T + b and as a result
            # we allocate the transpose.
            # Initialize weight.
            self._init_params = {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "partition_dim": 0,
                "per_partition_size": self.output_size_per_partition,
                "init_method": nn.init.xavier_normal_,  # actually not used
                "init_method_forced": init_method,  # if set, use this instead of param_init_fn's init_fn_
                "return_master_weight": keep_master_weight_for_test,
                "device": config.init_device,
                "use_master_weight": config.use_master_weight,
                "init_type": config.init_type,
            }

            self.fused_gather_input = getattr(config, "enable_async_tp", False)

            self._tp_linear_submodule = nn.Linear(
                in_features=self.input_size,
                out_features=self.output_size_per_partition,
                bias=self.use_bias,
                device=config.init_device,
                dtype=getattr(config, "dtype", None),
            )
            if self.use_bias:
                set_tensor_model_parallel_attributes(
                    self._tp_linear_submodule.bias, True, 0
                )

        # Initialize weights if on cpu device and skip for meta device.
        # For meta device params are initialized on FSDP model wrapping in Composer
        # using `param_init_fn`.

        if config.init_device != "meta":
            self.reset_parameters()

        self._skip_init_module = config.skip_init_tp_modules
        self._init_rules()

    def reset_parameters(self) -> None:
        if self.tp_size > 1:
            # Update device before init to account for tensor being moved
            self._init_params["device"] = self._tp_linear_submodule.weight.device
            assert torch.device("meta") != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            # If init_method_forced is set, use it instead of the default init method
            init_method_forced = self._init_params.pop("init_method_forced", None)
            if init_method_forced is not None:
                self._init_params["init_method"] = init_method_forced
            _initialize_tp_weight(self._tp_linear_submodule.weight, **self._init_params)
            # reset_parameters will be called twice for cpu init -> still need this function
            self._init_params["init_method_forced"] = init_method_forced
            if self.use_bias:
                torch.nn.init.zeros_(self._tp_linear_submodule.bias)
        else:
            init_method_forced = self._init_params.pop("init_method_forced", None)
            if init_method_forced is not None:
                self._init_params["init_method"] = init_method_forced
            self._init_params["init_method"](self.weight)
            if self.use_bias:
                torch.nn.init.zeros_(self.bias)

    def _init_rules(self):
        if self.tp_size > 1:

            def raise_init_error(*args, **kwargs):
                raise RuntimeError(
                    "This module must be inited by it's parent reset_parameters"
                )

            self._tp_linear_submodule.skip_init = self._skip_init_module
            self._tp_linear_submodule.reset_parameters = raise_init_error

    def forward(
        self,
        input_: torch.Tensor,
        target: Union[torch.Tensor, None] = None,
        gather_output: Optional[bool] = None,
        logical_batch_size: Optional[int] = None,
        use_liger: Optional[bool] = False,
        reduction: str = "none",
        lse_square_scale: float = 0.0,
        ignore_index: int = -100,
        process_group=None,  # type: ignore
        return_z_loss: bool = False,
        precomputed_grad_output: Optional[torch.Tensor] = None,
        no_recompute: bool = False,
    ) -> torch.Tensor:
        if use_liger:
            loss = self.liger_forward(
                input_,
                target,
                reduction,
                lse_square_scale,
                ignore_index,
                process_group,
                self.fused_gather_input,
                return_z_loss,
                precomputed_grad_output,
            )

            return loss

        if self.tp_size > 1:
            logits = self.tp_linear_forward(
                input_=input_,
                gather_output=gather_output,
                logical_batch_size=logical_batch_size,
                no_recompute=no_recompute,
            )
        elif self.config.pretraining_tp > 1:
            lm_head_slices = self.weight.split(
                self.config.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(input_, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = F.linear(input_, self.weight, self.bias)

        return logits

    def liger_forward(
        self,
        input_: torch.Tensor,
        target: Union[torch.Tensor, None] = None,
        reduction: str = "none",
        lse_square_scale: float = 0.0,
        ignore_index: int = -100,
        process_group=None,  # type: ignore
        enable_async_tp: bool = False,
        return_z_loss: bool = False,
        precomputed_grad_output: Optional[torch.Tensor] = None,
    ):
        if self.tp_size == 1:
            lin_weight = self.weight
            bias = self.bias
        else:
            assert process_group is not None, (
                "In tensor parallelism > 1, when using Liger CE, process_group can't be None."
            )

            lin_weight = self._tp_linear_submodule.weight
            bias = self._tp_linear_submodule.bias

        loss_fct = FusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            lse_square_scale=lse_square_scale,
            process_group=process_group,
            enable_async_tp=enable_async_tp,
            return_z_loss=return_z_loss,
            precomputed_grad_output=precomputed_grad_output,
        )

        loss = loss_fct(input_, lin_weight, target, bias)

        return loss

    def tp_linear_forward(
        self,
        input_: torch.Tensor,
        gather_output: Optional[bool] = None,
        logical_batch_size: Optional[int] = None,
        no_recompute: bool = False,
    ) -> torch.Tensor:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        if gather_output is None:
            gather_output = self.gather_output

        weight = self._tp_linear_submodule.weight
        bias = self._tp_linear_submodule.bias

        if self.fused_gather_input:
            forward_impl = lambda input, weight, bias: all_gather_linear(
                input,
                weight,
                bias=bias,
                logical_batch_size=logical_batch_size,
                no_recompute=no_recompute,
            )
            output_parallel = forward_impl(input=input_, weight=weight, bias=bias)
        else:
            input_ = copy_to_tensor_model_parallel_region(input_)
            output_parallel = self._tp_linear_submodule(input_)

        if gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output

    @staticmethod
    def gigafsdp_normalize_fqns(param_name: str) -> str:
        return re.sub(r"\.?_tp_linear_submodule", "", param_name)


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
       is_expert: If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name: Communication buffer name. Not used in
                             non-Transformer-Engine modules.
        config: ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        bias: bool,
        input_is_parallel: bool,
        keep_master_weight_for_test: Optional[bool] = False,
        init_method: Optional[Callable] = None,
    ):
        super(RowParallelLinear, self).__init__()

        self.config = config
        self.use_bias: bool = bias
        self.tp_size: int = get_tp_group_size() if get_tp_group_size() else 1

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        if self.tp_size == 1:
            factory_kwargs = {
                "device": config.init_device,
                "dtype": getattr(config, "dtype", None),
            }

            self._init_params = {
                "init_method": nn.init.xavier_normal_,
                "init_method_forced": init_method,
            }

            self.weight = nn.Parameter(
                torch.empty((output_size, input_size), **factory_kwargs)
            )
            if self.use_bias:
                self.bias: Union[torch.Tensor, None] = nn.Parameter(
                    torch.empty(output_size, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)

        else:
            # Divide the weight matrix along the last dimension.
            assert self.tp_size == 1 or input_size % self.tp_size == 0, (
                f"input_size = {input_size} is not divisible by self.tp_size = {self.tp_size}"
            )

            self.input_size_per_partition = input_size // self.tp_size

            # Parameters.
            # Note: torch.nn.functional.linear performs XA^T + b and as a result
            # we allocate the transpose.
            # Initialize weight.
            self._init_params = {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "partition_dim": 1,
                "per_partition_size": self.input_size_per_partition,
                "init_method": nn.init.xavier_normal_,
                "init_method_forced": init_method,
                "return_master_weight": keep_master_weight_for_test,
                "device": config.init_device,
                "use_master_weight": config.use_master_weight,
                "init_type": config.init_type,
            }

            self.fused_reduce_scatter_output = getattr(config, "enable_async_tp", False)

            self._tp_linear_submodule = nn.Linear(
                in_features=self.input_size_per_partition,
                out_features=self.output_size,
                bias=False,
                device=config.init_device,
                dtype=getattr(config, "dtype", None),
            )

            if self.use_bias:
                self.bias = nn.Parameter(
                    torch.empty(self.output_size, device=config.init_device)
                )
                setattr(self.bias, "allreduce", True)
            else:
                self.register_parameter("bias", None)

        # Initialize weights if on cpu device and skip for meta device.
        # For meta device params are initialized on FSDP model wrapping in Composer
        # using param_init_fn.
        if config.init_device != "meta":
            self.reset_parameters()

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
        self._skip_init_module = config.skip_init_tp_modules
        self._init_rules()

    def reset_parameters(self) -> None:
        if self.tp_size > 1:
            # Update device before init to account for tensor being moved
            self._init_params["device"] = self._tp_linear_submodule.weight.device

            assert torch.device("meta") != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            # If init_method_forced is set, use it instead of the default init method
            init_method_forced = self._init_params.pop("init_method_forced", None)
            if init_method_forced is not None:
                self._init_params["init_method"] = init_method_forced
            _initialize_tp_weight(self._tp_linear_submodule.weight, **self._init_params)
            # reset_parameters will be called twice for cpu init -> still need this function
            self._init_params["init_method_forced"] = init_method_forced
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        else:
            init_method_forced = self._init_params.pop("init_method_forced", None)
            if init_method_forced is not None:
                self._init_params["init_method"] = init_method_forced
            self._init_params["init_method"](self.weight)
            if self.use_bias:
                torch.nn.init.zeros_(self.bias)

    def _init_rules(self):
        if self.tp_size > 1:

            def raise_init_error(*args, **kwargs):
                raise RuntimeError(
                    "This module must be inited by it's parent reset_parameters"
                )

            self._tp_linear_submodule.skip_init = self._skip_init_module
            self._tp_linear_submodule.reset_parameters = raise_init_error

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            logits = self.tp_linear_forward(input_=input_)
        else:
            logits = F.linear(input_, self.weight, self.bias)
        return logits

    def tp_linear_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        elif self.fused_reduce_scatter_output:
            # In asynctp case
            # input shape is [1, b x seqlen // tp, hidden],
            # output shape is [1, b x seqlen, hidden // tp]
            input_parallel = SeqAllToAll.apply(input_, 1, 2, get_tp_group())
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        if self.fused_reduce_scatter_output:
            return linear_reduce_scatter(
                input_parallel, self._tp_linear_submodule.weight, self.bias
            )

        output_parallel = self._tp_linear_submodule(input_parallel)

        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = output_ + self.bias if self.bias is not None else output_
        return output

    @staticmethod
    def gigafsdp_normalize_fqns(param_name: str) -> str:
        return re.sub(r"\.?_tp_linear_submodule", "", param_name)


FC_CLASS_REGISTRY = {
    "torch": nn.Linear,
    "RowParallelLinear": RowParallelLinear,
    "ColumnParallelLinear": ColumnParallelLinear,
}
