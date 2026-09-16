import math
from abc import ABC, abstractmethod
import typing as tp

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from composer.utils import global_actlog_state
from composer.utils.dist import get_ep_group

from llmfoundry.models.layers.ffn import LlamaMLP, ParallelMLP
from llmfoundry.models.ops import grouped_gemm as gg
from llmfoundry.utils.misc import divide
from llmfoundry.models.parallel.tensor.utils import _initialize_ep_weight
from llmfoundry.models.layers.moe.sonic_cuda import require_sonic_moe_cuda_device
from llmfoundry.models.ops.float8.grouped_gemm import GroupedGemmFp8Wrapper
from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization import fused_swiglu_quantized_fn
from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices
from llmfoundry.models.ops.float8.triton_kernels.dequantization_deprecated import dequantize_kernel as fp8_dequantize
from llmfoundry.models.ops.float8.cuda_kernels.row2col_dq_quantization import warmup_ext as _warmup_row2col_dq_ext
from composer.utils.dist import get_ep_group_size
from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import unpad_and_unpermute_fn
from llmfoundry.models.ops.float8.triton_kernels.fixed_transformer_engine_permutation import (
    moe_unpermute as fused_unpermute,
)


# from sonicmoe.moe import MoE
try:
    from llmfoundry.models.layers.moe.sonicmoe.enums import ActivationType
    from llmfoundry.models.layers.moe.sonicmoe.functional import moe_general_routing_inputs
except:
    ActivationType = None
    moe_general_routing_inputs = None

# TODO: could this be made into something like / replaced by llmfoundry.models.ops.mlp.FusedGatedMLPFunc?
@torch.compile
def fused_swiglu(
    x: torch.Tensor, 
    probs: torch.Tensor,
    swiglu_limit: float = 0.0,
    clip_counts: tp.Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fused SwiGLU with optional activation clipping (DeepSeek-style).

    When ``swiglu_limit > 0``:
        gate = clamp(gate, -inf, swiglu_limit)
        up   = clamp(up,   -swiglu_limit, swiglu_limit)

    ``clip_counts`` (int32[2]) is mutated in-place when provided and
    ``swiglu_limit > 0``, accumulating [gate_clamped, up_clamped] counts.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    if swiglu_limit > 0:
        if clip_counts is not None:
            clip_counts[0].add_(gate.ge(swiglu_limit).sum().to(torch.int32))
            clip_counts[1].add_(up.le(-swiglu_limit).logical_or(up.ge(swiglu_limit)).sum().to(torch.int32))
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    x = F.silu(gate) * up
    res = x * probs
    return res.to(x.dtype)


# TODO: move this to utils
def _get_init_gain(packed_param: nn.Parameter,
                   logical_shape: tp.Iterable[int],
                   ep_size: int,
                   split_dim: int) -> float:
    logical_shape = torch.Size(logical_shape)
    packed_shape = packed_param.size()
    assert len(packed_shape) == 2
    assert len(logical_shape) == 2

    # row parallel
    if split_dim == 0:
        full_packed_shape = packed_shape[0] * ep_size, packed_shape[1]
    elif split_dim == 1:
        full_packed_shape = packed_shape[0], packed_shape[1] * ep_size
    else:
        raise ValueError(f"Invalid split_dim: {split_dim}")

    return math.sqrt(sum(full_packed_shape) / sum(logical_shape))


class GroupedAbstractMLP(nn.Module, ABC):
    """Abstract base for grouped MLP experts. Subclasses set fused_permute and implement forward."""

    def __init__(self, config, ep_size: int) -> None:
        super().__init__()
        self.config = config
        self.num_local_experts = divide(self.config.num_routed_experts, ep_size)

        self.tp_size = 1
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"

        self.ep_size = ep_size

        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        intermediate_size = self.intermediate_size

        fc1_output_size = intermediate_size * 2 * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, self.tp_size)
        fc2_input_size = intermediate_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, self.tp_size)

        self.fc1_output_size_per_partition = fc1_output_size_per_partition
        self.fc2_input_size_per_partition = fc2_input_size_per_partition

        self._use_new_expert_weight_layout = config.use_new_expert_weight_layout
        if self._use_new_expert_weight_layout:
            weight1_shape = (self.num_local_experts, intermediate_size * 2, self.config.hidden_size,)
            weight2_shape = (self.num_local_experts, self.config.hidden_size, intermediate_size,)
        else:
            weight1_shape = (self.hidden_size, self.fc1_output_size_per_partition,)
            weight2_shape = (self.fc2_input_size_per_partition, self.hidden_size,)

        self.weight1 = nn.Parameter(torch.empty(weight1_shape))
        self.weight2 = nn.Parameter(torch.empty(weight2_shape))

        # after gigafsdp wrap, it deletes the weight attributes from the module
        # so we need to store them here for log in repr
        self._weight1_shape = self.weight1.shape
        self._weight2_shape = self.weight2.shape

        self._use_float8_grouped_gemm = self.config.use_float8_grouped_gemm
        self._float8_sparse_fused_swiglu_quant = self.config.float8_sparse_fused_swiglu_quant
        self._float8_wgrad_backend = self.config.float8_wgrad_backend
        self._float8_triton_row2col = self.config.float8_triton_row2col
        self._swiglu_limit = getattr(self.config, "swiglu_limit", 0.0)

        # int32[2] buffers: [gate_clamped_count, up_clamped_count].
        # Filled by the forward kernel when ACTIVATION_MONITOR_ON=True.
        # persistent=False: excluded from state_dict (transient monitoring state).
        self.register_buffer('clip_counts', torch.zeros(2, dtype=torch.int32), persistent=False)

        if (
            self._use_float8_grouped_gemm
            and self._float8_wgrad_backend == "deep_gemm"
            and not self._float8_triton_row2col
        ):
            # Compile/load the CUDA extension eagerly on all ranks and
            # synchronise via barrier.  This prevents a 30-60 s JIT stall
            # inside the first backward pass (under FirstExpertGroupGate only
            # rank 0 has tokens, so it would block alone while other ranks
            # time out waiting for the DeepEP combine result).
            _warmup_row2col_dq_ext()

        _fp8_num_sms = getattr(config, 'float8_deep_gemm_num_sms', None)
        self.handle1 = GroupedGemmFp8Wrapper(self.num_local_experts, num_sms=_fp8_num_sms) if self._use_float8_grouped_gemm else None
        self.handle2 = GroupedGemmFp8Wrapper(self.num_local_experts, num_sms=_fp8_num_sms) if self._use_float8_grouped_gemm else None

        self._deep_ep_enabled = self.config.moe_enable_deepep

        self._init_params = dict(init_method=nn.init.xavier_normal_, use_master_weight=config.use_master_weight, init_type=config.init_type)
        self._init_rules()

    def _init_rules(self):
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        if self.ep_size > 1:
            def raise_init_error(*args, **kwargs):
                raise RuntimeError("This module must be inited by it's parent reset_parameters")

            self.weight1.skip_init = True
            self.weight1.reset_parameters = raise_init_error

            self.weight2.skip_init = True
            self.weight2.reset_parameters = raise_init_error

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(weight1_shape={self._weight1_shape}, weight2_shape={self._weight2_shape})"

    def _report_swiglu_clip_counts(self, fc1_output: torch.Tensor) -> None:
        """Report SwiGLU activation-clip counts to the global activation log.

        Shared between the FP8-quantised and plain (non-FP8) code paths.
        ``self.clip_counts`` must have been filled by the preceding
        ``fused_swiglu`` / ``fused_swiglu_quantized_fn`` call.
        """
        if not (global_actlog_state.enable_monitor and self._swiglu_limit > 0):
            return
        # Sum across EP group when ep_size > 1; every EP rank then holds the
        # layer-wide total.  With ep_size == 1 the per-rank counts are already
        # the full totals — no communication needed.
        if self.ep_size > 1:
            dist.all_reduce(self.clip_counts, op=dist.ReduceOp.SUM, group=get_ep_group())
        # Approximate global per-half-element total = (local elements) * ep_size.
        total = float(max((fc1_output.numel() // 2) * self.ep_size, 1))
        gate_share = self.clip_counts[0].float() / total
        up_share = self.clip_counts[1].float() / total
        global_actlog_state.use_monitor_variable(
            self.clip_counts[0].float().unsqueeze(0),
            "model.model.layers.{}.block_sparse_moe",
            "_gate_clip_count")
        global_actlog_state.use_monitor_variable(
            gate_share.unsqueeze(0),
            "model.model.layers.{}.block_sparse_moe",
            "_gate_clip_share")
        global_actlog_state.use_monitor_variable(
            self.clip_counts[1].float().unsqueeze(0),
            "model.model.layers.{}.block_sparse_moe",
            "_up_clip_count")
        global_actlog_state.use_monitor_variable(
            up_share.unsqueeze(0),
            "model.model.layers.{}.block_sparse_moe",
            "_up_clip_share")

    def reset_parameters_giga(self):
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        if self._use_new_expert_weight_layout:
            packed_w1 = self.weight1.new_empty(hidden_size, self.fc1_output_size_per_partition)
            packed_w2 = self.weight2.new_empty(self.fc2_input_size_per_partition, hidden_size)
        else:
            packed_w1 = self.weight1
            packed_w2 = self.weight2
        weight1_gain = _get_init_gain(
            packed_w1,
            (hidden_size, intermediate_size),
            ep_size=self.ep_size,
            split_dim=1
        )
        weight2_gain = _get_init_gain(
            packed_w2,
            (intermediate_size, hidden_size),
            ep_size=self.ep_size,
            split_dim=0
        )

        if self.ep_size > 1:
            self._init_params["device"] = self.weight1.device
            assert torch.device('meta') != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            if self._use_new_expert_weight_layout:
                w1 = torch.empty(
                    self.config.hidden_size, self.fc1_output_size_per_partition,
                    device=self._init_params["device"], dtype=self.weight1.dtype,
                )
            else:
                w1 = self.weight1

            _initialize_ep_weight(
                w1,
                input_size=self.fc1_output_size_per_partition * self.ep_size,
                output_size=self.hidden_size,
                partition_dim=1,
                per_partition_size=self.fc1_output_size_per_partition,
                gain=weight1_gain,
                return_master_weight=False,
                **self._init_params
            )
            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight1.copy_(
                        w1.view(self.config.hidden_size, self.num_local_experts, -1).permute(1, 2, 0)
                    )

            if self._use_new_expert_weight_layout:
                w2 = torch.empty(
                    self.fc2_input_size_per_partition, self.config.hidden_size,
                    device=self._init_params["device"], dtype=self.weight2.dtype,
                )
            else:
                w2 = self.weight2

            _initialize_ep_weight(
                w2,
                input_size=self.hidden_size,
                output_size=self.fc2_input_size_per_partition * self.ep_size,
                partition_dim=0,
                per_partition_size=self.fc2_input_size_per_partition,
                gain=weight2_gain,
                return_master_weight=False,
                **self._init_params
            )
            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight2.copy_(
                        w2.view(self.num_local_experts, -1, self.config.hidden_size).permute(0, 2, 1)
                    )

        else:
            self._init_params["init_method"](self.weight1, gain=weight1_gain)
            self._init_params["init_method"](self.weight2, gain=weight2_gain)

    def reset_parameters_deepseek(self):
        # Same as Giga init, but without `gain` parameter for normal init func.
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        if self.ep_size > 1:
            self._init_params["device"] = self.weight1.device
            assert torch.device('meta') != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            if self._use_new_expert_weight_layout:
                w1 = torch.empty(
                    self.config.hidden_size, self.fc1_output_size_per_partition,
                    device=self._init_params["device"], dtype=self.weight1.dtype,
                )
            else:
                w1 = self.weight1

            _initialize_ep_weight(
                w1,
                input_size=self.fc1_output_size_per_partition * self.ep_size,
                output_size=self.hidden_size,
                partition_dim=1,
                per_partition_size=self.fc1_output_size_per_partition,
                return_master_weight=False,
                **self._init_params
            )

            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight1.copy_(
                        w1.view(self.config.hidden_size, self.num_local_experts, -1).permute(1, 2, 0)
                    )

            if self._use_new_expert_weight_layout:
                w2 = torch.empty(
                    self.fc2_input_size_per_partition, self.config.hidden_size,
                    device=self._init_params["device"], dtype=self.weight2.dtype,
                )
            else:
                w2 = self.weight2

            _initialize_ep_weight(
                w2,
                input_size=self.hidden_size,
                output_size=self.fc2_input_size_per_partition * self.ep_size,
                partition_dim=0,
                per_partition_size=self.fc2_input_size_per_partition,
                return_master_weight=False,
                **self._init_params
            )

            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight2.copy_(
                        w2.view(self.num_local_experts, -1, self.config.hidden_size).permute(0, 2, 1)
                    )

        else:
            self._init_params["init_method"](self.weight1)
            self._init_params["init_method"](self.weight2)

    def reset_parameters(self):
        # TODO(m1kol): Combine these init functions, make them more universal.
        # Now some params assume xavier init (gain).
        if self.config.init_type == "giga":
            self.reset_parameters_giga()
        elif self.config.init_type == "deepseek":
            self.reset_parameters_deepseek()
        else:
            raise ValueError(f'Expected `init_type` to be one of "giga" or "deepseek", got {self.config.init_type}.')


class GroupedLlamaMLP(GroupedAbstractMLP):
    def __init__(self, config, ep_size: int) -> None:
        self.fused_permute = False
        super().__init__(config, ep_size)

    def forward(self,
                permuted_local_hidden_states: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                tokens_per_expert_cpu: torch.Tensor,
                tokens_per_expert_list: list[int],
                permuted_probs: torch.Tensor,
                weights_quantized: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:

        if permuted_local_hidden_states.nelement() == 0:
            assert torch.count_nonzero(tokens_per_expert_cpu) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = fused_swiglu(h, permuted_probs.unsqueeze(-1))
            return torch.matmul(h, w2)

        
        monitor_clips = global_actlog_state.enable_monitor and self._swiglu_limit > 0
        if monitor_clips:
            self.clip_counts.zero_()

        if self.training and self._use_float8_grouped_gemm and (self.ep_size == 1 or self._deep_ep_enabled):
            assert self._use_new_expert_weight_layout, "Old expert weight layout is not supported for float8 training."
            w1, w2 = self.weight1, self.weight2

            # TODO (fedorovgv): remove duplicate here! we can hold it in memory
            groups_padded_sizes_list = [(m + 127) // 128 * 128 for m in tokens_per_expert_list]
            groups_padded_sizes_tensor = torch.tensor(
                groups_padded_sizes_list,
                dtype=torch.int32,
                device=permuted_local_hidden_states.device)
            m_indices = build_m_indices(groups_padded_sizes_list, groups_padded_sizes_tensor)

            global_actlog_state.use_monitor_variable(
                permuted_local_hidden_states,
                "model.model.layers.{}.block_sparse_moe.gmm_upgate",
                "_input.0")
            assert self.handle1 is not None, "GroupedGemmFp8Wrapper is not available!"
            fc1_output = self.handle1.group_gemm(
                tensor_groups=permuted_local_hidden_states,
                weights=w1,
                weights_quantized=weights_quantized[0] if weights_quantized is not None else None,
                wgrad_backend=self._float8_wgrad_backend,
                tensor_groups_sizes=groups_padded_sizes_list,
                tensor_groups_sizes_padded=groups_padded_sizes_tensor,
                m_indices=m_indices,
                triton_row2col=self._float8_triton_row2col,
            )
            global_actlog_state.use_monitor_variable(
                fc1_output,
                "model.model.layers.{}.block_sparse_moe.fused_swiglu",
                "_input.0")

            if self._float8_sparse_fused_swiglu_quant:
                intermediate = fused_swiglu_quantized_fn(
                    fc1_output, permuted_probs,
                    swiglu_limit=self._swiglu_limit,
                    clip_counts=self.clip_counts if monitor_clips else None,
                )
            else:
                permuted_probs = permuted_probs.unsqueeze(-1)
                intermediate = fused_swiglu(
                    fc1_output, permuted_probs,
                    swiglu_limit=self._swiglu_limit,
                    clip_counts=self.clip_counts if monitor_clips else None,
                )

            self._report_swiglu_clip_counts(fc1_output)

            if global_actlog_state.enable_monitor:
                global_actlog_state.use_monitor_variable(
                    intermediate if not self._float8_sparse_fused_swiglu_quant else \
                        fp8_dequantize(intermediate._rowwise_data, intermediate._rowwise_scale_inv),
                    "model.model.layers.{}.block_sparse_moe.gmm_down",
                    "_input.0")

            assert self.handle2 is not None, "GroupedGemmFp8Wrapper is not available!"
            fc2_output = self.handle2.group_gemm(
                tensor_groups=intermediate,
                weights=w2,
                weights_quantized=weights_quantized[1] if weights_quantized is not None else None,
                wgrad_backend=self._float8_wgrad_backend,
                tensor_groups_sizes=groups_padded_sizes_list,
                tensor_groups_sizes_padded=groups_padded_sizes_tensor,
                m_indices=m_indices,
                triton_row2col=self._float8_triton_row2col,
            )
        else:
            if self._use_new_expert_weight_layout:
                w1 = self.weight1.view(self.num_local_experts, -1, self.config.hidden_size)
                w2 = self.weight2.view(self.num_local_experts, self.config.hidden_size, -1)
                trans_b = True
            else:
                w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
                w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
                trans_b = False

            global_actlog_state.use_monitor_variable(
                permuted_local_hidden_states,
                "model.model.layers.{}.block_sparse_moe.gmm_upgate",
                "_input.0")
            fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert_cpu, trans_b=trans_b)
            global_actlog_state.use_monitor_variable(
                fc1_output,
                "model.model.layers.{}.block_sparse_moe.fused_swiglu",
                "_input.0")

            intermediate = fused_swiglu(
                fc1_output, permuted_probs.unsqueeze(-1),
                swiglu_limit=self._swiglu_limit,
                clip_counts=self.clip_counts if monitor_clips else None,
            )

            self._report_swiglu_clip_counts(fc1_output)

            global_actlog_state.use_monitor_variable(
                intermediate,
                "model.model.layers.{}.block_sparse_moe.gmm_down",
                "_input.0")
            fc2_output = gg.ops.gmm(intermediate, w2, tokens_per_expert_cpu, trans_b=trans_b)

        global_actlog_state.use_monitor_variable(
            fc2_output,
            "model.model.layers.{}.block_sparse_moe.gmm_down",
            "_output.0")

        return fc2_output


if moe_general_routing_inputs is None or ActivationType is None:
    class GroupedSonicMLP(GroupedAbstractMLP):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GroupedSonicMLP requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "GroupedSonicMLP requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )
else:
    class GroupedSonicMLP(GroupedAbstractMLP):
        def __init__(self, config, ep_size: int) -> None:
            require_sonic_moe_cuda_device()
            assert not config.use_float8_grouped_gemm, (
                "GroupedSonicMLP does not support float8 grouped GEMM. "
                "Float8 is only supported for dispatch via DeepEP, not for SonicMoE experts."
            )
            assert not config.use_new_expert_weight_layout, (
                "GroupedSonicMLP does not support use_new_expert_weight_layout."
            )
            self.fused_permute = True
            self.intermediate_size = config.intermediate_size
            super().__init__(config, ep_size)
            self.activation_function = ActivationType.SWIGLU
            self.stream_id = torch.cuda.current_stream().cuda_stream

        def forward(
            self,
            local_hidden_states: torch.Tensor,
            token_idxs: torch.Tensor,
            expert_idxs: torch.Tensor,
            expert_scores: torch.Tensor,
        ) -> torch.Tensor:

            # weight1: (H, 2*I*E) -> view(E, 2*I, H) -> permute(1,2,0) -> (2*I, H, E)
            #   strides: (2*I*H, H, 1)  -> (H, 1, 2*I*H) -- strides[1]=1 ✓
            # weight2: (I*E, H)  -> view(E, H, I)   -> permute(1,2,0) -> (H, I, E)
            #   strides: (H*I, I, 1)    -> (I, 1, H*I)   -- strides[1]=1 ✓
            w1 = self.weight1.view(
                self.num_local_experts, 2 * self.intermediate_size, self.hidden_size
            ).permute(1, 2, 0)
            w2 = self.weight2.view(
                self.num_local_experts, self.hidden_size, self.intermediate_size
            ).permute(1, 2, 0)

            output_hidden_states, _ = moe_general_routing_inputs(
                x=local_hidden_states,
                router_scores=expert_scores.view(-1),
                token_indices=token_idxs.view(-1),
                expert_indices=expert_idxs.view(-1),
                w1=w1,
                b1=None,
                w2=w2,
                b2=None,
                E=self.num_local_experts,
                stream_id=self.stream_id,
                activation_type=self.activation_function,
                is_inference_mode_enabled=False
            )
            return output_hidden_states



# shared expert computation can be overlapped with MoE dispatch
# a separate class facilitates this
# inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py
class SharedExpertLlamaMLP(LlamaMLP):
    def __init__(self, config) -> None:
        ffn_dim = config.intermediate_size * config.n_shared_experts
        super().__init__(
            config,
            intermediate_size=ffn_dim,
            fused_mlp_checkpoint_lvl=config.fused_mlp_checkpoint_lvl,
        )


class SharedExpertParallelLlamaMLP(ParallelMLP):
    def __init__(self, config) -> None:
        ffn_dim = config.intermediate_size * config.n_shared_experts
        super().__init__(
            config,
            intermediate_size=ffn_dim,
            fused_mlp_checkpoint_lvl=config.fused_mlp_checkpoint_lvl,
        )

class ScMoEBlockGeMM(nn.Module):
    def __init__(self, config, token_dispatcher) -> None:
        super().__init__()

        self.config = config

        ep_size = get_ep_group_size()
        if ep_size is None:
            ep_size = 1

        self.experts = GroupedLlamaMLP(config, ep_size)
        self.token_dispatcher = token_dispatcher

    def forward(
        self,
        recv_x,
        dispatched_indices,
        dispatched_probs,
        tokens_per_expert,
        ctx,
        is_float8_training: bool = False,
        weights_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        (
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
        ) = self.token_dispatcher.post_dispatch_scmoe(
            recv_x,
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
            ctx,
            is_float8_training=is_float8_training,
        )
        expert_output = self.experts(
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
            weights_quantized,
        )
        if is_float8_training:
            recv_order_hidden = unpad_and_unpermute_fn(
                expert_output,
                ctx.deepep_reversed_mapping_for_combine,
                ctx.deepep_reversed_mapping_for_combine_padded,
                ctx.deepep_recv_hidden_shape,
                ctx.deepep_pad_starts,
                ctx.deepep_pad_counts,
            )
        else:
            recv_order_hidden = fused_unpermute(
                expert_output,
                ctx.deepep_reversed_mapping_for_combine,
                merging_probs=None,
                restore_shape=ctx.deepep_recv_hidden_shape,
            )
        return recv_order_hidden
