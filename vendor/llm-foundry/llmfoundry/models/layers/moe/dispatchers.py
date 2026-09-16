import typing as tp
from dataclasses import dataclass

import torch

from llmfoundry.models.ops.float8.triton_kernels.fixed_transformer_engine_permutation import (
    moe_permute_with_probs as fused_permute_with_probs,
    moe_unpermute as fused_unpermute,
    moe_sort_chunks_by_index_with_probs as fused_sort_chunks_by_index_with_probs,
    moe_sort_chunks_by_index as fused_sort_chunks_by_index,
)

from composer.utils import dist
from composer.utils.profiler_annotation import profiler_annotation

from composer.utils.dist import (
    get_ep_group, get_ep_group_rank, get_ep_group_size, all_reduce
)
from llmfoundry.utils.misc import divide
from llmfoundry.models.parallel.tensor import gather_from_tensor_model_parallel_region
from llmfoundry.models.parallel.tensor.mappings import all_to_all

try:
    from llmfoundry.models.layers.moe.fused_a2a import fused_dispatch, fused_combine, set_deepep_num_sms, scmoe_layer
    _DEEPEP_AVAILABLE = fused_dispatch is not None

except ImportError as e:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None
    scmoe_layer = None
    _DEEPEP_AVAILABLE = False

from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
    permute_and_pad_fn, unpad_and_unpermute_fn
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor


@dataclass
class DispatcherContext:
    """Context needed for unpermuting tokens"""
    num_local_tokens_per_expert: torch.Tensor | None = None
    input_splits: torch.Tensor | None = None
    output_splits: torch.Tensor | None = None
    reversed_local_input_permutation_mapping: int | None = None
    reversed_local_input_permutation_mapping_padded: int | None = None
    hidden_shape_before_permute: torch.Size | None = None
    chunk_size_per_local_expert: torch.Tensor | None = None

    # padding info for torch.empty optimisation in unpad_and_unpermute_fn backward
    reversed_local_input_permutation_pad_starts: torch.Tensor | None = None
    reversed_local_input_permutation_pad_counts: torch.Tensor | None = None

    # DeepEP
    deepep_handle: torch.Tensor | None = None
    deepep_recv_hidden_shape: torch.Size | None = None
    deepep_reversed_mapping_for_combine: torch.Tensor | None = None
    deepep_reversed_mapping_for_combine_padded: torch.Tensor | None = None
    deepep_pad_starts: torch.Tensor | None = None
    deepep_pad_counts: torch.Tensor | None = None


class TokenDispatcher:

    def __init__(self, config):
        self.config = config
        self.ep_size = get_ep_group_size() or 1
        self.num_local_experts = divide(config.num_routed_experts, self.ep_size)
        self.dispatcher_balancing_strategy = config.dispatcher_balancing_strategy
        self.expert_capacity_factor = config.expert_capacity_factor if hasattr(config, 'expert_capacity_factor') else None
        self.expert_capacity_threshold = config.expert_capacity_tokens if config.expert_capacity_tokens else None
        self.drop_by_group_capacity = config.drop_by_group_capacity
        self._disbalance_flag = False
        self.max_capacity = 1e10

        if self.expert_capacity_factor and (not self.dispatcher_balancing_strategy or self.dispatcher_balancing_strategy not in ["capacity_factor", "peak_capacity_factor"]):
            raise AttributeError(f"to use 'expert_capacity_factor' choose dispatcher_balancing_strategy='capacity_factor' or 'peak_capacity_factor' ")

        if self.expert_capacity_threshold and (not self.dispatcher_balancing_strategy or self.dispatcher_balancing_strategy != "fixed_capacity"):
            raise AttributeError(f"to use 'expert_capacity_threshold' choose dispatcher_balancing_strategy='fixed_capacity' ")

        if self.dispatcher_balancing_strategy and self.dispatcher_balancing_strategy == "fixed_capacity" and not self.expert_capacity_threshold:
            raise AttributeError(f"set 'expert_capacity_threshold' to usedispatcher_balancing_strategy='fixed_capacity' ")

        if  self.dispatcher_balancing_strategy and self.dispatcher_balancing_strategy in ["capacity_factor", "peak_capacity_factor"] and not self.expert_capacity_factor:
            raise AttributeError(f"set 'expert_capacity_factor' to use dispatcher_balancing_strategy='capacity_factor' or 'peak_capacity_factor' ")

        if self.dispatcher_balancing_strategy == "peak_capacity_factor":
            self.expert_capacity_threshold = self.init_capacity_treshold_from_peak_seqlen()

        # permute related ixes
        self.num_experts = self.config.num_routed_experts
        self.tp_size = 1
        self.num_experts_per_tok = self.config.num_experts_per_tok

        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=torch.cuda.current_device()
        )
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        self.init_monitor_done = False
        self._active_dispatcher_monitor = False

        # ---------- DeepEP -----------
        self.enable_deepep = config.moe_enable_deepep
        self.deepep_num_sms = config.moe_deepep_num_sms
        if self.enable_deepep and _DEEPEP_AVAILABLE and self.deepep_num_sms is not None:
            set_deepep_num_sms(int(self.deepep_num_sms))

    def _activate_dispatcher_monitor(self):
        self._active_dispatcher_monitor = True

    def _deactivate_dispatcher_monitor(self):
        self._active_dispatcher_monitor = False


    def _init_dispatcher_monitor(self, batch_log_interval:int=10, microbatch_num:int=1):
        self._local_expert_capacity = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._tokens_per_local_expert = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )

        self._tokens_per_local_expert_no_pad = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._tokens_per_local_expert_no_pad_no_prefix = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )

    def _reset_dispatcher_monitor(self):
        self._local_expert_capacity = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._tokens_per_local_expert = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._tokens_per_local_expert_no_pad = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._tokens_per_local_expert_no_pad_no_prefix = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            requires_grad=False,
            device=f"cuda:{torch.cuda.current_device()}",
        )

    def _update_dispatcher_monitor(
        self,
        local_expert_capacity_threshold: int,
        tokens_per_local_expert: torch.Tensor,
        tokens_per_local_expert_no_pad: torch.Tensor,
        tokens_per_local_expert_no_pad_no_prefix: torch.Tensor,
    ):

        self._local_expert_capacity = torch.zeros(
                self.num_experts,
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            ) + local_expert_capacity_threshold
        self._tokens_per_local_expert = torch.zeros(
                self.num_experts,
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            ) + tokens_per_local_expert
        self._tokens_per_local_expert_no_pad = torch.zeros(
                self.num_experts,
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            ) + tokens_per_local_expert_no_pad
        self._tokens_per_local_expert_no_pad_no_prefix = torch.zeros(
                self.num_experts,
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            ) + tokens_per_local_expert_no_pad_no_prefix

    def get_dispatcher_monitor(self):
        return (
            self._local_expert_capacity,
            self._tokens_per_local_expert,
            self._tokens_per_local_expert_no_pad,
            self._tokens_per_local_expert_no_pad_no_prefix
        )

    def init_capacity_treshold_from_peak_seqlen(self):
        max_seq_len = self.config.max_position_embeddings
        microbatch_size = self.config.peak_capacity_microbatchsize

        tp_size = dist.get_tp_group_size() or 1
        ep_size = dist.get_ep_group_size() or 1
        sp_size = dist.get_sp_group_size() or 1

        num_experts_per_token = self.config.num_experts_per_tok
        experts_num = self.config.num_routed_experts

        capacity_thresold = int(max_seq_len * microbatch_size * num_experts_per_token * ep_size * self.expert_capacity_factor / (tp_size * sp_size * experts_num))

        return capacity_thresold

    def process(self, routing_map: torch.Tensor, ctx: DispatcherContext) -> torch.Tensor:
        num_tokens_per_local_expert = routing_map.sum(0).long()

        if self.ep_size > 1:
            ctx.input_splits = num_tokens_per_local_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1).cpu().tolist()

            # ep_size x ep_size x num_local_experts
            global_num_tokens_per_expert = gather_from_tensor_model_parallel_region(
                num_tokens_per_local_expert, get_ep_group()
            ).view(self.ep_size, self.ep_size, self.num_local_experts)

            ep_rank = get_ep_group_rank()
            ctx.output_splits = global_num_tokens_per_expert.sum(dim=-1)[:, ep_rank].cpu().tolist()

            if self.num_local_experts > 1:
                ctx.chunk_size_per_local_expert = (
                    global_num_tokens_per_expert
                    .transpose(0, 1)
                    [ep_rank]
                    .view(-1, self.num_local_experts)
                )
            num_tokens_per_local_expert = global_num_tokens_per_expert[:, ep_rank].sum(dim=0)

        num_tokens_per_local_expert = num_tokens_per_local_expert.cpu()
        return num_tokens_per_local_expert

    def _get_expert_capacity_threshold(self, num_tokens: int):
        if self.dispatcher_balancing_strategy in set(["fixed_capacity", "peak_capacity_factor"]):
            # calc expert capacity for 1 ep rank from sum capacity from n ep ranks
            expert_capacity_threshold = self.expert_capacity_threshold // self.ep_size
        elif self.dispatcher_balancing_strategy == "capacity_factor":
            avg_capacity = num_tokens // self.num_experts
            expert_capacity_threshold = int(avg_capacity * self.expert_capacity_factor)
        elif self.dispatcher_balancing_strategy == None:
            expert_capacity_threshold = self.max_capacity
        else:
            AssertionError(f"Unknown dispatcher balancing stratege. Please, choose from set ['fixed_capacity','peak_capacity_factor', 'capacity_factor', None] ")

        return expert_capacity_threshold

    def _drop_token_by_capacity_factor(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: tp.Optional[torch.Tensor] = None,
        prefix_mask: tp.Optional[torch.Tensor] = None,
    ):
        # num_tokens on on 1 ep rank
        num_tokens = routing_map.shape[0] * self.num_experts_per_tok

        # expert capacity inside ep group of input tokens
        local_expert_capacity_threshold = self._get_expert_capacity_threshold(
                num_tokens=num_tokens
        )

        tokens_per_local_expert = routing_map.sum(axis=0)
        overloaded_local_experts = tokens_per_local_expert > local_expert_capacity_threshold

        num_experts = tokens_per_local_expert.shape[-1]

        # NOTE: drop paddings and prefixies only for monitorings
        # --------------------------------------------------------------------
        routing_map_no_pad, probs_no_pad = self._drop_pad_tokens(routing_map, probs, padding_mask)
        routing_map_no_pad_no_prefix, probs_no_pad_no_prefix = self._drop_pad_tokens(routing_map_no_pad, probs_no_pad, prefix_mask)
        tokens_per_local_expert_no_pad = routing_map_no_pad.sum(axis=0)
        tokens_per_local_expert_no_pad_no_prefix = routing_map_no_pad_no_prefix.sum(axis=0)
        # --------------------------------------------------------------------

        if self._active_dispatcher_monitor:
            with torch.no_grad():
                self._update_dispatcher_monitor(
                    local_expert_capacity_threshold=local_expert_capacity_threshold, # sum capacity of all experts
                    tokens_per_local_expert=tokens_per_local_expert,
                    tokens_per_local_expert_no_pad=tokens_per_local_expert_no_pad,
                    tokens_per_local_expert_no_pad_no_prefix=tokens_per_local_expert_no_pad_no_prefix,
                )

        # add extra communication operation
        # reduce token drop usage frequency
        if self.drop_by_group_capacity:
            global_tokens_per_expert = tokens_per_local_expert.clone()
            ep_group = get_ep_group()
            all_reduce(
                tensor=global_tokens_per_expert,
                reduce_operation = 'SUM',
                group=ep_group
            )
            global_tokens_per_expert_group = global_tokens_per_expert.view((self.ep_size, -1)).sum(1)

            # сравниваем с capacity одной группы
            # multiply by ep_size, as global_tokens_per_expert_group is aggregated from all ep ranks
            group_expert_capacity_threshold = local_expert_capacity_threshold * self.ep_size
            group_capacity = group_expert_capacity_threshold * (num_experts // self.ep_size)
            overloaded_global_expert_groups = global_tokens_per_expert_group > group_capacity
            if not overloaded_global_expert_groups.any():
                return routing_map, probs

        if overloaded_local_experts.any():
            self._disbalance_flag = True
            sorted_indices = probs.argsort(dim=0, descending=True, stable=True)[:local_expert_capacity_threshold, :]

            mask = torch.zeros_like(
                routing_map,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}"
            )
            mask = mask.scatter_(0, sorted_indices, 1)

            routing_map = routing_map*mask
            probs = probs*mask

        return routing_map, probs

    def _balance_tokens(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: tp.Optional[torch.Tensor] = None,
        prefix_mask: tp.Optional[torch.Tensor] = None,
    ):

        if self.dispatcher_balancing_strategy in ["peak_capacity_factor", "capacity_factor", "fixed_capacity", None]:
            routing_map, probs = self._drop_token_by_capacity_factor(
                probs=probs,
                routing_map=routing_map,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
            )
        else:
            raise ValueError(f'Unknown dispatcher_balancing_strategу: {self.dispatcher_balancing_strategy}')

        return routing_map, probs

    # TODO : move padding drop to routers file
    # fill zero all padding tokens in routing map and probs

    def _drop_pad_tokens(self, routing_map, probs, padding_mask):
        if padding_mask is None:
            return routing_map, probs

        padding_mask_flat = padding_mask.flatten().unsqueeze(1)
        assert routing_map.shape[0] == padding_mask_flat.shape[0], f"padding mask should be (num_tok, 1) shape {routing_map.shape[0]}"

        routing_map = routing_map * padding_mask_flat
        probs = probs * padding_mask_flat

        return routing_map, probs

    # ----------------------------------------------------------------
    # Decomposed building blocks (from sonic-moe refactor)
    # ----------------------------------------------------------------

    def _preprocess_context(self, hidden_states: torch.Tensor) -> DispatcherContext:
        """Prepares context before token distribution."""
        ctx = DispatcherContext()
        assert len(hidden_states.size()) == 2, f"Please reshape inputs to [-1, hidden_size] before applying permutations. Curr {hidden_states.shape = }"
        ctx.hidden_shape_before_permute = hidden_states.size()
        return ctx

    def _postprocess_context(self, hidden_states: torch.Tensor, ctx: DispatcherContext) -> torch.Tensor:
        """Restores tensor shape from context after combine/unpermute."""
        if ctx.hidden_shape_before_permute is not None:
            return hidden_states.view(ctx.hidden_shape_before_permute)
        return hidden_states

    def _permute(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        ctx: DispatcherContext,
        is_float8_training: bool = False,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatcherContext]:
        """Permutes tokens only within one ep-rank (no all2all)."""
        tokens_per_local_expert = self.process(routing_map, ctx)

        if is_float8_training:
            (
                permuted_hidden_states,
                permuted_probs,
                ctx.reversed_local_input_permutation_mapping,
                ctx.reversed_local_input_permutation_mapping_padded,
                ctx.reversed_local_input_permutation_pad_starts,
                ctx.reversed_local_input_permutation_pad_counts,
            ) = permute_and_pad_fn(
                hidden_states, routing_map, probs, routing_map.sum(0)
            )
        else:
            num_out_tokens = routing_map.sum()
            (
                permuted_hidden_states,
                permuted_probs,
                ctx.reversed_local_input_permutation_mapping,
            ) = fused_permute_with_probs(
                hidden_states, probs, routing_map, num_out_tokens
            )

        return permuted_hidden_states, tokens_per_local_expert, permuted_probs, ctx

    def _unpermute(
        self,
        hidden_states: torch.Tensor,
        ctx: DispatcherContext,
        is_float8_training: bool = False,
    ) -> torch.Tensor:
        """Unpermutes tokens only within one ep-rank (no all2all)."""
        if is_float8_training:
            assert ctx.hidden_shape_before_permute is not None
            return unpad_and_unpermute_fn(
                hidden_states,
                ctx.reversed_local_input_permutation_mapping,
                ctx.reversed_local_input_permutation_mapping_padded,
                ctx.hidden_shape_before_permute,
                ctx.reversed_local_input_permutation_pad_starts,
                ctx.reversed_local_input_permutation_pad_counts,
            )
        else:
            return fused_unpermute(
                hidden_states,
                ctx.reversed_local_input_permutation_mapping,
                merging_probs=None,
                restore_shape=ctx.hidden_shape_before_permute,
            )

    def _default_dispatch(
        self,
        permuted_hidden_states: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_local_expert: torch.Tensor,
        ctx: DispatcherContext,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatcherContext]:
        """Dispatch via all2all (after local permute)."""
        if self.ep_size > 1:
            permuted_hidden_states = all_to_all(
                get_ep_group(), permuted_hidden_states, ctx.output_splits, ctx.input_splits
            )
            permuted_probs = all_to_all(
                get_ep_group(), permuted_probs, ctx.output_splits, ctx.input_splits
            )
            if self.num_local_experts > 1:
                permuted_hidden_states, permuted_probs = sort_chunks_by_idxs(
                    permuted_hidden_states,
                    split_sizes=ctx.chunk_size_per_local_expert.ravel(),
                    sorted_idxs=self.sort_input_by_local_experts,
                    probs=permuted_probs,
                )
        return permuted_hidden_states, tokens_per_local_expert, permuted_probs, ctx

    def _deepep_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        ctx: DispatcherContext,
        is_float8_training: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatcherContext,
                   torch.Tensor, tp.List[int], tp.Optional[tp.Tuple]]:
        """Dispatch using DeepEP (fused_dispatch).

        Returns (recv_x, local_map, local_probs, ctx,
                 tokens_per_expert_cpu, tokens_per_expert_list, weights_quantized).
        """
        if self.ep_size == 1:
            return hidden_states, routing_map, probs, ctx, None, None, None
        assert self.enable_deepep and _DEEPEP_AVAILABLE, "DeepEP dispatch requires enable_deepep and fused_dispatch"

        num_local_tokens = routing_map.shape[0]
        rm_flat = routing_map.view(num_local_tokens, -1)
        pr_flat = probs.view(num_local_tokens, -1)
        token_probs, token_indices = torch.topk(pr_flat, k=self.num_experts_per_tok, dim=-1)
        token_probs = token_probs.float()
        if self.dispatcher_balancing_strategy:
            # TODO: rewrite without extra allocations
            not_selected_token_mask = rm_flat.gather(dim=-1, index=token_indices) == 0
            token_indices = token_indices.masked_fill(not_selected_token_mask, -1)

        assert fused_dispatch is not None, "DeepEP is not available!"
        (
            recv_x,
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert_cpu,
            tokens_per_expert_list,
            handle,
            weights_quantized,
        ) = fused_dispatch(
            x=hidden_states if isinstance(hidden_states, Float8BlockwiseQTensor) else hidden_states.contiguous(),
            token_indices=token_indices.contiguous(),
            token_probs=token_probs.contiguous(),
            num_experts=self.num_experts,
            group=get_ep_group(),
            is_float8_dispatch=is_float8_training,
            weight_quantizer_handles=weight_quantizer_handles,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        ctx.deepep_handle = handle
        ctx.deepep_recv_hidden_shape = recv_x.shape
        batch_size = dispatched_indices.shape[0]
        device = dispatched_indices.device
        safe_indices = torch.where(
            dispatched_indices == -1,
            torch.tensor(self.num_local_experts, device=device, dtype=dispatched_indices.dtype),
            dispatched_indices,
        )
        padded_map = torch.zeros(
            (batch_size, self.num_local_experts + 1),
            dtype=torch.bool,
            device=device,
        )
        padded_probs = torch.zeros(
            (batch_size, self.num_local_experts + 1),
            dtype=torch.float,
            device=device,
        )
        padded_map.scatter_(dim=1, index=safe_indices, value=True)
        padded_probs.scatter_(dim=1, index=safe_indices, src=dispatched_probs.float())
        local_map = padded_map[:, :-1]
        local_probs = padded_probs[:, :-1]

        assert int(local_map.sum().item()) == int(tokens_per_expert_cpu.sum().item()), (
            "DeepEP: mismatch between local multihot and tokens_per_expert"
        )

        return recv_x, local_map, local_probs, ctx, tokens_per_expert_cpu, tokens_per_expert_list, weights_quantized

    def _deepep_permute_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        ctx: DispatcherContext,
        is_float8_training: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor], torch.Tensor,
                   tp.List[int], torch.Tensor, DispatcherContext, tp.Optional[tp.Tuple]]:
        """DeepEP dispatch + local permute (for ggemm path with ep>1 during training).

        Returns (permuted_hidden_states, tokens_per_expert_gpu, tokens_per_expert_cpu,
                 tokens_per_expert_list, permuted_probs, ctx, weights_quantized).
        """
        # Step 1: DeepEP dispatch (all-to-all + expert assignment)
        (
            recv_x, local_map, local_probs, ctx,
            tokens_per_expert_cpu, tokens_per_expert_list, weights_quantized,
        ) = self._deepep_dispatch(
            hidden_states, probs, routing_map, ctx,
            is_float8_training=is_float8_training,
            weight_quantizer_handles=weight_quantizer_handles,
        )

        # Step 2: Local permute on received data (group tokens by expert)
        tokens_per_expert_gpu = tokens_per_expert_cpu.to(recv_x.device)

        if not is_float8_training:
            num_out_tokens = tokens_per_expert_cpu.sum().item()
            permuted_hidden_states, permuted_probs, reversed_mapping = fused_permute_with_probs(
                recv_x, local_probs, local_map, num_out_tokens
            )
            ctx.deepep_reversed_mapping_for_combine = reversed_mapping
        else:
            (
                permuted_hidden_states,
                permuted_probs,
                ctx.deepep_reversed_mapping_for_combine,
                ctx.deepep_reversed_mapping_for_combine_padded,
                ctx.deepep_pad_starts,
                ctx.deepep_pad_counts,
            ) = permute_and_pad_fn(
                recv_x=recv_x,
                local_map=local_map,
                probs=local_probs,
                tokens_per_expert=tokens_per_expert_gpu,
            )

        return (permuted_hidden_states, tokens_per_expert_gpu, tokens_per_expert_cpu,
                tokens_per_expert_list, permuted_probs, ctx, weights_quantized)

    def _default_combine(
        self,
        hidden_states: torch.Tensor,
        ctx: DispatcherContext,
    ) -> torch.Tensor:
        """Combine after default dispatch (all2all reverse + sort)."""
        if self.ep_size > 1:
            if self.num_local_experts > 1:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    ctx.chunk_size_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                )
            hidden_states = all_to_all(
                get_ep_group(), hidden_states, ctx.input_splits, ctx.output_splits
            )
        return hidden_states

    def _deepep_combine(
        self,
        hidden_states: torch.Tensor,
        ctx: DispatcherContext,
        is_float8_training: bool = False,
    ) -> torch.Tensor:
        """Combine after dispatch with DeepEP (fused_combine). Atomic implementation."""
        if self.ep_size == 1:
            return hidden_states
        assert self.enable_deepep and _DEEPEP_AVAILABLE, "we can't combine tokens without permute with regular deepep"
        assert ctx.deepep_handle is not None, "DeepEP: missing handle for fused_combine"
        assert ctx.deepep_recv_hidden_shape is not None, "DeepEP: missing recv shape"
        assert fused_combine is not None, "DeepEP: fused_combine is not available!"
        combined_x, _ = fused_combine(
            hidden_states,
            get_ep_group(),
            ctx.deepep_handle,
            async_finish=True,
            allocate_on_comm_stream=True,
            is_float8_training=is_float8_training,
        )
        ctx.deepep_handle = None
        return combined_x.view(ctx.hidden_shape_before_permute)

    # ----------------------------------------------------------------
    # Public API: dispatch-only (for SonicMoE — fused permute inside experts)
    # ----------------------------------------------------------------

    def preprocess_experts_input_dispatch_only(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        training: bool = False,
        padding_mask: tp.Optional[torch.Tensor] = None,
        prefix_mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatcherContext]:
        """_preprocess_context + _balance_tokens + _deepep_dispatch + _postprocess_context (no permute)."""
        hidden_states = hidden_states.flatten(0, -2)
        ctx = self._preprocess_context(hidden_states)
        routing_map, probs = self._balance_tokens(
            probs=probs,
            routing_map=routing_map,
            padding_mask=padding_mask,
            prefix_mask=prefix_mask,
        )
        if self.enable_deepep and _DEEPEP_AVAILABLE and self.ep_size > 1:
            dispatched_hidden_states, local_map, local_probs, ctx, _, _, _ = self._deepep_dispatch(
                hidden_states, probs, routing_map, ctx
            )
        else:
            if self.ep_size == 1:
                dispatched_hidden_states, local_map, local_probs = hidden_states, routing_map, probs
            else:
                raise RuntimeError("dispatch_only path requires DeepEP when ep_size > 1")
        return dispatched_hidden_states, local_map, local_probs, ctx

    def postprocess_experts_output_dispatch_only(
        self,
        expert_output: torch.Tensor,
        ctx: DispatcherContext,
    ) -> torch.Tensor:
        """_deepep_combine + _postprocess_context."""
        combined = self._deepep_combine(expert_output, ctx)
        return self._postprocess_context(combined, ctx)

    # ----------------------------------------------------------------
    # Public API: full permute + dispatch (for GroupedLlamaMLP — ggemm path)
    # ----------------------------------------------------------------

    def preprocess_experts_input(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        training: bool = False,
        padding_mask: tp.Optional[torch.Tensor] = None,
        prefix_mask: tp.Optional[torch.Tensor] = None,
        is_float8_training: bool = False,
        weight_quantizer_handles: tp.Optional[tp.Tuple] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor], torch.Tensor,
                   tp.List[int], torch.Tensor, DispatcherContext, tp.Optional[tp.Tuple]]:
        """_preprocess_context + _balance_tokens + (deepep_permute_dispatch | _permute + _default_dispatch).

        Returns (permuted_hidden_states, tokens_per_expert_gpu, tokens_per_expert_cpu,
                 tokens_per_expert_list, permuted_probs, ctx, weights_quantized).
        """
        if not isinstance(hidden_states, Float8BlockwiseQTensor):
            hidden_states = hidden_states.flatten(0, -2)
        # Float8BlockwiseQTensor is always 2D — already flattened by FP8 dispatch path
        ctx = self._preprocess_context(hidden_states)
        routing_map, probs = self._balance_tokens(
            probs=probs,
            routing_map=routing_map,
            padding_mask=padding_mask,
            prefix_mask=prefix_mask,
        )

        if training and self.enable_deepep and _DEEPEP_AVAILABLE and self.ep_size > 1:
            return self._deepep_permute_dispatch(
                hidden_states, probs, routing_map, ctx,
                is_float8_training=is_float8_training,
                weight_quantizer_handles=weight_quantizer_handles,
            )

        permuted_hidden_states, tokens_per_local_expert, permuted_probs, ctx = self._permute(
            hidden_states, probs, routing_map, ctx,
            is_float8_training=is_float8_training,
        )
        permuted_hidden_states, tokens_per_local_expert, permuted_probs, ctx = self._default_dispatch(
            permuted_hidden_states, permuted_probs, tokens_per_local_expert, ctx
        )
        tokens_per_expert_list = tokens_per_local_expert.tolist()
        return (permuted_hidden_states, None, tokens_per_local_expert,
                tokens_per_expert_list, permuted_probs, ctx, None)

    def postprocess_experts_output(
        self,
        expert_output: torch.Tensor,
        ctx: DispatcherContext,
        training: bool = False,
        is_float8_training: bool = False,
    ) -> torch.Tensor:
        """_unpermute + _default_combine / _deepep_combine + _postprocess_context."""
        if training and self.enable_deepep and _DEEPEP_AVAILABLE and getattr(ctx, "deepep_handle", None) is not None:
            # Unpermute (reverse local permute from _deepep_permute_dispatch)
            if not is_float8_training:
                unpermuted = fused_unpermute(
                    expert_output,
                    ctx.deepep_reversed_mapping_for_combine,
                    merging_probs=None,
                    restore_shape=ctx.deepep_recv_hidden_shape,
                )
            else:
                unpermuted = unpad_and_unpermute_fn(
                    expert_output,
                    ctx.deepep_reversed_mapping_for_combine,
                    ctx.deepep_reversed_mapping_for_combine_padded,
                    ctx.deepep_recv_hidden_shape,
                    ctx.deepep_pad_starts,
                    ctx.deepep_pad_counts,
                )
            combined = self._deepep_combine(unpermuted, ctx, is_float8_training=is_float8_training)
            return combined

        combined = self._default_combine(expert_output, ctx)
        unpermuted = self._unpermute(combined, ctx,
                                     is_float8_training=is_float8_training)
        return self._postprocess_context(unpermuted, ctx)


    def deepep_available(self, training: bool = False):
        return training and self.enable_deepep and _DEEPEP_AVAILABLE and self.ep_size > 1
    
    def preprocess(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        training: bool = False,
        padding_mask: tp.Optional[torch.Tensor] = None,
        prefix_mask: tp.Optional[torch.Tensor] = None,
    ):
        # # fill zero all padding tokens in routing map
        # routing_map, probs = self._drop_pad_tokens(routing_map, probs, padding_mask)

        assert len(hidden_states.size()) == 2, "Please reshape inputs to [-1, hidden_size] before applying permutations"
        hidden_states = hidden_states.flatten(0, -2)

        routing_map, probs = self._balance_tokens(
            probs=probs,
            routing_map=routing_map,
            padding_mask=padding_mask,
            prefix_mask=prefix_mask,
        )

        # dispatch_preprocess
        ## 1. _initialize_metadata
        num_local_tokens = routing_map.shape[0]

        ## 2. setup_metadata
        rm_flat = routing_map.view(num_local_tokens, -1)
        pr_flat = probs.view(num_local_tokens, -1)

        token_probs, token_indices = torch.topk(pr_flat, k=self.num_experts_per_tok, dim=-1)
        token_probs = token_probs.float()

        if self.dispatcher_balancing_strategy:
            # TODO: rewrite without extra allocations
            not_selected_token_mask = rm_flat.gather(dim=-1, index=token_indices) == 0
            token_indices = token_indices.masked_fill(not_selected_token_mask, -1)
        return probs, routing_map, token_indices, token_probs
    
    def post_dispatch_scmoe(
        self,
        recv_x,
        dispatched_indices,
        dispatched_probs,
        tokens_per_expert,
        ctx,
        is_float8_training: bool = False,
    ):

        # dispatch_postprocess
        ## get_permuted_hidden_states_by_experts
        ## 1. _indices_to_multihot
        batch_size = dispatched_indices.shape[0]
        device = dispatched_indices.device
        safe_indices = torch.where(
            dispatched_indices == -1,
            torch.tensor(self.num_local_experts, device=device, dtype=dispatched_indices.dtype),
            dispatched_indices
        )
        # Allocate target tensors with ONE extra column (+1) for the dummy writes
        padded_map = torch.zeros(
            (batch_size, self.num_local_experts + 1),
            dtype=torch.bool,
            device=device
        )
        padded_probs = torch.zeros(
            (batch_size, self.num_local_experts + 1),
            dtype=torch.float,
            device=device
        )
        padded_map.scatter_(dim=1, index=safe_indices, value=True)
        padded_probs.scatter_(dim=1, index=safe_indices, src=dispatched_probs.float())
        local_map = padded_map[:, :-1]
        local_probs = padded_probs[:, :-1]

        tokens_per_expert_cpu = (
            tokens_per_expert if tokens_per_expert.device.type == "cpu" else tokens_per_expert.cpu()
        )
        assert int(local_map.sum().item()) == int(tokens_per_expert_cpu.sum().item()), (
            "DeepEP: mismatch between local multihot and tokens_per_expert"
        )

        ## 2. permute
        tokens_per_expert_gpu = (
            tokens_per_expert_cpu.to(device) if is_float8_training else tokens_per_expert_cpu
        )
        tokens_per_expert_list = tokens_per_expert_cpu.tolist()
        if is_float8_training:
            (
                permuted_hidden_states,
                permuted_probs,
                ctx.deepep_reversed_mapping_for_combine,
                ctx.deepep_reversed_mapping_for_combine_padded,
                ctx.deepep_pad_starts,
                ctx.deepep_pad_counts,
            ) = permute_and_pad_fn(
                recv_x=recv_x,
                local_map=local_map,
                probs=local_probs,
                tokens_per_expert=tokens_per_expert_gpu,
            )
        else:
            (
                permuted_hidden_states,
                permuted_probs,
                ctx.deepep_reversed_mapping_for_combine,
            ) = fused_permute_with_probs(
                recv_x,
                local_probs,
                local_map,
                tokens_per_expert_cpu.sum().item(),
            )
            ctx.deepep_reversed_mapping_for_combine_padded = None
            ctx.deepep_pad_starts = None
            ctx.deepep_pad_counts = None
        return (
            permuted_hidden_states,
            tokens_per_expert_gpu,
            tokens_per_expert_cpu,
            tokens_per_expert_list,
            permuted_probs,
        )

        
def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: torch.Tensor | None = None,
):
    if probs is None:
        return fused_sort_chunks_by_index(input, split_sizes, sorted_idxs), None
    return fused_sort_chunks_by_index_with_probs(input, probs, split_sizes, sorted_idxs)
