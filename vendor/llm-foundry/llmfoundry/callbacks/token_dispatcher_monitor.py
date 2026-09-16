# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import torch
import torch.distributed as torch_dist

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['TokenDispatcherMonitor']

from llmfoundry.models.giga_mix import ComposerGigaMixCausalLM
from llmfoundry.models.dpo import ComposerDPOModel
from llmfoundry.models.gigavision import ComposerGigaVisionCausalLM


def get_experts_num(dispatcher_layer_list):
    num_experts = None
    num_experts = dispatcher_layer_list[0].config.num_routed_experts

    assert num_experts is not None, "MoE layer doesn't found"
    return num_experts



def get_moe_token_dispatchers(state: State):
    if isinstance(state.model, ComposerGigaMixCausalLM):
        model = state.model.model.model
    elif isinstance(state.model, ComposerDPOModel):
        model = state.model.model.model
    elif isinstance(state.model, ComposerGigaVisionCausalLM):
        model = state.model.model._language_model.model
    else:
        AssertionError(f"Not implemented logic for another models: {type(state.model)=}")
    dispatchers = []
    name_tags = []

    for i, layer in enumerate(getattr(model, "layers", [])):
        blk = getattr(layer, "block_sparse_moe", None)
        if blk is not None and hasattr(blk, "token_dispatcher"):
            dispatchers.append(blk.token_dispatcher)
            name_tags.append(f"layer_{i}")

    # MTP
    mtp_layers = getattr(getattr(model, "mtp_block", None), "mtp_layers", []) or []
    for j, mtp in enumerate(mtp_layers):
        dec = getattr(mtp, "decoder_layer", None)
        if dec is None:
            continue
        blk = getattr(dec, "block_sparse_moe", None)
        if blk is not None and hasattr(blk, "token_dispatcher"):
            dispatchers.append(blk.token_dispatcher)
            name_tags.append(f"mtp_layer_{j}")

    assert len(dispatchers) != 0, "MoE layers not found (neither in layers nor in mtp_layers)"
    return dispatchers, name_tags


class TokenDispatcherMonitor(Callback):
    """
    Token dispatcher monitor callback.

    Periodically logs MoE dispatching stats. If a layer's dispatcher has no
    balancing strategy, capacity-based metrics for that layer are skipped.

    Args:
        batch_log_interval (int, optional): Logging interval in batches. Defaults to 100.
    """

    supported_models = [
        ComposerGigaMixCausalLM,
        ComposerDPOModel,
        ComposerGigaVisionCausalLM,
    ]

    def __init__(
        self,
        batch_log_interval: int = 100,
        activate_per_expert_metrics: bool = False,
        activate_per_layer_metrics: bool = True,
        activate_ep_rank_metrics: bool = False,
        activate_eval_metrics: bool = True,
    ):
        self.batch_log_interval = batch_log_interval or 100
        self.activate_per_expert_metrics = activate_per_expert_metrics
        self.activate_per_layer_metrics = activate_per_layer_metrics
        self.activate_ep_rank_metrics = activate_ep_rank_metrics
        self.activate_eval_metrics = activate_eval_metrics

        self.token_drop_step_counter = 0
        self.eval_token_drop_step_inner_counter = 0
        self.ep_group_size = dist.get_ep_group_size() or 1
        self.microbatch_num = None
        self.microbatch_log_interval = None
        self.num_layers = None
        self.moe_layer_dispatchers = []
        self.name_tags = []
        self.curr_microbatch_step = 0
        self.world_size = dist.get_world_size() or 1
        self.ep_fsdp_group_size = dist.get_ep_fsdp_group_size() or self.world_size

        self.capacity_enabled = False
        self.drop_by_group_capacity = None

    def _activate_dispatcher_monitor(self, dispatchers_list):
        for dispatcher_layer in dispatchers_list:
            dispatcher_layer._activate_dispatcher_monitor()

    def _deactivate_dispatcher_monitor(self, dispatchers_list):
        for dispatcher_layer in dispatchers_list:
            dispatcher_layer._deactivate_dispatcher_monitor()

    def _init_dispatcher_monitor(self, dispatchers_list):
        for dispatcher_layer in dispatchers_list:
            dispatcher_layer._init_dispatcher_monitor()

    def _reset_statistics(self):
        # TODO: vectorize operations between layers and remove for loops
        self._layer_local_expert_capacity_hist = [
            torch.zeros(
                (self.microbatch_log_interval, self.num_experts),
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            for _ in range(self.num_layers)
        ]
        self._layer_tokens_per_local_expert_hist = [
            torch.zeros(
                (self.microbatch_log_interval, self.num_experts),
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            for _ in range(self.num_layers)
        ]
        self._layer_tokens_per_local_expert_no_pad_no_prefix_hist = [
            torch.zeros(
                (self.microbatch_log_interval, self.num_experts),
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            for _ in range(self.num_layers)
        ]
        self._layer_tokens_per_local_expert_no_pad_hist = [
            torch.zeros(
                (self.microbatch_log_interval, self.num_experts),
                dtype=torch.float32,
                requires_grad=False,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            for _ in range(self.num_layers)
        ]
        self.curr_microbatch_step = 0

    def _fit_start(self, state: State, logger: Logger):
        metadata_dict = state._get_state_metadata()
        batch_size = metadata_dict['train_dataloader_batch_size']
        microbatch_size = metadata_dict['device_train_microbatch_size']


        microbatch_size_quotient, microbatch_size_remainder = divmod(batch_size, microbatch_size)
        self.microbatch_num = microbatch_size_quotient + (microbatch_size_remainder > 0)
        self.microbatch_log_interval = self.batch_log_interval * self.microbatch_num

        self.moe_layer_dispatchers, self.name_tags = get_moe_token_dispatchers(state=state)
        self.num_experts = get_experts_num(self.moe_layer_dispatchers)
        self.num_layers = len(self.moe_layer_dispatchers)

        if self.num_layers > 0:
            strategy = getattr(self.moe_layer_dispatchers[0].config, 'dispatcher_balancing_strategy', None)
            self.capacity_enabled = strategy in ('capacity_factor', 'fixed_capacity', 'peak_capacity_factor')
        else:
            self.capacity_enabled = False

        if self.num_layers > 0:
            self.drop_by_group_capacity = getattr(self.moe_layer_dispatchers[0].config, 'drop_by_group_capacity', None)
        else:
            self.capacity_enabled = None

        self.group_size = (self.num_experts // self.ep_group_size) if self.ep_group_size > 0 and self.num_experts > 0 else 1

        self._init_dispatcher_monitor(self.moe_layer_dispatchers)
        self._activate_dispatcher_monitor(self.moe_layer_dispatchers)
        self._reset_statistics()

        self.curr_microbatch_step = 0

    def fit_start(self, state: State, logger: Logger):
        self._fit_start(state, logger)

    def after_load(self, state: State, logger: Logger):
        self._fit_start(state, logger)

    def before_forward(self, state: State, logger: Logger):
        pass

    def after_forward(self, state: State, logger: Logger):
        for layer_num, dispatcher_layer in enumerate(self.moe_layer_dispatchers):
            local_expert_capacity, tokens_per_local_expert, tokens_per_local_expert_no_pad, tokens_per_local_expert_no_pad_no_prefix = dispatcher_layer.get_dispatcher_monitor()

            self._layer_local_expert_capacity_hist[layer_num][self.curr_microbatch_step, :] = local_expert_capacity
            self._layer_tokens_per_local_expert_hist[layer_num][self.curr_microbatch_step, :] = tokens_per_local_expert
            self._layer_tokens_per_local_expert_no_pad_hist[layer_num][self.curr_microbatch_step, :] = tokens_per_local_expert_no_pad
            self._layer_tokens_per_local_expert_no_pad_no_prefix_hist[layer_num][self.curr_microbatch_step, :] = tokens_per_local_expert_no_pad_no_prefix

        self.curr_microbatch_step += 1

    def eval_after_forward(self, state: State, logger: Logger):
        if  self.activate_eval_metrics:
            # NOTE: Return pytorch memory pool to CUDA runtime
            torch.cuda.empty_cache()

            for layer_num, dispatcher_layer in enumerate(self.moe_layer_dispatchers):
                local_expert_capacity, local_tokens_per_expert, local_tokens_per_expert_no_pad, local_tokens_per_expert_no_pad_no_token = dispatcher_layer.get_dispatcher_monitor()

                global_expert_capacity = local_expert_capacity.clone()
                global_tokens_per_expert = local_tokens_per_expert.clone()

                ep_group = dist.get_ep_group()
                tp_sp_group = dist.get_tp_sp_group()
                ep_fsdp_group = dist.get_ep_fsdp_group()

                # HOTFIX: remove coalescing_manager
                # ep_ctx_mgr = torch_dist._coalescing_manager(group=ep_group)
                # with ep_ctx_mgr:
                #     if ep_group is not None:

                dist.all_reduce(global_tokens_per_expert, reduce_operation="SUM", group=ep_group, async_op=False)
                dist.all_reduce(global_expert_capacity, reduce_operation="SUM", group=ep_group, async_op=False)

                if tp_sp_group is not None:
                    dist.all_reduce(global_tokens_per_expert, reduce_operation="SUM", group=tp_sp_group, async_op=False)
                    dist.all_reduce(global_expert_capacity, reduce_operation="SUM", group=tp_sp_group, async_op=False)


                global_ep_rank_capacity = global_expert_capacity.reshape(self.ep_group_size, self.group_size).sum(1)
                global_tokens_per_ep_rank = global_tokens_per_expert.reshape(self.ep_group_size, self.group_size).sum(1)

                global_overloaded_experts = global_tokens_per_expert > global_expert_capacity
                global_overloaded_ep_ranks = global_tokens_per_ep_rank > global_ep_rank_capacity

                # ep_fsdp_ctx_mgr = torch_dist._coalescing_manager(group=ep_fsdp_group)
                # with ep_fsdp_ctx_mgr:
                if ep_group is not None:
                    dist.all_reduce(global_overloaded_experts, reduce_operation="MAX", group=ep_fsdp_group, async_op=False)
                    dist.all_reduce(global_overloaded_ep_ranks, reduce_operation="MAX", group=ep_fsdp_group, async_op=False)

                if self.capacity_enabled:
                    # count steps with overloaded groups per log interval
                    if self.drop_by_group_capacity:
                        if sum(global_overloaded_ep_ranks) > 0:
                            self.eval_token_drop_step_inner_counter += 1
                            break
                    else:
                        if sum(global_overloaded_experts) > 0:
                            self.eval_token_drop_step_inner_counter += 1
                            break


    def batch_start(self, state: State, logger: Logger):
        self._activate_dispatcher_monitor(self.moe_layer_dispatchers)

    def eval_batch_start(self, state: State, logger: Logger):
        if  self.activate_eval_metrics:
            self._activate_dispatcher_monitor(self.moe_layer_dispatchers)

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        # NOTE: Return pytorch memory pool to CUDA runtime
        torch.cuda.empty_cache()

        training_metrics = {
            "dispatcher_sum_metrics/experts_max_vio_microbatch_avg": 0.0,
            "dispatcher_sum_metrics/experts_max_vio_batch_avg": 0.0,
            "dispatcher_sum_metrics/experts_max_vio_batch_no_pad_avg": 0.0,
            "dispatcher_sum_metrics/experts_max_vio_batch_no_pad_no_prefix_avg": 0.0,
            "dispatcher_sum_metrics/processed_batch_tokens": 0.0,
            "dispatcher_sum_metrics/processed_batch_tokens_no_pad": 0.0,
            "dispatcher_sum_metrics/processed_batch_tokens_no_pad_no_prefix": 0.0,

        }
        if self.activate_ep_rank_metrics:
            training_metrics["dispatcher_sum_metrics/ep_ranks_max_vio_microbatch_avg"] = 0.0
        step_max_expert_load = 0
        step_max_group_load = 0

        if self.capacity_enabled:
            training_metrics["dispatcher_sum_metrics/overloaded_expert_ratio"] = 0.0
            training_metrics["dispatcher_sum_metrics/token_drop_level"] = 0.0
            training_metrics["dispatcher_sum_metrics/empty_token_level"] = 0.0
            if self.activate_ep_rank_metrics:
                training_metrics["dispatcher_sum_metrics/overloaded_ep_rank_ratio"] = 0.0
            overloaded_groups_step = torch.zeros(
                (self.batch_log_interval, self.ep_group_size),
                dtype=torch.bool,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            overloaded_experts_step = torch.zeros(
                (self.batch_log_interval, self.num_experts),
                dtype=torch.bool,
                device=f"cuda:{torch.cuda.current_device()}",
            )

        # skip_updates count
        assert isinstance(state.model, tuple(self.supported_models)), (
            f"Logic not implemented for this model type. "
            f"Supported: {self.supported_models}. Got: {type(state.model).__name__}"
        )
        ep_group = dist.get_ep_group()
        tp_sp_group = dist.get_tp_sp_group()
        ep_fsdp_group = dist.get_ep_fsdp_group()

        for layer_num, (local_expert_capacity_hist, tokens_per_local_expert_hist, tokens_per_local_expert_hist_no_pad, tokens_per_local_expert_hist_no_pad_no_prefix) in enumerate(
            zip(
                self._layer_local_expert_capacity_hist,
                self._layer_tokens_per_local_expert_hist,
                self._layer_tokens_per_local_expert_no_pad_hist,
                self._layer_tokens_per_local_expert_no_pad_no_prefix_hist
            )
        ):
            tokens_per_global_expert_hist = tokens_per_local_expert_hist.clone()
            tokens_per_global_expert_hist_no_pad = tokens_per_local_expert_hist_no_pad.clone()
            tokens_per_global_expert_hist_no_pad_no_prefix = tokens_per_local_expert_hist_no_pad_no_prefix.clone()
            if ep_group is not None:
                dist.all_reduce(tokens_per_global_expert_hist, reduce_operation="SUM", group=ep_group, async_op=False)
                dist.all_reduce(tokens_per_global_expert_hist_no_pad, reduce_operation="SUM", group=ep_group, async_op=False)
                dist.all_reduce(tokens_per_global_expert_hist_no_pad_no_prefix, reduce_operation="SUM", group=ep_group, async_op=False)

            global_tokens_per_expert_group_hist = tokens_per_global_expert_hist.view(
                (self.microbatch_log_interval, self.ep_group_size, self.group_size)
            ).sum(2)

            step_max_expert_load = max(tokens_per_global_expert_hist.max(), step_max_expert_load)
            step_max_group_load = max(global_tokens_per_expert_group_hist.max(), step_max_group_load)

            expected_load_hist = tokens_per_global_expert_hist.sum(dim=-1) / max(1, self.num_experts)
            max_load_hist = tokens_per_global_expert_hist.max(dim=-1).values
            denom_exp = torch.clamp(expected_load_hist, min=1.0)
            max_vio = ((max_load_hist - expected_load_hist) / denom_exp).sum() / max(1, self.microbatch_log_interval)

            expected_group_load_hist = global_tokens_per_expert_group_hist.sum(dim=-1) / max(1, self.ep_group_size)
            group_max_load_hist = global_tokens_per_expert_group_hist.max(dim=-1).values
            denom_grp = torch.clamp(expected_group_load_hist, min=1.0)
            group_max_vio = ((group_max_load_hist - expected_group_load_hist) / denom_grp).sum() / max(1, self.microbatch_log_interval)

            if tp_sp_group is not None:
                dist.all_reduce(tokens_per_global_expert_hist, reduce_operation="SUM", group=tp_sp_group, async_op=False)
                dist.all_reduce(tokens_per_global_expert_hist_no_pad, reduce_operation="SUM", group=tp_sp_group, async_op=False)
                dist.all_reduce(tokens_per_global_expert_hist_no_pad_no_prefix, reduce_operation="SUM", group=tp_sp_group, async_op=False)

            # batch_statistics pre communication
            tokens_per_global_expert_batch_hist = tokens_per_global_expert_hist.reshape(
                self.batch_log_interval,
                self.microbatch_num,
                self.num_experts
            ).sum(1)
            tokens_per_global_expert_batch_hist_no_pad = tokens_per_global_expert_hist_no_pad.reshape(
                self.batch_log_interval,
                self.microbatch_num,
                self.num_experts
            ).sum(1)
            tokens_per_global_expert_batch_hist_no_pad_no_prefix = tokens_per_global_expert_hist_no_pad_no_prefix.reshape(
                self.batch_log_interval,
                self.microbatch_num,
                self.num_experts
            ).sum(1)

            max_vio /= self.ep_fsdp_group_size
            group_max_vio /= self.ep_fsdp_group_size

            # ep_fsdp_ctx_mgr = torch_dist._coalescing_manager(group=ep_fsdp_group)
            # with ep_fsdp_ctx_mgr:
            # calc avg max_vio values
            dist.all_reduce(max_vio, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
            dist.all_reduce(group_max_vio, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)

            # calc max per group values
            dist.all_reduce(step_max_group_load, reduce_operation="MAX", group=ep_fsdp_group, async_op=False)
            dist.all_reduce(step_max_expert_load, reduce_operation="MAX", group=ep_fsdp_group, async_op=False)

            # calc batched statistics
            dist.all_reduce(tokens_per_global_expert_batch_hist, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
            dist.all_reduce(tokens_per_global_expert_batch_hist_no_pad, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
            dist.all_reduce(tokens_per_global_expert_batch_hist_no_pad_no_prefix, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)

            # batch_statistics post communication
            global_processed_batch_tokens = tokens_per_global_expert_batch_hist.view(self.batch_log_interval, -1).sum(dim=-1).mean()
            expected_load_batch_hist = tokens_per_global_expert_batch_hist.sum(dim=-1) / max(1, self.num_experts)
            max_load_batch_hist = tokens_per_global_expert_batch_hist.max(dim=-1).values
            denom_exp_batch = torch.clamp(expected_load_batch_hist, min=1.0)
            max_vio_batch = ((max_load_batch_hist - expected_load_batch_hist) / denom_exp_batch).sum() / max(1, self.batch_log_interval)

            # batch_statistics post communication
            global_processed_batch_tokens_no_pad = tokens_per_global_expert_batch_hist_no_pad.view(self.batch_log_interval, -1).sum(dim=-1).mean()
            expected_load_batch_hist_no_pad = tokens_per_global_expert_batch_hist_no_pad.sum(dim=-1) / max(1, self.num_experts)
            max_load_batch_hist_no_pad = tokens_per_global_expert_batch_hist_no_pad.max(dim=-1).values
            denom_exp_batch_no_pad = torch.clamp(expected_load_batch_hist_no_pad, min=1.0)
            max_vio_batch_no_pad = ((max_load_batch_hist_no_pad - expected_load_batch_hist_no_pad) / denom_exp_batch_no_pad).sum() / max(1, self.batch_log_interval)

            # batch_statistics post communication
            global_processed_batch_tokens_no_pad_no_prefix = tokens_per_global_expert_batch_hist_no_pad_no_prefix.view(self.batch_log_interval, -1).sum(dim=-1).mean()
            expected_load_batch_hist_no_pad_no_prefix = tokens_per_global_expert_batch_hist_no_pad_no_prefix.sum(dim=-1) / max(1, self.num_experts)
            max_load_batch_hist_no_pad_no_prefix = tokens_per_global_expert_batch_hist_no_pad_no_prefix.max(dim=-1).values
            denom_exp_batch_no_pad_no_prefix = torch.clamp(expected_load_batch_hist_no_pad_no_prefix, min=1.0)
            max_vio_batch_no_pad_no_prefix = ((max_load_batch_hist_no_pad_no_prefix - expected_load_batch_hist_no_pad_no_prefix) / denom_exp_batch_no_pad_no_prefix).sum() / max(1, self.batch_log_interval)

            training_metrics["dispatcher_sum_metrics/experts_max_vio_microbatch_avg"] += max_vio
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_avg"] += max_vio_batch
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_no_pad_avg"] += max_vio_batch_no_pad
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_no_pad_no_prefix_avg"] += max_vio_batch_no_pad_no_prefix
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens"] += global_processed_batch_tokens
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad"] += global_processed_batch_tokens_no_pad
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad_no_prefix"] += global_processed_batch_tokens_no_pad_no_prefix
            if self.activate_ep_rank_metrics:
                training_metrics["dispatcher_sum_metrics/ep_ranks_max_vio_microbatch_avg"] += group_max_vio
            if self.activate_per_layer_metrics:
                tag = self.name_tags[layer_num]
                training_metrics[f"dispatcher_layer_experts_max_vio_microbatch_avg/{tag}"] = max_vio
                training_metrics[f"dispatcher_layer_experts_max_vio_batch_avg/{tag}"] = max_vio_batch
                training_metrics[f"dispatcher_layer_experts_max_vio_batch_no_pad_avg/{tag}"] = max_vio_batch_no_pad
                training_metrics[f"dispatcher_layer_experts_max_vio_batch_no_pad_no_prefix_avg/{tag}"] = max_vio_batch_no_pad_no_prefix
                if self.activate_ep_rank_metrics:
                    training_metrics[f"dispatcher_layer_ep_ranks_max_vio_microbatch_avg/{tag}"] = group_max_vio

            if self.capacity_enabled:
                diff = (tokens_per_local_expert_hist - local_expert_capacity_hist).reshape(
                    (self.batch_log_interval, self.microbatch_num, self.num_experts)
                )
                mask_positive = diff > 0
                mask_negative = diff < 0
                droped_tokens_per_training_step = torch.sum(diff * mask_positive, dim=(1, 2))
                empty_tokens = -torch.sum(diff * mask_negative)

                overloaded_local_experts_hist = tokens_per_local_expert_hist > local_expert_capacity_hist
                global_expert_capacity_hist = local_expert_capacity_hist.clone()

                if ep_group is not None:
                    # don't wrap with coalescing manager because of different data type
                    dist.all_reduce(overloaded_local_experts_hist, reduce_operation="MAX", group=ep_group, async_op=False)

                    # ep_ctx_mgr_fp = torch_dist._coalescing_manager(group=ep_group)
                    # with ep_ctx_mgr_fp:
                    dist.all_reduce(global_expert_capacity_hist, reduce_operation="SUM", group=ep_group, async_op=False)
                    dist.all_reduce(droped_tokens_per_training_step, reduce_operation="SUM", group=ep_group, async_op=False)
                    dist.all_reduce(empty_tokens, reduce_operation="SUM", group=ep_group, async_op=False)


                global_group_capacity_hist = global_expert_capacity_hist.view(
                    (self.microbatch_log_interval, self.ep_group_size, self.group_size)
                ).sum(2)
                overloaded_global_expert_groups_hist = global_tokens_per_expert_group_hist > global_group_capacity_hist

                overloaded_experts_training_step = overloaded_local_experts_hist.view(
                    self.batch_log_interval, self.microbatch_num, self.num_experts
                ).max(1).values

                overloaded_groups_training_step = overloaded_global_expert_groups_hist.view(
                    self.batch_log_interval, self.microbatch_num, self.ep_group_size
                ).max(1).values

                overloaded_experts_ratio_log_step = overloaded_experts_training_step.sum(0) / max(1, self.batch_log_interval)
                overloaded_groups_ratio_log_step = overloaded_groups_training_step.sum(0) / max(1, self.batch_log_interval)

                # TODO: add if/else condition for flag drop_by_group_capacity. Current result is calculated for drop_by_group_capacity: true
                # calc droped tokens only for overloaded steps
                if self.drop_by_group_capacity:
                    droped_tokens = (droped_tokens_per_training_step * overloaded_groups_training_step.max(1).values).sum()
                else:
                    droped_tokens = (droped_tokens_per_training_step * overloaded_experts_training_step.max(1).values).sum()

                overloaded_experts_ratio_log_step /= self.ep_fsdp_group_size
                overloaded_groups_ratio_log_step /= self.ep_fsdp_group_size

                global_expert_capacity_hist /= self.ep_fsdp_group_size
                droped_tokens /= self.ep_fsdp_group_size
                empty_tokens /= self.ep_fsdp_group_size

                # ep_fsdp_ctx_mgr = torch_dist._coalescing_manager(group=ep_fsdp_group)
                # with ep_fsdp_ctx_mgr:
                dist.all_reduce(overloaded_experts_ratio_log_step, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
                dist.all_reduce(overloaded_groups_ratio_log_step, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)

                # calc absolute values
                dist.all_reduce(global_expert_capacity_hist, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
                dist.all_reduce(droped_tokens, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)
                dist.all_reduce(empty_tokens, reduce_operation="SUM", group=ep_fsdp_group, async_op=False)

                cap_sum = torch.clamp(global_expert_capacity_hist.sum(), min=1.0)
                training_metrics["dispatcher_sum_metrics/overloaded_expert_ratio"] += (
                    overloaded_experts_ratio_log_step.sum() / max(1, self.num_experts)
                )
                if self.activate_per_layer_metrics:
                    tag = self.name_tags[layer_num]
                    training_metrics[f"dispatcher_layer_overloaded_ratio/{tag}"] = (
                        overloaded_experts_ratio_log_step.sum() / self.num_experts
                    )

                if self.activate_per_expert_metrics:
                    for expert_num in range(self.num_experts):
                        tag = self.name_tags[layer_num]
                        training_metrics[
                            f"dispatcher_experts_overloaded_ratio/{tag}_expert{expert_num}"
                        ] = overloaded_experts_ratio_log_step[expert_num]
                if self.activate_ep_rank_metrics:
                    training_metrics["dispatcher_sum_metrics/overloaded_ep_rank_ratio"] += (
                        overloaded_groups_ratio_log_step.sum() / max(1, self.ep_group_size)
                    )
                training_metrics["dispatcher_sum_metrics/token_drop_level"] += droped_tokens / cap_sum
                training_metrics["dispatcher_sum_metrics/empty_token_level"] += empty_tokens / cap_sum

                if self.activate_per_layer_metrics:
                    tag = self.name_tags[layer_num]
                    training_metrics[f"dispatcher_layer_experts_overloaded_ratio/{tag}"] = (
                        overloaded_experts_ratio_log_step.sum() / max(1, self.num_experts)
                    )

                overloaded_groups_step = (overloaded_groups_training_step > 0) | overloaded_groups_step
                overloaded_experts_step = (overloaded_experts_training_step > 0) | overloaded_experts_step

        if self.num_layers and self.num_layers > 0:
            training_metrics["dispatcher_sum_metrics/experts_max_vio_microbatch_avg"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_avg"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_no_pad_avg"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/experts_max_vio_batch_no_pad_no_prefix_avg"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad_no_prefix"] /= self.num_layers
            if self.activate_ep_rank_metrics:
                training_metrics["dispatcher_sum_metrics/ep_ranks_max_vio_microbatch_avg"] /= self.num_layers

        if self.capacity_enabled and self.num_layers and self.num_layers > 0:
            training_metrics["dispatcher_sum_metrics/overloaded_expert_ratio"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/token_drop_level"] /= self.num_layers
            training_metrics["dispatcher_sum_metrics/empty_token_level"] /= self.num_layers
            if self.activate_ep_rank_metrics:
                training_metrics["dispatcher_sum_metrics/overloaded_ep_rank_ratio"] /= self.num_layers


        training_metrics["dispatcher_sum_metrics/step_max_expert_load"] = step_max_expert_load
        if self.activate_ep_rank_metrics:
            training_metrics["dispatcher_sum_metrics/step_max_ep_rank_load"] = step_max_group_load

        if self.capacity_enabled:
            # count steps with overloaded groups per log interval
            if self.drop_by_group_capacity:
                self.token_drop_step_counter += overloaded_groups_step.max(1).values.sum(0)
            else:
                self.token_drop_step_counter += overloaded_experts_step.max(1).values.sum(0)
            training_metrics["dispatcher_sum_metrics/num_drop_steps"] = self.token_drop_step_counter

        logger.log_metrics(training_metrics)

        self._reset_statistics()
        self._deactivate_dispatcher_monitor(self.moe_layer_dispatchers)


    def eval_end(self, state: State, logger: Logger):

         if  self.activate_eval_metrics:

            training_metrics = {
                "dispatcher_sum_metrics/eval_num_drop_steps": self.eval_token_drop_step_inner_counter
                }
            logger.log_metrics(training_metrics)

            self.eval_token_drop_step_inner_counter = 0

            self._deactivate_dispatcher_monitor(self.moe_layer_dispatchers)