# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import torch
import torch.distributed as torch_dist

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

import math

__all__ = ['ExpertBalancingMonitor']

from llmfoundry.models.giga_mix import ComposerGigaMixCausalLM
from llmfoundry.models.dpo import ComposerDPOModel
from llmfoundry.models.gigavision import ComposerGigaVisionCausalLM


def get_moe_gates(state: State):
    required_methods = (
        "activate_router_monitor",
        "get_router_monitor",
        "reset_router_monitor",
    )

    def _assert_supports_router_monitor(gate_layer, gate_name: str) -> None:
        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(getattr(gate_layer, method_name, None))
        ]
        assert len(missing_methods) == 0, (
            f"Gate `{gate_name}` ({type(gate_layer).__name__}) does not support router monitor. "
            f"Missing methods: {missing_methods}"
        )

    if isinstance(state.model, ComposerGigaMixCausalLM):
        model = state.model.model.model
    elif isinstance(state.model, ComposerDPOModel):
        model = state.model.model.model
    elif isinstance(state.model, ComposerGigaVisionCausalLM):
        model = state.model.model._language_model.model
    else:
        AssertionError(f"Not implemented logic for another models: {type(state.model)=}")

    gate_layers = []
    name_tags = []

    layers = getattr(model, "layers", [])
    mtp_layers = getattr(getattr(model, "mtp_block", None), "mtp_layers", []) or []

    for i, layer in enumerate(layers):
        if not hasattr(layer, 'self_attn') or not getattr(layer.self_attn, "is_gate", False):
            continue
        gate_layer = getattr(layer.self_attn, "_gate", None)

        if gate_layer is None:
            continue
        _assert_supports_router_monitor(gate_layer, f"layer_{i}")

        gate_layers.append(gate_layer)
        name_tags.append(f"layer_{i}")

    for i, mtp in enumerate(mtp_layers):
        dec = getattr(mtp, "decoder_layer", None)
        if dec is None or not hasattr(dec, "self_attn"):
            continue
        base_attn = getattr(dec.self_attn, "base_attn", None)
        if base_attn is None or not getattr(base_attn, "is_gate", False):
            continue
        gate_layer = getattr(base_attn, "_gate", None)


        if gate_layer is None:
            continue
        _assert_supports_router_monitor(gate_layer, f"mtp_layer_{i}")

        gate_layers.append(gate_layer)
        name_tags.append(f"mtp_layer_{i}")

    assert len(gate_layers) != 0, "gate layers not found (neither in layers nor in mtp_layers)"
    return gate_layers, name_tags


class ExpertBalancingMonitor(Callback):
    """
    Expert balancing monitor callback.

    This callback periodically logs the number of tokens processed by each expert in the MoE router.
    The data is logged to the `moe_routing/layer{layer_num}/expert{expert_num}` key, where layer_num and expert_num
    correspond to the layer and expert number in the MoE router, respectively.

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
        activate_h_balance_metrics: bool = False,
        activate_per_expert_metrics: bool = False,
        activate_per_layer_metrics: bool = True
    ):
        self.batch_log_interval = batch_log_interval or 100
        self.activate_h_balance_metrics = activate_h_balance_metrics
        self.activate_per_expert_metrics = activate_per_expert_metrics
        self.activate_per_layer_metrics = activate_per_layer_metrics

        self.moe_layer_gates = []
        self.name_tags = []

    def _fit_start(self, state: State, logger: Logger):
        self.moe_layer_gates, self.name_tags = get_moe_gates(state=state)

    def fit_start(self, state: State, logger: Logger):
        self._fit_start(state, logger)

    def batch_start(self, state: State, logger: Logger):
        if ((state.timestamp.batch.value+1) % self.batch_log_interval) != 0:
            return

        for gate_layer in self.moe_layer_gates:
            gate_layer.activate_router_monitor()

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        training_metrics = {}
        assert isinstance(state.model, tuple(self.supported_models)), (
            f"Logic not implemented for this model type. "
            f"Supported: {self.supported_models}. Got: {type(state.model).__name__}"
        )
        utilization = []
        sparsity = []
        n_experts = 0


        for gate_layer, layer_name in zip(self.moe_layer_gates, self.name_tags):
            gate_layer.activate_router_monitor()

            router_stats = gate_layer.get_router_monitor()

            tokens_per_layer_and_expert = router_stats["tokens_per_layer_and_expert"]
            mean_top_k_scores_sum = router_stats["mean_top_k_scores_sum"]
            mean_top_k_unnormed_scores_sum = router_stats["mean_top_k_unnormed_scores_sum"]
            mean_top_k_unnormed_biased_scores_sum = router_stats["mean_top_k_unnormed_biased_scores_sum"]
            scores_per_layer_and_expert = router_stats["scores_per_layer_and_expert"]
            entropy_per_layer_and_expert = router_stats["entropy_per_layer_and_expert"]
            expert_biases = router_stats["expert_biases"]
            z_loss_router = router_stats["z_loss_router"]
            seq_aux_loss = router_stats["seq_aux_loss"]

            fsdp_group = dist.get_fsdp_group()
            if gate_layer.aux_loss_free:
                expert_biases_mean_over_rank = expert_biases.clone()

                # Vars to check if expert_biases are identical over ranks
                expert_biases_min_over_rank = expert_biases_mean_over_rank.clone()
                expert_biases_max_over_rank = expert_biases_mean_over_rank.clone()


                tp_sp_group_size = dist.get_tp_sp_group_size() or 1
                if fsdp_group is None:
                    expert_biases_mean_over_rank /= dist.get_world_size()
                    z_loss_router /= dist.get_world_size()
                    seq_aux_loss /= dist.get_world_size()
                else:
                    expert_biases_mean_over_rank /= dist.get_fsdp_group_size()
                    z_loss_router /= dist.get_fsdp_group_size()
                    seq_aux_loss /= dist.get_fsdp_group_size()

                # fsdp_ctx_mgr = torch_dist._coalescing_manager(group=fsdp_group)
                # with fsdp_ctx_mgr:
                dist.all_reduce(expert_biases_min_over_rank, reduce_operation='MIN', group=fsdp_group, async_op=False)
                dist.all_reduce(expert_biases_max_over_rank, reduce_operation='MAX', group=fsdp_group, async_op=False)
                dist.all_reduce(expert_biases_mean_over_rank, reduce_operation='SUM', group=fsdp_group, async_op=False)
                dist.all_reduce(seq_aux_loss, reduce_operation='SUM', group=fsdp_group, async_op=False)
                dist.all_reduce(z_loss_router, reduce_operation='SUM', group=fsdp_group, async_op=False)

                assert torch.equal(expert_biases_min_over_rank, expert_biases_max_over_rank), "biases should be the same over differnt ranks"




            if fsdp_group is None:
                tokens_per_layer_and_expert /= dist.get_world_size()
                mean_top_k_scores_sum /= dist.get_world_size()
                mean_top_k_unnormed_scores_sum /= dist.get_world_size()
                mean_top_k_unnormed_biased_scores_sum /= dist.get_world_size()
                scores_per_layer_and_expert /= dist.get_world_size()
                entropy_per_layer_and_expert /= dist.get_world_size()
            else:
                tokens_per_layer_and_expert /= dist.get_fsdp_group_size()
                mean_top_k_scores_sum /= dist.get_fsdp_group_size()
                mean_top_k_unnormed_scores_sum /= dist.get_fsdp_group_size()
                mean_top_k_unnormed_biased_scores_sum /= dist.get_fsdp_group_size()
                scores_per_layer_and_expert /= dist.get_fsdp_group_size()
                entropy_per_layer_and_expert /= dist.get_fsdp_group_size()

            # fsdp_ctx_mgr = torch_dist._coalescing_manager(group=fsdp_group)
            # with fsdp_ctx_mgr:
            dist.all_reduce(tokens_per_layer_and_expert, reduce_operation='SUM', group=fsdp_group, async_op=False)
            dist.all_reduce(mean_top_k_scores_sum, reduce_operation='SUM', group=fsdp_group, async_op=False)
            dist.all_reduce(mean_top_k_unnormed_scores_sum, reduce_operation='SUM', group=fsdp_group, async_op=False)
            dist.all_reduce(mean_top_k_unnormed_biased_scores_sum, reduce_operation='SUM', group=fsdp_group, async_op=False)
            dist.all_reduce(scores_per_layer_and_expert, reduce_operation='SUM', group=fsdp_group, async_op=False)
            dist.all_reduce(entropy_per_layer_and_expert, reduce_operation='SUM', group=fsdp_group, async_op=False)

            raw = tokens_per_layer_and_expert.cpu()
            n_experts = raw.shape[-1]

            max_load = raw.max().item()
            min_load = raw.min().item()

            if self.activate_per_layer_metrics:
                training_metrics.update({
                    f"old_moe_routing_max_load/{layer_name}": max_load,
                    f"old_moe_routing_min_load/{layer_name}": min_load,
                    f"moe_seq_aux/{layer_name}": getattr(gate_layer, "alpha", None),
                    f"moe_z_loss_router/{layer_name}": z_loss_router.item(),
                    f"moe_seq_aux_loss/{layer_name}": seq_aux_loss.item(),
                })

            if gate_layer.aux_loss_free:
                expert_biases_mean_over_rank_min = expert_biases_mean_over_rank.min()
                expert_biases_mean_over_rank_max = expert_biases_mean_over_rank.max()

            tokens_per_layer_and_expert = torch.nn.functional.normalize(tokens_per_layer_and_expert, p=1)
            tokens_per_layer_and_expert = tokens_per_layer_and_expert.cpu()
            mean_top_k_scores_sum = mean_top_k_scores_sum.cpu()
            mean_top_k_unnormed_scores_sum = mean_top_k_unnormed_scores_sum.cpu()
            mean_top_k_unnormed_biased_scores_sum = mean_top_k_unnormed_biased_scores_sum.cpu()
            scores_per_layer_and_expert = scores_per_layer_and_expert.cpu()
            entropy_per_layer_and_expert = entropy_per_layer_and_expert.cpu()
            if self.activate_per_expert_metrics:
                training_metrics.update({
                    f"moe_routing_token_balancing/{layer_name}/expert{e}":
                        tokens_per_layer_and_expert[0, e].item()
                    for e in range(tokens_per_layer_and_expert.shape[-1])
                })

            if self.activate_per_layer_metrics:
                metrics = {
                    f"moe_routing_top_k_scores_sum/{layer_name}": mean_top_k_scores_sum.item(),
                    f"moe_routing_top_k_unnormed_scores_sum/{layer_name}": mean_top_k_unnormed_scores_sum.item(),
                    f"moe_routing_top_k_unnormed_biased_scores_sum/{layer_name}":
                        mean_top_k_unnormed_biased_scores_sum.item(),
                }
                if getattr(gate_layer, "aux_loss_free", False):
                    metrics.update({
                        f"moe_routing_expert_biases_mean_over_rank_min/{layer_name}": expert_biases_mean_over_rank_min.item(),
                        f"moe_routing_expert_biases_mean_over_rank_max/{layer_name}": expert_biases_mean_over_rank_max.item(),
                    })
                training_metrics.update(metrics)


            scores = scores_per_layer_and_expert.unsqueeze(0)
            entropy_per_layer = -scores * (scores + 1e-12).log()
            utilization.append(entropy_per_layer.sum().item())
            sparsity.append(entropy_per_layer_and_expert.item())

            gate_layer.reset_router_monitor()

        if self.activate_h_balance_metrics:
            if n_experts > 0:
                h_max = math.log(n_experts)
                training_metrics.update({
                    f"moe_routing_h_{name}_norm": (sum(vals) / len(vals) if vals else 0.0) / h_max
                    for name, vals in [("utilization", utilization), ("sparsity", sparsity)]
                })

        logger.log_metrics(training_metrics)

    # def eval_after_all(self, state: State, logger: Logger) -> None:
    #     self.batch_end(state, logger)
