import math
import warnings
from typing import Optional, Tuple, Type, Dict

import torch
import torch.nn.functional as F
from torch import nn

from composer.utils import dist, global_actlog_state
from composer.utils.dist import  get_ep_group_size
from composer.utils.profiler_annotation import decorator_forward_backward

from llmfoundry.models.layers.fc import ColumnParallelLinear
from llmfoundry.models.parallel.tensor import distribute_async_tp_embeddings
from llmfoundry.models.parallel.sequence import all_reduce_from_tensor_sequence_parallel_region
from llmfoundry.models.ops.zloss import z_loss_fn

TOPK_GATE_TYPE = "top-k"
UNIFORM_GATE_TYPE = "dummy_uniform"
FIRST_EXPERT_GROUP_GATE_TYPE = "first_expert_group"
DYNAMIC_GATE_TYPE = "dynamic"

class AddRouterZLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_router_z_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_router_z_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoEGate(nn.Module):
    def __init__(self, config, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.num_routed_experts
        self.init_device = config.init_device

        k_dense = getattr(config, "first_k_dense_replace", 0) or 0
        n_first_moe = getattr(config, "first_seq_aux_moe_layers", 0)

        if layer_idx == -1:
            alpha = config.router_aux_loss_coef # for MTP
        else:
            moe_pos = layer_idx - k_dense
            if moe_pos >= 0 and moe_pos < n_first_moe:
                alpha = config.first_router_aux_loss_coef
            else:
                alpha = config.router_aux_loss_coef

        self.alpha = alpha

        self.seq_aux = config.router_seq_aux
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.moe_router_routed_scaling_factor

        self.tp_size = config.tp_size or 1
        self.renorm_router_weights = config.renorm_router_weights
        if self.renorm_router_weights and self.tp_size > 1:
            raise AssertionError(
                "`renorm_router_weights=True` is not supported with tensor parallel `tp_size > 1`."
            )

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        self.num_groups = config.moe_router_num_groups
        self.group_topk = config.moe_router_group_topk

        # free loss
        self.aux_loss_free = config.aux_loss_free

        self.gate_type = TOPK_GATE_TYPE

        if self.tp_size > 1:
            assert config.enable_async_tp
            self.gate = ColumnParallelLinear(
                self.gating_dim,
                self.n_routed_experts,
                config=config,
                bias=False,
            )
        else:
            self.gate = nn.Linear(self.gating_dim, self.n_routed_experts, bias=False)

        if self.aux_loss_free:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.n_routed_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.n_routed_experts, dtype=torch.float32)
            )

        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Mark the gate for custom initialization in param_init_fns.py
        self.gate._is_moe_gate = True
        self.gate._renorm_router_weights = self.renorm_router_weights

        # Initialize weights and apply final processing
        self._batch_counter = 0
        self._upd_counter = 0
        self._tokens_per_layer_and_expert: Optional[torch.Tensor] = None
        self._top_k_scores_expert_sum: Optional[torch.Tensor] = None
        self._top_k_unnormed_scores_expert_sum: Optional[torch.Tensor] = None
        self._top_k_unnormed_biased_scores_expert_sum: Optional[torch.Tensor] = None
        self._scores_per_layer_and_expert: Optional[torch.Tensor] = None
        self._entropy_per_layer: Optional[torch.Tensor] = None
        self._z_loss_router: Optional[torch.Tensor] = None

        # TODO (Sbr): maybe need to move router monitor init to meta too,
        # as for now it is done in any case and initialized values are put
        # on current cuda device from the start
        self.init_router_monitor()

        # check recomputation state for activation checkpointing
        self._is_recompute = False

        if hasattr(self.config, "pad_token_id") and config.pad_token_id is not None:
            self._pad_token_id = int(config.pad_token_id)
        else:
            self._pad_token_id = 0

        # router z-loss
        self.router_z_loss_eps = config.router_z_loss_eps
        if self.router_z_loss_eps > 0:
            self.router_logits_zloss = True
        else:
            self.router_logits_zloss = False

        if hasattr(config, "balance_without_prefix") and config.balance_without_prefix is True:
            self._balance_without_prefix = True
        else:
            self._balance_without_prefix = False

    def _maintain_float32_expert_bias(self):
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def _get_balance_mask(
        self,
        padding_mask: Optional[torch.Tensor],
        prefix_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self._balance_without_prefix or prefix_mask is None:
            return padding_mask
        if padding_mask is None:
            return prefix_mask
        return padding_mask * prefix_mask


    def init_router_monitor(self):
        # TODO (Sbr): this works because we move our models to cuda but it would
        # be right to generalize the function and put monitors to `self.weight.device`.
        # This way we would need to be sure that the weight is on the proper device
        # by the time we call this. Another way is to put on cpu and then move when
        # needed to an appropriate device.
        self._activate_router_monitor = False
        self._tokens_per_layer_and_expert = torch.zeros(
            (1, self.n_routed_experts),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._top_k_scores_expert_sum = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._top_k_unnormed_scores_expert_sum = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._top_k_unnormed_biased_scores_expert_sum = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._scores_per_layer_and_expert = torch.zeros(
            (1, self.n_routed_experts),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._entropy_per_layer = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._z_loss_router = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        self._seq_aux_loss = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            requires_grad=False,
            device=torch.cuda.current_device(),
        )
        
        self._batch_counter = 0
        self._upd_counter = 0

    def set_is_recompute_state(self, v: bool):
        self._is_recompute = v

    def activate_router_monitor(self):
        self._activate_router_monitor = True

    def reset_router_monitor(self):
        torch.nn.init.zeros_(self._tokens_per_layer_and_expert)
        torch.nn.init.zeros_(self._top_k_scores_expert_sum)
        torch.nn.init.zeros_(self._top_k_unnormed_scores_expert_sum)
        torch.nn.init.zeros_(self._top_k_unnormed_biased_scores_expert_sum)
        torch.nn.init.zeros_(self._scores_per_layer_and_expert)
        torch.nn.init.zeros_(self._entropy_per_layer)
        torch.nn.init.zeros_(self._z_loss_router)
        torch.nn.init.zeros_(self._seq_aux_loss)
        self._batch_counter = 0
        self._upd_counter = 0
        self._activate_router_monitor = False

    def update_router_monitor(
        self,
        tokens_per_layer_and_expert,
        batch_top_k_scores_expert_sum,
        batch_top_k_unnormed_scores_expert_sum,
        batch_top_k_unnormed_biased_scores_expert_sum,
        scores,
        z_loss_router,
        seq_aux_loss,
    ):
        assert tokens_per_layer_and_expert.shape[-1] == self.n_routed_experts
        self._batch_counter += batch_top_k_scores_expert_sum.shape[0]
        self._upd_counter += 1
        self._tokens_per_layer_and_expert.add_(tokens_per_layer_and_expert) #.detach()
        self._top_k_scores_expert_sum.add_(batch_top_k_scores_expert_sum.sum()) #.detach()
        self._top_k_unnormed_scores_expert_sum.add_(batch_top_k_unnormed_scores_expert_sum.sum())
        self._top_k_unnormed_biased_scores_expert_sum.add_(batch_top_k_unnormed_biased_scores_expert_sum.sum())
        self._scores_per_layer_and_expert.add_(scores.mean(dim=0))
        entropy = -scores * (scores + 1e-12).log()
        self._entropy_per_layer.add_(entropy.sum(dim=1).mean(dim=0))
        self._z_loss_router.add_(z_loss_router)
        self._seq_aux_loss.add_(seq_aux_loss)

    def get_router_monitor(self):
        router_stats = {
            "tokens_per_layer_and_expert": self._tokens_per_layer_and_expert,
            "mean_top_k_scores_sum": self._top_k_scores_expert_sum / self._batch_counter,
            "mean_top_k_unnormed_scores_sum": self._top_k_unnormed_scores_expert_sum / self._batch_counter,
            "mean_top_k_unnormed_biased_scores_sum": self._top_k_unnormed_biased_scores_expert_sum / self._batch_counter,
            "scores_per_layer_and_expert": self._scores_per_layer_and_expert / self._upd_counter,
            "entropy_per_layer_and_expert": self._entropy_per_layer / self._upd_counter,
            "expert_biases": self.expert_bias, # TODO: add cumulative value for expert_bias during logging step
            "z_loss_router": self._z_loss_router / self._upd_counter,
            "seq_aux_loss": self._seq_aux_loss / self._upd_counter,
        }
        return router_stats

    def group_limited_topk(self, scores: torch.Tensor, num_tokens: int):
        assert self.n_routed_experts % self.num_groups == 0, \
            f"total_experts ({self.n_routed_experts}) must be divisible by num_groups ({self.num_groups})"

        assert self.group_topk <= self.num_groups, "group_topk cannot exceed num_groups"

        group_scores = (
            scores.view(num_tokens, self.num_groups, -1).topk(self.top_k // self.group_topk, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.group_topk, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.num_groups, self.n_routed_experts // self.num_groups)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        return torch.topk(masked_scores, k=self.top_k, dim=-1)[1]

    # fill zero all padding tokens in routing map and probs

    def _mask_pad_tokens(self, scores, padding_mask):
        if padding_mask is None:
            return scores
        padding_mask_flat = padding_mask.view(-1, 1)
        scores_res = scores * padding_mask_flat
        return scores_res

    def _update_monitor_from_topk(
        self,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        batch_size: int,
        seq_len: int,
        token_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.training or not torch.is_grad_enabled() or self._is_recompute:
            return
        if not self._activate_router_monitor:
            return

        with torch.no_grad():
            flat_topk_idx = topk_idx.reshape(-1, self.top_k)
            flat_topk_weight = topk_weight.reshape(-1, self.top_k).to(torch.float32)

            if token_mask is not None:
                weights = token_mask.unsqueeze(-1).expand(-1, -1, self.top_k).reshape(batch_size, -1).float()
            else:
                weights = torch.ones(batch_size, seq_len * self.top_k, device=flat_topk_idx.device, dtype=torch.float32)

            ce_router_monitor = torch.zeros(
                batch_size,
                self.n_routed_experts,
                device=flat_topk_idx.device,
                dtype=torch.float32,
            )
            ce_router_monitor.scatter_add_(1, flat_topk_idx.reshape(batch_size, -1), weights)
            tokens_per_layer_and_expert = ce_router_monitor.sum(dim=0)

            scores_for_router_monitor = torch.zeros(
                batch_size * seq_len,
                self.n_routed_experts,
                device=flat_topk_idx.device,
                dtype=torch.float32,
            )
            scores_for_router_monitor.scatter_add_(1, flat_topk_idx, flat_topk_weight)

            zeros_scalar = torch.zeros((), dtype=torch.float32, device=flat_topk_idx.device)
            topk_scores_sum = self._mask_pad_tokens(flat_topk_weight, token_mask).sum(dim=-1)
            self.update_router_monitor(
                tokens_per_layer_and_expert=tokens_per_layer_and_expert.detach().clone(),
                batch_top_k_scores_expert_sum=topk_scores_sum.detach().clone(),
                batch_top_k_unnormed_scores_expert_sum=topk_scores_sum.detach().clone(),
                batch_top_k_unnormed_biased_scores_expert_sum=topk_scores_sum.detach().clone(),
                scores=scores_for_router_monitor.detach().clone(),
                z_loss_router=zeros_scalar,
                seq_aux_loss=zeros_scalar,
            )

    @decorator_forward_backward()
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_sparse_outputs: bool = False,
        logical_batch_size: int | None = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._maintain_float32_expert_bias()
        bsz, seq_len, _ = hidden_states.shape
        balance_mask = self._get_balance_mask(padding_mask, prefix_mask)

        device_type = hidden_states.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, dtype=torch.float32):
            if self.tp_size > 1:
                logits = self.gate(
                    hidden_states,
                    logical_batch_size=logical_batch_size,
                    no_recompute=True,
                )
                logits = distribute_async_tp_embeddings(logits)
            else:
                global_actlog_state.use_monitor_variable(hidden_states, "model.model.layers.{}.self_attn._gate.gate", "_input.0")

                logits = self.gate(hidden_states)

                global_actlog_state.use_monitor_variable(logits, "model.model.layers.{}.self_attn._gate.gate", "_output.0")

            flat_logits = logits.view(-1, self.n_routed_experts)

            if self.training and self.router_logits_zloss:
                tp_sp_group = dist.get_tp_sp_group()
                tp_sp_group_size = dist.get_tp_sp_group_size() or 1
                z_loss_flat_logits = z_loss_fn(flat_logits, eps=self.router_z_loss_eps)
                z_loss_logits = z_loss_flat_logits.reshape(bsz, seq_len)
                if balance_mask is not None:
                    z_loss_logits = z_loss_logits * balance_mask
                    z_loss_logits_seq_sum = z_loss_logits.sum(1).clone()
                    tokens_no_pad = balance_mask.sum(1).detach().clone()

                    if tp_sp_group is not None:
                        tokens_no_pad = all_reduce_from_tensor_sequence_parallel_region(tokens_no_pad)
                        z_loss_logits_seq_sum = all_reduce_from_tensor_sequence_parallel_region(z_loss_logits_seq_sum)

                    z_loss_router = (z_loss_logits_seq_sum/tokens_no_pad).mean()
                else:
                    z_loss_router = z_loss_flat_logits.mean()
                    if tp_sp_group is not None:
                        z_loss_router /= tp_sp_group_size
                        z_loss_router = all_reduce_from_tensor_sequence_parallel_region(z_loss_router)
            else:
                z_loss_router  = None

            if z_loss_router is not None:
                flat_logits = AddRouterZLoss.apply(flat_logits, z_loss_router)
            else:
                z_loss_router = torch.tensor(0)


            if self.scoring_func == 'softmax':
                scores = F.softmax(flat_logits, dim=-1, dtype=torch.float32)
            elif self.scoring_func == 'sigmoid':
                scores = flat_logits.to(torch.float32).sigmoid()
            else:
                raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

            orig_scores = scores

            if self.expert_bias is not None:
                scores = orig_scores + self.expert_bias

        ### select top-k experts
        if self.group_topk:
            topk_idx = self.group_limited_topk(scores, bsz * seq_len)
        else:
            _, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = torch.gather(orig_scores, dim=-1, index=topk_idx)

        ### extra statistics for monitoring
        topk_weight_unnormed = topk_weight.clone()
        if self.expert_bias is not None:
            topk_weight_unnormed_biased = torch.gather(scores, dim=-1, index=topk_idx)
        else:
            topk_weight_unnormed_biased = topk_weight.clone()

        with torch.autocast(device_type=device_type, dtype=torch.float32):
            ### norm gate to sum 1
            if self.top_k > 1 and self.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator


        ### expert-level computation auxiliary loss
        if self.training:
            tp_sp_group = dist.get_tp_sp_group()
            tp_sp_group_size = dist.get_tp_sp_group_size() or 1
            with torch.autocast(device_type=device_type, dtype=torch.float32):
                aux_topk = self.top_k

                if balance_mask is not None:
                    weights = balance_mask.unsqueeze(-1).expand(-1, -1, aux_topk).reshape(bsz, -1).float()
                else:
                    weights = torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)

                scores_for_aux = orig_scores
                # always compute aux loss based on the naive greedy topk method
                if self.aux_loss_free:
                    # it needs to be done without bias and without scaling factor
                    _, topk_idx_for_aux_loss = torch.topk(scores_for_aux, k=self.top_k, dim=-1, sorted=False)
                else:
                    topk_idx_for_aux_loss = topk_idx

                topk_idx_for_aux_loss = topk_idx_for_aux_loss.view(bsz, -1) # [DEBUG] (bsz, seq_len*top_k)

                if self.seq_aux:
                    if self.alpha > 0:
                        assert self.tp_size == 1, "Seq aux loss is not tested yet with tp_size > 1"
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # [DEBUG] (bsz, seq_len, top_k)

                    if self.scoring_func == 'sigmoid':
                        denominator = scores_for_seq_aux.sum(dim=-1, keepdim=True)
                        denominator += 1e-20
                        scores_for_seq_aux = scores_for_seq_aux / denominator

                    # mask padding tokens at
                    if balance_mask is not None:
                        padding_mask_unsqueezed = balance_mask.unsqueeze(-1)
                        seq_len_no_pad = padding_mask_unsqueezed.sum(dim=1).detach().clone()
                        scores_for_seq_aux = scores_for_seq_aux * padding_mask_unsqueezed # [DEBUG] padding_mask (bsz, seq_len) -> (bsz, seq_len, 1)
                        scores_for_seq_aux_sum = scores_for_seq_aux.sum(dim=1).clone()

                        if tp_sp_group is not None:
                            seq_len_no_pad = all_reduce_from_tensor_sequence_parallel_region(seq_len_no_pad)
                            scores_for_seq_aux_sum = all_reduce_from_tensor_sequence_parallel_region(scores_for_seq_aux_sum)

                        scores_for_seq_aux_mean = scores_for_seq_aux_sum/seq_len_no_pad
                    else:
                        scores_for_seq_aux_mean = scores_for_seq_aux.mean(dim=1).clone()
                        seq_len_no_pad = seq_len.detach().clone()
                            
                        if tp_sp_group is not None:
                            seq_len_no_pad = all_reduce_from_tensor_sequence_parallel_region(seq_len_no_pad)
                            scores_for_seq_aux_mean = all_reduce_from_tensor_sequence_parallel_region(scores_for_seq_aux_mean)
    
                            scores_for_seq_aux_mean /= tp_sp_group_size

                    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device, dtype=torch.float32)
                    ce.scatter_add_(1, topk_idx_for_aux_loss, weights)

                    if tp_sp_group is not None:
                        ce = all_reduce_from_tensor_sequence_parallel_region(ce)

                    ce.div_(seq_len_no_pad * aux_topk / self.n_routed_experts)
                    aux_loss = (ce * scores_for_seq_aux_mean).sum(dim=1).mean() * self.alpha
                else:
                    aux_loss = 0
                    raise NotImplementedError('seq_aux=False is not supported')

                if torch.is_grad_enabled() and not self._is_recompute:
                    with torch.no_grad():
                        scores_for_router_monitor = scores.view(bsz, seq_len, -1)
                        ce_router_monitor = torch.zeros(
                            bsz, self.n_routed_experts,
                            device=hidden_states.device, dtype=torch.float32
                        )

                        ce_router_monitor.scatter_add_(
                            1, topk_idx.view(bsz, -1), weights
                        )

                        tokens_per_layer_and_expert = ce_router_monitor.sum(dim=0)

                        if self._activate_router_monitor:
                            self.update_router_monitor(
                                tokens_per_layer_and_expert=tokens_per_layer_and_expert.detach().clone(),
                                batch_top_k_scores_expert_sum=self._mask_pad_tokens(topk_weight, balance_mask).sum(dim=-1).detach().clone(),
                                batch_top_k_unnormed_scores_expert_sum=self._mask_pad_tokens(topk_weight_unnormed, balance_mask).sum(dim=-1).detach().clone(),
                                batch_top_k_unnormed_biased_scores_expert_sum=self._mask_pad_tokens(topk_weight_unnormed_biased, balance_mask).sum(dim=-1).detach().clone(),
                                scores=scores_for_router_monitor[0].detach().clone(),
                                z_loss_router=z_loss_router.detach().clone(),
                                seq_aux_loss=aux_loss.detach().clone()
                            )

                    if self.aux_loss_free:
                        self.local_tokens_per_expert.add_(tokens_per_layer_and_expert)

        else:
            aux_loss = None

        with torch.autocast(device_type=device_type, dtype=torch.float32):
            topk_weight = topk_weight * self.routed_scaling_factor

        if return_sparse_outputs:
            logits = logits.flatten(0, -2)
            topk_weight = torch.zeros_like(logits, dtype=topk_weight.dtype).scatter(1, topk_idx, topk_weight)
            topk_idx = torch.zeros_like(logits).int().scatter(1, topk_idx, 1).bool()

        return topk_idx, topk_weight, aux_loss


class UniformGate(MoEGate):  # TODO: Add a base MoE class with all necessary methods for third-party classes and inherit from it
    """
    Deterministic/stochastic round-robin gate for performance benchmarking.

    Two routing modes, selected via config.moe_uniform_mode:

      * "random" (default): per-token uniform random sample of top_k experts
        (without replacement). The most neutral baseline — no structural bias,
        approximates an untrained/fully-random router. Expected expert load is
        total_tokens * top_k / N per expert (uniform in expectation, with
        sqrt-scale variance).

      * "cross_rank": K copies on K distinct non-local ranks (skip-base).
        Requires top_k <= ep_size - 1. Upper bound on cross-rank comm — zero
        local traffic, but touches only one expert per non-local rank. Uses
        a per-rank base shift (rank * experts_per_rank) so different EP ranks
        produce different dispatch patterns.

    Config:
      * moe_uniform_mode: str in {"random", "cross_rank"}, default "random"
    """

    _VALID_MODES = ("random", "cross_rank")

    def __init__(self, config, layer_idx: int = 0) -> None:
        super().__init__(config, layer_idx)
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.gate_type = UNIFORM_GATE_TYPE

        assert self.top_k < self.num_routed_experts, (
            f"top_k ({self.top_k}) must be < num_routed_experts ({self.num_routed_experts})"
        )

        # Resolve mode. Keep backward-compat with the old bool flag.
        mode = getattr(config, "moe_uniform_mode", None)
        if mode is None:
            mode = "random"
        assert mode in self._VALID_MODES, (
            f"moe_uniform_mode must be one of {self._VALID_MODES}, got {mode!r}"
        )
        self.mode = mode
        self.ep_size = get_ep_group_size() or 1

        if self.mode == "cross_rank":
            assert self.top_k <= self.ep_size - 1, (
                f"mode='cross_rank' requires top_k ({self.top_k}) <= ep_size - 1 ({self.ep_size - 1})"
            )
            assert self.num_routed_experts % self.ep_size == 0, (
                f"num_routed_experts ({self.num_routed_experts}) must be divisible by ep_size ({self.ep_size})"
            )

    def _resolve_rank(self) -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def _route_random(self, total_tokens: int, device: torch.device) -> torch.Tensor:
        # Uniform random top_k without replacement. topk over random scores is
        # cheap and vectorized; avoids Python-level loops and multinomial overhead.
        scores = torch.rand(total_tokens, self.num_routed_experts, device=device)
        return scores.topk(self.top_k, dim=-1).indices  # [T, K]

    def _route_cross_rank(self, total_tokens: int, device: torch.device) -> torch.Tensor:
        rank = self._resolve_rank()
        experts_per_rank = self.num_routed_experts // self.ep_size
        base_shift = rank * experts_per_rank

        base_expert = (torch.arange(total_tokens, device=device) + base_shift) % self.num_routed_experts  # [T]
        step = experts_per_rank
        offsets = torch.arange(1, self.top_k + 1, device=device) * step  # skip base rank
        return (base_expert.view(-1, 1) + offsets.view(1, -1)) % self.num_routed_experts  # [T, K]

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_sparse_outputs: bool = False,
        logical_batch_size: int | None = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = hidden_states.shape[:-1]
        total_tokens = batch_size * seq_len
        device = hidden_states.device
        K = self.top_k
        N = self.num_routed_experts
        balance_mask = self._get_balance_mask(padding_mask, prefix_mask)

        if self.mode == "random":
            topk_idx = self._route_random(total_tokens, device)
        elif self.mode == "cross_rank":
            topk_idx = self._route_cross_rank(total_tokens, device)
        else:  # pragma: no cover — guarded in __init__
            raise ValueError(f"Unknown moe_uniform_mode: {self.mode!r}")

        topk_idx = topk_idx.view(batch_size, seq_len, K)

        topk_weight = torch.full(
            (batch_size, seq_len, K),
            fill_value=1.0 / K,
            device=device,
            dtype=hidden_states.dtype,
        )

        self._update_monitor_from_topk(
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            batch_size=batch_size,
            seq_len=seq_len,
            token_mask=balance_mask,
        )

        if return_sparse_outputs:
            flat_tokens = total_tokens
            topk_idx_flat = topk_idx.view(-1, K)
            topk_weight_flat = topk_weight.view(-1, K)

            sparse_topk_weight = torch.zeros(
                (flat_tokens, N),
                dtype=topk_weight.dtype,
                device=device,
            )
            sparse_topk_idx = torch.zeros(
                (flat_tokens, N),
                dtype=torch.bool,
                device=device,
            )
            sparse_topk_weight.scatter_(1, topk_idx_flat, topk_weight_flat)
            sparse_topk_idx.scatter_(1, topk_idx_flat, True)

            return sparse_topk_idx, sparse_topk_weight, None

        return topk_idx, topk_weight, None


class FirstExpertGroupGate(MoEGate): # TODO: Add a base MoE class with all necessary methods for third-party classes and inherit from it
    def __init__(self, config, layer_idx: int = 0) -> None:
        super().__init__(config, layer_idx)
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self._activate_router_monitor = False
        self.ep_size = get_ep_group_size() or 1

        self.gate_type = FIRST_EXPERT_GROUP_GATE_TYPE

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_sparse_outputs: bool = False,
        logical_batch_size: int | None = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = hidden_states.shape[:-1]
        total_tokens = batch_size * seq_len
        n_experts = self.num_routed_experts
        device = hidden_states.device
        balance_mask = self._get_balance_mask(padding_mask, prefix_mask)

        topk_idx = torch.arange(self.top_k, device=device).view(1, 1, self.top_k).expand(batch_size, seq_len, self.top_k)
        topk_weight = torch.full(
            (batch_size, seq_len, self.top_k),
            fill_value=1.0 / self.top_k,
            device=device,
            dtype=hidden_states.dtype,
        )

        self._update_monitor_from_topk(
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            batch_size=batch_size,
            seq_len=seq_len,
            token_mask=balance_mask,
        )

        if return_sparse_outputs:
            sparse_topk_weight = torch.zeros((total_tokens, n_experts), dtype=topk_weight.dtype, device=device)
            sparse_topk_idx = torch.zeros((total_tokens, n_experts), dtype=torch.bool, device=device)
            flat_topk_idx = topk_idx.reshape(-1, self.top_k)
            flat_topk_weight = topk_weight.reshape(-1, self.top_k)
            sparse_topk_weight.scatter_(1, flat_topk_idx, flat_topk_weight)
            sparse_topk_idx.scatter_(1, flat_topk_idx, True)
            return sparse_topk_idx, sparse_topk_weight, None

        return topk_idx, topk_weight, None

class DynamicGate(MoEGate):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self._gate_types = nn.ModuleDict({
            "first_group": FirstExpertGroupGate(config),
            "uniform": UniformGate(config),
        })

        self.gate_type = DYNAMIC_GATE_TYPE

        self._current_gate_type = None
        self.set_first_group_gate()

        assert config.first_group_steps is not None, "Set up number of steps before switch from FirstExpertGroupGate to UniformGate"
        self._first_group_steps = config.first_group_steps

    @property
    def _current_gate(self) -> Optional[nn.Module]:
        if self._current_gate_type is None:
            return None
        return self._gate_types[self._current_gate_type]

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_sparse_outputs: bool = False,
        logical_batch_size: int | None = None,
        padding_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        return self._current_gate.forward(
                        hidden_states,
                        return_sparse_outputs,
                        logical_batch_size,
                        padding_mask,
                        prefix_mask,
                    )
    def set_first_group_gate(self):
        self._current_gate_type = "first_group"
        self._current_gate.set_is_recompute_state(self._is_recompute)
        if self._activate_router_monitor:
            self._current_gate.activate_router_monitor()

    def set_uniform_gate(self):
        self._current_gate_type = "uniform"
        self._current_gate.set_is_recompute_state(self._is_recompute)
        if self._activate_router_monitor:
            self._current_gate.activate_router_monitor()

    def switch2uniform(self, curr_step:int=0):
        if self._current_gate_type != "uniform" and curr_step >= self._first_group_steps:
            self.set_uniform_gate()

    def init_router_monitor(self):
        if not hasattr(self, "_gate_types"):
            MoEGate.init_router_monitor(self)
            return
        for gate in self._gate_types.values():
            gate.init_router_monitor()
        self._batch_counter = 0
        self._upd_counter = 0
        self._activate_router_monitor = False

    def set_is_recompute_state(self, v: bool):
        self._is_recompute = v
        if hasattr(self, "_gate_types"):
            for gate in self._gate_types.values():
                gate.set_is_recompute_state(v)

    def activate_router_monitor(self):
        self._activate_router_monitor = True
        if hasattr(self, "_gate_types"):
            for gate in self._gate_types.values():
                gate.activate_router_monitor()

    def reset_router_monitor(self):
        self._batch_counter = 0
        self._upd_counter = 0
        self._activate_router_monitor = False
        if hasattr(self, "_gate_types"):
            for gate in self._gate_types.values():
                gate.reset_router_monitor()

    def update_router_monitor(
        self,
        tokens_per_layer_and_expert,
        batch_top_k_scores_expert_sum,
        batch_top_k_unnormed_scores_expert_sum,
        batch_top_k_unnormed_biased_scores_expert_sum,
        scores,
        z_loss_router,
        seq_aux_loss,
    ):
        if self._current_gate is not None:
            self._current_gate.update_router_monitor(
                tokens_per_layer_and_expert=tokens_per_layer_and_expert,
                batch_top_k_scores_expert_sum=batch_top_k_scores_expert_sum,
                batch_top_k_unnormed_scores_expert_sum=batch_top_k_unnormed_scores_expert_sum,
                batch_top_k_unnormed_biased_scores_expert_sum=batch_top_k_unnormed_biased_scores_expert_sum,
                scores=scores,
                z_loss_router=z_loss_router,
                seq_aux_loss=seq_aux_loss,
            )

    def get_router_monitor(self):
        if not hasattr(self, "_gate_types") or len(self._gate_types) == 0:
            return MoEGate.get_router_monitor(self)

        gates = list(self._gate_types.values())
        tokens_per_layer_and_expert = gates[0]._tokens_per_layer_and_expert.detach().clone()
        top_k_scores_expert_sum = gates[0]._top_k_scores_expert_sum.detach().clone()
        top_k_unnormed_scores_expert_sum = gates[0]._top_k_unnormed_scores_expert_sum.detach().clone()
        top_k_unnormed_biased_scores_expert_sum = gates[0]._top_k_unnormed_biased_scores_expert_sum.detach().clone()
        scores_per_layer_and_expert = gates[0]._scores_per_layer_and_expert.detach().clone()
        entropy_per_layer = gates[0]._entropy_per_layer.detach().clone()
        z_loss_router = gates[0]._z_loss_router.detach().clone()
        seq_aux_loss = gates[0]._seq_aux_loss.detach().clone()
        batch_counter = gates[0]._batch_counter
        upd_counter = gates[0]._upd_counter

        for gate in gates[1:]:
            tokens_per_layer_and_expert.add_(gate._tokens_per_layer_and_expert)
            top_k_scores_expert_sum.add_(gate._top_k_scores_expert_sum)
            top_k_unnormed_scores_expert_sum.add_(gate._top_k_unnormed_scores_expert_sum)
            top_k_unnormed_biased_scores_expert_sum.add_(gate._top_k_unnormed_biased_scores_expert_sum)
            scores_per_layer_and_expert.add_(gate._scores_per_layer_and_expert)
            entropy_per_layer.add_(gate._entropy_per_layer)
            z_loss_router.add_(gate._z_loss_router)
            seq_aux_loss.add_(gate._seq_aux_loss)
            batch_counter += gate._batch_counter
            upd_counter += gate._upd_counter

        batch_denom = max(batch_counter, 1)
        upd_denom = max(upd_counter, 1)
        expert_biases = self._current_gate.expert_bias if self._current_gate is not None else None
        return {
            "tokens_per_layer_and_expert": tokens_per_layer_and_expert,
            "mean_top_k_scores_sum": top_k_scores_expert_sum / batch_denom,
            "mean_top_k_unnormed_scores_sum": top_k_unnormed_scores_expert_sum / batch_denom,
            "mean_top_k_unnormed_biased_scores_sum": top_k_unnormed_biased_scores_expert_sum / batch_denom,
            "scores_per_layer_and_expert": scores_per_layer_and_expert / upd_denom,
            "entropy_per_layer_and_expert": entropy_per_layer / upd_denom,
            "expert_biases": expert_biases,
            "z_loss_router": z_loss_router / upd_denom,
            "seq_aux_loss": seq_aux_loss / upd_denom,
        }


GATE_CLASS_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    TOPK_GATE_TYPE: MoEGate,
    UNIFORM_GATE_TYPE: UniformGate,
    FIRST_EXPERT_GROUP_GATE_TYPE: FirstExpertGroupGate,
    DYNAMIC_GATE_TYPE: DynamicGate,
}
