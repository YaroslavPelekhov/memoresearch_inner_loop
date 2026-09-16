# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from composer.utils import dist, global_actlog_state

from llmfoundry.models.layers.activations import ACT2FN
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.parallel.tensor import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region)

from composer.utils import dist, global_actlog_state

try:
    import transformer_engine.pytorch as te
except:
    te = None

log = logging.getLogger(__name__)


class ParallelMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        intermediate_size=None,
        fused_mlp_checkpoint_lvl=None,
        layer_idx: int | None = None,
    ):
        super(ParallelMLP, self).__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.add_bias = False

        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            self.intermediate_size,
            config=config,
            bias=self.add_bias,
            gather_output=False,
        )
        self.up_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            self.intermediate_size,
            config=config,
            bias=self.add_bias,
            gather_output=False,
        )

        self._is_fused = config.fused_mlp
        if self._is_fused:
            assert not self.add_bias, "bias are not supported"
            from llmfoundry.models.ops.mlp import fused_mlp_func

            self._fused_func = fused_mlp_func

        self._checkpoint_lvl = (
            fused_mlp_checkpoint_lvl or config.fused_mlp_checkpoint_lvl
        )

        def swiglu(gate_proj, up_proj):
            return torch.nn.functional.silu(gate_proj) * up_proj

        self.activation_func = swiglu

        self.down_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
            self.intermediate_size,
            config.hidden_size,
            config=config,
            bias=self.add_bias,
            input_is_parallel=True,
        )

        self.down_proj._is_residual = True

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.gate_proj._init_std = init_std
            self.up_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = (
                        1
                        / math.sqrt(self.intermediate_size)
                        / math.sqrt(2 * (self.layer_idx + 1))
                    )
                else:
                    log.info(
                        "`ParallelMLP` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `down_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.down_proj._init_std = init_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        activation_checkpointing_on_layer: bool = False,
        logical_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._is_fused:
            input_parallel = copy_to_tensor_model_parallel_region(hidden_states)
            output_parallel = self._fused_func(
                x=input_parallel,
                weight1=self.gate_proj._tp_linear_submodule.weight,
                weight2=self.up_proj._tp_linear_submodule.weight,
                weight3=self.down_proj._tp_linear_submodule.weight,
                # we fully recompute the activations, so no need to checkpoint them
                checkpoint_lvl=0
                if activation_checkpointing_on_layer
                else self._checkpoint_lvl,
            )
            output = reduce_from_tensor_model_parallel_region(output_parallel)
            return output

        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )

        global_actlog_state.use_monitor_variable(
            hidden_states, "model.model.layers.{}.mlp.up_proj", "_input.0"
        )
        if self._checkpoint_lvl and not activation_checkpointing_on_layer:
            output = checkpoint.checkpoint(
                self.swiglu_fn, hidden_states, tp_kwargs, use_reentrant=False
            )
        else:
            output = self.swiglu_fn(hidden_states, tp_kwargs)

        return output

    def swiglu_fn(self, x, tp_kwargs):
        x1 = self.gate_proj(x, **tp_kwargs)
        x2 = self.up_proj(x, **tp_kwargs)
        return self.down_proj(self.activation_func(x1, x2))


class LlamaMLP(nn.Module):
    def __init__(
        self,
        config,
        intermediate_size=None,
        fused_mlp_checkpoint_lvl=None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=config.init_device,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=config.init_device,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            device=config.init_device,
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self._is_fused = config.fused_mlp
        if self._is_fused:
            from llmfoundry.models.ops.mlp import fused_mlp_func

            self._fused_func = fused_mlp_func

        self._sep_product_silu_quant = None
        self._swiglu_limit = getattr(config, "swiglu_limit", 0.0)
        if getattr(config, "float8_dense_fused_swiglu_quant", False):
            from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization_dense import SepProductSiluQuant
            self._sep_product_silu_quant = SepProductSiluQuant
        if self._swiglu_limit > 0:
            self.register_buffer(
                'clip_counts', torch.zeros(2, dtype=torch.int32), persistent=False
            )

        self._checkpoint_lvl = fused_mlp_checkpoint_lvl or config.fused_mlp_checkpoint_lvl
        self.down_proj._is_residual = True

        self.down_proj._is_residual = True

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.gate_proj._init_std = init_std
            self.up_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = (
                        1
                        / math.sqrt(self.intermediate_size)
                        / math.sqrt(2 * (self.layer_idx + 1))
                    )
                else:
                    log.info(
                        "`LlamaMLP` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `down_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.down_proj._init_std = init_std

    def forward(self, x, residual=None, activation_checkpointing_on_layer=False):
        if getattr(self.config, "float8_dense_fused_swiglu_quant", False):
            # assert not self._checkpoint_lvl > 0, "checkpoint_lvl is not implemented for FP8"
            assert self._is_fused is False, "fused_mlp=True and float8_dense_fused_swiglu_quant=True not compatible"
            down_proj = self.float8_dense_fused_swiglu_quant_fn(x, residual)
            return down_proj

        if self._is_fused:
            assert self._swiglu_limit <= 0, (
                "fused_mlp=True and swiglu_limit>0 are not compatible: "
                "the fused MLP kernel does not support SwiGLU activation clipping"
            )
            return self._fused_func(
                x=x,
                weight1=self.gate_proj.weight,
                weight2=self.up_proj.weight,
                weight3=self.down_proj.weight,
                # we fully recompute the activations, so no need to checkpoint them
                checkpoint_lvl=0
                if activation_checkpointing_on_layer
                else self._checkpoint_lvl,
            )

        monitor_clips = global_actlog_state.enable_monitor and self._swiglu_limit > 0 and hasattr(self, 'clip_counts')
        if monitor_clips:
            self.clip_counts.zero_()

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            if self._swiglu_limit > 0:
                if hasattr(self, 'clip_counts'):
                    self.clip_counts[0].add_(gate_proj.ge(self._swiglu_limit).sum().to(torch.int32))
                    self.clip_counts[1].add_(
                        up_proj.le(-self._swiglu_limit).logical_or(up_proj.ge(self._swiglu_limit)).sum().to(torch.int32))
                gate_proj = gate_proj.clamp(max=self._swiglu_limit)
                up_proj = up_proj.clamp(-self._swiglu_limit, self._swiglu_limit)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
            clip_total = gate_proj.numel()
        else:
            global_actlog_state.use_monitor_variable(
                x, "model.model.layers.{}.mlp.up_proj", "_input.0"
            )
            if self._checkpoint_lvl and not activation_checkpointing_on_layer:
                down_proj = checkpoint.checkpoint(
                    self.swiglu_fn, x, use_reentrant=False
                )
            else:
                down_proj = self.swiglu_fn(x)
            clip_total = x.shape[0] * self.intermediate_size

        if monitor_clips:
            total = float(max(clip_total, 1))
            gate_share = self.clip_counts[0].float() / total
            up_share = self.clip_counts[1].float() / total
            global_actlog_state.use_monitor_variable(
                self.clip_counts[0].float().unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_gate_clip_count")
            global_actlog_state.use_monitor_variable(
                gate_share.unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_gate_clip_share")
            global_actlog_state.use_monitor_variable(
                self.clip_counts[1].float().unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_up_clip_count")
            global_actlog_state.use_monitor_variable(
                up_share.unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_up_clip_share")

        return down_proj

    def swiglu_fn(self, x):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        if self._swiglu_limit > 0:
            if hasattr(self, 'clip_counts'):
                self.clip_counts[0].add_(gate_out.ge(self._swiglu_limit).sum().to(torch.int32))
                self.clip_counts[1].add_(
                    up_out.le(-self._swiglu_limit).logical_or(up_out.ge(self._swiglu_limit)).sum().to(torch.int32))
            gate_out = gate_out.clamp(max=self._swiglu_limit)
            up_out = up_out.clamp(-self._swiglu_limit, self._swiglu_limit)
        return self.down_proj(self.act_fn(gate_out) * up_out)

    def float8_dense_fused_swiglu_quant_fn(self, x, residual):
        gate_out = self.gate_proj(x, do_unpad_out=False)
        up_out = self.up_proj(x, do_unpad_out=False)

        monitor_clips = global_actlog_state.enable_monitor and self._swiglu_limit > 0
        if monitor_clips:
            self.clip_counts.zero_()

        prod_silu_out = self._sep_product_silu_quant.apply(
            gate_out, up_out,
            torch.float8_e4m3fn,
            True,   # power_two_max_round
            self._swiglu_limit,
            self.clip_counts if monitor_clips else None,
        )

        if monitor_clips:
            # Dense path is single-rank (no EP/TP wrapper around LlamaMLP), so per-rank
            # element count is already the full layer total. clamp_min(1) avoids div-by-0
            # on degenerate empty inputs.
            total = float(max(gate_out.numel(), 1))
            gate_share = self.clip_counts[0].float() / total
            up_share = self.clip_counts[1].float() / total
            global_actlog_state.use_monitor_variable(
                self.clip_counts[0].float().unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_gate_clip_count")
            global_actlog_state.use_monitor_variable(
                gate_share.unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_gate_clip_share")
            global_actlog_state.use_monitor_variable(
                self.clip_counts[1].float().unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_up_clip_count")
            global_actlog_state.use_monitor_variable(
                up_share.unsqueeze(0),
                "model.model.layers.{}.mlp",
                "_up_clip_share")

        return self.down_proj(prod_silu_out, residual=residual)


FFN_CLASS_REGISTRY = {"ParallelMLP": ParallelMLP, "LlamaMLP": LlamaMLP}


def build_ffn(
    tp_size: int,
    config: Any,
    layer_idx: int | None = None,
) -> LlamaMLP | ParallelMLP:
    ffn_type = (
        FFN_CLASS_REGISTRY["LlamaMLP"]
        if tp_size == 1
        else FFN_CLASS_REGISTRY["ParallelMLP"]
    )
    return ffn_type(
        config,
        intermediate_size=config.dense_intermediate_size,
        fused_mlp_checkpoint_lvl=config.dense_fused_mlp_checkpoint_lvl,
        layer_idx=layer_idx,
    )
