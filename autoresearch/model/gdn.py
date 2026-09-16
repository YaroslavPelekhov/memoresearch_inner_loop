"""Mutable Gated DeltaNet implementation.

This is the only model source file changed by idea implementations and the
inner evolution loop. The seed deliberately preserves LLM Foundry's supplied
GDN behavior; research ideas evolve subclasses from this starting point.
"""

from __future__ import annotations

import math
from typing import Any

from llmfoundry.models.layers.attention import Qwen3NextGatedDeltaNet


class AutoresearchGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Baseline GDN used on the fixed hybrid model's recurrent layers."""

    def __init__(self, config: Any, layer_idx: int | None = None) -> None:
        # Dense Gigar exposes the GDN dimensions but not these GigaMix defaults.
        # Keep them here because gate behavior and initialization are part of
        # the mutable research implementation.
        config.linear_gating_type = getattr(
            config,
            "linear_gating_type",
            "gated_rmsnorm",
        )
        config.linear_sigmoid_gate_scale = getattr(
            config,
            "linear_sigmoid_gate_scale",
            1.0,
        )
        config.linear_conv_init_std = getattr(
            config,
            "linear_conv_init_std",
            0.006,
        )
        config.linear_use_legacy_qkvz_layout = getattr(
            config,
            "linear_use_legacy_qkvz_layout",
            False,
        )
        super().__init__(config=config, layer_idx=layer_idx)

        input_std = 1 / math.sqrt(config.hidden_size)
        self.ba_proj._init_std = input_std
        if self.use_legacy_qkvz_layout:
            self.qkvz_proj._init_std = input_std
        else:
            self.qkv_proj._init_std = input_std
            self.z_proj._init_std = input_std

        output_std = input_std
        if config.init_type == "dclm" and layer_idx is not None and layer_idx >= 0:
            output_std /= math.sqrt(2 * (layer_idx + 1))
        elif config.init_type == "olmo3":
            output_std /= math.sqrt(2 * config.num_hidden_layers)
        self.o_proj._init_std = output_std
