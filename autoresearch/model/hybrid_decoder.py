"""Immutable Transformer/GDN decoder topology for architecture research."""

from __future__ import annotations

from typing import Any

from llmfoundry.models.layers.blocks import BLOCK_CLASS_REGISTRY, LlamaDecoderLayer

from autoresearch.model.gdn import AutoresearchGatedDeltaNet

GDN_LAYER_INDICES = frozenset(range(0, 16, 2))


class AutoresearchHybridDecoderLayer(LlamaDecoderLayer):
    """Use fixed Transformer layers and the mutable GDN on configured layers."""

    def __init__(
        self,
        config: Any,
        layer_idx: int = 0,
        **candidate_kwargs: Any,
    ) -> None:
        if candidate_kwargs:
            unknown = ", ".join(sorted(candidate_kwargs))
            raise TypeError(f"Unknown hybrid decoder arguments: {unknown}")

        # Gigar's dense configuration intentionally keeps linear_attention_type
        # unset. Build its standard Transformer layer first, then replace only
        # the attention module on the fixed GDN layer indices.
        super().__init__(config=config, layer_idx=layer_idx)
        if layer_idx in GDN_LAYER_INDICES:
            self.self_attn = AutoresearchGatedDeltaNet(
                config=config,
                layer_idx=layer_idx,
            )


BLOCK_CLASS_REGISTRY["AutoresearchHybridDecoderLayer"] = AutoresearchHybridDecoderLayer
