"""Attach gigatimers to local MoE forward methods on every rank."""

import re
from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from llmfoundry.models.layers.moe import MOE_CLASS_REGISTRY

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils.timers import TimerType, gigatimer

import logging
logger = logging.getLogger(__name__)

__all__ = ['MoeRanksTiming']

_LAYER_PATTERN = re.compile(r'\.layers\.(\d+)\.')


class _MoeLayerDiscoveryHandler:
    """Find local MoE modules and map them to layer ids."""

    def __init__(self, moe_layer_name: str) -> None:
        self.moe_layer_name: str = moe_layer_name

    def is_moe_module(self, module_name: str, module: torch.nn.Module) -> bool:
        if module.__class__.__name__ == self.moe_layer_name:
            return True
        return module_name.endswith('block_sparse_moe')

    @staticmethod
    def extract_layer_id(module_name: str) -> int:
        """Extract layer id from module"""
        match = _LAYER_PATTERN.search(module_name)
        if match is None:
            raise ValueError(
                f"Could not extract layer id from module name: {module_name!r}.\nExpected pattern like '.layers.<id>.'"
            )
        return int(match.group(1))

    def discover_local_layers(self, model: torch.nn.Module) -> Dict[int, torch.nn.Module]:
        local_layers: Dict[int, torch.nn.Module] = {}
        for module_name, module in model.named_modules():
            if not self.is_moe_module(module_name=module_name, module=module):
                continue
            layer_id = self.extract_layer_id(module_name)
            local_layers[layer_id] = module
        return local_layers


@dataclass
class _MoeForwardWrappingHandler:
    """Manage wrapping of local MoE forward methods."""

    modules_by_layer: Dict[int, torch.nn.Module] = field(default_factory=dict)

    def attach(self, layer_modules: Dict[int, torch.nn.Module], timers_level: int, prefix: str) -> None:
        self.modules_by_layer = dict(layer_modules)

        for layer_id, module in sorted(self.modules_by_layer.items()):
            layer_key = f'layer_{layer_id}'
            forward_timer_name = f'timers_{prefix}.{layer_key}.forward'

            module.forward = gigatimer(
                forward_timer_name,
                timer_type=TimerType.CUDA,
                timer_level=timers_level,
            )(module.forward)


class MoeRanksTiming(Callback):
    """Attach gigatimers to local MoE forward methods."""

    def __init__(
        self,
        timers_level: int = 1,
        prefix: str = 'moe_rank_timers',
        moe_layer_name: str = 'DeepseekGMMMoeBlock',
    ) -> None:
        self.prefix: str = prefix
        self.timers_level: int = timers_level
        self.moe_layer_name: str = moe_layer_name
        self._layer_discovery = _MoeLayerDiscoveryHandler(moe_layer_name=self.moe_layer_name)
        self._wrapping_handler = _MoeForwardWrappingHandler()

    def _attach_forward_wrappers(self, model: torch.nn.Module) -> None:
        local_layers = self._layer_discovery.discover_local_layers(model)
        self._wrapping_handler.attach(layer_modules=local_layers, timers_level=self.timers_level, prefix=self.prefix)

    def fit_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self._attach_forward_wrappers(state.model)
