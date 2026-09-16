import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist, TimerType, callback_timer
from giga_fsdp import GigaFSDP

import logging

log = logging.getLogger(__name__)


class ParallelismConsistencyCheck(Callback):
    def __init__(self, batch_interval):
        self.batch_interval = batch_interval

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger) -> None:
        if dist.get_tp_group_size() is None:
            return
        if state.timestamp.batch.value % self.batch_interval != 0:
            return
        for module in state.model.children():
            if isinstance(module, GigaFSDP):
                if module._separate_layer_norm_enabled:
                    layernorm_tensor = module._layernorm_parameter
                    compare_layernorms(layernorm_tensor, state.timestamp.batch.value)

def compare_layernorms(layernorm_tensor, batch):
    tensor_list = [torch.zeros_like(layernorm_tensor) for _ in range(dist.get_tp_group_size())]
    torch.distributed.all_gather(tensor_list, layernorm_tensor, group=dist.get_tp_group())

    for i in range(1, len(tensor_list)):
        if not torch.allclose(tensor_list[0], tensor_list[i], rtol=1e-5, atol=1e-8):
            log.critical(f"The difference of _layernorm_parameter values between ranks 0 and {i}, batch_number = {batch}")
