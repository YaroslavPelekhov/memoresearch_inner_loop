"""Compatibility bridge from the supplied GigaChat LLM Foundry to public Composer.

The supplied LLM Foundry snapshot references a private Composer fork, but its
``contrib/composer`` git submodule is not present in the snapshot.  The first
autoresearch experiment uses only ordinary single-GPU Composer features.  This
module supplies the missing names needed to import that snapshot while making
unsupported private-only features fail explicitly if a config requests them.
"""

from __future__ import annotations

from contextlib import nullcontext
import sys
from types import ModuleType
from typing import Any


class _UnavailablePrivateFeature:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "This config requested a feature from the unavailable private "
            "GigaChat Composer submodule. The autoresearch v1 config must use "
            "public Composer features only."
        )


class _GlobalActivationLogState:
    enable_monitor = False

    @staticmethod
    def set_monitor_variable(_monitor: Any) -> None:
        return None

    @staticmethod
    def use_monitor_variable(value: Any, *_args: Any, **_kwargs: Any) -> Any:
        return value

    @staticmethod
    def apply_layer_counter(value: Any) -> Any:
        return value

    @staticmethod
    def set_layer_idx_variable(*_args: Any, **_kwargs: Any) -> None:
        return None


class _GigaTimer:
    interval = 1

    @staticmethod
    def set_timers_threshold(_threshold: int) -> None:
        return None

    @classmethod
    def set_interval(cls, interval: int) -> None:
        cls.interval = interval


def install() -> None:
    """Patch only names absent from public Composer; never replace real ones."""

    # Composer 0.20 (the last public release containing its ICL evaluator) used
    # a tiny AMP helper that newer PyTorch versions no longer export.
    import torch.cuda.amp.grad_scaler as grad_scaler

    if not hasattr(grad_scaler, "_refresh_per_optimizer_state"):
        def _refresh_per_optimizer_state() -> dict[str, Any]:
            return {
                "stage": grad_scaler.OptState.READY,
                "found_inf_per_device": {},
            }

        grad_scaler._refresh_per_optimizer_state = _refresh_per_optimizer_state  # type: ignore[attr-defined]

    import composer.callbacks as callbacks
    import composer.models.huggingface as composer_huggingface
    import composer.optim as optim
    import composer.optim.scheduler as scheduler
    import composer.utils as utils
    from composer.utils import dist
    try:
        import composer.utils.timers as timers
    except ModuleNotFoundError:
        timers = ModuleType("composer.utils.timers")
        sys.modules[timers.__name__] = timers
    if "composer.utils.profiler_annotation" not in sys.modules:
        profiler_annotation = ModuleType("composer.utils.profiler_annotation")

        def decorator_forward_backward(fn: Any = None):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        profiler_annotation.decorator_forward_backward = decorator_forward_backward  # type: ignore[attr-defined]
        sys.modules[profiler_annotation.__name__] = profiler_annotation
    if "composer.trainer.sac_utils" not in sys.modules:
        sac_utils = ModuleType("composer.trainer.sac_utils")
        sac_utils.cpu_offload_context = nullcontext()  # type: ignore[attr-defined]
        sac_utils.cpu_offload_sync_function = lambda value: value  # type: ignore[attr-defined]
        sys.modules[sac_utils.__name__] = sac_utils

    for name in (
        "TrainingMetricsMonitor",
        "EMA",
        "OnlineProfiler",
    ):
        if not hasattr(callbacks, name):
            setattr(callbacks, name, _UnavailablePrivateFeature)

    for name in ("DecoupledFusedAdamW", "Adan", "Muon"):
        if not hasattr(optim, name):
            setattr(optim, name, _UnavailablePrivateFeature)

    for name in (
        "InvSquareRootWithWarmupAndCooldownScheduler",
        "ConstantMultiStepWithGammaWithWarmupScheduler",
    ):
        if not hasattr(scheduler, name):
            setattr(scheduler, name, _UnavailablePrivateFeature)

    if not hasattr(utils, "global_actlog_state"):
        utils.global_actlog_state = _GlobalActivationLogState()  # type: ignore[attr-defined]
    if not hasattr(utils, "get_torch_version"):
        from packaging.version import parse as parse_version
        import torch

        utils.get_torch_version = lambda: parse_version(torch.__version__.split("+")[0])  # type: ignore[attr-defined]
    if not hasattr(timers, "gigatimer"):
        timers.gigatimer = _GigaTimer()  # type: ignore[attr-defined]

    # Composer 0.20 scans every modern Transformers causal-LM mapping and trips
    # over unrelated dynamically documented models. Gigar is unambiguously a
    # causal LM and already passes shift_labels=True to the wrapper.
    composer_huggingface._is_registered_causal_lm = lambda _model: True

    original_initialize_dist = dist.initialize_dist
    if not getattr(original_initialize_dist, "_autoresearch_compat", False):
        def initialize_dist(device: Any = None, timeout: float = 300.0, **_kwargs: Any) -> None:
            original_initialize_dist(device=device, timeout=timeout)

        initialize_dist._autoresearch_compat = True  # type: ignore[attr-defined]
        dist.initialize_dist = initialize_dist

    scalar_defaults = {
        "get_tp_group_size": 1,
        "get_sp_group_size": 1,
        "get_ep_group_size": 1,
        "get_tp_sp_group_size": 1,
        "get_tp_group_rank": 0,
        "get_sp_group_rank": 0,
        "get_ep_group_rank": 0,
        "get_fsdp_group_rank": 0,
    }
    for name, value in scalar_defaults.items():
        if not hasattr(dist, name):
            setattr(dist, name, lambda value=value: value)
    for name in ("get_tp_group", "get_sp_group", "get_ep_group", "get_tp_sp_group"):
        if not hasattr(dist, name):
            setattr(dist, name, lambda: None)
    if not hasattr(dist, "run_local_rank_zero_first"):
        dist.run_local_rank_zero_first = nullcontext  # type: ignore[attr-defined]
    if not hasattr(dist, "is_trivial_process_group"):
        dist.is_trivial_process_group = lambda _group=None: True  # type: ignore[attr-defined]
