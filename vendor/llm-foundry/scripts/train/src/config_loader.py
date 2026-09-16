from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf as om
from omegaconf import DictConfig, ListConfig


def register_relative_config_loader(base_path: Path) -> None:
    """
    Register a resolver for loading configs relative to a base path.

    Args:
        base_path: The base path to resolve paths relative to.
    Returns:
        None
    """
    base = Path(base_path).resolve()

    def rp_load(path: str):
        return om.load((base / path).resolve())

    def rp_merge(*config_paths: list[str]):
        if not config_paths:
            return om.create({})

        result_config = om.create({})

        for c_p in config_paths:
            result_config = om.merge(result_config, om.load((base / c_p).resolve()))

        return result_config

    om.register_new_resolver("rp.load", rp_load, replace=True)
    om.register_new_resolver("rp.merge", rp_merge, replace=True)


def _drop_key(cfg: DictConfig, *path: str) -> None:
    current = cfg
    for part in path[:-1]:
        if not isinstance(current, DictConfig) or part not in current:
            return
        current = current[part]
    if isinstance(current, DictConfig):
        current.pop(path[-1], None)


def _has_key(cfg: DictConfig, *path: str) -> bool:
    current = cfg
    for part in path:
        if not isinstance(current, DictConfig) or part not in current:
            return False
        current = current[part]
    return True


def _has_runtime_evaluators(cfg: DictConfig) -> bool:
    return any(
        (
            _has_key(cfg, "eval_loader"),
            _has_key(cfg, "eval_loaders"),
            _has_key(cfg, "icl_tasks"),
        )
    )


def _disable_all_evaluators(cfg: DictConfig) -> None:
    _drop_key(cfg, "eval_loader")
    _drop_key(cfg, "eval_loaders")
    _drop_key(cfg, "icl_tasks")
    _drop_key(cfg, "eval_gauntlet")
    cfg["eval_interval"] = 0
    if "eval_before_train" in cfg:
        cfg["eval_before_train"] = "skip"


def _disable_runtime_benchmark_callbacks(cfg: DictConfig) -> None:
    for callback_name in (
        "autometrics",
        "job_dispatcher_callback",
        "job_dispatcher_callback_web",
        "online_profiler",
    ):
        _drop_key(cfg, "callbacks", callback_name)


def _reduce_train_streams(cfg: DictConfig, stream_name: Optional[str] = None) -> None:
    streams = cfg.get("train_loader", {}).get("dataset", {}).get("streams")
    if not isinstance(streams, DictConfig) or not streams:
        return

    selected_name = stream_name or next(iter(streams.keys()))
    if selected_name not in streams:
        raise ValueError(
            f"Requested debug train stream '{selected_name}' is missing from train_loader.dataset.streams."
        )

    cfg.train_loader.dataset.streams = om.create({selected_name: streams[selected_name]})


def _filter_icl_tasks(cfg: DictConfig, labels: ListConfig | list[str] | None) -> None:
    if labels is None:
        _drop_key(cfg, "icl_tasks")
        _drop_key(cfg, "eval_gauntlet")
        return

    if not _has_key(cfg, "icl_tasks"):
        return

    requested_labels = set(labels)
    filtered_tasks = [
        task
        for task in cfg.icl_tasks
        if task.get("label") in requested_labels
    ]
    if filtered_tasks:
        cfg.icl_tasks = om.create(filtered_tasks)
        return

    _drop_key(cfg, "icl_tasks")
    _drop_key(cfg, "eval_gauntlet")


def _disable_checkpoint_loading(cfg: DictConfig) -> None:
    for key in (
        "load_path",
        "teacher_load_path",
        "load_ignore_keys",
    ):
        _drop_key(cfg, key)
    cfg["load_weights_only"] = False
    cfg["autoresume"] = False


def _enable_fast_model_init(cfg: DictConfig) -> None:
    has_sharding = cfg.get("fsdp_config") is not None or cfg.get("gigafsdp_config") is not None
    if not has_sharding:
        return

    def maybe_switch_init_device(model_cfg: Optional[DictConfig]) -> None:
        if not isinstance(model_cfg, DictConfig):
            return
        if model_cfg.get("pretrained", False):
            return
        model_cfg["init_device"] = "meta"

    maybe_switch_init_device(cfg.get("model"))

    if isinstance(cfg.get("model"), DictConfig):
        maybe_switch_init_device(cfg.model.get("teacher_model"))
        maybe_switch_init_device(cfg.model.get("student_model"))


def apply_debug_config(cfg: DictConfig) -> DictConfig:
    """Apply debug-specific config transforms before generic overrides."""
    debug_cfg = cfg.get("debug")
    if not isinstance(debug_cfg, DictConfig) or not debug_cfg.get("use_debug", False):
        return cfg

    debugged_cfg = om.create(cfg)
    startup_speedups = debug_cfg.get("startup_speedups") or om.create({})

    use_one_train_stream = startup_speedups.get(
        "use_one_train_stream",
        startup_speedups.get("use_one_train_dataloader", True),
    )
    if use_one_train_stream:
        _reduce_train_streams(
            debugged_cfg,
            startup_speedups.get("train_stream_name"),
        )

    if startup_speedups.get("disable_runtime_benchmark_callbacks", True):
        _disable_runtime_benchmark_callbacks(debugged_cfg)

    if startup_speedups.get("disable_eval_loader", True):
        _drop_key(debugged_cfg, "eval_loader")
        _drop_key(debugged_cfg, "eval_loaders")

    if "icl_tasks_metrics" in startup_speedups:
        _filter_icl_tasks(debugged_cfg, startup_speedups.get("icl_tasks_metrics"))
    else:
        _filter_icl_tasks(debugged_cfg, None)

    if startup_speedups.get("disable_all_evaluators", False):
        _disable_all_evaluators(debugged_cfg)
    elif not _has_runtime_evaluators(debugged_cfg):
        debugged_cfg["eval_interval"] = 0
        if "eval_before_train" in debugged_cfg:
            debugged_cfg["eval_before_train"] = "skip"

    if not startup_speedups.get("load_checkpoints", False):
        _disable_checkpoint_loading(debugged_cfg)

    if startup_speedups.get("fast_model_init", True):
        _enable_fast_model_init(debugged_cfg)

    if debug_cfg.get("overrides") is not None:
        debugged_cfg = om.merge(debugged_cfg, debug_cfg.overrides)

    return debugged_cfg


def apply_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply overrides to the config.

    Args:
        cfg: The config to apply overrides to.

    Returns:
        The config with overrides applied.
    """
    if "overrides" not in cfg or cfg.overrides is None:
        return cfg
    base = om.create(cfg)
    del base["overrides"]
    return om.merge(base, cfg.overrides)


def _get_active_model_overrides(cfg: DictConfig) -> DictConfig | None:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        return None

    if model_cfg.get("student_model"):
        return model_cfg.get("student_model", {}).get("config_overrides")

    return model_cfg.get("config_overrides")


def sync_peak_capacity_microbatchsize(cfg: DictConfig) -> DictConfig:
    """
    Fill `peak_capacity_microbatchsize` from `device_train_microbatch_size`.

    The `peak_capacity_factor` dispatcher strategy computes a fixed expert
    capacity threshold once during model construction from model config only.
    Since `device_train_microbatch_size` lives at the trainer level, copy it
    into `model.config_overrides` when the user did not provide an explicit
    value.
    """
    model_overrides = _get_active_model_overrides(cfg)
    if model_overrides is None:
        return cfg

    if model_overrides.get("dispatcher_balancing_strategy") != "peak_capacity_factor":
        return cfg

    if model_overrides.get("peak_capacity_microbatchsize") is not None:
        return cfg

    device_train_microbatch_size = cfg.get("device_train_microbatch_size")
    if not isinstance(device_train_microbatch_size, int):
        raise ValueError(
            "dispatcher_balancing_strategy='peak_capacity_factor' requires either "
            "an explicit model.config_overrides.peak_capacity_microbatchsize or "
            f"an integer device_train_microbatch_size, got "
            f"{device_train_microbatch_size!r}"
        )

    model_overrides["peak_capacity_microbatchsize"] = device_train_microbatch_size
    return cfg


def prepare_train_config(cfg: DictConfig) -> DictConfig:
    """Apply train-specific config transforms in startup-safe order."""
    cfg = apply_debug_config(cfg)
    cfg = apply_overrides(cfg)
    om.resolve(cfg)
    return cfg
