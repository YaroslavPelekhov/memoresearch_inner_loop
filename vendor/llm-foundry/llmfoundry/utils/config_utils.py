# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import warnings
from typing import (
    Any,
    Optional,
    Union,
    Dict,
    List
)

from composer.utils import dist
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

import os
from copy import deepcopy

from llmfoundry.models.utils import init_empty_weights

# get gigar config and write lora params
def write_lora_config(cfg: DictConfig, om_model_config):
    if om_model_config.get("lora", False):
        lora_config = to_dict_container(om_model_config.get("lora", False)['args'])
        cfg.lora_config = lora_config
    return cfg


def calculate_batch_size_info(global_batch_size: int,
                              device_microbatch_size: Union[int, str]):
    tp_sp_size = dist.get_tp_sp_group_size()
    if tp_sp_size is None:
        world_size = dist.get_world_size()
    else:
        world_size = dist.get_world_size() // tp_sp_size
    if global_batch_size % world_size != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {world_size} '
            +
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            + f'to be divisible by data parallel world size, {world_size}.')
    device_batch_size = global_batch_size // world_size
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            print(
                'WARNING: device_microbatch_size > device_batch_size, ' +
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.'
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size /
                                      device_microbatch_size)
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: DictConfig):
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size, cfg.device_train_microbatch_size)
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    print(
        f"Global batch_size = {cfg.global_train_batch_size}, " +
        f"micro batch_size = {cfg.device_train_microbatch_size}, " +
        f"gradient accumulation = {cfg.device_train_grad_accum}"
    )
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    if cfg.get('fsdp_config') is not None:
        cfg.fsdp_config.gradient_accumulation_steps = cfg.device_train_grad_accum
    return cfg


def process_init_device(model_cfg: DictConfig, fsdp_config: Optional[Dict]):
    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in model_cfg:
        assert model_cfg.init_device in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and model_cfg.init_device == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            model_cfg.init_device = 'cpu'
        if model_cfg.init_device == 'meta':
            init_context = init_empty_weights()
        if model_cfg.init_device == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. ' +
                    'Please add a FSDP config.')
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('use_orig_params', False)
            fsdp_config.setdefault('load_monolith_rank0_only', True)
    return init_context


def _drop_path(d: Dict[str, Any], path: List[str]) -> None:
    cur = d
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def _prepare_config_yaml(cfg: DictConfig) -> str:
    sanitized = deepcopy(om.to_container(cfg, resolve=True))
    for p in [
        # ["train_loader", "dataset"],
        ["eval_loader", "dataset"],
        ["eval_gauntlet"],
        ["icl_tasks"],
    ]:
        _drop_path(sanitized, p)

    yaml_str = om.to_yaml(om.create(sanitized))
    return yaml_str


def log_config(cfg: DictConfig):
    full_yaml = om.to_yaml(cfg, resolve=True)
    print(full_yaml)

    tb_cfg = cfg.get('loggers', {}).get('tensorboard', None)
    if tb_cfg is not None:
        yaml_str = _prepare_config_yaml(cfg)
        rank_zero_only = bool(getattr(tb_cfg, 'rank_zero_only', True))
        from torch.utils.tensorboard import SummaryWriter
        if (not rank_zero_only) or dist.get_global_rank() == 0:
            base_dir = str(getattr(tb_cfg, 'log_dir', 'runs'))
            run_name = str(getattr(cfg, 'run_name', ''))
            run_dir = os.path.join(base_dir, run_name) if run_name else base_dir
            os.makedirs(run_dir, exist_ok=True)

            with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
                f.write(yaml_str)

            writer = SummaryWriter(log_dir=run_dir)
            writer.add_text('config/', f'```yaml\n{yaml_str}\n```', global_step=0)
            writer.flush()
            writer.close()

    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True), allow_val_change=True)

    if 'mlflow' in cfg.get('loggers', {}):
        try:
            import mlflow
        except ImportError as e:
            raise e
        if mlflow.active_run():
            mlflow.log_params(params=om.to_container(cfg, resolve=True))

def _load_stats_info(stats_path: str) -> dict:
    """Load and return contents of a stats.txt JSON file."""
    import json
    with open(stats_path, 'r') as f:
        return json.load(f)


def get_batches_in_epoch(config) -> Optional[int]:
    """Determine the number of batches per epoch.

    Auto-discovers stats.txt for single-stream datasets,
    falls back to ceil(epoch_size / global_train_batch_size).
    Works with both plain dicts and OmegaConf DictConfig.
    """
    try:
        streams = dict(config.get('train_loader', {}).get('dataset', {}).get('streams', {}))
    except Exception:
        streams = {}
    if len(streams) == 1:
        stream = list(streams.values())[0]
        local_path = stream['local'] if isinstance(stream, dict) else stream.local
        stats_path = os.path.join(local_path, 'stats.txt')
        if os.path.exists(stats_path):
            return int(_load_stats_info(stats_path)['long_epoch_size'])
    epoch_size = config.get('train_loader', {}).get('dataset', {}).get('epoch_size')
    gbs = config.get('global_train_batch_size')
    if epoch_size is not None and gbs is not None:
        return math.ceil(int(epoch_size) / int(gbs))
    return None


def process_epoch_value(val, batches_in_epoch: Optional[int]):
    """Convert an epoch-based interval value to batches.

    Handles both integer (2ep) and fractional (0.5ep) epoch values.
    If batches_in_epoch is None, integer epochs are left as-is
    (handled natively by Composer), fractional epochs produce a warning.
    """
    if not isinstance(val, str) or not val.endswith('ep'):
        return val

    epoch_num = float(val[:-2])
    is_integer_epoch = int(epoch_num) == epoch_num

    if batches_in_epoch is not None:
        return f'{math.ceil(epoch_num * batches_in_epoch)}ba'

    if not is_integer_epoch:
        warnings.warn(
            f"Fractional epoch value '{val}' requires stats_path or computed "
            "batches_in_epoch for conversion. Leaving as-is."
        )
    return val


def resolve_epoch_intervals(cfg, logger=None):
    """Convert epoch-based interval values to batches in-place.

    Processes max_duration, eval_interval, save_interval,
    stable_save_interval, console_log_interval and scheduler.milestones.
    Works with both plain dicts and OmegaConf DictConfig.
    """
    def _has_ep(val):
        if isinstance(val, str):
            return val.endswith('ep')
        if isinstance(val, (list, ListConfig)):
            return any(isinstance(v, str) and v.endswith('ep') for v in val)
        return False

    interval_keys = ('max_duration', 'eval_interval', 'save_interval',
                     'stable_save_interval', 'console_log_interval')

    has_ep_values = any(_has_ep(cfg.get(k)) for k in interval_keys)
    milestones = cfg.get('scheduler', {}).get('milestones', None)
    if milestones:
        has_ep_values = has_ep_values or _has_ep(milestones)

    if not has_ep_values:
        return

    bie = get_batches_in_epoch(cfg)
    if bie is not None:
        if logger:
            logger.info("Batches in epoch: %d", bie)
    else:
        if logger:
            logger.warning("Config has epoch-based intervals but cannot determine batches_in_epoch. Leaving ep values as-is.")
        return

    for key in interval_keys:
        val = cfg.get(key, None)
        if val is None:
            continue
        if isinstance(val, (list, ListConfig)):
            new_list = [process_epoch_value(v, bie) for v in val]
            if logger:
                logger.info("Resolved %s: %s -> %s", key, list(val), new_list)
            cfg[key] = new_list
        elif isinstance(val, str) and val.endswith('ep'):
            new_val = process_epoch_value(val, bie)
            if logger:
                logger.info("Resolved %s: %s -> %s", key, val, new_val)
            cfg[key] = new_val

    if milestones:
        new_milestones = [process_epoch_value(m, bie) for m in milestones]
        if logger:
            logger.info("Resolved scheduler.milestones: %s -> %s", list(milestones), new_milestones)
        cfg['scheduler']['milestones'] = new_milestones


def make_combined_interval_scheduler(intervals):
    """Convert a list of specific training steps into a single callable scheduler.

    Each item specifies an exact batch number at which the action should
    trigger (one-shot, not periodic).

    Example YAML::

        save_interval:
          - 500ba
          - 1000ba
          - 5000ba
    """
    batch_steps = set()
    for iv in intervals:
        if isinstance(iv, str) and iv.endswith('ba'):
            batch_steps.add(int(iv[:-2]))
        elif isinstance(iv, int):
            batch_steps.add(iv)
        else:
            raise ValueError(
                f"List interval items must be in batch units (e.g. '500ba'), got: {iv}"
            )

    def check_at_steps(state, event):
        current_batch = int(state.timestamp.batch)
        return current_batch in batch_steps

    check_at_steps.batch_steps = batch_steps
    return check_at_steps


def resolve_interval(cfg, key, default=None, logger=None):
    """Read an interval value from cfg. If it's a list, convert to a combined callable."""
    val = cfg.get(key, default)
    if isinstance(val, (list, ListConfig)) and len(val) > 0:
        if logger:
            logger.info("Converting list %s=%s to combined interval scheduler", key, list(val))
        return make_combined_interval_scheduler(list(val))
    return val


def to_dict_container(cfg: Union[DictConfig, dict[str, Any]]) -> dict[str, Any]:
    maybe_dict = to_container(cfg)
    if isinstance(maybe_dict, dict):
        return maybe_dict
    else:
        raise ValueError(f'Expected a dict-like type, got {type(maybe_dict)}')

def to_container(
    cfg: Optional[Union[DictConfig, ListConfig, dict[str, Any],
                        list[dict[str, Any]]]],
) -> Union[dict[str, Any], list[dict[str, Any]]]:
    """Converts a DictConfig or ListConfig to a dict or list.

    `omegaconf.to_container` does not handle nested DictConfig or ListConfig
    objects, so this function is used to convert them to dicts or lists.
    """
    if isinstance(cfg, DictConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, dict)
        return ret  # type: ignore (return type is correct and converting all keys to str would be unnecessarily costly)
    elif isinstance(cfg, ListConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, list)
        return ret  # type: ignore (see above)
    else:
        return cfg  # type: ignore (dicts and lists are already in the correct format)
