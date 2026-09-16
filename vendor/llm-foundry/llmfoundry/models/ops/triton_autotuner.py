import os
import pickle
import types
import typing as tp
import warnings

import triton


def set_best_config(
    best_config_path: str,
    best_config_path_bwd: tp.Optional[str] = None,
    keys: tp.Optional[tp.List[str]] = None,
    ignore_keys_on_save: tp.Optional[tp.List[str]] = None):
    if not best_config_path or not os.path.exists(best_config_path):
        warnings.warn(f"No best config found: {best_config_path}")
        return lambda kernel: kernel

    if ignore_keys_on_save is None:
        ignore_keys_on_save = []

    with open(best_config_path, "rb") as f:
        best_config = pickle.load(f)

    best_config_bwd = None
    if best_config_path_bwd:
        if not os.path.exists(best_config_path_bwd):
            warnings.warn(f"No best config (bwd) found: {best_config_path_bwd}")
            return lambda kernel: kernel
        with open(best_config_path_bwd, "rb") as f:
            best_config_bwd = pickle.load(f)
            best_config_bwd.pop("time")

    def decorator(kernel):
        orig_run = kernel.run
        def run_with_best_config(self, *args, **kwargs):
            nargs = dict(zip(kernel.arg_names, args))
            all_args = {**nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in kernel.arg_names}
            key = [_args[key] for key in keys if key in _args and key not in ignore_keys_on_save]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            cfg_kwargs = best_config.get(key)
            if cfg_kwargs is None and best_config_bwd is not None:
                cfg_kwargs = best_config_bwd.get(key)
            # in triton.jit() all parameters for kernel passed in kwargs,
            # so we need to update kwargs with best config.
            if cfg_kwargs is not None:
                kwargs.update(cfg_kwargs)
            else:
                warnings.warn(f"No best config found for key: {key} from best_configs: {best_config}")
            return orig_run(*args, **kwargs)
        kernel.run = types.MethodType(run_with_best_config, kernel)
        return kernel
    return decorator


def lookup_autotuner(
    mode: str,
    best_config_path: tp.Optional[str] = None,
    best_config_path_bwd: tp.Optional[str] = None,
    ignore_keys_on_save: tp.Optional[tp.List[str]] = None,
    **kwargs):
    if mode == "warmup":
        return triton.autotune(**kwargs)
    elif mode == "training":
        keys = kwargs['key']
        return set_best_config(best_config_path,
                               best_config_path_bwd,
                               keys=keys,
                               ignore_keys_on_save=ignore_keys_on_save)
    else:
        return lambda kernel: kernel
