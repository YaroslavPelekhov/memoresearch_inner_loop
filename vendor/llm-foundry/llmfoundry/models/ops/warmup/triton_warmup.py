import argparse
import logging
import os
import pickle
import typing as tp

import numpy as np
import torch
import triton
from triton_functions import (GROUP_TO_AUTOTUNE_KERNELS, KERNEL_KEY_METADATA,
                              WARMUP_FNS, _get_autotuned_kernels)
from utils import load_warmup_config

import logging
import sys

log = logging.getLogger("triton_warmup")
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)

log.propagate = False

# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def save_autotune_report(cache_dir: str) -> None:
    """Build per-kernel TSV reports from every *.pkl autotune cache file in ``cache_dir``.

    Writes ``<cache_dir>/{kernel_name}_autotune_report.tsv`` for each kernel with
    one row per autotune-key tuple.  Columns:

        kernel          — kernel cache name
        <key dims>      — one column per autotune-key dimension, named after the
                          actual kernel parameter (e.g. tensor_size_row).
        num_warps       — winning config meta-parameter
        num_stages      — winning config meta-parameter
        maxnreg         — winning config meta-parameter (NaN if unset)
        <config kwargs> — one column per constexpr kwarg in the config dict
                          (e.g. BLOCK_SIZE, POWER_TWO_MAX_ROUND, MAX_NUM_GROUPS)
    """
    import pandas as pd

    written = 0
    for pkl_file in sorted(os.listdir(cache_dir)):
        if not pkl_file.endswith(".pkl"):
            continue
        kernel_name = pkl_file[:-4]
        with open(os.path.join(cache_dir, pkl_file), "rb") as fh:
            cache: dict = pickle.load(fh)

        if not cache:
            log.warning("[report] %s: empty cache, skipping", kernel_name)
            continue

        key_meta = KERNEL_KEY_METADATA.get(kernel_name, {})
        key_names = key_meta.get("keys", [])
        rows: tp.List[tp.Dict[str, tp.Any]] = []
        for key_tuple, config in cache.items():
            row: tp.Dict[str, tp.Any] = {"kernel": kernel_name}
            for i, val in enumerate(key_tuple):
                col = key_names[i] if i < len(key_names) else f"key_{i}"
                row[col] = val
            if "time" in config:
                time = np.mean(config.pop("time"))
                row["mean_time"] = time
            for key in config:
                row[key] = config[key]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Canonical column order: kernel | keys | config meta | kwargs.
        fixed_cols = ["kernel"] + key_names + ["num_warps", "num_stages", "maxnreg", "mean_time"]
        fixed_cols = [c for c in fixed_cols if c in df.columns]
        kwargs_cols = [c for c in df.columns if c not in fixed_cols]
        df = df[fixed_cols + kwargs_cols]

        out_path = os.path.join(cache_dir, f"{kernel_name}_autotune_report.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        log.info("[report] %s: %d entries -> %s", kernel_name, len(df), out_path)
        written += 1

    if written == 0:
        log.warning("[report] no cache entries found in %s — no TSV written", cache_dir)


def autotune_get_best_config(cache_dir: str, kernel_names: tp.List[str]) -> None:
    """Get the best config for a given kernel from the autotune cache.
       For getting the best config, we perform the following steps:
          1. Separate the information about the ignore keys from the full key.
          2. For all remaining unique keys, we select the most frequent config.
          3. We output the result in JSON format.       
    """
    import hashlib
    from collections import Counter, defaultdict

    def _hashkey_params(key_tuple: tuple) -> str:
        hash_key = " ".join(list(map(str, key_tuple)))
        return hashlib.sha256(hash_key.encode("utf-8")).hexdigest()

    def _drop_ignore_keys(
        key_tuple: tuple,
        key_names: tp.List[str],
        ignore_keys_on_save: tp.List[str]) -> tuple:
        keep_indices = [
            i for i, name in enumerate(key_names)
            if name not in ignore_keys_on_save
        ]
        return tuple(key_tuple[i] for i in keep_indices) + \
               tuple(key_tuple[len(key_names):])

    kernels = _get_autotuned_kernels()
    for name in kernel_names:
        kernel = kernels[name]
        assert isinstance(kernel, triton.runtime.autotuner.Autotuner), \
            f"Kernel {name} is not an Autotuner." + \
             " Set @lookup_autotuner decorator on the kernel." + \
             " Check /llmfoundry/models/ops/warmup/README.md (Triton section) for more information."
        cache = dict(kernel.cache)
        timings = dict(kernel.configs_timings)
        best_configs = defaultdict(dict)
        map_hashkey_params = defaultdict(dict)
        counter_hashkey_params = defaultdict(Counter)
        key_meta = KERNEL_KEY_METADATA.get(name, {})
        key_names = key_meta.get("keys", [])
        ignore_keys_on_save = key_meta.get("ignore_keys_on_save", [])
        for key, params in cache.items():
            if ignore_keys_on_save:
                key = _drop_ignore_keys(key, key_names, ignore_keys_on_save)
            cache_params = params.all_kwargs()
            times = timings[params]
            hash_params = _hashkey_params(cache_params.values())
            map_hashkey_params[key][hash_params] = cache_params
            map_hashkey_params[key][hash_params]["time"] = times
            counter_hashkey_params[key].update([hash_params])

        for key, params in cache.items():
            if ignore_keys_on_save:
                key = _drop_ignore_keys(key, key_names, ignore_keys_on_save)
            hash_key = counter_hashkey_params[key].most_common(1)[0][0]
            best_config = map_hashkey_params[key][hash_key]
            best_configs[key] = best_config
            best_configs[key]["time"] = map_hashkey_params[key][hash_key]["time"]

        out_path = os.path.join(cache_dir, f"{name}_best_config.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump(best_configs, fh)

# ---------------------------------------------------------------------------
# Running warmup
# ---------------------------------------------------------------------------

def _run_single_group(cfg, group: str, device_id: int) -> None:
    """Run a single warmup group, isolated in a per-process Triton cache dir."""

    cache_dir = cfg.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    log.info("=== Warmup group='%s' device=cuda:%d ===", group, device_id)
    WARMUP_FNS[group](cfg, device)

    kernel_names = GROUP_TO_AUTOTUNE_KERNELS[group]
    if kernel_names:
        autotune_get_best_config(cache_dir, kernel_names)

    log.info("=== Done group='%s' ===", group)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the warmup config YAML file.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--kernel-group",
        choices=list(WARMUP_FNS),
        help="Warm up a single kernel group on --device.  Omit to run all groups sequentially.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="ID",
        help=(
            "CUDA device index to use for single-process modes"
            " (--kernel-group or sequential fallback).  Default: 0."
        ),
    )

    args = parser.parse_args()

    cfg = load_warmup_config(args.config)

    if args.kernel_group:
        _run_single_group(cfg, args.kernel_group, args.device)
    else:
        # Sequential: run all groups one after another on the specified device.
        log.info("Running all groups sequentially on cuda:%d.", args.device)
        for group in WARMUP_FNS:
            _run_single_group(cfg, group, args.device)
    save_autotune_report(cfg.cache_dir)
