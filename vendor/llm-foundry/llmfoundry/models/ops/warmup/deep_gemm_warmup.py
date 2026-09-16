import argparse
import gc
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import typing as tp
from typing import Iterable

import torch
from tqdm import tqdm
from utils import DeepGemmWarmupCfg, _setup_standalone_path, load_config

_setup_standalone_path()

import deep_gemm

from composer.utils import dist

from llmfoundry.models.ops.float8.deep_gemm_handle import _deep_gemm_handle
from llmfoundry.models.ops.float8.triton_kernels.block_quantization import \
    block_quantization_128x128_fn as block_quantization
from llmfoundry.models.ops.float8.triton_kernels.block_transpose_fused_quantization import \
    block_transpose_fused_quantization_128x128_fn as \
    block_quantization_transpose
from llmfoundry.models.ops.float8.triton_kernels.row2col_deepgemm_quantization import \
    row2col_requantization_deepgemm_layout_fn as row2col_deepgemm_layout_fn
from llmfoundry.models.ops.float8.triton_kernels.row_quantization import \
    row_quantization_1x128_fn as row_quantization
from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices

log = logging.getLogger("deepgemm_warmup")
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)

log.propagate = False

def _quantize_inputs_for_m_grouped(x_bf16: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    x_q = row_quantization(x_bf16)
    return (x_q[0].contiguous(), x_q[1].contiguous())


def _quantize_weights_for_m_grouped(w_bf16: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    w_q = block_quantization(w_bf16)
    num_groups = w_bf16.shape[0]
    w_q = (
        w_q[0].reshape(num_groups, -1, w_q[0].size(-1)),
        w_q[1].reshape(num_groups, -1, w_q[1].size(-1)),
    )
    return w_q

# ---------------- NEW: cover totals, not just uniform size*num_groups ----------------

def _make_group_sizes_for_total(
    total: int,
    *,
    num_groups: int,
    step: int,
    max_per_group: int,
    pattern: str = "balanced",    # "balanced" | "one_hot" | "random"
    rng: tp.Optional[random.Random] = None,
) -> tp.List[int]:
    """
    Build per-group sizes (multiples of step) that sum to `total`.
    Caps each group to `max_per_group`.
    Allows zeros (important for real MoE routing).
    """
    assert total >= 0, total
    assert step > 0, step
    assert total % step == 0, (total, step)
    assert max_per_group % step == 0, (max_per_group, step)
    assert num_groups > 0, num_groups

    total_units = total // step
    max_units = max_per_group // step

    sizes_u = [0] * num_groups
    rng = rng or random.Random(0)

    if pattern == "one_hot":
        g = 0
        while total_units > 0:
            take = min(total_units, max_units)
            sizes_u[g] += take
            total_units -= take
            g += 1
            if g >= num_groups:
                g = 0

    elif pattern == "balanced":
        base = min(total_units // num_groups, max_units)
        sizes_u = [base] * num_groups
        total_units -= base * num_groups

        g = 0
        while total_units > 0:
            if sizes_u[g] < max_units:
                sizes_u[g] += 1
                total_units -= 1
            g += 1
            if g >= num_groups:
                g = 0

    elif pattern == "random":
        for _ in range(total_units):
            candidates = [i for i in range(num_groups) if sizes_u[i] < max_units]
            if not candidates:
                break
            sizes_u[rng.choice(candidates)] += 1

        # If caps prevented placement, fall back to balanced (deterministic)
        if sum(sizes_u) < total // step:
            return _make_group_sizes_for_total(
                total,
                num_groups=num_groups,
                step=step,
                max_per_group=max_per_group,
                pattern="balanced",
                rng=rng,
            )
    else:
        raise ValueError(f"unknown pattern={pattern}")

    return [u * step for u in sizes_u]


def _iter_totals(min_total: int, max_total: int, step: int) -> Iterable[int]:
    assert step > 0
    if min_total < 0:
        min_total = 0
    # align up
    if min_total % step != 0:
        min_total = ((min_total + step - 1) // step) * step
    # align down
    if max_total % step != 0:
        max_total = (max_total // step) * step
    if max_total < min_total:
        return []
    return range(min_total, max_total + 1, step)


@torch.no_grad()
def warmup_m_grouped_sweep(
    cfg: DeepGemmWarmupCfg,
    *,
    num_groups: int,
    min_total: tp.Optional[int] = None,
    max_per_group: tp.Optional[int] = None,
    step: tp.Optional[int] = None,
    # patterns: Tuple[str, ...] = ("balanced", "one_hot", "random"),
    patterns: tp.Tuple[str, ...] = ("random",),
) -> None:
    """
    Warm m_grouped over TOTAL tokens (m_sum) in increments of `step`,
    generating multiple per-expert distributions for each total.
    """
    step = cfg.step if step is None else int(step)
    min_total = cfg.min_size if min_total is None else int(min_total)
    max_per_group = cfg.max_size if max_per_group is None else int(max_per_group)

    assert step > 0
    assert min_total >= 0
    assert max_per_group > 0
    assert max_per_group % step == 0, "max_per_group should be aligned to step"
    assert num_groups > 0

    device = torch.device(cfg.device)
    torch.cuda.set_device(device.index if device.type == "cuda" else 0)

    disable_pbar = dist.get_global_rank() != 0

    max_total_tokens = int(max_per_group) * int(num_groups)
    if cfg.max_buffer_tokens is not None:
        max_total_tokens = min(max_total_tokens, cfg.max_buffer_tokens)

    totals = list(_iter_totals(min_total, max_total_tokens, step))
    log.info("[m_grouped] START  num_groups=%d  hid=%d  int=%d  "
             "max_per_group=%d  max_total_tokens=%d  steps=%d",
             num_groups, cfg.hid_dim, cfg.int_dim,
             max_per_group, max_total_tokens, len(totals))
    t0 = time.monotonic()

    x_max = torch.empty((max_total_tokens, cfg.hid_dim), device=device, dtype=cfg.dtype)
    x_max.normal_()
    x_q_max = _quantize_inputs_for_m_grouped(x_max)

    w = torch.empty((num_groups, cfg.int_dim, cfg.hid_dim), device=device, dtype=cfg.dtype)
    w.normal_()
    w_q = _quantize_weights_for_m_grouped(w)

    out_dim = w_q[0].size(1)
    out_max = torch.empty((max_total_tokens, out_dim), device=device, dtype=cfg.dtype)

    rng = random.Random(0)

    for total_tokens in tqdm(
        totals,
        desc="DeepGEMM warmup: m_grouped (by total)",
        unit="total",
        disable=disable_pbar,
    ):
        for pattern in patterns:

            group_sizes = _make_group_sizes_for_total(
                total_tokens,
                num_groups=num_groups,
                step=step,
                max_per_group=max_per_group,
                pattern=pattern,
                rng=rng,
            )

            group_sizes_t = torch.tensor(group_sizes, dtype=torch.int32, device=device)
            m_indices = build_m_indices(group_sizes, group_sizes_t)

            x_q = (x_q_max[0][:total_tokens], x_q_max[1][:total_tokens])
            x_q = (x_q[0].contiguous(), x_q[1].contiguous())
            out = out_max[:total_tokens]

            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x_q, w_q, out, m_indices)

            del group_sizes_t, m_indices
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    log.info("[m_grouped] DONE   %d totals in %.1fs (%.2f total/s)",
             len(totals), elapsed, len(totals) / max(elapsed, 1e-9))


@torch.no_grad()
def warmup_m_grouped_sweep_nt_for_dgrad_transposed(
    cfg: DeepGemmWarmupCfg,
    *,
    num_groups: int,
    min_total: tp.Optional[int] = None,
    max_per_group: tp.Optional[int] = None,
    step: tp.Optional[int] = None,
    # patterns: Tuple[str, ...] = ("balanced", "one_hot", "random"),
    patterns: tp.Tuple[str, ...] = ("random",),
) -> None:
    """
    Warm m_grouped NT path for dgrad with transposed weights over TOTAL tokens.
    """
    step = cfg.step if step is None else int(step)
    min_total = cfg.min_size if min_total is None else int(min_total)
    max_per_group = cfg.max_size if max_per_group is None else int(max_per_group)

    assert step > 0
    assert min_total >= 0
    assert max_per_group > 0
    assert max_per_group % step == 0, "max_per_group should be aligned to step"
    assert num_groups > 0

    device = torch.device(cfg.device)
    torch.cuda.set_device(device.index if device.type == "cuda" else 0)

    disable_pbar = dist.get_global_rank() != 0

    max_total_tokens = int(max_per_group) * int(num_groups)
    if cfg.max_buffer_tokens is not None:
        max_total_tokens = min(max_total_tokens, cfg.max_buffer_tokens)

    totals = list(_iter_totals(min_total, max_total_tokens, step))
    log.info("[m_grouped_dgrad] START  num_groups=%d  hid=%d  int=%d  "
             "max_per_group=%d  max_total_tokens=%d  steps=%d",
             num_groups, cfg.hid_dim, cfg.int_dim,
             max_per_group, max_total_tokens, len(totals))
    t0 = time.monotonic()

    dy_max = torch.empty((max_total_tokens, cfg.int_dim), device=device, dtype=cfg.dtype)
    dy_max.normal_()
    dy_q_max = _quantize_inputs_for_m_grouped(dy_max)

    w = torch.empty((num_groups, cfg.int_dim, cfg.hid_dim), device=device, dtype=cfg.dtype)
    w.normal_()
    w_q_t = block_quantization_transpose(w)

    out_max = torch.empty((max_total_tokens, cfg.hid_dim), device=device, dtype=cfg.dtype)

    rng = random.Random(0)

    for total_tokens in tqdm(
        totals,
        desc="DeepGEMM warmup: m_grouped (NT dgrad, W^T) (by total)",
        unit="total",
        disable=disable_pbar,
    ):
        for pattern in patterns:

            group_sizes = _make_group_sizes_for_total(
                total_tokens,
                num_groups=num_groups,
                step=step,
                max_per_group=max_per_group,
                pattern=pattern,
                rng=rng,
            )

            group_sizes_t = torch.tensor(group_sizes, dtype=torch.int32, device=device)
            m_indices = build_m_indices(group_sizes, group_sizes_t)

            dy_q = (dy_q_max[0][:total_tokens], dy_q_max[1][:total_tokens])
            dy_q = (dy_q[0].contiguous(), dy_q[1].contiguous())
            out = out_max[:total_tokens]

            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dy_q, w_q_t, out, m_indices)

            del group_sizes_t, m_indices
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    log.info("[m_grouped_dgrad] DONE   %d totals in %.1fs (%.2f total/s)",
             len(totals), elapsed, len(totals) / max(elapsed, 1e-9))


def clear_tensor_data(*tensors) -> None:
    for t in tensors:
        if t is not None:
            t.data = torch.Tensor()
            del t


@torch.no_grad()
def warmup_k_grouped_sweep(
    cfg: DeepGemmWarmupCfg,
    *,
    num_groups: int,
    min_total: tp.Optional[int] = None,
    max_per_group: tp.Optional[int] = None,
    step: tp.Optional[int] = None,
    # patterns: Tuple[str, ...] = ("balanced", "one_hot", "random"),
    patterns: tp.Tuple[str, ...] = ("random",),
) -> None:
    """
    Warm k_grouped over TOTAL K (k_sum) in increments of `step`,
    synthesizing multiple ks distributions per total.
    """
    step = cfg.step if step is None else int(step)
    min_total = cfg.min_size if min_total is None else int(min_total)
    max_per_group = cfg.max_size if max_per_group is None else int(max_per_group)

    assert step > 0
    assert min_total >= 0
    assert max_per_group > 0
    assert max_per_group % step == 0, "max_per_group should be aligned to step"
    assert num_groups > 0

    device = torch.device(cfg.device)
    torch.cuda.set_device(device.index if device.type == "cuda" else 0)

    disable_pbar = dist.get_global_rank() != 0

    max_total_tokens = int(max_per_group) * int(num_groups)
    if cfg.max_buffer_tokens is not None:
        max_total_tokens = min(max_total_tokens, cfg.max_buffer_tokens)

    totals = list(_iter_totals(min_total, max_total_tokens, step))
    log.info("[k_grouped] START  num_groups=%d  hid=%d  int=%d  "
             "max_per_group=%d  max_total_tokens=%d  steps=%d",
             num_groups, cfg.hid_dim, cfg.int_dim,
             max_per_group, max_total_tokens, len(totals))
    t0 = time.monotonic()

    handle = _deep_gemm_handle(num_sms=cfg.num_sms)

    x_max = torch.empty((max_total_tokens, cfg.hid_dim), device=device, dtype=cfg.dtype)
    x_max.normal_()
    x_q_max = row_quantization(x_max)  # (fp8, scales)

    grad_output_max = torch.empty((max_total_tokens, cfg.int_dim), device=device, dtype=cfg.dtype)
    grad_output_max.normal_()
    grad_output_q_max = row_quantization(grad_output_max)

    rng = random.Random(0)

    for total_k in tqdm(
        totals,
        desc="DeepGEMM warmup: k_grouped (by total)",
        unit="total",
        disable=disable_pbar,
    ):
        for pattern in patterns:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            wgrad = torch.zeros(
                (num_groups, cfg.int_dim, cfg.hid_dim),
                dtype=torch.bfloat16,
                device=device,
            )

            ks = _make_group_sizes_for_total(
                total_k,
                num_groups=num_groups,
                step=step,
                max_per_group=max_per_group,
                pattern=pattern,
                rng=rng,
            )
            total_tokens = int(sum(ks))  # == total_k

            ks_tensor = torch.tensor(ks, dtype=torch.int32, device=device)
            m_indices = build_m_indices(ks, ks_tensor)

            x_q = (x_q_max[0][:total_tokens], x_q_max[1][:total_tokens])
            x_q = (x_q[0].contiguous(), x_q[1].contiguous())
            grad_output_q = (grad_output_q_max[0][:total_tokens], grad_output_q_max[1][:total_tokens])
            grad_output_q = (grad_output_q[0].contiguous(), grad_output_q[1].contiguous())

            x_q = row2col_deepgemm_layout_fn(x_q[0], m_indices, ks_tensor, x_q[1])
            grad_output_q = row2col_deepgemm_layout_fn(grad_output_q[0], m_indices, ks_tensor, grad_output_q[1])

            grad_output_q = (grad_output_q[0].view(torch.float8_e4m3fn).contiguous(), grad_output_q[1].mT)
            x_q = (x_q[0].view(torch.float8_e4m3fn).contiguous(), x_q[1].mT)

            handle.k_grouped_fp8_gemm_contiguous(
                a=grad_output_q,
                b=x_q,
                d=wgrad,
                ks=ks,
                ks_tensor=ks_tensor,
            )

            clear_tensor_data(*grad_output_q)
            clear_tensor_data(*x_q)

            del ks_tensor, m_indices, ks, wgrad
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - t0
    log.info("[k_grouped] DONE   %d totals in %.1fs (%.2f total/s)",
             len(totals), elapsed, len(totals) / max(elapsed, 1e-9))


def warmup_deep_gemm_before_training(cfg: DeepGemmWarmupCfg, gemm: str) -> None:
    log.info(">>> warmup_deep_gemm_before_training  gemm=%s  "
             "hid=%d  int=%d  groups=%d  max_size=%d  max_buf=%s",
             gemm, cfg.hid_dim, cfg.int_dim, cfg.num_groups,
             cfg.max_size, cfg.max_buffer_tokens)

    if "m_grouped_nt_for_dgrad_transposed" in gemm:
        warmup_m_grouped_sweep_nt_for_dgrad_transposed(
            cfg,
            num_groups=cfg.num_groups,
            min_total=cfg.min_size,
            max_per_group=cfg.max_size,
            step=cfg.step,
        )
    elif "m_grouped" in gemm:
        warmup_m_grouped_sweep(
            cfg,
            num_groups=cfg.num_groups,
            min_total=cfg.min_size,
            max_per_group=cfg.max_size,
            step=cfg.step,
        )
    elif "k_grouped" in gemm:
        warmup_k_grouped_sweep(
            cfg,
            num_groups=cfg.num_groups,
            min_total=cfg.min_size,
            max_per_group=cfg.max_size,
            step=cfg.step,
        )
    else:
        log.warning("Unknown gemm type '%s' — skipped", gemm)
        return

    log.info("<<< warmup_deep_gemm_before_training  gemm=%s  FINISHED", gemm)


def _init_single_process_dist() -> None:
    """Initialize a gloo single-rank process group so dist.get_global_rank() works."""
    if torch.distributed.is_initialized():
        return
    import socket

    # Always allocate a fresh free port — never inherit MASTER_PORT from the
    # environment (a training job may have set it to a port that is still in use).
    os.environ["MASTER_ADDR"] = "localhost"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        os.environ["MASTER_PORT"] = str(_s.getsockname()[1])
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    log.info("[dist] single-process gloo group initialized on port %s", os.environ["MASTER_PORT"])


def _make_cfg(
    cfg: dict,
    gemm: str,
    device: str,
) -> DeepGemmWarmupCfg:
    """
    DeepGemmWarmupCfg uses:
      hid_dim       = input dimension  (K in the GEMM)
      int_dim = output dimension (N in the GEMM)

    Model layers:
      w2: x(total, intermediate) @ W2(intermediate, hidden)  →  (total, hidden)
          → hid_dim=int_dim_model, int_dim=hid_dim_model
      w1: x(total, hidden) @ W1(hidden, 2*intermediate)      →  (total, 2*intermediate)
          → hid_dim=hid_dim_model, int_dim=2*int_dim_model
    """
    if "w2" in gemm:
        h, i = cfg["int_dim"], cfg["hid_dim"]
    elif "w1" in gemm:
        h, i = cfg["hid_dim"], 2 * cfg["int_dim"]
    max_size = cfg.get("_max_size_override") or cfg["max_size"]
    max_buffer_tokens = cfg.get("_max_buffer_tokens")
    return DeepGemmWarmupCfg(
        cache_dir=cfg["cache_dir"],
        device=device,
        dtype=torch.bfloat16,
        hid_dim=h,
        int_dim=i,
        num_groups=cfg["num_groups"],
        step=128,
        min_size=cfg["min_size"],
        max_size=max_size,
        max_buffer_tokens=max_buffer_tokens,
        num_sms=cfg.get("num_sms"),
    )


def _run_single_gemm(cfg: dict, gemm: str, device_id: int) -> None:
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    cfg: DeepGemmWarmupCfg = _make_cfg(cfg, gemm, device)
    log.info("=== Warmup  gemm='%s'  device=%s  hidden=%d  intermediate=%d ===",
             gemm, device, cfg.hid_dim, cfg.int_dim)
    warmup_deep_gemm_before_training(cfg, gemm)
    log.info("=== Done    gemm='%s'  device=%s ===", gemm, device)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepGEMM cache warmup. "
            "Sweeps all (total_tokens, group_sizes) combinations and writes the "
            "compiled kernels to DG_JIT_CACHE_DIR so training starts immediately."
        ),
    )

    # ---- cache & device ----
    parser.add_argument(
        "--config", required=True, metavar="DIR",
        help="Path to the warmup config YAML file.",
    )
    parser.add_argument(
        "--device", type=int, default=0, metavar="ID",
        help="CUDA device index for single-process runs (default: 0).",
    )

    # ---- overrides for stages-coverage pass ----
    parser.add_argument(
        "--max-size-override", type=int, default=None, metavar="N",
        help="Override max_size from the YAML config (used by stages-coverage).",
    )
    parser.add_argument(
        "--max-buffer-tokens", type=int, default=None, metavar="N",
        help="Cap the token-buffer allocation to avoid OOM when "
             "max_size * num_groups is very large.",
    )

    # ---- mode ----
    parser.add_argument(
        "--gemm", metavar="{w1,w2}", default=None,
        help=(
            "Warm a single GEMM variant: "
            "w1 = hidden→2*intermediate (fwd+wgrad), "
            "w2 = intermediate→hidden (fwd+wgrad+dgrad). "
            "Omit to run both variants sequentially."
        ),
    )
    
    args = parser.parse_args()
    cfg = load_config(args.config)["deep_gemm"]

    num_sms = cfg.get("num_sms")
    if num_sms is not None:
        deep_gemm.set_num_sms(int(num_sms))

    if args.max_size_override is not None:
        cfg["_max_size_override"] = args.max_size_override
    if args.max_buffer_tokens is not None:
        cfg["_max_buffer_tokens"] = args.max_buffer_tokens

    final_cache_dir = cfg["cache_dir"]

    gemm_list = [args.gemm] if args.gemm else cfg["gemm_groups"]
    log.info("=" * 72)
    log.info("DeepGEMM warmup START")
    log.info("  config       : %s", args.config)
    log.info("  gemm groups  : %s", gemm_list)
    log.info("  hid_dim      : %d", cfg["hid_dim"])
    log.info("  int_dim      : %d", cfg["int_dim"])
    log.info("  num_groups   : %d", cfg["num_groups"])
    log.info("  max_size     : %d  (override: %s)",
             cfg["max_size"], cfg.get("_max_size_override", "none"))
    log.info("  max_buf_tok  : %s", cfg.get("_max_buffer_tokens", "none"))
    log.info("  cache_dir    : %s", final_cache_dir)
    log.info("  device       : cuda:%d", args.device)
    log.info("=" * 72)

    wall_t0 = time.monotonic()
    completed = []
    failed = []

    with tempfile.TemporaryDirectory(prefix="dg_warmup_") as tmp_cache:
        os.environ["DG_JIT_CACHE_DIR"] = tmp_cache
        log.info("JIT cache redirected to node-local %s (final dest: %s)",
                 tmp_cache, final_cache_dir)

        _init_single_process_dist()
        for gemm in gemm_list:
            try:
                _run_single_gemm(cfg, gemm, args.device)
                completed.append(gemm)
            except Exception:
                log.exception("FAILED  gemm=%s", gemm)
                failed.append(gemm)

        os.makedirs(final_cache_dir, exist_ok=True)
        shutil.copytree(tmp_cache, final_cache_dir, dirs_exist_ok=True)
        log.info("Copied compiled kernels from %s -> %s", tmp_cache, final_cache_dir)

    wall_elapsed = time.monotonic() - wall_t0
    log.info("=" * 72)
    log.info("DeepGEMM warmup SUMMARY  (%.1fs total)", wall_elapsed)
    log.info("  completed : %d / %d  %s",
             len(completed), len(gemm_list), completed)
    if failed:
        log.error("  FAILED    : %d  %s", len(failed), failed)
    else:
        log.info("  FAILED    : 0")
    log.info("  cache_dir : %s", final_cache_dir)
    log.info("=" * 72)

    if failed:
        sys.exit(1)
