"""GigaEvo benchmark adapter for the fixed LLM Foundry experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf

from autoresearch.config_builder import write_effective_config
from autoresearch.mds_provenance import (
    MDS_PROVENANCE_POLICY,
    clear_derived_mds_cache,
    mds_source_tree_sha256,
)
from autoresearch.metrics import read_core_metric

STRUCTURED_FEEDBACK_MARKER = "[gigaevo] structured feedback:"


def _torchrun() -> str:
    configured = os.environ.get("LLMFOUNDRY_TORCHRUN")
    if configured:
        return configured
    environment_root = Path(sys.executable).resolve().parent.parent
    sibling_runtime = (
        environment_root.with_name(f"{environment_root.name}-lmf") / "bin/torchrun"
    )
    if sibling_runtime.is_file():
        return str(sibling_runtime)
    return shutil.which("torchrun") or "torchrun"


def _build_command(
    root: Path,
    config_path: Path,
    gpu_count: int,
    train_overrides: tuple[str, ...] = (),
) -> list[str]:
    return [
        _torchrun(),
        "--standalone",
        "--nproc_per_node",
        str(gpu_count),
        str(root / "vendor/llm-foundry/scripts/train/train.py"),
        str(config_path),
        *train_overrides,
    ]


def _apply_runtime_overrides(
    config_path: Path,
    *,
    train_microbatch_size: int | None,
    eval_batch_size: int | None,
    loader_workers: int | None,
    gradient_log_interval: int | None,
    disable_optimizer_metrics: bool,
) -> dict[str, object]:
    """Apply human-owned performance settings and return their exact contract."""

    config = OmegaConf.load(config_path)
    global_batch_size = int(config.global_train_batch_size)
    if train_microbatch_size is not None:
        if train_microbatch_size < 1:
            raise ValueError("train microbatch size must be at least 1")
        if global_batch_size % train_microbatch_size:
            raise ValueError(
                "train microbatch size must divide global_train_batch_size "
                f"({global_batch_size})"
            )
        config.device_train_microbatch_size = train_microbatch_size
    if eval_batch_size is not None:
        if eval_batch_size < 1:
            raise ValueError("evaluation batch size must be at least 1")
        config.device_eval_batch_size = eval_batch_size
    if loader_workers is not None:
        if loader_workers < 0:
            raise ValueError("loader workers cannot be negative")
        config.train_loader.num_workers = loader_workers
    if gradient_log_interval is not None:
        if gradient_log_interval < 1:
            raise ValueError("gradient log interval must be at least 1")
        config.algorithms.gradient_clipping.diagnostics.log_interval = (
            gradient_log_interval
        )
    if disable_optimizer_metrics:
        config.callbacks.pop("optimizer_monitor", None)

    OmegaConf.save(config, config_path)
    optimizer_monitor = config.callbacks.get("optimizer_monitor")
    return {
        "global_train_batch_size": global_batch_size,
        "device_train_microbatch_size": int(config.device_train_microbatch_size),
        "gradient_accumulation_steps": (
            global_batch_size // int(config.device_train_microbatch_size)
        ),
        "device_eval_batch_size": int(config.device_eval_batch_size),
        "train_loader_num_workers": int(config.train_loader.num_workers),
        "gradient_diagnostics_log_interval": int(
            config.algorithms.gradient_clipping.diagnostics.log_interval
        ),
        "optimizer_monitor_enabled": optimizer_monitor is not None,
    }


def _emit(metrics: dict[str, float], feedback: dict[str, object]) -> None:
    print(
        f"{STRUCTURED_FEEDBACK_MARKER} {json.dumps(feedback, sort_keys=True)}",
        file=sys.stderr,
    )
    print(json.dumps(metrics, sort_keys=True))


def _tail(path: Path, max_characters: int = 4000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_characters * 4))
        return stream.read().decode("utf-8", errors="replace")[-max_characters:]


def _tree_sha256(path: Path) -> str:
    digest = sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(file_path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _parameter_count(stdout_path: Path) -> int | None:
    if not stdout_path.exists():
        return None
    match = re.findall(
        r"\[autoresearch\] n_params=(\d+)",
        stdout_path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(match[-1]) if match else None


def _steady_state_tokens_per_second(
    stderr_path: Path, run_dir: Path | None = None
) -> float | None:
    """Read Composer's last warmed SpeedMonitor sample."""

    if not stderr_path.exists():
        return None
    match = re.findall(
        r"Train throughput/tokens_per_sec:\s*([0-9]+(?:\.[0-9]+)?)",
        stderr_path.read_text(encoding="utf-8", errors="replace"),
    )
    if match:
        return float(match[-1])
    if run_dir is None:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    samples: list[tuple[int, float]] = []
    for event_path in run_dir.rglob("events.out.tfevents.*"):
        try:
            accumulator = EventAccumulator(
                str(event_path), size_guidance={"scalars": 0}
            ).Reload()
            for scalar in accumulator.Scalars("throughput/tokens_per_sec"):
                if math.isfinite(scalar.value):
                    samples.append((scalar.step, float(scalar.value)))
        except (KeyError, OSError):
            continue
    if not samples:
        return None
    return max(samples, key=lambda item: item[0])[1]


def _summarize_gradient_samples(
    gradient_norms: list[float], clipping_events: list[float]
) -> dict[str, float]:
    """Summarize finite gradient telemetry without adding a NumPy dependency."""

    finite_norms = sorted(value for value in gradient_norms if math.isfinite(value))
    finite_clipping = [value for value in clipping_events if math.isfinite(value)]
    if not finite_norms:
        return {}
    percentile_index = max(0, math.ceil(0.95 * len(finite_norms)) - 1)
    metrics = {
        "gradient_norm_pre_clip_max": finite_norms[-1],
        "gradient_norm_pre_clip_p95": finite_norms[percentile_index],
        "gradient_diagnostic_observations": float(len(finite_norms)),
    }
    if finite_clipping:
        metrics["gradient_clipping_fraction"] = sum(finite_clipping) / len(
            finite_clipping
        )
    return metrics


def _gradient_diagnostic_metrics(run_dir: Path) -> dict[str, float]:
    """Read pre-clipping norm and clipping-rate summaries from TensorBoard."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}
    by_tag_and_step: dict[str, dict[int, tuple[float, float]]] = {
        "l2_norm/grad/pre_clip_global": {},
        "gradient_clipping/was_applied": {},
    }
    for event_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        try:
            accumulator = EventAccumulator(
                str(event_path), size_guidance={"scalars": 0}
            ).Reload()
            for tag, values in by_tag_and_step.items():
                for scalar in accumulator.Scalars(tag):
                    previous = values.get(scalar.step)
                    if previous is None or scalar.wall_time >= previous[0]:
                        values[scalar.step] = (
                            float(scalar.wall_time),
                            float(scalar.value),
                        )
        except (KeyError, OSError):
            continue
    gradient_norms = [
        value for _, value in by_tag_and_step["l2_norm/grad/pre_clip_global"].values()
    ]
    clipping_events = [
        value for _, value in by_tag_and_step["gradient_clipping/was_applied"].values()
    ]
    return _summarize_gradient_samples(gradient_norms, clipping_events)


def _heldout_loss_metrics(stderr_path: Path) -> dict[str, float] | None:
    """Parse the fixed validation-loss trajectory printed by Composer."""

    if not stderr_path.exists():
        return None
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"Eval metrics/eval/LanguageCrossEntropy:\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        r"Eval loss/eval/total:\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    )
    values: list[float] = []
    for pattern in patterns:
        values = [float(value) for value in re.findall(pattern, text)]
        values = [value for value in values if math.isfinite(value)]
        if values:
            break
    if not values:
        return None
    tail = values[-min(2, len(values)) :]
    # Evaluations are evenly spaced at 25%, 50%, 75%, and 100%, so the
    # arithmetic mean is the normalized discrete loss AUC used for ranking.
    loss_auc = sum(values) / len(values)
    return {
        "heldout_loss_final": values[-1],
        "heldout_loss_tail_mean": sum(tail) / len(tail),
        "heldout_loss_auc": loss_auc,
        "heldout_evaluations": float(len(values)),
    }


def _failure_details(stderr_path: Path, returncode: int) -> dict[str, object]:
    """Classify common candidate failures without asking an agent to infer them."""

    tail = _tail(stderr_path, max_characters=12_000)
    lowered = tail.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        error_type = "cuda_oom"
    elif "expected mat1 and mat2 to have the same dtype" in lowered or (
        "dtype" in lowered and "bfloat16" in lowered and "float32" in lowered
    ):
        error_type = "dtype_mismatch"
    elif "torch._inductor" in lowered or "triton" in lowered:
        error_type = "kernel_compile_failure"
    elif "no space left on device" in lowered:
        error_type = "storage_exhausted"
    elif "connection" in lowered and ("timeout" in lowered or "refused" in lowered):
        error_type = "network_failure"
    else:
        error_type = "unknown_process_failure"
    return {
        "failure_stage": "training",
        "error_type": error_type,
        "returncode": returncode,
        "stderr_tail": tail[-4000:],
    }


def _gpu_memory_used_mib() -> float | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if not visible or visible == "-1":
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "--id",
        visible,
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _wait_with_gpu_monitor(
    process: subprocess.Popen[object], timeout: float
) -> tuple[int, float | None]:
    """Wait for training while sampling whole-device peak memory."""

    deadline = time.monotonic() + timeout
    peak: float | None = None
    while True:
        used = _gpu_memory_used_mib()
        if used is not None:
            peak = used if peak is None else max(peak, used)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(2.0, remaining)), peak
        except subprocess.TimeoutExpired:
            continue


def _nonfinite_training_loss(stderr_path: Path) -> dict[str, object] | None:
    """Return the first logged non-finite training loss, if present."""

    if not stderr_path.exists():
        return None
    current_batch: int | None = None
    batch_pattern = re.compile(r"\[batch=(\d+)/\d+\]")
    loss_pattern = re.compile(
        r"Train loss/train/(?P<metric>total|loss_ce|z_loss):\s*"
        r"(?P<value>[+-]?(?:nan|inf(?:inity)?))\b",
        re.IGNORECASE,
    )
    with stderr_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            batch_match = batch_pattern.search(line)
            if batch_match is not None:
                current_batch = int(batch_match.group(1))
            loss_match = loss_pattern.search(line)
            if loss_match is not None:
                return {
                    "batch": current_batch,
                    "metric": loss_match.group("metric"),
                    "value": loss_match.group("value").lower(),
                }
    return None


def _nonfinite_training_gradient(stderr_path: Path) -> dict[str, object] | None:
    """Return fail-fast gradient diagnostics emitted by clipping, if present."""

    if not stderr_path.exists():
        return None
    pattern = re.compile(
        r"\[autoresearch\] nonfinite_gradient batch=(?P<batch>\d+) "
        r"parameters=(?P<parameters>\S+)"
    )
    with stderr_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if match is not None:
                return {
                    "batch": int(match.group("batch")),
                    "parameters": match.group("parameters").split(","),
                }
    return None


def _completed_evaluation_after_nonzero_exit(
    stderr_path: Path,
    *,
    expected_batches: int,
    evaluation_provenance: dict[str, object],
) -> dict[str, object] | None:
    """Prove that a nonzero worker exit happened only after a complete evaluation."""

    tasks = evaluation_provenance.get("tasks")
    if not isinstance(tasks, list):
        return None
    expected_labels = {
        str(task["label"])
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("label"), str)
    }
    if len(expected_labels) != len(tasks) or not expected_labels:
        return None

    final_batch = False
    final_time_batch = False
    trainer_done = False
    engine_closed = False
    core_value: float | None = None
    observed_labels: set[str] = set()
    evaluation_pattern = re.compile(
        r"Eval metrics/(?P<label>[^/]+)/[^:]+:\s*"
        r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )
    core_pattern = re.compile(
        r"Train (?:metrics/)?metrics_gauntlet/core:\s*"
        r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )
    with stderr_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if f"[batch={expected_batches}/{expected_batches}]:" in line:
                final_batch = True
            if f"Train time/batch: {expected_batches}" in line:
                final_time_batch = True
            if "| INFO |  Done." in line:
                trainer_done = True
            if "| DEBUG |  Engine closed." in line:
                engine_closed = True
            evaluation_match = evaluation_pattern.search(line)
            if evaluation_match is not None:
                value = float(evaluation_match.group("value"))
                if math.isfinite(value):
                    observed_labels.add(evaluation_match.group("label"))
            core_match = core_pattern.search(line)
            if core_match is not None:
                value = float(core_match.group("value"))
                if math.isfinite(value):
                    core_value = value

    if not (
        final_batch
        and final_time_batch
        and trainer_done
        and engine_closed
        and core_value is not None
        and expected_labels == observed_labels
    ):
        return None
    return {
        "expected_batches": expected_batches,
        "completed_evaluation_labels": sorted(observed_labels),
        "console_core": core_value,
        "trainer_done": True,
        "engine_closed": True,
    }


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _configure_cuda_linker_path(environment: dict[str, str]) -> None:
    """Let Triton's compiler resolve -lcuda without requiring a root symlink."""

    configured = os.environ.get("AUTORESEARCH_CUDA_STUBS")
    candidates = [
        Path(configured) if configured else None,
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs"),
        Path("/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"),
    ]
    cuda_stubs = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None and (candidate / "libcuda.so").is_file()
        ),
        None,
    )
    if cuda_stubs is None:
        return
    existing = environment.get("LIBRARY_PATH")
    entries = [str(cuda_stubs)]
    if existing:
        entries.append(existing)
    environment["LIBRARY_PATH"] = os.pathsep.join(entries)


def _training_overrides(
    *, smoke_batches: int, diagnostic_batches: int
) -> tuple[str, ...]:
    """Build debug-mode overrides for smoke or baseline diagnostic runs."""

    batches = smoke_batches or diagnostic_batches
    if not batches:
        return ()
    overrides = [
        "debug.use_debug=true",
        "debug.startup_speedups.disable_all_evaluators=true",
        f"debug.overrides.max_duration={batches}ba",
        "debug.overrides.console_log_interval=1ba",
        "debug.overrides.save_folder=null",
    ]
    if smoke_batches:
        # Smoke mode measures plumbing/throughput and intentionally avoids all
        # persistent logging. Diagnostic mode retains TensorBoard telemetry.
        overrides.append("debug.overrides.loggers={}")
    return tuple(overrides)


def _apply_screen_contract(
    config_path: Path, *, batches: int, eval_batches: int
) -> dict[str, object]:
    """Make a two-horizon screen use only the fixed held-out LM evaluator."""

    if batches < 1 or eval_batches < 1:
        raise ValueError("screen and evaluation batch counts must be positive")
    config = OmegaConf.load(config_path)
    if "eval_loader" not in config:
        raise ValueError("screening requires eval_loader")
    train_streams = config.get("train_loader", {}).get("dataset", {}).get("streams")
    eval_streams = config.get("eval_loader", {}).get("dataset", {}).get("streams")
    if not OmegaConf.is_dict(train_streams) or len(train_streams) != 1:
        raise ValueError("screening requires exactly one training stream")
    if not OmegaConf.is_dict(eval_streams) or len(eval_streams) != 1:
        raise ValueError("screening requires exactly one validation stream")
    train_stream = next(iter(train_streams.values()))
    eval_stream = next(iter(eval_streams.values()))
    screen_root = str(eval_stream["local"])
    train_stream["local"] = screen_root
    train_stream["split"] = "train"
    interval = max(1, batches // 4)
    config.max_duration = f"{batches}ba"
    config.eval_interval = f"{interval}ba"
    config.eval_before_train = "skip"
    config.eval_subset_num_batches = eval_batches
    config.console_log_interval = min(20, interval)
    config.pop("icl_tasks", None)
    config.pop("eval_gauntlet", None)
    config.pop("cached_datasets_directory_template", None)
    OmegaConf.save(config, config_path)
    return {
        "mode": "heldout_loss_screen",
        "training_batches": batches,
        "evaluation_interval_batches": interval,
        "evaluation_subset_batches": eval_batches,
        "expected_evaluations": math.ceil(batches / interval),
        "training_root": screen_root,
        "training_split": "train",
        "validation_split": str(eval_stream["split"]),
    }


def _absolute_stacked_schedule(
    schedule_rows: list[object], *, reference_batches: int
) -> list[list[object]]:
    """Resolve fractional scheduler boundaries against one fixed full horizon."""

    if reference_batches < 1:
        raise ValueError("schedule reference must be positive")
    resolved: list[list[object]] = []
    previous = 0
    for raw_row in schedule_rows:
        if not isinstance(raw_row, list | tuple) or len(raw_row) != 3:
            raise ValueError("stacked scheduler rows must have three entries")
        raw_boundary, target, kind = raw_row
        boundary_value = float(raw_boundary)
        boundary = (
            int(boundary_value * reference_batches)
            if -1.0 < boundary_value <= 1.0
            else int(boundary_value)
        )
        if boundary <= previous:
            raise ValueError("resolved scheduler boundaries must be increasing")
        if kind not in {"linear", "cosine"}:
            raise ValueError(f"unsupported stacked scheduler segment: {kind}")
        resolved.append([boundary, float(target), str(kind)])
        previous = boundary
    return resolved


def _apply_schedule_reference(
    config_path: Path, *, reference_batches: int
) -> dict[str, object]:
    """Keep the main LR trajectory identical across checkpoint rungs."""

    config = OmegaConf.load(config_path)
    scheduler = config.get("scheduler")
    if scheduler is None or scheduler.get("name") != "stacked":
        raise ValueError("multi-fidelity runs require the stacked scheduler")
    rows = OmegaConf.to_container(scheduler.get("schedule_rows"), resolve=True)
    if not isinstance(rows, list):
        raise ValueError("stacked scheduler is missing schedule_rows")
    absolute_rows = _absolute_stacked_schedule(
        rows, reference_batches=reference_batches
    )
    scheduler.schedule_rows = absolute_rows
    OmegaConf.save(config, config_path)
    return {
        "reference_batches": reference_batches,
        "absolute_schedule_rows": absolute_rows,
    }


def _schedule_value(rows: list[list[object]], step: int) -> float:
    first_step = 0
    first_value = 0.0
    for boundary, raw_target, kind in rows:
        last_step = int(boundary)
        target = float(raw_target)
        if step < last_step:
            fraction = (step - first_step) / (last_step - first_step)
            if kind == "linear":
                return first_value + (target - first_value) * fraction
            if kind == "cosine":
                weight = (1.0 - math.cos(fraction * math.pi)) / 2.0
                return first_value + (target - first_value) * weight
            raise ValueError(f"unsupported stacked scheduler segment: {kind}")
        first_step = last_step
        first_value = target
    return first_value


def _apply_convergence_probe_contract(
    config_path: Path, *, fork_batch: int, probe_batches: int
) -> dict[str, object]:
    """Replace the post-fork LR path with an isolated cosine descent to zero."""

    if fork_batch < 1 or probe_batches < 1:
        raise ValueError("probe fork and duration must be positive")
    config = OmegaConf.load(config_path)
    scheduler = config.get("scheduler")
    if scheduler is None or scheduler.get("name") != "stacked":
        raise ValueError("convergence probes require the stacked scheduler")
    raw_rows = OmegaConf.to_container(scheduler.get("schedule_rows"), resolve=True)
    if not isinstance(raw_rows, list):
        raise ValueError("stacked scheduler is missing schedule_rows")
    rows = [list(row) for row in raw_rows]
    if any(float(row[0]) <= 1.0 for row in rows):
        raise ValueError("freeze the scheduler horizon before applying a probe")

    fork_value = _schedule_value(rows, fork_batch)
    probe_rows = [row for row in rows if int(row[0]) < fork_batch]
    probe_rows.append([fork_batch, fork_value, "linear"])
    end_batch = fork_batch + probe_batches
    probe_rows.append([end_batch, 0.0, "cosine"])
    scheduler.schedule_rows = probe_rows
    config.max_duration = f"{end_batch}ba"
    # Composer intervals are evaluated against the restored absolute timestamp.
    # Using the absolute end guarantees that the one probe evaluation observes
    # the zero-LR endpoint rather than an intermediate batch.
    config.eval_interval = f"{end_batch}ba"
    config.console_log_interval = min(20, probe_batches)
    OmegaConf.save(config, config_path)
    return {
        "fork_batch": fork_batch,
        "probe_batches": probe_batches,
        "end_batch": end_batch,
        "evaluation_interval_batches": end_batch,
        "expected_evaluations": 1,
        "fork_lr_multiplier": fork_value,
        "schedule_rows": probe_rows,
    }


def _apply_checkpoint_contract(
    config_path: Path,
    *,
    load_path: Path | None,
    save_folder: Path | None,
    save_interval_batches: int | None,
) -> dict[str, object]:
    """Configure full-state continuation without allowing probe-side writes."""

    if save_interval_batches is not None and save_interval_batches < 1:
        raise ValueError("checkpoint save interval must be positive")
    if (save_folder is None) != (save_interval_batches is None):
        raise ValueError(
            "--save-folder and --save-interval-batches must be supplied together"
        )
    config = OmegaConf.load(config_path)
    config.load_path = str(load_path.resolve()) if load_path is not None else None
    config.load_weights_only = False
    config.autoresume = False
    config.save_folder = str(save_folder.resolve()) if save_folder is not None else None
    if save_folder is not None:
        assert save_interval_batches is not None
        config.save_interval = f"{save_interval_batches}ba"
        config.save_num_checkpoints_to_keep = 2
        config.save_latest_filename = "latest-rank{rank}.pt"
        config.compute_checkpoint_hashes = True
    OmegaConf.save(config, config_path)
    return {
        "load_path": str(load_path.resolve()) if load_path is not None else None,
        "save_folder": (
            str(save_folder.resolve()) if save_folder is not None else None
        ),
        "save_interval_batches": save_interval_batches,
        "full_state_resume": load_path is not None,
        "writes_checkpoint": save_folder is not None,
    }


def _heldout_provenance(config_path: Path) -> dict[str, object]:
    """Prove that screen training and validation use disjoint pinned shards."""

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Effective training config must be a mapping")
    train_streams = config["train_loader"]["dataset"]["streams"]
    if not isinstance(train_streams, dict) or len(train_streams) != 1:
        raise ValueError("screening requires exactly one training stream")
    train_stream = next(iter(train_streams.values()))
    eval_streams = config["eval_loader"]["dataset"]["streams"]
    if not isinstance(eval_streams, dict) or len(eval_streams) != 1:
        raise ValueError("screening requires exactly one validation stream")
    eval_stream = next(iter(eval_streams.values()))
    train_root = Path(str(train_stream["local"]))
    eval_root = Path(str(eval_stream["local"]))
    if train_root.resolve() != eval_root.resolve():
        raise ValueError(
            "screen train and validation roots must share one split manifest"
        )
    train_index = train_root / str(train_stream["split"]) / "index.json"
    eval_index = eval_root / str(eval_stream["split"]) / "index.json"
    split_manifest_path = train_root / "holdout-split-manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train = json.loads(train_index.read_text(encoding="utf-8"))
    validation = json.loads(eval_index.read_text(encoding="utf-8"))

    def basenames(index: dict[str, object]) -> set[str]:
        shards = index.get("shards")
        if not isinstance(shards, list):
            raise ValueError("MDS split index is missing shards")
        return {
            str(shard["zip_data"]["basename"])
            for shard in shards
            if isinstance(shard, dict) and isinstance(shard.get("zip_data"), dict)
        }

    overlap = basenames(train) & basenames(validation)
    train_hash = sha256(train_index.read_bytes()).hexdigest()
    validation_hash = sha256(eval_index.read_bytes()).hexdigest()
    if overlap:
        raise ValueError(f"held-out split overlaps training shards: {sorted(overlap)}")
    if split_manifest.get("train_index_sha256") != train_hash:
        raise ValueError("held-out split train index changed")
    if split_manifest.get("validation_index_sha256") != validation_hash:
        raise ValueError("held-out split validation index changed")
    validation_tree_hash = mds_source_tree_sha256(eval_index.parent, validation)
    if split_manifest.get("validation_tree_sha256") != validation_tree_hash:
        raise ValueError("held-out split validation shards changed")
    return {
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256(split_manifest_path.read_bytes()).hexdigest(),
        "train_index": str(train_index),
        "train_index_sha256": train_hash,
        "validation_index": str(eval_index),
        "validation_index_sha256": validation_hash,
        "validation_tree_sha256": validation_tree_hash,
        "overlapping_compressed_shards": [],
        "validation_samples": split_manifest.get("validation_samples"),
    }


def _dataset_capacity(config_path: Path) -> dict[str, object]:
    """Resolve the fixed MDS corpus capacity and requested training tokens."""

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Effective training config must be a mapping")
    sequence_length = int(config["max_seq_len"])
    global_batch_size = int(config["global_train_batch_size"])
    duration = str(config["max_duration"])
    duration_match = re.fullmatch(r"(\d+)ba", duration)
    if duration_match is None:
        raise ValueError(f"Scientific max_duration must use batch units: {duration}")
    batches = int(duration_match.group(1))

    train_loader = config.get("train_loader")
    if not isinstance(train_loader, dict):
        raise ValueError("Effective config is missing train_loader")
    dataset = train_loader.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Effective config is missing train_loader.dataset")
    streams = dataset.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("Effective config is missing training streams")
    explicit_epoch_size = dataset.get("epoch_size")
    weighted_streams = [
        str(name)
        for name, stream in streams.items()
        if isinstance(stream, dict)
        and any(key in stream for key in ("proportion", "repeat", "choose"))
    ]
    uses_distinct_samples = explicit_epoch_size is None and not weighted_streams

    total_samples = 0
    indexes: list[str] = []
    corpus_manifests: list[dict[str, object]] = []
    for stream in streams.values():
        if not isinstance(stream, dict):
            raise ValueError("Training stream configuration must be a mapping")
        local = Path(str(stream["local"]))
        split = str(stream.get("split", ""))
        index_path = local / split / "index.json" if split else local / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = index.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"Invalid MDS index: {index_path}")
        total_samples += sum(int(shard["samples"]) for shard in shards)
        indexes.append(str(index_path))
        manifest_path = local / "corpus-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_samples = int(manifest["samples"])
            manifest_tokens = int(manifest["tokens"])
            manifest_sequence_length = int(manifest["sequence_length"])
            if manifest_samples != sum(int(shard["samples"]) for shard in shards):
                raise ValueError(f"Corpus manifest sample mismatch: {manifest_path}")
            if manifest_tokens != manifest_samples * manifest_sequence_length:
                raise ValueError(f"Corpus manifest token mismatch: {manifest_path}")
            if manifest_sequence_length != sequence_length:
                raise ValueError(
                    f"Corpus manifest sequence-length mismatch: {manifest_path}"
                )
            index_hash = sha256(index_path.read_bytes()).hexdigest()
            if manifest.get("mds_index_sha256") != index_hash:
                raise ValueError(
                    f"Corpus manifest index hash mismatch: {manifest_path}"
                )
            if manifest.get("mds_provenance_policy") != MDS_PROVENANCE_POLICY:
                raise ValueError(
                    f"Corpus manifest has an obsolete MDS provenance policy: "
                    f"{manifest_path}"
                )
            mds_tree_hash = mds_source_tree_sha256(index_path.parent, index)
            if manifest.get("mds_tree_sha256") != mds_tree_hash:
                raise ValueError(
                    f"Corpus manifest MDS tree hash mismatch: {manifest_path}"
                )
            source_manifest_path = Path(str(manifest["source_manifest"]))
            source_manifest_hash = sha256(source_manifest_path.read_bytes()).hexdigest()
            if manifest.get("source_manifest_sha256") != source_manifest_hash:
                raise ValueError(
                    f"Corpus source-manifest hash mismatch: {manifest_path}"
                )
            tokenizer_path = Path(str(manifest["tokenizer"]))
            tokenizer_hash = _tree_sha256(tokenizer_path)
            if manifest.get("tokenizer_sha256") != tokenizer_hash:
                raise ValueError(f"Corpus tokenizer hash mismatch: {manifest_path}")
            expected_boundary = {
                "policy": "no_implicit_special_tokens_plus_one_explicit_eos",
                "implicit_empty_ids": [],
                "explicit_eos_ids": [0],
                "eos_token_id": 0,
            }
            if manifest.get("tokenizer_boundary") != expected_boundary:
                raise ValueError(
                    f"Corpus tokenizer-boundary policy mismatch: {manifest_path}"
                )
            if (
                manifest.get("eos_text") != "<|endoftext|>"
                or manifest.get("tokens_dtype") != "uint16"
                or manifest.get("compression") != "zstd"
            ):
                raise ValueError(f"Corpus encoding policy mismatch: {manifest_path}")
            corpus_manifests.append(
                {
                    "path": str(manifest_path),
                    "sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                    "source_manifest_sha256": source_manifest_hash,
                    "tokenizer_sha256": tokenizer_hash,
                    "mds_index_sha256": index_hash,
                    "mds_tree_sha256": mds_tree_hash,
                }
            )

    available_tokens = total_samples * sequence_length
    required_tokens = batches * global_batch_size * sequence_length
    return {
        "available_samples": total_samples,
        "available_tokens": available_tokens,
        "required_tokens": required_tokens,
        "sequence_length": sequence_length,
        "global_batch_size": global_batch_size,
        "batches": batches,
        "index_paths": indexes,
        "corpus_manifests": corpus_manifests,
        "explicit_epoch_size": explicit_epoch_size,
        "weighted_streams": weighted_streams,
        "uses_distinct_samples": uses_distinct_samples,
        "has_complete_provenance": len(corpus_manifests) == len(streams),
        "is_sufficient": available_tokens >= required_tokens,
    }


def _evaluation_provenance(config_path: Path) -> dict[str, object]:
    """Hash the exact CORE task configuration and source JSONL fixtures."""

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Effective training config must be a mapping")
    tasks = config.get("icl_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Effective config is missing ICL evaluation tasks")
    records: list[dict[str, object]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("ICL task configuration must be a mapping")
        dataset_path = Path(str(task["dataset_uri"]))
        record = {
            "label": str(task["label"]),
            "dataset_uri": str(dataset_path),
            "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_bytes": dataset_path.stat().st_size,
            "num_fewshot": task.get("num_fewshot"),
            "fewshot_random_seed": task.get("fewshot_random_seed", 1234),
            "icl_task_type": task.get("icl_task_type"),
            "continuation_delimiter": task.get("continuation_delimiter"),
            "gauntlet_tags": task.get("gauntlet_tags"),
        }
        records.append(record)
    gauntlet_weighting = (config.get("eval_gauntlet") or {}).get("weighting")
    contract = {
        "gauntlet_weighting": gauntlet_weighting,
        "tasks": records,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "contract_sha256": sha256(serialized.encode()).hexdigest(),
        "gauntlet_weighting": gauntlet_weighting,
        "tasks": records,
    }


def _clear_dataset_caches(dataset_capacity: dict[str, object]) -> dict[str, int]:
    removed_files = 0
    removed_bytes = 0
    for raw_index_path in dataset_capacity["index_paths"]:
        cleanup = clear_derived_mds_cache(Path(str(raw_index_path)).parent)
        removed_files += cleanup["removed_files"]
        removed_bytes += cleanup["removed_bytes"]
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("autoresearch/config/dclm-140m.yaml"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("autoresearch/config/candidate.yaml"),
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--train-microbatch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--loader-workers", type=int)
    parser.add_argument("--gradient-log-interval", type=int)
    parser.add_argument("--disable-optimizer-metrics", action="store_true")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--smoke-batches",
        type=int,
        default=int(os.environ.get("AUTORESEARCH_SMOKE_BATCHES", "0")),
        help=(
            "Train this many batches and skip evaluation/checkpointing. "
            "Defaults to AUTORESEARCH_SMOKE_BATCHES, or zero."
        ),
    )
    run_mode.add_argument(
        "--diagnostic-batches",
        type=int,
        default=int(os.environ.get("AUTORESEARCH_DIAGNOSTIC_BATCHES", "0")),
        help=(
            "Train this many baseline batches without evaluation/checkpointing "
            "while retaining TensorBoard diagnostic metrics."
        ),
    )
    run_mode.add_argument(
        "--screen-batches",
        type=int,
        default=int(os.environ.get("AUTORESEARCH_SCREEN_BATCHES", "0")),
        help=(
            "Train this many batches and rank by fixed held-out LM loss. "
            "The idea campaign uses only 1024 and 4096."
        ),
    )
    parser.add_argument(
        "--screen-eval-batches",
        type=int,
        default=int(os.environ.get("AUTORESEARCH_SCREEN_EVAL_BATCHES", "32")),
    )
    parser.add_argument(
        "--schedule-reference-batches",
        type=int,
        help="Resolve fractional LR boundaries against this fixed full horizon.",
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        help="Resume model, optimizer, timestamp, and RNG from this checkpoint.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        help="Write a resumable main-branch checkpoint to this isolated folder.",
    )
    parser.add_argument("--save-interval-batches", type=int)
    parser.add_argument("--probe-from-batch", type=int)
    parser.add_argument("--probe-batches", type=int)
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(
            os.environ.get("AUTORESEARCH_EXPERIMENT_TIMEOUT_SECONDS", 12 * 60 * 60)
        ),
        help=(
            "Maximum wall time for one baseline or candidate run. Defaults to "
            "AUTORESEARCH_EXPERIMENT_TIMEOUT_SECONDS, or 12 hours."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/current"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpus < 1:
        parser.error("--gpus must be at least 1")

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    effective_config = write_effective_config(
        args.base_config,
        args.candidate_config,
        run_dir / "effective-config.yaml",
    )
    screen_contract: dict[str, object] | None = None
    if args.screen_batches:
        try:
            screen_contract = _apply_screen_contract(
                effective_config,
                batches=args.screen_batches,
                eval_batches=args.screen_eval_batches,
            )
        except ValueError as exc:
            parser.error(str(exc))
    schedule_contract: dict[str, object] | None = None
    probe_contract: dict[str, object] | None = None
    checkpoint_contract: dict[str, object] | None = None
    try:
        if args.schedule_reference_batches is not None:
            schedule_contract = _apply_schedule_reference(
                effective_config,
                reference_batches=args.schedule_reference_batches,
            )
        if (args.probe_from_batch is None) != (args.probe_batches is None):
            raise ValueError(
                "--probe-from-batch and --probe-batches must be supplied together"
            )
        if args.probe_from_batch is not None and args.probe_batches is not None:
            expected_end = args.probe_from_batch + args.probe_batches
            if args.screen_batches != expected_end:
                raise ValueError(
                    "a probe screen horizon must equal fork batch plus probe batches"
                )
            if args.save_folder is not None:
                raise ValueError("a convergence probe cannot write checkpoints")
            probe_contract = _apply_convergence_probe_contract(
                effective_config,
                fork_batch=args.probe_from_batch,
                probe_batches=args.probe_batches,
            )
            screen_contract = {
                "mode": "convergence_probe",
                "training_batches": expected_end,
                "evaluation_interval_batches": expected_end,
                "evaluation_subset_batches": args.screen_eval_batches,
                "expected_evaluations": 1,
            }
        checkpoint_contract = _apply_checkpoint_contract(
            effective_config,
            load_path=args.load_path,
            save_folder=args.save_folder,
            save_interval_batches=args.save_interval_batches,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        runtime_contract = _apply_runtime_overrides(
            effective_config,
            train_microbatch_size=args.train_microbatch_size,
            eval_batch_size=args.eval_batch_size,
            loader_workers=args.loader_workers,
            gradient_log_interval=args.gradient_log_interval,
            disable_optimizer_metrics=args.disable_optimizer_metrics,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke_batches < 0:
        parser.error("--smoke-batches must be positive")
    if args.diagnostic_batches < 0:
        parser.error("--diagnostic-batches must be positive")
    if args.screen_batches < 0:
        parser.error("--screen-batches must be positive")
    train_overrides = _training_overrides(
        smoke_batches=args.smoke_batches,
        diagnostic_batches=args.diagnostic_batches,
    )
    command = _build_command(
        root,
        effective_config,
        args.gpus,
        train_overrides,
    )
    try:
        dataset_capacity = _dataset_capacity(effective_config)
    except Exception as exc:
        _emit(
            {"fitness": 0.0, "is_valid": 0.0},
            {
                "status": "dataset_preflight_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "effective_config": str(effective_config),
            },
        )
        return 0
    try:
        evaluation_provenance = (
            _heldout_provenance(effective_config)
            if args.screen_batches
            else _evaluation_provenance(effective_config)
        )
    except Exception as exc:
        _emit(
            {"fitness": 0.0, "is_valid": 0.0},
            {
                "status": "evaluation_preflight_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "effective_config": str(effective_config),
                "dataset_capacity": dataset_capacity,
            },
        )
        return 0
    execution_contract = {
        "dataset": dataset_capacity,
        "evaluation": evaluation_provenance,
        "runtime": runtime_contract,
        **({"screen": screen_contract} if screen_contract is not None else {}),
        **(
            {"schedule_reference": schedule_contract}
            if schedule_contract is not None
            else {}
        ),
        **({"convergence_probe": probe_contract} if probe_contract is not None else {}),
        **(
            {"checkpoint": checkpoint_contract}
            if checkpoint_contract is not None
            else {}
        ),
    }
    if args.dry_run:
        _emit(
            {"fitness": 0.0, "is_valid": 1.0},
            {
                "status": "dry_run",
                "command": command,
                "effective_config": str(effective_config),
                "execution_contract": execution_contract,
            },
        )
        return 0
    if (
        not args.smoke_batches
        and not args.diagnostic_batches
        and (
            not dataset_capacity["is_sufficient"]
            or not dataset_capacity["has_complete_provenance"]
            or not dataset_capacity["uses_distinct_samples"]
        )
    ):
        _emit(
            {"fitness": 0.0, "is_valid": 0.0},
            {
                "status": (
                    "insufficient_training_corpus"
                    if not dataset_capacity["is_sufficient"]
                    else "unproven_training_corpus"
                    if not dataset_capacity["has_complete_provenance"]
                    else "resampled_training_corpus"
                ),
                "dataset_capacity": dataset_capacity,
                "execution_contract": execution_contract,
                "effective_config": str(effective_config),
            },
        )
        return 0

    if dataset_capacity["has_complete_provenance"]:
        try:
            dataset_capacity["derived_cache_cleanup"] = _clear_dataset_caches(
                dataset_capacity
            )
        except Exception as exc:
            _emit(
                {"fitness": 0.0, "is_valid": 0.0},
                {
                    "status": "dataset_cache_cleanup_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "effective_config": str(effective_config),
                    "execution_contract": execution_contract,
                },
            )
            return 0

    environment = os.environ.copy()
    python_paths = [str(root), str(root / "vendor/llm-foundry")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["AUTORESEARCH_RUN_ROOT"] = str(run_dir)
    environment.setdefault("RUN_NAME", run_dir.name)
    _configure_cuda_linker_path(environment)
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train.stdout.log"
    stderr_path = run_dir / "train.stderr.log"
    started = time.monotonic()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        try:
            returncode, peak_gpu_memory_mib = _wait_with_gpu_monitor(
                process, args.timeout
            )
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            elapsed = time.monotonic() - started
            _emit(
                {"fitness": 0.0, "is_valid": 0.0, "wall_time_seconds": elapsed},
                {
                    "status": "timeout",
                    "timeout_seconds": args.timeout,
                    "failure_stage": "training",
                    "error_type": "timeout",
                    "stderr_tail": _tail(stderr_path),
                    "execution_contract": execution_contract,
                },
            )
            return 0

    elapsed = time.monotonic() - started
    n_params = _parameter_count(stdout_path)
    nonfinite_gradient = _nonfinite_training_gradient(stderr_path)
    if nonfinite_gradient is not None:
        _emit(
            {"fitness": 0.0, "is_valid": 0.0, "wall_time_seconds": elapsed},
            {
                "status": "nonfinite_training_gradient",
                "nonfinite_gradient": nonfinite_gradient,
                "returncode": returncode,
                "stderr_tail": _tail(stderr_path),
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0
    nonfinite_loss = _nonfinite_training_loss(stderr_path)
    if nonfinite_loss is not None:
        _emit(
            {"fitness": 0.0, "is_valid": 0.0, "wall_time_seconds": elapsed},
            {
                "status": "nonfinite_training_loss",
                "nonfinite_loss": nonfinite_loss,
                "returncode": returncode,
                "stderr_tail": _tail(stderr_path),
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0
    post_completion_failure = None
    if (
        returncode != 0
        and not args.smoke_batches
        and not args.diagnostic_batches
        and not args.screen_batches
    ):
        post_completion_failure = _completed_evaluation_after_nonzero_exit(
            stderr_path,
            expected_batches=int(dataset_capacity["batches"]),
            evaluation_provenance=evaluation_provenance,
        )
    if returncode != 0 and post_completion_failure is None:
        failure = _failure_details(stderr_path, returncode)
        _emit(
            {
                "fitness": 0.0,
                "is_valid": 0.0,
                "wall_time_seconds": elapsed,
                **(
                    {"peak_gpu_memory_mib": peak_gpu_memory_mib}
                    if peak_gpu_memory_mib is not None
                    else {}
                ),
            },
            {
                "status": "train_failed",
                **failure,
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0

    if args.screen_batches:
        heldout = _heldout_loss_metrics(stderr_path)
        if heldout is None:
            _emit(
                {
                    "fitness": 0.0,
                    "is_valid": 0.0,
                    "wall_time_seconds": elapsed,
                    **(
                        {"peak_gpu_memory_mib": peak_gpu_memory_mib}
                        if peak_gpu_memory_mib is not None
                        else {}
                    ),
                },
                {
                    "status": "heldout_metric_missing",
                    "failure_stage": "evaluation",
                    "error_type": "metric_missing",
                    "stderr_tail": _tail(stderr_path),
                    "n_params": n_params,
                    "execution_contract": execution_contract,
                },
            )
            return 0
        fitness = 1.0 / (1.0 + heldout["heldout_loss_auc"])
        steady_state_tokens_per_second = _steady_state_tokens_per_second(
            stderr_path, run_dir
        )
        gradient_diagnostics = _gradient_diagnostic_metrics(run_dir)
        _emit(
            {
                "fitness": fitness,
                "is_valid": 1.0,
                "llmfoundry_core_equal_raw": 0.0,
                **heldout,
                "wall_time_seconds": elapsed,
                "steady_state_tokens_per_second": (
                    steady_state_tokens_per_second or 0.0
                ),
                "peak_gpu_memory_mib": (
                    peak_gpu_memory_mib if peak_gpu_memory_mib is not None else 81920.0
                ),
                **gradient_diagnostics,
                **({"n_params": float(n_params)} if n_params is not None else {}),
            },
            {
                "status": "screen_complete",
                "screen_batches": args.screen_batches,
                "fitness_definition": "1 / (1 + heldout_loss_auc)",
                "effective_config": str(effective_config),
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0

    if args.smoke_batches:
        smoke_tokens = args.smoke_batches * 16 * 2048
        steady_state_tokens_per_second = _steady_state_tokens_per_second(
            stderr_path, run_dir
        )
        _emit(
            {
                "fitness": 0.0,
                "is_valid": 1.0,
                # Smoke mode deliberately skips CORE. Emit its neutral
                # placeholder so the retained required-metrics stage records a
                # valid execution card instead of replacing all fields with
                # invalid sentinels. Smoke fitness is never used for ranking.
                "llmfoundry_core_equal_raw": 0.0,
                "wall_time_seconds": elapsed,
                **(
                    {"peak_gpu_memory_mib": peak_gpu_memory_mib}
                    if peak_gpu_memory_mib is not None
                    else {}
                ),
                "smoke_tokens_per_second_including_startup": smoke_tokens / elapsed,
                **(
                    {"steady_state_tokens_per_second": (steady_state_tokens_per_second)}
                    if steady_state_tokens_per_second is not None
                    else {}
                ),
                **({"n_params": float(n_params)} if n_params is not None else {}),
            },
            {
                "status": "smoke_complete",
                "smoke_batches": args.smoke_batches,
                "smoke_tokens": smoke_tokens,
                "effective_config": str(effective_config),
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0

    if args.diagnostic_batches:
        diagnostic_tokens = args.diagnostic_batches * 16 * 2048
        _emit(
            {
                "fitness": 0.0,
                "is_valid": 1.0,
                "llmfoundry_core_equal_raw": 0.0,
                "wall_time_seconds": elapsed,
                **({"n_params": float(n_params)} if n_params is not None else {}),
            },
            {
                "status": "diagnostic_complete",
                "diagnostic_batches": args.diagnostic_batches,
                "diagnostic_tokens": diagnostic_tokens,
                "effective_config": str(effective_config),
                "n_params": n_params,
                "execution_contract": execution_contract,
            },
        )
        return 0

    try:
        core_value, core_tag, event_file = read_core_metric(run_dir)
    except Exception as exc:
        _emit(
            {"fitness": 0.0, "is_valid": 0.0, "wall_time_seconds": elapsed},
            {
                "status": "metric_missing",
                "error": f"{type(exc).__name__}: {exc}",
                "execution_contract": execution_contract,
            },
        )
        return 0

    if not math.isfinite(core_value):
        _emit(
            {"fitness": 0.0, "is_valid": 0.0, "wall_time_seconds": elapsed},
            {
                "status": "nonfinite_core_metric",
                "core_tag": core_tag,
                "event_file": str(event_file),
                "execution_contract": execution_contract,
            },
        )
        return 0

    _emit(
        {
            "fitness": core_value,
            "is_valid": 1.0,
            "llmfoundry_core_equal_raw": core_value,
            "wall_time_seconds": elapsed,
            **(
                {"peak_gpu_memory_mib": peak_gpu_memory_mib}
                if peak_gpu_memory_mib is not None
                else {}
            ),
            **({"n_params": float(n_params)} if n_params is not None else {}),
        },
        {
            "status": "complete",
            "core_tag": core_tag,
            "event_file": str(event_file),
            "effective_config": str(effective_config),
            "n_params": n_params,
            "execution_contract": execution_contract,
            **(
                {
                    "post_completion_failure": {
                        "returncode": returncode,
                        "stderr_tail": _tail(stderr_path),
                        **post_completion_failure,
                    }
                }
                if post_completion_failure is not None
                else {}
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
