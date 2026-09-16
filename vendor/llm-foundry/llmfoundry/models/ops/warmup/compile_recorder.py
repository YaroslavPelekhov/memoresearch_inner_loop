import functools
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
from triton import knobs

_tls = threading.local()


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _summarize_value(x):
    info = {"py_type": type(x).__name__}

    # tensor-like objects (torch.Tensor, etc.)
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        info["dtype"] = str(getattr(x, "dtype", None))
        info["shape"] = _safe_call(lambda: list(x.shape))
        info["device"] = str(getattr(x, "device", None))
        info["stride"] = _safe_call(lambda: list(x.stride())) if hasattr(x, "stride") else None
        info["numel"] = _safe_call(lambda: int(x.numel())) if hasattr(x, "numel") else None

        if hasattr(x, "element_size") and info.get("numel") is not None:
            info["nbytes"] = _safe_call(lambda: int(x.numel() * x.element_size()))
        else:
            info["nbytes"] = None

        return info

    # scalars / constexpr / launch params
    if isinstance(x, (int, float, bool, str)):
        info["value"] = x
        return info

    info["repr"] = repr(x)[:200]
    return info


def _summarize_call(args, kwargs):
    return {
        "args": [_summarize_value(x) for x in args],
        "kwargs": {k: _summarize_value(v) for k, v in kwargs.items()},
    }


class _CompileInputRecorder:
    def __init__(self, kernel, sidecar_name="runtime_inputs.json"):
        self._kernel = kernel
        self._sidecar_name = sidecar_name

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        launcher = self._kernel[grid]

        @functools.wraps(launcher)
        def wrapped(*args, **kwargs):
            ctx = {
                "inputs": _summarize_call(args, kwargs),
                "sidecar_name": self._sidecar_name,
            }

            if not hasattr(_tls, "stack"):
                _tls.stack = []
            _tls.stack.append(ctx)

            prev_listener = knobs.compilation.listener

            def listener(*, src, metadata, metadata_group, times, cache_hit):
                # get previuous listener, if exists
                if prev_listener is not None:
                    prev_listener(
                        src=src,
                        metadata=metadata,
                        metadata_group=metadata_group,
                        times=times,
                        cache_hit=cache_hit,
                    )

                # skip, if cache_hit=True
                if cache_hit:
                    return

                stack = getattr(_tls, "stack", [])
                if not stack:
                    return

                active = stack[-1]

                record = {
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "kernel_name": metadata.get("name", getattr(src, "name", None)),
                    "cache_hit": cache_hit,
                    "target": str(metadata.get("target")),
                    "compile_times_us": {
                        "total": getattr(times, "total", None),
                        "ir_initialization": getattr(times, "ir_initialization", None),
                        "lowering_stages": dict(getattr(times, "lowering_stages", [])),
                        "store_results": getattr(times, "store_results", None),
                    },
                    "inputs": active["inputs"],
                }

                # write file next to Triton cache artifacts of this kernel variant.
                if metadata_group:
                    any_artifact = next(iter(metadata_group.values()), None)
                    if any_artifact is not None:
                        cache_dir = Path(any_artifact).parent
                        sidecar_path = cache_dir / active["sidecar_name"]

                        if torch.distributed.get_rank() == 0:
                            if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
                                with open(sidecar_path, "r", encoding="utf-8") as f:
                                    prev_records = json.load(f)
                                if not isinstance(prev_records, list):
                                    prev_records = [prev_records] if prev_records else []
                                prev_records.append(record)
                                with open(sidecar_path, "w", encoding="utf-8") as f:
                                    json.dump(prev_records, f)
                            else:
                                with open(sidecar_path, "w", encoding="utf-8") as f:
                                    json.dump([record], f)

            try:
                # listener is global, so limit its scope to the current call.
                with knobs.compilation.scope():
                    knobs.compilation.listener = listener
                    return launcher(*args, **kwargs)
            finally:
                _tls.stack.pop()

        return wrapped


def save_inputs_on_real_compile(sidecar_name="runtime_inputs.json"):
    """
    Decorator to be placed outside of Triton kernel / autotune object.
    """
    if not sidecar_name:
        return lambda kernel: kernel
    def decorator(kernel):
        return _CompileInputRecorder(kernel, sidecar_name=sidecar_name)
    return decorator
