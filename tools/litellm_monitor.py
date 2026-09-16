"""LiteLLM proxy + vLLM backend health monitor.

Scrapes vLLM /metrics on each backend plus LiteLLM /health, appends the
snapshot to a rolling JSONL history, and emits a 24h diagnostic plot.

Usage:
  python tools/litellm_monitor.py             # collect + plot + print
  python tools/litellm_monitor.py --send      # also send plot to Telegram
  python tools/litellm_monitor.py --plot-only # plot from existing history
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys

import httpx
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INFRA = ROOT / "experiments" / "infrastructure.yaml"
HISTORY = HERE / ".litellm_monitor.jsonl"
PLOT_PATH = HERE / ".litellm_monitor.png"
HISTORY_KEEP_DAYS = 7

VLLM_METRICS = {
    "success": "vllm:request_success_total",
    "preemptions": "vllm:num_preemptions_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}


def _load_backends() -> list[dict]:
    with open(INFRA) as f:
        infra = yaml.safe_load(f)
    backends: list[dict] = []
    chain = infra["chain_servers"]
    for ep in chain["endpoints"]:
        port = ep.get("port", chain.get("port", 8000))
        backends.append(
            {
                "role": "chain",
                "label": ep.get("label", f"{ep['host']}:{port}"),
                "host": ep["host"],
                "port": port,
                "url": f"http://{ep['host']}:{port}/metrics",
            }
        )
    mut = infra["mutation_servers"]
    mut_port = mut.get("port", 8000)
    for ep in mut["endpoints"]:
        if ep.get("status", "active") != "active":
            continue
        port = ep.get("port", mut_port)
        backends.append(
            {
                "role": "mutation",
                "label": ep.get("label", f"{ep['host']}:{port}"),
                "host": ep["host"],
                "port": port,
                "url": f"http://{ep['host']}:{port}/metrics",
            }
        )
    return backends


def _scrape_prom(text: str) -> dict[str, float]:
    """Sum numeric values for each Prometheus metric name (ignoring labels)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


def _scrape_backend(client: httpx.Client, url: str) -> dict | None:
    try:
        r = client.get(url, timeout=5.0)
        if r.status_code != 200:
            return None
        scraped = _scrape_prom(r.text)
        return {k: scraped.get(v, 0.0) for k, v in VLLM_METRICS.items()}
    except Exception:
        return None


def _scrape_proxy(client: httpx.Client) -> dict:
    """Liveness check for the proxy process itself.

    NOTE: the full `/health` endpoint probes every backend serially and routinely
    exceeds 30 s, so we use `/health/liveliness` (instant) and derive per-backend
    health counts from our own vLLM /metrics scrapes in collect_snapshot().
    """
    try:
        r = client.get("http://localhost:4000/health/liveliness", timeout=5.0)
        return {"alive": r.status_code == 200}
    except Exception:
        return {"alive": False}


def collect_snapshot() -> dict:
    # Internal IPs must bypass the system Squid proxy.
    no_proxy_env = os.environ.get("NO_PROXY", "")
    extra = ",".join([b["host"] for b in _load_backends()] + ["localhost", "127.0.0.1"])
    os.environ["NO_PROXY"] = no_proxy_env + "," + extra if no_proxy_env else extra
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    with httpx.Client(trust_env=False) as client:
        backends = _load_backends()
        per_backend = {}
        healthy, unhealthy = 0, 0
        for b in backends:
            m = _scrape_backend(client, b["url"])
            if m is None:
                unhealthy += 1
            else:
                healthy += 1
            per_backend[b["label"]] = {
                "role": b["role"],
                "host": b["host"],
                "port": b["port"],
                **(m or {k: None for k in VLLM_METRICS}),
            }
        proxy = _scrape_proxy(client)
        # Derive backend counts from vLLM /metrics reachability — more reliable
        # than LiteLLM /health (which probes serially and times out).
        proxy["healthy"] = healthy
        proxy["unhealthy"] = unhealthy
    return {
        "ts": datetime.now(UTC).isoformat(),
        "proxy": proxy,
        "backends": per_backend,
    }


def append_history(snap: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a") as f:
        f.write(json.dumps(snap) + "\n")


def load_history(keep_days: float = 7.0) -> list[dict]:
    if not HISTORY.exists():
        return []
    cutoff = datetime.now(UTC) - timedelta(days=keep_days)
    rows: list[dict] = []
    with open(HISTORY) as f:
        for line in f:
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(row["ts"])
                if ts >= cutoff:
                    rows.append(row)
            except Exception:
                continue
    return rows


def trim_history(keep_days: float = HISTORY_KEEP_DAYS) -> None:
    rows = load_history(keep_days)
    with open(HISTORY, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rate_matrix(
    rows: list[dict], metric: str, role: str
) -> tuple[np.ndarray, list, np.ndarray]:
    """Build a (T-1, N_backends) rate matrix between adjacent samples.

    Returns (midpoint_times, labels, rates) where rates[i, j] is the average
    per-second rate of `metric` for backend `labels[j]` over interval i.
    Missing values become NaN — downstream nanmean/nanstd handles gaps.
    """
    times = [datetime.fromisoformat(r["ts"]) for r in rows]
    labels = sorted(
        {lbl for r in rows for lbl, d in r["backends"].items() if d.get("role") == role}
    )
    if len(rows) < 2 or not labels:
        return np.array([]), labels, np.empty((0, len(labels)))

    T = len(rows)
    N = len(labels)
    raw = np.full((T, N), np.nan)
    for i, r in enumerate(rows):
        for j, lbl in enumerate(labels):
            v = r["backends"].get(lbl, {}).get(metric)
            if v is not None:
                raw[i, j] = float(v)

    dt = np.array([(times[i + 1] - times[i]).total_seconds() for i in range(T - 1)])
    dt_safe = np.where(dt > 0, dt, np.nan)
    diff = raw[1:] - raw[:-1]
    # Clip negative (counter reset → NaN so we don't plot a misleading zero)
    diff = np.where(diff < 0, np.nan, diff)
    rates = diff / dt_safe[:, None]
    mids = [times[i] + (times[i + 1] - times[i]) / 2 for i in range(T - 1)]
    return np.array(mids), labels, rates


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Edge-aware moving average. window<=1 returns arr unchanged."""
    if window <= 1 or arr.size == 0:
        return arr.copy()
    out = np.empty_like(arr, dtype=float)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        with np.errstate(all="ignore"):
            out[i] = np.nanmean(arr[lo:hi])
    return out


def _pick_window(n_points: int) -> int:
    if n_points < 6:
        return 1
    if n_points < 24:
        return 3
    if n_points < 72:
        return 5
    return max(5, n_points // 24)


def _fmt_num(n: float) -> str:
    """Compact integer-like number: 1234 → '1.2k', 1e6 → '1.0M'."""
    if n is None or not np.isfinite(n):
        return "—"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def _raw_matrix(
    rows: list[dict], metric: str, role: str
) -> tuple[np.ndarray, list, np.ndarray]:
    """Build (T, N) raw-value matrix (no differencing) for gauge metrics.

    Used for instantaneous gauges like `running` or `waiting` — counters
    should go through _rate_matrix instead.
    """
    times = [datetime.fromisoformat(r["ts"]) for r in rows]
    labels = sorted(
        {lbl for r in rows for lbl, d in r["backends"].items() if d.get("role") == role}
    )
    T, N = len(rows), len(labels)
    raw = np.full((T, N), np.nan)
    for i, r in enumerate(rows):
        for j, lbl in enumerate(labels):
            v = r["backends"].get(lbl, {}).get(metric)
            if v is not None:
                raw[i, j] = float(v)
    return np.array(times), labels, raw


def _load_caps() -> dict[str, int]:
    """Parse max_parallel_requests per host:port from the live litellm config."""
    cfg_path = HERE / ".litellm_config.yaml"
    caps: dict[str, int] = {}
    if not cfg_path.exists():
        return caps
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for dep in cfg.get("model_list", []):
            params = dep.get("litellm_params", {})
            api = params.get("api_base", "")
            mpr = params.get("max_parallel_requests")
            # api_base: http://HOST:PORT/v1  →  key = "HOST:PORT"
            if api and mpr:
                hp = api.replace("http://", "").replace("https://", "").split("/", 1)[0]
                caps[hp] = int(mpr)
    except Exception:
        pass
    return caps


# ─────────────────────────────  Scientific plot  ────────────────────────────────

# Color palette for the per-backend lines. tab10 + Set2 give 10+6 distinct hues.
_CHAIN_COLORS = plt.get_cmap("tab10").colors
_MUT_COLORS = plt.get_cmap("Set2").colors


def _plot_timeseries(
    ax,
    rows: list[dict],
    role: str,
    metric: str,
    *,
    kind: str = "rate",
    ylabel: str = "req/s",
    title: str = "",
    colors=None,
) -> None:
    """Overlay one line per backend. kind in {"rate","gauge"}."""
    if kind == "rate":
        mids, labels, vals = _rate_matrix(rows, metric, role)
        x = mids
    else:
        x, labels, vals = _raw_matrix(rows, metric, role)

    if len(labels) == 0 or x.size == 0 or vals.size == 0:
        ax.text(
            0.5,
            0.5,
            "accumulating samples\u2026",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888",
            fontsize=11,
            style="italic",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        return

    win = _pick_window(len(rows))
    palette = colors or _CHAIN_COLORS

    for j, lbl in enumerate(labels):
        y = _rolling_mean(vals[:, j], win)
        ax.plot(x, y, color=palette[j % len(palette)], lw=1.4, label=lbl, alpha=0.9)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        labelspacing=0.3,
    )


def render_plot(rows: list[dict], hours: float = 24.0) -> Path:
    """Clean scientific dashboard: per-backend time series.

    2x2 grid:
      A  chain       req/s   (one line per backend)
      B  mutation    req/s
      C  chain       running requests (gauge)
      D  chain       preemption rate (req/s)
    """
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows = [r for r in rows if datetime.fromisoformat(r["ts"]) >= cutoff]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    (axA, axB), (axC, axD) = axes

    _plot_timeseries(
        axA,
        rows,
        "chain",
        "success",
        kind="rate",
        ylabel="req/s",
        title="Chain (Qwen3-8B) — throughput per backend",
        colors=_CHAIN_COLORS,
    )

    _plot_timeseries(
        axB,
        rows,
        "mutation",
        "success",
        kind="rate",
        ylabel="req/s",
        title="Mutation (Qwen3-235B) — throughput per backend",
        colors=_MUT_COLORS,
    )

    _plot_timeseries(
        axC,
        rows,
        "chain",
        "running",
        kind="gauge",
        ylabel="num_requests_running",
        title="Chain — in-flight requests per backend",
        colors=_CHAIN_COLORS,
    )

    _plot_timeseries(
        axD,
        rows,
        "chain",
        "preemptions",
        kind="rate",
        ylabel="preemptions / s",
        title="Chain — KV-cache preemption rate per backend",
        colors=_CHAIN_COLORS,
    )

    for ax in (axC, axD):
        ax.set_xlabel("UTC time")

    n_samples = len(rows)
    if n_samples >= 2:
        window_h = (
            datetime.fromisoformat(rows[-1]["ts"])
            - datetime.fromisoformat(rows[0]["ts"])
        ).total_seconds() / 3600.0
    else:
        window_h = 0.0
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"LiteLLM backends — {window_h:.1f} h window, {n_samples} samples  ·  {now_str}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return PLOT_PATH


def send_telegram(path: Path, caption: str) -> bool:
    sys.path.insert(0, str(ROOT))
    from tools.telegram_notify import send_photo  # noqa: E402

    return send_photo(str(path), caption=caption, parse_mode="")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="Send plot to Telegram")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip collection; plot from existing history",
    )
    ap.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect one snapshot and exit (no plot, no Telegram). "
        "Used by the 1-minute sampler daemon.",
    )
    ap.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Window (hours) for the plot. Default 24.",
    )
    args = ap.parse_args()

    if not args.plot_only:
        snap = collect_snapshot()
        append_history(snap)
        trim_history()
        print(
            f"[{snap['ts']}] proxy healthy={snap['proxy']['healthy']} "
            f"unhealthy={snap['proxy']['unhealthy']}",
            flush=True,
        )

    if args.collect_only:
        return 0

    rows = load_history(keep_days=HISTORY_KEEP_DAYS)
    path = render_plot(rows, hours=args.hours)
    print(f"Plot: {path}", flush=True)

    if args.send:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        caption = f"LiteLLM proxy {int(args.hours)}h health @ {now}"
        ok = send_telegram(path, caption)
        print(f"Telegram: {'sent' if ok else 'FAILED'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
