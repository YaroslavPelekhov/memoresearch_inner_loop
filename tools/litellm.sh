#!/usr/bin/env bash
# tools/litellm.sh — Start a LiteLLM proxy that load-balances across all
# backend LLM servers defined in experiments/infrastructure.yaml.
#
# Usage:
#   bash tools/litellm.sh              # foreground (Ctrl-C to stop)
#   bash tools/litellm.sh --background # daemonize with nohup
#   bash tools/litellm.sh --stop       # kill running litellm proxy
#   bash tools/litellm.sh --status     # check if proxy is running
#
# Requires: litellm conda env (set LITELLM_PYTHON / LITELLM_BIN to override defaults)
#           PyYAML (for parsing infrastructure.yaml)

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
INFRA_YAML="$PROJ/experiments/infrastructure.yaml"
LITELLM_PYTHON="${LITELLM_PYTHON:-/home/jovyan/.mlspace/envs/litellm/bin/python}"
LITELLM_BIN="${LITELLM_BIN:-/home/jovyan/.mlspace/envs/litellm/bin/litellm}"
LITELLM_CONFIG="$PROJ/tools/.litellm_config.yaml"
LITELLM_LOG="$PROJ/tools/.litellm.log"
LITELLM_PID="$PROJ/tools/.litellm.pid"
PORT=4000

# --- Helper: parse infrastructure.yaml and generate litellm config ---
#
# Design notes (see `tools/litellm.sh` header comment for context):
#
# RELIABILITY POLICY: scientific experiments — no dropped requests.
# The proxy queues and retries; it does NOT fail fast on capacity.
# Timeouts below are safety valves for genuinely hung backends, set
# well above any legitimate completion time.
#
# - Route by least-busy. Each vLLM chain instance has ~equal hardware, but one
#   instance holding a slow request is exactly where simple-shuffle breaks down.
#   least-busy tracks in-flight requests per deployment (in-process counter) and
#   sends new traffic to the deployment with the fewest.
# - Cap per-deployment concurrency to protect vLLM from KV-cache preemption.
#   Cap sized for ~35% KV-cache at avg request size; aggregate cap large
#   enough that typical bursts don't queue long (320 chain / 36 mutation).
# - Generous timeouts (chain 900s, mutation 1800s). Chain finishes in 60-130s
#   normally; mutation thinking traces up to ~30 min. Timeout only fires on
#   truly stuck backends — never on a legitimate slow generation.
# - num_retries=2. Covers transient 5xx / bouncing backend / one preemption
#   without hammering a degraded backend.
generate_config() {
    "$LITELLM_PYTHON" - "$INFRA_YAML" "$LITELLM_CONFIG" <<'PYEOF'
import sys
import yaml

infra_path, out_path = sys.argv[1], sys.argv[2]

with open(infra_path) as f:
    infra = yaml.safe_load(f)

proxy_cfg = infra.get("litellm_proxy", {})
chain_alias = proxy_cfg.get("alias_chain", "Qwen/Qwen3-8B")
mutation_alias = proxy_cfg.get("alias_mutation", "Qwen3-235B-A22B-Thinking-2507")

# Per-deployment concurrency + timeout. Tune these if vLLM capacity changes.
#
# Capacity math (probed from /metrics on 2026-04-20):
#   Chain   : num_gpu_blocks=26829, block_size=16 → 429k KV tokens; max_model_len=32768
#             Worst case (all full context) fits ~13 seqs. Observed ~14.6% of
#             requests are long-gen (>=10k tokens) and stay resident in KV for
#             minutes — enough to tip instances into a preemption feedback loop
#             (100% KV, 100+ preemptions/min, TTFT p99 >10 min) while peers sit idle.
#   Mutation: num_gpu_blocks=16946 → 271k KV tokens; max_model_len=160000
#             Full context fits just ~1.7 seqs. 8 concurrent thinking-mode runs at
#             ~20k tokens each = 160k (59% of cache). Mild preemption possible.
#
# LiteLLM 1.82.6 streaming-semaphore bug (router.py:2148 and siblings):
#   `max_parallel_requests` releases its semaphore at TTFT, not at stream
#   completion. For chain workload (TTFT ~0.3s, gen 30-130s) the effective
#   concurrency cap is inflated ~gen_time/TTFT (100-400×), so it does
#   essentially nothing under streaming load. Confirmed 2026-04-22: with
#   max_parallel_requests=40 per chain, 9/11 chains pinned at 96-99.9% KV
#   with 500+ queued requests.
#   Workaround: use `rpm` (requests-per-minute, enforced by a time-window
#   counter independent of stream duration). At per-deployment rpm=40,
#   aggregate is 11 × 40 = 440 rpm ≈ 7.3 req/s — well above observed
#   steady-state ~2 req/s chain load and below the burst-to-preemption
#   threshold (~20 concurrent long-gens per instance). Mutation keeps
#   max_parallel_requests since its short/uniform gen times make the bug
#   irrelevant there.
CHAIN_RPM = 40              # per-deployment rate cap (11 × 40 = 440 rpm aggregate)
CHAIN_TIMEOUT_S = 3600      # 1h; chain 4k-token gen is 60-130s, so ~30x headroom
MUTATION_CONCURRENCY = 8    # → 48 total concurrent mutation calls (6 × 8)
MUTATION_TIMEOUT_S = 7200   # 2h; 235B thinking traces ~30 min, so ~4x headroom

models = []

# Chain servers (Qwen3-8B, 8 instances on one H100 box)
chain = infra["chain_servers"]
chain_model = chain["model"]
chain_endpoints = []
for ep in chain["endpoints"]:
    port = ep.get("port", chain.get("port", 8000))
    # Per-endpoint override: some vLLM instances are launched without
    # --served-model-name, so they advertise the HF cache path as the
    # model id. `served_model_name` lets us keep a unified client alias
    # while sending the backend the name it actually recognizes.
    served = ep.get("served_model_name", chain_model)
    chain_endpoints.append({
        "model_name": chain_alias,
        "litellm_params": {
            "model": f"openai/{served}",
            "api_base": f"http://{ep['host']}:{port}/v1",
            "api_key": "None",
            "rpm": CHAIN_RPM,
            "timeout": CHAIN_TIMEOUT_S,
        },
    })
models.extend(chain_endpoints)

# Mutation servers (Qwen3-235B, one instance per host)
mut = infra["mutation_servers"]
mut_model = mut["model"]
mut_port = mut.get("port", 8000)
for ep in mut["endpoints"]:
    if ep.get("status", "active") != "active":
        continue
    port = ep.get("port", mut_port)
    models.append({
        "model_name": mutation_alias,
        "litellm_params": {
            "model": f"openai/{mut_model}",
            "api_base": f"http://{ep['host']}:{port}/v1",
            "api_key": "None",
            "max_parallel_requests": MUTATION_CONCURRENCY,
            "timeout": MUTATION_TIMEOUT_S,
        },
    })

config = {
    "model_list": models,
    "general_settings": {
        "master_key": "sk-gigaevo",
    },
    "litellm_settings": {
        "drop_params": True,
        # Retries on transient failures protect scientific reliability:
        # a 5xx, a momentarily-bouncing backend, or a vLLM preemption
        # should NOT surface as a drop to the experiment. With 10 chain
        # and 6 mutation backends, 3 retries picks a fresh deployment
        # each time — a single correlated flap can't drop a request.
        "num_retries": 3,
        "request_timeout": MUTATION_TIMEOUT_S,  # upper-bound fallback (2h)
    },
    "router_settings": {
        # Route to the deployment with the fewest in-flight requests.
        # When an instance has a stuck call, its in-flight count stays
        # elevated until the timeout fires — so least-busy naturally
        # steers new traffic away from degraded backends.
        "routing_strategy": "least-busy",
        "num_retries": 3,
        "timeout": MUTATION_TIMEOUT_S,
        # Circuit breaker: take a backend out of rotation for 30s after
        # 5 consecutive failures. A loose threshold keeps capacity online
        # through transient flaps; a short cooldown recovers fast.
        # Cooldown isolates a bad backend, it does NOT drop user requests —
        # in-flight and queued requests re-route to healthy peers.
        "allowed_fails": 5,
        "cooldown_time": 30,
    },
}

with open(out_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Generated {out_path}")
print(f"  Chain model alias:    {chain_alias} ({len(chain_endpoints)} endpoints,"
      f" rpm={CHAIN_RPM}, timeout={CHAIN_TIMEOUT_S}s)")
print(f"  Mutation model alias: {mutation_alias} "
      f"({len(models) - len(chain_endpoints)} endpoints,"
      f" max_parallel_requests={MUTATION_CONCURRENCY}, timeout={MUTATION_TIMEOUT_S}s)")
print(f"  Routing strategy:     least-busy")
print(f"  num_retries:          1 (was 3)")
PYEOF
}

# --- Commands ---

do_stop() {
    if [ -f "$LITELLM_PID" ]; then
        pid=$(cat "$LITELLM_PID")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping litellm proxy (PID $pid)..."
            kill "$pid"
            rm -f "$LITELLM_PID"
            echo "Stopped."
        else
            echo "PID $pid not running. Cleaning up stale pidfile."
            rm -f "$LITELLM_PID"
        fi
    else
        echo "No pidfile found. Checking for running litellm processes..."
        pkill -f "litellm --config" && echo "Killed." || echo "No litellm proxy running."
    fi
}

do_status() {
    if [ -f "$LITELLM_PID" ]; then
        pid=$(cat "$LITELLM_PID")
        if kill -0 "$pid" 2>/dev/null; then
            echo "LiteLLM proxy running (PID $pid) on port $PORT"
            curl -s "http://localhost:$PORT/health" 2>/dev/null && echo "" || echo "  (health check failed — may still be starting)"
            return 0
        else
            echo "PID $pid not running (stale pidfile)"
            rm -f "$LITELLM_PID"
            return 1
        fi
    else
        echo "No litellm proxy running."
        return 1
    fi
}

do_start() {
    local background="${1:-false}"

    # Check if already running
    if [ -f "$LITELLM_PID" ] && kill -0 "$(cat "$LITELLM_PID")" 2>/dev/null; then
        echo "LiteLLM proxy already running (PID $(cat "$LITELLM_PID")). Use --stop first."
        exit 1
    fi

    # Set NO_PROXY for backend access
    no_proxy_ips=$("$LITELLM_PYTHON" -c "
import yaml
with open('$INFRA_YAML') as f:
    infra = yaml.safe_load(f)
print(','.join(infra.get('no_proxy_hosts', [])))
")
    export NO_PROXY="$no_proxy_ips"
    export no_proxy="$no_proxy_ips"

    # Generate config from infrastructure.yaml
    echo "--- Generating LiteLLM config from $INFRA_YAML ---"
    generate_config
    echo ""

    if [ "$background" = "true" ]; then
        echo "--- Starting LiteLLM proxy (background, port $PORT) ---"
        nohup "$LITELLM_BIN" \
            --config "$LITELLM_CONFIG" \
            --port "$PORT" \
            --host 0.0.0.0 \
            > "$LITELLM_LOG" 2>&1 &
        echo $! > "$LITELLM_PID"
        echo "PID: $(cat "$LITELLM_PID")"
        echo "Log: $LITELLM_LOG"
        echo ""
        echo "Waiting for proxy to be ready..."
        for i in $(seq 1 30); do
            if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
                echo "LiteLLM proxy ready on port $PORT"
                echo ""
                echo "Use in experiments:"
                echo "  Chain:    http://localhost:$PORT/v1  model=$("$LITELLM_PYTHON" -c "import yaml; print(yaml.safe_load(open('$INFRA_YAML'))['litellm_proxy']['alias_chain'])")"
                echo "  Mutation: http://localhost:$PORT/v1  model=$("$LITELLM_PYTHON" -c "import yaml; print(yaml.safe_load(open('$INFRA_YAML'))['litellm_proxy']['alias_mutation'])")"
                return 0
            fi
            sleep 2
        done
        echo "WARNING: proxy did not become healthy within 60s. Check $LITELLM_LOG"
    else
        echo "--- Starting LiteLLM proxy (foreground, port $PORT) ---"
        echo "Press Ctrl-C to stop."
        echo ""
        exec "$LITELLM_BIN" \
            --config "$LITELLM_CONFIG" \
            --port "$PORT" \
            --host 0.0.0.0
    fi
}

# --- Main ---
case "${1:-}" in
    --stop)
        do_stop
        ;;
    --status)
        do_status
        ;;
    --background)
        do_start true
        ;;
    ""|--foreground)
        do_start false
        ;;
    *)
        echo "Usage: bash tools/litellm.sh [--background|--foreground|--stop|--status]"
        exit 1
        ;;
esac
