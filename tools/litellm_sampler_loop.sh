#!/usr/bin/env bash
# 1-minute sampler for LiteLLM + vLLM backend metrics.
#
# Runs `litellm_monitor.py --collect-only` every 60 s and appends a snapshot
# to the shared history file (tools/.litellm_monitor.jsonl). The 4-hour
# reporter (litellm_monitor_loop.sh) reads that same history to render plots,
# so this daemon is purely a fast data collector — no rendering, no Telegram.
#
# Start:  nohup bash tools/litellm_sampler_loop.sh > /dev/null 2>&1 &  disown
# Stop:   bash tools/litellm_sampler_loop.sh --stop
# Status: bash tools/litellm_sampler_loop.sh --status

set -euo pipefail
PROJ="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal"
cd "$PROJ"
PID_FILE="$PROJ/tools/.litellm_sampler.pid"
LOG="$PROJ/tools/.litellm_sampler.log"
INTERVAL_S=60

case "${1:-}" in
    --stop)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            kill "$(cat "$PID_FILE")"
            rm -f "$PID_FILE"
            echo "Stopped."
        else
            echo "Not running (or stale pidfile)."
            rm -f "$PID_FILE"
        fi
        exit 0 ;;
    --status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Running (PID $(cat "$PID_FILE"))"
            tail -n 5 "$LOG" 2>/dev/null || true
        else
            echo "Not running."
        fi
        exit 0 ;;
esac

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"; exit 0' TERM INT

# Keep log small: rotate when >5 MB.
rotate_log() {
    if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ]; then
        mv "$LOG" "$LOG.1"
    fi
}

while true; do
    rotate_log
    /home/jovyan/.mlspace/envs/evo/bin/python3 "$PROJ/tools/litellm_monitor.py" --collect-only \
        >> "$LOG" 2>&1 || echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] sample failed (exit=$?)" >> "$LOG"
    sleep "$INTERVAL_S"
done
