#!/usr/bin/env bash
# Persistent 4-hour monitor loop. No crontab on this box → run as a daemon.
#
# Start:  nohup bash tools/litellm_monitor_loop.sh > /dev/null 2>&1 &  disown
# Stop:   tools/litellm_monitor_loop.sh --stop
# Status: tools/litellm_monitor_loop.sh --status

set -euo pipefail
PROJ="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal"
cd "$PROJ"
PID_FILE="$PROJ/tools/.litellm_monitor_loop.pid"
LOG="$PROJ/tools/.litellm_monitor.loop.log"
INTERVAL_S=14400   # 4 hours

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
            tail -n 10 "$LOG" 2>/dev/null || true
        else
            echo "Not running."
        fi
        exit 0 ;;
esac

# Daemon body
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
echo $$ > "$PID_FILE"

trap 'rm -f "$PID_FILE"; exit 0' TERM INT

while true; do
    {
        echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
        # Plot-and-send only — the 1-min sampler daemon is the source of data.
        # (Falling back to --send here would add a duplicate sample; harmless
        #  but redundant. --plot-only --send renders from existing history.)
        /home/jovyan/.mlspace/envs/evo/bin/python3 "$PROJ/tools/litellm_monitor.py" --plot-only --send || echo "run failed (exit=$?)"
    } >> "$LOG" 2>&1
    sleep "$INTERVAL_S"
done
