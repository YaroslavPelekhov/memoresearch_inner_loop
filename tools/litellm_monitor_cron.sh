#!/usr/bin/env bash
# Cron wrapper for tools/litellm_monitor.py.
# Sources .env (via python-dotenv inside the script) and runs with a stable
# working directory so relative paths resolve the infra yaml correctly.

set -euo pipefail
PROJ="/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal"
cd "$PROJ"
LOG="$PROJ/tools/.litellm_monitor.cron.log"
{
    echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
    /home/jovyan/.mlspace/envs/evo/bin/python3 "$PROJ/tools/litellm_monitor.py" --send
} >> "$LOG" 2>&1
