#!/usr/bin/env bash
# Cron-based health check for all running experiment watchdogs.
# Checks Redis heartbeats and alerts via GitHub PR comment if stale.
#
# Install: crontab -e
#   */30 * * * * /path/to/gigaevo/tools/experiment/check_all_watchdogs.sh >> /tmp/watchdog_cron.log 2>&1
#
# Remove: crontab -e (delete the line)

set -euo pipefail

PYTHON=${GIGAEVO_PYTHON:-$(command -v python3)}
PROJ="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
STALE_THRESHOLD=10800  # 3 hours (3x default 1h poll interval)

NOW=$(date +%s)

# Find all watchdog heartbeat keys in Redis DB 0
HEARTBEATS=$($PYTHON -c "
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
keys = r.keys('experiments:*:watchdog_heartbeat')
for key in keys:
    val = r.get(key)
    if val:
        exp = key.decode().split(':')[1]
        print(f'{exp}|{val.decode()}')
" 2>/dev/null || true)

if [ -z "$HEARTBEATS" ]; then
    echo "[$(date -u)] No active watchdog heartbeats found"
    exit 0
fi

while IFS='|' read -r exp_name last_beat; do
    AGE=$((NOW - last_beat))
    if [ "$AGE" -gt "$STALE_THRESHOLD" ]; then
        echo "[$(date -u)] STALE: $exp_name — last heartbeat ${AGE}s ago"

        # Try to find PR number from experiment.yaml
        PR=$($PYTHON -c "
import yaml
with open('$PROJ/experiments/$exp_name/experiment.yaml') as f:
    m = yaml.safe_load(f)
print(m.get('experiment',{}).get('pr_number',''))
" 2>/dev/null || true)

        if [ -n "$PR" ]; then
            gh pr comment "$PR" --repo KhrulkovV/gigaevo-core-internal \
                --body "**WATCHDOG STALE** for \`$exp_name\`. Last heartbeat: $(date -d @$last_beat -u '+%Y-%m-%d %H:%M UTC') (${AGE}s ago). Restart watchdog or check experiment status." \
                2>/dev/null || echo "[$(date -u)] Failed to post PR comment for $exp_name"
        fi
    else
        echo "[$(date -u)] OK: $exp_name — heartbeat ${AGE}s ago"
    fi
done <<< "$HEARTBEATS"
