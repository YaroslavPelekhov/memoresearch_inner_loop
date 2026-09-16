#!/usr/bin/env bash
# GENERATED from experiment.yaml — do not edit manually.
# Regenerate: PYTHONPATH=. $GIGAEVO_PYTHON tools/experiment/generate_launch.py --experiment toy/kadane_speedrun
#
# Experiment: toy/kadane_speedrun
# Branch: exp/toy-kadane-speedrun
#
# Runs: R1, R2

set -euo pipefail

PROJ="/workspace-SR008.fs2/mathemage/gigaevo-core"
PYTHON="/home/jovyan/envs/evo_fast/bin/python"
LOG_DIR="$PROJ/experiments/toy/kadane_speedrun"

export NO_PROXY="localhost,127.0.0.1,api.github.com"
export no_proxy="$NO_PROXY"
export GIGAEVO_PYTHON="$PYTHON"

echo "================================================================"
echo "toy/kadane_speedrun experiment launch — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "R1: Gemini Flash on Kadane's algorithm — pipeline=standard"
echo "R2: Gemini Flash on Kadane's algorithm — pipeline=standard"
echo "================================================================"
echo ""

# ── Preflight check (hard gate) ──────────────────────────────────────────
PYTHONPATH="$PROJ" "$PYTHON" "$PROJ/tools/experiment/preflight_check.py" --experiment toy/kadane_speedrun
echo ""

# ── Config verification (--cfg job) ───────────────────────────────────────
echo "--- R1 config ---"
"$PYTHON" "$PROJ/run.py" \
    problem.name=toy_kadane \
    pipeline=standard \
    prompts=default \
    redis.db=15 \
    stage_timeout=120 \
    dag_timeout=300 \
    max_generations=3 \
    max_mutations_per_generation=2 \
    max_elites_per_generation=2 \
    num_parents=1 \
    model_name=openai/gpt-4o-mini \
    llm_base_url="https://openrouter.ai/api/v1" \
    mutation_mode=rewrite \
    --cfg job \
    > "$LOG_DIR/cfg_run_R1.txt" 2>&1
cat "$LOG_DIR/cfg_run_R1.txt" | head -40
echo ""
echo "--- R2 config ---"
"$PYTHON" "$PROJ/run.py" \
    problem.name=toy_kadane \
    pipeline=standard \
    prompts=default \
    redis.db=14 \
    stage_timeout=120 \
    dag_timeout=300 \
    max_generations=3 \
    max_mutations_per_generation=2 \
    max_elites_per_generation=2 \
    num_parents=1 \
    model_name=openai/gpt-4o-mini \
    llm_base_url="https://openrouter.ai/api/v1" \
    mutation_mode=rewrite \
    --cfg job \
    > "$LOG_DIR/cfg_run_R2.txt" 2>&1
cat "$LOG_DIR/cfg_run_R2.txt" | head -40
echo ""

echo "================================================================"
echo "Config verified."
echo "Launching runs..."
echo ""

# ── Launch from project root (Hydra resolves paths relative to CWD) ───────
cd "$PROJ"

# ── Launch runs ────────────────────────────────────────────────────────────
# ── Run R1: Gemini Flash on Kadane's algorithm
nohup "$PYTHON" "$PROJ/run.py" \
    problem.name=toy_kadane \
    pipeline=standard \
    prompts=default \
    redis.db=15 \
    stage_timeout=120 \
    dag_timeout=300 \
    max_generations=3 \
    max_mutations_per_generation=2 \
    max_elites_per_generation=2 \
    num_parents=1 \
    model_name=openai/gpt-4o-mini \
    llm_base_url="https://openrouter.ai/api/v1" \
    mutation_mode=rewrite \
    > "$LOG_DIR/run_R1.log" 2>&1 &
PID_R1=$!
echo "Run R1 started: PID=$PID_R1  DB=15  pipeline=standard"

# ── Run R2: Gemini Flash on Kadane's algorithm
nohup "$PYTHON" "$PROJ/run.py" \
    problem.name=toy_kadane \
    pipeline=standard \
    prompts=default \
    redis.db=14 \
    stage_timeout=120 \
    dag_timeout=300 \
    max_generations=3 \
    max_mutations_per_generation=2 \
    max_elites_per_generation=2 \
    num_parents=1 \
    model_name=openai/gpt-4o-mini \
    llm_base_url="https://openrouter.ai/api/v1" \
    mutation_mode=rewrite \
    > "$LOG_DIR/run_R2.log" 2>&1 &
PID_R2=$!
echo "Run R2 started: PID=$PID_R2  DB=14  pipeline=standard"

echo ""
echo "================================================================"
echo "All 2 runs launched."
echo "PIDs: R1=$PID_R1  R2=$PID_R2"
echo "$PID_R1 $PID_R2" > "$LOG_DIR/pids.txt"

# ── Verify all PIDs alive ──────────────────────────────────────────────────
sleep 5
ALL_ALIVE=true
kill -0 $PID_R1 2>/dev/null || { echo "DEAD: R1 (PID=$PID_R1)"; ALL_ALIVE=false; }
kill -0 $PID_R2 2>/dev/null || { echo "DEAD: R2 (PID=$PID_R2)"; ALL_ALIVE=false; }
if [ "$ALL_ALIVE" = "false" ]; then
    echo "ABORT: not all runs alive. Check logs."
    exit 1
fi
echo "All PIDs verified alive."

# ── Record PIDs in experiment.yaml ────────────────────────────────────────
gigaevo -e toy/kadane_speedrun manifest record-pids --pids-file "$LOG_DIR/pids.txt" --labels "R1 R2"

echo ""
echo "Launch watchdog:"
echo "  NO_PROXY=\"$NO_PROXY\" no_proxy=\"$NO_PROXY\" \\"
echo "  nohup $PYTHON experiments/toy/kadane_speedrun/run_watchdog.py \\"
echo "      > experiments/toy/kadane_speedrun/watchdog.log 2>&1 &"
echo "================================================================"
