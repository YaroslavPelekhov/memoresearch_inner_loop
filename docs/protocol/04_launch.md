# Phase 4: Implementation and Launch
<!-- Protocol version: 1.0 -->

**Actor**: Researcher + Claude Code
**Prerequisite**: `experiments/<task>/<name>/03_plan.md` committed to git before any code changes
**Gate**: All dry-run checks pass, preflight green, researcher presses Enter

---

## Step 0: Implement Per-Experiment Tools

Watchdog is CLI-driven — `gigaevo -e <task>/<name> watchdog` reads the
`control_plane.watchdog` section of `experiment.yaml` directly. No per-experiment
Python entry point is needed.

Test eval is still per-experiment:

```bash
cp experiments/_template/run_test_eval.sh experiments/<task>/<name>/run_test_eval.sh
```

- [ ] `run_test_eval.sh` — implement if the problem has a held-out test set
  (reference: `experiments/hotpotqa/push/run_test_eval.sh`); skip for optimization
  problems without a separate test split (e.g. Heilbronn, hexagon packing)

Generic reusable tools are available via the `gigaevo` CLI (e.g. `gigaevo status`,
`gigaevo flush`) or legacy scripts in `tools/` — do not reimplement them.

- [ ] Run the phase gate check to confirm all prior phases are complete and committed:
  ```bash
  bash tools/experiment/check_phase_order.sh <experiment-name>
  ```
  All checks must pass before proceeding.

---

## Step 1: Implementation

- [ ] Implement all code and config changes
- [ ] For every pipeline YAML used with `prompts=<custom>`:
  - [ ] `prompts_dir: ${prompts.dir}` present in the `evolution_context` block
  - [ ] `prompts_dir: ${prompts.dir}` present in `mutation_operator` (via `_base.yaml` or explicitly)
  - [ ] If either is missing: **stop and fix** — custom prompts are silently ignored with no error
- [ ] For every `validate.py` that returns `tuple[dict, list[...]]`:
  - [ ] Pipeline is `hotpotqa_asi`, `hotpotqa_reflective`, or another with a custom `FormatterStage`
  - [ ] `pipeline=standard` is **forbidden** — it calls `repr()` on the tuple
- [ ] Run unit tests: `$GIGAEVO_PYTHON -m pytest`
- [ ] Run linting: `ruff check . && ruff format --check .`

---

## Step 2: Write launch.sh

Create `experiments/<task>/<name>/launch.sh`. It must contain these sections in order:

1. **Header comment** — run labels, DBs, server IPs, manipulated variables, pre-registration ref
2. **`set -euo pipefail`** — abort on any error
3. **Server IP variables** — chain servers and mutation LLM servers, one per run
4. **`NO_PROXY` / `no_proxy` export** — must include ALL internal server IPs
5. **`--flush` block** — optional Redis flush (guarded by `[[ "${1:-}" == "--flush" ]]`)
6. **`[preflight]` server connectivity** — curl `/v1/models` for every server, abort on failure
7. **`[preflight]` thinking mode** — POST to each chain endpoint, check response contains `<think>`
8. **`[preflight]` Redis empty** — check `dbsize() == 0` for every DB, abort if not
9. **`[preflight]` seed directory** — confirm `initial_programs/` exists and has `.py` files
10. **`COMMON_PARAMS` array** — shared Hydra overrides (everything NOT in the design table IV column)
11. **`[verify]` config-dump block** — one `--cfg job` invocation per run, then human pause:
    ```bash
    "$PYTHON" "$PROJ/run.py" "${COMMON_PARAMS[@]}" problem.name=... redis.db=N --cfg job
    read -r _ </dev/tty 2>/dev/null || true
    ```
    `--cfg job` is the Hydra built-in that dumps the fully resolved config YAML and exits
    immediately without running the experiment. It does **not** start workers, touch Redis,
    or load problem code — so it is safe to run at any time. ~~`dry_run=true`~~ was removed
    from `config.yaml`; do not use it.
12. **`[launch]` block** — one `nohup ... &` per run, capture PID, echo to console and `launch.log`
13. **Post-launch instructions** — print exact commands to update watchdog PIDs and start it

Key syntax rules:
- Use full Python path: `$GIGAEVO_PYTHON`
- Config dump: `--cfg job` (Hydra built-in, exits immediately after printing resolved config)
- Redis flush uses Python, not `redis-cli` (may not be installed):
  ```bash
  "$PYTHON" -c "import redis; [redis.Redis(db=n).flushdb() for n in [DB1,DB2,...]]"
  ```
- Also: `GIGAEVO_PYTHON=$GIGAEVO_PYTHON` must be set (or hardcoded)
  for any script that falls back to `$(command -v python3)` — system python3 lacks `redis`

---

## Step 3: Configure Watchdog

Watchdog configuration lives in `experiment.yaml` under `control_plane.watchdog`.
The CLI (`gigaevo -e <task>/<name> watchdog`) reads `runs[]` (labels, DBs, prefixes,
roles, PIDs), `plot_metrics`, `plot_commands`, `alert_thresholds`, and
`checkpoint_milestones` directly from the manifest. Plugin (`adversarial` /
`solo` / `prompt_coevo`) is auto-resolved from `runs[].prefix`.

Verify in `experiment.yaml`:

- [ ] `runs[]` labels, `db`, `prefix`, and `role` match this experiment's design
- [ ] `control_plane.watchdog.plot_metrics` lists the metrics to chart
- [ ] `control_plane.watchdog.plot_commands` references commands available in the
  plugin (`arms-race`, `comparison`, `trajectory`)
- [ ] `control_plane.notifications.telegram.enabled: true` if Telegram alerts are wanted
- [ ] `.env` at project root has `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
  (or the env vars named in `token_env` / `chat_id_env`)

### Functional test (mandatory)

Start the watchdog, wait 60 seconds, confirm it is still alive:

```bash
nohup gigaevo -e <task>/<name> watchdog \
    > experiments/<task>/<name>/watchdog.log 2>&1 &
WATCHDOG_PID=$!
sleep 60
kill -0 $WATCHDOG_PID 2>/dev/null && echo "Watchdog alive" || echo "WATCHDOG DEAD — check log"
```

If dead: read `watchdog.log`, fix the crash, restart. Do not proceed until the watchdog
survives 60 seconds.

---

## Step 4: Config Verification

**Mandatory for every run, without exception.**

The `[verify]` block in `launch.sh` runs `--cfg job` for each run. Review the YAML output
against the design table before pressing Enter.

### 4a. Config fields (from `--cfg job` YAML output)

- [ ] `redis.db` — matches design table
- [ ] `redis.host` / `redis.port` — correct Redis instance
- [ ] `problem.name` — matches design table (e.g., `static` vs `static_r`)
- [ ] `problem.dir` — path exists on disk (`ls "$PROJ/problems/${problem.name}"`)
- [ ] `prompts.dir` — correct:
  - Control runs: `null` or package default
  - Treatment runs with custom prompts: correct absolute path
- [ ] `pipeline_builder._target_` — correct class
- [ ] `max_generations`, `max_mutations_per_generation`, `max_elites_per_generation`,
  `num_parents`, `primary_resolution` — all match design table
- [ ] `llm_base_url` — correct mutation LLM IP and port

### 4b. Manual code checks (run once before launch, not per-run)

`--cfg job` exits before loading problem code, so runtime metadata must be verified manually:

**Problem directory**
```bash
ls "$PROJ/problems/chains/hotpotqa/<problem_name>/"
# Must contain: validate.py, task_description.txt, metrics.yaml, baseline.py (if static)
ls "$PROJ/experiments/<seed>/initial_programs/"
# Must contain at least 1 .py file with def entrypoint()
```

**validate.py return type → pipeline constraint**
```bash
grep "def validate" problems/chains/hotpotqa/<problem_name>/validate.py
grep "return (" problems/chains/hotpotqa/<problem_name>/validate.py
```
- Returns `dict` → `pipeline=standard` acceptable
- Returns `tuple[dict, list[dict]]` → **must use `pipeline=hotpotqa_asi`** (never `standard`)

**Pipeline class**
```bash
grep "_target_" config/pipeline/hotpotqa_asi.yaml
```
- Confirm `FormatterStage` class is the custom one, not the base (which calls `repr()`)

**Prompt files** (only matters if `prompts=<custom>`)
```bash
grep "prompts_dir" config/pipeline/hotpotqa_asi.yaml
# Must appear in BOTH evolution_context AND mutation_operator blocks
```

**Seed programs**
```bash
ls "$PROJ/experiments/<seed>/initial_programs/"
python -c "import ast; ast.parse(open('<path>').read()); print('syntax ok')"
```

**Environment variables**
- [ ] Problem-specific env var (e.g., `HOTPOTQA_CHAIN_URL`) is exported in `launch.sh`
- [ ] `NO_PROXY` / `no_proxy` includes all internal server IPs
- [ ] `GIGAEVO_PYTHON` set to `$GIGAEVO_PYTHON` (or hardcoded in script)

---

## Step 5: Launch

After reviewing all dry-run output, press Enter at the `[verify]` pause in `launch.sh`.

- [ ] Pressed Enter
- [ ] All PIDs printed to console (one per run)
- [ ] PIDs recorded in `experiments/<task>/<name>/03_plan.md` — Actual Launch Record table
- [ ] Launch time (UTC) recorded in `03_plan.md`
- [ ] `launch.log` contains correct entries

**Capture server state and config output at launch** (before logs rotate):

```bash
# Save resolved config for each run — redirect --cfg job output to a text file
# Add to launch.sh [verify] block, one per run (before the read -r pause):
HOTPOTQA_CHAIN_URL="$CHAIN_URL_O" "$PYTHON" "$PROJ/run.py" \
    "${COMMON_PARAMS[@]}" problem.name=... redis.db=N \
    --cfg job > experiments/<task>/<name>/cfg_run_O.txt 2>&1

# Save LLM server model identity for each chain server at launch time
# Note: *.json is gitignored — save as .txt
for LABEL_URL in "O:$CHAIN_URL_O" "R:$CHAIN_URL_R"; do
    LABEL="${LABEL_URL%%:*}"; URL="${LABEL_URL#*:}"
    HOST="${URL#http://}"; HOST="${HOST%%:*}"
    curl --noproxy "$HOST" -s "$URL/models" \
        >> experiments/<task>/<name>/server_models_at_launch.txt
    echo "" >> experiments/<task>/<name>/server_models_at_launch.txt
done
```

- [ ] `cfg_run_<label>.txt` saved for each run (contains full resolved config YAML)
- [ ] `server_models_at_launch.txt` saved (model IDs + context window at launch time)
- [ ] Both committed to git:
  ```bash
  git add experiments/<task>/<name>/cfg_run_*.txt experiments/<task>/<name>/server_models_at_launch.txt
  git commit -m "launch-artifacts: config dumps and server model info for <name>"
  ```
  **Note**: `*.json` and `archives/` are gitignored — use `.txt` extension and save
  directly under `experiments/<task>/<name>/`, not in `archives/`.

---

## Step 6: Start Watchdog

```bash
nohup gigaevo -e <task>/<name> watchdog \
    > experiments/<task>/<name>/watchdog.log 2>&1 &
echo "Watchdog PID: $!"
```

The CLI sources `.env` for Telegram credentials and inherits `NO_PROXY` from the
current shell (launch.sh already exports it).

- [ ] Watchdog started
- [ ] Watchdog PID recorded in `experiment.yaml` (`control_plane.watchdog_pid`)
- [ ] After 60 seconds: confirm watchdog still alive (`kill -0 <PID>`)
- [ ] After the first poll tick (default 60 minutes): confirm watchdog still alive

**Set a recurring watchdog health check** — the watchdog is the only mechanism
producing checkpoint data during a multi-day run. If it crashes silently, you lose
all intermediate monitoring.

`crontab` is **not available** on this machine. Use a background health-check loop instead:

```bash
# Background health-check loop: checks every 30 minutes, logs if dead
(
  WATCHDOG_PID=<PID>
  HEALTH_LOG="experiments/<task>/<name>/watchdog_health.log"
  while true; do
    sleep 1800
    if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
      echo "[$(date -u '+%Y-%m-%d %H:%M UTC')] WATCHDOG DEAD (PID=$WATCHDOG_PID)" >> "$HEALTH_LOG"
    fi
  done
) &
HEALTH_PID=$!
echo "Health-check loop PID: $HEALTH_PID"
# Kill after experiment: kill $HEALTH_PID
```

- [ ] Health-check loop started; PID noted for cleanup after experiment

---

## Step 7: Post-Launch Monitoring

Any crash, stall, or unexpected result must be recorded as an amendment in `03_plan.md`.

Checkpoint generations and timing depend on `max_generations` and per-evaluation cost — both set
in the design table (`03_plan.md`). The structure below is fixed; fill in concrete numbers from
your design before launch.

### Early smoke check (≈10% of max_generations)

- [ ] All run PIDs alive (`kill -0 <PID>`)
- [ ] Watchdog PID alive
- [ ] Redis key count growing for each DB
- [ ] No `ERROR` or `Traceback` in run logs (`tail -50 run_<x>.log`)
- [ ] **Run `tools/status.py`** and check the `Invalid%` and `Val dur(s)` columns:
  - `Invalid%` > 75% at gen 3+ → **stop immediately** — `stage_timeout` is too short
    for this eval workload. Record as amendment, fix `stage_timeout`, relaunch.
  - `Val dur(s)` mean > 60% of `stage_timeout` → at risk; monitor closely next gen
  - Normal invalidity range: 20–50% (bad code, runtime errors — not a timeout symptom)

### Mid-run checkpoints (≈20%, 50% of max_generations)

- [ ] Check status across all runs:
  ```bash
  gigaevo -e <task>/<name> status
  ```
- [ ] **Save** top-10 programs for each run (not just inspect — data in Redis is ephemeral):
  ```bash
  gigaevo -r <prefix>@<db>:<label> top -n 10 \
      --save-dir experiments/<task>/<name>/archives/<label>/gen<N>/
  ```
- [ ] Run test evaluation on each best program (if applicable):
  ```bash
  bash experiments/<task>/<name>/run_test_eval.sh
  ```
- [ ] Record results in `03_plan.md` checkpoint log (gen, date UTC, primary metric per run)
- [ ] If val/test split exists: note val-test gap; flag runs where gap exceeds problem-specific
  overfitting threshold (set in design table)

### Final evaluation (max_generations reached)

- [ ] All runs completed (check Redis gen count or logs)
- [ ] Run full test evaluation for all runs (if applicable)
- [ ] Record final metrics in `03_plan.md`
- [ ] **Archive all runs immediately** (Step 8 below) — do not proceed to Phase 5 without this
- [ ] Hand off to Phase 5: invoke `ml-research-methodologist` for results analysis

---

## Step 8: Archive Run Data

**Redis is ephemeral. This step must run before any flush, reboot, or Phase 5 analysis.**
**Loss of Redis = loss of all evolved programs, all fitness histories, everything.**

### 8a. Record environment (once per experiment, at launch or immediately after)

```bash
pip freeze > experiments/<task>/<name>/environment_freeze.txt
uname -a >> experiments/<task>/<name>/environment_freeze.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> experiments/<task>/<name>/environment_freeze.txt 2>/dev/null || true
```

Commit this to git — it is small and text-only:
```bash
git add experiments/<task>/<name>/environment_freeze.txt
git commit -m "env: record environment for <name>"
```

### 8b. Export and upload archives for each run

```bash
# Dry run first — verify output before uploading
bash tools/experiment/archive_run.sh --exp <task>/<name> --run "<prefix>@<db>:<label>"

# Then upload to GitHub Release (creates release exp/<name> if needed)
bash tools/experiment/archive_run.sh --exp <task>/<name> --run "<prefix>@<db>:<label>" --upload
```

Each archive contains:
- `evolution_data.csv` — all programs, all generations, all metrics from Redis
- `programs/*.py` — source code of every evaluated program
- `top50.json` — top 50 programs with full metadata

The GitHub Release URL is printed at the end. Add it to the PR description.

### 8c. Checklist

- [ ] `archive_run.sh` run for **every** run in this experiment
- [ ] GitHub Release `exp/<name>` created with assets for all runs
- [ ] Release URL added to `experiments/<task>/<name>/PR_DESCRIPTION.md`
- [ ] `environment_freeze.txt` committed to git
- [ ] Verify CSV rows > 0 for each run before proceeding

**Only after Step 8c is complete may Redis be flushed.**

---

## Compatibility Reference

### validate.py return type → required pipeline

| `validate.py` returns | Allowed pipelines | Forbidden |
|---|---|---|
| `dict` | `standard`, `hotpotqa_asi`, `hotpotqa_reflective` | — |
| `tuple[dict, list[dict]]` | `hotpotqa_asi`, `hotpotqa_reflective` | **`standard`** |

### Custom prompts → required pipeline YAML fields

| Config location | Required field |
|---|---|
| `evolution_context` block in pipeline YAML | `prompts_dir: ${prompts.dir}` |
| `mutation_operator` block (or via `_base.yaml`) | `prompts_dir: ${prompts.dir}` |

Missing either field → custom prompts **silently ignored** with no error. Detected via
`[PROMPT FILES]` showing all `[default]` when treatment is expected.

### exec_runner cleanup before relaunch

Stale exec_runner workers (4 per run) survive after the main run process dies and will
repopulate Redis, causing the Redis empty preflight to fail — or worse, silently corrupt
a new run if not caught.

Use `gigaevo flush` — it enforces the correct kill-then-flush ordering:

```bash
# Preview what would happen (dry-run, default)
gigaevo flush --db 0 1 2 3

# Execute: kill workers then flush
gigaevo flush --db 0 1 2 3 --confirm
```

Never flush manually without first killing workers — they repopulate Redis immediately.

---

## Quick-Reference: Common Failure Modes

| Symptom | Root cause | Detection | Fix |
|---|---|---|---|
| Mutation prompts contain raw Python `repr()` | `pipeline=standard` + tuple-returning validate.py | `[PIPELINE BUILDER]` shows `none (base)` | Switch to `pipeline=hotpotqa_asi` |
| Custom prompts silently ignored | `prompts_dir` missing from `evolution_context` | `[PROMPT FILES]` shows all `[default]` | Add `prompts_dir: ${prompts.dir}` to pipeline YAML |
| Chain server non-thinking output | Server restarted in wrong mode | `[preflight]` thinking check fails | Restart server with correct config |
| Redis non-empty at launch | exec_runner workers repopulating | `[REDIS]` shows N keys | Kill workers first, then flush, then re-run dry-run |
| Watchdog crashes ~1h after start | `_last_gen` keyed with wrong labels (copy-paste from prior experiment) | `KeyError` in `watchdog.log` | Fix `_last_gen` to derive from `RUNS` list; restart watchdog |
| Watchdog crashes at start | Wrong PR number, branch, or generate_plot labels | Traceback in `watchdog.log` | Fix config fields; verify with 60s survival test |
| Config dump shows wrong values | Forgot to set `GIGAEVO_PYTHON` — system python3 lacks `redis`, config defaults differ | `--cfg job` fails or shows unexpected paths | Set `GIGAEVO_PYTHON=$GIGAEVO_PYTHON` |
| Wrong seed used | `program_loader.problem_dir` misconfigured | `[SEED PROGRAMS]` shows wrong path | Fix in launch.sh `COMMON_PARAMS` |
