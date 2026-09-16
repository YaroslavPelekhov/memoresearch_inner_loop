# Euler multi-fidelity pilot runbook

The isolated workspace is `/home/bulatov/yaroslav-multifidelity`. Code, conda
environment, DCLM train/heldout data, CORE fixtures, Codex state, run outputs,
and logs live below that directory.

The runtime also avoids shared-user service state: Redis uses the private
directory `redis/` and loopback port `16379`; the Codex SOCKS/HTTP proxy uses
loopback ports `11880`/`11881`, with binaries, configuration, PID files, and
logs under `proxy/`. The launcher refuses an occupied port unless its PID file
and process command both resolve to this workspace.

Connect through the shared-account isolation wrapper:

```bash
ssh -t bulatov@135.106.168.25 /home/bulatov/.guest-zsh/login
cd /home/bulatov/yaroslav-multifidelity/repo
```

Validate both lanes without starting them:

```bash
tools/launch-multifidelity-pilot --gpu 0
tools/launch-multifidelity-pilot --gpu 1
```

Starting a lane always requires the explicit `--start` guard:

```bash
tools/launch-multifidelity-pilot --gpu 0 --start
tools/launch-multifidelity-pilot --gpu 1 --start
```

The defaults create independent four-round campaigns under
`runs/multifidelity-pilot-gpu0` and `runs/multifidelity-pilot-gpu1`. Each
candidate uses one visible GPU and the checked-in observe-only multi-fidelity
plan. Override the campaign name or pilot size only before starting:

```bash
tools/launch-multifidelity-pilot \
  --gpu 0 \
  --campaign multifidelity-pilot-gpu0 \
  --target-rounds 4 \
  --evolution-generations 4 \
  --start
```

Inspect without changing state:

```bash
tmux ls
tail -f /home/bulatov/yaroslav-multifidelity/logs/multifidelity-pilot-gpu0.log
nvidia-smi
```

Do not activate pruning during the pilot. Complete observe-only trajectories
are required to fit the two probability models, calibrate gates, and preserve
the locked-test protocol.
