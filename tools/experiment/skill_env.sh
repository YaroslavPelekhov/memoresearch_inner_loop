#!/usr/bin/env bash
# Shared environment for all experiment skills. Source this, don't execute it.
# Derives PROJ from git root so it works on any checkout/worktree.
export GIGAEVO_PYTHON=${GIGAEVO_PYTHON:-/home/jovyan/.mlspace/envs/evo/bin/python3}
export PROJ="$(git rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJ"
