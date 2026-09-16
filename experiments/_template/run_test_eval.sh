#!/usr/bin/env bash
# Test evaluation script for <experiment name>.
#
# Evaluates the best-by-val program from each run on the held-out test set.
# Run at gen 10, 25, and 50 checkpoints. Record results in 03_plan.md.
#
# Usage:
#   bash experiments/<task>/<name>/run_test_eval.sh
#
# TODO: implement for this problem/experiment.
# Reference: experiments/hotpotqa/push/run_test_eval.sh

set -euo pipefail

PYTHON=${GIGAEVO_PYTHON:-$(command -v python3)}
PROJ=/workspace-SR008.fs2/mathemage/gigaevo-core

echo "TODO: implement run_test_eval.sh for this experiment"
exit 1
