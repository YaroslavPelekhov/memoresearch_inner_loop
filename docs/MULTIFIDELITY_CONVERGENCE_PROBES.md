# Adaptive multi-fidelity evaluation

This repository evaluates the following two primary hypotheses:

1. A short, isolated LR-to-zero convergence probe contains incremental signal
   about a candidate's full-budget outcome beyond its passive training curve.
2. A cascade with statistically calibrated, shrinking decision gates can save
   compute while preserving a preregistered lower confidence bound on eventual
   winner recall.

The weaker claim that early trajectories correlate with final outcomes is a
baseline, not the novelty claim.

## Runtime contract

The default plan is `config/multifidelity/dclm-140m.yaml`. It uses checkpoints
at 128, 256, 1,024, 2,048, and 4,096 batches. These are operational defaults
and may be changed before data collection; changing them after collection starts
creates a new protocol version.

At every non-final rung the runner:

1. advances the main branch to the rung and saves a full-state checkpoint;
2. hashes that checkpoint;
3. loads it in a separate run with no checkpoint output configured;
4. replaces only the post-fork LR path with a cosine decay to zero;
5. evaluates the probe and verifies that the main checkpoint hash is unchanged;
6. records `F`, `delta F`, `V`, `H = V - F`, and
   `E = V(b) - V(previous)`;
7. records the recommended gate action and advances the untouched main branch.

The main scheduler is resolved once against the full 4,096-batch horizon and
stored as absolute boundaries. A 128-batch checkpoint therefore does not
compress warmup or trigger an artificial early decay.

Each candidate writes an atomic, resumable
`multifidelity-trajectory.json`. Probe logs and effective configs remain for
audit, but probes never write model checkpoints.

## Decision rule

The probability model estimates the event "candidate finishes in the
full-budget Top-K" and returns a calibrated interval `[p_lower, p_upper]`.
For rung gates `L(b)` and `U(b)`:

- `KILL` only if `p_upper < L(b)`;
- `PROMOTE` only if `p_lower > U(b)`;
- otherwise `MORE_EVIDENCE`.

The plan validator requires `L(b)` to be nondecreasing, `U(b)` to be
nonincreasing, and the uncertainty width `U(b)-L(b)` to shrink with budget.
Active pruning is rejected unless both the plan and probability model are
marked calibrated.

The checked-in plan is intentionally `observe_only: true`. It records what the
policy would have done but continues every trajectory to 100%, which is needed
to obtain unbiased labels for the first training/calibration dataset.

## Statistical gate calibration

Gate significance concerns prediction of the final winner event, not whether
two early losses differ at `p < 0.05`.

For each rung, kill thresholds are selected on held-out, out-of-sample
predictions. A threshold is admissible only when the one-sided 95% Wilson lower
bound for eventual-winner recall is at least 0.95. Runtime calibration uses the
upper probability bound because that is the value used by the kill rule.

Promotion thresholds analogously require a lower confidence bound on precision
and use the lower probability bound. Monotonic post-processing may only widen
the uncertainty region, never weaken either statistical constraint.

Expected JSONL input to the calibration tool:

```json
{"budget_batches":128,"probability":0.42,"probability_lower":0.31,"probability_upper":0.54,"eventual_winner":false}
```

Build the labeled rung dataset from complete observe-only trajectories with:

```bash
python tools/build-multifidelity-dataset.py \
  --root runs/idea-campaign-001 \
  --output runs/calibration/rung-dataset.jsonl \
  --final-budget 4096 \
  --winner-fraction 0.10
```

Probability-model training must use run-level splits: all rungs from one run
belong to the same fold. Gate calibration consumes predictions for runs that
were not used to fit that prediction model. The runtime model artifact is a
JSON object with a `model_id`, `calibrated: true`, and one logistic model per
budget (`intercept`, feature `coefficients`, and a held-out
`interval_radius`).

Run calibration with:

```bash
python tools/calibrate-multifidelity-gates.py \
  --plan config/multifidelity/dclm-140m.yaml \
  --records runs/calibration/out-of-sample-predictions.jsonl \
  --output runs/calibration/calibrated-plan.yaml
```

The tool keeps the output in observe-only mode. Activation is a separate,
explicit protocol decision after checking probability calibration, sample size,
winner recall, false-kill rate, and compute saving.

## Primary comparisons

Report compute saving versus winner recall for full training, a fixed early
threshold, Successive Halving, trajectory-only learned gates, and
trajectory-plus-probe learned gates. The key ablation uses the same splits and
model class, adding only `V`, `H`, and `E`.

The slow-starter slice is preregistered: candidates below the early-fitness
cutoff that nevertheless finish in the full-budget Top-K. Compare their
survival under trajectory-only and probe-aware policies.

No positive scientific claim should be made from observe-only plumbing tests or
from in-sample probability estimates.
