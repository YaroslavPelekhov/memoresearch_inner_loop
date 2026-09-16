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
- `PROMOTE` only if `p_lower > U(b)`: skip all remaining intermediate
  decisions and train directly to the full budget;
- otherwise `MORE_EVIDENCE`: advance exactly one rung and decide again.

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

The statistical unit is one candidate/seed. All descendants of the same idea
lineage or campaign are assigned as a group, so correlated candidates cannot
cross split boundaries. Before fitting anything, groups are frozen into four
sets:

1. `train` fits the probability model and defines the winner threshold;
2. `probability_calibration` calibrates its probability intervals;
3. `policy_selection` chooses the gate thresholds;
4. `locked_test` is opened once for the final policy evaluation.

"Winner" is not redefined as the Top-10% of each test set. A fitness threshold
is derived from unique runs in `train`, frozen in the split manifest, and then
applied unchanged to every other split.

For each rung, kill thresholds are selected from out-of-sample predictions in
`policy_selection`. A threshold is admissible only when the exact one-sided 95%
Clopper-Pearson lower bound for eventual-winner recall is at least 0.95.
Runtime calibration uses the upper probability bound because that is the value
used by the kill rule.

Promotion thresholds analogously require an exact lower confidence bound on
precision and use the lower probability bound. Monotonic post-processing may
only widen the uncertainty region, never weaken either statistical constraint.
These bounds constrain policy selection; they are not the final significance
claim, because multiple candidate thresholds were inspected on that split.

First build an unlabeled rung table from complete observe-only trajectories:

```bash
python tools/build-multifidelity-dataset.py \
  --root runs/idea-campaign-001 \
  --output runs/statistics/raw-rungs.jsonl \
  --final-budget 4096
```

Then make the four immutable group-level splits and freeze the winner
definition:

```bash
python tools/split-multifidelity-dataset.py \
  --records runs/statistics/raw-rungs.jsonl \
  --output-dir runs/statistics/splits \
  --winner-fraction 0.10 \
  --seed 20260917
```

The command writes one JSONL file per split plus `split-manifest.json` with the
source hash, split hashes, group assignments, and frozen fitness threshold.
Probability-model training may read only `train`; probability calibration may
read only `probability_calibration`. The manifest deliberately omits the winner
count for `locked_test`.

Fit the paired model variants without opening either selection split:

```bash
python tools/fit-multifidelity-models.py \
  --split-dir runs/statistics/splits \
  --plan config/multifidelity/dclm-140m.yaml \
  --output-dir runs/statistics/models \
  --bootstrap-resamples 200 \
  --confidence 0.95
```

Both variants use the same per-rung regularized linear model class and training
runs. `trajectory_only` uses `F` and `delta F`; `probe_aware` additionally uses
`V`, `H`, and `E`. Each artifact contains a Platt-calibrated winner-probability
head and a final-fitness regression head. Probability intervals come from
resampling whole lineage groups in both train and probability-calibration
splits. Missing early features are imputed with frozen train means. The emitted
`model-freeze-manifest.json` binds both models, the original plan, and every
split hash while recording that locked test has not been opened.

Generate paired predictions on policy selection:

```bash
python tools/predict-multifidelity-models.py \
  --freeze-manifest runs/statistics/models/model-freeze-manifest.json \
  --records runs/statistics/splits/policy_selection.jsonl \
  --split policy_selection \
  --policy-variant probe_aware \
  --output runs/statistics/policy-selection-predictions.jsonl
```

Expected JSONL input to the gate-calibration tool consists of predictions for
`policy_selection` only:

```json
{"split":"policy_selection","run_id":"run-17","group_id":"lineage-4","budget_batches":128,"probability":0.42,"probability_lower":0.31,"probability_upper":0.54,"eventual_winner":false}
```

Run calibration with:

```bash
python tools/calibrate-multifidelity-gates.py \
  --plan config/multifidelity/dclm-140m.yaml \
  --records runs/statistics/policy-selection-predictions.jsonl \
  --output runs/statistics/calibrated-plan.yaml
```

The tool records the prediction-file hash and exact method in a sibling
`.calibration.json` evidence file. It keeps the plan in observe-only mode.
Before touching locked test, bind the models, selected gates, and prediction
receipt into a second immutable manifest:

```bash
python tools/freeze-multifidelity-policy.py \
  --model-freeze-manifest runs/statistics/models/model-freeze-manifest.json \
  --calibrated-plan runs/statistics/calibrated-plan.yaml \
  --calibration-evidence runs/statistics/calibrated-plan.calibration.json \
  --policy-predictions runs/statistics/policy-selection-predictions.jsonl \
  --output runs/statistics/policy-freeze-manifest.json
```

The prediction command refuses locked test unless this policy-freeze manifest
is supplied and the caller explicitly adds `--unlock-locked-test`:

```bash
python tools/predict-multifidelity-models.py \
  --freeze-manifest runs/statistics/policy-freeze-manifest.json \
  --records runs/statistics/splits/locked_test.jsonl \
  --split locked_test \
  --unlock-locked-test \
  --output runs/statistics/locked-test-predictions.jsonl
```

Every stage verifies hashes and emits a receipt. Activation remains a separate
decision after both locked-test claims pass.

## Locked-test decision

The overall cascade is the primary analysis; per-rung results are secondary
diagnostics. Replay the frozen active decisions on the complete observe-only
locked trajectories. Probe batches are counted as compute, so promoted
candidates can cost slightly more than the full-training baseline:

```bash
python tools/simulate-multifidelity-policy.py \
  --policy-freeze-manifest runs/statistics/policy-freeze-manifest.json \
  --predictions runs/statistics/locked-test-predictions.jsonl \
  --output runs/statistics/locked-test-outcomes.jsonl
```

This creates one outcome per candidate:

```json
{"split":"locked_test","run_id":"run-91","group_id":"lineage-22","eventual_winner":true,"survived":true,"full_compute":4096,"cascade_compute":2048}
```

Evaluate it with:

```bash
python tools/evaluate-multifidelity-policy.py \
  --outcomes runs/statistics/locked-test-outcomes.jsonl \
  --output runs/statistics/locked-test-report.json \
  --recall-floor 0.95 \
  --confidence 0.95
```

Certification requires all three conditions:

- the exact one-sided lower bound on winner recall is at least 0.95;
- a lineage-grouped bootstrap lower bound on winner recall is at least 0.95;
- the grouped-bootstrap lower bound on compute saving is above zero.

At 95% confidence, 59 winner candidate/seeds with zero misses are the minimum
for the exact lower recall bound to reach 0.95. Correlated winners do not create
59 independent lineages, so the grouped bootstrap plus at least 20 total groups
and 20 winner-bearing groups remain mandatory. Any failed condition is reported
as a negative or inconclusive result; gates are not retuned on the locked set.

## Probe-mechanism test

The first hypothesis is tested before the binary safety claim (fixed-sequence
gatekeeping controls the family-wise error rate at 5%). On the same locked
candidates, compare two models frozen before opening the split: identical model
class and training data, with the second model adding only `V`, `H`, and `E`.
The preregistered endpoint is the paired reduction in squared final-fitness
error. Its one-sided 95% lower bound is obtained by resampling whole
lineage/campaign groups and must be above zero:

```json
{"split":"locked_test","run_id":"run-91","group_id":"lineage-22","final_fitness":1.84,"trajectory_prediction":1.71,"probe_prediction":1.82}
```

```bash
python tools/evaluate-multifidelity-probe.py \
  --predictions runs/statistics/locked-test-predictions.jsonl \
  --output runs/statistics/locked-test-probe-report.json \
  --budget 128
```

Only if this first claim passes is the end-to-end safety claim interpreted as
the second confirmatory claim. Winner recall and compute saving remain the
endpoints of that frozen policy test; other losses and per-rung comparisons are
supporting analyses. The primary probe budget must be chosen before locked test
is unlocked; changing `--budget` after inspecting results invalidates the
confirmatory interpretation.

## Primary comparisons

Report compute saving versus winner recall for full training, a fixed early
threshold, Successive Halving, trajectory-only learned gates, and
trajectory-plus-probe learned gates. The key ablation uses the same splits and
model class, adding only `V`, `H`, and `E`.

The slow-starter slice is preregistered: candidates below the early-fitness
cutoff that nevertheless finish in the full-budget Top-K. Compare their
survival under trajectory-only and probe-aware policies.

No positive scientific claim should be made from observe-only plumbing tests or
from in-sample probability estimates, from the policy-selection split, or from
per-rung significance without a successful locked end-to-end test.
