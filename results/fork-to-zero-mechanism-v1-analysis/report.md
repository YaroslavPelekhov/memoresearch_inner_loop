# Fork-to-zero matched-control analysis

Verdict: **not supported for candidate ranking; possible instability stress-test signal**.

This analysis compares a 256-batch LR-to-zero fork at batch 1,024 against an equal-compute continuation of the original scheduler. All reported predictive targets are the untouched main-branch fitness at batch 4,096.

## Dataset

- 47 complete experimental runs with valid matched pairs.
- 36 unique experimental commits across 12 idea lineages and two campaigns.
- Cross-campaign models use 35 commits; commits observed in both campaigns are excluded from that comparison.
- Exclusions: {"all_trajectories": 52, "baselines": 2, "incomplete": 3}.

## Direct signal

- Equal-compute control vs final: rho=0.898 (exploratory 95% commit-bootstrap CI 0.750 to 0.969).
- Fork-to-zero vs final: rho=0.905 (CI 0.766 to 0.969).
- Difference in correlation (fork minus control): 0.006 (CI -0.055 to 0.074).
- Fork effect Z-C vs final: rho=0.134.
- Fork and control signals correlate at rho=0.962.

## Cross-campaign prediction

- Trajectory + control: MAE=0.003172, RMSE=0.014092.
- Trajectory + Fork-to-zero: MAE=0.002298, RMSE=0.009713.
- Paired squared-error gain, control minus fork: 0.000104253 (exploratory 95% commit-bootstrap CI -0.000000166 to 0.000312755).
- Campaign-specific squared-error gains: {"fork-to-zero-mechanism-v1-gpu0": 1.5900360992273678e-07, "fork-to-zero-mechanism-v1-gpu1": 0.0002603930768983335}.
- The apparent gain is dominated by commit `b3a210ded839250e22eb148718fc9d401ec90052`: main@1024=0.163977, control=0.103125, fork=0.030137, and final=0.097236.
- Without that commit, control MAE=0.000302 and Fork-to-zero MAE=0.000308; the absolute-error gain becomes -0.000006008.

## Candidate selection within idea lineages

- Mean top-1 regret, control: 0.000037.
- Mean top-1 regret, Fork-to-zero: 0.000040.
- Regret gain, control minus fork: -0.000003 (lineage-bootstrap CI -0.000008 to 0.000000).
- Pairwise-ranking gain, fork minus control: 0.052.
- Early-rank-reversal accuracy gain, fork minus control: 0.048.
- For reference, raw main@1024 has mean top-1 regret 0.000005 and top-1 hit rate 0.750, versus Fork-to-zero regret 0.000040 and hit rate 0.417.

## Incomplete-run audit

- `3e4d68594abb98030745fcef4ed6f1e8645d8d47` at main/budget-512/wrapper.stderr.log: status=train_failed, error=unknown_process_failure, nonfinite_batch=None.
- `c842b9869feea1c74e615a4151e5c7fc436e7ac9` at main/budget-2048/wrapper.stderr.log: status=nonfinite_training_gradient, error=None, nonfinite_batch=1129.
- `c842b9869feea1c74e615a4151e5c7fc436e7ac9` at probes/budget-1024/wrapper.stderr.log: status=nonfinite_training_gradient, error=None, nonfinite_batch=1199.
- `3e936d2dbe9888af4b59f74840b00a9fbe8ece62` at main/budget-2048/wrapper.stderr.log: status=train_failed, error=cuda_oom, nonfinite_batch=None.

## Interpretation limits

The comparison is post-collection exploratory. More importantly, repeated parent commits connect the six sequential lineages on each GPU. The dataset therefore contains only two independent campaign-level clusters, not twelve independent experiments. The next confirmatory run must freeze this analysis and use independently seeded candidate pools rather than a sequential chain.
