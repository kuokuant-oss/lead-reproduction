# M5 hotwater factorial tree-scaler audit

The scaler-invariance pilot compared the seed-42
`hw_pos_excluded__hw_neg_excluded` cell after cell-specific versus frozen
pooled-reference StandardScaler transforms. The maximum ensemble probability
difference was 0.163691, exceeding the 1e-6 tolerance. Both tree scaler arms
were therefore retained for all twelve contexts.

The runner audit proves the same F4 raw-index manifest, 137-feature matrix
construction, query order, label balance, dtype conversion to `float32`, and
StandardScaler transform contract for both arms. NaN values are permitted by
the canonical M3 feature contract; infinities are rejected before fitting.

The original run did not persist the four base booster models or per-booster
probabilities. A requested per-booster comparison of feature order, NaN routing,
dtype, transformed values, binning, and seed cannot be completed without
retraining the pilot cell, which is prohibited in this audit round. The observed
sensitivity is consequently a pipeline-audit finding, not a scientific result
and not a model-family claim. It must remain unresolved until durable booster
artifacts are recovered or a dedicated no-context-change audit refit is
authorized.
