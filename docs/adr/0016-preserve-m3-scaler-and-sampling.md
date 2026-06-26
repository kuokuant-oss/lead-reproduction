# Preserve M3 scaler and sampling semantics

## Status

Accepted (2026-06-26)

## Context

M4.4 reviewed two preserved-but-underexplained behaviors in the M3 reproduction
path:

+ M3 scripts fit a `StandardScaler` before tree models.
+ `downsample_indices` builds the fit index as `[negs1, pos, negs2, pos]`,
  duplicating positive rows in both positive blocks.

The tracked uses are `src/lead/sample.py`, `src/lead/__init__.py`, and the
M3 scripts `scripts/run_m3_3_budslab.py`, `scripts/run_m3_4_ensemble.py`,
`scripts/run_m3_50_50_ensemble.py`, `scripts/run_m3_5_postprocessing.py`, and
`scripts/run_m3_split_causality.py`.

## Decision

Keep both behaviors as intentional reproduction-compatibility code.

`StandardScaler` remains in the M3 fit path for numeric parity with the
accepted script lineage, even though LightGBM and the other tree boosters do not
require scaled features for their split semantics. `downsample_indices` keeps
the `[negs1, pos, negs2, pos]` shape so positive rows appear exactly twice and
the effective fit set remains 50:50.

## Rationale

M4 is a foundation milestone, not a modeling milestone. Removing either behavior
would create a model-path change solely to simplify code. Keeping the current
path makes labels, features, sampled indices, and fit preprocessing unchanged,
so the expected AUC delta for M3.2 and M3.4 is exactly `0`.

## Consequences

+ M3.2 and M3.4 reruns are not required for this keep/no-behavior-change
  decision because the executable path is byte-identical apart from comments and
  tests.
+ `tests/test_sampling_semantics.py` machine-checks that positives are
  duplicated exactly twice and that the sampled fit set remains 50:50.
+ Future sampling or preprocessing replacements must go through the regression
  gate in `tests/golden_metrics.json` before changing the M3 headline path.
