# Meter-aware M5/M6 comparison line

## Status

Accepted (2026-07-02)

## Context

M3 keeps `row_offset` as the frozen reproduction default. ADR 0011 and ADR 0024
preserve that numeric line so M3.2/M3.4 golden values, downsampling semantics,
`StandardScaler`, and existing M3 scripts remain comparable.

INV-1 measured the materiality of multi-meter row-offset meter crossing on GEPIII.
On the 50/50 split, `row_offset_meter_aware` changed the 4-model ensemble AUC by
about `+0.0009`, above the ADR 0013 noise floor of `0.0005`, and changed the
ensemble member ordering. The feature-layer check also showed that the default
row-offset value-change cells are frequently different from the meter-aware
version in multi-meter buildings.

Issue #52 therefore implements the comparison line additively. Existing M3
defaults and fixtures are unchanged; M5/M6 comparison scripts must opt into the
comparison regime explicitly.

## Decision

M5/M6 cross-model comparison uses `row_offset_meter_aware` value-change features.

`scripts/run_m5_phaseD_foundation_vs_gbdt.py` keeps `row_offset` as its default
`--value-change-regime` so the older M5 fixture remains reproducible. New M6-facing
comparison runs pass `--value-change-regime row_offset_meter_aware` and write
separate outputs under `data/processed/`.

The 50/50 six-model comparison uses:

+ 50% training / 50% testing held-out splits;
+ the same fit rows, train scoring rows, and test scoring rows for all models in
  each cell;
+ natural-prevalence train/test scoring subsamples;
+ LightGBM, XGBoost, CatBoost, HistGBT, Ensemble, and TabPFN;
+ AUC, PR-AUC, and confusion matrix operating points at threshold `0.5` and
  fixed recall `0.90`;
+ row-index fingerprints in output JSON rather than full row-index arrays, to
  keep tracked provenance below the repository large-file gate.

## Why not `timestamp_merge`

`timestamp_merge` is the strict timestamp-join regime, but it is not the chosen
comparison line here.

`row_offset_meter_aware` fixes the material meter-crossing problem while preserving
the same feature count and the same row-offset family used by the frozen M3
reproduction. It also matches the per-meter row-offset semantics proven in the
existing value-change regime tests. That makes it the smallest additive change
that removes the cross-model comparison confound.

`timestamp_merge` changes more than the meter grouping behavior. It is useful as
a sensitivity regime, but using it as the main comparison line would combine a
meter-crossing fix with a different time-alignment implementation. For M5/M6
model comparison, the intended change is narrower: same row-offset feature
family, meter-aware grouping.

## Consequences

+ M3 defaults, golden metrics, `load_m3_frame`, downsampling, `StandardScaler`,
  and public API exports remain unchanged.
+ M5/M6 comparison reports and JSON outputs must name the active
  `value_change_regime`.
+ Older row-offset M5 outputs are historical fixtures and are not used as current
  cross-model conclusions.
+ Future multi-meter comparison work should use the meter-aware line unless it
  explicitly declares another additive sensitivity regime.
