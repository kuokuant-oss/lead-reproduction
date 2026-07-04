# Meter-aware M5/M6 comparison line

## Status

Superseded by the 2026-07-04 timestamp-merge re-baseline.

## Context

This ADR recorded the 2026-07-02 meter-aware comparison line before the
2026-07-04 timestamp-merge re-baseline. M3 now keeps `timestamp_merge` as the
canonical reproduction default. ADR 0011 records the current default.

INV-1 measured the materiality of multi-meter row-offset meter crossing on GEPIII.
On the 50/50 split, `row_offset_meter_aware` changed the 4-model ensemble AUC by
about `+0.0009`, above the ADR 0013 noise floor of `0.0005`, and changed the
ensemble member ordering. The feature-layer check also showed that the default
row-offset value-change cells are frequently different from the meter-aware
version in multi-meter buildings.

Issue #52 originally implemented the comparison line additively. The current
M5/M6 comparison scripts now default to `timestamp_merge`; meter-aware outputs
remain historical ladder inputs.

## Decision

M5/M6 cross-model comparison uses `timestamp_merge` value-change features.

`scripts/run_m5_phaseD_foundation_vs_gbdt.py`,
`scripts/run_m5_phaseC_tabpfn_spike.py`, and
`scripts/run_m6_phaseD_50_50_full_models.py` default to `timestamp_merge`.
Older `row_offset` and `row_offset_meter_aware` outputs remain separate
historical sensitivity artifacts under `data/processed/`.

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

## Historical note

`row_offset_meter_aware` fixed the material meter-crossing problem while
preserving the row-offset family. It is still useful as a ladder/sensitivity
regime. The canonical line moved to `timestamp_merge` because that regime is
the buds-lab-faithful timestamp-join implementation.

## Consequences

+ M3 defaults and golden metrics were re-baselined to `timestamp_merge`.
+ M5/M6 comparison reports and JSON outputs must name the active
  `value_change_regime`.
+ Older row-offset M5 outputs are historical fixtures and are not used as current
  cross-model conclusions.
+ Future multi-meter comparison work should use `timestamp_merge` unless it
  explicitly declares another additive sensitivity regime.
