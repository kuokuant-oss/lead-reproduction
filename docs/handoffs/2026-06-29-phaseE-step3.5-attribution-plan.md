# Handoff: Phase E Step 3.5 attribution and full-transfer plan

**Date**: 2026-06-29
**Scope**: Attribute Step 3 smoke score split; plan full chilledwater transfer
without running it

## Attribution Target

Step 3 smoke on Fox/chilledwater produced an unlabeled score-distribution split:
BDG2-only median score `0.15755569665583008` versus GEPIII-overlap median score
`0.007009272301667491`, roughly `22x`. Under ADR 0019, this is not accuracy,
not readiness, and not a headline metric. Step 3.5 diagnoses whether this is
more likely true fault signal or detector extrapolation / OOD artifact.

Diagnostic output:

+ `.scratch/phaseE-step3.5-smoke-attribution.json`

Command:

+ `.venv\Scripts\python.exe scripts\diagnose_phaseE_step3_smoke_attribution.py --out .scratch\phaseE-step3.5-smoke-attribution.json`

## Evidence

| Check | GEPIII-overlap | BDG2-only |
| --- | ---: | ---: |
| Rows | `1,736,856` | `35,088` |
| Buildings | `99` | `2` |
| Median score | `0.007009272301667491` | `0.15755569665583008` |
| `primary_use_unseen_rate` | `0.0` | `0.0` |
| `log_square_feet` missing rate | `0.0` | `0.0` |
| `meter_reading` missing rate | `0.011166728847987398` | `0.5036194710442317` |
| `lag_value_diff_1` missing rate | `0.01361598198123506` | `0.5038189694482444` |
| `lag_value_diff_168` missing rate | `0.024833952843528767` | `0.5168148654810761` |
| `square_feet` median | `83,660.0` | `143,071.5` |
| `square_feet` max | `553,210.0` | `244,964.0` |
| chilledwater reading median | `517.8579` | `528.2474` |
| chilledwater reading max | `18,340.9776` | `4,125.5482` |

Reference GEPIII source-train chilledwater distribution:

+ `square_feet` median `86,166.9609375`, max `861,523.75`.
+ chilledwater `meter_reading` median `110.06900024414062`, max `880,374.0`.

## Attribution Call

Call: **(ii) detector OOD / feature-distribution artifact is more likely than
true anomaly signal**.

Reasoning:

+ There is no BDG2 label in this path, so true anomaly frequency cannot be
  inferred from the score split.
+ `primary_use_unseen_rate` is `0.0` in both strata, so unseen primary-use
  category is not the driver.
+ `log_square_feet` has no missingness in either stratum; BDG2-only buildings
  are larger at the median but still within the GEPIII chilledwater source range.
+ The dominant difference is missingness: BDG2-only has about `50%`
  `meter_reading` and value-change lag missingness, versus about `1-2.5%` in
  GEPIII-overlap rows. LightGBM can route missing values, so the higher scores
  are plausibly detector behavior on a sparse/OOD slice rather than validated
  anomalies.
+ Chilledwater reading medians are similar between overlap and BDG2-only, and
  both are inside the broad GEPIII source range. The large score split therefore
  tracks sparsity and stratum composition more than raw magnitude alone.

Unknown #26 records this as a full-transfer interpretation gate. Full transfer
may continue as unlabeled score transfer only if OOD/missingness flags travel
with every output and absolute scores are not treated as calibrated risk.

## Value-Change Regime Clarification

Current smoke:

+ GEPIII detector source summary: `value_change_regime="row_offset"`.
+ BDG2 scoring summary: `value_change_regime="row_offset_meter_aware"`.

For a single-meter chilledwater transfer, these are equivalent. Step 1 proved
that meter-aware scoring over a multi-meter frame equals per-meter row-offset
scoring row by row. This avoids train/serve skew for the single-meter smoke.

Full-transfer precondition: if Phase E expands to multiple meters, detector and
scoring value-change semantics must be made equivalent before scoring. Either
train and score through a meter-aware-equivalent path or restrict transfer to
one meter per detector. Do not combine multiple meters under source `row_offset`
and target `row_offset_meter_aware` without a fresh equivalence proof.

## Full Chilledwater Transfer Plan

This is a plan only; no full transfer was executed in Step 3.5.

### Scored Variant

Use **raw** chilledwater as the primary full-transfer scoring variant.

Reason: ADR 0017 records that cleaned meter files can remove abnormal readings.
For anomaly detection, scoring only cleaned data risks scoring after the
candidate anomalies have been removed. Cleaned chilledwater should still be
loaded as a bridge/sensitivity companion, but raw is the planned primary
unlabeled score-transfer input. If raw/cleaned differences are later used as
pseudo-labels, every metric and prose reference must say `pseudo-label` per ADR
0019.

### Detector

Use the **M3.4 seed-42 four-model ensemble** as the planned primary full-transfer
detector, because it is the accepted canonical GEPIII headline detector over
the M3.2 137-feature table. Keep the M3.2 LightGBM detector as the engineering
smoke/calibration detector because it is cheaper and already proved the BDG2
plumbing.

Full-transfer output should record both the primary detector identity and, if
run, the single-model calibration sidecar. No BDG2 ground-truth classification
metric is allowed for either detector.

### Reporting Contract

+ Headline transfer evidence, when eventually allowed, must be reported on
  BDG2-only buildings (`is_gepiii_overlap == False`) or held-out BDG2 sites.
+ GEPIII-overlap rows are bridge/calibration evidence only.
+ Every output must preserve `is_gepiii_overlap` stratification.
+ Every output must include OOD/missingness flags, including meter-reading
  missingness, value-change missingness, square-foot distribution position, and
  primary-use unseen status.
+ Prefer within-run rank/quantile summaries over absolute score interpretation
  until unknown #26 is resolved.

### Meter Scope

The first full-transfer version remains **chilledwater only**. Electricity stays
excluded until unknown #25 gets a per-site time-basis review. Other meters remain
future slices.

### Memory and Scale

Full-site chilledwater is about `555 x 17,544 = 9,736,920` rows per raw/cleaned
variant before filtering. Plan the runner as per-site or chunked scoring:

+ load one site at a time with `load_bdg2_frame`;
+ score raw first, optionally cleaned as companion;
+ write per-site JSON/Parquet shards under ignored `.scratch` or
  `data/processed`;
+ aggregate only score summaries and stratified counts in memory;
+ never load all eight meters or all raw/cleaned variants into memory at once.
