# Handoff: Phase E Stage 1 patch and Stage 2 BDG2 readiness

**Date**: 2026-06-29
**Scope**: BDG2 real-data bridge, ingestion contract, and GEPIII-only
assumption isolation

## BDG2 reality now in repo facts

Stage 0 inventory uses the real flat Kaggle archive copied into
`data/raw/bdg2/`, not the discarded Git LFS pointer checkout. The generated
report is `docs/reports/bdg2-data-reality.md`.

Key measured facts:

+ 18 CSV files: metadata, weather, and raw/cleaned files for 8 meter types.
+ 1,636 BDG2 buildings and 19 sites.
+ 1,449 buildings have non-empty `building_id_kaggle` and are GEPIII overlap;
  187 are BDG2-only.
+ No native per-row anomaly label exists in the archive.
+ Cleaned null rates are higher than raw null rates for every measured meter
  type, so raw/cleaned differences are a candidate pseudo-label source.

## Stage 1 contract

ADR 0017 records the BDG2 ingestion contract. `load_bdg2_frame` melts wide meter
files into `(building_id, meter, timestamp, meter_reading)` rows, joins metadata
and site-level weather, preserves `building_id_kaggle` / `site_id_kaggle`, and
derives `is_gepiii_overlap`.

The loader intentionally creates no `anomaly` column. The next Phase E decision
is the evaluation-paradigm ADR: forecasting residuals, unsupervised scoring,
GEPIII-trained detector transfer, or raw/cleaned difference pseudo-labels.

## Stage 2 isolation

ADR 0018 records the GEPIII-only assumptions isolated for BDG2:

+ holidays derive years from the frame and can map BDG2 timezones to country
  calendars;
+ the `0.2931` correction is named as GEPIII/Kaggle-only and is not in the BDG2
  loader path;
+ `row_offset` value-change remains the M3 default, while
  `row_offset_meter_aware` is available for BDG2 multi-meter frames;
+ post-processing end-of-year masks derive each row's year length;
+ post-processing meter names distinguish GEPIII numeric ids from BDG2 meter
  strings;
+ site-held-out logic routes through exported `leave_site_out_mask`.

## Verification

+ Full unit suite: `42` tests passed.
+ Golden regression gate: M3.2 actual `0.992011952` vs expected `0.9920`;
  M3.4 actual `0.992788643` vs expected `0.9928`; both within `0.0005`.
+ Real BDG2 smoke loaded actual `electricity_cleaned.csv` rows with
  `is_gepiii_overlap` present and no `anomaly` column.

## Next

Open the Phase E evaluation-paradigm ADR before any headline BDG2 FDD metric.
That ADR should decide whether to use forecasting residuals, unsupervised
scoring, GEPIII-detector transfer, or raw/cleaned difference pseudo-labels.
