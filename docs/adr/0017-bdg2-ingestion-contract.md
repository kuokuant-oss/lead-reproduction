# BDG2 ingestion contract

## Status

Accepted (2026-06-29)

## Context

Phase E moves FDD work from GEPIII to the real BDG2 corpus. A prior BDG2 loader
was retired because it guessed schema and labels from synthetic fixtures. Stage
0 therefore downloaded the real Kaggle archive
`claytonmiller/buildingdatagenomeproject2`, discarded the Git LFS pointer
checkout, and measured the flat CSV files in
`docs/reports/bdg2-data-reality.md`.

The measured archive contains:

+ `metadata.csv`: 1,636 buildings, 19 sites, string `building_id` values, and
  metadata columns including `site_id`, `building_id_kaggle`,
  `site_id_kaggle`, `primaryspaceusage`, `sqft`, `yearbuilt`,
  `numberoffloors`, and `timezone`.
+ `building_id_kaggle` is non-empty for 1,449 buildings and empty for 187
  BDG2-only buildings. These bridge columns must be preserved so GEPIII-overlap
  evaluation can be separated from true BDG2-only transfer.
+ Six site timezones: `Europe/Dublin`, `Europe/London`, `US/Central`,
  `US/Eastern`, `US/Mountain`, and `US/Pacific`.
+ Sixteen meter files: raw and cleaned variants for electricity, chilled water,
  steam, hot water, gas, water, irrigation, and solar.
+ Each meter file is a wide table with `timestamp` plus one column per building
  id; all measured meter files have 17,544 hourly rows from `2016-01-01
  00:00:00` through `2017-12-31 23:00:00`.
+ `weather.csv`: 331,166 rows keyed by `(site_id, timestamp)`, with no explicit
  timezone column.
+ No per-row anomaly labels or label-like files.

## Decision

Add `src/lead/bdg2.py` and export `load_bdg2_frame` as an additive public API.
Do not modify `load_m3_frame` or any existing GEPIII numeric path.

The BDG2 frame contract is:

+ Long meter rows keyed by `(building_id, meter, timestamp)`.
+ `building_id` remains the BDG2 string id, not a GEPIII integer surrogate.
+ `meter` is the BDG2 meter type string from the file stem, not the GEPIII
  numeric meter code.
+ `meter_reading` is the melted cell value from the wide meter file.
+ Metadata joins map Stage 0 fields as follows: `primaryspaceusage` to
  `primary_use`, `sqft` to `square_feet`, `yearbuilt` to `year_built`, and
  `numberoffloors` to `floor_count`.
+ Metadata joins retain `building_id_kaggle` and `site_id_kaggle`, and derive
  `is_gepiii_overlap` from non-empty `building_id_kaggle`.
+ Weather joins on `(site_id, timestamp)` and renames weather fields to the
  existing snake_case style where possible.
+ The loader creates no `anomaly` column because BDG2 provides no per-row
  anomaly labels.

Use the cleaned BDG2 meter files by default for forecasting-oriented ingestion.
Keep raw files available through `variant="raw"` for audits and sensitivity
checks. Stage 0 measured higher cleaned null rates than raw null rates for every
meter type, so an FDD/anomaly-detection path must re-evaluate this default:
cleaning can remove the abnormal readings the detector is meant to find. The
GEPIII site-0/meter-0 correction must not be applied to the BDG2 raw/cleaned
path; BDG2 unit semantics are handled by its raw/cleaned release distinction
and are not the same as the GEPIII/Kaggle subset correction.

Weather joins use `(site_id, timestamp)`. `weather.csv` has no timezone column,
so site timezone comes from metadata. GEPIII had a known weather/meter timezone
alignment issue; BDG2 raw/cleaned should not inherit that assumption silently.
Phase E must verify the claimed timestamp alignment rather than treating it as
proven by column names.

For labels, Phase E will not fabricate supervised labels. Until a later ADR
chooses the evaluation paradigm, valid strategies are limited to:

+ unsupervised detection on BDG2 readings,
+ forecasting-residual scoring,
+ applying a GEPIII-trained detector as an unlabeled cross-dataset scoring
  baseline,
+ raw/cleaned difference pseudo-labels, where cells present in raw but set to
  `NaN` in cleaned are treated as candidate BDG2-cleaning-identified bad
  readings.

For splits, distinguish within-BDG2 building/site hold-outs from true
cross-dataset transfer:

+ building-level splits separate BDG2 buildings inside the BDG2 corpus;
+ site-held-out splits separate BDG2 sites inside the BDG2 corpus;
+ GEPIII-to-BDG2 transfer is the first true cross-dataset test and must not be
  described as equivalent to `site_id % k` inside one dataset.

## Rationale

The Stage 0 report shows that BDG2's real schema differs from GEPIII in the
places that matter most: string building ids, eight meter types, wide meter
files, site-level weather, multiple site timezones, and no row labels. Keeping
these differences visible in the loader prevents silent GEPIII assumptions from
entering Phase E.

The cleaned default gives Phase E a practical starting point while preserving a
raw-file audit path. The optional slicing arguments in `load_bdg2_frame` make
tests and exploratory probes possible without materializing the full corpus.

## Consequences

+ `lead.__all__` now has one additive export, `load_bdg2_frame`.
+ `tests/test_bdg2_loader.py` covers schema-compatible loading, meter-column
  metadata guards, weather-key guards, reshape length, and row-key uniqueness.
+ Existing M3 loaders, scripts, feature generation, and regression metrics are
  unchanged by this ADR.
+ Stage 2 must still address GEPIII-only assumptions in holidays, unit
  correction, meter-aware value-change computation, post-processing meter names,
  site-held-out helper export, and day-of-year/year-length boundaries such as
  the `dayofyear > 366.9583` post-processing rule that only matches a leap-year
  2016 frame.
