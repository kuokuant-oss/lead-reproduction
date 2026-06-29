# Handoff: Phase E Step 1 validation

**Date**: 2026-06-29
**Scope**: Stage 1/2 acceptance checks before resolving BDG2 evaluation
paradigm unknown #23

## 1a. Meter-aware value-change regression

`tests/test_value_change_regimes.py` now includes an explicit row-by-row
equivalence check for `row_offset_meter_aware`: the multi-meter frame is split
by meter, each single-meter slice is computed with ordinary `row_offset`, and
the resulting `lag_value_diff_1` and `lag_value_ratio_1` series are compared
against the matching rows from the meter-aware multi-meter run with
`pd.testing.assert_series_equal`.

Focused validation:

+ `.venv\Scripts\python.exe -m unittest tests.test_value_change_regimes`
+ Result: `Ran 8 tests in 0.041s`, `OK`.

## 1b. M3 golden regression gate

The M3 default `row_offset` path remains unchanged.

+ M3.2 row-offset actual AUC: `0.992011952050056`
+ M3.2 expected AUC: `0.9920`
+ M3.2 delta: `+0.000011952050056`
+ M3.4 seed-42 ensemble actual AUC: `0.992788643212651`
+ M3.4 expected AUC: `0.9928`
+ M3.4 delta: `-0.000011356787349`
+ Noise floor: `0.0005`

Both deltas are below the required `0.0005` gate.

Commands:

+ `.venv\Scripts\python.exe -c "... fit_m3_2_regime(..., 'row_offset') ..."`
+ `.venv\Scripts\python.exe scripts\run_m3_4_ensemble.py --model-seeds 42 --out .scratch\phaseE-stage3-gates\m3_4_seed42_results.json`

The first full `run_m4_3_timestamp_value_change.py` attempt also produced the
same M3.2 row-offset AUC before timing out in the non-gate `timestamp_merge`
branch.

## 1c. BDG2 meter/weather timestamp alignment diagnostic

Added `scripts/diagnose_bdg2_timezone_alignment.py` as a read-only diagnostic.
It reads `metadata.csv`, `weather.csv`, and raw/cleaned `electricity` and
`chilledwater` meter CSVs. It does not modify `load_bdg2_frame`, perform joins,
or write data files.

Method:

+ For each site/meter aggregate with at least 10 buildings, compute the hourly
  site-level mean load from BDG2 meter files.
+ For weather, use `airTemperature`, reindex per site to the hourly timestamp
  range, and interpolate in time before phase/correlation estimation.
+ Compare the daily temperature peak hour against the aggregate meter-load peak
  hour.
+ Compute the strongest temperature-load Pearson correlation over lags
  `-12..+12` hours.
+ Run the same diagnostic on raw and cleaned meter variants.

Measured output:

+ `airTemperature` missing rate: `0.000387`. The Stage 0 weather null rate
  `0.197133` is the overall weather-table null rate across all weather fields,
  not the `airTemperature` field used for this diagnostic.
+ Chilledwater cleaned: 10 sites, median absolute peak delta `1.0` hour,
  median absolute best lag `1.0` hour, median absolute correlation `0.72825`.
+ Chilledwater raw: 10 sites, median absolute peak delta `1.5` hours, median
  absolute best lag `1.0` hour, median absolute correlation `0.72865`.
+ Electricity cleaned: 16 sites, median absolute peak delta `2.5` hours,
  median absolute best lag `5.5` hours, median absolute correlation `0.28520`.
+ Electricity raw: 16 sites, median absolute peak delta `2.5` hours, median
  absolute best lag `5.5` hours, median absolute correlation `0.27515`.

Conclusion:

The strongest physical signal is chilledwater, where raw and cleaned variants
both align temperature and load peaks within roughly one to two hours and show
high positive temperature-load correlation. This is evidence that BDG2 meter
timestamps and weather timestamps are on the same practical clock basis for the
tested sites. Electricity is weaker and more occupancy-driven, but raw/cleaned
results are internally consistent and do not show a systematic 12-hour or
timezone-sized displacement.

No timezone-misalignment unknown is opened from this diagnostic. The conclusion
is empirical and scoped: it supports the current `(site_id, timestamp)` weather
join for Phase E diagnostics, but it is not a supervised BDG2 anomaly-label
claim and does not resolve unknown #23.
