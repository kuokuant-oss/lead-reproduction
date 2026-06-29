# Handoff: Phase E Step 3 smoke transfer

**Date**: 2026-06-29
**Scope**: First BDG2 execution under ADR 0019; smoke only, no headline metric

## Forced decisions before execution

### Detector

Step 3 uses the GBDT line as the transfer detector, specifically the M3.2
LightGBM detector trained on the canonical GEPIII 80/20 source split. This keeps
the detector in the existing GBDT family selected for real-time feasibility:
sub-second-style inference, no TabPFN context window, and a path that can later
scale to all BDG2 sites.

M3.4's four-model ensemble remains the stronger canonical GEPIII headline, but
the single LightGBM member is the first smoke detector because it is materially
cheaper and matches the same M3.2 137-feature offline table. TabPFN is not used
for Step 3 full-transfer smoke: Phase D measured about `6.3` ms/row, roughly
`100x` slower than GBDT, and a melted BDG2 all-meter frame is roughly tens of
millions of rows, beyond the practical full-transfer shape for TabPFN-3's
documented `1,000,000 x 200` row/feature limit. TabPFN is reserved for later
small in-context slices.

### Meter and Site

The first BDG2 smoke locks to `meter == "chilledwater"` and the single
chilledwater site with the most buildings. The measured max site is `Fox` with
101 chilledwater buildings, including GEPIII-overlap and BDG2-only rows for ADR
0019 stratification.

Chilledwater is chosen because Phase E Step 1 empirically supported the
weather-load time basis for this meter: median absolute correlation about
`0.73` and median best lag `1` hour. Electricity is excluded from this smoke
because unknown #25 remains open (`5.5` hour median best lag, correlation about
`0.28`) and must be reviewed per site if electricity enters scoring.

### Feature and Causal Regime

This smoke is explicitly **offline** scoring. It uses the M3.2 offline shift set
for compatibility with the selected GEPIII detector. BDG2 value-change features
must use `row_offset_meter_aware`, even in a single-meter smoke, so the pipeline
does not accidentally normalize the unsafe multi-meter `row_offset` path. Any
later real-time/online FDD claim must switch to `PAST_SHIFTS` only per ADR 0007
and ADR 0011.

Single-meter regime note: the GEPIII source detector records
`value_change_regime="row_offset"` while BDG2 scoring records
`"row_offset_meter_aware"`. For this one-meter chilledwater slice these are
semantically equivalent; Step 1 proved row-by-row equality between meter-aware
multi-meter scoring and per-meter row-offset scoring. A future multi-meter
transfer must either train/score through an equivalent meter-aware path or keep
one detector per meter, so train/serve value-change semantics do not diverge.

## Smoke Output

Output JSON:

+ `.scratch/phaseE-step3-bdg2-transfer-smoke.json`

Command:

+ `.venv\Scripts\python.exe scripts\run_phaseE_step3_bdg2_transfer_smoke.py --out .scratch\phaseE-step3-bdg2-transfer-smoke.json`

Loaded BDG2 scope:

+ Site: `Fox`
+ Meter: `chilledwater`
+ Buildings: `101`
+ Raw rows loaded: `1,771,944`
+ Cleaned rows loaded/scored: `1,771,944`
+ Raw and cleaned frames both satisfied the ADR 0017 schema: `(building_id,
  meter, timestamp, meter_reading)` plus `building_id_kaggle`,
  `site_id_kaggle`, `is_gepiii_overlap`, and weather columns from the
  `(site_id, timestamp)` join.

Unlabeled score-transfer smoke:

+ Scored variant: `cleaned`
+ Score coverage: `1.0`
+ Missing-score rate: `0.0`
+ All-row score median: `0.007116834175596456`
+ GEPIII-overlap rows: `1,736,856`; median score:
  `0.007009272301667491`
+ BDG2-only rows: `35,088`; median score: `0.15755569665583008`
+ Median score delta, overlap minus BDG2-only: `-0.1505464243541626`
+ Scoring runtime: `0.41858649998903275` seconds for `1,771,944` rows
  (`4,233,160.888003856` rows/second)
+ End-to-end smoke runtime, including source-detector training and data load:
  `108.82870780001394` seconds

Metric-contract check:

+ The JSON marks `bdg2_ground_truth_metrics_reported=false`.
+ The smoke reports unlabeled score coverage, score distribution, runtime, and
  `is_gepiii_overlap` stratification only.
+ It does not report BDG2 ground-truth ROC-AUC, PR-AUC, precision, recall, or
  F1.

Interpretation boundary:

This is a smoke result, not a headline result. The BDG2-only median is higher
than the GEPIII-overlap median in this single-site chilledwater slice, but that
is only an unlabeled score-distribution contrast. It is not accuracy, not
pseudo-label agreement, and not BDG2 readiness evidence until the next stage
plans and validates the full transfer scope under ADR 0019.

Next step after user confirmation: plan full chilledwater transfer across all
sites, preserving BDG2-only / GEPIII-overlap reporting and still excluding
electricity until unknown #25 is reviewed.
