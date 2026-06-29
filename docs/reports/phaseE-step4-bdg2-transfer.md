# Phase E Step 4: BDG2 chilledwater transfer gate correction

**Date**: 2026-06-29
**Issue**: [#39](https://github.com/kuokuant-oss/lead-reproduction/issues/39)
**Accepted evidence**:

+ `.scratch/phaseE-step4a-bdg2-transfer-pilot.json`
+ `.scratch/phaseE-step4c-pooled-powered-fallback.json`

**Quarantined evidence from the prior overrun**:

+ `.scratch/phaseE-step4a-bdg2-transfer-full.json`
+ `.scratch/phaseE-step4b-tabpfn-vs-gbdt-bdg2.json`

The full and Step 4b artifacts above were produced before the corrected pilot
gate. They are retained as local diagnostic artifacts only. They are not accepted
Phase E results, not a passed full transfer, and not a basis for a readiness
claim. Both JSON files now carry `quarantined=true`,
`quarantine_reason`, and `metric_contract.headline_metric=false`.

## Contract

This slice follows ADR 0019. BDG2 has no native per-row anomaly label in the
measured archive, so all accepted outputs are unlabeled score-transfer
diagnostics. They are not BDG2 ground-truth ROC-AUC, PR-AUC, precision, recall,
F1, anomaly prevalence, calibrated risk, or real-time FDD evidence.

Unknown #26 remains the controlling interpretation gate: BDG2 score contrasts
must travel with OOD and missingness summaries, and score/rank summaries must be
read by completeness stratum rather than as a mixed headline.

## Correction

The first Step 4 implementation treated `score_coverage=1.0` plus any BDG2-only
rows as enough to pass the pilot. That was only a plumbing check. The corrected
gate now splits each overlap stratum by `(building_id, meter)` direct-reading
completeness:

+ `sufficient_obs`: building/meter `meter_reading` missing rate is at most 50%.
+ `high_missing`: building/meter `meter_reading` missing rate is above 50%.

The gate requires a powered `bdg2_only__sufficient_obs` stratum and a powered
`gepiii_overlap__sufficient_obs` baseline before any next-stage scoring can be
accepted. The current minimum is at least 5 buildings and at least 17,544 rows;
row count does not substitute for building diversity. If the BDG2-only powered
stratum is absent, the verdict is `underpowered` and the next step is
`stop_and_report`. If the BDG2-only stratum is powered but the overlap baseline
is not, the verdict is `indeterminate_no_overlap_baseline` and the next step is
also `stop_and_report`.

## Pilot Rerun

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_phaseE_step4a_bdg2_transfer.py --mode pilot --include-cleaned --out .scratch\phaseE-step4a-bdg2-transfer-pilot.json
```

Pilot sites were `Fox` and `Swan`. The corrected gate result:

+ `status`: `failed`
+ `verdict`: `underpowered`
+ `allowed_next_step`: `stop_and_report`
+ `failures`: `pilot has no powered bdg2_only__sufficient_obs stratum`

Raw pilot strata:

| Site | BDG2-only sufficient rows | BDG2-only sufficient buildings | BDG2-only sufficient median | BDG2-only high-missing rows | BDG2-only high-missing buildings | GEPIII-overlap sufficient rows | GEPIII-overlap sufficient median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fox | 17,544 | 1 | 0.1302 | 17,544 | 1 | 1,736,856 | 0.0043 |
| Swan | 0 | 0 | n/a | 350,880 | 20 | 0 | n/a |

Fox has a visible sufficient-observation BDG2-only score contrast, but it is one
building only and therefore underpowered. Swan has many BDG2-only buildings, but
all are high-missing under the corrected building/meter completeness split.

## Control Anchor

The pilot artifact now includes a Fox cleaned M3.2 LightGBM control anchor:

+ BDG2-only median score: `0.15755569665583008`
+ GEPIII-overlap median score: `0.007009272301667491`

This reproduces the Step 3 smoke shape as a control anchor. It does not make the
Step 4 pilot pass, because the corrected gate is about powered
sufficient-observation BDG2-only evidence, not merely reproducing the plumbing
or the earlier score split.

## Pooled Raw Fallback

After the corrected pilot remained underpowered, Step 4c pooled raw
chilledwater scoring across the available BDG2 sites as a diagnostic
stratum-power fallback. This pooled artifact is still ADR 0019 unlabeled
score-transfer evidence. It is not a full-transfer headline, not Step 4b, not a
BDG2 ground-truth metric, and not a readiness claim.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_phaseE_step4c_pooled_powered_fallback.py --out .scratch\phaseE-step4c-pooled-powered-fallback.json
```

Pooled gate result:

+ `status`: `failed`
+ `verdict`: `underpowered_even_pooled`
+ `allowed_next_step`: `stop_and_report`
+ `failure`: `pooled chilledwater has no powered bdg2_only__sufficient_obs stratum`

Pooled raw completeness strata:

| Stratum | Buildings | Rows | Median score | p05 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEPIII-overlap sufficient_obs | 516 | 9,052,704 | 0.0076 | 0.0017 | 0.9912 |
| GEPIII-overlap high_missing | 13 | 228,072 | 0.2211 | 0.0062 | 0.4819 |
| BDG2-only sufficient_obs | 3 | 52,632 | 0.9911 | 0.0052 | 0.9952 |
| BDG2-only high_missing | 23 | 403,512 | 0.1077 | 0.0021 | 0.5613 |

The pooled BDG2-only sufficient-observation stratum has enough rows but only 3
buildings, below the 5-building minimum. Rows still do not substitute for
building diversity, so the pooled fallback remains a stop point.

The pooled sufficient-observation median contrast is large but not powered for
BDG2-only: BDG2-only median `0.9911189331269352`, GEPIII-overlap median
`0.007638401990504516`, median ratio `129.75474901151046`. The attached OOD
evidence flags `ood_signal=true`, including meter-reading median ratio `0.0`
and model-feature missing-rate delta `0.1427258502998009`; because the BDG2-only
side is not powered, this remains diagnostic context rather than an accepted
transfer conclusion.

The Step 4c Fox cleaned M3.2 LightGBM control anchor matches the earlier pilot
anchor shape:

+ BDG2-only median score: `0.15755569665583008`
+ GEPIII-overlap median score: `0.007009272301667491`

## Decision

Do not run or accept full chilledwater transfer yet. Do not run or accept Step
4b TabPFN-vs-GBDT BDG2 comparison yet.

Next work should either:

+ choose a pilot site/slice with enough BDG2-only sufficient-observation
  chilledwater buildings, or
+ report that the current chilledwater BDG2-only pilot is underpowered and
  redesign the Phase E transfer evidence around a different meter/site/sampling
  frame.

Until that happens, Phase E Step 4 is a corrected pilot-gate stop, not a full
transfer result. The pooled fallback does not change that decision; it confirms
the chilledwater BDG2-only sufficient-observation evidence is still
underpowered after cross-site pooling.

## Validation Notes

The runner command exited `0` and saved strict JSON. Windows emitted a
non-fatal `cp950` reader-thread warning after the successful write, plus sklearn
feature-name warnings during scoring.

The Step 4c pooled fallback command also exited `0` and saved strict JSON.
Windows again emitted the non-fatal `cp950` reader-thread warning after the
successful write, plus sklearn feature-name warnings during scoring.
