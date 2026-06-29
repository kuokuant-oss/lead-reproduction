# Handoff: Phase E Step 4 gate correction

**Date**: 2026-06-29
**Scope**: Correct the Step 4 pilot gate after the previous overrun executed
pilot/full/4b before validating unknown #26.

## What Changed

+ `scripts/phaseE_transfer.py`
  + Added building/meter completeness classification.
  + `sufficient_obs` means `(building_id, meter)` missing rate for
    `meter_reading` is at most 50%.
  + `high_missing` means the missing rate is above 50%.
  + `stratified_score_report` now reports the cross product of
    `is_gepiii_overlap` and completeness:
    `gepiii_overlap__sufficient_obs`, `gepiii_overlap__high_missing`,
    `bdg2_only__sufficient_obs`, and `bdg2_only__high_missing`.
  + Added strict powered-stratum helper with defaults of at least 5 buildings and
    at least 17,544 rows.
+ `scripts/run_phaseE_step4a_bdg2_transfer.py`
  + Pilot gate no longer passes on plumbing alone.
  + Gate returns `verdict`, `allowed_next_step`, powered site lists, and
    sufficient-observation comparisons.
  + Pilot artifact includes a Fox cleaned M3.2 LightGBM control anchor.
+ `scripts/run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py`
  + Rank agreement now also reports by overlap/completeness stratum.
  + The Step 4b runner remains present but should not be accepted until the
    corrected pilot gate allows the next stage.
+ `scripts/run_phaseE_step4c_pooled_powered_fallback.py`
  + Added a raw chilledwater pooled fallback that scores sites independently and
    aggregates only the unlabeled score/OOD stratum evidence.
  + The pooled gate maps an unpowered pooled BDG2-only sufficient-observation
    stratum to `underpowered_even_pooled` and `stop_and_report`.
+ `tests/test_phaseE_step4_transfer.py`
  + Added regression coverage proving plumbing-only BDG2 rows fail the gate.
  + Added completeness and rank-by-stratum tests.
  + Added pooled fallback tests for the four cross strata, underpowered pooled
    verdict, and powered OOD stop without full-transfer permission.
+ `docs/reports/phaseE-step4-bdg2-transfer.md`, README, and
  `docs/plans/m5-plan.md` now describe the corrected gate stop.

## Accepted Pilot Evidence

Accepted artifact:

+ `.scratch/phaseE-step4a-bdg2-transfer-pilot.json`

Command rerun:

```powershell
.\.venv\Scripts\python.exe scripts\run_phaseE_step4a_bdg2_transfer.py --mode pilot --include-cleaned --out .scratch\phaseE-step4a-bdg2-transfer-pilot.json
```

Gate result:

+ `status`: `failed`
+ `verdict`: `underpowered`
+ `allowed_next_step`: `stop_and_report`
+ failure: `pilot has no powered bdg2_only__sufficient_obs stratum`

Raw pilot details:

+ Fox: BDG2-only sufficient-observation stratum has 17,544 rows but only 1
  building; median score `0.1301750424683641`. Fox GEPIII-overlap sufficient
  stratum has 1,736,856 rows and 99 buildings; median score
  `0.004250435627313853`.
+ Swan: BDG2-only sufficient-observation stratum has 0 rows and 0 buildings.
  Swan BDG2-only high-missing stratum has 350,880 rows and 20 buildings; median
  score `0.09571689850728261`.

Control anchor:

+ Fox cleaned M3.2 LightGBM control anchor BDG2-only median:
  `0.15755569665583008`.
+ Fox cleaned M3.2 LightGBM control anchor GEPIII-overlap median:
  `0.007009272301667491`.

## Pooled Raw Fallback Evidence

Accepted diagnostic artifact:

+ `.scratch/phaseE-step4c-pooled-powered-fallback.json`

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_phaseE_step4c_pooled_powered_fallback.py --out .scratch\phaseE-step4c-pooled-powered-fallback.json
```

Pooled gate result:

+ `status`: `failed`
+ `verdict`: `underpowered_even_pooled`
+ `allowed_next_step`: `stop_and_report`
+ failure: `pooled chilledwater has no powered bdg2_only__sufficient_obs stratum`

Pooled raw strata:

+ GEPIII-overlap sufficient-observation: 516 buildings, 9,052,704 rows, median
  score `0.007638401990504516`.
+ GEPIII-overlap high-missing: 13 buildings, 228,072 rows, median score
  `0.2211491656241341`.
+ BDG2-only sufficient-observation: 3 buildings, 52,632 rows, median score
  `0.9911189331269352`.
+ BDG2-only high-missing: 23 buildings, 403,512 rows, median score
  `0.10770437155813975`.

The pooled BDG2-only sufficient-observation stratum has enough rows but only 3
buildings, so it still fails the 5-building minimum. The sufficient-observation
median ratio is `129.75474901151046`, with `ood_signal=true`, but the BDG2-only
side is not powered; treat that contrast as diagnostic context only.

Step 4c control anchor:

+ Fox cleaned M3.2 LightGBM control anchor BDG2-only median:
  `0.15755569665583008`.
+ Fox cleaned M3.2 LightGBM control anchor GEPIII-overlap median:
  `0.007009272301667491`.

## Quarantined Artifacts

These files exist from the previous overrun and should not be treated as passed
evidence:

+ `.scratch/phaseE-step4a-bdg2-transfer-full.json`
+ `.scratch/phaseE-step4b-tabpfn-vs-gbdt-bdg2.json`

They were produced before the corrected gate and are retained only as local
diagnostic artifacts. Do not cite them as full-transfer success, readiness, or a
green Step 4b result. Both JSON files now carry `quarantined=true`,
`quarantine_reason`, and `metric_contract.headline_metric=false`.

## Next Step

Stop after the corrected pilot and pooled fallback. Do not run full or Step 4b
again until a new pilot or scope has a powered `bdg2_only__sufficient_obs`
stratum, or until the project explicitly chooses a held-out-BDG2-site scope
under ADR 0019. The current chilledwater pooled fallback is
`underpowered_even_pooled`, not permission to proceed.

## Validation

+ `.\.venv\Scripts\python.exe -m ruff check scripts\phaseE_transfer.py scripts\run_phaseE_step4a_bdg2_transfer.py scripts\run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py tests\test_phaseE_step4_transfer.py`
+ `.\.venv\Scripts\python.exe -m unittest tests.test_phaseE_step4_transfer -v`
+ `.\.venv\Scripts\python.exe scripts\run_phaseE_step4c_pooled_powered_fallback.py --out .scratch\phaseE-step4c-pooled-powered-fallback.json`

Both passed before the pilot rerun. The pilot command exited `0` and saved
strict JSON; Windows emitted the same non-fatal `cp950` reader-thread warning
after the successful write.

The Step 4c pooled fallback command exited `0` and saved strict JSON. Windows
again emitted the same non-fatal `cp950` reader-thread warning after the
successful write, plus sklearn feature-name warnings during scoring.
