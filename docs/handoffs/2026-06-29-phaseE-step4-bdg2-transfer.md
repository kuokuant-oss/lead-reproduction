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
+ `tests/test_phaseE_step4_transfer.py`
  + Added regression coverage proving plumbing-only BDG2 rows fail the gate.
  + Added completeness and rank-by-stratum tests.
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

Stop after the corrected pilot. Do not run full or Step 4b again until a new
pilot has a powered `bdg2_only__sufficient_obs` stratum, or until the project
explicitly chooses to report the current chilledwater pilot as underpowered and
redesigns the Phase E transfer evidence.

## Validation

+ `.\.venv\Scripts\python.exe -m ruff check scripts\phaseE_transfer.py scripts\run_phaseE_step4a_bdg2_transfer.py scripts\run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py tests\test_phaseE_step4_transfer.py`
+ `.\.venv\Scripts\python.exe -m unittest tests.test_phaseE_step4_transfer -v`

Both passed before the pilot rerun. The pilot command exited `0` and saved
strict JSON; Windows emitted the same non-fatal `cp950` reader-thread warning
after the successful write.
