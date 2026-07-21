# M6 Site Transfer — manifest complete，正式模型待執行

Written: 2026-07-17
Branch: `main`
HEAD at handoff: `955f65e Clarify M5 combined timing semantics`
Execution environment: Windows PowerShell，repo venv at `.venv\Scripts\python.exe`

> **State.** Runs are **stopped**. Local disk has **59 prepared manifests and 0 completed
> M6 model cells**. The PowerShell B2 stdout-capture failure has been fixed and regression-tested.
> Existing manifests are valid; do not delete or reinterpret them as completed experiments.
>
> The authoritative experiment protocol is
> [`docs/reports/m6-site-transfer.md`](../reports/m6-site-transfer.md). This handoff only records
> execution state, order, commands, and traps. Trust JSON `status` fields over prose.

---

## Prompt for Claude Code — paste this

```text
Read docs/handoffs/2026-07-17-m6-site-transfer-rerun-ready.md in full, including
"Order of work" and "Known traps". Work from the repository root.

First verify the local state and tests. Then execute M6 one stage at a time in this order:
B2 manifest resume, A2, A3, A5, B1, B2, Aggregate. Do not run A4 until the A3/A5 evidence
has been reviewed and the stage gate is explicitly reopened.

Use scripts/run_m6_site_transfer_suite.ps1. It prepares and verifies each manifest before a
formal run. Do not modify src/lead, existing M3 runners, frozen split/feature/downsampling/model
logic, or existing M3 artifacts. Do not treat manifest_prepared as completed. Do not calibrate
fixed recall from test labels. Source-calibrated cells are separate additive variants.

After each stage, inspect exit code and completed JSON status before continuing. A model score
being poor is not a failure. Stop only for a command/runtime/artifact error, preserve the error,
and diagnose it without deleting valid completed outputs.

When all requested stages finish, run Aggregate and report: completed/failed cell counts,
artifact paths, elapsed times, pooled and macro-site PR-AUC, worst sites, oracle gaps, and whether
source-calibrated recall-0.90 data are present. Do not update report conclusions before numeric
artifacts exist.
```

---

## Current disk state

At handoff creation, `data/processed/m6_site_transfer_*.json` contains:

| Family | Prepared manifests | Completed model cells | Notes |
|---|---:|---:|---|
| A2 | 2 | 0 | canonical + source-calibrated |
| A3 | 8 | 0 | four folds，canonical + source-calibrated |
| A5 | 16 | 0 | one canonical in-site oracle manifest per site |
| B1 | 30 | 0 | 2 directions × 5 budgets × 3 seeds |
| B2 probe | 3 | 0 | `a0/a1/a2`, `pos1`, seed 42；support discovery only |
| A4 | 0 | 0 | deliberately not prepared；LOSO remains behind stage gate |
| **Total** | **59** | **0** | no full-data model has run |

The three B2 `pos1` manifests are valid probes. Their only purpose is to expose
`b2_available_source_anomalies`; they are not B2 experiment cells and should not be aggregated.

Re-check state from disk:

```powershell
Set-Location C:\Users\tonykuo\projects\lead-reproduction

$rows = Get-ChildItem data\processed -File -Filter "m6_site_transfer_*.json" |
  Sort-Object Name |
  ForEach-Object {
    $j = Get-Content -Raw -Encoding UTF8 -LiteralPath $_.FullName | ConvertFrom-Json
    [pscustomobject]@{
      Name = $_.Name
      Status = $j.status
      Cell = $j.cell
      Variant = $j.variant
    }
  }

$rows | Group-Object Status | Select-Object Name, Count
$rows | Format-Table -AutoSize
```

Expected before formal execution: `manifest_prepared = 59`, `completed = 0`.

---

## What is implemented

| Path | Contract |
|---|---|
| [`scripts/m6_site_transfer_protocol.py`](../../scripts/m6_site_transfer_protocol.py) | A2/A3/A4/A5 masks；B1 source-site-stratified nested meters；B2 matched unique anomalies；source-building calibration |
| [`scripts/run_m6_site_transfer.py`](../../scripts/run_m6_site_transfer.py) | Manifest-first additive runner for A2/A3/A4/A5/B1/B2 |
| [`scripts/run_m6_site_transfer_suite.ps1`](../../scripts/run_m6_site_transfer_suite.ps1) | PowerShell orchestration，stage ordering，B2 common support discovery，A5 pairing，aggregation |
| [`scripts/compare_m6_site_oracle.py`](../../scripts/compare_m6_site_oracle.py) | Paired cross-site vs in-site oracle comparison on identical ordered rows |
| [`scripts/aggregate_m6_site_transfer.py`](../../scripts/aggregate_m6_site_transfer.py) | Completed-cell metrics/curves/slices/operating-points/runtime → plot-ready JSON |
| [`tests/test_m6_site_transfer.py`](../../tests/test_m6_site_transfer.py) | Split、sampling、calibration、plot-data、oracle、aggregation contracts |
| [`tests/test_m6_powershell_suite.py`](../../tests/test_m6_powershell_suite.py) | Guards Python stdout from contaminating B2 budget return value |

The runner stores exact test probabilities plus raw index, timestamp, site, building, meter, and
label. JSON also stores pooled curves, per-site/per-meter metrics, macro-site distribution,
score histograms, source/test/calibration support, matrix footprint, and segmented timing.
`--source-calibration` additionally stores source-only calibration probabilities, learned
thresholds, and pooled/per-site fixed-recall-0.90 confusion data. This is enough to redraw the
planned figures without refitting models.

---

## Order of work — do not deviate

### 0. Environment and verification

```powershell
Set-Location C:\Users\tonykuo\projects\lead-reproduction

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src;scripts"

.\.venv\Scripts\ruff.exe check `
  scripts/m6_site_transfer_protocol.py `
  scripts/run_m6_site_transfer.py `
  scripts/compare_m6_site_oracle.py `
  scripts/aggregate_m6_site_transfer.py `
  tests/test_m6_site_transfer.py `
  tests/test_m6_powershell_suite.py

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected at handoff: Ruff passes and `110` unit tests pass.

### 1. Resume B2 manifest preparation only

The prior all-manifest command stopped after writing the three valid `pos1` probes because Python
stdout was captured with the numeric return. The fix pipes native stdout to `Out-Host`, so
`Get-B2CommonPositiveBudget` now returns one `Int64` rather than `Object[]`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage B2 `
  -ManifestOnly
```

This rechecks the three probes, calculates the common unique-anomaly budget, and writes the nine
formal B2 manifests. It does not fit models.

### 2. A2 reverse direction

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage A2 `
  -IncludeCalibration
```

Produces canonical and source-calibrated A2 result JSON/NPZ pairs.

### 3. A3 four-fold grouped-site evaluation

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage A3 `
  -IncludeCalibration
```

Produces four canonical folds and four source-calibrated folds. Every site is a test site once in
the canonical A3 out-of-fold set.

### 4. A5 per-site in-site oracle and paired comparison

A3 canonical predictions must exist first. A5 automatically pairs site `s` with A3 fold
`s % 4` on the exact oracle-test rows.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage A5
```

Produces 16 oracle JSON/NPZ pairs and up to 16 `m6_paired_oracle_site*.json` comparisons.

### 5. B1 source-meter learning curves

This is 30 cells: two directions × budgets `50,100,200,400,all` × seeds `42,123,999`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage B1
```

Each budget preserves all source sites via the frozen stratified nested meter manifest.

### 6. B2 matched anomaly support

This is nine formal cells: `a0/a1/a2` × seeds `42,123,999`, all using the same unique anomaly
budget discovered from disk.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage B2
```

The frozen M3 sampling shape remains `[negs1, pos, negs2, pos]`. If the common unique anomaly
budget is `N_pos`, fit length is `4 * N_pos`; matching is based on unique positive evidence, not
duplicated fit rows.

### 7. Aggregate completed cells for plotting

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage Aggregate
```

Expected output:

```text
data\processed\m6_site_transfer_plot_data.json
```

The aggregate references exact NPZ arrays rather than duplicating them. It includes pooled curves,
site metrics, threshold-0.5 data, source-calibrated recall-0.90 data, B1/B2 points, oracle gaps,
score distributions, timing, and matrix footprint.

### 8. A4 LOSO — only after the A3/A5 stage gate is reopened

Do not include A4 merely because the script supports it. Review A3 dispersion and A5 oracle gaps
first. If A4 is approved:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage A4

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_m6_site_transfer_suite.ps1 `
  -Stage Aggregate
```

---

## Known traps

1. **Prepared is not completed.** A JSON with `status=manifest_prepared` contains no model result.
2. **Do not rerun M3.** M6 is additive. Do not edit `src/lead`, M3 scripts, existing M3 JSON/NPZ,
   split rules, feature order, sampling semantics, scaler, model parameters, or ensemble weights.
3. **B2 `pos1` files are probes.** Keep them; do not aggregate or report them as experiments.
4. **B2 stdout bug is fixed.** `Invoke-Python` must keep `| Out-Host`. Removing it causes logs to
   join `$CommonPositiveBudget` and argparse receives a long non-integer string.
5. **A5 needs A3 canonical predictions.** Without them, oracle runs can finish but paired
   comparisons are skipped with a warning.
6. **Canonical and source-calibrated are different variants.** Do not overwrite or average them
   together. Fixed recall 0.90 is deployable only for `sourcecal` cells.
7. **Never derive recall-0.90 threshold from test labels.** Canonical cells intentionally mark this
   operating point unavailable.
8. **A4 is gated.** `-Stage All` deliberately excludes A4 unless `-IncludeA4` is added. Do not add
   it before reviewing A3/A5.
9. **Poor site performance is evidence, not failure.** Preserve site 11/13 or any other unfavorable
   result; only runtime/schema/fingerprint errors block a cell.
10. **Large local artifacts are ignored.** `data/processed/*` may not appear in normal `git status`.
    Inspect disk and JSON status directly.
11. **Preserve the dirty worktree.** Existing M3 figure/report changes predate this M6 handoff and
    belong to the user. Do not reset, clean, overwrite, or fold them into an unrelated repair.
12. **Do not publish yet.** No GitHub issue was opened at slice start and no commit/push was
    requested. Follow `docs/reference/change-checklist.md`; do not fabricate retroactive issue
    history. Decide the tracking/publish shape explicitly after the run checkpoint.

---

## Validation at handoff

- `ruff check` on the M6 Python/test files: pass.
- `python -m unittest discover -s tests -p "test_*.py"`: `110/110` pass.
- PowerShell AST parse for `run_m6_site_transfer_suite.ps1`: pass.
- B2 minimal regression: before fix returned `Object[] = log + 410394`; after fix returns one
  `System.Int64 = 410394`.
- `git diff --check`: pass; only pre-existing CRLF warnings on unrelated dirty tracked files.
- Full pre-commit is not green-confirmed: the repo-local pre-commit cache previously raised
  `InvalidManifestError`. Repair/re-run that gate only when preparing a commit.
- No M6 full-data model cell was executed by Codex in this slice.

---

## Repository state and close-out boundary

The checkout is on `main` and dirty. At handoff creation, tracked modifications already existed in
`README.md`, `docs/plans/m3-plan.md`, and `docs/reports/m3-report.md`; multiple M3 figure assets,
scripts, tests, and the M6 files are untracked. Preserve all of them.

This M6 slice currently has no issue/commit/push checkpoint. The repo checklist requires issue,
README/plan/ADR review, handoff, and full gates before publication. Because work began before an
issue was opened, apply the checklist backfill policy honestly rather than inventing history.

---

## Suggested skills

- `diagnose`：only if a stage fails, fingerprints drift, argparse receives malformed values, or a
  completed artifact cannot be aggregated. Reproduce with the smallest cell/manifest first.
- `handoff`：use again after a long stage or before changing agents; record exact completed counts
  from disk and the next command.
- Do not invoke architecture/refactor skills during the formal run. The frozen M3 boundary and
  additive M6 runner are deliberate experiment controls.
