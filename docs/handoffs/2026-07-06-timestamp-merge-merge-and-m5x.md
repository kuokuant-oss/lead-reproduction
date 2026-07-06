# 2026-07-06 Timestamp-Merge Merge into main and M5.x Split

## Reason

`debug_timestamp_merge` (9 commits, diverged from `b7fd55b`) carried the
`timestamp_merge` re-baseline plus a deep-comparison redo. It had to land on
`main`, which had independently gained the row-offset M5.1 deep comparison via
PR #56. This is a re-baselining, not an accuracy gain: ADR 0011 measures the two
regimes at `+0.00047` AUC apart, inside the `+/-0.0005` noise floor.

## Merge

+ Merge commit `6c428fd` (`--no-ff`, two parents `0b80417` + `efd695e`):
  "rebaseline value-change regime to timestamp_merge (within noise floor)".
+ Local `main` was first fast-forwarded to `origin/main` so PR #56 (`00bda2a`,
  `0b80417`) was not dropped.
+ Six add/add + content conflicts, all in the M5.1 deep comparison, resolved to
  the debug-branch versions (they redo the same M5.1 work on the
  `timestamp_merge` pipeline and supersede PR #56's row-offset M5.1):
  + `data/processed/m5_phaseD_deep_comparison.json`
  + `docs/handoffs/m5-phaseD-deep-comparison.md`
  + `docs/reports/m5-1-deep-comparison.md`
  + `scripts/run_m5_phaseD_deep_comparison.py`
  + `docs/reports/m3-report.md` (content)
  + `docs/reports/m5-foundation-vs-gbdt.md` (content)
+ The two content conflicts were reviewed segment-by-segment: PR #56's only
  edits to them were provenance text (Kaggle source link, dataset list,
  `bad_meter_readings.zip`), which the debug versions already carry. No PR #56
  content was overwritten.

## M5.x Split (PR #57)

+ The debug tip `4bbb8c6` (M5.x partition-granularity experiment, ~16.9k lines,
  unrelated to the re-baseline) was excluded from the merge; `main` was merged
  only through `efd695e`.
+ M5.x was cherry-picked onto merged `main` as branch
  `m5x-partition-granularity` (`f78a691`), pushed, and landed via PR #57
  (merge commit `83026c3`). Branch deleted after merge.

## Sanity Gate

`scripts/run_m4_3_timestamp_value_change.py` was run in full on the 20.2M-row M3
frame (16.6 min, `provenance.commit=6c428fd`):

| Regime | val AUC | Delta vs golden |
|---|---:|---:|
| `row_offset` | 0.9920119520500562 | -0.00047115662424412896 |
| `timestamp_merge` | 0.9924831086743003 | 0.0 |

`gate_status=within_noise_floor`; `timestamp_merge` reproduces the golden
`m3_2_lightgbm_80_20_offline_auc` exactly. `building_overlap=0`.

## README Updates

+ `ad14c62`: synced project-structure and tracked-code trees to `git ls-files`
  (new inv/deep-comparison/M5.x scripts; M5.x and report-consistency tests;
  M5.1/M5.x/BDG2 reports); added M5.1 deep-comparison and M5.x run commands to
  the execution section; added the M5.x result to the M5 milestone summary and
  status row.
+ `095fd8d`: added the M5.1 deep-comparison result and linked
  `m5-1-deep-comparison.md` / `m5x-partition-granularity.md` from primary docs.
+ `06c6c9f`: retitled the three M5 reports by their distinguishing focus and
  regrouped the README M5 results under those three foci:
  + M5  -> `TabPFN 與 GBDT 跨部署情境總覽比較`
  + M5.1 -> `TabPFN 調參敏感度與小樣本標註效率深入比較`
  + M5.x -> `per-unit 建模切分粒度比較`
+ `32f95e8`: made the small-sample result concrete in the M5 status row
  (TabPFN leads PR-AUC at support `100`-`2,000`; strong small-sample baseline).

Report file names were left unchanged (only titles and README presentation
changed) so test paths, internal links, and handoff references stay valid.

## Verification

+ Full suite green throughout (`python -m unittest discover -s tests`): 68 tests
  at the merge commit (M5.x excluded), 78 after PR #57 merged M5.x back.
+ `test_report_metric_consistency`, `test_m5_timestamp_merge_regime`,
  `test_readme_freshness` all pass; report AUC display values match
  `tests/golden_metrics.json` (canonical M3.2 is `0.9925`; residual `0.9920`
  occurrences are historical/superseded provenance only).
+ pre-commit hooks (markdownlint, ruff, whitespace/EOF, json) green on every
  commit.

## Scope Notes

+ No git history was rewritten; both merges use merge commits.
+ `src/lead/features.py` default `timestamp_merge` came in from the debug branch
  unchanged; no regime was deleted and no test tolerance was widened.
+ M5 report basenames were not renamed on disk.
