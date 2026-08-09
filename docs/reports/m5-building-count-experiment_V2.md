# M5 building-count experiment V2

**Status:** pipeline and recording schema prepared; formal models not started.

V2 studies the effect of source-building count under constrained site-stratified random building draws. It uses the no-early-stopping matched-context tree contract and requires trees and TabPFN to consume byte-identical manifest-allocated context rows.

Protocol decisions and run gates are fixed in [m5-building-count-experiment-v2-plan.md](../plans/m5-building-count-experiment-v2-plan.md). This file is separate from the published V1 report and will only be regenerated from a complete V2 sweep.

## Fixed protocol

- Building seeds: 42, 43, 44, 45, 46.
- K budgets: 10, 20, 50, 100, as strict prefixes of one accepted ladder per seed.
- Candidate pool: 725 even-ID training buildings; odd-ID buildings are excluded from selection and retained as the canonical holdout.
- Building draw: seeded PCG64 site-stratified random sampling without replacement.
- Feasibility: at K=10 every evaluation meter appears in at least 2 distinct source buildings; every later K transition adds at least 1 source building for every meter.
- Rejection: whole-ladder deterministic redraw, at most 10,000 attempts; no greedy correction or silent relaxation.
- Selection does not use anomaly labels, anomaly rate, building size, meter row share or post-sampling diagnostics.
- Row allocation: `row_seed=42`, average cap 500 rows per building, natural prevalence, no post-manifest redraw.
- Expected context rows: 5,000 / 10,000 / 25,000 / 50,000 for K=10 / 20 / 50 / 100.
- Features: 137.
- Model seed: 42.
- Tree ensemble: LightGBM 100, XGBoost 100, CatBoost 1,000, HistGradientBoosting 100; no early stopping and no validation split.
- TabPFN: `n_estimators=8`; early stopping is not applicable.
- Evaluation: one identical canonical odd-building natural-prevalence holdout for every seed/K/model cell.

The manifest role column is diagnostic metadata only. V2 fits trees on all K selected buildings and does not create an early-stop subset.

## Sampling audit already completed

All 20 seed/K prefixes passed the meter constraints and all five draws are distinct at every K.

| building_seed | accepted attempt (zero-based) | attempts used | K10 digest | K20 digest | K50 digest | K100 digest |
|---:|---:|---:|---|---|---|---|
| 42 | 8 | 9 | b7ef075724ac… | 61b1c3085303… | a32746bf4dc7… | 0c388f30d1d5… |
| 43 | 7 | 8 | 452bed9f8bbf… | edc334200495… | 136bd7d7cb3a… | f35c0144fe61… |
| 44 | 7 | 8 | b7addce65d0a… | 45e543d9060a… | 97f4d28d2165… | ff0158882216… |
| 45 | 7 | 8 | 4bb2103ece4a… | c219835b15a4… | af2992d71aa6… | 545cc2c5e722… |
| 46 | 22 | 23 | ead42107d41e… | f5573ea471fb… | 3f16604a96a4… | 35c694a38f83… |

Machine-readable records:

- Full building IDs, site counts, per-meter source-building counts, pass/fail and full digests: `data/processed/m5_building_curve/sensitivity/building_candidate_pilot/sampling_prefix_audit.csv`.
- One ladder CSV and JSON manifest per seed: `data/processed/m5_building_curve/sensitivity/building_candidate_pilot/building_ladder_seed{seed}.{csv,json}`.
- Sampling summary: `data/processed/m5_building_curve/sensitivity/building_candidate_pilot/summary.json`.
- Post-selection composition diagnostics: `data/processed/m5_building_curve/sensitivity/building_candidate_pilot/composition_audit.csv`.

## Run census

| item | expected | current |
|---|---:|---:|
| building seeds | 5 | 5 audited |
| K budgets per seed | 4 | 4 audited |
| model families | 2 | pipeline prepared |
| total model cells | 40 | 0 formally run |
| matched-context seed/K gates | 20 | pending model artifacts |
| aggregate report | 1 | pending complete sweep |

No validation or formal metric is claimed in this document yet.

## Overall performance across building draws

This table is populated only after all five raw seed results exist for every model/K.

| model | K | n_building_seeds | PR-AUC mean | PR-AUC SD | PR-AUC min | PR-AUC max | ROC-AUC mean | ROC-AUC SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TabPFN | 10 | pending | pending | pending | pending | pending | pending | pending |
| TabPFN | 20 | pending | pending | pending | pending | pending | pending | pending |
| TabPFN | 50 | pending | pending | pending | pending | pending | pending | pending |
| TabPFN | 100 | pending | pending | pending | pending | pending | pending | pending |
| ensemble | 10 | pending | pending | pending | pending | pending | pending | pending |
| ensemble | 20 | pending | pending | pending | pending | pending | pending | pending |
| ensemble | 50 | pending | pending | pending | pending | pending | pending | pending |
| ensemble | 100 | pending | pending | pending | pending | pending | pending | pending |

The final report will also retain one raw overall result per building seed and the full per-meter/per-site rows. Means never replace raw results.

## Required matched-context record

Before aggregation, each of the 20 seed/K pairs must record and pass:

| field | tree vs TabPFN requirement |
|---|---|
| context row count | equal |
| context row SHA-256 | equal |
| row policy / row seed | equal |
| model seed | equal |
| holdout row SHA-256 | equal |
| holdout raw indices | array-identical |
| holdout labels | array-identical |
| holdout building/site/meter | array-identical |

The completed gate is written to:

```text
data/processed/m5_building_curve/v2/
  building_seed_sweep_42-43-44-45-46/
  matched_context_gate.json
```

## Result artifact schema

Raw and aggregate V2 artifacts are isolated under:

```text
data/processed/m5_building_curve/v2/
  building_seed_sweep_42-43-44-45-46/
    model_runs/building_seed{seed}/
      tree_no_es_k{K}_f137/
      tabpfn_k{K}_f137/
    aggregate/
      metrics.csv
      curves.csv
      building_seed_summary.csv
```

Each cell stores provenance, fit/checkpoint state, prediction chunks, `predictions.npz`, `cell.json`, heartbeat and `COMPLETE.json`. The artifact identity contains `building_seed`, so independent draws cannot overwrite one another.

## Commands

Plan only; this does not load data or start a model:

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py --mode plan
```

Small non-scientific pipeline validation:

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py \
  --mode validation \
  --validation-context-rows 200 \
  --validation-holdout-rows 200
```

Formal 40-cell sweep:

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py --mode formal
```

The orchestrator resumes compatible partial cells, runs the matched-context gate, generates raw and cross-seed aggregates, and then replaces this prepared template with an artifact-derived completed report through [update_m5_building_count_v2_report.py](../../scripts/update_m5_building_count_v2_report.py).

## Authorized overnight execution

The detached multi-day queue uses this pair order:

```text
42/K10, 42/K20, 42/K50, 42/K100,
43/K10, 44/K10, 45/K10, 46/K10,
43/K20, 44/K20, 45/K20, 46/K20,
43/K50, 44/K50, 45/K50, 46/K50,
43/K100, 44/K100, 45/K100, 46/K100
```

The operational entry point is
[run_m5_building_count_v2_overnight.py](../../scripts/run_m5_building_count_v2_overnight.py),
guarded by
[ensure_m5_building_count_v2_overnight.sh](../../scripts/ensure_m5_building_count_v2_overnight.sh).
It uses resume-compatible cell checkpoints, three attempts per model unit,
bounded GPU waits, bounded Git-push retries and failed-stage markers. Exhausted
units are skipped so later pairs can continue. After each pair,
[update_m5_building_count_v2_progress.py](../../scripts/update_m5_building_count_v2_progress.py)
updates this tracked report and the supervisor commits/pushes it. The full
artifact-derived report is generated only after all 40 cells pass.

<!-- BEGIN M5 BUILDING COUNT V2 RUN PROGRESS -->

## Overnight formal-run progress

- Last update: 2026-08-09T08:06:08+08:00.
- Last checkpointed pair: building_seed43_k10.
- Completed seed/K pairs: 5/20.
- Failed/skipped seed/K pairs: 0.
- A pair is complete only after both frozen no-ES trees and TabPFN finish.
- Raw model artifacts remain under the ignored V2 data root; this tracked table is committed and pushed after each completed pair.

| order | building_seed | K | status | tree | TabPFN | ensemble PR-AUC | ensemble ROC-AUC | TabPFN PR-AUC | TabPFN ROC-AUC |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 1 | 42 | 10 | complete | yes | yes | 0.828712 | 0.974761 | 0.731425 | 0.963171 |
| 2 | 42 | 20 | complete | yes | yes | 0.781904 | 0.974136 | 0.743220 | 0.970135 |
| 3 | 42 | 50 | complete | yes | yes | 0.841498 | 0.983131 | 0.831862 | 0.981946 |
| 4 | 42 | 100 | complete | yes | yes | 0.899520 | 0.986899 | 0.872776 | 0.984554 |
| 5 | 43 | 10 | complete | yes | yes | 0.673867 | 0.967551 | 0.693309 | 0.966330 |
| 6 | 44 | 10 | pending | no | no |  |  |  |  |
| 7 | 45 | 10 | pending | no | no |  |  |  |  |
| 8 | 46 | 10 | pending | no | no |  |  |  |  |
| 9 | 43 | 20 | pending | no | no |  |  |  |  |
| 10 | 44 | 20 | pending | no | no |  |  |  |  |
| 11 | 45 | 20 | pending | no | no |  |  |  |  |
| 12 | 46 | 20 | pending | no | no |  |  |  |  |
| 13 | 43 | 50 | pending | no | no |  |  |  |  |
| 14 | 44 | 50 | pending | no | no |  |  |  |  |
| 15 | 45 | 50 | pending | no | no |  |  |  |  |
| 16 | 46 | 50 | pending | no | no |  |  |  |  |
| 17 | 43 | 100 | pending | no | no |  |  |  |  |
| 18 | 44 | 100 | pending | no | no |  |  |  |  |
| 19 | 45 | 100 | pending | no | no |  |  |  |  |
| 20 | 46 | 100 | pending | no | no |  |  |  |  |

<!-- END M5 BUILDING COUNT V2 RUN PROGRESS -->
