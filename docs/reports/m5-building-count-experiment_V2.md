# M5 building-count experiment V2

**Status:** complete

V2 uses constrained site-stratified random source-building ladders and compares TabPFN with the frozen four-model tree ensemble on byte-identical manifest-allocated rows. Tree early stopping is disabled.

## Fixed protocol

- Building seeds: 42, 43, 44, 45, 46.
- K budgets: 10, 20, 50, 100; strict nested prefixes.
- Candidate buildings: even IDs only; odd IDs are the canonical holdout.
- Source sampling: PCG64 site-stratified random sampling without replacement, with whole-ladder meter-feasibility rejection.
- Row policy: fixed row_seed=42 manifest allocation; natural prevalence; no additional 50:50 redraw and no M3 anomaly duplication.
- Features: 137 timestamp-merge features.
- Tree contract: LightGBM 100, XGBoost 100, CatBoost 1000, HistGBT 100; fixed model_seed=42; no validation split and no early stopping.
- TabPFN: n_estimators=8; no task-specific weight-update early stopping.
- Evaluation: identical canonical odd-building natural-prevalence holdout.
- Matched-context gate: passed for all 20 seed/K pairs.

## Sampling records

| building_seed | K | sampling_attempt | attempts_used | constraint_pass | reproducibility_digest |
|---|---|---|---|---|---|
| 42 | 10 | 8 | 9 | True | b7ef075724ac3377788bb6fc1c48aa80c24a8bd4d052d4a68766a7219115524a |
| 42 | 20 | 8 | 9 | True | 61b1c3085303f9f6fc7f132bcd24b23a29dab8e6f006441d3c099e1a465755c4 |
| 42 | 50 | 8 | 9 | True | a32746bf4dc71ca6b7a3df6806052d0a1f70301c41dbd98a44edd5e3da6b4dcd |
| 42 | 100 | 8 | 9 | True | 0c388f30d1d5c8e198ab68e67d67ea3edc2bf66eca541d06444ada532d226657 |
| 43 | 10 | 7 | 8 | True | 452bed9f8bbf9d5ed3e73a5538ef81b282f6f55c5bb2c2e37bc2b0c2f0d685f6 |
| 43 | 20 | 7 | 8 | True | edc3342004951da4b8ea176688d0ef96bc5dc767a73b3dd26b62b836a3280533 |
| 43 | 50 | 7 | 8 | True | 136bd7d7cb3a649f6ee8c95daf08b08a105b6b859c43191c348a5817d0aed43a |
| 43 | 100 | 7 | 8 | True | f35c0144fe61b153beaf947dbad1163cf7cdc007a31cf2746705002fefdf3728 |
| 44 | 10 | 7 | 8 | True | b7addce65d0a717e30edd8e0211766907853d6ee0f4bd51f292ee4e3c85e7e7d |
| 44 | 20 | 7 | 8 | True | 45e543d9060ada3f6fa4dadaefefb34ff4318c711ad013c651d83205f183a02c |
| 44 | 50 | 7 | 8 | True | 97f4d28d2165eb4b0e80f4803152f6cfa762f05ef58066546bf84b9129d24186 |
| 44 | 100 | 7 | 8 | True | ff0158882216b8fd379f6b420690c1f0142c65d3a4ef5bee88c45d01111401fa |
| 45 | 10 | 7 | 8 | True | 4bb2103ece4a87c0e661711d8c981fd15537bfd0df8ca8428840cd128dcd30cb |
| 45 | 20 | 7 | 8 | True | c219835b15a4310fa883debfc4714528d92192ae65734f438910655a4fa1881b |
| 45 | 50 | 7 | 8 | True | af2992d71aa6f0057f2d7bf813689158a56f70467a30592fb4e29ff8002fa427 |
| 45 | 100 | 7 | 8 | True | 545cc2c5e722e12da72092a7bd0d85f7c84c9ec833c2560a009aaf1f0d1c30ff |
| 46 | 10 | 22 | 23 | True | ead42107d41ec80563dbbe0782a093e128fdcf89e2f7e677fa49d4bf74608814 |
| 46 | 20 | 22 | 23 | True | f5573ea471fb621577ece6bd8d4b21de4a8f379547b42b420c0af83a10f7c5ab |
| 46 | 50 | 22 | 23 | True | 3f16604a96a428723a2a811b0ed1c91c99679fc86cdf80c46810f4884dea11ef |
| 46 | 100 | 22 | 23 | True | 35c694a38f83e37dce51b48dfab07271b20b971273fa3f0b7bdfaaa60293e0d0 |

Full building IDs, site counts, per-meter source-building counts and digests: data/processed/m5_building_curve/sensitivity/building_candidate_pilot/sampling_prefix_audit.csv.

## Cross-seed overall results

| model | building_budget | n_building_seeds | pr_auc_mean | pr_auc_std | pr_auc_min | pr_auc_max | roc_auc_mean | roc_auc_std |
|---|---|---|---|---|---|---|---|---|
| ensemble | 10 | 5 | 0.759397 | 0.065109 | 0.673867 | 0.828712 | 0.971859 | 0.006836 |
| ensemble | 20 | 5 | 0.710646 | 0.099976 | 0.564427 | 0.800865 | 0.968943 | 0.013505 |
| ensemble | 50 | 5 | 0.798740 | 0.031949 | 0.760378 | 0.841498 | 0.977537 | 0.005389 |
| ensemble | 100 | 5 | 0.863657 | 0.024056 | 0.833187 | 0.899520 | 0.984370 | 0.002221 |
| tabpfn | 10 | 5 | 0.722049 | 0.031490 | 0.687507 | 0.763260 | 0.968159 | 0.005138 |
| tabpfn | 20 | 5 | 0.705693 | 0.054063 | 0.654403 | 0.777562 | 0.968033 | 0.006751 |
| tabpfn | 50 | 5 | 0.755926 | 0.068302 | 0.677478 | 0.831862 | 0.976659 | 0.003104 |
| tabpfn | 100 | 5 | 0.839677 | 0.028518 | 0.796319 | 0.872776 | 0.980453 | 0.003408 |

## Per-seed overall results

| model | K | building_seed | pr_auc | roc_auc |
|---|---|---|---|---|
| ensemble | 10 | 42 | 0.828712 | 0.974761 |
| ensemble | 10 | 43 | 0.673867 | 0.967551 |
| ensemble | 10 | 44 | 0.801459 | 0.978415 |
| ensemble | 10 | 45 | 0.783444 | 0.976483 |
| ensemble | 10 | 46 | 0.709506 | 0.962084 |
| ensemble | 20 | 42 | 0.781904 | 0.974136 |
| ensemble | 20 | 43 | 0.651699 | 0.968799 |
| ensemble | 20 | 44 | 0.754335 | 0.975951 |
| ensemble | 20 | 45 | 0.800865 | 0.979955 |
| ensemble | 20 | 46 | 0.564427 | 0.945874 |
| ensemble | 50 | 42 | 0.841498 | 0.983131 |
| ensemble | 50 | 43 | 0.802587 | 0.979386 |
| ensemble | 50 | 44 | 0.775459 | 0.973420 |
| ensemble | 50 | 45 | 0.813778 | 0.981305 |
| ensemble | 50 | 46 | 0.760378 | 0.970443 |
| ensemble | 100 | 42 | 0.899520 | 0.986899 |
| ensemble | 100 | 43 | 0.856368 | 0.982424 |
| ensemble | 100 | 44 | 0.859844 | 0.984624 |
| ensemble | 100 | 45 | 0.869366 | 0.986091 |
| ensemble | 100 | 46 | 0.833187 | 0.981814 |
| tabpfn | 10 | 42 | 0.731425 | 0.963171 |
| tabpfn | 10 | 43 | 0.693309 | 0.966330 |
| tabpfn | 10 | 44 | 0.763260 | 0.975056 |
| tabpfn | 10 | 45 | 0.734746 | 0.971977 |
| tabpfn | 10 | 46 | 0.687507 | 0.964259 |
| tabpfn | 20 | 42 | 0.743220 | 0.970135 |
| tabpfn | 20 | 43 | 0.654403 | 0.956288 |
| tabpfn | 20 | 44 | 0.696626 | 0.972342 |
| tabpfn | 20 | 45 | 0.777562 | 0.972589 |
| tabpfn | 20 | 46 | 0.656653 | 0.968812 |
| tabpfn | 50 | 42 | 0.831862 | 0.981946 |
| tabpfn | 50 | 43 | 0.689758 | 0.975893 |
| tabpfn | 50 | 44 | 0.677478 | 0.974382 |
| tabpfn | 50 | 45 | 0.791562 | 0.974468 |
| tabpfn | 50 | 46 | 0.788973 | 0.976605 |
| tabpfn | 100 | 42 | 0.872776 | 0.984554 |
| tabpfn | 100 | 43 | 0.830886 | 0.975376 |
| tabpfn | 100 | 44 | 0.852266 | 0.982142 |
| tabpfn | 100 | 45 | 0.846139 | 0.980713 |
| tabpfn | 100 | 46 | 0.796319 | 0.979479 |

## Detailed outputs

- Raw metrics: data/processed/m5_building_curve/v2/building_seed_sweep_42-43-44-45-46/aggregate/metrics.csv.
- Cross-seed summary: data/processed/m5_building_curve/v2/building_seed_sweep_42-43-44-45-46/aggregate/building_seed_summary.csv.
- ROC/PR curve points: data/processed/m5_building_curve/v2/building_seed_sweep_42-43-44-45-46/aggregate/curves.csv.
- Matched-context gate: data/processed/m5_building_curve/v2/building_seed_sweep_42-43-44-45-46/matched_context_gate.json.
- Sampling composition diagnostics: data/processed/m5_building_curve/sensitivity/building_candidate_pilot/composition_audit.csv.

Per-meter and per-site rows remain in metrics.csv; raw per-seed results are preserved and are not replaced by the mean.
