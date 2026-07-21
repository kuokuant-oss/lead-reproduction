# Handoff: M3 50/50 Report Figures Complete

**Date**: 2026-07-16
**Issue**: [#59](https://github.com/kuokuant-oss/lead-reproduction/issues/59)
**Baseline**: 50/50 `building_id % 2`, offline 137-feature `timestamp_merge`, seed 42

## Outcome

The M3 report now contains thirteen standalone figures, organized into five
report sections and generated from a separate observation-only runner:

1. Tree Ensemble confusion matrix at threshold `0.5`.
2. Real validation-series Difference/Ratio illustration.
3. Four separate ROC/PR comparisons for the tree models and 17 vs 137 features.
4. Six separate model, ensemble, and consensus permutation-importance charts.
5. End-to-end M3 workflow.

The observer reproduced the accepted 50/50 baseline exactly. It did not modify
`src/lead`, existing M3 runners, split rules, seeds, feature definitions,
preprocessing, model parameters, execution order, ensemble behavior, or
threshold.

## Numeric Evidence

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| LightGBM | 0.991045274 | 0.923947484 |
| XGBoost | 0.989169641 | 0.927683969 |
| CatBoost | 0.988282317 | 0.925605463 |
| HistGBT | 0.991459981 | 0.926224188 |
| Tree Ensemble | 0.991806391 | 0.930290630 |

The ensemble confusion matrix is `TN=9,278,088`, `FP=221,670`, `FN=32,982`,
`TP=604,415`. M3.1 LightGBM on the same 50/50 split scored ROC-AUC `0.965022231`
and PR-AUC `0.823508940`; adding the canonical value-change feature set raised
the 137-feature LightGBM scores to `0.991045274` and `0.923947484`.

## Feature-Utility Check

Single-column permutation importance screened 44 zero, negative, or
repeat-indistinguishable candidates. Screening was not treated as removal
proof. Correlated groups were checked, and the first three targeted groups were
constantized and retrained through the same four-model ensemble.

All three groups were classified `harmful_to_remove` because at least one of
ROC-AUC, PR-AUC, or Recall@0.5 exceeded its allowed degradation. The canonical
137-feature set remains unchanged.

## Artifacts

| Path | Purpose |
|---|---|
| `scripts/run_m3_figure_observations.py` | Separate observation-only 50/50 runner |
| `scripts/plot_m3_figures.py` | Deterministic five-figure renderer |
| `docs/reports/assets/m3/*.png` | Five report figures |
| `docs/metrics/m3-figures.json` | Figure hashes and compact provenance |
| `data/processed/m3_figure_observations_50_50.json` | Local full observation artifact |
| `data/processed/m3_figure_predictions_50_50.npz` | Local full validation predictions |
| `tests/test_m3_figure_pipeline.py` | Frozen contract and rendering tests |
| `docs/reference/plot-style-rules.md` | Repository plotting rules |

The large observation and prediction artifacts remain local/ignored. The
tracked manifest contains hashes for traceability without committing large
numeric outputs.

## Validation

- Frozen split: 725 train buildings, 724 validation buildings, zero overlap.
- Rows: 10,078,945 train and 10,137,155 validation.
- Feature count: 137, including 120 `timestamp_merge` value-change features.
- Numeric gates: all five accepted 50/50 ROC-AUC values reproduced exactly.
- Visual QA: all five PNGs inspected; axis labels, legends, margins, dates, and
  workflow nodes are readable and unclipped.
- Existing M3 implementation paths remained unchanged.

## Follow-up Contract

Do not remove any screened feature based only on single-column permutation
importance. Any future pruning proposal must remain isolated from the frozen
137-feature baseline and pass grouped, retrained ROC-AUC, PR-AUC, and recall
gates on the same split.
