# M5 building-count curve and tree early stopping: implementation ready

**Date**: 2026-08-04
**State**: implementation and bounded synthetic validation complete; formal run not launched

## Delivered

+ A deterministic building-source ladder for K=10/20/50/100 with strict nested
  prefixes and representative/site/meter/anomaly-balanced profiles.
+ A hard even-building-only training contract.  Profiles, TabPFN contexts, tree
  fit and tree early-stop data reject odd building IDs.  Odd buildings remain
  the canonical final holdout.
+ Permanent 80/20 building roles inside every K: four fit buildings for each
  one external early-stop building.  Fit and ES sets remain nested as K grows.
+ Explicit external-validation early stopping for LightGBM, XGBoost, CatBoost
  and HistGradientBoosting, including history, best iteration, ceiling-hit and
  ES ROC-AUC/PR-AUC provenance.
+ A TabPFN building cell that uses all K available even buildings as context and
  records why traditional early stopping is not applicable.  It performs no
  dataset-specific checkpoint weight updates.
+ Checkpointed tree/TabPFN predictions with row identities, labels, building,
  site and meter arrays; aggregation emits overall/per-meter/per-site ROC-AUC,
  PR-AUC and reconstructable ROC/PR curve points.
+ Plan/validation/formal guards.  Plan mode cannot load data or models;
  validation requires deterministic row/iteration caps; formal mode rejects
  caps and requires a clean worktree.

## Validation

Fourteen new unittests passed.  They cover deterministic and seed-sensitive
ordering, strict nesting, even-only rejection, role partitions, class gates,
all four early-stopping adapters, detailed reporting, curve generation,
canonical identity failures and execution-mode guards.  New Python files pass
ruff.

The full 20.2M-row ladder and model cells were deliberately not launched.
Repository policy requires a clean committed implementation plus separate
operator authorization for formal scientific execution.

## Intended execution order after authorization

1. Generate and review the representative ladder with
   `prepare_m5_building_curve.py`.
2. Review per-K row/anomaly/site/meter census and confirm GPU context feasibility.
3. Run bounded non-scientific tree and fake-TabPFN cells in an isolated output
   root; validate resume and prediction identity.
4. Commit the reviewed implementation.
5. Explicitly launch formal tree and TabPFN cells per K/feature line.
6. Aggregate both families together with `report_m5_building_curve.py`, then
   render overall, meter and site plots with `plot_m5_building_curve.py`.
