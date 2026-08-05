# Use even-building budgets and external-building tree early stopping

## Status

Accepted.

## Context

The existing M5 matched-context curve varies the number of balanced labelled
rows.  It does not isolate how performance changes as labels come from more
source buildings.  Its tree arm also reuses fixed-iteration M3 models and has
no early stopping, while TabPFN performs in-context inference with pretrained
weights and has no task-specific epoch loop to stop.

A naive row-random validation split would leak the same buildings into tree fit
and model selection.  Borrowing validation buildings outside the stated K would
also give trees a larger labelled-source budget than TabPFN.

## Decision

Add an independent building-count comparison line with these rules:

+ Candidate, profile, balance, TabPFN context, tree fit and tree early-stop data
  come only from `building_id % 2 == 0`.
+ `building_id % 2 == 1` remains the fixed canonical test set and cannot affect
  building ordering, scaling, early stopping or model selection.
+ One deterministic balanced building order defines all K cells as strict
  prefixes.  The primary budgets are 10, 20, 50 and 100 buildings.
+ Every fifth selected building is assigned permanently to the tree external
  early-stop role.  The remaining four are tree fit buildings.  Both role sets
  remain nested as K grows.
+ K is the common labelled-building acquisition budget.  TabPFN may use all K
  buildings as context because it does not consume a validation set for weight
  updates; trees use the same K split between fit and early stopping.  Reports
  must disclose effective context, fit and early-stop buildings and rows.
+ LightGBM, XGBoost, CatBoost and HistGradientBoosting monitor an explicit
  building-disjoint validation set using ROC-AUC.  Best iteration, history,
  ceiling hits and validation PR-AUC are persisted.
 Tree fit rows retain the frozen M3 `[negs1, pos, negs2, pos]` downsampling
  with seeds 10 and 20, while external early-stop rows retain their natural
  distribution.  After iteration selection, final refit applies the same M3
  downsampling to all available training-source rows.  The scaler is fitted
  only on each downsampled training matrix.
+ Existing frozen M3 models and row-context results are historical artifacts and
  are not redefined.

## Consequences

+ Building-source diversity, row count and anomaly prevalence still co-vary
  under the primary all-rows policy, so every cell records all three.
 Available source-row counts and effective downsampled tree fit-row counts are
  both reported; `all_rows` never means bypassing M3 class balancing.
+ Site-, meter- and anomaly-balanced ladders are sensitivity profiles, not
  interchangeable headline samples.
+ TabPFN receives no artificial early stopping.  Any future fine-tuning or
  metric-tuning variant requires a separately named protocol.
+ Formal runs remain explicit operator actions after a clean commit and bounded
  validation; odd-building labels never enter training or early stopping.
