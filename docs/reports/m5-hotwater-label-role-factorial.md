# M5 hotwater label-role factorial

**Date:** 2026-07-30
**Status:** first fixed-query factorial complete; do not expand
**Scope:** F4/137 features, N=20,000, 50/50 labels, existing 352-row fixed
query, three context-draw seeds (42, 123, 999), fixed model seed 42.

## Design and provenance

For every seed a pooled-reference context supplied the same 20,000 ordered row
slots for four cells: hotwater positive / negative support present or excluded.
An excluded hotwater row was replaced one-for-one with a unique same-label
non-hotwater reserve row. The deterministic builder preserved total N, label
counts, row-slot order, and the non-hotwater conditional meter-allocation rule.

The manifests, ordered digests, replacements, composition audit, and overlap
audit are under `data/processed/m5_hotwater_label_factorial/`. The run used 48
prediction cells: two models × three context seeds × four factorial cells × two
scaler arms. No full-holdout, 5k/100k, retrieval, site-transfer, transfer
matrix, 500k scaling, or representation-ablation job was run.

TabPFN used cell-specific and per-seed frozen pooled-reference scalers. The
tree scaler-invariance pilot did **not** pass: its maximum absolute prediction
difference was 0.163691 versus the 1e-6 tolerance. Consequently both tree
scaler arms were run; no tree result is being treated as scaler-invariant.

## Pre-specified factorial effects

Values are seed means of the cell-specific-scaler contrasts. A main effect is
the average present-minus-excluded contrast; interaction is
`present/present − present/excluded − excluded/present + excluded/excluded`.

| Estimand | TabPFN positive effect | TabPFN negative effect | TabPFN interaction | Direction over 3 seeds |
| --- | ---: | ---: | ---: | --- |
| HW 0–1 within-meter rank gap | +0.1127 | +0.0314 | +0.1669 | positive-support and interaction: 3/3; negative: 2/3 |
| HW 0–1 pairwise AUC | +0.1771 | +0.0382 | +0.2292 | positive-support and interaction: 3/3; negative: 2/3 |
| Steam-positive × HW-negative AUC | −0.1924 | +0.4111 | +0.3158 | all three effects: 3/3 in stated directions |
| Global recall at FPR=0.001 | −0.1373 | +0.4119 | +0.3617 | negative and interaction: 3/3; positive: 3/3 negative |

The contrast pattern distinguishes the two support roles in this fixed-query
setting. Positive hotwater support improves the two local HW 0–1 ranking
readouts, while negative support strongly improves steam-positive versus
hotwater-negative ordering and low-FPR recall. The positive interaction is
consistent with complementarity, but remains a fixed-query intervention result,
not a general mechanism claim.

Trees share the external spillover direction, but differ on local hotwater
readouts: the TabPFN-minus-tree positive-support effect is +0.2049 for HW 0–1
pairwise AUC and +0.1377 for its rank gap. This is the relevant model-response
difference; it is more informative than comparing unpaired pooled scores.

## Clustered uncertainty and scaler result

The report artifacts contain 1,000-draw building- and segment-clustered
bootstrap intervals. For the TabPFN interaction, the steam-positive ×
hotwater-negative AUC interval excludes zero for each seed and both cluster
definitions:

| Seed | Building 95% CI | Segment 95% CI |
| ---: | ---: | ---: |
| 42 | +0.197 to +0.481 | +0.190 to +0.455 |
| 123 | +0.136 to +0.375 | +0.142 to +0.388 |
| 999 | +0.189 to +0.579 | +0.201 to +0.538 |

The original query's post-fit audit changes how these results must be read.
Its HW 0–1 negative cell has only **3 rows / 3 buildings / 3 segments** (versus
16 HW 0–1 positive rows), so its local rank/AUC bootstrap is not a viable
mechanism test. It also has only 176 normal rows: the minimum empirical FPR
resolution is 1/176 = 0.00568, and the linear 0.999 normal quantile produced
one false positive in 47 of 48 cells (two in the remaining cell). Therefore
`recall@FPR=0.001` is descriptive only, not evidence for or against the
mechanism. Pairwise and rank results are retained as screening diagnostics, not
as adequate local confirmation.

Frozen scaling does not erase the external interaction: its cell-specific minus
frozen interaction difference averages −0.0098 for steam-positive ×
hotwater-negative AUC and −0.0076 for recall. It does alter the local rank-gap
interaction (+0.0320 on average), so preprocessing geometry is a modifier of
the local score geometry rather than the explanation for the stable steam
spillover.

## Decision and convergence table

| Candidate claim | Novelty | Supporting evidence | Counterevidence | Missing evidence | Minimum next experiment | Stopping condition |
| --- | --- | --- | --- | --- | --- | --- |
| Label roles have separable effects on heterogeneous support | Direct positive/negative allocation intervention | Three-seed TabPFN external interaction; building and segment CIs exclude 0; frozen scaler retains it | Trees show substantial external response too; local HW cluster CIs are unstable | Natural-prevalence full holdout; larger/fixed-query replication; stable local HW uncertainty | **Do not expand now.** Preserve this result and audit query-cluster coverage | Stop composition line if a larger fixed query reverses the external sign or reduces it to a few clusters |
| Positive support preserves HW anomaly morphology | Local ranking mechanism | Positive main effect is 3/3 for TabPFN HW rank gap and AUC | Both 1,000-draw clustered CIs cross 0; tree local response differs | More independent HW 0–1 segments/buildings | No new fit under current gate | Treat as descriptive only unless a cluster-stable local effect appears |
| Frozen scaler explains the factorial | Preprocessing alternative | Tree pilot proves scaling is not invariant | External TabPFN interaction persists under frozen scaler; scaler interaction is small for steam/recall | Larger query replication | None | Reject as sole explanation |

The first-round screen is not an expansion decision. Its local HW negative
stratum is too small and the external result includes an AUC ceiling, so the
screen supplies directionally useful composition hypotheses only. It must not
be used to select or score the independent query.

## Post-fit audit and replication status

`factorial_raw_estimand_audit.csv` records the raw row/building/segment count,
score quantiles, normal count, threshold rule, and false-positive count for
every model × seed × scaler × cell. `factorial_interaction_dominance_audit.csv`
shows that no single algebraic cell term exceeds 0.654 of a local interaction,
but it also flags the steam AUC ceiling. Thus the original external interaction
is not a single-cell arithmetic artifact, while its scale is constrained by the
small fixed query.

An independent 192-row mechanism query has been frozen before reading any new
predictions: 64 HW 0–1 positives (34 buildings / 52 segments), 64 HW 0–1
negatives (34 / 64), and 64 steam positives (40 / 58). It excludes all 352
original rows and all 142 original-query buildings, with at most two rows per
building and segment. However, the original runners persisted scores and
scalers only; no TabPFN fitted states, tree models, or per-booster predictions
were retained. Under the no-retraining instruction this query cannot be scored.
The replication is therefore blocked rather than silently replaced by a refit.

That blocker was subsequently addressed by an authorized exact-design refit
which saved every state. Its original deterministic reproduction gate failed
for TabPFN and its portable-state reload was variable, while trees reproduced
and reloaded exactly. This is execution-variation evidence, not a scientific
closure of Path A. The independent query remains unscored. The next authorized
design is a fixed-8.0.8 four-cell repeated-inference variance pilot with one
fit per cell and replicate-aware factorial estimation; no 24-cell grid, Path B,
or query scoring is authorized here.

## Reproducible outputs

- `scripts/prepare_m5_hotwater_label_role_factorial.py`
- `scripts/run_m5_hotwater_label_role_factorial.py`
- `scripts/analyze_m5_hotwater_label_role_factorial.py`
- `factorial_cell_estimands.csv`, `factorial_effects.csv`, and
  `factorial_cluster_bootstrap.csv` in the factorial report directory.
