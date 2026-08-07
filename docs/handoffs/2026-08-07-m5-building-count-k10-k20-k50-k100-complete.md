# M5 building-count K=10/20/50/100 complete — active handoff

**Date**: 2026-08-07
**State**: formal K=10/20/50/100 grid complete; aggregate report published; no
model process remains to launch or resume

This handoff is authoritative for the completed building-count run, its actual
formal configuration, its publication history, and the post-completion
Hot-water diagnosis. It supersedes the run-state portion of
[`2026-08-04-m5-building-count-early-stopping-ready.md`](2026-08-04-m5-building-count-early-stopping-ready.md).
That earlier handoff remains useful as the implementation-readiness record.

## 0. Final state

The formal queue is complete. The supervisor records `10/10` completed stages,
`failed_stages=[]`, an atomic `COMPLETE.json`, and a final publication to
`github/main` at commit `16712f0` (`Record M5 building stage
tabpfn_k100_f137`).

The ten supervised stages were:

1. full-building Tree references at 17 and 137 features;
2. Tree K=10, 20, 50, and 100 at 137 features;
3. TabPFN K=10, 20, 50, and 100 at 137 features.

All four K cells have model artifacts, predictions, provenance, heartbeats, and
atomic `COMPLETE.json` markers. The final aggregate report contains every Tree
and TabPFN K point. **Do not restart the overnight queue merely to regenerate
these completed cells.**

## 1. Scientific question and scope

This experiment asks how anomaly-detection performance changes when labels are
available from more source buildings:

\[
K \in \{10,20,50,100\}.
\]

K is the number of selected training buildings. It is not the number of
buildings available for each individual meter type. This distinction is
material: at K=100 only 10 selected buildings contain Hot-water rows.

The primary comparison is the 137-feature Tree ensemble against TabPFN on one
fixed odd-building canonical holdout. The experiment measures building-source
scarcity under a natural, deterministic training composition. It is different
from the matched-row context curve, which explicitly fixes row counts and class
composition.

## 2. Frozen protocol

| Property | Formal value |
| --- | --- |
| training-building split | even `building_id` only |
| final holdout | odd `building_id` only |
| candidate training buildings | 725 |
| building budgets | 10, 20, 50, 100 |
| feature line | 137 features |
| building/row seed | 42 |
| sampling profile | `representative` |
| row policy | `average_building_cap` |
| average row allocation | at most 500 rows per selected building |
| total context ceiling | 50,000 rows |
| K context rows | 5,000 / 10,000 / 25,000 / 50,000 |
| holdout rows | 10,137,155 |
| holdout anomalies | 637,397 |
| Tree models | LightGBM, XGBoost, CatBoost, HistGradientBoosting, and their mean ensemble |
| Tree roles | fixed 80% fit / 20% external early-stop buildings |
| TabPFN estimators | 8 |
| TabPFN query microbatch | 4,096 |
| TabPFN checkpoint rows | 20,000 |
| formal caps | none beyond the frozen K context policy |

### 2.1 Building ladder

The same deterministic ladder is used by every K cell. K=10, K=20, K=50, and
K=100 are strict nested prefixes; neither buildings nor selected row identities
are redrawn between cells.

The `representative` objective balances these building-level dimensions against
the even-building candidate population:

- site membership;
- overall anomaly-rate strata;
- building-size strata;
- per-meter presence;
- per-meter row share;
- overall anomaly rate and zero-anomaly share.

It does **not** explicitly balance `meter × site × label`, per-meter anomaly
rate, or the number of positive buildings within each meter. That boundary
explains the Hot-water composition discussed in Section 7.

Within each incremental K block, a building receives a quota proportional to
its available rows. Rows are chosen by a seed-42 stable hash of raw row identity
without consulting labels. Individual buildings can therefore contribute more
or fewer than 500 rows while the K-prefix average remains 500.

### 2.2 Model data contracts

Both families begin from the same selected K source rows, but their effective
fit rows are not identical:

- TabPFN consumes every selected K row as one natural-prevalence in-context
  dataset. It performs no task-specific epoch loop or checkpoint weight update,
  so conventional early stopping is not applicable.
- Trees reserve every fifth building for external early stopping. Their final
  fit uses the M3 post-feature-sort `[negs1,pos,negs2,pos]` class-downsampling
  path, selects iteration counts on external validation, and then refits at
  those counts on downsampled rows from all K source buildings.

The actual Tree early-stopping selection metric in the formal cell provenance
is **ROC-AUC**. The opening paragraph of the generated
[`m5-building-count-experiment.md`](../reports/m5-building-count-experiment.md)
currently says PR-AUC; that sentence is inconsistent with the cell artifacts
and the report's own early-stopping audit table. Treat the cell provenance and
audit table as authoritative until the report generator is corrected.

| K | Tree fit-source rows | Tree ES rows | Tree final-refit rows | TabPFN context rows |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 4,146 | 854 | 1,628 | 5,000 |
| 20 | 8,116 | 1,884 | 2,404 | 10,000 |
| 50 | 20,102 | 4,898 | 6,476 | 25,000 |
| 100 | 40,108 | 9,892 | 12,036 | 50,000 |

## 3. Training composition

The overall K-prefixes preserve a natural anomaly rate near 6% after K=10.
Meter row proportions are also stable, but this does not guarantee stable
within-meter label composition.

| K | Rows | Anomalies | Anomaly rate | Electricity | Chilled water | Steam | Hot water | Anomalous building-meter pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5,000 | 407 | 8.140% | 3,107 | 991 | 580 | 322 | 10/17 |
| 20 | 10,000 | 601 | 6.010% | 6,450 | 1,636 | 1,267 | 647 | 17/32 |
| 50 | 25,000 | 1,619 | 6.476% | 15,383 | 4,651 | 3,372 | 1,594 | 50/81 |
| 100 | 50,000 | 3,009 | 6.018% | 30,376 | 9,816 | 6,691 | 3,117 | 99/162 |

At K=100 all 16 sites appear somewhere in the full context. That global site
coverage must not be interpreted as all 16 sites appearing inside every meter.

## 4. Execution chronology

The watchdog started the resumable `m5-building-overnight` tmux session on
2026-08-05 17:57 Asia/Taipei. Before K execution, commit `56c07c1` increased
TabPFN's query microbatch from 512 to 4,096 and added the value to provenance.
Every formal K artifact records 4,096, seed 42, eight estimators, the same model
checkpoint, and no internal context subsampling.

| K | Tree wall time | TabPFN supervisor wall time | Final TabPFN artifact time | Publication time (Asia/Taipei) | Publication commit |
| ---: | ---: | ---: | ---: | --- | --- |
| 10 | 5m 07s | 5h 34m 44s | 2h 12m 07s | 2026-08-06 00:09 | `9c422f0` |
| 20 | 5m 26s | 3h 36m 40s | 3h 32m 18s | 2026-08-06 03:57 | `fdb6f31` |
| 50 | 5m 45s | 8h 32m 34s | 8h 28m 04s | 2026-08-06 12:45 | `6d6bc54` |
| 100 | 8m 07s | 20h 11m 24s | 20h 06m 04s | 2026-08-07 09:16 | `16712f0` |

`cell.json:elapsed_seconds` begins after the holdout feature matrix is built and
can reuse existing prediction chunks. Supervisor wall time includes process
startup and feature construction, so the two columns are not interchangeable
throughput measurements.

K=10 crossed a supervisor restart. The event log retains one unmatched earlier
`command_start`, while the resumed command completed with return code 0 and
reused durable artifacts under `--resume`. There are no recorded failed stages,
and the final queue marker has `failed_stages=[]`.

All four Tree families stopped before their ceilings. Formal best iterations:

| K | LightGBM | XGBoost | CatBoost | HistGradientBoosting |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 8 | 28 | 4 | 24 |
| 20 | 4 | 18 | 58 | 46 |
| 50 | 57 | 20 | 705 | 58 |
| 100 | 114 | 188 | 616 | 105 |

## 5. Primary results

### 5.1 Pooled holdout

These metrics pool all 10,137,155 holdout rows. Electricity contributes
6,035,071 rows, so pooled PR-AUC is not an equal-meter summary.

| K | Tree PR-AUC | TabPFN PR-AUC | TabPFN − Tree | Tree ROC-AUC | TabPFN ROC-AUC | TabPFN − Tree |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.722452 | 0.710155 | -0.012297 | 0.968506 | 0.972253 | +0.003747 |
| 20 | 0.727457 | 0.757629 | +0.030172 | 0.966187 | 0.980230 | +0.014043 |
| 50 | 0.852387 | 0.849813 | -0.002574 | 0.983452 | 0.975575 | -0.007877 |
| 100 | 0.856724 | 0.863110 | +0.006386 | 0.982828 | 0.983862 | +0.001034 |

There is no monotonic model-family winner. TabPFN loses pooled PR-AUC at K=10,
wins at K=20, is essentially tied but slightly behind at K=50, and is slightly
ahead at K=100.

### 5.2 Per-meter PR-AUC and equal-meter macro

The macro column is the unweighted mean of Electricity, Chilled water, Steam,
and Hot water. It prevents Electricity's row count from dominating the summary.

| K | Model | Electricity | Chilled water | Steam | Hot water | Meter macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | Tree ensemble | 0.940382 | 0.461733 | 0.320400 | 0.589285 | 0.577950 |
| 10 | TabPFN | 0.956596 | 0.446448 | 0.395821 | 0.720603 | 0.629867 |
| 20 | Tree ensemble | 0.917038 | 0.429845 | 0.296928 | 0.651913 | 0.573931 |
| 20 | TabPFN | 0.957103 | 0.487510 | 0.428093 | 0.735351 | 0.652014 |
| 50 | Tree ensemble | 0.967365 | 0.597643 | 0.420515 | 0.628849 | 0.653593 |
| 50 | TabPFN | 0.967275 | 0.604821 | 0.502285 | 0.697948 | 0.693082 |
| 100 | Tree ensemble | 0.968728 | 0.645239 | 0.589363 | 0.612808 | 0.704034 |
| 100 | TabPFN | 0.971769 | 0.626835 | 0.582334 | 0.660247 | 0.710296 |

| K | Electricity delta | Chilled-water delta | Steam delta | Hot-water delta | Macro delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | +0.016214 | -0.015285 | +0.075421 | +0.131318 | +0.051917 |
| 20 | +0.040065 | +0.057665 | +0.131165 | +0.083438 | +0.078083 |
| 50 | -0.000090 | +0.007177 | +0.081770 | +0.069099 | +0.039489 |
| 100 | +0.003041 | -0.018404 | -0.007029 | +0.047439 | +0.006262 |

The equal-meter TabPFN advantage narrows from +0.078083 at K=20 to +0.006262
at K=100. At K=100, TabPFN leads on Electricity and Hot water, while the Tree
ensemble leads on Chilled water and Steam.

## 6. What the completed curve supports

- Increasing K substantially improves Chilled-water and Steam PR-AUC for both
  families after K=20.
- Electricity remains easy and high-scoring throughout. It must remain visible
  in per-meter reporting, but it should not be allowed to stand in for overall
  cross-meter quality.
- Pooled PR-AUC and equal-meter macro answer different questions and should be
  reported together rather than substituted for one another.
- The TabPFN advantage is largest under the smaller K settings on the
  equal-meter summary; by K=100 the two families are near parity.
- Hot water is non-monotonic for both families and requires the conditional
  composition diagnosis below.

## 7. Hot-water post-completion diagnosis

This section records a read-only, post-hoc diagnostic derived from the frozen
manifest and the completed prediction artifacts. It is not a new trained cell
and is not a pre-registered primary metric.

### 7.1 Effective Hot-water training support

| K | Hot-water buildings | Rows | Anomalies | Anomaly rate | Positive buildings |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 322 | 172 | 53.416% | 1 |
| 20 | 2 | 647 | 172 | 26.584% | 1 |
| 50 | 5 | 1,594 | 173 | 10.853% | 2 |
| 100 | 10 | 3,117 | 201 | 6.448% | 6 |
| fixed test | 73 | 636,121 | 90,691 | 14.257% | — |

The new Hot-water rows added between K points are overwhelmingly negative:

| Increment | Added rows | Added anomalies | Increment anomaly rate |
| --- | ---: | ---: | ---: |
| K10 → K20 | 325 | 0 | 0.000% |
| K20 → K50 | 947 | 1 | 0.106% |
| K50 → K100 | 1,523 | 28 | 1.838% |

At K=100, building 1302 supplies 172 of 201 Hot-water positives (85.572%). The
three largest positive-contributing buildings supply 195 of 201 (97.015%).

### 7.2 K=100 Hot-water source sites

| Site | Selected building IDs | Rows | Anomalies | Anomaly rate |
| ---: | --- | ---: | ---: | ---: |
| 14 | 1302, 1300, 1258, 1294 | 1,204 | 181 | 15.033% |
| 2 | 234, 232, 184 | 961 | 0 | 0.000% |
| 1 | 114, 138 | 670 | 20 | 2.985% |
| 11 | 1032 | 282 | 0 | 0.000% |

The most consequential mismatch is Site 2. Its three selected Hot-water source
buildings contain zero positives, while the fixed test has 59,827 positives in
236,727 Site-2 Hot-water rows (25.273%). Site 2 alone contains 65.968% of all
Hot-water test positives. Sites 10 and 15 also have Hot-water test positives but
no selected K=100 Hot-water source building.

`site_id` is not a direct feature in the 137-feature matrix. Nevertheless,
meter, weather, building metadata, and lag-value features carry strong
site/building-conditioned structure, so this conditional support mismatch can
still affect ranking.

### 7.3 Observed score movement

| K | Hot-water PR-AUC | ROC-AUC | Recall at 0.5 | Mean score on positives |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.720603 | 0.937102 | 0.828329 | 0.766099 |
| 20 | 0.735351 | 0.942879 | 0.796794 | 0.690593 |
| 50 | 0.697948 | 0.933620 | 0.503699 | 0.517267 |
| 100 | 0.660247 | 0.938212 | 0.369629 | 0.404006 |

The thresholded failure is mainly a recall collapse: added mostly-normal
context pushes positive scores downward. The PR-AUC decline is not only a
threshold-calibration artifact, because PR-AUC is threshold-free and also
falls. Performance is heterogeneous: Site-2 Hot-water PR-AUC falls from
0.875933 at K=20 to 0.725652 at K=100, while Site 14 rises from 0.670172 to
0.766153.

An equal-building diagnostic over the 65 mixed-label Hot-water test buildings
moves in the opposite direction:

| K | Mixed-building macro PR-AUC |
| ---: | ---: |
| 20 | 0.566806 |
| 50 | 0.610832 |
| 100 | 0.643062 |

Therefore K=100 is not uniformly worse across buildings. The pooled Hot-water
metric is pulled down by the high-volume/high-positive test segments on which
the selected training context has poor positive support.

### 7.4 What was ruled out

The K=20, K=50, and K=100 cells use:

- seed 42;
- eight TabPFN estimators;
- query microbatch 4,096;
- the same checkpoint SHA-256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`;
- the full requested 10,000 / 25,000 / 50,000 effective context with
  `sample_subsampling=null`;
- no relevant source diff in the TabPFN runner, feature builder, evaluator, or
  building protocol between the recorded cell-provenance commits.

The evidence therefore points to context-composition shift, not a different
checkpoint, hidden context truncation, changed estimator count, or changed
feature implementation. A causal claim would still require a controlled rerun
that changes only Hot-water conditional support; that rerun has not been done.

## 8. Reproducibility and artifacts

Primary tracked report and implementation:

- [`m5-building-count-experiment.md`](../reports/m5-building-count-experiment.md)
- [`m5_building_curve_protocol.py`](../../scripts/m5_building_curve_protocol.py)
- [`prepare_m5_building_curve.py`](../../scripts/prepare_m5_building_curve.py)
- [`run_m5_building_curve_tree_cell.py`](../../scripts/run_m5_building_curve_tree_cell.py)
- [`run_m5_building_curve_tabpfn_cell.py`](../../scripts/run_m5_building_curve_tabpfn_cell.py)
- [`run_m5_building_curve_overnight.py`](../../scripts/run_m5_building_curve_overnight.py)
- [`report_m5_building_curve.py`](../../scripts/report_m5_building_curve.py)
- [`update_m5_building_curve_report.py`](../../scripts/update_m5_building_curve_report.py)

Frozen data and result artifacts:

- `data/processed/m5_building_curve/protocol/representative/seed42/building_ladder.json`
- `data/processed/m5_building_curve/protocol/representative/seed42/building_ladder.csv`
- `data/processed/m5_building_curve/formal/{tree,tabpfn}_k{10,20,50,100}_f137/`
- `data/processed/m5_building_curve/aggregate/metrics.csv`
- `data/processed/m5_building_curve/aggregate/curves.csv`
- `data/processed/m5_building_curve/aggregate/summary.json`
- `data/processed/m5_building_curve/supervisor/status.json`
- `data/processed/m5_building_curve/supervisor/events.jsonl`
- `data/processed/m5_building_curve/supervisor/COMPLETE.json`

Identity gates:

| Artifact | SHA-256 |
| --- | --- |
| building manifest | `2b55a4ca56709caf238a530a856190f030619eba83c2d2de831f8bdc0140b834` |
| canonical holdout rows | `6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893` |
| K=10 context rows | `34c3d72a93e178f323347b071b0934eeec75d4d0625ce844ea1e4c7b6f07de5b` |
| K=20 context rows | `408cc790b280b2d98b8a6cf3165390208163a79dfbe76cad336439e89d81c966` |
| K=50 context rows | `25ed5794fd7ae5aa17071275e43ff1efc811d0cdefa77e39d0a32227e23a58a8` |
| K=100 context rows | `bdbd30d0d1f421737816858175ddc046e2bb3f3d904c48ae4b135d62f844168e` |

Every accepted aggregate row was gated on byte-identical canonical holdout row
identity and labels. Meter/site breakdowns and pooled metrics can be regenerated
from the stored predictions; no model refit is required.

## 9. Carry-forward issues and decisions not taken

1. **K is global, not meter-specific.** Any manuscript statement about
   performance versus K must describe K as total selected source buildings and
   report effective per-meter source support where relevant.
2. **The current sampler does not constrain meter-specific labels.** Decide
   whether the present curve remains the main natural-composition experiment or
   whether a separate meter-conditioned building-scarcity experiment is needed.
   Do not silently replace this completed protocol.
3. **One seed only.** The curve fixes seed 42. It has no building-ladder error
   bars and does not estimate TabPFN context-draw variance.
4. **Tree and TabPFN effective fit rows differ.** The common object is the K
   source-row pool; Trees subsequently apply the M3 class-downsampling path.
   Describe this explicitly in comparisons.
5. **Pooled PR-AUC is Electricity-heavy.** Retain per-meter results and the
   equal-meter macro in submission figures/tables.
6. **Generated-report wording bug.** The report introduction says Tree early
   stopping selects PR-AUC, but formal cell provenance selects ROC-AUC. Fix the
   generator before relying on that prose in a manuscript.
7. **Hot-water causal test remains open.** The read-only evidence strongly
   supports conditional-composition shift, but no controlled context-rebalance
   rerun has been authorized or performed.
8. **Existing untracked visualization work is user work.** At handoff creation,
   `docs/reports/assets/m5-scarcity/`,
   `docs/reports/m5-scarcity-submission-draft.md`,
   `docs/reports/tables/`, `scripts/plot_m5_scarcity_submission.py`, and
   `tests/test_plot_m5_scarcity_submission.py` are untracked. Preserve them and
   do not mistake them for formal K-run provenance.

Nothing in this handoff authorizes a new formal run, a sampler replacement,
additional GPU work, deletion of completed artifacts, or manuscript edits.
Those are separate human decisions.
