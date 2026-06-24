# M4 Evaluation Report

## Foundation Health

The current M3 pipeline is numerically strong but structurally fragile.

P1: core helper logic is copied across scripts. Before M4, `load_m3_frame`
appears in three tracked M3 scripts, `add_value_change_features` appears in four
places, and `downsample_indices` appears in three places. The package directory
`src/lead/` is only a shell and no tracked pipeline imports `lead`.

P2: the M3 anomaly label join is positional. The code checks only that
`bad_meter_readings.csv` and `train.csv` have equal length, then assigns
`bad["is_bad_meter_reading"].values`. A row-order mismatch would not fail.

P3: M3 value-change features use `groupby().shift()` row offsets. Missing hours
inside a building's observed timestamp range can therefore make an `n`-row
shift differ from an `n`-hour shift. The timestamp-merge version exists in the
M2.4 test-side notebook path, but M3 has not adopted it.

P4: `StandardScaler` is used before tree models, where it is not expected to
change split behavior. Current downsampling also duplicates positive examples
through `[negs1, pos, negs2, pos]`. M4.1 must preserve these semantics; M4.4 is
the first slice allowed to revisit them.

## Numeric Inventory

Golden metrics to preserve during M4.1:

| Result | AUC | Source |
|---|---:|---|
| M3.2 LightGBM 80/20 offline | 0.9920 | `data/processed/m3_2_results.json` |
| M3.4 4-model ensemble 80/20 offline | 0.9928 | `data/processed/m3_4_results.json` |
| 50/50 offline ensemble | 0.9921 | `docs/metrics/m3-50-50-ensemble.json` |
| 50/50 causal ensemble | 0.9911 | `docs/metrics/m3-50-50-ensemble.json` |
| Site-held-out ensemble | 0.9774 | `data/processed/m3_5_results.json` |
| Steam meter | 0.9553 | `data/processed/m3_5_results.json` |

The regression noise floor is +/- `0.0005`. M4.1 should treat movements within
that band as refactor noise and movements outside that band as a stop-and-review
event.

## Evaluation Protocol

M4 uses three layers:

| Layer | Purpose | Gate |
|---|---|---|
| Golden regression | Preserve current M3 behavior during refactor | AUC delta < +/- `0.0005` |
| Diagnostic robustness | Keep known limitations visible | Site-held-out and steam metrics remain tracked |
| Readiness review | Decide whether `src/lead` can support M5 | Public API documented and duplicated helpers removed |

Metric rules:

+ AUC is the primary gate for M3.2 and M3.4 compatibility.
+ Precision, recall, and F1 at threshold `0.5` are recorded when available but
  do not supersede the AUC gate.
+ A metric value used as a gate must record source file and commit provenance.
+ If paper text, docs, and executable code disagree, code is the baseline until
  an ADR intentionally changes behavior.

## BDG2 Readiness Conclusion

M4 is necessary before BDG2/FDD work starts. The current scripts are sufficient
to reproduce M3, but they are not yet a stable foundation for another dataset:
data loading, feature generation, split logic, sampling, and evaluation are
embedded in experiment runners. M4.0 and M4.1 should therefore lock the existing
numbers and extract a minimal importable API without changing semantics.

BDG2 readiness should not be claimed until M4.5. After M4.1, the repo should be
importable and regression-protected, but P2/P3/P4 will still be known
limitations awaiting explicit follow-up slices.
