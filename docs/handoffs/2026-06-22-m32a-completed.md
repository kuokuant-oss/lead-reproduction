# Handoff: M3 Split/Causality PI Response

**Date**: 2026-06-22
**Status**: Complete
**GitHub Issue**: [#18](https://github.com/kuokuant-oss/lead-reproduction/issues/18)
**Scope**: Experimental design only. No M3.3 feature work.

## Why This Exists

The PI requested train/test to each use half the buildings. Prior M3.2 used
`building_id % 5 == 4` for validation, giving 1160/289 buildings (about 80/20).

M3.2 value-change features also include negative shifts, which are valid for
offline batch labeling but use future meter readings if the task is interpreted
as real-time FDD. This handoff records both regimes.

## Artifacts

| Artifact | Purpose |
|---|---|
| `notebooks/07-m3-split-causality.ipynb` | Notebook record with 2x2 grid and rerun command |
| `scripts/run_m3_split_causality.py` | Tracked reproducible experiment runner |
| `.scratch/run_m3_split_causality.py` | Original scratch copy, gitignored |
| `data/processed/m3_split_causality_results.json` | Machine-readable result payload |
| `docs/m3-report.md` | Report section with 2x2 table and framing |
| `docs/m3-plan.md` | M3.2a design step inserted before M3.3 |

## Experimental Setup

+ Model: LightGBM only
+ Model seed: `random_state=42`
+ Downsampling: unchanged M3.2 logic, normal samples from seeds `10` and `20`
+ Baseline features: existing 17 M3.1 features
+ Offline value-change: 60 shifts x 2 feature types = 120 value-change features
+ Causal value-change: positive/past shifts only = 60 value-change features
+ Building-level separation: all splits have train/val building overlap 0

Shift set:

```python
shifts = (
    list(range(-24, 0))
    + list(range(1, 25))
    + list(range(-168, -24, 24))
    + list(range(48, 169, 24))
)
```

In pandas `groupby().shift(n)`, positive `n` is past-looking and negative `n`
is future-looking.

## Results

| Split | Regime | Features | Train buildings | Val buildings | Train anomaly | Val anomaly | Val AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80/20 mod5 | offline | 137 | 1160 | 289 | 6.65% | 5.93% | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| 80/20 mod5 | causal | 77 | 1160 | 289 | 6.65% | 5.93% | 0.9908 | 0.6237 | 0.9603 | 0.7562 |
| 50/50 mod2 | offline | 137 | 725 | 724 | 6.72% | 6.29% | 0.9914 | 0.6878 | 0.9421 | 0.7951 |
| 50/50 mod2 | causal | 77 | 725 | 724 | 6.72% | 6.29% | 0.9903 | 0.6646 | 0.9355 | 0.7772 |

## Robustness and Sanity

+ 80/20-offline reproduces M3.2: AUC `0.9920`.
+ Seeded-random 50/50 building split (`random_state=42`, offline) gives AUC
  `0.9910`, close to deterministic 50/50 offline AUC `0.9914`.
+ 50/50-causal label-shuffle sanity check gives AUC `0.4527`.

## Interpretation

+ The 50/50 AUC dip is the cost of the PI protocol. Training uses 725 buildings
  instead of 1160, so this is not a model regression.
+ The causal AUC dip is the cost of real-time deployability. The model no longer
  sees future meter readings.
+ This causal result connects directly to the existing M3.2 past/future leakage
  check. M3.2 measured future shifts as only about `+0.0012` AUC over past-only;
  M3.2a operationalizes real-time FDD by removing that future contribution.

## Next Step

Proceed to M3.3 buds-lab feature alignment using the PI-response result as the
design baseline. Do not treat the 50/50 or causal AUC changes as regressions.

## Open Questions

+ Which regime/split is the canonical line for M3.3+? Answer for now: 80/20
  offline, because it is LEAD-aligned and reproduces M3.2 AUC `0.9920`.
+ Should 50/50 or 80/20 be the headline reporting split going forward? Current
  archival keeps 80/20 offline as the headline line and records 50/50 as the
  PI-response protocol unless the reporting target changes.
+ Does the 50/50-causal label-shuffle AUC `0.4527` raise concern versus the M3.2
  label-shuffle AUC `0.5669`? No. It is cleaner and farther from the real
  50/50-causal AUC `0.9903`.

## Next Session Context Requirements

+ `docs/handoffs/2026-06-22-m32a-completed.md`
+ `docs/handoffs/2026-05-29-m3.2-completed.md`
+ `docs/m3-plan.md` M3.3 section and feature-gap table
+ `docs/m3-report.md` Ch4.2 feature-gap table
+ `.scratch/02_preprocess_data.py`
+ `notebooks/06-m3-baseline.ipynb`
+ `notebooks/07-m3-split-causality.ipynb`
