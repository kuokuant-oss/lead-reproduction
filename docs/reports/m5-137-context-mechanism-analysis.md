# M5 137-feature context mechanism analysis

**Date:** 2026-07-29
**Status:** Phase-1 CPU analysis complete; no new fit launched
**Protocol:** Existing natural-prevalence GEPIII building-disjoint full holdout,
5k/10k/20k/50k/100k contexts, F4 (137 temporal value-change features),
TabPFN-3 and matched-row tree ensembles.

## Scope and reproducibility

This is a post-hoc analysis of pre-existing predictions, not a new fit. The
entry point is:

```powershell
.venv\Scripts\python.exe scripts\analyze_m5_137_context_mechanism.py --bootstrap-draws 1000
```

It writes CPU-derived artifacts to
`data/processed/m5_context_mechanism_137/`:

- `m5_137_row_score_rank_movement_{tabpfn,trees}.parquet` — 10,137,155 rows
  each; frozen row identity, five scores, paired score movement, and global and
  within-meter percentile-rank movement.
- `m5_137_pairwise.csv`, `m5_137_distributions.csv`, `m5_137_operating.csv`,
  and `m5_137_crossings.csv` — cell-level score, ranking, and operating-point
  evidence.
- `m5_137_bootstrap.csv` and `m5_137_influence.csv` — 1,000-draw
  building-clustered intervals and leave-one-building diagnostics.
- `m5_137_anomaly_segments.parquet` and `m5_137_anomaly_segment_phases.parquet`
  — 13,334 consecutive-hour anomaly segments and onset/middle/recovery slices.

The raw-reading regime is label-blind and meter-specific. For hotwater it is
explicitly `0–1` versus `>1`; other meters use low/middle/high holdout reading
quartiles. Segment temporal statistics are computed in each holdout
building×meter stream, so their 24h/168h values are descriptive and do not
introduce training information.

## Main result: context improves ranking for both learners, without a clear winner swap

The table separates every positive-meter × negative-meter comparison into
within-meter and cross-meter components. Pair weights are fixed by the
full-holdout label and meter counts (cross 0.6008; within 0.3992).

| Model | Component | AUC 5k | AUC 100k | Change |
| --- | --- | ---: | ---: | ---: |
| TabPFN | cross-meter | 0.985142 | 0.989678 | +0.004536 |
| TabPFN | within-meter | 0.991770 | 0.995229 | +0.003459 |
| Trees | cross-meter | 0.985348 | 0.989831 | +0.004483 |
| Trees | within-meter | 0.991722 | 0.995203 | +0.003481 |

Both components improve from 5k to 100k for both learners. The gains are
nearly matched: at 100k, trees remain ahead by 0.000153 in the cross-meter
component, while TabPFN is ahead by 0.000026 within meter. Thus this evidence
does not support a primary claim that larger contexts reverse the overall
TabPFN/tree ordering. It does support analysing *how* shared score references
move, rather than relying on a pooled AUC.

The fixed-FPR results rule out an effect confined to a 0.5 threshold. At global
FPR=0.001, recall rises from 0.5016 to 0.6032 for TabPFN and from 0.4846 to
0.6148 for trees. At global FPR=0.01, it rises from 0.7613 to 0.8320 and from
0.7416 to 0.8285, respectively. The corresponding fixed-TPR operating points
also reduce FPR (for example, FPR at TPR=0.8 falls from 0.01245 to 0.00757 for
TabPFN and from 0.01369 to 0.00809 for trees).

## Where movement occurs

### Label-specific rank movement is stable

Building-clustered 95% intervals for 5k→100k within-meter percentile-rank
movement are positive for anomaly rows in every meter and both learners. Normal
rank movement is near zero, with intervals spanning zero. Selected estimates:

| Meter | TabPFN anomaly rank movement (95% CI) | Trees anomaly rank movement (95% CI) |
| --- | ---: | ---: |
| electricity | +0.0176 (+0.0147, +0.0205) | +0.0183 (+0.0147, +0.0221) |
| chilledwater | +0.0148 (+0.0117, +0.0176) | +0.0130 (+0.0095, +0.0165) |
| steam | +0.0205 (+0.0152, +0.0261) | +0.0220 (+0.0164, +0.0278) |
| hotwater | +0.0579 (+0.0443, +0.0708) | +0.0356 (+0.0240, +0.0464) |

No single building dominates these means: the largest leave-one-building change
in the hotwater-positive TabPFN rank estimate is 0.00245, and in electricity it
is 0.00043. This is building-level robustness only; segment-clustered
uncertainty remains a requirement for a submission-ready claim.

### Pre-specified diagnostic cuts

- **Hotwater 0–1:** absolute scores decline substantially for both labels, but
  the normal decline is larger. TabPFN means move from 0.5027 to 0.3197 for
  normal rows and from 0.8717 to 0.7526 for anomaly rows. The positive
  within-meter rank nevertheless rises by +0.0579. This is evidence of a
  changing shared score reference, not evidence that an absolute 0.5 threshold
  is stable.
- **Steam at ≥100k:** TabPFN exhibits the largest pairwise movement: steam
  positives versus hotwater negatives improve by +0.03715 AUC (0.94095→0.97810),
  versus +0.01712 for trees. TabPFN steam-positive score movement is +0.0722
  (building-bootstrap 95% CI +0.0538 to +0.0906), while steam normals move
  −0.0331 (−0.0406 to −0.0261). This is the strongest localized mechanism
  candidate, but it has only one context draw.
- **Electricity larger-context misses:** at an electricity-specific FPR=0.001
  threshold re-estimated at each context, TabPFN has 21,271 5k-above→100k-below
  anomaly crossings and 15,917 reverse crossings. The direction is therefore
  threshold-sensitive despite a positive rank movement. This diagnostic must
  not be promoted as a generic “new miss” claim without freezing an operating
  threshold and repeating contexts.
- **Chilledwater non-monotonicity:** some raw score regimes are non-monotone
  (TabPFN middle-reading anomaly means: 0.7982, 0.7938, 0.7796, 0.8119,
  0.8217), while its positive within-meter rank movement remains positive and
  clustered-stable. This supports reporting distributional movement rather than
  treating every raw-score trajectory as a ranking effect.

## Segment morphology

The 637,397 anomaly rows merge into 13,334 segments (median duration two hourly
rows; 90th percentile 60; 99th percentile 983.4). Segment-mean TabPFN score
movement is +0.0954 and tree movement +0.0663; their difference moves +0.0291.
These figures are descriptive because duration is extremely right-skewed (max
7,404 hours). Electricity contributes 7,999 segments and 356,679 anomaly rows,
so all segment claims must use cluster-aware uncertainty and concentration
checks before publication.

## Convergence table and next decision

| Candidate claim | Novelty | Supporting evidence | Counterevidence | Missing evidence | Minimum next experiment | Stopping condition |
| --- | --- | --- | --- | --- | --- | --- |
| A. Label-specific support establishes a shared score reference | A mechanism for heterogeneous-context ranking rather than a generic curve | Positive rank movement in all four meters with building CIs excluding zero; fixed-FPR and pairwise AUC move; hotwater 0–1 shows relative improvement despite score-scale decline | Matched trees show the same broad pattern; TabPFN/tree ordering does not clearly cross | ≥3 context seeds, explicit positive/negative support intervention, frozen-scaler control, segment-clustered uncertainty | **Select Path A:** F4 hotwater positive-support × negative-support 2×2, N=20k, fixed query, matched TabPFN/trees | No interaction; seed signs disagree; effect reduces to one threshold; building/segment concentration |
| B. Temporal operator family makes heterogeneous support usable | Would localize the 137-feature mechanism to an operator/direction | Steam score and pairwise movement are larger for TabPFN; reading-regime trajectories are heterogeneous | No operator ablation has been run; trees share substantial movement | Targeted feature contrasts and slope comparison | Do not fit B now; retain as fallback only if Path A fails or its interaction predicts no response | Contrasts lack a stable sign/slope change |
| C. Calibration/threshold only | Negative result; prevents overclaim | Hotwater and electricity have threshold-sensitive crossings | Pairwise ranking and fixed-FPR metrics change, so C is not sufficient for the current pooled effect | Repeated contexts to decide whether this is stable | No independent fit for C | If replicated contexts erase pairwise/fixed-FPR change, stop the mechanism claim |

**Decision:** select only **Path A** for any future new fit. The choice is
provisional and does not authorize a GPU run in this phase. Do not expand past
the N=20k factorial unless the interaction is clear; only then add N={5k,
100k}, at least three context seeds, frozen-scaler control, and full holdout.

## Submission gates

This analysis does not yet meet the Q2 threshold: it has one natural-prevalence
full-holdout draw per context size, but no intervention and no three-seed
confirmation. A Q2 path needs one clear finding, a supporting intervention or
representation contrast, at least three context seeds, building- and
segment-clustered uncertainty, and full-holdout/natural-prevalence confirmation.
Q1 additionally requires a reusable support-allocation diagnostic,
cross-domain-alignment metric, or context selector and confirmation with a
second modern tabular learner or usable data construction.
