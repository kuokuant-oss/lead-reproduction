# M5 chilledwater mechanism localization (C1)

**Status:** planned CPU-only analysis; no fit authorized.
**Inputs:** existing 5k/10k/20k/50k/100k natural-prevalence full-holdout
predictions and the existing row/segment artifacts.

## Purpose

C1 determines what the meter-specific chilledwater learner response is
consistent with before any chilledwater intervention is considered. It is not a
new claim, an ablation, or an extension of the hotwater factorial.

## Required outputs

1. Separate chilledwater TabPFN movement into within-chilledwater
   anomaly-vs-normal ranking, cross-meter ranking, and absolute-score/
   calibration components.
2. Compare chilledwater-positive movement against electricity-, chilledwater-,
   steam-, and hotwater-negative groups.
3. Localize movement by raw-reading quartile, anomaly phase
   (onset/middle/recovery), duration, slope, 24h/168h deviation, diff/ratio
   morphology, building, and segment.
4. Decompose the TabPFN-minus-tree gap into anomaly rows, normal rows,
   within-meter morphology, cross-meter reference, and threshold/calibration
   contributions.
5. Check 10k/20k/50k/100k direction, building- and segment-clustered
   uncertainty, leave-one-building influence, valid-pair resolution, and
   concentration.

## Decision gate

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| Same hotwater-negative reference | Chilledwater-positive vs hotwater-negative moves in the same stable direction as steam | retain as a Path-A boundary readout; do not add a new factorial |
| Different support source | Another meter/label support group better explains the stable movement | propose one narrowly scoped support-source intervention only after review |
| Within-meter morphology/representation | Effects localize to morphology or temporal operator behaviour | consider a targeted support or feature contrast; do not automatically start Path B |
| No stable localization | Meter-specific advantage exists but mechanism is unresolved | report the performance gap only; make no chilledwater mechanism claim |

## Audit before any factorial readout

The original 352-row screening query must be audited for each required
chilledwater stratum: row, building, and segment count; valid positive-negative
pair count and AUC resolution; and building/segment concentration. Continuous
margins are feasibility readouts, not a substitute for inadequate pair
resolution.

## Prohibitions

C1 must not fit a model, score the 192-row query, rerun a context curve, refit
trees, launch Path B, or change paper text.
