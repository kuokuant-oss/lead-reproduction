# Require raw BDG2 data for transfer/FDD scoring

## Status

Accepted (2026-06-30)

## Context

ADR 0017 defines `load_bdg2_frame` as the general BDG2 loader. Its default
variant remains `cleaned` because the loader is a broad data-access surface and
existing loader tests cover the cleaned release behavior.

ADR 0019 fixes the Phase E evaluation paradigm: a GEPIII-trained detector is
applied to BDG2 as unlabeled transfer evidence, with BDG2-only and
GEPIII-overlap strata reported separately. ADR 0020 then defines the useful
output as within-context evidence packets and audit-yield measurement rather
than absolute cross-dataset risk.

Miller et al. 2020 describe the BDG2 cleaned release as applying Twitter
AnomalyDetection, removing zero-reading runs longer than 24 hours, and removing
electricity zeros. The tracked paper note in
`docs/reference/papers/bdg2-miller-2020.md` records the same release-level
cleaning rules. Scoring electricity FDD on cleaned data would remove exactly
the zero-run and electricity-zero candidates that M6.1 is meant to surface.

Unknown #27 is now open for a separate caveat: the GEPIII/Kaggle source kept UTC
weather timestamps and unit-conversion errors left as-is, while BDG2 raw and
cleaned data use local-time weather and corrected units. That source-vs-target
regime shift must travel with M6 transfer outputs, but it does not block
within-context ranking.

## Decision

Require raw BDG2 data as the default scoring variant for the Phase E transfer
and M6 FDD scoring path.

Implement the guarantee above the general loader through the
`load_bdg2_scoring_frame` wrapper in `scripts/phaseE_transfer.py`. The wrapper
defaults to `variant="raw"` and delegates to `load_bdg2_frame`. Cleaned remains
available only as an explicit sensitivity, bridge, or raw-vs-cleaned convergence
companion.

Phase E Step 4 transfer scripts must route through this raw-first scoring entry
point. `load_bdg2_frame` keeps its general default, and `lead.__all__` is not
changed.

This is a correctness precondition for M6.1. The full-corpus electricity scan
must use raw BDG2 electricity data for scoring, with any cleaned comparison
reported as a companion data-quality screen rather than the primary scoring
surface.

This does not change the transfer paradigm. GEPIII-trained detector transfer to
BDG2 remains fixed under ADR 0019.

This does not touch the M3 numeric line. `load_m3_frame` defaults, M3.2/M3.4
golden values, downsampling, scaler fitting, and the existing GEPIII/Kaggle
source path remain unchanged.

This preserves and carries TabPFN forward. ADR 0020's TabPFN audit roles remain
offline re-ranking, disagreement diagnostics, active-learning audit-set
selection, and few-shot calibration after human review data exists.

## Rationale

FDD candidate surfacing must start from the raw signal when the cleaned release
has already removed a class of candidate events. Otherwise the first M6
electricity scan would be biased toward what survived BDG2's release cleaning
instead of what the detector should review.

Keeping the raw-first rule in a transfer/FDD wrapper preserves ADR 0017's
general loader contract. It also gives later M6 code a named entry point that
communicates intent at call sites without changing the reusable loader default.

Unknown #27 does not force a different scoring source because M6 outputs remain
within-context ranks and quantiles. ADR 0019 still forbids absolute-score risk
claims, so the weather/unit regime shift is a required caveat rather than a
blocking metric invalidation.

ADR 0018 already isolates the GEPIII/Kaggle-only `0.2931` unit correction and
keeps it out of `load_bdg2_frame`; this ADR reuses that boundary to avoid double
conversion in the BDG2 path.

## Consequences

+ `scripts/phaseE_transfer.py` owns the raw-first transfer/FDD scoring wrapper.
+ Phase E Step 4 runners use the wrapper instead of importing the general loader
  directly.
+ Cleaned BDG2 scoring remains explicit and secondary.
+ M6.1's raw-first precondition is satisfied, but A4 does not run the M6.1 scan.
+ Unknown #27 must travel with every M6 transfer output as a non-blocking
  source-vs-target regime caveat.
+ No BDG2 supervised metrics, confirmed fault labels, absolute cross-dataset
  top-K, M3 numeric-line changes, or TabPFN full-corpus scanner role are
  introduced.
