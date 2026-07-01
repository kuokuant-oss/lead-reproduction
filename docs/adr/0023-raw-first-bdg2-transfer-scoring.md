# Require raw BDG2 data for transfer/FDD scoring

## Status

Accepted (2026-06-30)

Amended by ADR 0025 (2026-07-01): raw-first remains required, but the primary
rationale now includes supervised evaluation correctness. Scoring only cleaned
BDG2 would evaluate after the release cleaning process may have removed
candidate positives.

## Context

ADR 0017 defines `load_bdg2_frame` as the general BDG2 loader. Its default
variant remains `cleaned` because the loader is a broad data-access surface and
existing loader tests cover the cleaned release behavior.

At acceptance time, ADR 0019 framed Phase E as unlabeled transfer evidence and
ADR 0020 framed the useful output as audit evidence. ADR 0025 later superseded
that primary paradigm for the GEPIII-overlap, 2016, meters-0-3 subset, where
bridged rank-1 GEPIII annotations make supervised metrics legitimate.

Miller et al. 2020 describe the BDG2 cleaned release as applying Twitter
AnomalyDetection, removing zero-reading runs longer than 24 hours, and removing
electricity zeros. The tracked paper note in
`docs/reference/papers/bdg2-miller-2020.md` records the same release-level
cleaning rules. Scoring electricity FDD on cleaned data would remove exactly
the zero-run and electricity-zero candidates that M6.1 is meant to surface.

Unknown #27 is now open for a separate caveat: the GEPIII/Kaggle source kept UTC
weather timestamps and unit-conversion errors left as-is, while BDG2 raw and
cleaned data use local-time weather and corrected units. That source-vs-target
regime shift must be measured in M6.2, but it does not block M6.1 label-bridge
integrity.

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

This is a correctness precondition for M6.1. The label-bridge integrity slice
and later supervised scoring must use raw BDG2 data as the primary scoring
surface, with any cleaned comparison reported as a companion sensitivity rather
than the primary metric surface.

This paragraph is superseded by ADR 0025 for the primary M6 path. GEPIII-trained
detector scoring remains the transfer object, but the GEPIII-overlap subset now
supports supervised metrics through the bridged rank-1 annotations.

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

Unknown #27 does not force a different scoring source because M6.1 keys labels
by building, meter, and timestamp, independent of `meter_reading` value. Under
ADR 0025, M6.2 should measure the weather/unit regime shift as a supervised
overlap delta rather than treat it as a blocking label-integrity problem.

ADR 0018 already isolates the GEPIII/Kaggle-only `0.2931` unit correction and
keeps it out of `load_bdg2_frame`; this ADR reuses that boundary to avoid double
conversion in the BDG2 path.

## Consequences

+ `scripts/phaseE_transfer.py` owns the raw-first transfer/FDD scoring wrapper.
+ Existing Phase E Step 4 runners used the wrapper instead of importing the
  general loader directly; their old metric contract is a known code/docs gap
  for a later M6.2 refactor.
+ Cleaned BDG2 scoring remains explicit and secondary.
+ M6.1's raw-first precondition is satisfied, but A4 does not run the M6.1 scan.
+ Unknown #27 must be measured in M6.2 as a non-blocking source-vs-target
  regime delta.
+ This ADR itself introduced no supervised metric, label bridge execution, M3
  numeric-line change, or TabPFN full-corpus scanner role.
