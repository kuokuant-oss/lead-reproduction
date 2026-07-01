# BDG2 label-bridge integrity

## Status

Proposed (2026-07-01)

## Context

ADR 0010 already protects the M3 positional join between Kaggle `train.csv` and
`bad_meter_readings.csv`: the label file must have exactly the expected schema,
the same length as `train.csv`, a default row index, and a safe alignment before
labels are attached.

M6 needs a different guard. BDG2 rows must not receive labels positionally.
Labels must be keyed from Kaggle/GEPIII rows and merged onto BDG2 overlap rows
through the preserved Kaggle building id plus meter code and timestamp.

The bridge is valid because the label key is independent of `meter_reading`
value. BDG2 raw/cleaned unit corrections can change feature distributions seen
by a GEPIII-trained detector, but they do not change the row identity
`(building, meter, timestamp)` used to attach the rank-1 manual annotation.

## Decision

M6 must use a keyed label bridge before reporting supervised BDG2 metrics.
P0 is documentation-only: this ADR defines the required M6.1 gate but does not
run the bridge, read BDG2 data, create provenance, or report metrics.

The bridge contract is:

+ read `data/raw/m3/train.csv` with `building_id`, `meter`, and `timestamp`;
+ read `data/raw/m3/bad_meter_readings.csv` with exactly
  `is_bad_meter_reading`;
+ apply ADR 0010-style positional guards before constructing the keyed GEPIII
  label table;
+ assert keyed uniqueness for `(building_id, meter, timestamp)`;
+ map BDG2 overlap buildings through `building_id_kaggle`;
+ restrict eligible BDG2 rows to GEPIII-overlap, year 2016, and meters
  `electricity`, `chilledwater`, `steam`, `hotwater`;
+ map those meters to GEPIII codes `0`, `1`, `2`, `3`;
+ left-merge labels on `(kaggle_building_id, meter_code, timestamp)`;
+ fail loudly on unexpected null-label rates or duplicate bridge keys.

The bridge must produce a coverage report before any supervised scoring:

+ eligible BDG2 rows by meter;
+ bridged label hit rate and null-label rate by meter;
+ positive label counts by meter;
+ excluded BDG2-only, 2017, and other-meter row counts.

## Integrity Gates

Timestamp-grid identity must be sampled and reported. For selected overlap
buildings and meters, compare the BDG2 raw 2016 timestamp grid to the Kaggle
`train.csv` grid for the same building and meter. If grids differ, the keyed
merge protects correctness, but the coverage report must expose the mismatch.

Label-value independence must be explicit. The bridge attaches labels by
building, meter, and timestamp only. It must not infer labels from BDG2
`meter_reading`, raw-vs-cleaned differences, score values, or unit-correction
effects.

The GEPIII-only `0.2931` correction remains outside the BDG2 path per ADR 0018.
Any resulting feature distribution shift is evaluated in M6.2 as unknown #27,
not treated as a label-integrity failure.

## Consequences

+ M6.1 is a bridge-and-integrity slice, not a metric slice.
+ The public `lead.__all__` remains frozen unless a later implementation exports
  a new helper additively with tests and ADR coverage.
+ The old unlabeled BDG2 loader default remains valid. A labeled overlap helper
  must be optional and scoped to M6 supervised evaluation.
+ No supervised M6.2 result can be reported until this integrity gate passes and
  records provenance.
