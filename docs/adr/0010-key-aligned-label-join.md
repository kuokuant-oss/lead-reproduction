# Key-aligned label join

## Status

Accepted

## Context

M3 currently loads `bad_meter_readings.csv` and assigns labels positionally:
equal row count is checked, then values are copied into `train["anomaly"]`.
This reproduces the current code path but does not protect against row-order
drift.

M4.2 inspected the raw GEPIII label file and found that
`data/raw/m3/bad_meter_readings.csv` contains exactly one column:
`is_bad_meter_reading`. It does not contain `building_id`, `timestamp`, or
`meter`. A true key-aligned merge is therefore not available from the raw files.
The executable M3 baseline remains the reproduction authority.

## Decision

M4.2 keeps the positional label assignment, but accepts it only as guarded
positional alignment. The guard now fails loudly unless:

+ `bad_meter_readings.csv` has exactly the single expected label column,
  `is_bad_meter_reading`.
+ The label file and `train.csv` frame have equal length.
+ The train frame still has the raw row-order `RangeIndex`.
+ `(building_id, meter, timestamp)` is unique in the train frame.

This is not a key join. It is the only join available from the raw files, made
explicit and test-protected.

## Rationale

An anomaly label is a data-integrity boundary. Equal length is not enough to
prove label correctness. However, without label-side keys, constructing a
key-aligned merge would require fabricating row identity that is not present in
the source file.

The guarded positional path keeps labels byte-identical to M3 while moving the
known fragility into an explicit integrity gate. The agreement rate with the old
assignment is trivially 100% because the assigned label vector is unchanged; the
value of this slice is that schema drift, length drift, non-canonical row order,
and undefined train row identity now fail at the label boundary.

Verification evidence:

+ Full raw train probe: 20,216,100 rows.
+ `(building_id, meter, timestamp)` duplicate count: 0.
+ Raw train index: default `RangeIndex`.
+ `tests.test_label_join_integrity`: passes, including a synthetic row-order
  mismatch case.
+ `tests.test_refactor_regression` and `tests.test_call_arity`: pass.
+ M3.2/M3.4 reruns were not required for this guard-only slice because labels
  are unchanged; expected AUC delta is 0 and remains within the +/- `0.0005`
  noise floor.

## Consequences

+ M4.2 does not change M3 label semantics or headline metrics.
+ Future keyed labels can replace this path with a real merge, but only when
  source labels carry stable keys.
+ A future train reorder with preserved original index now fails before labels
  are assigned.
+ A train reorder followed by index reset remains fundamentally undetectable
  without label-side keys; this limitation is documented rather than hidden.
+ M4.3 timestamp value-change work remains out of scope and ADR 0011 remains
  Proposed.
