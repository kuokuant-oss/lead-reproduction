# Key-aligned label join

## Status

Proposed

## Context

M3 currently loads `bad_meter_readings.csv` and assigns labels positionally:
equal row count is checked, then values are copied into `train["anomaly"]`.
This reproduces the current code path but does not protect against row-order
drift.

## Decision

M4.2 should replace positional assignment with a key-aligned join or add a
documented invariant check that proves the positional files are aligned by the
same keys before assignment.

## Rationale

An anomaly label is a data-integrity boundary. Equal length is not enough to
prove label correctness. A key-aligned path gives future BDG2 work a safer
interface and makes failures local and testable.

## Consequences

+ M4.1 must not change label semantics.
+ M4.2 must measure M3.2 regression against the golden AUC `0.9920`.
+ Any movement beyond +/- `0.0005` requires review before proceeding.
+ Tests should include a row-order mismatch case.
