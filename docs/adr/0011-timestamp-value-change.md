# Timestamp value-change features

## Status

Accepted (2026-06-25)

## Context

M3 value-change features use `groupby("building_id").shift(n)`. That gives an
`n`-row offset, not necessarily an `n`-hour timestamp offset when a building has
missing timestamps. The M2.4 test-side implementation used timestamp shifting
and merge semantics.

## Decision

M4.3 adds timestamp-merge value-change as an explicit regime while keeping the
current row-offset regime available for reproduction compatibility.

`row_offset` remains the default regime for the M3 reproduction line. It
preserves the accepted M3.2/M3.4 behavior and keeps existing script calls
compatible. `timestamp_merge` is opt-in and computes n-hour offsets by joining
on timestamp-shifted readings. When a `meter` column is present, the timestamp
merge includes `meter` in the join key so each output row remains aligned with
the original label row.

## Rationale

The current behavior is the code baseline and must remain reproducible. The
timestamp-merge behavior is likely the cleaner semantic model for downstream FDD
and BDG2, but it needs a measured AUC comparison before replacement.

## Consequences

+ M4.1 preserves row-offset features.
+ M4.3 measured timestamp-merge M3.2 in the same harness as row-offset:
  row-offset AUC `0.9920119520500562`, timestamp-merge AUC
  `0.9924831086743003`, same-run Delta AUC `+0.00047115662424412896`.
+ The measured regime delta is within the +/- `0.0005` noise floor, so it does
  not replace the row-offset default.
+ Reports must name the value-change regime when discussing results.
+ M5 should not inherit row-offset semantics silently.
