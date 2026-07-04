# Timestamp value-change features

## Status

Accepted (2026-06-25)

## Context

M3 value-change features use `groupby("building_id").shift(n)`. That gives an
`n`-row offset, not necessarily an `n`-hour timestamp offset when a building has
missing timestamps. The M2.4 test-side implementation used timestamp shifting
and merge semantics.

## Decision

M4.3 added timestamp-merge value-change as an explicit regime while keeping the
current row-offset regime available for reproduction compatibility.

As of the 2026-07-04 M4.1 re-baseline, `timestamp_merge` is the default and
canonical regime for the M3 reproduction line. It computes n-hour offsets by
joining on timestamp-shifted readings. When a `meter` column is present, the
timestamp merge includes `meter` in the join key so each output row remains
aligned with the original label row. `row_offset` and `row_offset_meter_aware`
remain available for historical ablation.

## Rationale

The row-offset behavior was the original code baseline and must remain
reproducible as an ablation. The measured timestamp-merge delta is within the
noise floor while matching the buds-lab timestamp-join semantics more faithfully,
so timestamp-merge is now the canonical default.

## Consequences

+ M4.1 now preserves timestamp-merge as the canonical value-change baseline.
+ M4.3 measured timestamp-merge M3.2 in the same harness as row-offset:
  row-offset AUC `0.9920119520500562`, timestamp-merge AUC
  `0.9924831086743003`, same-run Delta AUC `+0.00047115662424412896`.
+ The measured regime delta is within the +/- `0.0005` noise floor, so it does
  not indicate a material AUC-only regression from changing the default.
+ Reports must name the value-change regime when discussing results.
+ M5 should not inherit row-offset semantics silently.
