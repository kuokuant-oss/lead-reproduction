# Timestamp value-change features

## Status

Proposed

## Context

M3 value-change features use `groupby("building_id").shift(n)`. That gives an
`n`-row offset, not necessarily an `n`-hour timestamp offset when a building has
missing timestamps. The M2.4 test-side implementation used timestamp shifting
and merge semantics.

## Decision

M4.3 should add timestamp-merge value-change as an explicit regime while
keeping the current row-offset regime available for reproduction compatibility.

## Rationale

The current behavior is the code baseline and must remain reproducible. The
timestamp-merge behavior is likely the cleaner semantic model for downstream FDD
and BDG2, but it needs a measured AUC comparison before replacement.

## Consequences

+ M4.1 preserves row-offset features.
+ M4.3 resolves whether timestamp merge moves M3.2 beyond +/- `0.0005`.
+ Reports must name the value-change regime when discussing results.
+ M5 should not inherit row-offset semantics silently.
