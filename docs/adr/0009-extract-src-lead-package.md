# Extract `src/lead` package

## Status

Accepted (2026-06-24)

## Context

The M3 pipeline currently lives in notebooks and experiment scripts. Core
helpers are duplicated across tracked scripts: `load_m3_frame`,
`add_value_change_features`, and `downsample_indices` each have multiple local
copies. `src/lead/` exists but has no functional implementation, and no tracked
pipeline imports `lead`.

M5 will need to extend the pipeline to BDG2/FDD. That should build on importable
modules rather than notebook cells or script-to-script imports.

## Decision

Create an importable `src/lead/` package with modules for data loading,
features, splitting, sampling, evaluation, and provenance IO. M4.1 will move the
current M3 helper behavior into that package and refactor M3 scripts to import
from it.

## Rationale

This makes the pipeline testable and reusable while preserving the current M3
numeric line. The first extraction should avoid semantic fixes so any movement
in M3.2 or M3.4 can be attributed to refactor risk, not feature changes.

## Consequences

+ M3 scripts become thinner experiment runners.
+ Future M4 slices can change behavior behind explicit ADR and regression gates.
+ `src/lead` becomes the public API surface for M5.
+ Until M4.2/M4.3, the package preserves known label-join and row-offset
  value-change limitations.
