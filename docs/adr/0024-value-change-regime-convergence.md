# Align value-change regimes before multi-meter transfer

## Status

Accepted (2026-07-01)

## Context

ADR 0007 separates offline batch features from causal online features. ADR 0011
then keeps `row_offset` as the M3 reproduction default while adding
`timestamp_merge` as an opt-in value-change regime. ADR 0018 adds
`row_offset_meter_aware` for BDG2-style multi-meter rows while preserving the
M3 numeric line.

Phase E transfer currently has a train/serve value-change asymmetry: the GEPIII
source detector is trained under `row_offset`, while BDG2 scoring uses
`row_offset_meter_aware`. Phase E Step 1 proved row-by-row equivalence for
single-meter scoring: when only one meter type is present,
`row_offset_meter_aware` matches per-meter `row_offset` and does not change the
M3 semantics. The same equivalence breaks when one detector scores multiple
meter types, because plain `row_offset` can cross meter series inside a building
while `row_offset_meter_aware` cannot.

ADR 0019 fixes the transfer paradigm as GEPIII-trained detector transfer to
BDG2 with BDG2-only and GEPIII-overlap separated and no supervised BDG2 metric
claims. ADR 0023 makes BDG2 transfer/FDD scoring raw-first. A5 decides how
value-change semantics must converge before transfer expands beyond a
single-meter electricity path.

## Decision

Choose the additive meter-aware-equivalent path.

Future multi-meter transfer must train and score through matching value-change
semantics. The preferred implementation path is an additive opt-in source
training path that uses `row_offset_meter_aware` when the GEPIII source table
contains multiple meter types and includes a `meter` column, paired with BDG2
target scoring through the same regime. This keeps train/serve value-change
semantics aligned without moving the M3 reproduction default.

The one-detector-per-meter alternative is not chosen as the default decision.
It remains an allowed fallback only if a later implementation slice proves that
the additive source-side meter-aware path cannot be wired without moving the M3
numeric line or otherwise violating the fixed public API and regression gates.

Decided now:

+ Multi-meter transfer must not mix source `row_offset` training with target
  `row_offset_meter_aware` serving.
+ The convergence direction is additive opt-in meter-aware source and target
  scoring, not a change to M3 defaults.
+ M6.1 single-meter electricity is unaffected because Phase E Step 1 already
  proved single-meter equivalence.

Deferred to implementation:

+ the exact Phase E/M6 source-training helper or script flag that selects
  `row_offset_meter_aware`;
+ any multi-meter detector artifact and provenance schema;
+ whether a one-detector-per-meter fallback is needed for a specific later
  multi-meter run.

This does not change the transfer paradigm. GEPIII-trained detector transfer to
BDG2 remains fixed under ADR 0019.

This does not touch the M3 numeric line. `row_offset` remains the default
regime for the M3 reproduction path, and `load_m3_frame` defaults, M3.2/M3.4
golden values, downsampling, StandardScaler fitting, and existing M3 scripts
remain unchanged.

This preserves and carries TabPFN forward. ADR 0020's TabPFN audit roles remain
available, and this value-change decision applies to the shared feature
semantics a TabPFN comparison or audit path would consume.

## Rationale

The additive meter-aware-equivalent path directly resolves the train/serve skew
without fragmenting the detector family by meter type. It uses an existing
opt-in regime and existing tests: `row_offset` remains the default, and
`row_offset_meter_aware` already has a row-by-row equivalence test against
per-meter `row_offset`.

One-detector-per-meter would also avoid cross-meter row-offset leakage, but it
would make the default transfer design more fragmented before M6 has evidence
that such fragmentation is necessary. It is better kept as a fallback if the
additive aligned path cannot be implemented cleanly later.

Keeping this as a decision slice avoids premature multi-meter wiring. M6.1 is
single-meter electricity and remains safe under the existing equivalence proof.
The decision matters before later multi-meter transfer, chilledwater expansion,
or comparison work reuses one source detector across multiple meter types.

## Consequences

+ A5 adds no behavior change and no public API change.
+ Existing regime tests remain the guard for M3 default behavior and
  single-meter equivalence.
+ Later multi-meter Phase E/M6 work must either implement the additive
  meter-aware source/target path or explicitly invoke this ADR's fallback
  condition and choose one-detector-per-meter for that slice.
+ The M6.1 electricity scan remains single-meter and does not need new
  value-change wiring before it can run.
+ No unknown #25, #26, or #27 status changes are made by this ADR.
