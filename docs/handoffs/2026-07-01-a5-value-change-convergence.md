# A5 value-change convergence handoff

## Status

Slice A5 completed under GitHub issue
[#48](https://github.com/kuokuant-oss/lead-reproduction/issues/48).

## Decision

ADR 0024 chooses the additive meter-aware-equivalent path. Future multi-meter
transfer must train and score through matching value-change semantics, using an
opt-in source path with `row_offset_meter_aware` paired with BDG2 target scoring
through the same regime.

The one-detector-per-meter alternative remains a fallback only if a later
implementation slice proves the additive aligned path cannot be wired without
moving the frozen M3 numeric line.

## Decided Now

+ Multi-meter transfer must not train with plain source `row_offset` and serve
  with target `row_offset_meter_aware`.
+ M3 `row_offset` remains the reproduction default.
+ M6.1 single-meter electricity is unaffected because Phase E Step 1 already
  proved single-meter equivalence.

## Deferred

+ No multi-meter transfer wiring was implemented in A5.
+ The exact source-training helper, script flag, detector artifact, and
  provenance schema are deferred until a later multi-meter slice needs them.
+ A3, the queued M6 comparison redesign, and all M6 implementation remain
  unstarted.

## Validation Expectation

A5 is decision-only, so no new behavior test is needed. Existing
`tests/test_value_change_regimes.py` covers default `row_offset` behavior and
the `row_offset_meter_aware` equivalence check.
