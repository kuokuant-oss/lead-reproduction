# A3 Swan chilledwater off critical path handoff

## Status

Slice A3 completed under GitHub issue
[#49](https://github.com/kuokuant-oss/lead-reproduction/issues/49).

## What Changed

+ Reclassified Swan chilledwater structural-missingness contiguity from a
  blocking gating task to optional future chilledwater work.
+ Marked A3 done and Part A complete in the Phase E to M6 roadmap.
+ Updated the M5 plan, BDG2 FDD eval plan, BDG2 EDA report wording, README, and
  the earlier BDG2 FDD evaluation handoff where Swan had been described as
  gating.

## ADR Note

No new ADR was added. A3 is plan housekeeping rather than a new architectural
decision. The reclassification follows:

+ ADR 0021: the powered chilledwater bar is reporting-confidence metadata, not a
  blocking entry gate.
+ ADR 0022: electricity is the first transfer/FDD entry meter, while chilledwater
  is deferred to later Level-3 weather-conditioned work.

## Preserved

+ The Swan EDA findings remain intact: Swan has about 20 BDG2-only chilledwater
  columns near the `0.50` missingness threshold, and those facts may still
  matter for later chilledwater work.
+ Unknowns #25, #26, and #27 were not resolved or changed.
+ No code, tests, M3 numeric-line behavior, public API, M6 comparison redesign,
  M6 ADR, scan, or M6 implementation was started.

## Next

Part A is complete. The next queued item is the M6 comparison redesign, but it
must not start until A3 is reviewed and approved.
