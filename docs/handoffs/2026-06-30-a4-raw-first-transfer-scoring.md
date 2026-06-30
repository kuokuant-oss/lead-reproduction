# A4 raw-first transfer scoring handoff

## Status

Slice A4 completed under GitHub issue
[#45](https://github.com/kuokuant-oss/lead-reproduction/issues/45).

## What changed

+ Added `load_bdg2_scoring_frame` in `scripts/phaseE_transfer.py`. It defaults
  the transfer/FDD scoring path to `variant="raw"` while delegating to the
  general `load_bdg2_frame` loader.
+ Routed Phase E Step 4a, Step 4b, and Step 4c through the raw-first scoring
  wrapper. Cleaned remains explicit for companion checks.
+ Added tests proving raw is the transfer scoring default, cleaned must be
  explicitly requested, Step 4 scripts route through the wrapper, and the
  general BDG2 loader tests still pass unchanged.
+ Added ADR 0023 recording raw-first BDG2 transfer/FDD scoring as a correctness
  precondition for M6.1.
+ Opened unknown #27 as a non-blocking GEPIII-to-BDG2 weather/unit regime caveat
  that must travel with M6 transfer outputs.
+ Updated the Phase E roadmap, M5 plan, and README status/count references.

## Constraints preserved

+ Transfer paradigm remains GEPIII-trained detector transfer to BDG2 under
  ADR 0019.
+ M3 numeric line is untouched: no `load_m3_frame`, golden metric,
  downsampling, scaler, or source-training changes.
+ `lead.__all__` is untouched because the wrapper stays in scripts.
+ TabPFN audit roles from ADR 0020 remain preserved and carried forward.

## Next

A5 value-change regime decision is the next queued Part A slice. Do not start
A5 until A4 is reviewed and approved. M6.1 is now unblocked on the raw-first
precondition but still must be opened and run as its own later slice.
