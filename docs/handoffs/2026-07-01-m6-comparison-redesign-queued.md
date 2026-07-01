# M6 comparison redesign queue handoff

## Status

Queueing slice completed under GitHub issue
[#47](https://github.com/kuokuant-oss/lead-reproduction/issues/47).

## What changed

+ Added a queued M6 comparison redesign section to the Phase E to M6 roadmap.
+ Added a tracker row placing the redesign after A5 and A3 and before M6.1.
+ Updated the M5 plan to mention the queued redesign after Part A.

## What did not change

+ A5 and A3 were not started.
+ The current M6 ladder was not rewritten.
+ No M6-comparison ADR was added yet. That belongs to the queued redesign slice
  after A5 and A3 approval.
+ No code, scans, packets, tests, ADR status fields, unknown statuses, M3
  defaults, or public API surfaces changed.

## Future redesign brief

The future slice should compare GBDT and TabPFN on BDG2 instead of assigning
GBDT as scanner and TabPFN as auditor in advance. It should define independent
M6 phases for forecasting, synthetic injection, union event packets, and human
audit top-K, each with its own comparison basis and definition of done.

The future ADR must explicitly preserve: forecasting error is not fault; injected
metrics are synthetic-only; audit labels are triage judgments; no confirmed
fault status; no absolute cross-dataset top-K; BDG2-only and GEPIII-overlap
separation; raw-first scoring; unknown #25 and #27 caveats; frozen M3 numeric
line; frozen `lead.__all__`; TabPFN as research focus with license and latency
caveats.
