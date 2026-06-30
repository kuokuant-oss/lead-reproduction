# 2026-06-30 Slice 0 Phase E FDD roadmap archive

## Slice

Issue: [#43](https://github.com/kuokuant-oss/lead-reproduction/issues/43)

Scope: archive the full Part A plus M6 roadmap as a tracked repo artifact before
starting A2.

## What changed

+ Added `docs/plans/phaseE-fdd-roadmap.md` as the authoritative living roadmap
  for clearing Phase E baggage and implementing M6.
+ Captured fixed constraints, Part A order (A1, A2, A4, A5, A3),
  BDG2-paper-derived folds, planned unknown #27, M6.1-M6.6, and a slice tracker.
+ Marked A1 as done with issue #42 and ADR 0021.
+ Added roadmap discovery pointers from README and `docs/plans/m5-plan.md`.

## Explicitly unchanged

+ Docs only: no code changes, no ADR status changes, and no behavior changes.
+ A2 was not started.
+ No A4 raw-first wrapper, no unknown #27, no A5 value-change decision, no A3
  Swan downgrade, no M6 implementation, and no `docs/plans/m6-plan.md`.
+ Transfer paradigm, frozen M3 numeric line, and TabPFN audit roles remain
  unchanged.

## Next step

Stop for review. After approval, execute A2 as its own issue, ADR, commit, and
handoff: make electricity the entry meter while preserving chilledwater as a
later Level-3 weather-conditioned path.
