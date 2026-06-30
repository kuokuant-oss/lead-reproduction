# 2026-06-30 A2 electricity entry meter

## Slice

Issue: [#44](https://github.com/kuokuant-oss/lead-reproduction/issues/44)

Scope: make electricity the entry meter for the first Phase E transfer/FDD
within-context scoring path while keeping chilledwater support intact.

## What changed

+ Added ADR 0022. It records electricity as the entry meter and cross-references
  ADR 0019, ADR 0020, and ADR 0021.
+ Updated Step 4 transfer script parser defaults so electricity is the default
  meter and chilledwater remains an accepted choice.
+ Updated focused tests to assert electricity is the default and chilledwater is
  still accepted.
+ Updated `docs/plans/m5-plan.md`, `docs/plans/phaseE-fdd-roadmap.md`,
  `docs/plans/bdg2-fdd-eval-plan.md`, README, and unknown #25.
+ Reframed unknown #25 as an open per-site/per-meter weather-feature-validity
  caveat for electricity Level-3 evidence, not an electricity-wide disqualifier.

## Explicitly unchanged

+ Chilledwater support was not removed. It remains available in script choices
  and the loader, and is deferred to a later Level-3 weather-conditioned path.
+ No A4 raw-first wrapper and no unknown #27.
+ No A5 value-change regime decision.
+ No A3 Swan downgrade.
+ No M6 work, no full electricity scan, and no `docs/plans/m6-plan.md`.
+ No M3 numeric-line changes, no `load_m3_frame` default changes, no golden
  metric updates, no scaler/downsample changes, and no `lead.__all__` change.

## Next step

Stop for review. If approved, the next slice is A4: make the transfer/FDD
scoring path raw-first and open unknown #27.
