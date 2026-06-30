# 2026-06-30 A1 powered-gate demotion

## Slice

Issue: [#42](https://github.com/kuokuant-oss/lead-reproduction/issues/42)

Scope: demote the Phase E powered BDG2-only sufficient-observation gate from a
blocking entry gate to an after-the-fact reporting-confidence label.

## What changed

+ Added ADR 0021. It supersedes only the powered-entry-gate clause of ADR 0019.
  The GEPIII-trained transfer paradigm, no-supervised-BDG2-metric rule,
  BDG2-only / GEPIII-overlap separation, and no absolute cross-dataset top-K
  rules remain in force.
+ Added `multi_building_transfer_stability` reporting metadata in
  `scripts/phaseE_transfer.py`. It reports whether a stratum reaches the prior
  5-building / 17,544-row bar, but it does not block within-context packets.
+ Updated Step 4a and Step 4c gates so score plumbing failures still stop, absent
  BDG2-only sufficient-observation evidence still stops, and one or more
  BDG2-only sufficient-observation buildings allow the
  `within_context_packet_path`.
+ Updated Step 4 transfer tests to replace the old "underpowered means stop"
  assertion with the new non-blocking single-building behavior.
+ Updated README and `docs/plans/m5-plan.md` so `underpowered_even_pooled` is a
  confidence measurement rather than a hard dead-end.

## Explicitly unchanged

+ No A2 electricity-entry decision.
+ No A4 raw-first wrapper, no unknown #27, and no raw/cleaned default change.
+ No A5 value-change regime decision.
+ No A3 Swan downgrade.
+ No M6 plan or implementation.
+ No M3 numeric-line changes, no `load_m3_frame` default changes, no golden
  metric updates, no scaler/downsample changes, and no `lead.__all__` change.

## Next step

Wait for review of the A1 diff. If approved, the next separate slice is A2:
choose electricity as the entry meter while preserving chilledwater as a later
weather-conditioned path.
