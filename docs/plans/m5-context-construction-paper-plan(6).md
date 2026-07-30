# M5 137-feature meter-specific mechanism plan - synchronized snapshot (6)

**Snapshot date:** 2026-07-30
**Canonical source:** m5-context-construction-paper-plan.md

This snapshot records the canonical plan at the commit that introduced the
steam/chilledwater framing. The canonical plan remains the source of truth.

- Research question: meter-specific TabPFN gains for steam and chilledwater
  under the 137-feature representation.
- Steam Path A: hotwater support allocation is the intervention; steam
  cross-meter ordering and continuous margin are principal outcomes.
- Hotwater local readouts: manipulation checks and mechanism diagnostics.
- Chilledwater: CPU-only C1 localization first, then conditional boundary
  readouts only after a query-resolution audit.
- TabPFN 8.0.8: fixed scientific version with repeated-inference estimands.
- TabPFN 8.1.0: diagnostic only.
- Locked: 192-row query, Path B, tree refit, context-curve rerun, full-holdout
  refit, site transfer, and manuscript conclusions.
