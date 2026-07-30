# M5 context-mechanism handoff

**Date:** 2026-07-30
**Repository:** lead-reproduction
**Branch:** m5-tabpfn-repro-audit
**HEAD before this documentation commit:** ed66e799db86cf4a89212a923ed583734b4ea629

## Completed evidence

- Existing full-holdout 5k-100k analysis (E1) found rank improvement for both
  learners, anomaly within-meter rank increase in all meters, hotwater 0-1
  normal score decrease larger than anomaly decrease, and a large
  steam-positive versus hotwater-negative movement.
- Building influence did not support one-building domination in the completed
  CPU analysis.
- Original Path-A recovery persisted TabPFN states and tree ensembles; tree
  recovery/reload is bit-exact.
- TabPFN 8.0.8 low_memory and fit_preprocessors lifecycle diagnostics show
  live-repeat variation before save; same/fresh reload does not remove it.
- Isolated 8.1.0 diagnostic R1-R3 also shows live-repeat variation and is
  excluded from science.

## Decisions now fixed

- Scientific question: meter-specific TabPFN response for steam and
  chilledwater under 137 temporal features.
- TabPFN 8.0.8 is the scientific version. 8.1.0 is diagnostic only.
- TabPFN is analysed with Y[c,s,a,r] = mu[c,s,a] + epsilon[c,s,a,r].
- Steam is the principal Path-A outcome; hotwater is the support-allocation
  lever and its local readouts are diagnostics.
- Chilledwater first receives CPU-only C1 localization, not a new intervention.
- Electricity is a counterexample; trees are matched-row comparators.
- The 192-row query is frozen and unscored. Path B is paused.

## Planned but unauthorized

1. E0 meter-specific learner-gap CPU analysis from existing predictions.
2. C1 chilledwater localization from existing row/segment/prediction artifacts.
3. Chilledwater query-resolution audit and endpoint freeze.
4. Only then, after explicit authorization, the 8.0.8 four-cell variance
   pilot (one fit/cell, repeated inference).
5. Independent 192-row replication remains locked until the protocol is frozen.

## Hard prohibitions

No 8.1.0 science; no model fit or inference replicate in this handoff round; no
192-row scoring; no 24-cell grid; no Path B; no tree refit; no context-curve or
full-holdout refit; no site transfer; no paper-manuscript change.

## Long-running execution policy

All future research execution must follow the repository-wide
[`long-running research execution policy`](../policies/long-running-research-execution.md):
**NO CHECKPOINT, NO LAUNCH**, no automatic timeout, atomic checkpoint/resume
with provenance validation, and an explicit human authorization before a formal
run.

## Artifact map

- Canonical plan: docs/plans/m5-context-construction-paper-plan.md
- C1 specification: docs/plans/m5-chilledwater-mechanism-localization.md
- E1 report: docs/reports/m5-137-context-mechanism-analysis.md
- Execution diagnostic: docs/reports/m5-tabpfn-deterministic-execution-audit.md
- Repeated-inference policy: docs/reports/m5-tabpfn-repeated-inference-policy.json
- Ignored artifacts: data/processed/m5_hotwater_label_factorial/ and
  data/processed/m5_context_mechanism_137/; do not commit predictions, states,
  checkpoints, or query-score artifacts.

## Open questions

- Does E0 show a stable steam/chilledwater learner gap under clustered
  uncertainty and positive-ranking checks?
- Does C1 support the hotwater-negative reference, a distinct support source,
  within-meter morphology, or no localization?
- Are the original-query chilledwater strata sufficiently resolved for future
  boundary readouts?

## Formal E0 execution status — Tranche 1

Completed evidence: Stage I framework commit
`28815776ad735db624755bcb08c5676bd18dfd60` and formal tranche implementation
HEAD `d8e59da2c40cb5102367d6a73299e807680f6ca6`.  Formal preflight passed with
ten inputs and a 4,000-unit manifest.  Identity is 10/10 and base metrics are
40/40, both completed.  Bootstrap is partially checkpointed at 168/4,000:
draw IDs 0–41 for each of four meters, with 3,832 pending.  Elapsed execution
time was 1:16:38.  There is no bootstrap completion marker, no active research
Python process, and no LOO, segment, finalization, or scientific
interpretation.  The exact resume point is draw ID 42 per meter under the same
formal root, manifest, seed mapping, and provenance; it requires explicit
future authorization.  This partial bootstrap is not completed E0.
