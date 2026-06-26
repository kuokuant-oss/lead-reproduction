# Handoff: M5 Phase D — retire premature BDG2-ingestion skeleton

**Date**: 2026-06-26
**Issue**: [#34](https://github.com/kuokuant-oss/lead-reproduction/issues/34)
**Supersedes**: Phase D slice 1 (#33), the BDG2 ingestion skeleton

## What changed

Retired the BDG2-ingestion detour from the Phase D line in one commit:

+ Removed `src/lead/bdg2.py`, `tests/test_bdg2_loader.py`, and the invented Fox
  smoke fixtures (`tests/fixtures/bdg2_fox_smoke.csv`,
  `tests/fixtures/bdg2_fox_smoke_labels.csv`). They used no real data — the
  fixtures and their labels were synthetic and the schema was guessed.
+ Rewrote the m5-plan Phase D prose: Phase D is no longer "docs-only" and no
  longer presents BDG2 ingestion as the current slice. Phase D is now a rigorous
  TabPFN-vs-GBDT comparison on the existing M3 GEPIII data.
+ Updated the README M5 row and repo-structure tree, and the m5-plan issue map.

The frozen `lead.__all__` public API is unchanged — `bdg2` was never exported,
so this removal is purely additive-in-reverse.

## Why

The data-scaling path is LEAD subset → full ASHRAE GEPIII (M3, done) → BDG2
(later). A guessed BDG2 loader with synthetic labels validates nothing and risks
anchoring later work on a wrong schema. Phase D's real job is depth on data we
already have, labels included: a foundation-model vs tree-model comparison on
GEPIII through the frozen pipeline.

## What is deferred

Real BDG2 ingestion — real download, real schema, real per-row label semantics —
is a later BDG2 scale-out milestone, to begin only after the GEPIII
foundation-vs-tree comparison lands (#35).

## Next

Issue [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35): build
the comparison harness on existing M3 GEPIII data and run the four-axis
TabPFN-vs-GBDT comparison (in-domain, site transfer, label scarcity, minimal
feature engineering), then write `docs/reports/m5-foundation-vs-gbdt.md`.
