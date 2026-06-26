# Handoff: M5 FDD framing fix, model-track close-out, and Phase E entry

**Date**: 2026-06-26
**Issue**: [#36](https://github.com/kuokuant-oss/lead-reproduction/issues/36)

## What changed (docs-only, one commit)

### 1. Framing fix — M5 is FDD, not a foundation-model benchmark

The milestone is **FDD on BDG2**. The GEPIII TabPFN-vs-GBDT four-axis comparison
is M5's **FDD model-selection stage (model track), now complete** — TabPFN is one
compared FDD model, not the goal and not an independent track. BDG2 transfer is
M5's **next stage, Phase E**, not a separate milestone. Aligned across:

+ README M5 row (scope → "FDD on BDG2"; status → "Model track (Phase A–D)
  complete; Phase E (BDG2) 規劃中").
+ m5-plan: model-track heading → "FDD model selection (TabPFN vs GBDT)", status
  line, and "deferred to a later milestone" → "next stage (Phase E)".
+ ADR 0015: added a framing note (FDD model-selection stage; Phase E next; GBDT is
  the real-time deployment candidate; TabPFN is a license/latency-bounded offline
  bootstrapper).
+ Report heading and unknown #21 wording → "Phase E (BDG2)".

Old session handoffs were left unchanged as historical record. `test_readme_freshness`
checks only M4 / ADR rows and stays green.

### 2. Model-track close-out

README and m5-plan mark the FDD model-selection stage (Phase A–D) complete, with
a close-out section summarising the outcome: GBDT is the real-time deployment
candidate; TabPFN is retained as an offline / label-scarce bootstrapper.

### 3. Report

`docs/reports/m5-foundation-vs-gbdt.md` is the single canonical M5 report, written
in Traditional Chinese. Its provenance is the result JSON
(`data/processed/m5_phaseD_foundation_vs_gbdt.json`, commit `8f4373b`). It is
linked once from README (milestone table and 主要文件). (Issue #37 later made the
report Chinese-only by removing the earlier English copy and promoting it to this
canonical filename.)

### 4. Phase E plan (docs-only)

m5-plan gains a "Phase E: FDD transfer to BDG2" slice with a gated order:
labels (#22) → evaluation-paradigm ADR (#23) → ingestion contract (#24, rebuild
the retired skeleton on real data, single owner) → transfer-eval contract
(0.9774 / 0.9833 are internal references; `site_id % 5 == 4` ≠ cross-dataset) →
causal discipline (PAST_SHIFTS-only, ADR 0007/0011) → roles/limits (TabPFN
license + latency; GBDT remains the real-time candidate). Unknowns #22–#24
registered.

## Phase E entry conditions

Phase E starts only when unknown #22 (BDG2 Fox per-row labels) is resolved and the
evaluation-paradigm ADR (#23) is recorded. No BDG2 download or ingestion code
until then. A Phase E issue is opened at that point.
