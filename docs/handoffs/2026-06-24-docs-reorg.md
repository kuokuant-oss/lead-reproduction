# Docs Reorganization Handoff

**Date**: 2026-06-24
**Scope**: Phase A housekeeping only. No code, model, data, or numeric behavior
changed.

## Move Map

Plans:

+ `docs/m1-plan.md` -> `docs/plans/m1-plan.md`
+ `docs/m2-plan.md` -> `docs/plans/m2-plan.md`
+ `docs/m3-plan.md` -> `docs/plans/m3-plan.md`
+ `docs/m4-plan.md` -> `docs/plans/m4-plan.md`

Reports:

+ `docs/reproduction-report.md` -> `docs/reports/reproduction-report.md`
+ `docs/m3-report.md` -> `docs/reports/m3-report.md`
+ `docs/m4-evaluation-report.md` -> `docs/reports/m4-evaluation-report.md`

Reference:

+ `docs/workflow.md` -> `docs/reference/workflow.md`
+ `docs/unknowns.md` -> `docs/reference/unknowns.md`
+ `docs/paper-notes.md` -> `docs/reference/paper-notes.md`
+ `docs/feature-engineering-rules.md` ->
  `docs/reference/feature-engineering-rules.md`
+ `docs/notebooks-map.md` -> `docs/reference/notebooks-map.md`

Metrics:

+ `docs/m3-50-50-ensemble.json` ->
  `docs/metrics/m3-50-50-ensemble.json`
+ `docs/m3-primary-use-auc.json` ->
  `docs/metrics/m3-primary-use-auc.json`

Unchanged:

+ `docs/adr/`
+ `docs/handoffs/`
+ `docs/agents/`
+ `docs/assets/`

## Reference Updates

Updated live references in:

+ `README.md`
+ `CONTEXT.md`
+ `tests/golden_metrics.json`
+ non-handoff files under `docs/`

The two `reported_source_file` provenance strings in
`tests/golden_metrics.json` now point to:

```text
docs/metrics/m3-50-50-ensemble.json
```

## README Rewrite

`README.md` was rewritten with Traditional Chinese as the primary narrative
language. The milestone sections now read chronologically from M1 through M4,
the stale structure tree was replaced with the current layout, and the ADR count
was corrected to 13 total ADRs while preserving the historical statement that
M1 produced ADR 0001-0006.

## Handoffs Policy

Pre-reorg files under `docs/handoffs/` intentionally still reference pre-reorg
paths. They are point-in-time records, so their bodies were not rewritten.

## Verification

Phase A verification performed before commit:

+ old top-level `docs/...` references were re-grepped in live files
+ non-handoff Markdown links were checked for existing file targets
+ `markdownlint-cli2` was run through pre-commit
+ `ruff` was run through pre-commit
+ `tests.test_refactor_regression` passed
+ `tests.test_call_arity` passed

## Suggested Skills

+ `handoff` for continuing Phase B context transfer
+ `diagnose` if Phase B label-join checks uncover row-order or key-integrity
  failures
+ `grill-with-docs` if ADR 0010 needs terminology or decision-status review
