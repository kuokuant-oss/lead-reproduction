# 2026-06-25 Phase A Governance Hardening

## Scope

Phase A restored repo governance before Phase B foundation-model planning. The
slice is tracked by GitHub issue #24 and is intended to close with `Closes #24`.

Phase B has not started. Its planning issue is #27.

## Changes

+ Added `docs/reference/change-checklist.md` as the authoritative close-out
  checklist for every slice.
+ Wired the checklist into `docs/reference/workflow.md`, `CLAUDE.md`, and the
  remaining M4 plan close-out criteria.
+ Added `tests/test_readme_freshness.py` to compare README M4/ADR claims against
  `docs/plans/m4-plan.md` and ADR status blocks.
+ Fixed README drift: M4.0-M4.3 are now shown complete, and ADR 0010/0011 are
  shown Accepted.
+ Updated the M4 issue tracker map honestly:
  + M4.0-M4.3 are marked done before issue policy restoration.
  + M4.4 is #25.
  + M4.5 is #26.
  + Phase A governance is #24.
+ Added ADR 0014 as Accepted.

## Red to green proof

Command:

```bash
uv run python -m unittest tests.test_readme_freshness
```

Initial stale README result:

```text
FAILED (failures=2)
README says M4.2/M4.3 are not executed, but the M4 plan marks them Done
README claims ADR 0010 is Proposed, actual status is Accepted
```

After README fix:

```text
Ran 2 tests in 0.001s
OK
```

Machine-readable local provenance is stored at
`data/processed/phase_a_governance_red_green.json`.

## CJK encoding check

`git diff -- README.md` showed only the intended M4 status, ADR status,
structure, command, and checklist-link lines changed. `git diff --
docs/reference/workflow.md` showed only the appended mandatory checklist
section. No encoding repair or mass CJK rewrite was performed.

## Suggested next skills

+ `handoff` for compacting Phase B context after the Phase A commit.
+ `to-issues` only if the Phase B plan expands into additional implementation
  tickets beyond issue #27.
+ `zoom-out` if M5 planning needs a broader architecture review before ADR 0015.
