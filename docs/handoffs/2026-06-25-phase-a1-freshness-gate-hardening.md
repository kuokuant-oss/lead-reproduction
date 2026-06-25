# 2026-06-25 Phase A.1 Freshness Gate Hardening

## Scope

Phase A.1 hardened the README freshness gate before Phase B foundation-model
planning. The slice is tracked by GitHub issue #28 and should close with
`Closes #28`.

Phase B has not started.

## Changes

+ Replaced README M4 milestone freshness magic-string checks with parsers for:
  + the README milestone table M4 status cell;
  + the README M4 section completed-through claim;
  + M4.x status blocks in `docs/plans/m4-plan.md`.
+ Added a guard for the M4 tracker governance row so issue #24 cannot remain
  `In progress` or `TBD` after closure.
+ Fixed the actual M4 plan drift: `M4 governance hardening` is now `Done`.

## Red to green proof

Old false-green proof before hardening:

```text
Temporary edit:
README M4 status-table cell = M4.1 complete

uv run python -m unittest tests.test_readme_freshness
Ran 2 tests in 0.001s
OK
```

New tracker guard red before fixing the plan row:

```text
FAILED (failures=1)
AssertionError: 'In progress' != 'Done'
 : M4 governance issue #24 is closed; tracker must be Done
```

New status-cell red proof after hardening:

```text
Temporary edit:
README M4 status-table cell = M4.1 complete

FAILED (failures=1)
AssertionError: 'M4.1 complete' != 'M4.0-M4.3 complete'
 : README M4 milestone table status is stale
```

Green after restoring README and fixing the plan row:

```text
uv run python -m unittest tests.test_readme_freshness
Ran 3 tests in 0.002s
OK
```

## CJK encoding check

Temporary README edits were restored before commit. `git diff -- README.md`
shows no persistent README changes. `git diff -- docs/plans/m4-plan.md` shows
only the governance tracker row changing from `In progress` to `Done`.

## Suggested next skills

+ `handoff` after Phase B is committed and green.
+ `to-issues` only if Phase B planning discovers additional independently
  grabbable M5 tasks beyond issue #27.
