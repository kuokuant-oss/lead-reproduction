# Change-management checklist and README freshness gate

## Status

Accepted

## Context

The repo already had an issue-opening rule in `docs/agents/issue-tracker.md`,
`CLAUDE.md`, and `docs/reference/workflow.md`. M2 followed that rule with
issues #8-#12, but by M4 the rule had decayed: M4.0-M4.3 were complete while
the M4 issue tracker map still showed `TBD`.

README freshness had a separate gap. No rule required README or related status
documents to be updated when milestone state changed. As a result, README
claimed M4.2 and M4.3 were not executed and ADR 0010/0011 were Proposed even
after the M4 plan and ADR files marked those slices and decisions complete.

The root cause is that rules living only in prose decay unless they are embedded
in the slice checklist and protected by a machine-checked gate.

## Decision

Create `docs/reference/change-checklist.md` as the authoritative close-out
checklist for every slice. Wire it into the workflow, agent instructions, and
remaining M4 plan close-out criteria. The checklist requires issue tracking,
README updates when status or structure changes, plan tracker updates, ADR
updates, handoff writing, provenance placement, CJK UTF-8 diff review, and the
local verification gate before commit.

Add `tests/test_readme_freshness.py` as a machine-checked guard. The test parses
README status claims for M4 and ADR 0010/0011, compares them with
`docs/plans/m4-plan.md` and the ADR status blocks, and fails when README drifts.

Backfill issue history honestly. Work completed before the restored issue
policy is recorded as completed before issue tracking was restored, not assigned
retroactive fabricated issues. Upcoming work gets issues before implementation.

## Rationale

README and tracker drift are governance failures, not modeling failures. The
right fix is to move close-out rules into the workflow path agents actually
execute and add a failing test for the drift that already occurred.

The README guard is intentionally narrow. It protects the current decay mode
without turning README into a full duplicate of every plan and ADR.

## Consequences

+ Future slices must open and close a GitHub issue through the checklist.
+ Changes affecting status, structure, or commands must update README before
  commit.
+ README cannot claim M4.2/M4.3 or ADR 0010/0011 are stale without failing the
  test suite.
+ Historical M4.0-M4.3 issue gaps remain visible as a provenance fact instead
  of being rewritten.
+ Phase B and later M5 planning must use this checklist before commit.
