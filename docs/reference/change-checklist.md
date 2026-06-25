# Change close-out checklist

Every slice must satisfy this checklist before commit. The checklist is the
authoritative close-out gate for repo changes; milestone plans may add stricter
slice-specific criteria, but they do not replace this list.

## Required checks

+ Open a GitHub issue at slice start and record it in the relevant tracker.
+ Close the slice with `Closes #N` in the commit message.
+ Update README when the change affects milestone status, repo structure,
  commands, public workflow, or current project status.
+ Update the relevant plan tracker under `docs/plans/`.
+ Update ADR status or add a new ADR when the change records a decision.
+ Write a handoff under `docs/handoffs/`.
+ Put result or provenance JSON under `data/processed/` unless it is a tracked
  report fixture; do not leave loose result JSON in `docs/` root.
+ For CJK documents, read and write UTF-8, then inspect `git diff` and confirm
  only intended lines changed. Do not repair encoding as part of unrelated work.
+ Run the repo verification gate before commit: tests, `ruff`, markdownlint,
  and `pre-commit run --all-files`.

## Backfill policy

Do not fabricate retroactive issue history. Work completed before this policy
was restored should be marked honestly as completed before issue tracking was
restored, with a pointer to this checklist.
