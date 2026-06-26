# ADR freshness fix handoff

## Scope

Docs-only housekeeping after M4 closure. Issue: #29.

## Changes

+ ADR 0012 status changed from `Proposed` to `Accepted (2026-06-26)` because
  the M4 evaluation protocol was enforced across M4.
+ ADR 0013 status changed from `Proposed` to `Accepted (2026-06-26)` because
  the +/- `0.0005` AUC noise-floor gate was enforced across M4.
+ ADR 0015 remains `Proposed` because the TabPFN decision belongs to M5.
+ README ADR count updated from 13 to 16 while preserving the historical M1
  sentence that M1 produced ADR 0001-0006.
+ `tests/test_readme_freshness.py` now checks that the README ADR count matches
  the number of Markdown files under `docs/adr/`.

## Close-out Notes

+ No code, model behavior, metric values, or `data/processed` JSON values were
  changed.
+ The CJK README edit was made as UTF-8 and checked with `git diff -- README.md`
  to confirm only the ADR-count line changed.
+ Full gate required before commit: unit tests, ruff, markdownlint, and
  pre-commit.
