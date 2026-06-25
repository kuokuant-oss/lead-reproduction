# CLAUDE.md

## Agent skills

### Issue tracker

Issues live in GitHub Issues for this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage role labels without modification. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Auto-flow conventions

This repo uses pre-commit hooks + GitHub Actions for quality automation.
Agents working on this repo can assume:

- Trailing whitespace, EOF newline, basic YAML/JSON syntax are auto-fixed by hooks
  → don't pre-emptively fix these issues; let hooks do it
- Pre-commit will block bad commits → it's OK to attempt commit, see what fails
- GitHub Actions runs same checks on push → push only after local pre-commit passes
- See `.pre-commit-config.yaml` for the full hook list
- Every slice must satisfy `docs/reference/change-checklist.md` before commit

Permission philosophy (see `.claude/settings.local.json`):

- Read operations: auto-allowed (git status/diff/log, ls, cat, find, etc.)
- Test/verify operations: auto-allowed (uv run pre-commit, ruff, pytest)
- Git history / remote / file deletion: always confirm
- New dependencies / GitHub state changes: always confirm
