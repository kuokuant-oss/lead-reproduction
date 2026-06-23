# Handoff: Report/code consistency audit follow-up

**Date**: 2026-06-23
**Status**: Complete; pre-commit and count checks passed locally
**Scope**: Documentation/provenance fixes plus fail-fast feature-count assert;
no metric-changing code changes

## Summary

The audit found no true contradictions in reported numeric results. The only
substantive code-level inconsistency was the M2-vs-M3 value-change orientation:
both lines use the same 60-shift family, but M2 computes `shift(n) - current`
and `(shift(n)+1)/(current+1)`, while M3 computes `current - shift(n)` and
`(current+1)/(shift(n)+1)`. This was documented as tree-invariant rather than
changed in code.

## Fixes

| Task | Fix |
|---|---|
| M2-vs-M3 value-change direction | Added ADR 0008 and updated `docs/m3-report.md` §3.2 to say same shift family, opposite diff/ratio orientation. |
| M2 column-name mislabel | Updated `docs/reproduction-report.md` Ch3.1 table to use real M2 column name `lag_value_{n}` and note `lag_value_diff` is only a category key. |
| M3 50/50 ensemble provenance | Added committed `docs/m3-50-50-ensemble.json` with 725/724 split, 137/77 feature counts, and the reported offline/causal metrics; referenced it from `docs/m3-report.md` §2. |
| M3 Ch4 clarity | Added a Ch4 note that §4.1/§4.2 checks are 80/20 development evidence, not final 50/50 split diagnostics. |
| Recall caveat | Added caveat that M2 recall `81.2%` is on 50:50 downsampled validation while paper §3 `81.9%` is on the real distribution, so `Δ 0.7%` is indicative only. |
| ClusterNo ARI provenance | Softened the ARI=1.0 claim to state it was measured against external buds-lab ClusterNo reference labels, which are not committed in-repo. |
| Count guards | Added a fail-fast `170` feature-count assert to `scripts/run_m3_3_budslab.py`; existing M3 50/50 guards cover `137/77` and `60/30/30`. |

## Verification

Passed:

```powershell
uv run pre-commit run --all-files
uv run python -c "import run_m3_3_budslab as b, run_m3_50_50_ensemble as e; ..."
```

Count check output:

```text
run_m3_3_budslab shifts 60 30 30
run_m3_3_budslab features 170
run_m3_50_50 shifts 60 30 30
run_m3_50_50 features 137 77
```

No leaderboard probing was run.

## Suggested skills

- `handoff` if another agent needs to continue from this state.
- `diagnose` only if pre-commit or feature-count verification fails.

## 2026-06-23 prose-tightening note

Tightened only `docs/reproduction-report.md` and `docs/m3-report.md` to remove
duplicate caveats and repeated framing. Numeric token multisets were unchanged
for both reports before hooks; report diff before this note was net `-24` lines.
