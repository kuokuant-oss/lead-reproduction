# Value-change direction differs between M2 and M3

## Status

Accepted (2026-06-23)

## Context

M2 and M3 use the same 60-shift value-change family:

- `-24..-1`
- `1..24`
- `-168..-48 step 24`
- `48..168 step 24`

The direction of the derived value-change columns differs:

| Line | Difference column | Ratio column |
|---|---|---|
| M2 | `lag_value_{n} = shift(n) - meter_reading` | `lag_value_ratio_{n} = (shift(n) + 1) / (meter_reading + 1)` |
| M3 | `lag_value_diff_{n} = meter_reading - shift(n)` | `lag_value_ratio_{n} = (meter_reading + 1) / (shift(n) + 1)` |

The M2 formula appears in `notebooks/05-m2-integration.ipynb` Cell 3 and in the
M2 follow-up scripts `notebooks/ablations_m25.py` and
`notebooks/submission_m25.py`. The M3 formula appears in the M3 runners and in
`notebooks/06-m3-baseline.ipynb`.

## Decision

Keep the code as-is and document the difference. M2 and M3 share the same shift
coverage, but the difference feature is sign-flipped and the ratio feature is
reciprocal-oriented.

## Consequences

Reported metrics are unaffected. The models used for the reported M2/M3 results
are tree-based GBDT models, and M2 Lesson #1 established that tree models are
invariant to monotonic feature transformations because splits depend on feature
ordering, not absolute scale. Negating a difference reverses its ordering, and
taking a positive reciprocal reverses the ratio ordering; both are monotonic
one-to-one transformations for the observed feature values. They therefore do
not change the information available to the tree model or the reported AUC,
precision, recall, or F1 results.

Changing one line to match the other would add unnecessary risk without changing
the reported metrics. Future reports should say that the shift family is shared,
not that the exact diff/ratio orientation is identical.

## References

- `notebooks/05-m2-integration.ipynb` Cell 3
- `notebooks/ablations_m25.py`
- `notebooks/submission_m25.py`
- `scripts/run_m3_3_budslab.py`
- `scripts/run_m3_4_ensemble.py`
- `scripts/run_m3_50_50_ensemble.py`
- `scripts/run_m3_split_causality.py`
- `notebooks/06-m3-baseline.ipynb`
- M2 Lesson #1 in `docs/reproduction-report.md`
