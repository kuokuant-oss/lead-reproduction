# M5 C1 — 352-row screening query resolution audit

## Scope

Pre-fit resolution audit of the original 352-row screening query, required
before any chilledwater boundary readout can be added to a future pilot.

**No model was scored.** No fit, no inference. The frozen 192-row independent
query was not touched. This audit reports what the query can and cannot resolve;
it does not evaluate any learner on it.

## Thresholds (declared before reading results)

| Rule | Value |
| --- | --- |
| minimum valid positive-negative pairs | 100 |
| maximum share of a stratum from one building | 0.50 |

A continuous score margin is a feasibility readout only. It cannot substitute
for inadequate pair resolution.

## Stratum composition

All five required strata are present.

| Stratum | Rows | Buildings | Segments | Rows inside a segment | Top-building share |
| --- | ---: | ---: | ---: | ---: | ---: |
| chilledwater positive | 64 | 34 | 48 | 64 | 0.094 |
| chilledwater negative | 64 | 45 | 0 | 0 | 0.078 |
| electricity negative | 64 | 53 | 0 | 0 | 0.047 |
| steam negative | 32 | 21 | 0 | 0 | 0.125 |
| hotwater negative | 16 | 12 | 0 | 0 | 0.125 |

Negative strata have no segment membership by construction: segments are
anomaly episodes, and negative rows lie outside them. All 64 chilledwater
positives fall inside a segment, spread over 48 distinct segments — no single
episode dominates the positive stratum.

Building concentration is low everywhere: the most concentrated required stratum
is hotwater-negative at 0.125, far under the 0.50 limit.

## Pair resolution

| Comparison | Positives | Negatives | Valid pairs | AUC resolution | Adequate |
| --- | ---: | ---: | ---: | ---: | :-: |
| chilledwater-pos vs hotwater-neg | 64 | 16 | 1,024 | 9.77 × 10⁻⁴ | yes |
| chilledwater-pos vs steam-neg | 64 | 32 | 2,048 | 4.88 × 10⁻⁴ | yes |
| chilledwater-pos vs electricity-neg | 64 | 64 | 4,096 | 2.44 × 10⁻⁴ | yes |
| chilledwater-pos vs chilledwater-neg | 64 | 64 | 4,096 | 2.44 × 10⁻⁴ | yes |

`valid pairs` is the number of positive-negative row pairs a pairwise AUC can
use; `AUC resolution` is the smallest AUC increment those pairs can express.

## Which boundary readouts have adequate resolution

All four comparisons clear the declared thresholds. But clearing a threshold is
not the same as being able to resolve the effect of interest, and one comparison
must be flagged explicitly:

**chilledwater-positive vs hotwater-negative is the weakest of the four and is
close to unusable for the effect size E0 reports.** It has only 16 negative rows,
giving an AUC resolution of 9.77 × 10⁻⁴. The chilledwater ROC-AUC learner gap
measured on full holdout is 0.0043 — only **4.4 times** that resolution. A single
discordant pair moves the estimate by ~0.001, so a pilot readout on this stratum
could not distinguish the reported effect from zero with any confidence.

The other three comparisons resolve to 2.4-4.9 × 10⁻⁴, which is 9-18 times
finer than the same 0.0043 effect — usable, though still coarse.

### Consequence for a future pilot

- `chilledwater-positive vs chilledwater-negative` (within-meter) has the best
  resolution and is the comparison C1 finds carries the entire effect. It is the
  readout worth adding.
- `chilledwater-positive vs hotwater-negative` should **not** be relied on as a
  boundary readout at this query size. If that contrast matters to a future
  design, the hotwater-negative stratum needs more rows, not a continuous-margin
  substitute.
- Continuous score margins may accompany these readouts as feasibility
  indicators only, and must not be reported as if they resolved the pairwise
  question.

Full record: `data/processed/m5_chilledwater_c1/c1_query_resolution_audit.json`.
