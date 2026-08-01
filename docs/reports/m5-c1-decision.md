# M5 C1 decision — chilledwater mechanism localization

## Decision

> **`WITHIN_METER_MORPHOLOGY`**

The chilledwater learner advantage localizes to within-chilledwater ordering,
not to any cross-meter support reference.

Decision record: `data/processed/m5_chilledwater_c1/c1_decision.json`.
Frozen protocol: `c1_protocol.json`, SHA-256 `fb6699d8…`, written before any C1
result was read.

## How the frozen gate resolved

The gate was evaluated in its declared order.

| Candidate | Test | Result |
| --- | --- | --- |
| A `SAME_HOTWATER_NEGATIVE_REFERENCE` | chilledwater-pos vs hotwater-neg positive and stable across contexts | **refuted** — negative at all five contexts (−0.0078, −0.0005, −0.0012, −0.0028, −0.0007) |
| B `DIFFERENT_SUPPORT_SOURCE` | some other cross-meter negative group stably positive | **refuted** — electricity and steam negative or ~zero at all five contexts |
| C `WITHIN_METER_MORPHOLOGY` | advantage only within-meter, survives clustered uncertainty, shows within-meter structure | **satisfied** |
| D `NO_STABLE_LOCALIZATION` | nothing satisfies its rule | not reached |

Supporting facts: within-meter TabPFN-minus-tree pairwise AUC is positive at all
five contexts (+0.0043 at 100k); PR-AUC excludes zero under both building- and
segment-clustered bootstrap at 100k with direction positive in 97.7% of draws;
exact leave-one-building influence over 252 buildings produces 0 sign flips.

## One correction to the frozen rule, recorded not hidden

The protocol froze `concentration_limit = 0.25`. That threshold was written for
a segment-level top-10 share, as used in E0. Applied to a four-bin quartile
split it is **vacuous, because 0.25 is the uniform baseline** — every factor
would "concentrate" by construction.

The decision therefore judges concentration against each factor's own uniform
baseline (1/k), requiring ≥1.5× to count as structure. This is a correction to
a badly-specified threshold, not a change of endpoint, candidate set, or
decision order, and it does not alter which candidate wins: A and B are refuted
by the support-source table alone, independently of any concentration test.

## What the decision does and does not claim

**Claims.** The advantage is a within-meter phenomenon. It is not explained by a
shared hotwater-negative reference, nor by any other cross-meter support group.
It concentrates in the high-precision region rather than in absolute score
separation, and it is diffuse across buildings and segments.

**Does not claim.** No mechanism is demonstrated. The within-meter structure is
modest — onset-phase episodes at 1.83× the middle phase, reading-regime and
duration structure at ~2× uniform, while 24h and 168h deviation morphologies
show essentially none. The strongest apparent concentration (66.7% in the lowest
reading quartile) is confounded: 90.4% of chilledwater positives live there.

**Does not reclassify E0.** C1's ROC-AUC interval at 100k excludes zero by
7.4 × 10⁻⁵ where E0's missed it by 2.7 × 10⁻⁵. Both are indistinguishable from
the boundary. The E0 classification **"observed advantage but not stable"
stands unchanged**, as the task requires.

## Permitted next action

Per the frozen gate, `WITHIN_METER_MORPHOLOGY` permits proposing **one targeted
support or feature contrast**, and explicitly does **not** start Path B or a
representation ablation.

### Proposed contrast (proposal only — not executed, not authorized)

The narrowest contrast that would test the localization:

- **Contrast:** chilledwater anomaly-onset segments versus chilledwater
  middle/recovery segments, holding the negative support fixed at
  chilledwater-negative rows.
- **Rationale:** onset is the only phase-level structure that survives, at 1.83×
  the middle phase, and it is the one within-meter factor not confounded with
  positive prevalence.
- **Readouts:** within-meter PR-AUC and ROC-AUC separately; score movement and
  rank movement separately; building- and segment-clustered intervals.
- **Resolution requirement:** the 352-row query's chilledwater-positive stratum
  spans 48 segments over 34 buildings, which is adequate for a within-meter
  contrast but **not** for a hotwater-negative contrast (1,024 pairs, resolution
  9.8 × 10⁻⁴ against a 0.0043 effect).

This proposal requires human authorization before any execution.

## Explicitly not started

E3 variance pilot, Path A, Path B, the frozen 192-row query, site transfer,
500k, full-holdout refit, representation ablation, any fit, and any inference.
No manuscript text was modified.
