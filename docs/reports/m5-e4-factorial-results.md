# M5 E4 — factorial results

All numbers recomputed from the raw per-repeat score vectors. Point estimates
are equal-weight averages over the three pre-specified context seeds; intervals
are percentile 95% intervals over 1,000 addressable clustered draws.

The four sources of variation are kept apart throughout, because they answer
different questions.

## 1. Inference variation (the 8 repeats of one fit)

| | steam AUC | steam margin |
|---|---:|---:|
| min half-width | 0.000000 | 0.001070 |
| median half-width | 0.005826 | 0.003965 |
| max half-width | 0.022683 | 0.009470 |

All 192 repeats produced distinct score digests — 8 of 8 in every fit. Repeated
inference on a fitted TabPFN 8.0.8 state is not bitwise reproducible, confirming
E3 at 24× the scale.

**Six fits have exactly zero AUC variance**: all three seeds × both arms of
cell 01, where the steam AUC saturates at 1.000000. A half-width of zero there
is saturation, not precision. This is why the margin is a co-primary and why
every conclusion below is required to hold on both endpoints.

## 2. Cell means (steam AUC, TabPFN, cell-specific arm)

| Seed | 00 | 01 | 10 | 11 |
|---|---:|---:|---:|---:|
| 42 | 0.8276 | **1.0000** | 0.4756 | 0.9712 |
| 123 | 0.7104 | **1.0000** | 0.3823 | 0.9619 |
| 999 | 0.7087 | **1.0000** | 0.3301 | 0.9727 |

Cell 10 sits below 0.5 in every seed — with hotwater-positive support present
and hotwater-negative support absent, steam positives rank *below* hotwater
negatives.

## 3. Primary steam contrasts (TabPFN, cell-specific arm)

| Effect | Endpoint | Overall | Per seed (42 / 123 / 999) | SD | Sign | Building CI | Segment CI |
|---|---|---:|---|---:|---|---|---|
| **negative-support main** | AUC | **+0.4118** | +0.334 / +0.435 / +0.467 | 0.069 | 3/3 + | [+0.220, +0.570] ✓ | [+0.259, +0.572] ✓ |
| | margin | **+0.6312** | +0.607 / +0.650 / +0.637 | 0.022 | 3/3 + | [+0.537, +0.708] ✓ | [+0.532, +0.721] ✓ |
| **positive-support main** | AUC | **−0.1922** | −0.190 / −0.183 / −0.203 | 0.010 | 3/3 − | [−0.258, −0.116] ✓ | [−0.250, −0.132] ✓ |
| | margin | **−0.2785** | −0.261 / −0.282 / −0.293 | 0.016 | 3/3 − | [−0.338, −0.232] ✓ | [−0.336, −0.220] ✓ |
| **positive × negative** | AUC | +0.3215 | +0.323 / +0.290 / +0.351 | 0.031 | 3/3 + | [+0.188, +0.449] ✓ | [+0.211, +0.429] ✓ |
| | margin | +0.0978 | +0.080 / +0.104 / +0.110 | 0.016 | 3/3 + | [−0.208, +0.374] ✗ | [−0.170, +0.343] ✗ |

✓ = excludes zero. The `frozen_reference` arm reproduces all six rows to within
0.008 on the AUC and 0.002 on the margin.

**The interaction is the one place the two primary endpoints disagree.** The AUC
interval excludes zero on both clusterings; the margin interval does not, on
either. Reporting only the AUC would have produced a confident interaction claim
that the margin does not support — which is exactly the failure mode the
co-primary requirement exists to prevent.

## 4. Against the matched fixed tree

| Effect | Endpoint | TabPFN | Tree | Gap | Building CI on the gap | Segment CI |
|---|---|---:|---:|---:|---|---|
| negative-support main | AUC | +0.4118 | +0.2881 | **+0.1237** | [+0.033, +0.197] ✓ | [+0.057, +0.191] ✓ |
| | margin | +0.6312 | +0.4790 | **+0.1522** | [+0.107, +0.196] ✓ | [+0.108, +0.200] ✓ |
| positive-support main | AUC | −0.1922 | −0.2360 | +0.0438 | [−0.032, +0.117] ✗ | [−0.035, +0.132] ✗ |
| | margin | −0.2785 | −0.3132 | +0.0347 | [−0.032, +0.088] ✗ | [−0.031, +0.094] ✗ |
| positive × negative | AUC | +0.3215 | +0.4212 | −0.0997 | [−0.270, +0.063] ✗ | [−0.284, +0.068] ✗ |
| | margin | +0.0978 | +0.0188 | +0.0790 | [−0.046, +0.214] ✗ | [−0.039, +0.200] ✗ |

The matched tree responds to composition too, and for the positive-support main
effect it responds *more* than TabPFN. Only the **negative-support main effect**
shows a gap that clears zero on both endpoints, both clusterings and 3/3 seeds.

The interaction gap has **opposite signs on the two endpoints** (−0.100 on AUC,
+0.079 on margin), neither excluding zero.

## 5. Scaler arm

The scaler-arm interaction excludes zero **nowhere** — not for any effect, any
primary endpoint, or either clustering. Magnitudes are 0.000–0.007 against main
effects of 0.19–0.63.

Cell 11 supplies an unplanned null control here. Its `frozen_reference` scaler
is fitted on cell 11's own rows, so both arms transform identical data with
identical statistics; any arm difference at cell 11 is pure noise. Measured:

| Seed | states differ | `ensemble_configs_` equal | mean \|score difference\| |
|---|---|---|---:|
| 42 | yes | yes | 1.73e−03 |
| 123 | yes | yes | 2.32e−03 |
| 999 | yes | yes | 2.49e−03 |

Two things follow. First, **TabPFN's fit is not bitwise deterministic either** —
identical inputs, identical seed, identical ensemble configuration, different
persisted state. E3 established this for inference; this establishes it for the
fit. Second, the arm-to-arm difference at cell 11 (1.7–2.5e−03) is *smaller*
than the within-fit repeat spread (3.74e−03), so the two arms there behave as
two draws of one quantity — which is what the scaler axis should look like when
it carries no signal.

## 6. Chilledwater secondary

Not pooled into the steam claim, and reported only as a secondary readout.

| Endpoint | Effect | Overall | Building | Segment |
|---|---|---:|---|---|
| within-meter AUC | positive main | −0.00020 | ✗ | ✗ |
| | negative main | −0.00008 | ✗ | ✗ |
| | interaction | −0.00033 | ✗ | ✗ |
| within-meter margin | positive main | −0.00356 | ✗ | ✗ |
| | negative main | −0.00096 | ✗ | ✗ |
| | interaction | +0.00742 | ✓ | ✗ |

One building-clustered interval excludes zero, and its segment-clustered
counterpart does not. A result that survives one clustering and not the other is
not established. Everything here is two to three orders of magnitude below the
steam effects.

`chilledwater_positive_vs_hotwater_negative` remains
`RESOLUTION_LIMITED_DIAGNOSTIC` and onset/middle/recovery remains
`UNRESOLVED_NOT_EXECUTED`. Neither carries a mechanism conclusion.

## 7. Coverage and provenance

| | |
|---|---|
| fits | 24 / 24 |
| same-process repeats | 192 / 192 |
| distinct fit states | 24 |
| distinct process UUIDs | 24 |
| effective `n_estimators_` | {8} across all 24, re-read from the persisted states |
| clustered draws valid | 1000/1000 in all eight bootstraps |
| clusters | 142 building, 352 segment |
| stderr | 0 bytes |
| interrupted / stray temp files | 0 / 0 |

## 8. What these numbers do not license

The clustered intervals are conditional on **these 24 fitted states and these
three context seeds**. They do not contain model-seed variation or fresh-fit
variation, neither of which E4 executed. An interval here answers "how much does
this contrast move when the query rows are resampled", not "how much would it
move under a different fit".
