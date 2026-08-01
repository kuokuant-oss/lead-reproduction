# M5 C1 — chilledwater mechanism localization

## Question

Why does chilledwater show a stable PR-AUC advantage at 100k context while its
ROC-AUC bootstrap interval very nearly includes zero?

CPU-only localization over existing artifacts. No fit, no refit, no TabPFN
inference, no tree refit, no scoring of the frozen 192-row query. Protocol frozen
before any result was read (SHA-256 `fb6699d8…`), fixing the four candidate
explanations, the analysis dimensions and the decision rule.

236 atomic units: 5 movement decompositions, 20 support-source comparisons,
11 morphology stratifications, 200 clustered-bootstrap blocks.

## 1. The PR/ROC disagreement is a prevalence-and-saturation effect

Chilledwater at 100k, within-meter (2,115,354 rows; 141,139 anomaly, 6.7%
prevalence):

| Quantity | TabPFN | Tree | Difference |
| --- | ---: | ---: | ---: |
| within-meter PR-AUC | 0.825149 | 0.776532 | **+0.048618** |
| within-meter ROC-AUC | 0.984294 | 0.980010 | **+0.004285** |
| score separation (mean pos − mean neg) | 0.851837 | 0.850688 | +0.001149 |

The PR-AUC gap is **11 times** the ROC-AUC gap. That ratio is not a mechanism
difference — it follows from where each metric has headroom. ROC-AUC is already
saturated at 0.98, so a real improvement in the high-precision region moves it
by ~0.004 while moving PR-AUC by ~0.049.

Critically, **the absolute score separation is essentially identical**
(+0.0011). The advantage is therefore *not* a calibration or absolute-score
effect: TabPFN and the tree place positives and negatives at nearly the same
average distance apart. What differs is the ordering inside the high-precision
region, which is exactly what PR-AUC weights and ROC-AUC does not.

## 2. The advantage exists only against chilledwater's own negatives

TabPFN-minus-tree pairwise AUC of chilledwater-positive against each negative
reference group, at every context:

| Context | vs **chilledwater**-neg | vs electricity-neg | vs **hotwater**-neg | vs steam-neg |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | **+0.008399** | −0.000609 | −0.007784 | −0.002616 |
| 10,000 | **+0.003207** | −0.000253 | −0.000472 | −0.001115 |
| 20,000 | **+0.005009** | +0.000034 | −0.001177 | −0.001800 |
| 50,000 | **+0.003395** | −0.000152 | −0.002763 | −0.001952 |
| 100,000 | **+0.004285** | −0.000172 | −0.000734 | −0.001252 |

**This refutes the shared-hotwater-negative-reference hypothesis directly.**
Chilledwater-positive versus hotwater-negative is negative at every one of the
five contexts. It does not move in the same stable direction as steam does; it
moves the other way.

It equally refutes a different cross-meter support source: electricity-negative
and steam-negative are also negative or indistinguishable from zero at every
context. No cross-meter reference group explains the movement.

The only positive column is the within-meter one. Consistency check: the
within-meter pairwise AUC gap at 100k is **+0.004285**, numerically identical to
the E0 within-meter ROC-AUC gap — as it must be, since the pairwise AUC of
positives against same-meter negatives *is* the within-meter ROC-AUC. Two
independent code paths produce the same number.

## 3. Where the within-meter movement comes from

Score movement from the 5k reference context to 100k, chilledwater rows:

| | TabPFN | Tree | TabPFN − tree |
| --- | ---: | ---: | ---: |
| anomaly rows | −0.012541 | −0.020577 | **+0.008036** |
| normal rows | −0.021370 | −0.011423 | **−0.009947** |

As context grows, TabPFN holds anomaly scores **up** relative to the tree
(+0.0080) and pushes normal scores **down** relative to the tree (−0.0099).
Both movements improve within-meter separation, and they are of similar
magnitude — the effect is not carried by one side alone.

### Score movement and rank movement disagree in sign

Reported separately, never pooled:

| Rank quantity (100k) | TabPFN | Tree | TabPFN − tree |
| --- | ---: | ---: | ---: |
| anomaly global rank movement | +0.002505 | +0.001171 | +0.001333 |
| anomaly within-meter rank movement | +0.005308 | +0.009148 | **−0.003840** |
| normal global rank movement | −0.036470 | +0.031667 | **−0.068137** |
| normal within-meter rank movement | −0.000380 | −0.000654 | +0.000274 |

Two things must be said plainly:

- The **within-meter anomaly rank gap is negative** (−0.0038) while the
  within-meter PR-AUC gap is positive (+0.0486). The tree lifts chilledwater
  anomalies further in average within-meter rank; TabPFN nevertheless orders the
  high-precision region better. Average rank and PR-AUC are not
  interchangeable, and any claim resting on "TabPFN ranks anomalies higher"
  would be wrong as stated.
- The largest single number in the table is a **normal-side cross-meter effect**:
  TabPFN moves chilledwater normal rows *down* the global ranking (−0.0365)
  while the tree moves them *up* (+0.0317). This is a real cross-meter
  behavioural difference, but it does not produce a positive cross-meter
  pairwise AUC for chilledwater positives (section 2), so it is not the
  mechanism the advantage rests on.

## 4. Morphology localization

Share of absolute learner-gap movement held by the largest stratum, judged
against each factor's own uniform baseline (1/k):

| Factor | Strata | Top share | Uniform baseline | Multiple |
| --- | ---: | ---: | ---: | ---: |
| raw reading quartile | 4 | 0.667 | 0.250 | 2.7× |
| reading regime | 3 | 0.667 | 0.333 | 2.0× |
| ratio morphology | 3 | 0.657 | 0.333 | 2.0× |
| duration | 4 | 0.550 | 0.250 | 2.2× |
| slope | 2 | 0.785 | 0.500 | 1.6× |
| diff morphology | 4 | 0.305 | 0.250 | 1.2× |
| 24h deviation | 4 | 0.270 | 0.250 | 1.1× |
| 168h deviation | 4 | 0.272 | 0.250 | 1.1× |
| building | 252 | 0.036 | 0.004 | 9.1× |
| segment | 2,604 | top-10 = 0.030 | — | diffuse |

Anomaly phase, mean learner-gap movement: **onset +0.035044**, recovery
+0.019645, middle +0.019195 — onset is **1.83×** the middle phase.

### What this does and does not establish

- **The effect is diffuse across buildings and segments.** The most influential
  of 252 buildings holds 3.6% of absolute movement; the top 10 of 2,604 segments
  hold 3.0% and the top 50 hold 12%. No handful of buildings or episodes carries
  it. The 9.1× building multiple is an artefact of a 252-bin baseline, not
  concentration — in absolute terms 3.6% is diffuse, and it is reported here so
  the multiple is not misread.
- **The 24h and 168h deviation morphologies show no structure at all** (1.1×,
  i.e. essentially uniform), and diff morphology is close to none.
- **The reading-level concentration is confounded.** Quartile q1 holds 66.7% of
  absolute movement, but 127,629 of the 141,139 chilledwater positives (90.4%)
  are in q1. That concentration is mostly a statement about where the labels
  are, not an independent morphology finding, and it should not be quoted as
  evidence of a low-reading mechanism.
- What survives as genuine within-meter structure is therefore **modest**:
  onset-phase episodes at 1.83× the middle phase, and reading-regime / duration
  / ratio structure at ~2× uniform.

## 5. Robustness

Clustered bootstrap, 1,000 draws each, rows never treated as independent:

| | Building-clustered (252 clusters) | Segment-clustered (460 clusters) |
| --- | --- | --- |
| PR gap 10k | +0.0266 [−0.0173, +0.0703] | +0.0252 [−0.0192, +0.0702] |
| ROC gap 10k | +0.0032 [−0.0006, +0.0080] | +0.0031 [−0.0008, +0.0079] |
| PR gap 20k | +0.0518 [+0.0104, +0.0987] ✔ | +0.0503 [+0.0085, +0.1018] ✔ |
| ROC gap 20k | +0.0050 [+0.0010, +0.0106] ✔ | +0.0048 [+0.0008, +0.0100] ✔ |
| PR gap 50k | +0.0434 [+0.0050, +0.0882] ✔ | +0.0427 [+0.0035, +0.0929] ✔ |
| ROC gap 50k | +0.0034 [−0.0003, +0.0083] | +0.0034 [−0.0004, +0.0082] |
| PR gap 100k | +0.0467 [+0.0016, +0.1022] ✔ | +0.0481 [+0.0008, +0.1030] ✔ |
| ROC gap 100k | +0.0042 [+0.0001, +0.0098] ✔ | +0.0044 [+0.0000, +0.0097] ✔ |

✔ = interval excludes zero. Direction is positive in 94-99% of draws at every
context and under both clusterings; 0 invalid draws.

**PR-AUC is robust at 20k, 50k and 100k under both clusterings. ROC-AUC excludes
zero at 20k and 100k but not at 10k or 50k** — the direction is consistent but
interval exclusion is not monotone in context, which is itself a sign that the
ROC-AUC estimate sits near its resolution limit.

Exact leave-one-building influence is read from the completed E0 LOO phase (the
identical estimand, 252 buildings, not recomputed): **0 sign flips** on PR-AUC,
ROC-AUC and the positive-rank endpoint, with the LOO range for PR-AUC spanning
+0.0376 to +0.0561 around a full estimate of +0.0486.

### On the ROC-AUC boundary

C1's building-clustered ROC-AUC interval at 100k excludes zero by 7.4 × 10⁻⁵.
E0's interval missed zero by 2.7 × 10⁻⁵ in the other direction. These two
implementations differ only in resampling details and seed. **Neither overturns
the other; both say the lower bound is indistinguishable from zero.** C1 does not
reclassify E0 on this basis, and the E0 classification "observed advantage but
not stable" stands unchanged.

## 6. What is source-derived, what is statistical, what is untested

- **Source-derived observation.** The advantage appears only against
  chilledwater's own negatives; every cross-meter reference is zero or negative
  at all five contexts. TabPFN raises anomaly scores and lowers normal scores
  relative to the tree as context grows. Absolute score separation is unchanged.
- **Statistical robustness.** PR-AUC survives building- and segment-clustered
  resampling at 20k/50k/100k and 252 single-building omissions with no sign
  flip. ROC-AUC's interval excludes zero at 20k and 100k only.
- **Candidate-mechanism consistency.** The pattern is consistent with a
  within-meter representation or ordering effect concentrated in the
  high-precision region, mildly stronger at anomaly onset and in the
  low-reading regime.
- **Not verified by any intervention.** Nothing here is an experiment. No
  support was allocated, no feature was ablated, no model was fit. A consistent
  pattern in frozen predictions is not a demonstrated mechanism, and the
  onset-phase and reading-regime structure in particular could have other
  explanations this analysis cannot exclude.
