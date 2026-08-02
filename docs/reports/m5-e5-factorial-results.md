# M5 E5 — factorial results on the frozen 192-row query

Everything recomputed from the raw per-repeat score vectors. Point estimates are
equal-weight averages over the three pre-specified context seeds; intervals are
percentile 95% intervals over 1,000 addressable clustered draws with namespace
5005. `*` marks an interval that excludes zero.

## 1. Coverage

| | |
|---|---|
| states reloaded | 24 / 24 |
| same-process repeats | 192 / 192 |
| score vector length | 192 |
| tree score vectors | 24 |
| **fits performed** | **0** |
| distinct state identities | 24 |
| distinct process UUIDs | 24 |
| effective `n_estimators_` | {8} |
| scaler verified exact | all 24 |
| clustered draws valid | 1000/1000 in all four bootstraps |
| clusters | 142 building, 192 segment |

## 2. Inference variation (the 8 repeats of one reloaded state)

| | steam AUC | steam margin |
|---|---:|---:|
| min half-width | 0.000415 | 0.000334 |
| median half-width | 0.003590 | 0.002466 |
| max half-width | 0.009790 | 0.006440 |
| states with zero variance | **0 / 24** | 0 / 24 |

All 192 repeats produced distinct score digests — 8 of 8 in every state.
Repeated inference on a reloaded TabPFN 8.0.8 state is not bitwise
reproducible, now confirmed on a third independent occasion.

**The AUC does not saturate here.** In E4 six of 24 fits gave exactly 1.000000
with zero variance, all of them cell 01. On the 192-row query cell 01 sits at
0.985–0.993, so every state carries usable variance and the ceiling that
qualified E4's AUC does not apply to E5.

## 3. Cell means (steam AUC, TabPFN, cell-specific arm)

| Seed | 00 | 01 | 10 | 11 |
|---|---:|---:|---:|---:|
| 42 | 0.7357 | 0.9930 | 0.3903 | 0.9062 |
| 123 | 0.7242 | 0.9884 | 0.3545 | 0.8870 |
| 999 | 0.7239 | 0.9852 | 0.2834 | 0.8812 |

Cell 10 again sits below 0.5 in every seed, as it did in E4.

## 4. The primary replication target: negative-support main effect

### TabPFN

| Endpoint | Arm | Overall | Per seed (42 / 123 / 999) | SD | Building CI | Segment CI |
|---|---|---:|---|---:|---|---|
| AUC | cell_specific | **+0.40487** | +0.3867 / +0.3984 / +0.4296 | 0.0222 | [+0.3312, +0.4753]* | [+0.3467, +0.4646]* |
| AUC | frozen_reference | **+0.40447** | +0.3837 / +0.4013 / +0.4284 | 0.0225 | [+0.3304, +0.4747]* | [+0.3454, +0.4650]* |
| margin | cell_specific | **+0.59916** | +0.5976 / +0.6008 / +0.5991 | 0.0016 | [+0.5679, +0.6308]* | [+0.5696, +0.6277]* |
| margin | frozen_reference | **+0.59823** | +0.5944 / +0.6036 / +0.5966 | 0.0048 | [+0.5673, +0.6302]* | [+0.5679, +0.6263]* |

3/3 seeds positive on both endpoints and both arms; all eight intervals exclude
zero upward.

### Fixed tree comparator

| Endpoint | Arm | Overall | Per seed | Building CI | Segment CI |
|---|---|---:|---|---|---|
| AUC | cell_specific | +0.32212 | +0.2897 / +0.3431 / +0.3336 | [+0.2587, +0.3850]* | [+0.2741, +0.3716]* |
| AUC | frozen_reference | +0.31549 | +0.2827 / +0.3315 / +0.3323 | [+0.2503, +0.3793]* | [+0.2642, +0.3670]* |
| margin | cell_specific | +0.49817 | +0.4809 / +0.5210 / +0.4926 | [+0.4652, +0.5297]* | [+0.4720, +0.5250]* |
| margin | frozen_reference | +0.49797 | +0.4773 / +0.5216 / +0.4951 | [+0.4634, +0.5299]* | [+0.4713, +0.5257]* |

The matched tree responds to the same intervention, and substantially — most of
the effect is not TabPFN's.

### TabPFN minus tree

| Endpoint | Arm | Overall | Per seed | Building CI | Segment CI |
|---|---|---:|---|---|---|
| AUC | cell_specific | **+0.08274** | +0.0969 / +0.0553 / +0.0960 | [+0.0317, +0.1301]* | [+0.0402, +0.1251]* |
| AUC | frozen_reference | **+0.08898** | +0.1010 / +0.0698 / +0.0961 | [+0.0389, +0.1365]* | [+0.0480, +0.1313]* |
| margin | cell_specific | **+0.10099** | +0.1166 / +0.0799 / +0.1065 | [+0.0603, +0.1425]* | [+0.0674, +0.1335]* |
| margin | frozen_reference | **+0.10026** | +0.1172 / +0.0821 / +0.1016 | [+0.0589, +0.1429]* | [+0.0662, +0.1340]* |

The gap clears the full bar too: 3/3 seeds, both endpoints, both arms, all eight
intervals excluding zero.

## 5. Side by side with E4

| Quantity | E4 (352-row) | E5 (192-row) |
|---|---:|---:|
| TabPFN negative-support, AUC | +0.4118 | +0.4049 |
| TabPFN negative-support, margin | +0.6312 | +0.5992 |
| TabPFN − tree, AUC | +0.1237 | +0.0827 |
| TabPFN − tree, margin | +0.1522 | +0.1010 |

The main effects land within about 0.01 (AUC) and 0.03 (margin) of E4 on a query
that shares no row with it. The TabPFN-minus-tree gaps are about a third smaller
than E4's while remaining positive on 3/3 seeds and excluding zero on both
clusterings and both endpoints. E5's job was to test direction and significance
against pre-declared thresholds, and the smaller gap changes neither.

## 6. Secondary readouts

These do not control the verdict. They are reported only as compatibility checks
against E4.

| Effect | Endpoint | E5 overall | Sign | Intervals | E4 said |
|---|---|---:|---|---|---|
| positive-support main | AUC | −0.24132 | 3/3 negative | both exclude zero | −0.1922, established, not TabPFN-specific |
| positive-support main | margin | −0.29097 | 3/3 negative | both exclude zero | −0.2785, same |
| positive × negative | AUC | +0.28780 | 3/3 positive | both exclude zero | +0.3215, excluded zero |
| positive × negative | margin | **−0.15798** | 3/3 negative | both exclude zero | +0.0978, did **not** exclude zero |

The positive-support effect reproduces closely and keeps its sign.

**The interaction is where E5 is most informative, and it is negative
information.** On the 192-row query the two co-primary endpoints do not merely
disagree about significance as they did in E4 — they now have **opposite signs**
and both exclude zero. An AUC-only reading would report a large positive
interaction; a margin-only reading would report a clear negative one. That is
compatible with E4's conclusion that the interaction is not established, and it
strengthens it: the interaction is not a stable quantity across endpoints.

Per the protocol, an interval containing zero is not proof of absence, and
nothing here is read that way.

### Scaler arm

| Endpoint | Scaler-arm interaction | Building | Segment |
|---|---:|---|---|
| AUC | −0.000392 | does not exclude zero | does not exclude zero |
| margin | −0.000928 | does not exclude zero | does not exclude zero |

Magnitudes of 4e−4 to 9e−4 against a main effect of 0.40 to 0.60. Compatible
with E4, where the scaler-arm interaction excluded zero nowhere either.

### Chilledwater

The 192-row query contains no chilledwater rows. No chilledwater endpoint was
computed, and no other query was substituted to supply one.

## 7. What these numbers do not license

The clustered intervals are conditional on these 24 fitted states and these
three context seeds. They contain no model-seed or fresh-fit variation, because
E5 executed neither — E5 refits nothing at all.

And the constraint carried from the scaler audit: because removing hotwater
support also flips `meter` from numerical to categorical in cell 00, E5 tests
independent reproduction of the negative-support intervention **as a whole**.
It does not isolate the hotwater-normal reference as the sole mechanism.
