# M5 E4 — decision

**Artifact:** `data/processed/m5_e4_formal_path_a/e4_decision.json`

## The question E4 was authorised to answer

> Does a controlled change in hotwater positive/negative support produce, across
> three context seeds and two scaler arms, a directionally consistent factorial
> response on steam that exceeds inference noise?

The protocol did not pre-specify a verdict vocabulary for E4 the way it did for
E3, so none was invented. Each contrast is answered against an explicit,
conjunctive bar, stated in the artifact beside the answer:

1. all three context seeds agree in sign
2. both cluster types (building, segment) exclude zero
3. both primary endpoints (AUC, margin) agree in sign
4. both scaler arms agree in sign

**TabPFN-specific** additionally requires the TabPFN-minus-tree gap to clear the
same bar. An effect the matched fixed tree shows equally is a property of the
data, not of TabPFN.

## Verdict

| Contrast | Response established | TabPFN-specific |
|---|---|---|
| **negative-support main effect** | **yes** | **yes** |
| **positive-support main effect** | **yes** | no |
| positive × negative interaction | **no** | no |

Scaler arm changes no conclusion anywhere.

### Negative-support main effect — established and TabPFN-specific

The one finding that survives every criterion. Adding hotwater-negative support
to the context raises steam-positive separation from hotwater negatives by
**+0.41 AUC** and **+0.63 margin**, with 3/3 seeds agreeing, both clusterings
excluding zero on both endpoints, and both scaler arms agreeing.

It is also larger than the matched tree's response by **+0.124 AUC** and
**+0.152 margin**, and that gap itself clears the full bar. The effect is about
70× the median inference half-width.

### Positive-support main effect — established, but not TabPFN's

Adding hotwater-positive support *lowers* steam separation: **−0.19 AUC**,
**−0.28 margin**, 3/3 seeds, both clusterings, both arms.

The matched fixed tree shows the same thing slightly more strongly (−0.236 AUC
against TabPFN's −0.192), and the TabPFN-minus-tree gap does not exclude zero on
either endpoint or either clustering. So the response is real but shared: it is
a property of the context composition, not of TabPFN's handling of it.

### Interaction — not established

The two primary endpoints disagree, which is precisely what the co-primary rule
exists to catch. The AUC interaction (+0.32) excludes zero on both clusterings;
the margin interaction (+0.098) excludes zero on neither. The TabPFN-minus-tree
interaction gap has **opposite signs** on the two endpoints (−0.100 AUC, +0.079
margin), neither excluding zero.

Reporting the AUC alone would have yielded a confident interaction claim the
margin does not support.

### Scaler arm — no effect

The scaler-arm interaction excludes zero nowhere: no effect, no primary
endpoint, no clustering. Magnitudes are 0.000–0.007 against main effects of
0.19–0.63. Cell 11, where both arms are the same transform by construction,
gives an unplanned null control and confirms this is a genuine null rather than
a low-power one.

## Two facts that constrain how this may be read

**The AUC saturates in cell 01.** All six cell-01 fits give exactly 1.000000
with zero repeat variance across all three seeds and both arms. An AUC
half-width of zero there is a ceiling, not precision. The margin, which does not
saturate, agrees on both main effects — that agreement is what keeps them
standing.

**The intervals are conditional.** They cover resampling of query rows, given
these 24 fitted states and these three context seeds. They contain no model-seed
variation and no fresh-fit variation, because E4 executed neither. An interval
here says how much a contrast moves when the query is resampled, not how much it
would move under a different fit.

## Carried forward unresolved

| Item | Status |
|---|---|
| chilledwater positive vs hotwater negative | `RESOLUTION_LIMITED_DIAGNOSTIC` |
| onset / middle / recovery phase contrast | `UNRESOLVED_NOT_EXECUTED` |

Both were settled before any fit and neither carries a mechanism conclusion.
Chilledwater within-meter readouts are two to three orders of magnitude smaller
than the steam effects, and the single interval that excludes zero under
building clustering fails under segment clustering. Chilledwater is not pooled
into the steam claim.

## What this authorises

Nothing. `authorises` is empty in the decision artifact. E5 (frozen 192-row
query), E6 (complete other-half full-test confirmation), Path B, representation
ablation, 500k, site transfer, tree refit, TabPFN 8.1.0 as science, and
manuscript changes all remain prohibited and require separate human
authorisation.

## Decision

E4 formal Path A is complete and frozen. One mechanism-bearing result stands:
hotwater-negative support raises steam separation, consistently across seeds and
clusterings, on both primary endpoints, and by more than the matched fixed tree.
The positive-support effect is real but not TabPFN's. The interaction is not
established. The next stage awaits explicit human authorisation.
