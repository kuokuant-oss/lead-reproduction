# M5 E5 — decision

**Artifact:** `data/processed/m5_e5_independent_replication/e5_decision.json`

## Verdicts

| | |
|---|---|
| **A. Response replication** | **REPLICATED** |
| **B. TabPFN-specific replication** | **REPLICATED** |

Both were decided by rules frozen in `e5_protocol.json` at commit `c5d38c4b…`,
before the 192-row query had ever been scored. No threshold was changed after
seeing a result, and the decision script reads the rules from the protocol
rather than restating them.

## The bar, and what met it

`REPLICATED` required all five, on the negative-support main effect:

| Condition | AUC | Margin |
|---|---|---|
| overall effect > 0 | +0.40487 ✓ | +0.59916 ✓ |
| 3/3 context seeds positive | +0.3867 / +0.3984 / +0.4296 ✓ | +0.5976 / +0.6008 / +0.5991 ✓ |
| both scaler arms positive | +0.40487, +0.40447 ✓ | +0.59916, +0.59823 ✓ |
| building interval excludes zero | [+0.3312, +0.4753] ✓ | [+0.5679, +0.6308] ✓ |
| segment interval excludes zero | [+0.3467, +0.4646] ✓ | [+0.5696, +0.6277] ✓ |

**TabPFN-specific** additionally required the TabPFN-minus-tree gap to clear the
same bar on both endpoints. It does:

| Endpoint | Gap | Per seed | Building | Segment |
|---|---:|---|---|---|
| AUC | +0.08274 | +0.0969 / +0.0553 / +0.0960 | [+0.0317, +0.1301] ✓ | [+0.0402, +0.1251] ✓ |
| margin | +0.10099 | +0.1166 / +0.0799 / +0.1065 | [+0.0603, +0.1425] ✓ | [+0.0674, +0.1335] ✓ |

## Read against E4

| Quantity | E4 (352-row) | E5 (192-row, independent) |
|---|---:|---:|
| negative-support main, AUC | +0.4118 | +0.4049 |
| negative-support main, margin | +0.6312 | +0.5992 |
| TabPFN − tree, AUC | +0.1237 | +0.0827 |
| TabPFN − tree, margin | +0.1522 | +0.1010 |

The main effect lands within about 0.01 (AUC) and 0.03 (margin) of E4 on a query
sharing no row with it. The TabPFN-minus-tree gap is about a third smaller than
E4's. It still meets every pre-declared condition, and E5's job was direction
and significance against fixed thresholds, not magnitude agreement — but the
shrinkage is real and should not be reported as if E5 reproduced E4's gap size.

## Three things the verdict does not say

**It does not isolate the mechanism.** In cell 00 the context has no hotwater
rows, so `meter` has three levels instead of four and TabPFN classifies it as
categorical and ordinal-encodes it after scaling; in the other three cells all
137 features stay numerical. Removing hotwater support therefore also flips a
feature's modality in the reference cell. E5 tests independent reproduction of
E4's negative-support intervention **as a whole**. A successful replication may
**not** be described as having isolated the hotwater-normal reference as the
sole mechanism.

**It does not cover fit variation.** The clustered intervals are conditional on
these 24 fitted states and these three context seeds. E5 refits nothing, so no
model-seed or fresh-fit variation is inside any interval here.

**It authorises nothing.** `authorises` is empty in the decision artifact. E6,
Path B, representation ablation, tree refit, 500k, site transfer, TabPFN 8.1.0
as science, manuscript changes, and scoring the 10,137,155-row holdout all
remain prohibited and require separate human authorisation.

## Secondary compatibility with E4

Secondary readouts do not control the verdict and did not.

| Effect | E5 | E4 | Compatible? |
|---|---|---|---|
| positive-support main | −0.241 AUC, −0.291 margin, 3/3 negative, intervals exclude zero | −0.192, −0.279, same pattern | yes |
| scaler-arm interaction | −0.0004 / −0.0009, excludes zero nowhere | excluded zero nowhere | yes |
| positive × negative | AUC +0.288 and margin **−0.158**, both excluding zero — **opposite signs** | AUC excluded zero, margin did not | yes, and stronger |

The interaction result is worth stating plainly. In E4 the two co-primary
endpoints disagreed about significance; in E5 they disagree about **sign**, and
both intervals exclude zero. An AUC-only reading would report a large positive
interaction and a margin-only reading a clear negative one. That is compatible
with E4's "not established" and makes it firmer: the interaction is not a stable
quantity across endpoints, and reporting either endpoint alone would mislead.

An interval containing zero is not treated as proof of absence anywhere in this
report.

## Execution-provenance note

The fixed tree comparator was scored in the laptop environment that fitted it,
because that is the only environment where the reloaded ensembles reproduce E4's
frozen comparator bit for bit — 352/352 rows, max difference 0.000e+00, on all
24 units. On gpu-host the same ensembles differ from E4's comparator by a mean
of 8.1e−03, comparable to the gap under test, so gpu-host tree output was
prohibited and none was used. TabPFN scored entirely on gpu-host, as in E4. No
tree was refit.

This is an execution-provenance limitation. It is not a scientific factor and
does not lower the TabPFN-specific threshold, which was met on its original
terms.

## Decision

E5 is complete and frozen. The steam negative-support response established in E4
reproduces on a completely independent, never-before-scored 192-row query with
zero row overlap: **REPLICATED**, and **REPLICATED as specific to TabPFN beyond
the matched fixed tree**. The next stage awaits explicit human authorisation.
