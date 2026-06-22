# Handoff: M3 Milestone Closed

**Date**: 2026-06-22
**Status**: ✅ M3 milestone fully closed (M3.1 → M3.5 all done)
**Scope**: Full ASHRAE GEPIII anomaly detection, 1,449 buildings, raw CSV → feature engineering → validation
**Final reports**: [docs/m3-report.md](../m3-report.md), [docs/m3-plan.md](../m3-plan.md)

## 摘要

M3 extends the LEAD reproduction workflow from the competition subset to full
ASHRAE GEPIII. The milestone is closed with M3.1-M3.5 complete and all six M3
issues (#13-#18) closed. The headline canonical line is the M3.4 80/20 offline
4-model ensemble at `0.9928` validation AUC; the PI-requested 50/50 ensemble
follow-up is also complete, with `0.9921` offline AUC and `0.9911` causal AUC.

M3.5 closes post-processing as a documented null result rather than forcing an
M2 rule transfer. The combined hard-rule post-processing delta is `-0.000054`,
so the downstream story should preserve the negative result and move the focus
to FDD transfer questions.

## 最終數字

| PI framing | M3 result | Interpretation |
|---|---:|---|
| Step 1: full GEPIII raw-to-feature pipeline | M3.2 value-change LGB AUC `0.9920` | 80/20 offline baseline with 137 features |
| Step 2: paper-style ensemble extension | M3.4 ensemble AUC `0.9928` | Canonical 80/20 offline headline |
| Step 3: train/test each use half the buildings | 50/50 offline `0.9921`, causal `0.9911` | Causal number is the deployable real-time FDD variant |

M2 remains the LEAD reproduction anchor: Kaggle Private `0.98616`, gap `0.05%`
from the original `0.98661`. M3 confirms that the value-change feature family
transfers strongly to full GEPIII, while the equal-weight ensemble adds a modest
positive lift and hard-rule post-processing does not transfer.

## Limitations

The main limitation for downstream work is transfer, not local validation
quality. Site-held-out ensemble AUC is `0.9774`, materially below the canonical
`0.9928`, so cross-site generalization should be treated as a first-class FDD
question. Steam is the weakest meter slice at `0.9553` AUC, consistent with the
meter-type error discussion in the GEPIII context.

The causal vs offline distinction must stay explicit. Offline value-change uses
past and future shifts and is appropriate for batch labeling; causal value-change
uses past-only shifts and is the real-time FDD deployability number. Residual
structure also remains visible in diagnostics: label-shuffle mean is `0.519`,
and `945/1449` buildings (`65.2%`) have missing hours inside their observed
timestamp range, so row-offset value-change features are approximations across
timestamp gaps.

## Next

Next work should move to downstream FDD on BDG2. The two design constraints to
carry forward are cross-site transfer and causal feature availability: use the
`0.9911` 50/50 causal ensemble as the deployable M3 reference, and treat the
`0.9774` site-held-out result as the warning that BDG2 transfer needs explicit
validation rather than assuming canonical GEPIII validation will generalize.
