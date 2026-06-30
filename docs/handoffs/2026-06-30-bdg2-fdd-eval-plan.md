# Handoff: BDG2 FDD audit-yield model roles

**Date**: 2026-06-30
**Issue**: [#41](https://github.com/kuokuant-oss/lead-reproduction/issues/41)
**Branch**: `bdg2-fdd-eval-plan`

## Completed

+ Extended ADR 0020 with a `Model roles` decision section.
+ Extended `docs/plans/bdg2-fdd-eval-plan.md` with a two-stage scanner plus
  audit re-ranker design.
+ Kept the slice design-only: no scoring, no modeling, no labels, no pipeline,
  and no meter-scope change.

## Model-Roles Summary

+ GBDT, trained on GEPIII under ADR 0019's transfer contract, is the primary
  full-corpus scanner because M5 Phase D kept it as the real-time deployment
  candidate with sub-second inference.
+ TabPFN is offline, label-scarce, second-stage, and audit-centered. Its roles
  are GBDT top-K case-level re-ranking, 50-200 case few-shot calibration,
  active-learning / audit-set selection with diverse disagreement plus random
  baseline, and model-disagreement diagnostics.
+ Manual or external evidence is the only route to higher plausibility.
  `confirmed` remains unavailable in BDG2 because the repo has no maintenance,
  BMS, work-order, or adjudicated review record.

## M5 Phase D Consistency

+ The design preserves Phase D's split: GBDT remains the full-corpus / real-time
  candidate; TabPFN is retained where Phase D showed value, especially
  label-scarce and cross-site settings.
+ The design also preserves Phase D's limits: TabPFN was about `6.3 ms/row`
  (`~100x` slower than GBDT), has a research/internal-use license boundary, and
  did not win the minimal-feature-engineering axis (`0.9587` raw-17 GBDT
  ROC-AUC vs `0.9499` TabPFN).

## Methodological Guardrails

+ Audit labels are reviewer triage judgments (`likely`, `data-quality`,
  `OOD-normal`, `unknown`), not confirmed faults.
+ TabPFN ranking utility can only be evaluated on held-out audit cases or
  within-audit cross-validation and must be called triage-utility, not accuracy
  or BDG2 performance.
+ Evidence features are meant to separate data-quality/OOD candidates from
  operational candidates; the re-ranker must not simply relearn OOD or
  missingness.

## Deferred

README and the canonical M5 report wording were intentionally left unchanged in
this branch. Update those only after ADR 0020 moves from Proposed to Accepted
and the design is merged.

## Stop Point

Stop for review here. Do not implement scoring, evidence-packet generation,
model transfer, a new meter scope, or a BDG2 pipeline from this handoff.
