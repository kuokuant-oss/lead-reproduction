# Evaluate a tabular foundation model in M5

## Status

Accepted

Phase C note (2026-06-26): accepted after a completed LEAD/M3 local spike on
the same reduced `1,000 x 137` table as the paired GBDT anchor. The run used
local weights only, no TabPFN Client or cloud path. TabPFN AUC was `0.9904`
versus GBDT AUC `0.9870`; both models produced the same `0.5`-threshold
confusion matrix on this bounded validation slice, so precision/recall/F1 are
identical despite different probability ranking. The regenerated cold
fit+predict wall-clock was `6.5070` seconds on an RTX 4070 Laptop GPU, including
`1.5884` seconds of model initialization. This is an offline feasibility result,
not a real-time FDD claim.

Phase D note (2026-06-26): the rigorous four-axis comparison ran on the existing
GEPIII data through the frozen pipeline (10k balanced fit rows, 4k fixed val
rows, seeds `{42, 123, 999}`, within the documented TabPFN-3 `1,000,000 × 200`
limit). Results in `docs/reports/m5-foundation-vs-gbdt.md` and
`data/processed/m5_phaseD_foundation_vs_gbdt.json`. Judged on this ADR's rubric
(transfer, label scarcity, minimal feature engineering, not headline AUC): TabPFN
wins **label scarcity** (`+0.100` PR-AUC at 200 labels) and **true cross-site
ROC-AUC** (`0.9833` vs GBDT-retrain `0.9797`), and matches the tuned M3.4 ensemble
(`0.9928`) in-domain at a 10k context. GBDT keeps the **inference-latency** edge
(~100× faster) and the **minimal-feature-engineering** edge (GBDT > TabPFN on raw
17 features; TabPFN degrades more without the engineered lags). Tuned GBDT is not
dethroned on headline AUC. The decision stands: TabPFN is an M5 model-track
candidate for label-scarce and cross-site settings, not a GBDT replacement. The
deferred BDG2 ingestion skeleton (Phase D slice 1) was retired as premature
(issue #34) before this comparison.

Framing note (2026-06-26, issue #36): this milestone is **FDD on BDG2**. The
GEPIII TabPFN-vs-GBDT comparison is M5's FDD *model-selection stage* (the model
track, Phase A–D), now complete — TabPFN is one compared FDD model, not the
milestone's goal or an independent track. The BDG2 transfer is M5's **next stage,
Phase E**, not a separate milestone. The real-time deployment candidate remains
GBDT; TabPFN is positioned as an offline / label-scarce bootstrapper, bounded by
its inference latency and the TabPFN-3.0 research/internal-use license.

## Context

M5 is expected to move from reproduction toward FDD on BDG2. The M4 foundation
created importable data, feature, split, sample, and evaluation helpers, but the
accepted M3 model line remains GBDT-centered.

TabPFN is a tabular foundation model family from Prior Labs. The Nature 2025
TabPFNv2 paper frames TabPFN as a model that learns a tabular prediction
algorithm through in-context learning and reports strong performance on small
to medium tabular data. The TabPFN-2.5 and TabPFN-3 reports extend the scale
claim, with TabPFN-3 documenting up to 1M training rows and 200 features on H100
class hardware. Prior Labs model documentation lists the current TabPFN-3 limits
as `1,000,000 x 200`, `100,000 x 2,000`, or `1,000 x 20,000` rows x features.

This repo's strongest unresolved M5 motivation is not headline AUC. M3 already
has strong in-domain AUC, while site-held-out evaluation dropped to `0.9774` and
BDG2 may not provide per-row anomaly labels. A foundation model is therefore
interesting mainly as a transfer and low-feature-engineering candidate.

## Decision

Evaluate TabPFN as an M5 model track, not as a replacement for the accepted GBDT
line. TabPFN will consume the same tabular feature table as the paired GBDT
anchor and slot in at the model stage. Upstream feature construction remains
responsible for causal discipline, including `PAST_SHIFTS` for real-time FDD.

The evaluation ladder is:

1. LEAD subset feasibility and local-validation anchor: GBDT vs TabPFN on the
   same split, seed, downsample, and feature table.
2. Primary contribution: few-shot transfer to held-out or unlabeled sites,
   comparing TabPFN in-context adaptation with GBDT retrain and GBDT transfer.
3. Minimal-feature-engineering comparison to test whether TabPFN reduces feature
   engineering burden without overclaiming headline AUC.

Success criteria are a rigorous benchmark and transfer evaluation. Success does
not require TabPFN to beat tuned GBDT on raw AUC.

## Rationale

Tree ensembles remain strong for tabular data and may still win the raw AUC
comparison. The reason to evaluate TabPFN is that it may offer value where the
current GBDT line is weakest: label scarcity, site transfer, and lower feature
engineering requirements.

Keeping TabPFN at the model stage prevents it from weakening upstream
reproduction guarantees. It also keeps causal-vs-offline feature semantics
visible instead of hiding them inside a model comparison.

## Consequences

+ Phase B remains docs-only: no torch dependency, no TabPFN install, no BDG2
  download, and no GPU run.
+ Phase C, if approved, should be a LEAD-only feasibility spike and not a full
  benchmark.
+ TabPFN-3.0 License v1.0 must be treated as suitable for research/internal
  evaluation planning only; production or business-decision use requires a
  commercial license or API agreement.
+ The 1M-row TabPFN-3 claim is not a laptop feasibility claim. It was
  demonstrated on H100-class hardware and must be separated from any local LEAD
  spike result.
+ Unknowns for GEPIII row count fit, GPU/VRAM, inference latency, and data
  egress consent must be tracked before implementation.
+ Phase C found that the full M3 downsampled table is `4,285,104 x 137`, which
  exceeds the documented TabPFN-3 `1,000,000 x 200` limit. The reduced
  `1,000 x 137` local table fits and produced a completed local GPU TabPFN
  metric.

## References

+ Hollmann, N., Müller, S., Purucker, L. et al. "Accurate predictions on small
  data with a tabular foundation model." Nature 637, 319-326 (2025).
  https://doi.org/10.1038/s41586-024-08328-6
+ Grinsztajn, L. et al. "TabPFN-2.5: Advancing the State of the Art in Tabular
  Foundation Models." arXiv:2511.08667. https://arxiv.org/abs/2511.08667
+ Grinsztajn, L. et al. "TabPFN-3: Technical Report." arXiv:2605.13986.
  https://arxiv.org/abs/2605.13986
+ Prior Labs model documentation. https://docs.priorlabs.ai/models
