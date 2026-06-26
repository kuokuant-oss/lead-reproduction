# Handoff: M5 Phase D — TabPFN vs GBDT comparison on GEPIII

**Date**: 2026-06-26
**Issue**: [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)
**Report**: `docs/reports/m5-foundation-vs-gbdt.md`
**Data**: `data/processed/m5_phaseD_foundation_vs_gbdt.json`
**Harness**: `scripts/run_m5_phaseD_foundation_vs_gbdt.py`

## What this slice did

Replaced the retired BDG2-ingestion detour with the real Phase D job: a rigorous,
multi-seed, four-axis TabPFN (foundation) vs GBDT (LightGBM) comparison on the
existing M3 GEPIII data, through the frozen `src/lead` pipeline. Every paired
cell reuses the same split, downsample, feature table, and fixed 4,000-row
validation subsample, so only the model varies. Metrics: ROC-AUC, PR-AUC,
precision/recall/F1@0.5, fit+predict latency; mean ± std over seeds `{42,123,999}`.

Budget: 10,000 balanced fit rows (well beyond the 1,000-row Phase C spike), 137
features. Within the documented TabPFN-3 `1,000,000 × 200` limit, so
`ignore_pretraining_limits` was never set; the budget is bounded by 8 GB laptop
VRAM. Local TabPFN weights only (`tabpfn==8.0.8`, RTX 4070), no cloud, no egress.
Full run: 559.8 s, all cells completed, zero failures.

## Headline results (mean over 3 seeds)

| Axis | TabPFN | GBDT | Read |
| --- | --- | --- | --- |
| In-domain ROC-AUC | 0.9925 | 0.9877 | TabPFN matches tuned M3.4 ensemble 0.9928 at 10k context |
| Site transfer ROC-AUC (true cross-site) | 0.9833 | 0.9797 (retrain) | TabPFN wins ROC; GBDT-retrain wins PR-AUC |
| Label scarcity @200 PR-AUC | 0.7953 | 0.6954 | **+0.100** — TabPFN's clearest win |
| Minimal FE, raw 17 ROC-AUC | 0.9499 | 0.9587 | GBDT wins; FE-saving hypothesis NOT supported |

Inference latency: TabPFN `predict_proba` ~25 s / 4,000 rows (~6.3 ms/row) vs
sub-second GBDT — ~100× slower.

## Honest verdict

TabPFN adds real value in **label-scarce** and **true cross-site** settings.
It does **not** dethrone the tuned GBDT headline (0.9928 ensemble), is far slower
at inference, and does **not** lower the feature-engineering burden (it depends on
the engineered value-change lags at least as much as GBDT). Nuance recorded in the
report: GBDT-transfer-without-retrain scores highest on the site split (0.9882)
but is an easier setting (all-sites training carries site familiarity), so it is
not apples-to-apples with the true cross-site models.

## Rails honored

Frozen `lead.__all__` unchanged (harness imports only public API; PR-AUC computed
additively in the script). Local weights, `TABPFN_NO_BROWSER=1`,
`TABPFN_DISABLE_TELEMETRY=1`, no token, no cloud. Result JSON carries provenance
(commit, UTC, command). Pre-commit gate passed.

## Next: deferred to the later BDG2 scale-out milestone

+ Real BDG2 cross-**dataset** transfer (real data, schema, labels — not a
  synthetic skeleton), unlabeled / few-shot target-site adaptation.
+ Real-time FDD latency engineering: TabPFN inference must drop by orders of
  magnitude, and features must be `PAST_SHIFTS`-only (ADR 0007/0011).
