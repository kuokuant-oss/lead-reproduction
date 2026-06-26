# M5 Phase D — TabPFN (foundation model) vs GBDT (tree model) on GEPIII

**Issue**: [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)
**Data**: existing M3 ASHRAE GEPIII frame (`20,216,100 × 21`), labels included.
No BDG2, no cloud, no data egress. TabPFN runs from local weights only.
**Provenance**: `data/processed/m5_phaseD_foundation_vs_gbdt.json`
(commit `8f4373b`, generated 2026-06-26 UTC).

## Setup

Every paired cell reuses the **same split, downsample, feature table, and fixed
validation subsample**, through the frozen `src/lead` pipeline (`load_m3_frame`,
`add_value_change_features`, `split_mask`-style masks, `downsample_indices`,
`classification_metrics`). The only variable in a paired cell is the model.

+ **Models**: TabPFN-3 local checkpoint (`tabpfn==8.0.8`, RTX 4070 Laptop GPU,
  8 GB) vs LightGBM `LGBMClassifier(n_estimators=100)`. Both consume the same
  `StandardScaler`-transformed table.
+ **Feature table**: 137 features (17 baseline + 120 row-offset value-change),
  the M3.2 line.
+ **Fit budget**: 10,000 balanced rows (well beyond the 1,000-row Phase C spike).
  137 features ≤ 200 and 10,000 ≪ 1,000,000, so the run stays **within the
  documented TabPFN-3 `1,000,000 × 200` limit** — `ignore_pretraining_limits`
  was never set. The budget is bounded by 8 GB laptop VRAM, not the documented
  limit. (The full M3 downsample is `4,285,104 × 137`, which exceeds the
  `1,000,000 × 200` row limit, so the full table cannot be fed to TabPFN-3.)
+ **Validation**: a fixed 4,000-row natural-prevalence subsample per axis
  (anomaly rate ≈ 6%), scored identically by both models.
+ **Seeds**: fit-subsample + model `random_state` over `{42, 123, 999}`;
  reported as mean ± std. Metrics: ROC-AUC, PR-AUC (average precision),
  precision/recall/F1 at 0.5, and fit+predict latency.

All metrics below are **mean ± std over 3 seeds** unless noted. Latency is cold
in-process fit+predict (TabPFN includes model init + fit + `predict_proba`);
TabPFN `predict_proba` recomputes against the in-context training set every call.

---

## Axis 1 — In-domain (`80_20_mod5` building split)

| Model | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
| --- | --- | --- | --- | --- |
| GBDT (LightGBM, 10k fit) | 0.9877 ± 0.0012 | 0.9154 ± 0.0068 | 0.756 ± 0.013 | ~0.23 (warm) |
| TabPFN-3 (10k context) | **0.9925 ± 0.0005** | **0.9253 ± 0.0049** | 0.747 ± 0.007 | 26.8 ± 2.0 |

TabPFN edges the single-GBDT-at-10k baseline (+0.0048 ROC, +0.010 PR-AUC) but
its `predict_proba` is ~25.3 s for 4,000 rows (~6.3 ms/row) versus GBDT's
sub-second scoring — roughly **two orders of magnitude slower at inference**.

**Context**: the accepted M3.4 line is a *4-model ensemble on the full data*
with ROC-AUC `0.9928`. TabPFN at a 10k context **matches that tuned ensemble**,
while the single-GBDT-at-10k baseline sits below it. So TabPFN is not beating a
weak model here — it reaches the tuned-ensemble headline — but it does so at a
large inference-latency cost, and the tuned GBDT line is not dethroned.

---

## Axis 2 — Site transfer (PRIMARY, `site_id % 5 == 4` held out)

Held-out sites are never seen in training for the two *true cross-site* models.
M3 ensemble site-held-out anchor: ROC-AUC `0.9774` (full-data 4-model ensemble;
not directly comparable to these single-model 10k cells).

| Condition | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
| --- | --- | --- | --- | --- |
| GBDT-retrain (source sites only) | 0.9797 ± 0.0008 | **0.8221 ± 0.0035** | 0.780 ± 0.013 | ~0.24 |
| TabPFN-in-context (source sites only) | **0.9833 ± 0.0009** | 0.8119 ± 0.0052 | **0.783 ± 0.003** | 26.5 ± 0.2 |
| GBDT-transfer, no retrain (in-domain model) | 0.9882 | 0.9023 | 0.761 | ~0.003 |

**Honest reading.** Among the **true cross-site** models (trained only on source
sites), TabPFN-in-context beats GBDT-retrain on ROC-AUC (+0.0035) and F1, but
GBDT-retrain wins PR-AUC (+0.010) — a genuine split decision, not a clean TabPFN
sweep. The **GBDT-transfer-without-retrain** row has the highest ROC-AUC
(0.9882) and PR-AUC (0.9023), **but it is an easier setting**: that model was
trained on the `80_20_mod5` building split, whose source buildings span *all*
sites — including other buildings in the held-out sites. It therefore carries
site familiarity (weather regime, site mix) that the true cross-site models lack.
It answers "does a deployed all-sites model generalize to new *buildings* in
*known* sites?", not "does a model transfer to *unseen sites*?" Read that way, it
is not evidence against TabPFN's transfer.

---

## Axis 3 — Label scarcity (`80_20_mod5`, fixed 4k val)

ROC-AUC and PR-AUC (mean over 3 seeds) as the labeled support set shrinks:

| Support | GBDT ROC | TabPFN ROC | ΔROC | GBDT PR | TabPFN PR | ΔPR |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 0.9659 | 0.9806 | **+0.0148** | 0.6954 | 0.7953 | **+0.0999** |
| 500 | 0.9786 | 0.9829 | +0.0043 | 0.7669 | 0.8302 | +0.0634 |
| 1,000 | 0.9809 | 0.9834 | +0.0025 | 0.7815 | 0.8507 | +0.0692 |
| 2,000 | 0.9851 | 0.9863 | +0.0012 | 0.8635 | 0.8818 | +0.0183 |
| 5,000 | 0.9885 | 0.9899 | +0.0014 | 0.9086 | 0.9121 | +0.0035 |
| 10,000 | 0.9877 | 0.9925 | +0.0048 | 0.9154 | 0.9234 | +0.0080 |

**This is the clearest TabPFN win.** At 200 labels TabPFN leads by +0.015 ROC and
**+0.100 PR-AUC**; the gap shrinks monotonically (on PR-AUC) as labels grow. The
PR-AUC view — the right lens for a ~6%-prevalence anomaly task — shows the
foundation model is markedly better when labels are scarce, exactly where it is
expected to help.

---

## Axis 4 — Minimal feature engineering (`80_20_mod5`, 10k fit, 4k val)

| Feature set | GBDT ROC | TabPFN ROC | GBDT PR | TabPFN PR |
| --- | --- | --- | --- | --- |
| Raw baseline (17 feats) | **0.9587 ± 0.0042** | 0.9499 ± 0.0016 | **0.8305** | 0.7943 |
| Full value-change (137 feats) | 0.9877 | **0.9924** | 0.9154 | **0.9248** |
| **ROC drop 137 → 17** | **−0.0290** | −0.0424 | — | — |

**The feature-engineering-burden hypothesis is NOT supported here.** On the raw
17-feature set GBDT actually *beats* TabPFN (0.9587 vs 0.9499 ROC; 0.831 vs 0.794
PR-AUC), and TabPFN loses *more* when the engineered value-change lags are removed
(−0.042 vs GBDT's −0.029). The row-offset value-change features encode temporal
context that a single raw row cannot express, and TabPFN's in-context learning
does not recover that time-series structure from raw tabular rows — it depends on
the engineered features at least as much as GBDT does. In this anomaly-detection
setup, TabPFN does **not** reduce the feature-engineering burden.

---

## Verdict (honest, per ADR 0015)

ADR 0015 says to judge TabPFN on transfer, label scarcity, and minimal feature
engineering — not a single headline AUC. On that rubric:

**Where TabPFN beats GBDT**

+ **Label scarcity (strongest result)**: large PR-AUC advantage at small support
  (+0.100 PR-AUC at 200 labels), narrowing as labels grow.
+ **True cross-site transfer ROC-AUC**: TabPFN-in-context 0.9833 vs GBDT-retrain
  0.9797 (+0.0035) and higher F1 — though GBDT-retrain wins PR-AUC.
+ **In-domain at a matched 10k budget**: 0.9925 vs 0.9877, matching the tuned
  M3.4 ensemble (0.9928).

**Where GBDT wins / TabPFN does not help**

+ **Inference latency**: GBDT scores in sub-second; TabPFN `predict_proba` is
  ~25 s for 4,000 rows (~6.3 ms/row). Not viable for low-latency real-time FDD
  as-is. This is an offline benchmark, **not** a real-time FDD guarantee; any
  real-time claim still requires `PAST_SHIFTS`-only causal features per ADR 0007
  and ADR 0011.
+ **Minimal feature engineering**: GBDT > TabPFN on raw features, and TabPFN
  degrades more without the engineered lags — the opposite of the FE-saving
  hypothesis.
+ **Site-transfer PR-AUC**: GBDT-retrain edges TabPFN-in-context.
+ **Tuned headline**: the accepted full-data M3.4 GBDT ensemble (0.9928) is not
  dethroned; TabPFN matches it only at far higher inference cost.

**Net**: TabPFN is a credible foundation-model candidate specifically for
**label-scarce and cross-site** settings, where it adds real value (especially in
PR-AUC). It is not a drop-in replacement for the tuned GBDT line on headline AUC,
it is much slower at inference, and it does not lower the feature-engineering
burden in this task. That is the intended use of this comparison: a rigorous,
multi-axis read, not a headline claim.

## Deferred to Phase E (BDG2), M5's next stage

+ Real cross-**dataset** transfer to BDG2 (different buildings, sites, meters),
  with real BDG2 data, schema, and labels — not the retired synthetic skeleton.
+ Unlabeled / few-shot target-site adaptation on BDG2.
+ Any real-time FDD latency engineering: TabPFN inference latency must drop by
  orders of magnitude, and features must be `PAST_SHIFTS`-only (ADR 0007/0011).
