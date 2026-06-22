# M3 Plan: Full ASHRAE GEPIII Reproduction

**Status**: M3.1/M3.2/M3.2a/M3.3/M3.4 complete; M3.5 post-processing next
**Started**: 2026-05-29
**Reference**:

+ 教授原話: 「用 ASHRAE GEPIII 完整 dataset (2000+ 電表), 從 raw 做 feature
  engineering, 用 anomaly label 做 binary classification, train 和 test 各別用一半
  的建築數量」
+ M2 baseline: docs/reproduction-report.md Ch5.6 (M3 outlook)
+ buds-lab reference: 02_preprocess_data.py (GEPIII rank-1 solution)
+ M3 baseline handoff: docs/handoffs/2026-05-29-m3-baseline.md

---

## M3 vs M2 對比

| 維度 | M2 (LEAD subset) | M3 (Full GEPIII) |
|---|---|---|
| Dataset | Kaggle energy-anomaly-detection | Kaggle ashrae-energy-prediction |
| Buildings | 406 (200+206) | 1,449 (自定 building_id % 5 split) |
| Anomaly rate | 2.13% | 6.50% |
| Feature source | pre-processed CSV (train_features.csv) | raw CSVs (from scratch) |
| Weather join | per-building in CSV | per-site: building → site → weather |
| Meter types | electricity dominant | 4 types (0=elec, 1=chill, 2=steam, 3=hot) |
| Test 評估 | Kaggle leaderboard | self-defined: building_id % 5 == 4 |

---

## ⚠️ M3 Feature Engineering 校準說明

M3 的 feature engineering 應以 `.scratch/02_preprocess_data.py`（buds-lab GEPIII
rank-1 solution）為參考基準。M3.1 + M3.2 目前**未包含**以下 buds-lab features：

| Feature 類別 | buds-lab 實作 | M3 現狀 |
|---|---|---|
| Cyclic time encodings | sin/cos(hour, day, month, weekday) | ❌ 缺 |
| Weather rolling lags | windows 7, 73 (lag/lead + rolling mean) | ❌ 缺 |
| Holiday flags | US Federal Calendar `holidays` library | ❌ 缺 |
| GaussianTargetEncoder (gte_*) | per-(site, meter) target encoding | ❌ 缺（需 anomaly label，M3 可做） |
| Building interaction strings | `primary_use + "_" + meter_str` | ❌ 缺 |
| Site 0 meter 0 correction | × 0.2931 (unit mismatch fix) | ❌ 缺 |
| Weather GMT offset | per-site UTC correction | ❌ 缺 |
| Weather interpolation + NA indicators | linear interp + `_na` flag cols | ❌ 缺 |

**M3.3 任務**：對齊以上 buds-lab features（優先），再考慮 ClusterNo + SavGol（次要）。

---

## Milestones

### M3.1: Baseline pipeline

**GitHub Issue**: [#13](https://github.com/kuokuant-oss/lead-reproduction/issues/13)
**Status (2026-05-29)**: ✅ Complete — commit ea3977d

**What**: 17-feature baseline (time + building metadata + weather), LightGBM only

**Done when**:

+ [x] Positional anomaly label join confirmed (6.50% rate)
+ [x] Building-level split: building_id % 5 == 4 → val (1160/289 buildings)
+ [x] 17 features from raw CSVs (no value-change, no ClusterNo)
+ [x] LightGBM baseline val AUC = **0.9562**
+ [x] Notebook: notebooks/06-m3-baseline.ipynb (Cells 1-11)
+ [x] Handoff: docs/handoffs/2026-05-29-m3-baseline.md

**Key results**:

| Metric | Value | M2 reference |
|---|---|---|
| Val AUC | 0.9562 | 0.8952 (M2.1, 57 feat) |
| Features | 17 | 57 |
| Buildings (train/val) | 1160/289 | 162/38 |
| Downsampled rows | 4,285,104 | 149,184 |
| Top feature | log_square_feet | dayofyear |

M3 baseline higher than M2 baseline despite fewer features: log_square_feet
(building size) is a strong anomaly signal not available in M2's feature set.

---

### M3.2: Value-change features

**GitHub Issue**: [#14](https://github.com/kuokuant-oss/lead-reproduction/issues/14)
**Status**: ✅ Complete (2026-05-29) — val AUC 0.9920, ΔAUC +0.0358

**What**: Add 60 shifts × 2 types (diff + ratio) → ~137 total features

```python
# 60 shifts: sub-day ±1-24h + multi-day ±48-168h (step 24h)
shifts = (
    list(range(-24, 0)) + list(range(1, 25))
    + list(range(-168, -24, 24)) + list(range(48, 169, 24))
)  # 60 shifts total (same as M2.2.b)
```

**Implementation approach** (vectorized, NOT per-building loop):

```python
df_sorted = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
for n in shifts:
    shifted = df_sorted.groupby('building_id')['meter_reading'].shift(n)
    df_sorted[f'lag_value_diff_{n}'] = df_sorted['meter_reading'] - shifted
    df_sorted[f'lag_value_ratio_{n}'] = (df_sorted['meter_reading'] + 1) / (shifted + 1)
```

**Done when**:

+ [x] 120 value-change features generated for train and val splits
+ [x] M3.2 val AUC = **0.9920** > 0.97 ✅ (exceeds M2 LightGBM 0.9818)
+ [x] ΔAUC vs M3.1: +0.0358
+ [x] Feature importance: value-change 46.7% / baseline 53.3%
+ [x] Notebook cells 12-15 added to 06-m3-baseline.ipynb
+ [x] Handoff: docs/handoffs/2026-05-29-m3.2-completed.md

**Risk**:

+ Memory: 20M rows × 120 cols × 4 bytes ≈ 9.6 GB — process train/val separately
+ Time: 60 groupby.shift ops on 16M train rows — estimate 5-15 min

**Depends on**: M3.1 ✅

---

### M3.2a: PI-response split/causality design check

**GitHub Issue**: [#18](https://github.com/kuokuant-oss/lead-reproduction/issues/18)
**Status**: Complete (2026-06-22) - run before M3.3 feature work

**What**: Experimental-design check requested by the PI. Reuse the M3.2
LightGBM pipeline and compare:

+ Building split: existing 80/20 `building_id % 5 == 4` vs PI 50/50
  `building_id % 2` protocol
+ Value-change regime: offline past+future shifts vs causal past-only shifts

**Done when**:

+ [x] 2x2 grid run with LightGBM only, `random_state=42`
+ [x] 80/20-offline reproduces M3.2 AUC 0.9920
+ [x] 50/50 split has building-level separation (725/724, overlap 0)
+ [x] Causal setting drops all negative/future shift features (77 total features)
+ [x] Seeded-random 50/50 robustness check run (`random_state=42`)
+ [x] 50/50-causal label-shuffle sanity check run
+ [x] Notebook: `notebooks/07-m3-split-causality.ipynb`
+ [x] Results: `data/processed/m3_split_causality_results.json`
+ [x] ADR: `docs/adr/0007-offline-batch-vs-causal-online-feature-regimes.md`
+ [x] Handoff: `docs/handoffs/2026-06-22-m32a-completed.md`
+ [x] Runner: `scripts/run_m3_split_causality.py`

**Key results**:

| Split | Regime | Features | Train/val buildings | Val AUC | P/R/F1 @ 0.5 |
|---|---|---:|---:|---:|---:|
| 80/20 mod5 | offline | 137 | 1160/289 | 0.9920 | 0.6409/0.9665/0.7707 |
| 80/20 mod5 | causal | 77 | 1160/289 | 0.9908 | 0.6237/0.9603/0.7562 |
| 50/50 mod2 | offline | 137 | 725/724 | 0.9914 | 0.6878/0.9421/0.7951 |
| 50/50 mod2 | causal | 77 | 725/724 | 0.9903 | 0.6646/0.9355/0.7772 |

Interpretation: the 50/50 dip is the cost of the PI protocol (fewer train
buildings), not a regression. The causal dip is the cost of real-time
deployability and operationalizes the M3.2 past/future leakage check by removing
future-shift contribution. ADR 0007 records that 80/20 offline remains the
canonical M3.3+ reproduction line, while causal past-only is the real-time-FDD
variant.

**Depends on**: M3.2. **Must precede** M3.3 feature work.

---

### M3.3: buds-lab Feature Alignment (Priority)

**GitHub Issue**: [#15](https://github.com/kuokuant-oss/lead-reproduction/issues/15)
**Status**: Complete (2026-06-22) - no robust AUC lift over M3.2

**What**: Add buds-lab 02_preprocess_data.py features not yet in M3 pipeline.
Priority order: cyclic encodings → weather rolling lags → holiday flags →
GaussianTargetEncoder → building interactions → site corrections.

**Done when**:

+ [x] Cyclic time encodings: sin/cos(hour, dayofweek, month) added (6 features)
+ [x] Weather rolling lags: windows 7, 73 (lag + rolling mean) added
+ [x] Holiday flags: US Federal Calendar `holidays` library
+ [x] GaussianTargetEncoder: per-(site, meter) target encoding on anomaly label
+ [x] Building interaction string: `primary_use + "_" + meter_str`
+ [x] Site 0 meter 0 correction: multiply by 0.2931 applied
+ [x] M3.3 val AUC = 0.9913 vs M3.2 0.9920; no-lift/negligible
+ [x] Label-shuffle diagnostics: seed range 0.3738-0.5700, GTE ablation 0.5680
+ [x] Notebook: `notebooks/08-m3-budslab.ipynb`
+ [x] Results: `data/processed/m3_3_results.json`
+ [x] Handoff: `docs/handoffs/2026-06-22-m33-completed.md`
**Risk**:

+ GaussianTargetEncoder needs anomaly label → potential leakage if not done correctly;
  fit on train only, apply to val with train parameters
+ Weather rolling lags: 20M rows × large window — memory and time significant
+ AUC did not beat M3.2 + 0.0005; document as no-lift/negligible, skip to M3.4

**Note on ClusterNo + SavGol**: Secondary priority. M2 ablation showed SavGol
had minimal effect (ΔAUC −0.001). Add after buds-lab features if M3.3 AUC improves.

**Result**: Val AUC `0.9913`, precision/recall/F1 `0.6668/0.9583/0.7864`.
Multi-seed AUC mean `0.9917` wraps around but does not robustly beat M3.2
`0.9920`. Label-shuffle seed 42 `0.5697` matches M3.2 `0.5669`; five shuffle
seeds range `0.3738-0.5700`, so M3.3 did not add a stable leakage signature.

**Depends on**: M3.2 ✅

---

### M3.4: 4-model ensemble (Stretch)

**GitHub Issue**: [#16](https://github.com/kuokuant-oss/lead-reproduction/issues/16)
**Status**: Complete (2026-06-22) - positive but modest lift over M3.2

**What**: LGB + XGB + CatBoost + HistGBT equal-weight ensemble

**Done when**:

+ [x] 4 models trained on M3.2 (137) features (M3.3 was no-lift)
+ [x] M3.4 ensemble val AUC = **0.9928** > 0.98
+ [x] Ranking vs M2 documented: M2 `LGB > Hist > Cat > XGB`; M3.4 seed 42 `LGB > Hist > XGB > Cat`
+ [x] Multi-seed sanity: ensemble AUC `0.9928/0.9932/0.9930`, mean `0.9930`, std `0.00018`

**Risk**: CatBoost 1000 iters on ~4.3M downsampled rows: estimate 30-60 min

**Result**: Seed-42 equal-weight ensemble AUC `0.9928`, P/R/F1
`0.6779/0.9664/0.7969`. This is a modest positive lift over M3.2 (`+0.00079`)
and over the best seed-42 single model (`+0.00078`). CatBoost ran all 1000
trees and took ~3.6 min in the local environment.

**Depends on**: M3.3

---

### M3.5: Post-processing (Stretch)

**GitHub Issue**: [#17](https://github.com/kuokuant-oss/lead-reproduction/issues/17)
**Status**: Pending M3.4

**What**: Rule 1 (meter_reading==1.0) + Rule 2b (year-end); Rule 2a needs EDA

**Notes**:

+ Rule 1: same as M2 (`predictions[test['meter_reading'] == 1] = 1`)
+ Rule 2b: same as M2 (`dayofyear > 366.9583`)
+ Rule 2a: M2 used building_id > 145 OR < 105 (LEAD-specific range).
  M3 building_id runs 0-1448 — need new EDA to find equivalent filter

**Done when**:

+ [ ] Rule 1 trigger rate quantified
+ [ ] Rule 2b applied and verified
+ [ ] Rule 2a EDA: find M3-appropriate filter or document as N/A
+ [ ] Post-processing ΔAUC recorded

**Depends on**: M3.4

---

## Out of M3 Scope

+ Kaggle submission (M3 not a competition entry)
+ Cross-validation across multiple random seeds
+ Hyperparameter tuning beyond library defaults
+ Paper §5 future work comparison (留給未來)

---

## Issue Tracker Map (M3)

| Milestone | GitHub Issue | Status |
|---|---|---|
| M3.1 baseline | [#13](https://github.com/kuokuant-oss/lead-reproduction/issues/13) | ✅ Closed |
| M3.2 value-change | [#14](https://github.com/kuokuant-oss/lead-reproduction/issues/14) | ✅ Closed |
| M3.2a PI-response split/causality | [#18](https://github.com/kuokuant-oss/lead-reproduction/issues/18) | ✅ Closed |
| M3.3 buds-lab alignment | [#15](https://github.com/kuokuant-oss/lead-reproduction/issues/15) | ✅ Closed |
| M3.4 4-model ensemble | [#16](https://github.com/kuokuant-oss/lead-reproduction/issues/16) | ✅ Closed |
| M3.5 post-processing | [#17](https://github.com/kuokuant-oss/lead-reproduction/issues/17) | 🚧 Open |

---

## M3 Exit Criteria

+ [x] M3.2 val AUC > 0.97 (0.9920 ✅)
+ [x] PI-response 50/50 split + causal/offline design check complete
+ [x] M3.3 buds-lab alignment complete; no robust AUC lift
+ [x] M3 pipeline (baseline + value-change) complete and reproducible
+ [x] M3.4 ensemble complete; seed-42 AUC 0.9928, multi-seed mean 0.9930
+ [ ] Handoff doc for each completed stage
+ [ ] GitHub Issues closed for completed milestones

---

**Last reviewed**: 2026-06-22 (M3.4 4-model ensemble complete; modest positive lift vs M3.2)
