# M3 Plan: Full ASHRAE GEPIII Reproduction

**Status**: M3.1 baseline complete; M3.2 value-change in progress
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

### M3.3: ClusterNo + SavGol (Stretch)

**GitHub Issue**: TBD
**Status**: Pending M3.2

**What**: K-means building clustering + SavGol residual → ~139 features total

**Done when**:

+ [ ] ClusterNo: 1449-building KMeans(n_clusters=10, n_init=10, random_state=666)
+ [ ] SavGol: per-building savgol_filter(ffill().bfill().fillna(0), 5, 3)
+ [ ] M3.3 val AUC > 0.975

**Risk**:

+ ClusterNo pivot: 8784 timestamps × 1449 buildings (larger than M2's × 406)
+ SavGol per-building loop: 1449 buildings × 8784 rows ≈ slower than M2 (× ~3.5)

**Depends on**: M3.2

---

### M3.4: 4-model ensemble (Stretch)

**GitHub Issue**: TBD
**Status**: Pending M3.3

**What**: LGB + XGB + CatBoost + HistGBT equal-weight ensemble

**Done when**:

+ [ ] 4 models trained on M3.3 features
+ [ ] M3.4 ensemble val AUC > 0.98
+ [ ] Ranking vs M2 (LGB > Hist > Cat > XGB in M2) documented

**Risk**: CatBoost 1000 iters on ~4.3M downsampled rows: estimate 30-60 min

**Depends on**: M3.3

---

### M3.5: Post-processing (Stretch)

**GitHub Issue**: TBD
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
| M3.3-M3.5 | TBD | Pending |

---

## M3 Exit Criteria

+ [ ] M3.2 val AUC > 0.97
+ [ ] M3 pipeline (baseline + value-change) complete and reproducible
+ [ ] Handoff doc for each completed stage
+ [ ] GitHub Issues closed for completed milestones

---

**Last reviewed**: 2026-05-29 (M3.1 + M3.2 complete; M3.3 ClusterNo+SavGol next)
