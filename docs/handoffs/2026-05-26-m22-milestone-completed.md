# Handoff: M2.2 Milestone Complete

**Date**: 2026-05-26
**Status**: ✅ Complete — M2.2 closed (Issue #9), ready for M2.3

---

## What was accomplished

### M2.1 baseline (carried forward)

Full pipeline with 57 raw features: cloud_coverage fix → downsampling → CV split →
StandardScaler → LightGBM(n_estimators=100).

- val AUC = **0.8952** (paper baseline 0.9311, gap 3.86% — <5% pass)
- Anomaly rate: 2.13% (paper says ~5%; documented unknowns.md #8)
- Downsampled df_eq: 149,184 rows, 50:50 class ratio
- CV split: 162 train buildings / 38 val buildings (building_id % 5)

### M2.2.0: cloud_coverage sentinel fix (ablation)

- Fixed `cloud_coverage = 255` → 10 (affects 797,545 rows = 45.6%)
- ΔAUC = **+0.0000** — disconfirmed as M2.1 gap source
- Reason: all four GBDT models are tree-based; monotonic feature transforms
  preserve split ordering → AUC invariant
- Fix kept in pipeline for buds-lab alignment

### M2.2.a: ClusterNo (per-building shape clustering)

- Pivot: (8784, 406) timestamps × buildings (train 200 + test 206)
- KMeans: n_clusters=10, random_state=666, **n_init=10 explicit**
- **ARI = 1.0** — 406/406 buildings match buds-lab reference exactly
- Saved: `data/interim/clusterno.csv` (406 rows × 2 cols)
- Critical finding: sklearn 1.4+ `n_init='auto'` with k-means++ = 1 (was 10);
  without explicit n_init=10, ARI=0.503

### M2.2.b: Value-change features (120 features)

- 60 shifts × `lag_value_{n}` (difference) + `lag_value_ratio_{n}` = 120 features
- Implementation: `groupby('building_id').shift(n)` (approximate; buds-lab used timestamp-based merge)
- Direction verified on building_id=107: `lag_value_1` = -175.183 ✓
- **Discovery**: 104/200 buildings have missing timestamps (min 7,471 ts = building_id=1353);
  groupby.shift() ≠ exact n-hour shift for these buildings → documented unknowns.md #11
- Output not saved (2.9 GB); M2.2.e regenerates in memory (3.3s)

### M2.2.c: Savitzky-Golay residual

- `Residual_savgol_w5p3 = meter_reading - savgol_filter(meter_reading, window=5, poly=3)`
- NaN fix: `.ffill().bfill().fillna(0)` required (`ffill` alone leaves leading NaN;
  `savgol_filter` rejects NaN input)
- Sample building residual mean = -0.0022 (≈ 0); max |mean| across 200 buildings = 0.117

### M2.2.d: dayofyear float feature

- `dayofyear = dt.dayofyear + dt.hour / 24`
- Range: 1.0 to 366.9583 (2016 leap year) ✓
- Pearson corr with anomaly: -0.0034 (low linear; tree importance still high — paper Fig 5 rank #5)

### M2.2.e: Integration + LightGBM val AUC

Full 169-feature pipeline run end-to-end in `notebooks/05-m2-integration.ipynb`.

| Feature class | Count |
|---------------|-------|
| raw_numeric (select_dtypes minus drops) | 46 |
| ClusterNo | 1 |
| lag_value_diff | 60 |
| lag_value_ratio | 60 |
| Residual_savgol_w5p3 | 1 |
| dayofyear | 1 |
| **Total** | **169** |

**val AUC = 0.9818** (paper Table 2: 0.9849, gap **0.31%** — <3% pass)
ΔAUC vs M2.1: **+0.0866**

Feature importance overlap with paper Fig 5 top 10: **8/10**
(missing: `gte_building_id` rank #7, `gte_meter_primary_use` rank #10)

---

## Methodology lessons (cumulative)

### 1. Tree-based monotonic invariance (M2.2.0)

LightGBM / XGBoost / CatBoost / HistGBT split decisions depend only on
feature value **ranking**, not absolute values. Any monotonic transform
(e.g., 255→10 preserving order) produces identical AUC. StandardScaler
similarly has no effect on these models.

**Implication**: Diagnosing AUC gaps in tree-based pipelines should focus on
rank-changing operations (NaN imputation, clipping, missing data handling),
not scaling or sentinel remapping.

### 2. sklearn n_init silent default change (M2.2.a)

sklearn 1.4+ changed `n_init='auto'` with k-means++ to mean **n_init=1** (was 10).
Result without explicit setting: ARI=0.503 (wrong clusters).

**Rule**: Explicitly set all version-sensitive sklearn defaults, especially
`n_init`, `max_iter`, and algorithm-selection parameters.

### 3. SavGol leading-NaN edge case (M2.2.c)

`ffill()` leaves leading NaN when a building's meter_reading starts with NaN
(no prior values to forward-fill). `savgol_filter` rejects NaN input with
a `ValueError`. Fix: `.ffill().bfill().fillna(0)` — `bfill` handles leading
NaN, `fillna(0)` handles all-NaN buildings.

**Rule**: For time-series preprocessing that rejects NaN, always chain
`.ffill().bfill().fillna(0)`, not just `.ffill()`.

### 4. Pearson correlation vs tree feature importance (M2.2.d)

`dayofyear` had Pearson corr = -0.0034 with anomaly labels — nearly zero
linear correlation. Yet it ranks #5 in paper Fig 5 (LightGBM split-count
importance). Tree models capture non-linear, interaction-based signal;
Pearson correlation is a poor proxy for tree feature importance.

**Rule**: Do not use Pearson correlation to pre-screen features for tree models.

### 5. Centered SavGol future-info legitimacy for batch tasks (M2.2.e)

`savgol_filter` with window=5, poly=3 uses points i±2 (future information).
For real-time deployment this would be test-time leakage. For LEAD
(batch evaluation over a full year of historical data), it is legitimate —
the "future" points are already available at inference time.

Verification: centered vs causal SavGol residual correlation = 0.333 (full
series), 0.600 (anomaly rows). The signal content is different but both are
valid for batch evaluation.

**Rule**: Distinguish between streaming (real-time) and batch (historical)
contexts before classifying future-data usage as leakage.

---

## Key findings still open (M2.5 ablation targets)

Ordered by estimated impact on closing the M2.1 gap (0.8952 vs paper 0.9311):

| Priority | Finding | Expected ΔAUC | Verification |
|----------|---------|---------------|-------------|
| 1 | **impute_nulls missing**: buds-lab Feature generator Cell 11 fills NaN rows with per-building mean meter_reading; M2 lets LightGBM handle NaN natively | Unknown (PRIMARY SUSPECT — NaN changes split paths) | M2.5: add/remove imputation |
| 2 | **CV fold variance**: only fold 4 (`building_id % 5 == 4`); paper may average 5 folds | ±0.01 range | M2.5: run all 5 folds |
| 3 | **Downsampling seed variance**: seeds 10/20 fixed; paper seed unknown | ±0.005 | M2.5: multiple seed combos |
| 4 | **LightGBM version differences**: 2022 vs 2026 default behavior (min_child_samples, num_leaves, etc.) | Small | M2.5: pin 2022 version or check changelog |
| 5 | **Paper Fig 4 number nature**: 0.9311 may be 5-fold average or test AUC, not fold-4 val AUC | Framing, not real gap | M2.5: accept as confound |
| 6 | **Missing timestamps + groupby.shift() approximation**: 104/200 buildings lack timestamps; shift(n) ≠ n-hour offset for these | Small–medium | M2.5: timestamp-merge vs groupby.shift() |
| 7 | **SavGol + ClusterNo AUC contribution**: paper §2.2.4 says "no apparent positive effect"; M2.2.e split-count importance rank #6 = 105 | Quantify ablation | M2.5: remove each feature, measure ΔAUC |

unknowns.md coverage: #10 (candidates 1–6), #11 (finding 6), #12 (finding 7), #5 (gte_* 8/10 overlap).

---

## AUC progression

| Step | val AUC | Features | Notes |
|------|---------|----------|-------|
| M2.1 baseline | 0.8952 | 57 | LightGBM defaults |
| M2.2.0 cloud_coverage fix | 0.8952 | 57 | ΔAUC = 0 (tree invariant) |
| M2.2.e integration | **0.9818** | 169 | ΔAUC = +0.0866 vs M2.1 |
| Paper Table 2 (LightGBM) | 0.9849 | 169 | gap **0.31%** |
| Paper Table 2 (ensemble) | 0.9866 | 169 | 4-model target for M2.3 |

---

## Commit history (M2.0–M2.2)

| Hash | Description |
|------|-------------|
| `fefde05` | M2.1 baseline pipeline (AUC=0.8952) |
| `8dcf3ca` | M2.1 reproducibility follow-up |
| `92638e3` | M2.1 closure + M2.2 prep |
| `c7e990a` | M2.2.0 cloud_coverage sanity check plan |
| `f8e349b` | M2.2.0 cloud_coverage fix; ΔAUC=0 (disconfirmed) |
| `23fd993` | M2.2.0 lesson learned: tree invariance narrows candidates |
| `4425f35` | M2.2.a ClusterNo ARI=1.0, n_init finding |
| `bd21952` | M2.2.a handoff doc |
| `ad0206b` | M2.2.b value-change 120 features; M2.2.c+d notebooks |
| `4e05312` | M2.2.c SavGol residual + M2.2.d dayofyear |
| `ad32a42` | M2.2.e 169-feature LightGBM val AUC 0.9818 (closes #9) |

---

## Next session plan: M2.3 (4-model ensemble)

### Prerequisites (already in place)

- `notebooks/05-m2-integration.ipynb` — full 169-feature pipeline, LightGBM AUC 0.9818
- All features: cloud_coverage fix → ClusterNo merge → sort → value-change (120) →
  SavGol residual → dayofyear → downsampling → CV split → StandardScaler

### Task

Add XGBoost, CatBoost, HistGBT; implement equal-weight ensemble per m2-plan.md M2.3.

```python
# XGBoost — native NaN support
model_xgb = XGBClassifier(n_estimators=100)
model_xgb.fit(X_train_scaled, y_train)

# CatBoost — verbose=False (not verbose=-1)
model_cat = CatBoostClassifier()
model_cat.fit(X_train_scaled, y_train, silent=True)

# HistGBT — NaN not supported; preprocess
model_hist = HistGradientBoostingClassifier()
model_hist.fit(np.nan_to_num(X_train_scaled), y_train)

# Ensemble
pred_ensemble = (pred_lgb + pred_xgb + pred_cat + pred_hist) / 4
```

### Expected results (paper Table 2)

| Model | Paper AUC | Our target |
|-------|-----------|-----------|
| LightGBM | 0.9849 | 0.9818 (done) |
| XGBoost | 0.9840 | ~0.98 |
| CatBoost | 0.9857 | ~0.98 |
| HistGBT | 0.9839 | ~0.98 |
| Ensemble | **0.9866** | **≥ 0.985** |

### Risks

- CatBoost 1,000 default iterations on CPU: 5–15 min wait
- HistGBT: `np.nan_to_num(X)` required before fit and predict
- XGBoost: native NaN support, no preprocessing needed

### Estimated time: 30–40 minutes

### Environment recovery

```bash
cd ~/projects/lead-reproduction
git pull
uv sync
claude
```

### Sync prompt for next session

> "讀 docs/m2-plan.md M2.3 section,
> docs/handoffs/2026-05-26-m22-milestone-completed.md,
> docs/unknowns.md (#10 候選清單 + #11/#12),
> notebooks/05-m2-integration.ipynb 的最終 cells.
> 報告 M2.3 工作範圍 + 第一步該做什麼. 純報告,不修改。"
