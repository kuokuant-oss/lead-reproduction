# Handoff: M2.2.a ClusterNo Complete

**Date**: 2026-05-26
**Status**: ✅ Complete — ready for M2.2.b

---

## What was accomplished

M2.2.a implemented ClusterNo (per-building shape clustering) and achieved
**ARI=1.0** numeric alignment with buds-lab's reference labels.

### Artifacts created

| File | Description |
|------|-------------|
| `notebooks/02-m2-clusterno.ipynb` | ClusterNo implementation (9 code cells) |
| `data/interim/clusterno.csv` | 406 buildings × ClusterNo label (int32, 0–9) |

### Key numbers

- Pivot shape: (8784, 406) — 8784 hourly timestamps × 406 buildings
- NaN ratio: 6.13% → 15.48% after log1p (expected; z < -1 produces NaN)
- KMeans inertia: 2,598,263.45, n_iter_=16 (converged, < 10000 max)
- **ARI = 1.0** (406/406 buildings match buds-lab reference exactly)

---

## Critical finding: sklearn n_init version trap

sklearn 1.4+ changed `n_init='auto'` with k-means++ initialization to mean
**n_init=1** (was 10 in sklearn ≤ 1.3, which is what buds-lab used in 2022).

Without explicit `n_init=10`:
- KMeans lands in different local minimum
- ARI = 0.503 (wrong cluster assignments)

With `n_init=10` explicit:
- ARI = 1.0 (perfect alignment)

**Rule for M2**: All sklearn estimators must explicitly set version-sensitive
default parameters. See `docs/unknowns.md` #10 "Lesson learned (M2.2.a)".

---

## Preprocessing chain (order is critical)

```python
# 1. Pivot: timestamp × building_id (train + test combined = 406 buildings)
merged = pd.concat([train_features, test_features], axis=0, ignore_index=True)
pivot = merged.pivot_table(index='timestamp', columns='building_id', values='meter_reading')

# 2. (z-score + ±10σ clip) × 2  ← FIRST
for _ in range(2):
    pivot = (pivot - pivot.mean()) / pivot.std()
    pivot = pivot[pivot < 10]
    pivot = pivot[pivot > -10]

# 3. log1p  ← AFTER z-score (z < -1 → NaN; handled by fillna(0) below)
pivot = np.log1p(pivot)

# 4. Transpose + StandardScaler + fillna(0)
df_buildings = pivot.T
X_cluster = StandardScaler().fit_transform(df_buildings.fillna(0))

# 5. KMeans — n_init=10 EXPLICIT (sklearn 1.4+ 'auto' = 1!)
km = KMeans(n_clusters=10, max_iter=10000, random_state=666, n_init=10)
labels = km.fit_predict(X_cluster)
```

The wrong order (log1p before z-score) was in the original plan and was caught
during systematic notebook review before implementation.

---

## Train cluster distribution (200 buildings)

| ClusterNo | Count |
|-----------|-------|
| 0 | 19 |
| 1 | 44 |
| 2 | 2 |
| 3 | 10 |
| 4 | 18 |
| 5 | 14 |
| 6 | 2 |
| 7 | 32 |
| 8 | 37 |
| 9 | 22 |

---

## Where things stand

### AUC progression

| Step | AUC | Features |
|------|-----|----------|
| M2.1 baseline | 0.8952 | 57 |
| M2.2.0 cloud_coverage fix | 0.8952 (ΔAUC=0) | 57 |
| M2.2.a ClusterNo | — | +1 (not yet integrated) |

ClusterNo is a prerequisite feature; AUC jump happens at M2.2.e integration.

### M2.1 gap status

Val AUC 0.8952 vs paper baseline 0.9311 (gap 3.86%). Candidates:

1. **impute_nulls** (PRIMARY SUSPECT) — per-building mean imputation skipped
2. CV fold variance
3. Downsampling seed variance
4. LightGBM version differences

All deferred to M2.5 ablation.

---

## Next session: M2.2.b value-change features

### Quick context recovery

```bash
cd ~/projects/lead-reproduction && git pull && uv sync
```

Read: `docs/m2-plan.md` M2.2.b section

### M2.2.b task

Implement 120 value-change features via `groupby().shift()`:

```python
shifts = (
    list(np.arange(-24, 0)) + list(np.arange(1, 25))
    + list(np.arange(-168, -24, 24)) + list(np.arange(48, 169, 24))
)  # 60 shifts
for n in shifts:
    train_features[f'lag_value_{n}'] = (
        train_features.groupby('building_id')['meter_reading'].shift(n)
        - train_features['meter_reading']
    )
    train_features[f'lag_value_ratio_{n}'] = (
        (train_features.groupby('building_id')['meter_reading'].shift(n) + 1)
        / (train_features['meter_reading'] + 1)
    )
```

Alignment strategy: **structural only** (shape correct + direction correct).
No numeric alignment required for value-change features.

Done when:
- `lag_value_*` × 60 + `lag_value_ratio_*` × 60 = 120 new columns confirmed
- Spot-check building_id=0 lag_value_1 first 5 rows: values = shifted - original
- Verify LEAD data has no time gaps (per-building row count = 8,784)

### Remaining M2.2 steps after M2.2.b

- M2.2.c: SavGol residual (`Residual_savgol_w5p3`)
- M2.2.d: dayofyear float feature
- M2.2.e: Integration + LightGBM val AUC ≥ 0.97

### Commits

| Hash | Description |
|------|-------------|
| fefde05 | M2.1 baseline pipeline (AUC=0.8952) |
| 8dcf3ca | M2.1 reproducibility follow-up |
| 92638e3 | M2.1 closure + M2.2 prep |
| (M2.2.0) | cloud_coverage fix + ΔAUC=0 |
| (M2.2.a docs) | 4425f35 — ClusterNo ARI=1.0, n_init finding |
