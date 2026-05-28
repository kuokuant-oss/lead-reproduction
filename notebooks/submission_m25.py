# -*- coding: utf-8 -*-
"""
M2.5 Ablation Submission Generator — produces 3 ablation CSVs for Kaggle.
Run from project root: uv run python notebooks/submission_m25.py

Ablation A: LightGBM only, no gte_* features
Ablation B: 4-model ensemble with per-bldg mean impute (buds-lab style)
Ablation C: M2.4 baseline + blanket Rule 2a (14 extra rows set to 0)

Note: test value-change uses groupby.shift (vs timestamp-merge in M2.4).
This is acceptable for ablation research data (not competition entry).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.signal import savgol_filter
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SEP = "=" * 65

# ============================================================
# Step 1: Train/val pipeline (replicate Cell 1-17 + 24 vars)
# ============================================================
print(SEP)
print("STEP 1: Train pipeline")
print(SEP)
t_total = time.time()

train = pd.read_csv(DATA / "train_features.csv")
train["cloud_coverage"] = train["cloud_coverage"].replace({255: 10})

clusterno_df = pd.read_csv(ROOT / "data" / "interim" / "clusterno.csv")
train = train.merge(clusterno_df, on="building_id", how="left")

train = train.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

shifts = (
    list(np.arange(-24, 0))
    + list(np.arange(1, 25))
    + list(np.arange(-168, -24, 24))
    + list(np.arange(48, 169, 24))
)
assert len(shifts) == 60

t0 = time.time()
new_cols = {}
for n in shifts:
    shifted = train.groupby("building_id")["meter_reading"].shift(n)
    new_cols[f"lag_value_{n}"] = shifted - train["meter_reading"]
    new_cols[f"lag_value_ratio_{n}"] = (shifted + 1) / (train["meter_reading"] + 1)
train = pd.concat([train, pd.DataFrame(new_cols, index=train.index)], axis=1)
print(f"  Train value-change: {time.time() - t0:.1f}s")

t0 = time.time()
results = []
for bid in train["building_id"].unique():
    tmp = train[train["building_id"] == bid].copy()
    from scipy.signal import savgol_filter as _sgf

    smoothed = _sgf(tmp["meter_reading"].ffill().bfill().fillna(0), 5, 3)
    tmp["Residual_savgol_w5p3"] = tmp["meter_reading"] - smoothed
    results.append(tmp)
train = pd.concat(results).sort_index()
print(f"  Train SavGol: {time.time() - t0:.1f}s")

ts = pd.to_datetime(train["timestamp"])
train["dayofyear"] = ts.dt.dayofyear + ts.dt.hour / 24

drop_cols = ["anomaly", "wind_direction", "air_temperature_std_lag73"]
X_full = train.select_dtypes(include=["float", "int"])
X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns])
assert X_full.shape[1] == 169, f"Expected 169, got {X_full.shape[1]}"
feature_cols = X_full.columns.tolist()

neg = train[train["anomaly"] == 0]
pos = train[train["anomaly"] == 1]
negs1 = neg.sample(n=pos.shape[0], random_state=10)
negs2 = neg.sample(n=pos.shape[0], random_state=20)
df_eq = pd.concat([negs1, pos, negs2, pos], axis=0).reset_index(drop=True)
target_all = df_eq["anomaly"]
print(f"  Downsampled: {df_eq.shape[0]:,} rows")
print(f"Step 1 done: {(time.time() - t_total):.1f}s")

# ============================================================
# Step 2: Test feature engineering (groupby.shift for speed)
# ============================================================
print()
print(SEP)
print("STEP 2: Test feature engineering")
print(SEP)
t0 = time.time()

test_raw = pd.read_csv(DATA / "test_features.csv")
test_raw["cloud_coverage"] = test_raw["cloud_coverage"].replace({255: 10})
test_raw = test_raw.merge(clusterno_df, on="building_id", how="left")
test_raw = test_raw.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

new_cols_test = {}
for n in shifts:
    shifted_t = test_raw.groupby("building_id")["meter_reading"].shift(n)
    new_cols_test[f"lag_value_{n}"] = shifted_t - test_raw["meter_reading"]
    new_cols_test[f"lag_value_ratio_{n}"] = (shifted_t + 1) / (
        test_raw["meter_reading"] + 1
    )
test_raw = pd.concat(
    [test_raw, pd.DataFrame(new_cols_test, index=test_raw.index)], axis=1
)
print(f"  Test value-change: {time.time() - t0:.1f}s")

t0 = time.time()
results_t = []
for bid in test_raw["building_id"].unique():
    tmp = test_raw[test_raw["building_id"] == bid].copy()
    smoothed = savgol_filter(tmp["meter_reading"].ffill().bfill().fillna(0), 5, 3)
    tmp["Residual_savgol_w5p3"] = tmp["meter_reading"] - smoothed
    results_t.append(tmp)
test_raw = pd.concat(results_t).sort_index()
print(f"  Test SavGol: {time.time() - t0:.1f}s")

ts_t = pd.to_datetime(test_raw["timestamp"])
test_raw["dayofyear"] = ts_t.dt.dayofyear + ts_t.dt.hour / 24

check = test_raw[feature_cols]
assert check.shape == (1_800_567, 169), f"Unexpected test shape: {check.shape}"
print(f"  Test features: {check.shape} OK")

# Post-processing masks
mask_r1_test = test_raw["meter_reading"].values == 1.0
mask_r2a_test = (test_raw["dayofyear"].values == 1) & (
    (test_raw["building_id"].values > 145) | (test_raw["building_id"].values < 105)
)
mask_r2b_test = test_raw["dayofyear"].values > 366.9583
print(
    f"  Rule 1 trigger: {mask_r1_test.sum():,}  Rule 2a: {mask_r2a_test.sum():,}  Rule 2b: {mask_r2b_test.sum():,}"
)

print(f"Step 2 done: {(time.time() - t_total):.1f}s total")

# ============================================================
# Step 3: Ablation A — LightGBM, no gte_*
# ============================================================
print()
print(SEP)
print("STEP 3: Ablation A submission (LightGBM, no gte_*)")
print(SEP)
t0 = time.time()

gte_cols = [c for c in feature_cols if c.startswith("gte_")]
non_gte_cols = [c for c in feature_cols if not c.startswith("gte_")]
print(f"  gte_* removed: {len(gte_cols)}  features remaining: {len(non_gte_cols)}")

features_all_nogte = df_eq[non_gte_cols].copy()
sc_a = StandardScaler()
X_all_nogte = sc_a.fit_transform(features_all_nogte)
X_test_nogte = sc_a.transform(test_raw[non_gte_cols])

lgb_a = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
lgb_a.fit(X_all_nogte, target_all)
pred_a = lgb_a.predict_proba(X_test_nogte)[:, 1]

pred_a_pp = pred_a.copy()
pred_a_pp[mask_r1_test] = 1
pred_a_pp[mask_r2a_test] = 0
pred_a_pp[mask_r2b_test] = 0

sub_a = pd.DataFrame({"row_id": test_raw["row_id"].values, "anomaly": pred_a_pp})
assert len(sub_a) == 1_800_567
sub_a.to_csv(PROC / "submission_m2_5_ablation_a_nogte.csv", index=False)
print(
    f"  Saved: submission_m2_5_ablation_a_nogte.csv  ({len(sub_a):,} rows)  {time.time() - t0:.0f}s"
)

# ============================================================
# Step 4: Ablation B — 4-model, per-bldg mean impute
# ============================================================
print()
print(SEP)
print("STEP 4: Ablation B submission (per-bldg mean impute)")
print(SEP)
t0 = time.time()

features_all_pbm = df_eq[feature_cols].copy()
for col in feature_cols:
    if features_all_pbm[col].isna().any():
        bm = features_all_pbm.groupby("building_id")[col].transform("mean")
        features_all_pbm[col] = features_all_pbm[col].fillna(bm).fillna(0)
print(f"  Train imputation done: {time.time() - t0:.0f}s")

t1 = time.time()
test_pbm = test_raw[feature_cols].copy()
for col in feature_cols:
    if test_pbm[col].isna().any():
        bm_t = test_pbm.groupby(test_raw["building_id"].values)[col].transform("mean")
        test_pbm[col] = test_pbm[col].fillna(bm_t).fillna(0)
print(f"  Test imputation done: {time.time() - t1:.0f}s")

sc_b = StandardScaler()
X_all_pbm = sc_b.fit_transform(features_all_pbm)
X_test_pbm = sc_b.transform(test_pbm)

print("  Refitting 4 models...")
t1 = time.time()
lgb_b = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
lgb_b.fit(X_all_pbm, target_all)
xgb_b = xgb.XGBClassifier(
    n_estimators=100, eval_metric="logloss", verbosity=0, random_state=42
)
xgb_b.fit(X_all_pbm, target_all)
cat_b = CatBoostClassifier(iterations=1000, verbose=False, random_seed=42)
cat_b.fit(X_all_pbm, target_all)
hist_b = HistGradientBoostingClassifier(max_iter=100, random_state=42)
hist_b.fit(np.nan_to_num(X_all_pbm), target_all)
print(f"  Models done: {time.time() - t1:.0f}s")

pred_b = (
    lgb_b.predict_proba(X_test_pbm)[:, 1]
    + xgb_b.predict_proba(X_test_pbm)[:, 1]
    + cat_b.predict_proba(X_test_pbm)[:, 1]
    + hist_b.predict_proba(np.nan_to_num(X_test_pbm))[:, 1]
) / 4

pred_b_pp = pred_b.copy()
pred_b_pp[mask_r1_test] = 1
pred_b_pp[mask_r2a_test] = 0
pred_b_pp[mask_r2b_test] = 0

sub_b = pd.DataFrame({"row_id": test_raw["row_id"].values, "anomaly": pred_b_pp})
assert len(sub_b) == 1_800_567
sub_b.to_csv(PROC / "submission_m2_5_ablation_b_pbmmean.csv", index=False)
print(
    f"  Saved: submission_m2_5_ablation_b_pbmmean.csv  ({len(sub_b):,} rows)  {time.time() - t0:.0f}s"
)

# ============================================================
# Step 5: Ablation C — M2.4 baseline + blanket Rule 2a
# ============================================================
print()
print(SEP)
print("STEP 5: Ablation C submission (blanket Rule 2a)")
print(SEP)
t0 = time.time()

# Apply blanket Rule 2a to ensemble (starting fresh, not from M2.4 file)
# We have test_raw loaded, so compute ensemble predictions from scratch
# using the standard X_all approach

# Build X_all and refit 4 models (no imputation — raw NaN baseline)
features_all = df_eq[feature_cols].copy()
sc_c = StandardScaler()
X_all = sc_c.fit_transform(features_all)
X_test_s = sc_c.transform(test_raw[feature_cols])
X_test_f = np.nan_to_num(X_test_s)

print("  Refitting 4 baseline models (for pred_ensemble_test)...")
t1 = time.time()
lgb_c = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
lgb_c.fit(X_all, target_all)
xgb_c = xgb.XGBClassifier(
    n_estimators=100, eval_metric="logloss", verbosity=0, random_state=42
)
xgb_c.fit(X_all, target_all)
cat_c = CatBoostClassifier(iterations=1000, verbose=False, random_seed=42)
cat_c.fit(X_all, target_all)
hist_c = HistGradientBoostingClassifier(max_iter=100, random_state=42)
hist_c.fit(np.nan_to_num(X_all), target_all)
print(f"  Models done: {time.time() - t1:.0f}s")

pred_ens_test = (
    lgb_c.predict_proba(X_test_s)[:, 1]
    + xgb_c.predict_proba(X_test_s)[:, 1]
    + cat_c.predict_proba(X_test_s)[:, 1]
    + hist_c.predict_proba(X_test_f)[:, 1]
) / 4

# Apply Rules with BLANKET Rule 2a
pred_c_pp = pred_ens_test.copy()
pred_c_pp[mask_r1_test] = 1
mask_r2a_blank = test_raw["dayofyear"].values == 1  # blanket: all buildings
pred_c_pp[mask_r2a_blank] = 0
pred_c_pp[mask_r2b_test] = 0

print(
    f"  Rule 1: {mask_r1_test.sum():,}  Rule 2a blanket: {mask_r2a_blank.sum():,}  Rule 2b: {mask_r2b_test.sum():,}"
)
print(
    f"  Extra rows vs buds-lab filter: {mask_r2a_blank.sum() - mask_r2a_test.sum():,}"
)

sub_c = pd.DataFrame({"row_id": test_raw["row_id"].values, "anomaly": pred_c_pp})
assert len(sub_c) == 1_800_567
sub_c.to_csv(PROC / "submission_m2_5_ablation_c_blanket.csv", index=False)
print(
    f"  Saved: submission_m2_5_ablation_c_blanket.csv  ({len(sub_c):,} rows)  {time.time() - t0:.0f}s"
)

# ============================================================
# Final check
# ============================================================
print()
print(SEP)
print("DONE — 3 ablation CSVs generated")
print(SEP)
for fname in [
    "submission_m2_5_ablation_a_nogte.csv",
    "submission_m2_5_ablation_b_pbmmean.csv",
    "submission_m2_5_ablation_c_blanket.csv",
]:
    path = PROC / fname
    df = pd.read_csv(path)
    r1 = (df["anomaly"] == 1.0).sum()
    r0 = (df["anomaly"] == 0.0).sum()
    print(f"  {fname}")
    print(
        f"    rows={len(df):,}  hard_1={r1:,}  hard_0={r0:,}  mean={df['anomaly'].mean():.4f}"
    )

print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
