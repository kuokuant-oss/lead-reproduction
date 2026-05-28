"""
M2.5 Ablation Script — runs A, B, C on val pipeline.
Run from project root: uv run python notebooks/ablations_m25.py
Approx time: 10-15 min (mostly val pipeline setup + Ablation B 3x ensemble)
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.signal import savgol_filter
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw"

SEP = "=" * 65
sep = "-" * 50

# ============================================================
# Step 1: Val pipeline setup (mirrors Cells 1-17)
# ============================================================
print(SEP)
print("SETUP: Reproducing val pipeline")
print(SEP)

t_total = time.time()

# Load + cloud_coverage fix
train = pd.read_csv(DATA / "train_features.csv")
train["cloud_coverage"] = train["cloud_coverage"].replace({255: 10})
print(f"Train loaded: {train.shape}")

# ClusterNo
clusterno_df = pd.read_csv(ROOT / "data" / "interim" / "clusterno.csv")
train = train.merge(clusterno_df, on="building_id", how="left")
assert train["ClusterNo"].isna().sum() == 0

# Value-change 120 features (groupby.shift)
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
print(f"Value-change done: {time.time() - t0:.1f}s")

# SavGol
t0 = time.time()
results = []
for bid in train["building_id"].unique():
    tmp = train[train["building_id"] == bid].copy()
    smoothed = savgol_filter(tmp["meter_reading"].ffill().bfill().fillna(0), 5, 3)
    tmp["Residual_savgol_w5p3"] = tmp["meter_reading"] - smoothed
    results.append(tmp)
train = pd.concat(results).sort_index()
print(f"SavGol done: {time.time() - t0:.1f}s")

# dayofyear
ts = pd.to_datetime(train["timestamp"])
train["dayofyear"] = ts.dt.dayofyear + ts.dt.hour / 24

# Feature matrix (169 features)
drop_cols = ["anomaly", "wind_direction", "air_temperature_std_lag73"]
X_full = train.select_dtypes(include=["float", "int"])
X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns])
assert X_full.shape[1] == 169, f"Got {X_full.shape[1]}, expected 169"
feature_cols = X_full.columns.tolist()
print(f"Features: {len(feature_cols)} OK")

# Downsampling
neg = train[train["anomaly"] == 0]
pos = train[train["anomaly"] == 1]
negs1 = neg.sample(n=pos.shape[0], random_state=10)
negs2 = neg.sample(n=pos.shape[0], random_state=20)
df_eq = pd.concat([negs1, pos, negs2, pos], axis=0).reset_index(drop=True)
print(f"Downsampled: {df_eq.shape[0]:,} rows")

# CV split
X_eq = df_eq.select_dtypes(include=["float", "int"])
X_eq = X_eq.drop(columns=[c for c in drop_cols if c in X_eq.columns])
train_mask = df_eq["building_id"] % 5 < 4
val_mask = df_eq["building_id"] % 5 == 4
X_train = X_eq[train_mask]
y_train = df_eq.loc[train_mask, "anomaly"]
X_val = X_eq[val_mask]
y_val = df_eq.loc[val_mask, "anomaly"]

# Scaler + 4 models
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_va_s = scaler.transform(X_val)
X_tr_f = np.nan_to_num(X_tr_s)
X_va_f = np.nan_to_num(X_va_s)

print("Fitting baseline 4 models...")
t0 = time.time()
model_lgb = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
model_lgb.fit(X_tr_s, y_train)
pred_lgb = model_lgb.predict_proba(X_va_s)[:, 1]
auc_lgb = roc_auc_score(y_val, pred_lgb)
print(f"  LightGBM: {auc_lgb:.4f}  ({time.time() - t0:.1f}s)")

t0 = time.time()
model_xgb = xgb.XGBClassifier(
    n_estimators=100, eval_metric="logloss", verbosity=0, random_state=42
)
model_xgb.fit(X_tr_s, y_train)
pred_xgb = model_xgb.predict_proba(X_va_s)[:, 1]
auc_xgb = roc_auc_score(y_val, pred_xgb)
print(f"  XGBoost:  {auc_xgb:.4f}  ({time.time() - t0:.1f}s)")

t0 = time.time()
model_cat = CatBoostClassifier(iterations=1000, verbose=False, random_seed=42)
model_cat.fit(X_tr_s, y_train)
pred_cat = model_cat.predict_proba(X_va_s)[:, 1]
auc_cat = roc_auc_score(y_val, pred_cat)
print(f"  CatBoost: {auc_cat:.4f}  ({time.time() - t0:.1f}s)")

t0 = time.time()
model_hist = HistGradientBoostingClassifier(max_iter=100, random_state=42)
model_hist.fit(X_tr_f, y_train)
pred_hist = model_hist.predict_proba(X_va_f)[:, 1]
auc_hist = roc_auc_score(y_val, pred_hist)
print(f"  HistGBT:  {auc_hist:.4f}  ({time.time() - t0:.1f}s)")

pred_ensemble = (pred_lgb + pred_xgb + pred_cat + pred_hist) / 4
auc_ensemble = roc_auc_score(y_val, pred_ensemble)
print(f"  Ensemble: {auc_ensemble:.4f}  (expected 0.9830)")
assert abs(auc_ensemble - 0.9830) < 0.001, f"Baseline mismatch: {auc_ensemble:.4f}"

# Val post-processing vars (for Ablation C)
val_dayofyear = df_eq.loc[val_mask, "dayofyear"].values
val_building_id = df_eq.loc[val_mask, "building_id"].values

print(f"\nSetup complete: {(time.time() - t_total) / 60:.1f} min")

# ============================================================
# ABLATION A: gte_* feature removal
# ============================================================
print("\n" + SEP)
print("ABLATION A: gte_* feature removal (Unknown #5)")
print(SEP)

gte_cols = [c for c in feature_cols if c.startswith("gte_")]
non_gte_cols = [c for c in feature_cols if not c.startswith("gte_")]
print(f"Total features: {len(feature_cols)}")
print(f"gte_* features ({len(gte_cols)}): {gte_cols}")
print(f"Non-gte features: {len(non_gte_cols)}")

if len(gte_cols) == 0:
    print("\n⚠️  No gte_* features found — train_features.csv may not include them")
    auc_nogte = auc_lgb
    delta_gte = 0.0
else:
    X_tr_nogte = X_train[non_gte_cols]
    X_va_nogte = X_val[non_gte_cols]
    sc_nogte = StandardScaler()
    X_tr_nogte_s = sc_nogte.fit_transform(X_tr_nogte)
    X_va_nogte_s = sc_nogte.transform(X_va_nogte)

    lgb_nogte = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    lgb_nogte.fit(X_tr_nogte_s, y_train)
    pred_nogte = lgb_nogte.predict_proba(X_va_nogte_s)[:, 1]
    auc_nogte = roc_auc_score(y_val, pred_nogte)
    delta_gte = auc_nogte - auc_lgb

print(f"\nLightGBM with gte_*:    {auc_lgb:.4f}")
print(f"LightGBM without gte_*: {auc_nogte:.4f}")
print(f"ΔAUC (no_gte - with):   {delta_gte:+.4f}")
print(f"Significant (>±0.0005): {abs(delta_gte) > 0.0005}")
if delta_gte > 0.005:
    print("→ gte_* HARMFUL (removing improves AUC)")
elif delta_gte < -0.005:
    print("→ gte_* HELPFUL (removing hurts AUC)")
elif abs(delta_gte) > 0.0005:
    print("→ gte_* small effect (above noise floor)")
else:
    print("→ gte_* NEUTRAL (within noise floor ±0.0005)")

# ============================================================
# ABLATION B: impute_nulls
# ============================================================
print("\n" + SEP)
print("ABLATION B: impute_nulls 3 variants (Unknown #10)")
print(SEP)


def run_ensemble(X_tr, X_va, y_tr, y_va):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xva = sc.transform(X_va)
    m_lgb = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    m_xgb = xgb.XGBClassifier(
        n_estimators=100, eval_metric="logloss", verbosity=0, random_state=42
    )
    m_cat = CatBoostClassifier(iterations=1000, verbose=False, random_seed=42)
    m_hst = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    m_lgb.fit(Xtr, y_tr)
    m_xgb.fit(Xtr, y_tr)
    m_cat.fit(Xtr, y_tr)
    m_hst.fit(np.nan_to_num(Xtr), y_tr)
    p = (
        m_lgb.predict_proba(Xva)[:, 1]
        + m_xgb.predict_proba(Xva)[:, 1]
        + m_cat.predict_proba(Xva)[:, 1]
        + m_hst.predict_proba(np.nan_to_num(Xva))[:, 1]
    ) / 4
    return roc_auc_score(y_va, p)


# Variant 0: Raw NaN (sanity check — should match auc_ensemble)
t0 = time.time()
auc_raw = run_ensemble(X_train, X_val, y_train, y_val)
print(f"  Raw NaN (baseline):        {auc_raw:.4f}  ({time.time() - t0:.0f}s)")

# Variant 1: fillna(per-building mean) — buds-lab approach
X_tr_pbm = X_train.copy()
X_va_pbm = X_val.copy()
for col in feature_cols:
    if X_tr_pbm[col].isna().any():
        bm_tr = X_tr_pbm.groupby("building_id")[col].transform("mean")
        X_tr_pbm[col] = X_tr_pbm[col].fillna(bm_tr).fillna(0)
        bm_va = X_va_pbm.groupby("building_id")[col].transform("mean")
        X_va_pbm[col] = X_va_pbm[col].fillna(bm_va).fillna(0)
t0 = time.time()
auc_pbm = run_ensemble(X_tr_pbm, X_va_pbm, y_train, y_val)
print(f"  fillna(per-bldg mean):     {auc_pbm:.4f}  ({time.time() - t0:.0f}s)")

# Variant 2: fillna(0)
X_tr_zero = X_train.fillna(0)
X_va_zero = X_val.fillna(0)
t0 = time.time()
auc_zero = run_ensemble(X_tr_zero, X_va_zero, y_train, y_val)
print(f"  fillna(0):                 {auc_zero:.4f}  ({time.time() - t0:.0f}s)")

noise_floor = 0.0005
print(f"\nBaseline ensemble (M2.3):       {auc_ensemble:.4f}")
print(
    f"Δ(raw NaN vs M2.3):             {auc_raw - auc_ensemble:+.4f}  (sanity, expected ~0)"
)
print(
    f"Δ(per-bldg mean - raw):         {auc_pbm - auc_raw:+.4f}  {'⭐ significant' if abs(auc_pbm - auc_raw) > noise_floor else 'within noise floor'}"
)
print(
    f"Δ(fillna(0) - raw):             {auc_zero - auc_raw:+.4f}  {'⭐ significant' if abs(auc_zero - auc_raw) > noise_floor else 'within noise floor'}"
)

# ============================================================
# ABLATION C: Rule 2a filter on/off
# ============================================================
print("\n" + SEP)
print("ABLATION C: Rule 2a building_id filter (Unknown #15)")
print(SEP)

val_r2a_buds = (val_dayofyear == 1) & (
    (val_building_id > 145) | (val_building_id < 105)
)
val_r2a_blank = val_dayofyear == 1

print("Val side (expected 0 for both — downsampling artifact):")
print(f"  buds-lab filter (id>145|<105): {val_r2a_buds.sum():,} rows")
print(f"  blanket (all buildings):       {val_r2a_blank.sum():,} rows")

# Test side — load just building_id + timestamp from test CSV
test_slim = pd.read_csv(
    DATA / "test_features.csv", usecols=["building_id", "timestamp"]
)
ts_test = pd.to_datetime(test_slim["timestamp"])
test_slim["dayofyear"] = ts_test.dt.dayofyear + ts_test.dt.hour / 24

mask_buds = (test_slim["dayofyear"] == 1) & (
    (test_slim["building_id"] > 145) | (test_slim["building_id"] < 105)
)
mask_blank = test_slim["dayofyear"] == 1

print("\nTest side:")
print(f"  buds-lab filter (id>145|<105): {mask_buds.sum():,} rows  (expected 192)")
print(f"  blanket (all buildings):       {mask_blank.sum():,} rows")
print(f"  Difference:                    {mask_blank.sum() - mask_buds.sum():,} rows")

excl = sorted(
    set(test_slim[mask_blank]["building_id"].unique())
    - set(test_slim[mask_buds]["building_id"].unique())
)
print(f"\nBuildings excluded by filter (105≤id≤145): {len(excl)} unique")
print(f"  IDs: {excl}")

# ============================================================
# Summary
# ============================================================
print("\n" + SEP)
print("M2.5 ABLATION SUMMARY")
print(SEP)
print("\nAblation A (gte_* removal):")
print(f"  LightGBM with gte_*:    {auc_lgb:.4f}")
print(f"  LightGBM without gte_*: {auc_nogte:.4f}")
print(f"  ΔAUC:                   {delta_gte:+.4f}")

print("\nAblation B (impute_nulls):")
print(f"  Raw NaN:                {auc_raw:.4f}")
print(f"  fillna(per-bldg mean):  {auc_pbm:.4f}  Δ={auc_pbm - auc_raw:+.4f}")
print(f"  fillna(0):              {auc_zero:.4f}  Δ={auc_zero - auc_raw:+.4f}")

print("\nAblation C (Rule 2a filter):")
print(
    f"  Val buds-lab:  {val_r2a_buds.sum()} rows  |  blanket: {val_r2a_blank.sum()} rows"
)
print(f"  Test buds-lab: {mask_buds.sum()} rows  |  blanket: {mask_blank.sum()} rows")
print(f"  Excluded buildings: {excl}")

print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
