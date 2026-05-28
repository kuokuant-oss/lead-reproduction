# -*- coding: utf-8 -*-
"""M2.5 Ablation C only — Rule 2a filter on/off. Fast, no model fitting."""

import sys
import pandas as pd
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw"

print("=" * 60)
print("ABLATION C: Rule 2a building_id filter (Unknown #15)")
print("=" * 60)

# Val side (reproduce from known data)
# From notebook Cell 18: val has 0 dayofyear==1 rows (downsampling artifact)
# We verify by loading df_eq[val_mask] dayofyear
train = pd.read_csv(DATA / "train_features.csv")
ts = pd.to_datetime(train["timestamp"])
train["dayofyear"] = ts.dt.dayofyear + ts.dt.hour / 24

neg = train[train["anomaly"] == 0]
pos = train[train["anomaly"] == 1]
negs1 = neg.sample(n=pos.shape[0], random_state=10)
negs2 = neg.sample(n=pos.shape[0], random_state=20)
df_eq = pd.concat([negs1, pos, negs2, pos], axis=0).reset_index(drop=True)

val_mask = df_eq["building_id"] % 5 == 4
val_dayofyear = df_eq.loc[val_mask, "dayofyear"].values
val_building_id = df_eq.loc[val_mask, "building_id"].values

val_r2a_buds = (val_dayofyear == 1) & (
    (val_building_id > 145) | (val_building_id < 105)
)
val_r2a_blank = val_dayofyear == 1

print("\nVal side (downsampling artifact expected: both = 0):")
print(f"  buds-lab filter (id>145|<105): {val_r2a_buds.sum():,} rows")
print(f"  blanket (all buildings):       {val_r2a_blank.sum():,} rows")
print(f"  Val dayofyear==1 rows total:   {(val_dayofyear == 1).sum():,}")

# Test side
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
print(f"  Difference (filter excluded):  {mask_blank.sum() - mask_buds.sum():,} rows")

excl_bldgs = sorted(
    set(test_slim[mask_blank]["building_id"].unique())
    - set(test_slim[mask_buds]["building_id"].unique())
)
print(f"\nBuildings excluded by filter (105<=id<=145): {len(excl_bldgs)} unique")
print(f"  IDs: {excl_bldgs}")
