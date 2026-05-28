# -*- coding: utf-8 -*-
"""
M3 Baseline Exploration — ASHRAE GEPIII full dataset.
Run from project root: uv run python notebooks/m3_exploration.py

Dataset: data/raw/m3/
  train.csv              — 20.2M rows, 4 cols (building_id, meter, timestamp, meter_reading)
  bad_meter_readings.csv — 20.2M rows, 1 col (is_bad_meter_reading), row-aligned with train.csv
  building_metadata.csv  — 1449 buildings
  weather_train.csv      — site-level weather

Key difference from M2 (LEAD):
  M2: 406 buildings, test rows ~1.8M, anomaly label from competition, rate 2.13%
  M3: 1449 buildings, train rows ~20.2M, label from bad_meter_readings.csv, rate ~6.5%
"""

import sys
import warnings
import time
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
M3 = ROOT / "data" / "raw" / "m3"

SEP = "=" * 65

# ============================================================
# Cell 1: Load train data
# ============================================================
print(SEP)
print("Cell 1: Load train.csv")
print(SEP)
t0 = time.time()

train_raw = pd.read_csv(M3 / "train.csv")
print(f"  Shape: {train_raw.shape}")
print(f"  Columns: {train_raw.columns.tolist()}")
print(f"  dtypes:\n{train_raw.dtypes}")
print(f"  Load time: {time.time() - t0:.1f}s")

# ============================================================
# Cell 2: Load bad_meter_readings (row-aligned, not key-joined)
# ============================================================
print()
print(SEP)
print("Cell 2: Load bad_meter_readings.csv")
print(SEP)
t0 = time.time()

bad_readings = pd.read_csv(M3 / "bad_meter_readings.csv")
print(f"  Shape: {bad_readings.shape}")
print(f"  Columns: {bad_readings.columns.tolist()}")
print(f"  Value counts:\n{bad_readings['is_bad_meter_reading'].value_counts()}")
print(f"  Load time: {time.time() - t0:.1f}s")

# ============================================================
# Cell 3: Join label (positional — row-aligned, not key-joined)
# ============================================================
print()
print(SEP)
print("Cell 3: Join anomaly label")
print(SEP)

# bad_meter_readings is row-aligned with train.csv (not key-joined)
assert len(bad_readings) == len(train_raw), "Row count mismatch"
train_raw["anomaly"] = bad_readings["is_bad_meter_reading"].values
print(f"Anomaly rate: {train_raw['anomaly'].mean() * 100:.2f}%")
# 預期 ~6.5% (M2 是 2.13%)
print("  M2 (LEAD subset): 2.13%")
print(f"  M3 (GEPIII full): {train_raw['anomaly'].mean() * 100:.2f}%")

# ============================================================
# Cell 4: Building metadata
# ============================================================
print()
print(SEP)
print("Cell 4: Building metadata")
print(SEP)

meta = pd.read_csv(M3 / "building_metadata.csv")
print(f"  Shape: {meta.shape}")
print(f"  Columns: {meta.columns.tolist()}")
print(f"  Unique buildings: {meta['building_id'].nunique()}")
print(f"  Unique sites: {meta['site_id'].nunique()}")
print(f"  Primary use counts:\n{meta['primary_use'].value_counts().head(10)}")

# ============================================================
# Cell 5: Weather data
# ============================================================
print()
print(SEP)
print("Cell 5: Weather data")
print(SEP)

weather = pd.read_csv(M3 / "weather_train.csv")
print(f"  Shape: {weather.shape}")
print(f"  Columns: {weather.columns.tolist()}")
print(f"  Sites: {weather['site_id'].nunique()}")

# ============================================================
# Cell 6: Time range and basic stats
# ============================================================
print()
print(SEP)
print("Cell 6: Time range + basic stats")
print(SEP)

train_raw["timestamp"] = pd.to_datetime(train_raw["timestamp"])
print(f"  Date range: {train_raw['timestamp'].min()} to {train_raw['timestamp'].max()}")
print(f"  Unique buildings: {train_raw['building_id'].nunique()}")
print(f"  Unique meters: {train_raw['meter'].unique()}")
print(
    f"  meter_reading: min={train_raw['meter_reading'].min():.2f}, "
    f"max={train_raw['meter_reading'].max():.2f}, "
    f"mean={train_raw['meter_reading'].mean():.2f}"
)
print(f"  Null counts:\n{train_raw.isnull().sum()}")

# ============================================================
# Cell 7: Anomaly by meter type
# ============================================================
print()
print(SEP)
print("Cell 7: Anomaly rate by meter type")
print(SEP)

meter_labels = {0: "electricity", 1: "chilled_water", 2: "steam", 3: "hot_water"}
for meter_id, meter_name in meter_labels.items():
    sub = train_raw[train_raw["meter"] == meter_id]
    rate = sub["anomaly"].mean() * 100
    print(
        f"  Meter {meter_id} ({meter_name}): {len(sub):,} rows, anomaly rate {rate:.2f}%"
    )

# ============================================================
# Cell 8: Anomaly by building — top anomalous buildings
# ============================================================
print()
print(SEP)
print("Cell 8: Top anomalous buildings")
print(SEP)

bldg_anomaly = train_raw.groupby("building_id")["anomaly"].agg(["mean", "sum", "count"])
bldg_anomaly.columns = ["rate", "n_anomaly", "n_total"]
bldg_anomaly = bldg_anomaly.sort_values("rate", ascending=False)
print(f"  Buildings with 0% anomaly rate: {(bldg_anomaly['rate'] == 0).sum()}")
print(f"  Buildings with >50% anomaly rate: {(bldg_anomaly['rate'] > 0.5).sum()}")
print("\n  Top 10 most anomalous buildings:")
print(bldg_anomaly.head(10).to_string())

# ============================================================
# Cell 9: Monthly anomaly distribution
# ============================================================
print()
print(SEP)
print("Cell 9: Monthly anomaly distribution")
print(SEP)

train_raw["month"] = train_raw["timestamp"].dt.month
monthly = train_raw.groupby("month")["anomaly"].agg(["mean", "sum"])
monthly.columns = ["rate", "count"]
print(monthly.to_string())

# ============================================================
# Cell 10: Rows per building — M2 vs M3 comparison
# ============================================================
print()
print(SEP)
print("Cell 10: Dataset size comparison M2 vs M3")
print(SEP)

rows_per_bldg_m3 = train_raw.groupby("building_id").size()
print(
    f"  M3: rows per building — mean={rows_per_bldg_m3.mean():.0f}, "
    f"min={rows_per_bldg_m3.min()}, max={rows_per_bldg_m3.max()}"
)
print()
print("  M2 (LEAD subset):")
print("    buildings = 406")
print("    test rows = ~1,800,567")
print("    anomaly rate = 2.13%")
print()
print("  M3 (ASHRAE GEPIII):")
print(f"    buildings = {train_raw['building_id'].nunique()}")
print(f"    train rows = {len(train_raw):,}")
print(f"    anomaly rate = {train_raw['anomaly'].mean() * 100:.2f}%")
print("    meters per building = 1–4 (electricity always present)")

# ============================================================
# Cell 11: 思考點 — M3 vs M2 analysis
# ============================================================
print()
print(SEP)
print("Cell 11: 思考點 — M3 vs M2 key differences")
print(SEP)

m3_rate = train_raw["anomaly"].mean() * 100

print("1. Dataset scope:")
print("   M2: LEAD competition subset (406/1449 buildings, test only)")
print("   M3: Full ASHRAE GEPIII (all 1449 buildings, full year train)")
print()
print("2. Label source:")
print("   M2: Competition-provided anomaly label (strict definition)")
print("   M3: bad_meter_readings.csv (row-aligned, not key-joined)")
print()
print("3. Train/val split:")
print("   M2: Competition defines train/test boundary")
print("   M3: Self-defined split required (by building_id, per ADR 0001)")
print()
print("4. Meter types:")
print("   M2: Likely electricity-only or mixed (check LEAD paper)")
print("   M3: 4 types — electricity(0), chilled_water(1), steam(2), hot_water(3)")
print()
print("5. Weather join:")
print("   M2: Building-level weather already in train_features.csv")
print("   M3: Site-level weather_train.csv — needs building→site→weather join")
print()
print("6. Downsampling strategy:")
print("   M2: 50:50 normal:anomaly downsampling (ADR 0002)")
print(f"   M3: Need to re-evaluate — anomaly rate {m3_rate:.2f}% vs M2's 2.13%")
print(f"   At {m3_rate:.2f}%, extreme downsampling may discard useful patterns")
print()
print("7. Anomaly rate 差異 (M3 vs M2):")
print("   M2 (LEAD subset): 2.13%")
print(f"   M3 (GEPIII full): {train_raw['anomaly'].mean() * 100:.2f}%")
print(f"   差異 {(m3_rate - 2.13) / 2.13 * 100:.0f}% (M3 約 3× M2)")
print("   可能原因: LEAD 是 GEPIII subset + anomaly 定義更嚴")

# ============================================================
# Cell 12: M3 baseline planning notes
# ============================================================
print()
print(SEP)
print("Cell 12: M3 baseline planning")
print(SEP)

print("Proposed M3 approach (carry over from M2 pipeline):")
print("  1. Split by building_id (ADR 0001): 80/20 or leave-out by site")
print("  2. Downsample: ratio TBD — 6.5% anomaly rate changes the calculus")
print("  3. Value-change features: same 60 shifts as M2 (hourly ±24h + weekly)")
print("  4. SavGol: keep (M2.5 Ablation A showed no strong negative effect)")
print("  5. Weather join: building_metadata → site_id → weather_train")
print("  6. Post-processing: Rule 1 (meter=1.0) still applies; Rule 2 TBD")
print()
print("Open questions for M3 planning (→ unknowns.md):")
print("  - What is the correct train/val boundary for GEPIII?")
print("  - Does 6.5% anomaly rate warrant different downsampling ratio?")
print("  - Are all 4 meter types treated equally, or per-meter model?")
print("  - How to handle buildings with 0% anomaly rate?")

print()
print(SEP)
print("M3 exploration complete")
print(SEP)
