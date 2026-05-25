# Feature Engineering Rules

LEAD-1st-solution 的特徵生成規則,由 Feature generator notebook 原始碼推導。

---

## 輸入

`train_features.csv` — 57 欄,由 GEPIII `02_preprocess_data.py` 產出。
見 `docs/unknowns.md` #7 確認此上游關係。

---

## 特徵總覽(169 欄)

| 來源 | 數量 | 說明 |
|------|------|------|
| 57 欄保留的數值欄 | 46 | `select_dtypes(include=['float','int'])` 排除 8 個 object 欄;再 drop 3 欄 |
| ClusterNo | 1 | K-means 分群 |
| Diff features | 60 | `lag_value_{n}` |
| Ratio features | 60 | `lag_value_ratio_{n}` |
| Residual_savgol_w5p3 | 1 | Savitzky-Golay 殘差 |
| dayofyear | 1 | 連續日期(含小時分數) |
| **Total** | **169** | |

---

## 排除規則

### 排除的 object 欄(8 欄)

`select_dtypes(include=['float','int'])` 自動排除:

- `timestamp`
- `primary_use`
- `weekday_hour`
- `building_weekday_hour`
- `building_weekday`
- `building_month`
- `building_hour`
- `building_meter`

### 明確 drop 的欄(3 欄)

- `anomaly` — 目標變數,不可作為特徵
- `wind_direction` — notebook Cell 10 明確 drop
- `air_temperature_std_lag73` — notebook Cell 10 明確 drop

---

## 新增特徵

### ClusterNo(Cell 9)

K-means 以 `meter_reading` 為輸入,`n_clusters=10`,結果存為 `ClusterNo`(int)。

### Value-change features(Cells 11–12)

**基礎欄**: `meter_reading` 唯一

**Shift 集合**(60 個值,單位:小時):

```python
shifts = (
    list(np.arange(-24, 0))      # -24 到 -1,共 24 個
    + list(np.arange(1, 25))     # 1 到 24,共 24 個
    + list(np.arange(-168, -24, 24))  # -168, -144, ..., -48,共 6 個
    + list(np.arange(48, 169, 24))    # 48, 72, ..., 168,共 6 個
)
# 共 24+24+6+6 = 60 個 shift
```

**公式**:

對每個 shift `n`(正 = 向未來位移,負 = 向過去位移):

```python
lag_value_{n}       = X(t - n) - X(t)           # diff
lag_value_ratio_{n} = (X(t - n) + 1) / (X(t) + 1)  # ratio
```

注意:公式方向與論文 §2.2 相反。論文寫 `X(t) - X(t-s)`,原始碼實際計算 `X(t-n) - X(t)`。
程式碼定義的 `lag_value` 代表「目前讀數相對於平移後讀數的差距」。

**NaN 處理**: 無額外填補。位移超出時間序列邊界的位置自然產生 NaN(left merge 語意)。

**欄名**: `lag_value_{n}` 和 `lag_value_ratio_{n}`,其中 `n` 為整數(含負值)。例:
`lag_value_-1`、`lag_value_1`、`lag_value_-168`、`lag_value_ratio_48`

### Residual_savgol_w5p3(Cells 13–14)

對 `meter_reading` 套用 Savitzky-Golay 濾波(window=5, polyorder=3),殘差存為 `Residual_savgol_w5p3`。

### dayofyear(Modeling notebook Cell 2)

```python
dayofyear = timestamp_as_day_of_year + hour / 24
```

連續浮點數,結合日期與時刻資訊。

---

## 注意事項

- 論文 §2.2.2 以 "i.e." 列舉 shift 舉例(sub-day {1,2,3,23},multi-day {24,48,72,168}),描述不精確。
  代碼 ground truth:60 shifts × 2 types = 120 value-change features。
- Shift 集合設計的物理意義:短期(±1–24h)捕捉日週期,長期(±48–168h,間隔 24h)捕捉週週期。

---

## Model hyperparameters

四個 GBDT 模型的超參數設定,由 Modeling notebook Cells 8–11 提取。

**原始碼結果:四個模型均使用各 GBDT 庫預設值(LightGBM/XGBoost: n_estimators=100,CatBoost: iterations=1000,HistGBT: max_iter=100)。論文 §2.3.3 提到 "hyperparameter tuning will be considered",但代碼中未見對應實作。Train AUC 0.9999 的高擬合來自特徵判別力 + balanced downsampling。**

| Model | Constructor call | 明確設定參數 | 值 | 庫預設 | 備註 |
|-------|-----------------|-------------|---|--------|------|
| XGBoost | `XGBClassifier(n_estimators=100)` | `n_estimators` | 100 | 100 | 與預設相同,明確寫出 |
| LightGBM | `LGBMClassifier(n_estimators=100)` | `n_estimators` | 100 | 100 | 與預設相同,明確寫出 |
| CatBoost | `CatBoostClassifier()` | 無 | — | iterations=1000 | 純預設;`.fit(silent=True)` 僅抑制輸出 |
| HistGBT | `HistGradientBoostingClassifier()` | 無 | — | max_iter=100 | 純預設;輸入以 `np.nan_to_num()` 預處理 |

### 訓練流程的關鍵設計(非論文揭露)

| 設計 | 說明 |
|------|------|
| StandardScaler | Cell 7:全部特徵在 train 上 `fit_transform`,val/test 只 `transform`。GBDT 理論上不需要正規化,此處可能是保守做法或遺留設定。 |
| Ensemble 權重 | 四個模型預測值**等權平均**(各 1/4),非加權或 stacking。Cell 12–13。 |
| Early stopping | **無**。所有模型跑完預設輪數後直接評估。 |
| NaN 處理 | HistGradientBoosting 使用 `np.nan_to_num(X)`(NaN → 0.0)。其餘三個模型直接接受 NaN。 |
| Final refit | Cell 13:在 `X_all`(train+val 合併,仍有 downsampled 結構)上重新訓練全部四個模型後再預測 test。 |

---

Last updated: 2026-05-26
