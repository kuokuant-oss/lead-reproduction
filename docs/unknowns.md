# Unknowns

論文沒有解釋清楚、重現時需要另外找資源或實驗才能確定的問題。

每個 unknown 列出:術語/問題、論文出現脈絡、為什麼重要。

---

## 1. Feature count 169 的完整組成

**術語**: 總特徵數 169(Table 3: "Raw, V-C (169)")

**狀態**: resolved ✓

**精確組成(B.1 確認)**:

| 來源 | 數量 | 說明 |
|------|------|------|
| Original 57 → 扣除 string cols | 46 | `select_dtypes(include=['float','int'])` 排除 8 個 object 欄(timestamp, primary_use, weekday_hour, building_weekday_hour, building_weekday, building_month, building_hour, building_meter);再 drop anomaly/wind_direction/air_temperature_std_lag73 |
| ClusterNo | 1 | K-means 分 10 群,Feature generator Cell 9 |
| Diff features | 60 | 60 shifts × `lag_value_{n}` |
| Ratio features | 60 | 60 shifts × `lag_value_ratio_{n}` |
| Residual_savgol_w5p3 | 1 | Savitzky-Golay 殘差,Feature generator Cells 13–14 |
| dayofyear | 1 | Modeling notebook Cell 2 新增(float = day + hour/24) |
| **Total** | **169** | ✓ |

**論文 §2.2.2 描述不精確**:以 "i.e." 列舉 sub-day shifts {1,2,3,23} 和 multi-day shifts {24,48,72,168},但同段也說 "shifts within one day were fully accounted for",暗示完整 1-day 涵蓋是設計意圖。代碼實際實作 60 個 shifts(sub-day {1..24} + multi-day {48,72,96,120,144,168})× 2 types = 120 value-change features。論文舉例不完整,但設計方向與代碼一致;以代碼為 ground truth。

**ADR**: 見 `docs/feature-engineering-rules.md` 完整 value-change 生成規則。

---

## 2. Validation split 的實作細節

**術語**: split by `building_id`

**狀態**: partially-resolved — split 機制已確認,建築數已量測;5-fold loop vs single-fold 待進一步確認

**出現脈絡**: §2.3.1 說「Train and validation datasets were split by building_id to ensure valid data were unseen during training」,且 validation score 與 leaderboard 差距 < 1%。但論文未說明:是單次 split 還是 k-fold?validation set 包含幾棟建築?用什麼 random seed?

**原始碼觀察(Modeling notebook Cell 6)**:
- 實作:`building_id % 5 < 4` → train;`building_id % 5 == 4` → validation
- **確定性切分,無 random seed** — building_id 模 5 等於 4 的建築固定為 validation
- 推論:modulo-5 本質上是 5-fold 結構,validation 約 40 棟建築(200 × 1/5 = 40)
- 但 notebook 只跑 fold 4(即 `% 5 == 4`),並非 5-fold cross-validation loop

**M2.1 measured**: train=162 buildings (119,520 rows), val=38 buildings (29,664 rows)
- 建築 ID 非連續(含 > 200 的 ID),導致 val 實際 38 棟而非估計的 40 棟
- 確認為 single holdout fold;5-fold loop 假設排除

**剩下待確認**: 確認 notebook 中是否有任何地方跑了多個 fold(目前判斷:無,為 single fold holdout)。

**為什麼重要**: Split 策略直接影響每次本地評估的可靠性。「差距 < 1%」這個特性也依賴此特定 split 設計才能重現。

---

## 3. Post-processing "start and end points" 的邊界定義

**術語**: start and end points of time series

**狀態**: resolved

**出現脈絡**: §2.4 Rule 2:「Set prediction to 0 (normal) for start and end points of time series.」論文未定義「start and end」是首尾各幾筆讀數,也未說明依據什麼邊界。

**原始碼觀察(Modeling notebook Cell 14)**:
- Rule 2 實際是兩個條件,非泛指「首尾幾筆」:
  1. `(dayofyear == 1) & ((building_id > 145) | (building_id < 105))` → 預測為 0
  2. `dayofyear > 366.9583` → 預測為 0(全年最後約 1 小時,即第 8,784 小時附近)
- 條件 1 排除特定建築 ID 範圍在 year day 1 的讀數(非所有建築)
- 條件 2 捕捉時間序列的最後一個(或最後幾個)時間點
- 論文的「start/end points」是對此具體 IF 條件的高度抽象描述

**ADR**: 見 `docs/adr/0004-postprocessing-hard-rules.md`(需補充更新)。

---

## 4. Downsampling 的作用域、策略與 random seed

**術語**: downsampling (of normal class)

**狀態**: resolved ✓

**出現脈絡**: §2.3.2 說「downsampling method for normal data is adopted so that normal and abnormal data can be balanced to occupy half of the training data.」未說明:是在整個 training set 做一次全域 downsampling?還是按建築做?還是在每個 CV fold 內重新抽?使用什麼 random seed?

**原始碼觀察(Modeling notebook Cell 3)**:
```python
negs1 = neg.sample(n=pos.shape[0], random_state=10)
negs2 = neg.sample(n=pos.shape[0], random_state=20)
df_eq = pd.concat([negs1, pos, negs2, pos], axis=0)
```
- Normal 資料被取樣兩次(seed 10、seed 20),各取 = anomaly 筆數
- Anomaly 資料被重複兩次
- 結果:`2 × pos.shape[0]` 正常 + `2 × pos.shape[0]` 異常 = 50:50,但**總量是 pos 的 4 倍**
- 全域一次操作(非按建築、非每 fold 重抽);在 CV split 之前執行
- Random seed 已確認:**10 和 20**

**M2.1 measured**:
- 原始 anomaly rate: **2.13%**(37,296 pos / 1,712,198 neg / 1,749,494 total)
- Downsampled df_eq: **149,184 rows**,class ratio = **50:50**(74,592 normal + 74,592 anomaly)
- 50:50 成立:2× neg samples(seed 10+20) + 2× pos = 等量正負
- 論文說「downsampling」不精確:neg 被 downsample,pos 被 upsample(×2);混合策略
- 全域一次操作,在 CV split 之前執行(已確認)

---

## 5. Target encoding 的 data leakage 防護方式

**術語**: target encoding

**狀態**: partially-resolved — encoding 機制已確認,leakage 影響尚待量化

**出現脈絡**: §2.2 Table 1 列出「Average values of the target variable aggregated by category (e.g., average values grouped by building_id)」作為一類特徵。若在 CV 切割前對完整訓練集計算 target encoding,validation set 的 label 資訊會洩漏進 training features,導致 validation AUC 虛高。論文未說明如何防範。

**原始碼觀察(`02_preprocess_data.py` lines 179–190)**:
- 使用 `GaussianTargetEncoder`,target 為 `log1p(meter_reading)`(非異常 label)
- 僅在 `good_train`(is_bad_meter_reading==0 的子集)上 `fit_transform`,再對全量 train/test `transform`
- **此 encoding 在 LEAD 競賽資料生成時已完成**,`train_features.csv` 的 `gte_*` 欄已是編碼後的值
- 重現時直接讀取 competition CSV 即可,**無需重算 target encoding**
- 能量讀數的平均值(非異常 label)不構成直接 label leakage,但 validation 建築的讀數統計仍有間接 leakage

**剩下待確認**: 量化此 `gte_*` leakage 對 validation AUC 的影響幅度;確認是否需要在 M2 重現中排除 `gte_*` 欄來測試純特徵貢獻。

**為什麼重要**: M2 直接使用 competition CSV 繞過了此問題;M3 從 GEPIII raw data 重建時需要決定是否沿用同一 GaussianTargetEncoder 策略。

---

## 6. 四個 GBDT 模型的超參數

**術語**: hyperparameters (LightGBM, XGBoost, CatBoost, HistGradientBoosting)

**狀態**: resolved ✓

**出現脈絡**: §2.3.3 提及四個模型各自訓練並做 hyperparameter tuning,但論文完全未揭露任何超參數數值(如 `num_leaves`、`learning_rate`、`max_depth`、`n_estimators` 等)。

**原始碼觀察(Modeling notebook Cells 8–11)**:

| Model | Constructor | 非預設設定 |
|-------|------------|-----------|
| XGBoost | `XGBClassifier(n_estimators=100)` | 無(100 是預設值) |
| LightGBM | `LGBMClassifier(n_estimators=100)` | 無(100 是預設值) |
| CatBoost | `CatBoostClassifier()` | 無(純預設,iterations=1000) |
| HistGBT | `HistGradientBoostingClassifier()` | 無(純預設,max_iter=100) |

**原始碼結果**:所有四個模型均使用庫預設超參數。論文 §2.3.3 以 "will be considered for modeling and hyperparameter tuning" 描述計畫意圖,未承諾調出非預設值;代碼最終使用預設值,與論文措辭不衝突。Train AUC 0.9999 來自特徵判別力 + balanced downsampling,而非特殊超參數設定。

**其他訓練設計**:無 early stopping;等權平均 ensemble(1/4 each);HistGBT 以 `np.nan_to_num()` 處理 NaN。

**ADR**: 見 `docs/feature-engineering-rules.md` "Model hyperparameters" 章節。

---

## 7. LEAD 比賽資料的上游 pipeline

**術語**: `train_features.csv` 的 57 欄來源

**狀態**: resolved ✓

**出現脈絡**: LEAD 競賽直接提供 `train_features.csv`(57 欄,含 building meta、weather、temporal、target encoding),但未說明這 57 欄是如何產生的。LEAD-1st-solution 的 Feature generator notebook 讀取這個 CSV 作為輸入,只在其上疊加 value-change features,沒有從更底層的 raw data 重新產生。

**確認結果**: `02_preprocess_data.py` 是 57 欄的直接上游來源。

**欄位對應**(`02_preprocess_data.py` → LEAD CSV,全部吻合):
- `meter_reading`、`site_id`、`building_id`、`meter`(建築/計量基本欄)
- `square_feet`、`year_built`、`floor_count`、`primary_use`(building meta)
- `air_temperature`、`dew_temperature`、`sea_level_pressure`、`wind_speed` 等氣象欄 + `had_*` NA 指示欄
- `air_temperature_mean_lag7`、`air_temperature_mean_lag73` 等滾動統計
- `hour`、`weekday`、`month`、`year`、`hour_x`/`y`、`month_x`/`y`、`weekday_x`/`y`(時間特徵)
- `weekday_hour`、`building_weekday_hour`、`building_weekday`、`building_month`、`building_hour`、`building_meter`(string 互動欄)
- `is_holiday`、`is_bad_meter_reading`(→ LEAD 改名為 `anomaly`)
- 16 個 `gte_*` target encoding 欄(GaussianTargetEncoder on `log1p(meter_reading)`)

**三個差異點**:
1. `meter` 欄:原始碼有,LEAD CSV 無(比賽資料僅一種 meter 或已合入 building_id)
2. `is_bad_meter_reading` → LEAD 改名為 `anomaly`
3. `had_*` 天氣 NA 指示欄:原始碼產生,LEAD CSV 無(或已刪除)

**M3 含義**: 從 GEPIII raw data 重建時,執行 `02_preprocess_data.py` 即可得到與 LEAD 相同的特徵基礎;差異點需補充處理。

**參考**:
- `02_preprocess_data.py`: https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/scripts/02_preprocess_data.py
- `bad_meter_readings.zip`: https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/input/bad_meter_readings.zip

---

Last reviewed: 2026-05-26 (Issue #6: resolved #6; #2/#4/#5 partially-resolved deferred to M2)
