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

**M2.2.e 觀察 (2026-05-26)**:
我們的 top 10 importance 缺 `gte_building_id`(論文 Fig 5 rank #7)跟
`gte_meter_primary_use`(論文 Fig 5 rank #10)。overlap = 8/10。
差異可能來自:(a) gte encoding 在我們的 169-feature 版本中排名 11–20,略低於前 10;
(b) importance 的計算可能依賴 downsampling seed 或 split 的 random ordering。
M2.5 ablation(移除全部 gte 欄)會直接量化影響。

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

**M2.3 設計觀察 (2026-05-26)**:

- buds-lab 4 model 都用 library default，但 default 本身不對稱：
  - LightGBM: n_estimators=100
  - XGBoost: n_estimators=100
  - HistGBT: max_iter=100
  - **CatBoost: iterations=1000** (10x 其他三個)
- Paper §2.3.3 沒 justify 這個不對稱，可能是「忘記統一」而非「故意設計」
- 對 final ensemble AUC 的影響未知

**M2.5 ablation 候選 #13**:
- Run CatBoost with iterations=100 (對齊其他 model)
- 量化 ΔAUC vs iterations=1000
- 若 ΔAUC < 0.001 → CatBoost 1000 對 ensemble 沒顯著貢獻，paper 設計有 implicit redundancy
- 若 ΔAUC > 0.003 → CatBoost 1000 是 ensemble AUC 提升關鍵，paper 應該 justify

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

## 8. Anomaly rate: 論文敘述與實測差距

**術語**: anomaly rate / prevalence

**狀態**: resolved ✓

**出現脈絡**: §2.3.2 描述「about 5%」的 anomaly rate 作為 downsampling 的背景。

**M2.1 measured**:
- 實測 anomaly rate: **2.13%**（37,296 pos / 1,712,198 neg / 1,749,494 total）
- 論文描述: "about 5%"

**解釋**:
- LEAD 競賽資料集為 GEPIII 1,449 棟的子集(200 棟),anomaly rate 因子集選取而異於全集
- 論文的 "about 5%" 可能描述的是 GEPIII 完整資料集的 anomaly rate,而非 LEAD 子集
- 不排除是概估或包含了更廣泛的異常定義

**不影響重現性**: Downsampling 策略(50:50)以實際 pos/neg 筆數為基準,
與論文描述的比例無關;2.13% 的實際 rate 代入公式後仍得到正確的 50:50 結果。

---

## 9. building_id 範圍非連續(LEAD 為 GEPIII 子集)

**術語**: building_id range, modulo split

**狀態**: resolved ✓

**出現脈絡**: CV split 用 `building_id % 5 == 4` 選出 validation set。M1 估計約 40 棟（200 × 1/5），實際測量為 38 棟。

**M2.1 measured**:
- building_id 最大值: 1319（非連續 0–199）
- 原因: LEAD 200 棟保留 GEPIII 原始 building_id，跨越更大範圍
- `% 5 == 4` 實際落入的建築數: **38 棟**（非理論 40 棟）

**設計含義**:
- building_id 稀疏分佈在 0–1319，`% 5` 的分佈不均勻，造成 val 為 38 棟而非 40 棟
- 但 modulo split 的核心優點仍成立：確定性分割、無 random seed、
  building 層級完整隔離
- 38/162 ≈ 19% val rate，接近理論 20%，不影響評估可靠性

---

## 10. M2.1 baseline AUC gap (3.86%)

**術語**: val AUC 0.8952 vs 論文 Fig 4 baseline 0.9311

**狀態**: open — cloud_coverage disconfirmed (M2.2.0); impute_nulls is new primary suspect

**出現脈絡**: M2.1 跑出 val AUC 0.8952,論文 Fig 4 報告 0.9311(feature engineering 前
baseline)。Gap 3.86% 在 m2-plan 的 < 5% pass 範圍,但顯著到值得列候選原因清單。

**候選原因**:

| 優先 | 原因 | 驗證時機 | 備註 |
|------|------|---------|------|
| **1st** | **缺 impute_nulls:跳過 per-building mean imputation(Feature generator Cell 11),LightGBM 自然處理 NaN** | M2.5 ablation | **NEW PRIMARY SUSPECT** — split-affecting:NaN 和 mean-imputed value 走不同 split path |
| 2nd | CV fold variance:只跑 fold 4;論文可能是 5-fold 平均 | M2.5 | split-affecting;跑 5-fold loop 看各 fold AUC std |
| 3rd | Downsampling seed variance:seeds 10/20 固定;論文 seed 未揭露 | M2.5 | split-affecting;多 seed 組合看 std |
| 4th | LightGBM 版本差異:2022 vs 2026 default 行為微調(min_child_samples、num_leaves 等) | M2.5 | possibly split-affecting;版本 pin 或查 changelog |
| 5th | 論文 Fig 4 數字本質:0.9311 是 val 還是 test AUC,或是 5-fold 平均而非 fold 4 | M2.5 | scaling-independent confound;論文未明說 |
| 6th | 時間點缺洞 + groupby().shift() 近似:104/200 棟缺時間點,shift(n) 不完全等於 n 小時 | M2.5 ablation | split-affecting;見 unknowns #11 |
| 7th | XGBoost/CatBoost NaN sensitivity:lag features 有 7.12% NaN,XGBoost gap 最大 0.91% 可能跟 NaN handling 策略相關 | M2.5 ablation | split-affecting;impute_nulls 升級 |
| — | Preprocessing 完整度 | **Resolved by Step (c)** | buds-lab 流程已確認 |
| — | Feature subset 差異(46 vs buds-lab list_variables) | **Resolved by Step (c)** | 邏輯完全一致 |
| — | cloud_coverage sentinel(255→10) | **Disconfirmed by M2.2.0** | Monotonic 轉換;ΔAUC = 0;tree models invariant |

**預期演進**: M2.2 加入 value-change features 後 AUC 應跳至 ≥ 0.97(+5% 跳幅)。
若 M2.2 後相對 gap 仍 ~0.04,代表是 pipeline-level 差異,需 M2.5 系統 ablation。
若 gap 隨 feature 增加而縮小,候選 4/5 可能性上升。

**M2.2.0 結果(2026-05-26)**:

- cloud_coverage fix(255→10):ΔAUC = **+0.0000**
- 判斷:**disconfirmed** — cloud_coverage 不是 M2.1 gap 的原因
- 原因:buds-lab 使用的四個模型全部是 tree-based(LightGBM / XGBoost /
  CatBoost / HistGBT)。Tree splits 只依賴 feature 值的相對大小(ranking),
  不依賴絕對值。255→10 保留了 cloud_coverage 的排序(10 仍是最大值,
  因為其他合法值均 ≤ 9),所以 LightGBM 的 splits 與 predictions 完全不變。
  StandardScaler 對 tree models 同樣無影響,進一步確認。
- **cloud_coverage fix 仍需保留在 pipeline**:原始碼確實有此步驟,對非 tree
  models 的未來擴展有影響,且維持與 buds-lab 精確對齊。

**M2.5 ablation 計畫**:
- impute_nulls 加/不加對照(驗證 1st suspect)
- 5-fold loop AUC variance(驗證 2nd suspect)
- 多 seed 組合:seeds {10,20}, {10,30}, {42,43} 等(驗證 3rd suspect)
- LightGBM 版本降級測試(驗證 4th suspect,若前三項不收斂)
- timestamp.merge vs groupby.shift 對照(驗證 6th suspect,見 #11)

**Lesson learned (M2.2.0)**:

- Tree-based models(LightGBM / XGBoost / CatBoost / HistGBT)對 monotonic
  feature 轉換完全 invariant:255→10 保留排序,AUC 完全相同
- StandardScaler 對這四個模型在數值上沒有影響(buds-lab Modeling Cell 8
  仍保留 StandardScaler,可能是 leftover code 或為未來 non-tree 模型預留)
- M2.1 gap 主因的搜索範圍縮小:**absolute scaling 類問題全部可排除**,
  主因應在 split-affecting 因素:NaN handling、seed variance、CV fold、版本差異

**Lesson learned (M2.2.a, 2026-05-26)**:

- **sklearn 1.4+ KMeans `n_init` default 變更**:
  - 2022 (buds-lab 寫 code 時): n_init default = 10
  - 2024+: n_init='auto' with k-means++ → 實際 = 1
  - 影響: 不明確設 n_init 會落入不同 local minimum (ARI=0.503)
  - 修正: 明確設 n_init=10 → ARI=1.0, 406/406 perfect alignment
- 重現舊 code 時必須明確設定所有可能在版本間變化的 default 參數
- M2 工作原則: 所有 sklearn estimator 都明確設定關鍵參數

---

## 11. 200 buildings 中有 104 棟缺時間點(完整 8784 ts 只有 96 棟)

**術語**: missing timestamps, groupby().shift() vs timestamp-based merge

**狀態**: documented — M2.2.b 採 groupby().shift() 為近似;M2.5 量化影響

**M2.2.b Stage 0 measured (2026-05-26)**:

- Complete buildings (8784 ts): 96/200
- Incomplete buildings: 104/200
- Min timestamps per building: 7471 (building_id=1353, 缺 1313 hours = 15%)
- Max timestamps per building: 8784

**M2.2.b 的選擇**:

- 採 `groupby('building_id').shift(n)` 而非 buds-lab 原版的 timestamp-based merge
- 兩者在無缺洞建築上等價;缺洞建築會有 divergence:
  - groupby().shift(n) 返回「往前 n rows」(可能跨越缺洞)
  - timestamp-based merge 返回「往前 n 小時」(對缺洞精確)
- 對 tree-based 模型,absolute 數值精確度影響小;split-affecting 差異未量化

**M2.5 ablation 計畫**:

- 加為 M2.1 gap candidate 6:時間點缺洞 + shift implementation choice
- 量化方式:跑兩版 lag features(groupby.shift vs timestamp.merge),
  對照 LightGBM val AUC

**為什麼重要**: 對 final AUC 與論文 0.9849 對齊有影響;若 M2.2.e 跑出 AUC < 0.95,
這是排第二位的可能原因(僅次於 #10 candidate 1 impute_nulls)。

**M2.2.b Stage 1 已知 issues**:

- pandas PerformanceWarning(DataFrame fragmentation)出現於 Cell 3 大量 column inserts
  - 不影響數值正確性;Cell 3 elapsed 3.3s
  - 若未來重構,改用 `pd.concat([train] + lag_dfs, axis=1)` 一次 merge
- value_change.csv 過大(2.9 GB),不存 CSV
  - M2.2.e 整合時 import notebook 邏輯直接重生成
  - 不影響 M2.5 ablation(M2.5 用 in-memory features)

---

## 12. SavGol residual importance vs paper §2.2.4 「no apparent positive effect」

**術語**: Residual_savgol_w5p3 importance, split count vs gain

**狀態**: documented — M2.5 ablation 量化實際 AUC 貢獻

**M2.2.e 測量 (2026-05-26)**:
- LightGBM feature importance(split count): `Residual_savgol_w5p3` = 105,排名 #6 / 169
- 論文 §2.2.4: 「did not show apparent positive effect」
- Paper Fig 5 top 10 不含 `Residual_savgol_w5p3`

**差異解釋**:
- LightGBM 預設 importance 是 split count(feature 被用來 split 的次數),不是 gain
  (split 帶來的 metric 改善)。Split count 高代表 feature 被頻繁用作分裂點,
  不等同於 AUC 貢獻大。
- SavGol residual 是 meter_reading 的近似導數,可能作為 proxy feature 提供輔助信號,
  但 gain-based importance 可能顯著較低。
- 論文的「no apparent positive effect」可能是基於 gain-based 或 permutation importance,
  而非 split count。

**M2.2.e SavGol 合法性確認**:
- LEAD 是 batch task(整年資料一次處理),centered SavGol(使用 i±2 未來點)不構成 test-time leakage
- 驗證:centered vs causal SavGol residual correlation = 0.333(全序列),0.600(anomaly rows)
- 結論:保留 centered SavGol 與 buds-lab 對齊;M2.5 ablation 量化實際 AUC 貢獻

**M2.3 cross-model 觀察 (2026-05-26)**:

- Residual_savgol_w5p3 importance ranking across models:
  - LightGBM: rank 6 (importance 105)
  - XGBoost: **rank 3** (top 5)
  - CatBoost: **rank 3** (top 5)
- Paper §2.2.4「no apparent positive effect」**可能只對 LightGBM 成立**
- XGBoost 跟 CatBoost 把 SavGol residual 當主要 signal

**M2.5 ablation 升級**:
- 不只測 with/without SavGol 對 LightGBM AUC
- 還要量化 Residual_savgol 對 XGBoost、CatBoost 的單獨 ΔAUC
- 若三個 model ΔAUC 不對稱 → paper §2.2.4 framing 過度 generalization

**M2.5 ablation 計畫**: 移除 `Residual_savgol_w5p3` 後重跑 LightGBM,量化 ΔAUC

---

## 13. Cross-model importance divergence: 4 models 看同樣 169 features 但學到不同 patterns

**術語**: feature importance, ensemble effectiveness, model-specific signals

**狀態**: documented — M2.5 ablation 進一步量化

**M2.3 觀察 (2026-05-26)**:

各 model top 5 importance:

| Rank | LightGBM | XGBoost | CatBoost |
|------|----------|---------|----------|
| 1 | **building_id** | meter_reading | meter_reading |
| 2 | lag_ratio_1 | lag_ratio_168 | lag_ratio_1 |
| 3 | meter_reading | **Residual_savgol** | **Residual_savgol** |
| 4 | lag_ratio_-1 | lag_ratio_1 | lag_ratio_-1 |
| 5 | dayofyear | lag_ratio_19 | building_id |

HistGBT: sklearn 1.7 不支援 `feature_importances_`，不納入對比。

**Cross-model 共同 top 10**: 6/10 features (`meter_reading`,
`building_id`, `lag_ratio_{1,-1,168,-168}`, `Residual_savgol_w5p3`)

**設計含義**:
- LightGBM strength: building-level personalization (building_id rank 1)
- XGBoost/CatBoost strength: absolute meter_reading level + SavGol smoothing
- Ensemble lift 來源: 6 個共同 features baseline + 各 model 4 個獨有 marginal

**Paper §2.3.4 的數學基礎**: "Each model has its own strengths and weaknesses"
— 我們提供了具體的量化證據(paper 沒給的 cross-model importance 對比)。

**為什麼重要**: 這是 paper Table 2 沒展示的洞察。M2 final report 的
discussion section 可以引用。

---

Last reviewed: 2026-05-26 (M2.3 complete: ensemble 0.9832, gap 0.34%; #13 added; #12 cross-model SavGol observation added)
