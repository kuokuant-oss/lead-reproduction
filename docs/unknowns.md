# Unknowns

論文沒有解釋清楚、重現時需要另外找資源或實驗才能確定的問題。

每個 unknown 列出:術語/問題、論文出現脈絡、為什麼重要。

---

## 1. Feature count 169 的完整組成

**術語**: 總特徵數 169(Table 3: "Raw, V-C (169)")

**狀態**: partially-resolved — shift 範圍已從原始碼確認,總特徵數差距 6 待階段 B 計算確認

**出現脈絡**: §2.2 說原始特徵「up to 57」;§2.2.1–2.2.3 描述的 value-change shifts 為 8 個值 × 2 方向 × 2 類型(差值/比值) = 32 個 value-change features。57 + 32 = 89,與 Table 3 標示的 169 差距 80 個,論文完全未解釋。

**Issue #7 發現(原始碼觀察)**:
- 競賽提供的 `train_features.csv` 已預計算 **57 個 original features** ✓
- 原始碼 Feature generator Cells 11–12 的實際 shift 範圍:
  - `np.arange(-24, 0)` + `np.arange(1, 25)` = 每小時 ±1~24(48 個)
  - `np.arange(-168, -24, 24)` + `np.arange(48, 169, 24)` = 每日 ±2~7 week(12 個)
  - 合計 **60 shifts × 2 類型(diff + ratio) = 120 value-change features**
- 論文宣稱 32 個 value-change features 是**嚴重低估**,實際是 120 個
- 初估總特徵數:57 + 120 + ClusterNo + Savitzky-Golay residual + dayofyear ≈ **175**
- 論文說 169,差距 **6 個** — 原因待階段 B 執行 Cell 4 的 `list_variables` 確認

**剩下待確認**: 精確的 175 vs 169 差距(可能某些欄被 `select_dtypes` 或 drop 排除);ClusterNo 和 Savitzky-Golay 是否真的在最終特徵集內。

---

## 2. Validation split 的實作細節

**術語**: split by `building_id`

**狀態**: partially-resolved — split 機制已確認,驗證建築數待計算

**出現脈絡**: §2.3.1 說「Train and validation datasets were split by building_id to ensure valid data were unseen during training」,且 validation score 與 leaderboard 差距 < 1%。但論文未說明:是單次 split 還是 k-fold?validation set 包含幾棟建築?用什麼 random seed?

**原始碼觀察(Modeling notebook Cell 6)**:
- 實作:`building_id % 5 < 4` → train;`building_id % 5 == 4` → validation
- **確定性切分,無 random seed** — building_id 模 5 等於 4 的建築固定為 validation
- 200 棟建築中約 40 棟(20%)進 validation,約 160 棟(80%)進 train
- 這是單次 split(非 k-fold);論文說「k-fold」的解讀有誤,就是單次

**剩下待確認**: 確切的 validation 建築數(需計算 200 棟中 building_id % 5 == 4 有幾棟)。

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

**狀態**: partially-resolved — 機制已確認,但「downsampling vs upsampling」語義待釐清

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

**剩下待確認**: 論文說「downsampling」,但實際上 anomaly 資料是被複製(upsampling 的成分),normal 才是 downsampling。這個語義差異對重現是否有影響需在階段 B 驗證。

---

## 5. Target encoding 的 data leakage 防護方式

**術語**: target encoding

**出現脈絡**: §2.2 Table 1 列出「Average values of the target variable aggregated by category (e.g., average values grouped by building_id)」作為一類特徵。若在 CV 切割前對完整訓練集計算 target encoding,validation set 的 label 資訊會洩漏進 training features,導致 validation AUC 虛高。論文未說明如何防範。

**為什麼重要**: 若重現時未防護 leakage,本地 validation 分數會過度樂觀,而 test set(206 棟新建築)上的性能不如預期。正確做法通常是 out-of-fold target encoding 或在 CV 內部計算;需看 GitHub 確認原版實作。

---

## 6. 四個 GBDT 模型的超參數

**術語**: hyperparameters (LightGBM, XGBoost, CatBoost, HistGradientBoosting)

**出現脈絡**: §2.3.3 提及四個模型各自訓練並做 hyperparameter tuning,但論文完全未揭露任何超參數數值(如 `num_leaves`、`learning_rate`、`max_depth`、`n_estimators` 等)。

**為什麼重要**: 超參數不同,Train/Test AUC 會有差距。XGBoost 和 CatBoost 的 Train AUC 高達 0.9999,暗示有特定的超參數設定導致高度 fitting。若重現時用預設值,各模型的 AUC 分布可能和論文 Table 2 不一致,ensemble 後的 AUC 也可能偏低。需看 buds-lab/LEAD-1st-solution 的各 notebook 確認。

---

Last reviewed: pending
