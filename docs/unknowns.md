# Unknowns

論文沒有解釋清楚、重現時需要另外找資源或實驗才能確定的問題。

每個 unknown 列出:術語/問題、論文出現脈絡、為什麼重要。

---

## 1. Feature count 169 的完整組成

**術語**: 總特徵數 169(Table 3: "Raw, V-C (169)")

**狀態**: upgraded — 缺口來源更精確,但仍未解

**出現脈絡**: §2.2 說原始特徵「up to 57」;§2.2.1–2.2.3 描述的 value-change shifts 為 8 個值 × 2 方向 × 2 類型(差值/比值) = 32 個 value-change features。57 + 32 = 89,與 Table 3 標示的 169 差距 80 個,論文完全未解釋。

**Issue #7 發現(精確化)**:
- 競賽提供的 `train_features.csv` 已預計算 **57 個 original features** ✓ — 含 building meta、weather、temporal、target encoding,欄位數完全吻合。
- 論文 §2.2 描述的 value-change features 只有 **32 個**(8 shifts × 2 方向 × 2 類型)。
- 缺口收斂為:**80 個未知的 value-change features**(169 − 57 − 32 = 80)。
- Original features 已確認,問題完全聚焦在 value-change 的計算方式上。

**為什麼重要**: 如果重現時只建出 89 個特徵,就漏掉了 80 個 value-change features,AUC 可能無法達到 0.9866。需要看 buds-lab/LEAD-1st-solution 的 notebook 確認完整 shift 清單與計算邏輯。

---

## 2. Validation split 的實作細節

**術語**: split by `building_id`

**出現脈絡**: §2.3.1 說「Train and validation datasets were split by building_id to ensure valid data were unseen during training」,且 validation score 與 leaderboard 差距 < 1%。但論文未說明:是單次 split 還是 k-fold?validation set 包含幾棟建築?用什麼 random seed?

**為什麼重要**: Split 策略直接影響每次本地評估的可靠性。如果我用不同的 validation 建築數或 fold 設定,同一套特徵工程實驗的 AUC 增益可能和論文報告的不一致,無法按照論文的節奏迭代改進。「差距 < 1%」這個特性也依賴特定的 split 設計才能重現。

---

## 3. Post-processing "start and end points" 的邊界定義

**術語**: start and end points of time series

**出現脈絡**: §2.4 Rule 2:「Set prediction to 0 (normal) for start and end points of time series.」論文未定義「start and end」是首尾各幾筆讀數,也未說明依據什麼邊界。

**為什麼重要**: 不同的邊界定義(首尾各 1 筆、各 24 筆、或某個固定時間窗)會改變最終提交的預測,影響 AUC。這個規則是 rule-based 的,需要完全照原版才能重現相同分數。需看 GitHub 確認實際實作。

---

## 4. Downsampling 的作用域、策略與 random seed

**術語**: downsampling (of normal class)

**出現脈絡**: §2.3.2 說「downsampling method for normal data is adopted so that normal and abnormal data can be balanced to occupy half of the training data.」未說明:是在整個 training set 做一次全域 downsampling?還是按建築做?還是在每個 CV fold 內重新抽?使用什麼 random seed?

**為什麼重要**: 不同的 downsampling 策略會影響訓練集的分布和模型穩定性。若在 CV fold 外做一次全域 downsampling,validation set 的評估可能偏樂觀;若每個 fold 重抽,結果的變異會不同。Random seed 決定結果的可重現性。

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
