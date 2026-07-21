# M3 報告：完整 ASHRAE GEPIII

+ **狀態**：完成
+ **日期**：2026-06-22
+ **任務**：使用完整 ASHRAE GEPIII 訓練資料，從原始 CSV 建立特徵，並以
  buds-lab `bad_meter_readings.zip` 異常標籤進行二元分類。

M3 是 M2 LEAD 重製工作的延伸實驗。M2 驗證 LEAD competition subset 上的 paper
reproduction；M3 驗證同一套方法論在完整 GEPIII train subset 上是否能建立穩定的
異常排序模型。M3 沒有 Kaggle leaderboard，因此所有結果皆以建物留出驗證 AUC
回報。

## 主要程式碼

+ M3.3 buds-lab alignment：[scripts/run_m3_3_budslab.py](https://github.com/kuokuant-oss/lead-reproduction/blob/main/scripts/run_m3_3_budslab.py)。
+ M3.4 ensemble：[scripts/run_m3_4_ensemble.py](https://github.com/kuokuant-oss/lead-reproduction/blob/main/scripts/run_m3_4_ensemble.py)。
+ M3.5 post-processing：[scripts/run_m3_5_postprocessing.py](https://github.com/kuokuant-oss/lead-reproduction/blob/main/scripts/run_m3_5_postprocessing.py)。
+ 50/50 offline/causal ensemble：[scripts/run_m3_50_50_ensemble.py](https://github.com/kuokuant-oss/lead-reproduction/blob/main/scripts/run_m3_50_50_ensemble.py)。
+ Split causality diagnostic：[scripts/run_m3_split_causality.py](../../scripts/run_m3_split_causality.py)。
+ 50/50 純觀測與出圖：[scripts/run_m3_figure_observations.py](../../scripts/run_m3_figure_observations.py)、[scripts/plot_m3_figures.py](../../scripts/plot_m3_figures.py)。
+ Golden gates 與 metrics：[tests/golden_metrics.json](../../tests/golden_metrics.json)、[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。
+ 圖表 provenance：[docs/metrics/m3-figures.json](../metrics/m3-figures.json)；作圖規範：[docs/reference/plot-style-rules.md](../reference/plot-style-rules.md)。

---

# 第 1 章：任務與評估設計

## 1.1 資料集

+ 資料來源：ASHRAE GEPIII，來源：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)。
+ 資料集：data/raw/m3/train.csv、data/raw/m3/bad_meter_readings.csv、data/raw/m3/building_metadata.csv、data/raw/m3/weather_train.csv。
+ Anomaly labels：來自 buds-lab bad_meter_readings.zip。

| 項目 | 值 |
|---|---:|
| 建物數 | 1,449 |
| 資料列數 | 20.2M |
| 電表類型 | electricity, chilled water, steam, hot water |
| 標籤合併方式 | 逐列位置對齊 |
| 整體異常比例 | 6.50% |

## 1.2 最終評估設計

最終報告採用 50/50 building-held-out split：

| Split | 訓練建物數 | 驗證建物數 | 建物重疊 |
|---|---:|---:|---:|
| `building_id % 2` | 725 | 724 | 0 |

報告同時列出兩種 feature availability 設定：

| 設定 | 特徵可用性 | 特徵數 | 解讀 |
|---|---|---:|---|
| Offline batch labeling | past-shift + future-shift value-change features | 137 | 批次回溯標註 |
| Past-only | 只使用 past-shift value-change features | 77 | 線上評分時不可使用 future meter readings |

早期 80/20 實驗（`building_id % 5 == 4`，1160/289 buildings，M3.1-M3.5）
集中放在第 3 章，作為開發階段證據與消融實驗。

## 1.3 解讀規則

以下結果中，最終結論以 50/50 building-held-out split 為準。80/20 結果用於說明
development-stage feature、model 與 diagnostic evidence。小於約 `0.001` 的 AUC
差異落在抽樣與建物層級變異範圍內，不作 stable lift claim。Value-change features
的貢獻以 17 features 到 137 features 的差異作為主要證據。

M2 可作重製錨點：Kaggle Private `0.98616`，與原作者 `0.98661` 的 gap 為
`0.05%`。M3 的資料、標籤與評估切分不同，因此 M2 分數只作背景參照。

---

# 第 2 章：最終結果

最終模型是 four-model equal-weight ensemble：LightGBM、XGBoost、CatBoost、
HistGradientBoosting。Offline batch labeling 使用 137-feature set；past-only
設定使用對應的 77-feature set。

| Split | 設定 | 特徵數 | Ensemble AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| 50/50 mod2 | Offline batch labeling | 137 | 0.9918 | 0.7317 | 0.9483 | 0.8260 |
| 50/50 mod2 | Past-only | 77 | 0.9913 | 0.7167 | 0.9430 | 0.8144 |

表 2.1：regime=`timestamp_merge`；split ratio=50/50 building-held-out。

機器可讀 provenance：
[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。

Offline batch labeling 分數是 50/50 protocol 下的批次回溯標註結果。Past-only
AUC 低 `0.0005`，量化了從 value-change features 中移除 future meter readings
的成本。

## 2.1 最終 50/50 模型圖表

以下獨立圖表全部來自同一條 50/50 building-held-out offline baseline。觀測 runner
只重建既有資料、特徵與模型並記錄 predictions、curves 與 permutation 結果；沒有
修改 `src/lead`、既有 M3 runners、切分、seed、前處理、模型參數、ensemble 或
threshold。

### Ensemble 混淆矩陣

![Tree Ensemble confusion matrix](./assets/m3/m3_tree_ensemble_confusion_threshold_0_5.png)

在 threshold `0.5` 下，Tree Ensemble 的 `TN/FP/FN/TP` 為
`9,278,088/221,670/32,982/604,415`。實際異常中有 `94.8%` 被偵測，漏報率為
`5.2%`；混淆矩陣只呈現最終 ensemble，不將四個 component models 重複拆圖。

### 數值變化特徵示意

![Value-change feature illustration](./assets/m3/m3_value_change_difference_ratio_illustration.png)

圖中使用真實 validation building/meter 時序，說明一小時
`Difference = current - previous` 與 `Ratio = (current + 1) / (previous + 1)`。
讀值突然降至零時，Difference 與 Ratio 同時偏離鄰近時段，顯示 value-change
features 如何把突變轉成模型可使用的訊號；缺少精確一小時配對時保留空值，不以相鄰列
替代。

### ROC、Precision-Recall 與 feature engineering

#### 四模型與 ensemble：Precision-Recall（局部放大）

![Model precision-recall comparison](./assets/m3/m3_model_comparison_precision_recall_zoomed.png)

#### 四模型與 ensemble：ROC（局部放大）

![Model ROC comparison](./assets/m3/m3_model_comparison_roc_zoomed.png)

#### Feature engineering：Precision-Recall

![Feature-engineering precision-recall comparison](./assets/m3/m3_feature_engineering_precision_recall.png)

#### Feature engineering：ROC

![Feature-engineering ROC comparison](./assets/m3/m3_feature_engineering_roc.png)

| 模型 | Features | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| M3.1 LightGBM | 17 | 0.9650 | 0.8235 |
| LightGBM | 137 | 0.9910 | 0.9239 |
| XGBoost | 137 | 0.9892 | 0.9277 |
| CatBoost | 137 | 0.9883 | 0.9256 |
| HistGBT | 137 | 0.9915 | 0.9262 |
| Tree Ensemble | 137 | 0.9918 | 0.9303 |

表 2.2：所有數字皆為 50/50 mod2、`timestamp_merge` offline validation；不是
80/20 development artifact。137-feature LightGBM 相較 17-feature baseline 的
ROC-AUC 與 PR-AUC 分別增加約 `0.0260` 與 `0.1004`。四個 component models 的
ROC-AUC 接近，ensemble 的 PR-AUC 最高。

### 四模型共識與 permutation importance

#### LightGBM

![LightGBM permutation importance](./assets/m3/m3_permutation_importance_lightgbm.png)

#### XGBoost

![XGBoost permutation importance](./assets/m3/m3_permutation_importance_xgboost.png)

#### CatBoost

![CatBoost permutation importance](./assets/m3/m3_permutation_importance_catboost.png)

#### HistGBT

![HistGBT permutation importance](./assets/m3/m3_permutation_importance_histgbt.png)

#### Tree Ensemble

![Tree Ensemble permutation importance](./assets/m3/m3_permutation_importance_tree_ensemble.png)

#### Four-model consensus

![Four-model consensus permutation importance](./assets/m3/m3_permutation_importance_four_model_consensus.png)

四模型與 ensemble 都把 `meter_reading`、`meter` 排在前段；ensemble 前十名亦包含
`floor_count`、`dayofyear`、`lag_value_diff_1`、building metadata 與 weather。
這表示模型不只依賴單一 value-change feature，但一小時 Difference 確實進入主要
判斷訊號。

單欄 permutation screening 找到 44 個零、負值或與重複變異不可區分的候選。這只
是篩選，不等同可刪除證據；觀測流程再做相關 feature group 檢查，並把前三組候選
分別設為標準化後的常數零、重新訓練相同四模型與 ensemble：

| Targeted group | Δ ROC-AUC | Δ PR-AUC | Δ Recall@0.5 | 判定 |
|---|---:|---:|---:|---|
| ratio -11；diff/ratio -10；ratio -9 | +0.000599 | +0.001346 | -0.002760 | harmful to remove |
| diff -15/-14/-13；ratio -14 | -0.000307 | -0.002110 | -0.004664 | harmful to remove |
| diff/ratio 120 | -0.000651 | -0.001777 | -0.004980 | harmful to remove |

表 2.3：第一組排序指標雖上升，但 Recall@0.5 超出 `0.001` 容許退化，因此仍不符合
移除條件；另外兩組同時降低 PR-AUC 與 Recall。結論是目前沒有足夠證據從 canonical
137-feature set 移除這些候選，既有 feature engineering 保持不變。

### 模型流程

![M3 anomaly-detection workflow](./assets/m3/m3_anomaly_detection_workflow.png)

流程圖區分 17 個 baseline features、120 個 `timestamp_merge` value-change
features、50/50 建物留出、training-only downsampling、保留的 StandardScaler、
四個 frozen tree models 與 equal-weight probability ensemble。Past-only 77-feature
版本是另行評估的 companion，不混入本節 offline 圖表。

既有 train/validation gap 檢查使用 80/20 development line 與 timestamp_merge
value-change features，目的只在檢查是否出現明顯 fit-set memorization。

| Score | LightGBM | XGBoost | CatBoost | HistGradientBoosting | Ensemble |
|---|---:|---:|---:|---:|---:|
| Fit-set AUC | 0.9984 | 0.9996 | 0.9999 | 0.9983 | 0.9997 |
| Full train-buildings AUC | 0.9984 | 0.9996 | 0.9999 | 0.9983 | 0.9996 |
| Validation AUC | 0.9925 | 0.9923 | 0.9904 | 0.9916 | 0.9934 |

表 2.4：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

Fit-set 與 full train-buildings AUC 接近，表示高分不是只來自重複 fit-set rows。
Full train-buildings 對 validation 約有 `0.006` AUC gap，列為 capacity/stability
限制，而不是 M3 headline result。

---

# 第 3 章：開發階段證據與消融實驗

本章使用 80/20 development line，目的是說明 features、models 與 post-processing
rules 對排序能力的影響。

## 3.1 AUC 進程

| 階段 | Split | 設定 | 特徵數 | AUC | 解讀 |
|---|---|---|---:|---:|---|
| M3.1 baseline | 80/20 | Offline | 17 | 0.9562 | Time, building, meter, weather baseline |
| M3.2 value-change | 80/20 | Offline | 137 | 0.9925 | 主要 feature-engineering 提升 |
| M3.2 value-change | 80/20 | Past-only | 77 | 0.9909 | past-shift features 仍提供穩定訊號 |
| M3.3 buds-lab alignment | 80/20 | Offline | 170 | 0.9928 | 補充特徵未帶來穩定 lift |
| M3.4 ensemble | 80/20 | Offline | 137 | 0.9934 | ensemble 小幅提升 |
| M3.5 post-processing | 80/20 | Offline | 137 | 0.9933 | 舊 rules 在本資料設定下未帶來改善 |

表 3.1：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

## 3.2 Value-change features

M3.1 使用 17 個 baseline features：time features、building metadata、meter type、
meter reading 與 weather。M3.2 加入 120 個 value-change features。

| 模型 | 特徵數 | AUC | 差異 |
|---|---:|---:|---:|
| M3.1 baseline LightGBM | 17 | 0.9562 | - |
| M3.2 value-change LightGBM | 137 | 0.9925 | +0.0363 |

表 3.2：regime=`timestamp_merge` for M3.2；split ratio=80/20 building-held-out。

Value-change features 使用與 M2 相同的 shift family：`-24..-1`, `1..24`,
`-168..-48 step 24`, `48..168 step 24`。M3.2 baseline 的權威產出由
[scripts/run_m3_2_baseline.py](../../scripts/run_m3_2_baseline.py) 產生，固定
`value_change_regime="timestamp_merge"`。此 regime 對應 buds-lab 原作的
`timestamp + timedelta` 後 merge，是 M3/M4/M5 後續比較的 canonical value-change
基準；`row_offset` 與 `row_offset_meter_aware` 保留為歷史近似與 M5 §7 ablation。
M3 的 diff sign 與 ratio orientation 和 M2 相反，屬 negation / reciprocal 的
monotonic 轉換；Ratio 的 +1 smoothing 公式見 ADR 0008。

## 3.3 buds-lab 對齊

M3.3 補上 buds-lab GEPIII preprocessing 中優先級最高的 feature 類別：
cyclic time encodings、weather trailing lags and rolling means、holiday flags、
train-only `(site, meter)` target encoding、primary-use/meter interaction，
以及 site 0 meter 0 correction。

| 執行 | 特徵數 | AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| M3.2 reference | 137 | 0.9925 | 0.6648 | 0.9688 | 0.7885 |
| M3.3 buds-lab 對齊 | 170 | 0.9928 | 0.6732 | 0.9533 | 0.7891 |

表 3.3：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

Full buds-lab alignment 作為驗證與消融步驟有價值；ranking AUC 未改善，
因此不納入最終模型。Threshold-0.5 指標列於上表。

## 3.4 Ensemble

Ensemble 使用 M3.2 feature set。

| 模型 | 80/20 AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|
| LightGBM | 0.9925 | 0.6648 | 0.9688 | 0.7885 |
| XGBoost | 0.9923 | 0.7033 | 0.9552 | 0.8101 |
| CatBoost | 0.9904 | 0.7284 | 0.9518 | 0.8252 |
| HistGBT | 0.9916 | 0.6561 | 0.9655 | 0.7813 |
| Ensemble | 0.9934 | 0.6961 | 0.9636 | 0.8083 |

表 3.4：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

Ensemble lift 為 `+0.0009`。Value-change features 是主要貢獻，ensemble 是次要
增益。補充的建物層級 bootstrap 檢查顯示 lift 方向為正，建物層級 CI 仍有重疊，
細節見第 4 章。

## 3.5 事後規則

M3.5 測試 M2/LEAD hard post-processing rules 是否能轉移到 raw GEPIII meter
readings。

| 規則 | 觸發列數 | 異常列數 | 相對 pre-rule 的 AUC 差異 |
|---|---:|---:|---:|
| Rule 1: `meter_reading == 1.0 -> 1` | 8 | 0 | -0.000002 |
| Rule 2a: Jan-1 start-point filter | 0 applied | 0 | 0.000000 |
| Rule 2b: `dayofyear > 366.9583 -> 0` | 478 | 13 | -0.000052 |
| Combined | - | - | -0.000054 |

表 3.5：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

Pre-rule 集成模型 AUC 是 `0.9933765`；combined post-processing AUC 是
`0.9933226`。舊規則在本資料設定下未帶來改善。

---

# 第 4 章：有效性檢查與限制

本章整理影響 M3 headline result 解讀的有效性檢查。完整補充檢查、程式碼與 JSON
輸出收錄於 appendix index。

## 4.1 切分與 leakage 檢查

M3 採用 building-held-out split，訓練與驗證建物沒有重疊。Target-encoder ablation
未出現異常分數提升，表示 target encoding 對 headline result 的影響有限。

Value-change features 可分為 past-shift features 與 future-shift features。在
offline batch labeling 中，兩者提供的訊號相近。若模型用於即時 FDD 或線上評分，
feature set 應限制為 past-shift features，因為 future-shift features 依賴未來
meter readings。

Label-shuffle 後的分數明顯低於真實標籤結果，表示模型表現並非主要由隨機標籤結構
解釋。不過，value-change features 在打亂標籤後仍保留少量可學習結構。基於這項
殘餘結構，本文將約 `0.001` AUC 量級的差異視為實質上不可區分，除非另有穩定性或
slice-level 證據支持。

## 4.2 Generalization 限制

Building-held-out 結果可作為 GEPIII 內部基準線。相比之下，site-held-out 驗證更
困難，顯示跨 site 泛化是 M3 後續部署與比較時的重要限制。各 meter type 中，steam
的表現最低，應作為後續誤差分析的優先對象。

Primary-use slices 中，多數類別維持高 AUC。部分類別的驗證建物數較少，因此這些
high-score small slices 主要用於定位資料分布與模型行為。完整表格見
`docs/metrics/m3-primary-use-auc.json`。

Time-holdout 檢查提供額外敏感性資訊。由於它同時改變 value-change regime 與 split，
其結果適合用來觀察模型在不同時間設定下的變化，不直接併入 M3 headline comparison。
細節見 appendix index。

## 4.3 Feature implementation 限制

buds-lab 原作的 value-change 對齊方式是 `timestamp + timedelta` 後 merge。
M3 現以 `timestamp_merge` 作為忠實基準，並由 frozen API
`add_value_change_features(..., value_change_regime="timestamp_merge")` 產生所有
canonical value-change 數字。`row_offset` 與 `row_offset_meter_aware` 是歷史近似；
前者以相鄰列近似時間變化，後者修正 multi-meter row-offset 的 meter-crossing，但
兩者都不再是 M3/M4/M5 的 canonical baseline。其影響集中收錄於 M5 §7 regime ladder。

Past-only 50/50 結果只比 offline batch labeling 低 `0.0005` AUC，顯示 M3 在即時
FDD 設定下仍保有接近 offline batch labeling 的表現。實際部署時，模型輸入應限制為
當下以前可取得的特徵。

整體來看，這些檢查界定了 M3 headline score 的適用條件：GEPIII 內部、
building-held-out split，以及 offline batch labeling 設定。在目前檢查範圍內，
未觀察到會系統性推高 headline result 的明顯 leakage pattern。後續比較應固定
split、label source 與 feature availability，並另外回報 cross-site、steam meter、
歷史行偏移近似與抽樣穩定度結果。

---

# 第 5 章：GEPIII 文獻對照

Miller et al. (2020, GEPIII overview/results) 將 GEPIII 描述為大型
energy-prediction competition，評分指標為 RMSLE。表現最好的 workflows 是
LightGBM 等 GBDT large ensembles，而 preprocessing / feature engineering 是
關鍵差異。M3 在 anomaly-label setting 中呈現相同方向：值變化特徵帶來主要提升，
GBDT 集成模型提供小幅增益。

Miller et al. (2022, GEPIII limitations/error analysis) 分析 top-50 competition
solutions 的 RMSLE prediction residuals。III2 研究的是 energy-prediction
residuals，M3 研究的是 anomaly-label ranking AUC，因此本報告只保留 per-meter
ordering 對照。

| 電表類型 | III2 prediction-error pattern | M3 anomaly AUC |
|---|---|---:|
| Electricity | 最容易 / 預測最好 | 0.9994 |
| Chilled water | 比 electricity 困難 | 0.9890 |
| Steam | 比 electricity 困難 | 0.9536 |
| Hot water | III2 中最難，good fit 約四成 | 0.9876 |

表 5.1：regime=`timestamp_merge`；split ratio=80/20 building-held-out。

---

# 第 6 章：結論

M3 已完成，主要結論如下。

第一，完整 GEPIII 訓練資料上，50/50 建物留出的集成模型取得穩定的異常排序能力；
回溯標註 AUC 為 `0.9918`，僅用過去資訊 AUC 為 `0.9913`。

第二，值變化特徵是主要效能來源；從 17 個基礎特徵加入值變化特徵後，AUC 由
`0.9562` 提升到 `0.9925`。

第三，僅用過去資訊的版本只小幅低於回溯標註版本，表示線上評分設定仍具可行性；
線上評分時需排除未來電表讀值。

第四，跨站泛化、蒸氣表、歷史行偏移近似、標籤打亂殘餘結構與建物層級變異仍是後續階段
需要保留的限制。

因此，M3 的角色是建立 GEPIII 上的穩定基準線；後續比較應以相同切分、相同標籤橋接
與相同特徵可用性條件進行。

---

# 附錄：數字來源與補充檢查索引

補充檢查的程式碼與 JSON 輸出包含標籤對齊、行偏移差異、標籤打亂、時間留出、
訓練／驗證 gap、建物分布與抽樣種子掃描。主要來源如下：

+ 最終 50/50 結果：
  [scripts/run_m3_50_50_ensemble.py](../../scripts/run_m3_50_50_ensemble.py)；
  [docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。
+ 開發線與 golden metrics：
  [scripts/run_m3_split_causality.py](../../scripts/run_m3_split_causality.py)、
  [scripts/run_m3_3_budslab.py](../../scripts/run_m3_3_budslab.py)、
  [scripts/run_m3_4_ensemble.py](../../scripts/run_m3_4_ensemble.py)、
  [scripts/run_m3_5_postprocessing.py](../../scripts/run_m3_5_postprocessing.py)；
  [tests/golden_metrics.json](../../tests/golden_metrics.json)。
+ 補充檢查輸出：
  [scripts/run_gate_label_join_integrity.py](../../scripts/run_gate_label_join_integrity.py)、
  [scripts/run_inv1_meter_aware_impact.py](../../scripts/run_inv1_meter_aware_impact.py)、
  [scripts/run_inv4_shuffle_ablation.py](../../scripts/run_inv4_shuffle_ablation.py)、
  [scripts/run_inv5_time_holdout.py](../../scripts/run_inv5_time_holdout.py)、
  [scripts/run_inv6_train_val_gap.py](../../scripts/run_inv6_train_val_gap.py)、
  [scripts/run_inv7_per_building_distribution.py](../../scripts/run_inv7_per_building_distribution.py)、
  [scripts/run_inv8_sampling_fragility.py](../../scripts/run_inv8_sampling_fragility.py)；
  對應 JSON 皆位於 [data/processed/](../../data/processed/)。

*Last updated: 2026-07-16 (Issue #59 50/50 observation-only figures)*
