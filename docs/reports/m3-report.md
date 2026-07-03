# M3 報告：完整 ASHRAE GEPIII

+ **狀態**：完成
+ **日期**：2026-06-22
+ **任務**：使用完整 ASHRAE GEPIII 訓練資料，從原始 CSV 建立特徵，並以
  `bad_meter_readings.csv` 異常標籤進行二元分類。

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
+ Golden gates 與 metrics：[tests/golden_metrics.json](../../tests/golden_metrics.json)、[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。

---

# 第 1 章：任務與評估設計

## 1.1 資料集

| 項目 | 值 |
|---|---:|
| 資料來源 | ASHRAE GEPIII：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data) |
| 資料集 | `data/raw/m3/train.csv`、`data/raw/m3/bad_meter_readings.csv`、`data/raw/m3/building_metadata.csv`、`data/raw/m3/weather_train.csv` |
| 建物數 | 1,449 |
| 資料列數 | 20.2M |
| 電表類型 | electricity, chilled water, steam, hot water |
| 標籤來源 | buds-lab `bad_meter_readings.zip` |
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
| 50/50 mod2 | Offline batch labeling | 137 | 0.9921 | 0.7175 | 0.9387 | 0.8133 |
| 50/50 mod2 | Past-only | 77 | 0.9911 | 0.7002 | 0.9311 | 0.7993 |

機器可讀 provenance：
[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。

Offline batch labeling 分數是 50/50 protocol 下的批次回溯標註結果。Past-only
AUC 低 `0.0010`，量化了從 value-change features 中移除 future meter readings
的成本。

既有 train/validation gap 檢查使用 80/20 development line 與 meter-aware
value-change features，目的只在檢查是否出現明顯 fit-set memorization。

| Score | LightGBM | XGBoost | CatBoost | HistGradientBoosting | Ensemble |
|---|---:|---:|---:|---:|---:|
| Fit-set AUC | 0.9983 | 0.9996 | 0.9999 | 0.9983 | 0.9997 |
| Full train-buildings AUC | 0.9983 | 0.9996 | 0.9999 | 0.9982 | 0.9996 |
| Validation AUC | 0.9925 | 0.9925 | 0.9884 | 0.9921 | 0.9937 |

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
| M3.2 value-change | 80/20 | Offline | 137 | 0.9920 | 主要 feature-engineering 提升 |
| M3.2 value-change | 80/20 | Past-only | 77 | 0.9908 | past-shift features 仍提供穩定訊號 |
| M3.3 buds-lab alignment | 80/20 | Offline | 170 | 0.9913 | 補充特徵未改善 AUC |
| M3.4 ensemble | 80/20 | Offline | 137 | 0.9928 | ensemble 小幅提升 |
| M3.5 post-processing | 80/20 | Offline | 137 | 0.9927 | 舊 rules 在本資料設定下未帶來改善 |

## 3.2 Value-change features

M3.1 使用 17 個 baseline features：time features、building metadata、meter type、
meter reading 與 weather。M3.2 加入 120 個 value-change features。

| 模型 | 特徵數 | AUC | 差異 |
|---|---:|---:|---:|
| M3.1 baseline LightGBM | 17 | 0.9562 | - |
| M3.2 value-change LightGBM | 137 | 0.9920 | +0.0358 |

Value-change features 使用與 M2 相同的 shift family：`-24..-1`, `1..24`,
`-168..-48 step 24`, `48..168 step 24`。M3 的 diff sign 與 ratio orientation
和 M2 相反，屬 negation / reciprocal 的 monotonic 轉換；這個結論只界定
diff/ratio 方向本身，meter-crossing 的 AUC 影響見 §4.3。Ratio 的 +1
smoothing 公式見 ADR 0008。

## 3.3 buds-lab 對齊

M3.3 補上 buds-lab GEPIII preprocessing 中優先級最高的 feature 類別：
cyclic time encodings、weather trailing lags and rolling means、holiday flags、
train-only `(site, meter)` target encoding、primary-use/meter interaction，
以及 site 0 meter 0 correction。

| 執行 | 特徵數 | AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| M3.2 reference | 137 | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| M3.3 buds-lab 對齊 | 170 | 0.9913 | 0.6668 | 0.9583 | 0.7864 |

Full buds-lab alignment 作為驗證與消融步驟有價值；ranking AUC 未改善，
因此不納入最終模型。Threshold-0.5 指標列於上表。

## 3.4 Ensemble

Ensemble 使用 M3.2 feature set。

| 模型 | 80/20 AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|
| LightGBM | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| XGBoost | 0.9909 | 0.6801 | 0.9559 | 0.7947 |
| CatBoost | 0.9891 | 0.7178 | 0.9579 | 0.8206 |
| HistGBT | 0.9915 | 0.6385 | 0.9650 | 0.7685 |
| Ensemble | 0.9928 | 0.6779 | 0.9664 | 0.7969 |

Ensemble lift 為 `+0.00079`。Value-change features 是主要貢獻，ensemble 是次要
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

Pre-rule 集成模型 AUC 是 `0.9927886`；combined post-processing AUC 是
`0.9927347`。舊規則在本資料設定下未帶來改善。

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

M3 的 value-change features 以 row-offset shift 實作，用相鄰列近似時間變化，
沒有進行精確的 timestamp join。GEPIII default 以 `building_id` 分組；在多電表
資料中，這會讓不同 meter 的讀值進入同一個 shift 序列，形成 meter-crossing。

補充檢查顯示，`row_offset_meter_aware` 能修正 meter-crossing，且分數略高於原始
`row_offset` 版本。為了維持與凍結版 GEPIII reproduction baseline 的一致性，M3
headline result 仍回報原始 row-offset implementation；後續 cross-model comparison
則固定採用 meter-aware implementation。

Past-only 50/50 結果只比 offline batch labeling 低 `0.0010` AUC，顯示 M3 在即時
FDD 設定下仍保有接近 offline batch labeling 的表現。實際部署時，模型輸入應限制為
當下以前可取得的特徵。

整體來看，這些檢查界定了 M3 headline score 的適用條件：GEPIII 內部、
building-held-out split，以及 offline batch labeling 設定。在目前檢查範圍內，
未觀察到會系統性推高 headline result 的明顯 leakage pattern。後續比較應固定
split、label source 與 feature availability，並另外回報 cross-site、steam meter、
meter-aware feature implementation 與抽樣穩定度結果。

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
| Electricity | 最容易 / 預測最好 | 0.9991 |
| Chilled water | 比 electricity 困難 | 0.9888 |
| Steam | 比 electricity 困難 | 0.9553 |
| Hot water | III2 中最難，good fit 約四成 | 0.9863 |

---

# 第 6 章：結論

M3 已完成，主要結論如下。

第一，完整 GEPIII 訓練資料上，50/50 建物留出的集成模型取得穩定的異常排序能力；
回溯標註 AUC 為 `0.9921`，僅用過去資訊 AUC 為 `0.9911`。

第二，值變化特徵是主要效能來源；從 17 個基礎特徵加入值變化特徵後，AUC 由
`0.9562` 提升到 `0.9920`。

第三，僅用過去資訊的版本只小幅低於回溯標註版本，表示線上評分設定仍具可行性；
線上評分時需排除未來電表讀值。

第四，跨站泛化、蒸氣表、行偏移近似、標籤打亂殘餘結構與建物層級變異仍是後續階段
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

*Last updated: 2026-07-02 (M3 report structure and check evidence links)*
