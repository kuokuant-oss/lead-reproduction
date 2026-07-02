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

+ M3.3 buds-lab 對齊：[scripts/run_m3_3_budslab.py](../../scripts/run_m3_3_budslab.py)。
+ M3.4 集成模型：[scripts/run_m3_4_ensemble.py](../../scripts/run_m3_4_ensemble.py)。
+ M3.5 事後規則：[scripts/run_m3_5_postprocessing.py](../../scripts/run_m3_5_postprocessing.py)。
+ 50/50 回溯標註與僅用過去資訊集成：[scripts/run_m3_50_50_ensemble.py](../../scripts/run_m3_50_50_ensemble.py)。
+ 切分與因果性診斷：[scripts/run_m3_split_causality.py](../../scripts/run_m3_split_causality.py)。
+ Golden gates 與 metrics：[tests/golden_metrics.json](../../tests/golden_metrics.json)、[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。

---

# 第 1 章：任務與評估設計

## 1.1 資料集

| 項目 | 值 |
|---|---:|
| 資料來源 | Kaggle `ashrae-energy-prediction` train data |
| 建物數 | 1,449 |
| 資料列數 | 20.2M |
| 電表類型 | electricity, chilled water, steam, hot water |
| 標籤來源 | buds-lab `bad_meter_readings.csv` |
| 標籤合併方式 | 逐列位置對齊 |
| 整體異常比例 | 6.50% |

## 1.2 最終評估設計

最終報告採用 50/50 建物切分：

| 切分 | 訓練建物數 | 驗證建物數 | 建物重疊 |
|---|---:|---:|---:|
| `building_id % 2` | 725 | 724 | 0 |

報告同時列出兩種特徵可用性設定：

| 設定 | 特徵可用性 | 特徵數 | 解讀 |
|---|---|---:|---|
| 回溯標註 | 過去與未來值變化位移 | 137 | 批次標註與事後分析 |
| 僅用過去資訊 | 只使用過去值變化位移 | 77 | 線上評分時不可使用未來讀值 |

早期 80/20 實驗（`building_id % 5 == 4`，1160/289 buildings，M3.1-M3.5）
集中放在第 3 章，作為開發階段證據與消融實驗。

## 1.3 解讀規則

以下結果中，最終結論以 50/50 建物切分為準。80/20 結果用於說明開發階段的特徵、
模型與診斷證據。小於約 `0.001` 的 AUC 差異落在抽樣與建物層級變異範圍內，
不作穩健提升宣稱。值變化特徵的貢獻以 17 特徵到 137 特徵的差異作為主要證據。

M2 可作重製錨點：Kaggle Private `0.98616`，與原作者 `0.98661` 的 gap 為
`0.05%`。M3 的資料、標籤與評估切分不同，因此 M2 分數只作背景參照。

---

# 第 2 章：最終結果

最終模型是 four-model equal-weight 集成模型：LightGBM、XGBoost、CatBoost、
HistGradientBoosting。回溯標註使用 137-feature set；僅用過去資訊使用對應的
77-feature set。

| 切分 | 設定 | 特徵數 | 集成模型 AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| 50/50 mod2 | 回溯標註 | 137 | 0.9921 | 0.7175 | 0.9387 | 0.8133 |
| 50/50 mod2 | 僅用過去資訊 | 77 | 0.9911 | 0.7002 | 0.9311 | 0.7993 |

機器可讀 provenance：
[docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json)。

回溯標註分數是 50/50 protocol 下的批次標註結果。僅用過去資訊的 AUC 低
`0.0010`，量化了從值變化特徵中移除未來電表讀值的成本。

---

# 第 3 章：開發階段證據與消融實驗

本章使用 80/20 開發線，目的是說明特徵、模型與事後規則對排序能力的影響。

## 3.1 AUC 進程

| 階段 | 切分 | 設定 | 特徵數 | AUC | 解讀 |
|---|---|---|---:|---:|---|
| M3.1 基礎特徵 | 80/20 | 回溯標註 | 17 | 0.9562 | 時間、建物、電表、天氣基礎特徵 |
| M3.2 值變化特徵 | 80/20 | 回溯標註 | 137 | 0.9920 | 主要 feature-engineering 提升 |
| M3.2 值變化特徵 | 80/20 | 僅用過去資訊 | 77 | 0.9908 | 過去值變化仍提供穩定訊號 |
| M3.3 buds-lab 對齊 | 80/20 | 回溯標註 | 170 | 0.9913 | 補充特徵未改善 AUC |
| M3.4 集成模型 | 80/20 | 回溯標註 | 137 | 0.9928 | 集成模型小幅提升 |
| M3.5 事後規則 | 80/20 | 回溯標註 | 137 | 0.9927 | 舊規則在本資料設定下未帶來改善 |

## 3.2 值變化特徵

M3.1 使用 17 個基礎特徵：時間特徵、建物 metadata、電表類型、電表讀值與天氣。
M3.2 加入 120 個值變化特徵。

| 模型 | 特徵數 | AUC | 差異 |
|---|---:|---:|---:|
| M3.1 基礎特徵 LightGBM | 17 | 0.9562 | - |
| M3.2 值變化特徵 LightGBM | 137 | 0.9920 | +0.0358 |

值變化特徵使用與 M2 相同的 shift family：`-24..-1`, `1..24`,
`-168..-48 step 24`, `48..168 step 24`。M3 的 diff sign 與 ratio orientation
和 M2 相反，屬 negation / reciprocal 的 monotonic 轉換，不影響 tree-based AUC；
ratio 的 +1 smoothing 公式見 ADR 0008。

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

## 3.4 集成模型

集成模型使用 M3.2 feature set。

| 模型 | 80/20 AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|
| LightGBM | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| XGBoost | 0.9909 | 0.6801 | 0.9559 | 0.7947 |
| CatBoost | 0.9891 | 0.7178 | 0.9579 | 0.8206 |
| HistGBT | 0.9915 | 0.6385 | 0.9650 | 0.7685 |
| 集成模型 | 0.9928 | 0.6779 | 0.9664 | 0.7969 |

集成模型 lift 為 `+0.00079`。值變化特徵是主要貢獻，集成模型是次要增益。補充的
建物層級 bootstrap 檢查顯示 lift 方向為正，建物層級 CI 仍有重疊，細節見第 4 章。

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

# 第 4 章：有效性檢查與補充診斷

本章彙整 80/20 開發線上的有效性檢查，以及後續補充調查的結果。這些結果用於界定
M3 分數的可信範圍與限制。

## 4.1 補充檢查索引

| 補充檢查 | 目的 | 主結論 | 位置 |
|---|---|---|---|
| 標籤打亂檢查 | 檢查異常殘餘訊號 | 未支持資料洩漏；值變化特徵在打亂標籤下仍有殘餘結構 | §4.2 |
| 行偏移檢查 | 量化值變化特徵的跨電表問題 | 電表感知版本略有改善；M3 default 維持凍結 | §4.3 |
| 時間留出檢查 | 衡量同建物時間外推 | held-out 時段未降低 AUC | §4.3 |
| 建物層級檢查 | 評估集成提升與高分切片 | lift 方向正向；建物層級 CI 有重疊 | §4.4 |
| 抽樣種子掃描 | 評估下採樣穩定度 | 平均分數穩定；小差異需保守解讀 | §4.5 |
| 訓練與驗證 gap | 檢查容量與穩定度 | full-train 對 validation 有約 `0.006` AUC gap | §4.5 |

## 4.2 切分與標籤檢查

切分檢查支持目前的建物留出設定：訓練與驗證建物沒有重疊，目標編碼消融也未顯示
異常提升來源。過去與未來位移在回溯標註設定中提供相近訊號。

| 檢查 | 結果 | 解讀 |
|---|---|---|
| 建物重疊 | 所有 reported splits 皆為 0 | 驗證建物以 `building_id` 留出。 |
| 僅用過去 / 僅用未來 / 全部位移 | Past `0.9908`, future `0.9908`, full `0.9920` | 過去與未來位移均提供相近訊號。 |
| 標籤打亂，8 seeds | mean 0.5197, std 0.0654 | Shuffle signal 不穩定，且遠低於 real-label result。 |
| 標籤打亂特徵消融 | remove value-change 0.5092; remove meter / building / weather 0.5251 / 0.5279 / 0.5253 | Shuffled-label 殘餘訊號主要來自 value-change 特徵。 |
| 移除電表特徵 | AUC drops to 0.8160 | Meter reading 與值變化特徵承載主要 anomaly signal。 |
| M3.3 target encoder 消融 | Removing `gte_site_meter_anomaly` does not reduce shuffle AUC | Target encoding 未形成 elevated shuffle result 的來源。 |

## 4.3 泛化診斷

| 診斷 | 結果 | 解讀 |
|---|---:|---|
| Site-held-out 集成模型 AUC (`site_id % 5 == 4`) | 0.9774 | Cross-site validation 明顯比 building-held-out validation 更難。 |
| Per-meter AUC: electricity / chilled water / steam / hot water | 0.9991 / 0.9888 / 0.9553 / 0.9863 | Steam 是最弱 meter slice。 |
| Observed range 內缺 hour 的建物 | 945/1449 (65.2%) | Row-offset value-change shifts 近似跨 timestamp gaps 的變化。 |
| 同建物時間留出 | single LightGBM 0.9907 → 0.9928; ensemble 0.9915 → 0.9937 | 使用 `row_offset_meter_aware` 與 `PAST_SHIFTS`；同一批建物的 held-out 時段未降低 AUC。 |

Value-change implementation 使用 `groupby().shift()`，因此 shifts 是 row-offset
features。GEPIII default 只以 `building_id` 分組；多 meter frame 會發生
meter-crossing。INV-1 量化顯示 `row_offset` 與 `row_offset_meter_aware` 在 GEPIII
上約 `57.6%` 至 `59.3%` 的 value-change cells 不同；80/20 split 的 meter-aware
single LightGBM AUC 差為 `+0.00053`，集成模型差為 `+0.00091`。M3 headline
保留 `row_offset` 作為凍結 reproduction default；M6 cross-model 比較線使用 opt-in
`row_offset_meter_aware`（見 #52）。

同建物時間留出檢驗衡量同一批建物的時間外推：train 為 2016-01-01 至
2016-08-31，validation 為 2016-09-01 至 2016-12-31；held-out 時段的建物在
訓練中出現過。此結果支持同建物時間外推設定下的 headline 穩定度。跨建物加跨時間的
泛化仍需另行量測。

## 4.4 建物用途切片與建物層級分布

下表 AUC 由 `data/processed/m3_5_val_predictions.csv.gz` join
`data/raw/m3/building_metadata.csv` 計算而來。完整 machine-readable table 存在
[docs/metrics/m3-primary-use-auc.json](../metrics/m3-primary-use-auc.json)。
部分 primary-use categories 的 validation buildings 很少，這些 slices 只作診斷用途。

INV-7 以 validation building 為單位重算分布。有效 validation buildings 為
`234`；per-building median AUC 為 single LightGBM `0.9996`、集成模型 `0.9999`；
single LightGBM p10/p90 為 `0.9751` / `1.0000`；minimum AUC 為 single LightGBM
`0.4061`、集成模型 `0.8042`。Building-bootstrap mean CI 為 single LightGBM
`[0.9802, 0.9928]`、集成模型 `[0.9928, 0.9970]`。High-score small slices 為
`primary_use_enc` 3、7、8、11、12、13、14、15，各自有不超過 4 個有效建物且
median AUC 接近 `0.999` 至 `1.000`。

| 建物用途 | AUC | 資料列數 | 異常列數 | 建物數 |
|---|---:|---:|---:|---:|
| Parking | 1.0000 | 26,349 | 7,063 | 3 |
| Retail | 1.0000 | 26,352 | 6,759 | 3 |
| Utility | 1.0000 | 14,944 | 62 | 1 |
| Warehouse/storage | 0.9997 | 34,078 | 1,093 | 4 |
| Healthcare | 0.9987 | 42,949 | 558 | 2 |
| Public services | 0.9984 | 317,758 | 9,918 | 32 |
| Lodging/residential | 0.9979 | 378,930 | 18,047 | 26 |
| Services | 0.9971 | 17,532 | 494 | 2 |
| Technology/science | 0.9970 | 22,276 | 279 | 1 |
| Food sales and service | 0.9969 | 26,343 | 18 | 1 |
| Entertainment/public assembly | 0.9950 | 520,050 | 20,453 | 41 |
| Office | 0.9941 | 863,244 | 62,886 | 52 |
| Other | 0.9932 | 35,134 | 737 | 3 |
| Education | 0.9894 | 1,766,403 | 113,683 | 117 |
| Manufacturing/industrial | 0.9876 | 6,677 | 1,148 | 1 |

## 4.5 穩定度與限制

Site-held-out validation 較 building-held-out 困難；steam 是最弱 meter slice；
label-shuffle 保留 value-change 殘餘結構；row-offset value-change shifts 近似跨
timestamp gaps 的變化且含 meter-crossing；primary-use slices 受 validation
building count 限制。

INV-6 train/validation gap 顯示 LightGBM fit-set AUC 為 `0.9983`、full
train-buildings AUC 為 `0.9983`、validation AUC 為 `0.9925`，full-train 減
validation 為 `+0.0058`。4-model 集成模型 fit-set AUC 為 `0.9997`、full
train-buildings AUC 為 `0.9996`、validation AUC 為 `0.9937`，gap 為 `+0.0059`。
Fit-set 與 full-train 分數接近，顯示分數不來自複製 fit-set 列的記憶；
full-train 對 validation 有 `0.006` 量級的容量 gap，列為 capacity/stability
caveat。

INV-8 sampling sweep 顯示 M3-style downsample seed 的 AUC mean 為 `0.9924`、
std 為 `0.00024`、range 為 `0.0008`；canonical seeds `(10,20)` 為 `0.9925`。
乾淨 50:50 且不複製正樣本的 mean 為 `0.9924`、std 為 `0.00027`；與 M3-style
mean 差為 `+0.00004`。Mean 不依賴正樣本複製；validation AUC 跨 sampling
seed 的 range `0.0008` 超過 `0.0005` noise floor，列為 sampling-seed
穩定度 caveat。

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

# 第 7 章：數字來源與程式碼索引

下表列出本報告中補充調查數字的來源，供讀者追溯到產生數字的程式與 JSON。

| 報告內容 | 程式碼 | 數字輸出 |
|---|---|---|
| 50/50 最終回溯標註與僅用過去資訊結果 | [scripts/run_m3_50_50_ensemble.py](../../scripts/run_m3_50_50_ensemble.py) | [docs/metrics/m3-50-50-ensemble.json](../metrics/m3-50-50-ensemble.json) |
| M3.1-M3.5 開發線、標籤打亂與因果性檢查 | [scripts/run_m3_split_causality.py](../../scripts/run_m3_split_causality.py), [scripts/run_m3_3_budslab.py](../../scripts/run_m3_3_budslab.py), [scripts/run_m3_4_ensemble.py](../../scripts/run_m3_4_ensemble.py), [scripts/run_m3_5_postprocessing.py](../../scripts/run_m3_5_postprocessing.py) | [tests/golden_metrics.json](../../tests/golden_metrics.json) |
| 標籤逐列對齊完整性 | [scripts/run_gate_label_join_integrity.py](../../scripts/run_gate_label_join_integrity.py) | [data/processed/gate_label_join_integrity.json](../../data/processed/gate_label_join_integrity.json) |
| 行偏移與電表感知值變化差異 | [scripts/run_inv1_meter_aware_impact.py](../../scripts/run_inv1_meter_aware_impact.py) | [data/processed/inv1_meter_aware_impact.json](../../data/processed/inv1_meter_aware_impact.json) |
| 標籤打亂特徵群消融 | [scripts/run_inv4_shuffle_ablation.py](../../scripts/run_inv4_shuffle_ablation.py) | [data/processed/inv4_shuffle_ablation.json](../../data/processed/inv4_shuffle_ablation.json) |
| 同建物時間留出 | [scripts/run_inv5_time_holdout.py](../../scripts/run_inv5_time_holdout.py) | [data/processed/inv5_time_holdout.json](../../data/processed/inv5_time_holdout.json) |
| Train/validation gap | [scripts/run_inv6_train_val_gap.py](../../scripts/run_inv6_train_val_gap.py) | [data/processed/inv6_train_val_gap.json](../../data/processed/inv6_train_val_gap.json) |
| Per-building AUC 分布與 bootstrap CI | [scripts/run_inv7_per_building_distribution.py](../../scripts/run_inv7_per_building_distribution.py) | [data/processed/inv7_per_building_distribution.json](../../data/processed/inv7_per_building_distribution.json) |
| Downsample seed 掃描與乾淨 50:50 對照 | [scripts/run_inv8_sampling_fragility.py](../../scripts/run_inv8_sampling_fragility.py) | [data/processed/inv8_sampling_fragility.json](../../data/processed/inv8_sampling_fragility.json) |

*Last updated: 2026-07-02 (M3 report structure and investigation evidence links)*
