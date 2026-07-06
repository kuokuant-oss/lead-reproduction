# M5.x：切分粒度比較（timestamp_merge causal77）

- **資料來源**：ASHRAE GEPIII，來源：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)。
- **資料集**：`data/raw/m3/train.csv`、`data/raw/m3/bad_meter_readings.csv`、`data/raw/m3/building_metadata.csv`、`data/raw/m3/weather_train.csv`。
- **Anomaly labels**：來自 buds-lab `bad_meter_readings.zip`。
- **Feature basis**：`timestamp_merge(causal77)` value-change features（17 baseline + 60 causal lag = 77 欄，只用過去 lag）。
- **Split**：時間切分。train `2016-01..2016-08`、calib `2016-09..2016-10`、test `2016-11..2016-12`；所有 config 共用同一組 test-window `eval_idx`。
- **輸出**：[data/processed/m5x_partition_granularity.json](../../data/processed/m5x_partition_granularity.json)。
- **Handoff**：[docs/handoffs/m5x-partition-granularity.md](../handoffs/m5x-partition-granularity.md)。

Last updated: 2026-07-06

---

## 1. 結論

本實驗在固定 `timestamp_merge(causal77)` 特徵與同一組 test-window 評估列下，比較四種 per-unit 建模粒度（C1 最細到 C3 最粗）對 tree family、tree ensemble 與 TabPFN 的影響。單 seed（42）、fit cap 10,000、calib cap 4,000、eval 4,000、每 config `max_units=400`。

**在每一個粒度，pooled PR-AUC 最高的都是 tree ensemble，TabPFN 的 pooled PR-AUC 在四個粒度全部低於 tree family。** 這與 M5.1 pooled building-heldout 的結論方向一致：TabPFN 的優勢在小樣本、低調參設定，而不是在 per-unit 切分下對 tree family 取得 PR-AUC 領先。

TabPFN 相對 tree family **最有利的粒度是 C2（`site_id, meter`）**：TabPFN pooled PR-AUC `0.8210`，與最高的 CatBoost `0.8504` 只差約 `0.03`，且 coverage 達 `99.75%`。**最不利的是 C3（`meter` only，最粗）**：TabPFN `0.6983`、tree ensemble `0.8606`，差距約 `0.16`。粒度與 TabPFN 相對表現非單調——中間粒度 C2 對 TabPFN 最友善。

最細的 C1（`building_id, meter`）不值得它的 coverage 代價。C1 在 eval 內共有 `1,928` 個 building×meter unit，受 `max_units=400` 限制只抽 400 個嘗試建模，其中僅 `303` 個 unit 的 train 段可 scorable（雙類別）而真正訓練，其餘 fallback 到 C3 meter-level 模型；換算成評估列，coverage 只有 `15.25%`、fallback rate `84.75%`。因此 C1 的 pooled 數字有 `84.75%` 由 C3 meter 模型的預測構成，與 C3 高度相關，並不是真正的 building-level 訊號。

per-unit NaN fraction 在四個粒度、non-fallback 與 fallback 兩個 cohort 都穩定在約 `3%`（global eval NaN fraction `0.0328`），**不隨粒度變細而上升**。TabPFN 在所有 config、所有 unit 都 `completed`、無 failure，在此 `~3%` NaN régime 下穩定;但這份穩定並未轉成 PR-AUC 領先。

**Decision：在 `timestamp_merge(causal77)` 下，per-unit 切分無法讓 TabPFN 在 pooled PR-AUC 上追上 tree family——tree ensemble 在每個粒度都領先。四個粒度中 C2（`site_id, meter`）是唯一讓 TabPFN 逼近最佳 tree（PR 差 ~0.03）且維持近全 coverage 的設定;最細的 C1 因 coverage 崩到 15% 而不具實用價值。**

---

## 2. 方法

每個 config 對 eval-window 的每個 unit 各自訓練 per-unit 模型;若 unit 的 train 段非 scorable（單一類別）或 calib 段為空，該 unit 的評估列 fallback 到 C3 meter-level 模型。六個模型在同一 unit 內共用相同 fit rows、calib rows 與 eval rows。JSON 記錄 row-index fingerprint、fairness asserts 與 provenance。

| 項目 | 設定 |
|---|---|
| value-change regime | `timestamp_merge(causal77)` |
| 特徵 | `BASELINE_FEATURE_COLS` + `PAST_SHIFTS` 的 `lag_value_diff_*` / `lag_value_ratio_*`，共 77 欄 |
| train window | 2016-01..2016-08 |
| calib window | 2016-09..2016-10 |
| test window | 2016-11..2016-12 |
| 評估列 | 全部 test-window 列的 natural-prevalence 隨機子抽樣，所有 config 共用同一組 `eval_idx`（4,000 列） |
| calib 列 | 每個 threshold scope 在 calib window 上用固定 natural-prevalence 子抽樣，cap = 4,000 |
| fit row budget | 每次模型呼叫先做 balanced fit，cap = 10,000 |
| max_units | 每 config 最多嘗試建模 400 個 unit，超出者 fallback |
| seed | 單 seed 42 |
| operating point | `threshold=0.5` 與 calib window 上固定 recall 0.90 後套用到 test |
| 模型 | LightGBM、XGBoost、CatBoost、HistGradientBoosting、四個 tree 等權 ensemble、TabPFN |

| Config | unit key | 說明 |
|---|---|---|
| C0_anchor | pooled building-heldout | 只引用 `m6_phaseD_50_50_full_models_timestamp_merge.json` 的 in-domain test ROC/PR，協定不同（pooled offline137），不列為競爭者 |
| C1 | `(building_id, meter)` | 最細的建築-表種粒度 |
| C2 | `(site_id, meter)` | site-表種粒度 |
| C3 | `(meter,)` | meter-level，同時是所有 config 的 fallback 來源 |
| C4 | `(primary_use, meter)` | 用 building metadata 補入 `primary_use` 後依用途-表種建模 |

Operating-point 限制：fallback 列使用 C3 meter-level 模型，pooled fixed-recall threshold 會將對應 C3 calib 子樣本併入，因此 `confusion@recall0.90` 反映實際 fallback row routing，而非對每個 C3 meter calib 子樣本去重後的分布。此限制只影響 fixed-recall operating point，**不影響 ROC-AUC / PR-AUC**;模型比較一律以 threshold-free ROC/PR-AUC 為準。

---

## 3. Pooled 結果（seed 42）

每個 config 一張表，列六模型的 pooled ROC-AUC、pooled PR-AUC、per-unit macro median ROC/PR、coverage、fallback rate。macro median 由該 config 的 scorable per-unit 求得（`n_scorable` 見表下），因 scorable unit 數少且多為可完美分離的小 unit，C1/C2 的 macro median 多為 `1.0`，屬小樣本 per-unit artifact，**pooled ROC/PR 才是本比較的主指標**。

### C0_anchor（參考，非競爭者;pooled building-heldout offline137）

| Model | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|
| LightGBM | 0.9871 | 0.9147 |
| XGBoost | 0.9869 | 0.8994 |
| CatBoost | 0.9884 | 0.9057 |
| HistGBT | 0.9888 | 0.9226 |
| Ensemble | 0.9895 | 0.9157 |
| TabPFN | 0.9915 | 0.9160 |

C0 是 pooled 多建築模型，PR-AUC（~0.91）遠高於任何 per-unit config（最佳 ~0.86），因為 per-unit 模型只用單一 unit 的少量列訓練。C0 只作錨點，協定不同不可直接比較。

### C1 `(building_id, meter)`｜coverage 15.25%、fallback 84.75%

available units `1,928`、selected `400`、trained `303`、`n_scorable = 47`。

| Model | Pooled ROC-AUC | Pooled PR-AUC | Macro ROC median | Macro PR median | Coverage | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9292 | 0.6817 | 1.0000 | 1.0000 | 0.1525 | 0.8475 |
| XGBoost | 0.9618 | 0.7843 | 1.0000 | 1.0000 | 0.1525 | 0.8475 |
| CatBoost | 0.9653 | 0.7962 | 1.0000 | 1.0000 | 0.1525 | 0.8475 |
| HistGBT | 0.9351 | 0.7435 | 1.0000 | 1.0000 | 0.1525 | 0.8475 |
| Ensemble | 0.9627 | **0.8168** | 1.0000 | 1.0000 | 0.1525 | 0.8475 |
| TabPFN | **0.9666** | 0.7369 | 1.0000 | 1.0000 | 0.1525 | 0.8475 |

C1 的 pooled PR-AUC 最高為 ensemble `0.8168`;TabPFN `0.7369` 低於 ensemble、CatBoost、XGBoost、HistGBT，僅高於 LightGBM。TabPFN 在 C1 的 pooled ROC 反而最高（`0.9666`），但這主要由 `84.75%` fallback（C3 meter 模型）主導，並非 building-level 判別力。

### C2 `(site_id, meter)`｜coverage 99.75%、fallback 0.25%

available units `39`、selected `39`、trained `37`、`n_scorable = 24`。

| Model | Pooled ROC-AUC | Pooled PR-AUC | Macro ROC median | Macro PR median | Coverage | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9873 | 0.8424 | 1.0000 | 1.0000 | 0.9975 | 0.0025 |
| XGBoost | 0.9848 | 0.8274 | 1.0000 | 1.0000 | 0.9975 | 0.0025 |
| CatBoost | 0.9863 | **0.8504** | 1.0000 | 1.0000 | 0.9975 | 0.0025 |
| HistGBT | 0.9871 | 0.8417 | 1.0000 | 1.0000 | 0.9975 | 0.0025 |
| Ensemble | 0.9869 | 0.8493 | 1.0000 | 1.0000 | 0.9975 | 0.0025 |
| TabPFN | 0.9691 | 0.8210 | 1.0000 | 1.0000 | 0.9975 | 0.0025 |

C2 是 TabPFN 相對 tree family 最有利的粒度：TabPFN pooled PR `0.8210`，距最高的 CatBoost `0.8504` 僅 `0.0294`，且 coverage `99.75%`。

### C3 `(meter,)`｜coverage 100%、fallback 0%

available units `4`、selected `4`、trained `4`、`n_scorable = 4`。此 config 無 fallback（自身即 fallback 來源）。

| Model | Pooled ROC-AUC | Pooled PR-AUC | Macro ROC median | Macro PR median | Coverage | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9878 | 0.8585 | 0.9828 | 0.8878 | 1.0000 | 0.0000 |
| XGBoost | 0.9896 | 0.8543 | 0.9823 | 0.8582 | 1.0000 | 0.0000 |
| CatBoost | 0.9893 | 0.8260 | 0.9905 | 0.8718 | 1.0000 | 0.0000 |
| HistGBT | 0.9876 | 0.8523 | 0.9814 | 0.8540 | 1.0000 | 0.0000 |
| Ensemble | 0.9899 | **0.8606** | 0.9875 | 0.8853 | 1.0000 | 0.0000 |
| TabPFN | 0.9671 | 0.6983 | 0.9737 | 0.7307 | 1.0000 | 0.0000 |

C3 是 TabPFN 相對最差的粒度：ensemble PR `0.8606`、TabPFN `0.6983`，差距 `0.1623`。C3 的 macro median 因 scorable unit（4 個 meter）皆有雙類別而非 `1.0`，是本實驗唯一 per-unit macro 有實質資訊的 config。

### C4 `(primary_use, meter)`｜coverage 99.78%、fallback 0.22%

available units `50`、selected `50`、trained `47`、`n_scorable = 21`。

| Model | Pooled ROC-AUC | Pooled PR-AUC | Macro ROC median | Macro PR median | Coverage | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9483 | 0.8331 | 0.9812 | 0.8819 | 0.9978 | 0.0022 |
| XGBoost | 0.9639 | 0.8369 | 0.9907 | 0.9107 | 0.9978 | 0.0022 |
| CatBoost | 0.9597 | 0.8221 | 0.9883 | 0.9235 | 0.9978 | 0.0022 |
| HistGBT | 0.9494 | 0.8195 | 0.9821 | 0.8857 | 0.9978 | 0.0022 |
| Ensemble | 0.9578 | **0.8380** | 0.9923 | 0.9480 | 0.9978 | 0.0022 |
| TabPFN | 0.9469 | 0.7452 | 0.9724 | 0.8833 | 0.9978 | 0.0022 |

C4 pooled PR 最高為 ensemble `0.8380`;TabPFN `0.7452` 低於全部 tree model，差距約 `0.09`。

---

## 4. Coverage 與 fallback 代價

| Config | unit key | available | selected | trained | Coverage | Fallback rate | 最佳 tree PR-AUC | TabPFN PR-AUC | TabPFN − 最佳 tree |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | `(building_id, meter)` | 1,928 | 400 | 303 | 0.1525 | 0.8475 | 0.8168 (Ensemble) | 0.7369 | −0.0799 |
| C2 | `(site_id, meter)` | 39 | 39 | 37 | 0.9975 | 0.0025 | 0.8504 (CatBoost) | 0.8210 | −0.0294 |
| C3 | `(meter,)` | 4 | 4 | 4 | 1.0000 | 0.0000 | 0.8606 (Ensemble) | 0.6983 | −0.1623 |
| C4 | `(primary_use, meter)` | 50 | 50 | 47 | 0.9978 | 0.0022 | 0.8380 (Ensemble) | 0.7452 | −0.0928 |

粒度越細，unit 數暴增、每 unit 可用列越少、scorable 比例越低，coverage 隨之崩壊。C1 只有 `15.25%` 的評估列由建築-表種模型覆蓋，其餘全數 fallback;C2/C4 的中間粒度維持近 `100%` coverage。TabPFN 相對 tree family 的 PR 差距在 C2 最小、C3 最大，與粒度非單調。

---

## 5. NaN fraction 與 TabPFN 穩定性

per-unit NaN fraction summary（median，non-fallback vs fallback cohort;eval split）。global eval NaN fraction `0.0328`。

| Config | non-fallback fit / calib / eval | fallback fit / calib / eval |
|---|---|---|
| C1 | 0.0370 / 0.0311 / 0.0303 | 0.0322 / 0.0280 / 0.0286 |
| C2 | 0.0377 / 0.0325 / 0.0328 | 0.0291 / 0.0266 / 0.0390 |
| C3 | 0.0355 / 0.0302 / 0.0329 | —（無 fallback） |
| C4 | 0.0366 / 0.0307 / 0.0335 | 0.0389 / 0.0323 / 0.0390 |

`timestamp_merge(causal77)` 的 lag 特徵在 eval 只有約 `3%` NaN，且 non-fallback 與 fallback cohort、四個粒度之間都維持在 `~3%`，**NaN fraction 不隨切分粒度變細而上升**。TabPFN 在所有 config、所有 unit 都 `status=completed`、無一 failure，在此 `~3%` NaN régime 下穩定。但 TabPFN 的穩定性在本任務並未轉成 PR-AUC 優勢——tree ensemble 在每個粒度的 pooled PR-AUC 都更高。

---

## 6. Fairness asserts（實算）

三個 fairness assert 由本次 JSON 實算，全部 `True`：

| Assert | 結果 | 說明 |
|---|---|---|
| `eval_idx_sha_all_equal` | True | 四個 config 共用同一組 `eval_idx`（sha256 一致） |
| `calib_idx_sha_all_models_equal_within_scope` | True | 同一 unit scope 內六模型共用同一 calib index |
| `no_future_leak` | True | 每個 trained unit 的 train 段時間戳早於其 test 段 |

另 `no_future_leak_global = True`、`feature_count = 77`。

---

## 7. 數字與程式碼索引

| 項目 | 程式碼 | 輸出 |
|---|---|---|
| M5.x partition granularity | [scripts/run_m5x_partition_granularity.py](../../scripts/run_m5x_partition_granularity.py) | [data/processed/m5x_partition_granularity.json](../../data/processed/m5x_partition_granularity.json) |
| Handoff notes | [docs/handoffs/m5x-partition-granularity.md](../handoffs/m5x-partition-granularity.md) | local handoff |
| Reference M5.1 report | [docs/reports/m5-1-deep-comparison.md](m5-1-deep-comparison.md) | report style reference |
| C0 anchor | [data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json](../../data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json) | pooled building-heldout 參考 |
| Frozen pipeline helpers | [src/lead/](../../src/lead/) | public helper surface |
