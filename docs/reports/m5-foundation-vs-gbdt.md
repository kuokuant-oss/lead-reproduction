# M5：TabPFN vs GBDT 於 GEPIII

- **Issue**：[#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)、[#52](https://github.com/kuokuant-oss/lead-reproduction/issues/52)
- **資料來源**：ASHRAE GEPIII，來源：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)。
- **資料集**：`data/raw/m3/train.csv`、`data/raw/m3/bad_meter_readings.csv`、`data/raw/m3/building_metadata.csv`、`data/raw/m3/weather_train.csv`。
- **Anomaly labels**：來自 buds-lab `bad_meter_readings.zip`。
- **Feature basis**：`row_offset_meter_aware` value-change features。
- **Split**：50% training / 50% testing。In-domain、label-scarcity、minimal-FE 使用 50/50 building split；site-transfer 使用 50/50 site split。
- **輸出**：[data/processed/m6_phaseD_50_50_full_models.json](../../data/processed/m6_phaseD_50_50_full_models.json)。

---

## 1. 結論

TabPFN 的主要優勢出現在低標註量的 PR-AUC。
當 support 為 200 與 500 時，TabPFN 的 test PR-AUC 最高；support 增加後，
各模型差距快速收斂。

在 full features 的 in-domain 50/50 test split 上，六個模型分數接近。Test AUC
介於 0.9938 到 0.9947，test PR-AUC 介於 0.9207 到 0.9304。無明確模型排序。

Minimal feature engineering 顯示，raw features 下不同指標給出的訊號不同。

在 site-transfer 設定下，TabPFN test AUC 仍高，但 test PR-AUC 是六個模型中最低。
此軸以 anomaly ranking 來看，tree family 較強。

---

## 2. 方法

每個 cell 內，六個模型使用相同 feature matrix、scaler、split、fit rows、train
scoring rows 與 test scoring rows。唯一變因是模型。

| 項目 | 設定 |
|---|---|
| Models | LightGBM、XGBoost、CatBoost、HistGBT、Ensemble、TabPFN |
| Feature regime | `row_offset_meter_aware` |
| Full feature count | 137 features：17 baseline + 120 value-change |
| Minimal-FE count | 17 raw baseline features |
| Fit budget | 10,000 balanced rows；label-scarcity 另用 support sizes |
| Scoring budget | 4,000 natural-prevalence train rows 與 4,000 natural-prevalence test rows |
| In-domain split | `50_50_mod2`，依 `building_id % 2` 留出 |
| Site-transfer split | `site_id_mod2_50_50`，依 `site_id % 2` 留出 |
| AUC | AUC 指 ROC-AUC；PR-AUC 另列 |
| Confusion matrix | threshold `0.5` 與 fixed recall `0.90` |

Train/test 指標與 confusion matrix 都使用 natural-prevalence scoring subsample。
同一個 cell 內，六個模型使用同一批 scoring rows。JSON 內記錄 row-index fingerprint、
seed、sample size 與 prevalence。

Train/test 欄位意義如下：

| 欄位 | 意義 |
|---|---|
| Fit-set AUC | balanced fit-set 上的 AUC |
| Train AUC | train half 的 natural-prevalence scoring subsample AUC |
| Test AUC | test half 的 natural-prevalence scoring subsample AUC |

---

## 3. In-Domain

本節列出 full features 下的 fit-set、train 與 test 分數。Fit-set AUC 幾乎為
`1.0000`，train AUC 與 test AUC 也接近，未呈現明顯 train/test gap。

| Model | Fit-set AUC | Train AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 1.0000 | 0.9968 | 0.9941 | 0.9255 |
| XGBoost | 1.0000 | 0.9961 | 0.9938 | 0.9207 |
| CatBoost | 0.9995 | 0.9954 | 0.9942 | 0.9280 |
| HistGBT | 1.0000 | 0.9967 | 0.9946 | 0.9304 |
| Ensemble | 1.0000 | 0.9967 | 0.9946 | 0.9294 |
| TabPFN | 0.9999 | 0.9963 | 0.9947 | 0.9295 |

Threshold `0.5` 下，所有模型都抓到超過 `96%` 的 test anomalies。TabPFN 抓到
`245/250`，漏掉 `5`，false alarms 為 `146`。Ensemble 抓到 `241/250`，漏掉 `9`，
false alarms 為 `128`。

![In-domain test confusion matrix, threshold 0.5](assets/m5/m5_confusion_in_domain_threshold_0_5.png)

Fixed recall `0.90` 下，六個模型都固定抓到 `225/250` anomalies。此時 false alarms
介於 `56` 到 `71`，差距不大。

![In-domain test confusion matrix, fixed recall 0.90](assets/m5/m5_confusion_in_domain_fixed_recall_0_90.png)

---

## 4. Label Scarcity

本節列出不同 support size 下的 test PR-AUC。

| Support | LightGBM | XGBoost | CatBoost | HistGBT | Ensemble | TabPFN |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.7264 | 0.7103 | 0.7183 | 0.7238 | 0.7196 | 0.7675 |
| 500 | 0.7792 | 0.7727 | 0.8197 | 0.7876 | 0.8155 | 0.8220 |
| 1,000 | 0.8404 | 0.8330 | 0.8514 | 0.8576 | 0.8582 | 0.8534 |
| 2,000 | 0.8876 | 0.8759 | 0.8829 | 0.8830 | 0.8905 | 0.9007 |
| 5,000 | 0.9182 | 0.9167 | 0.9141 | 0.9091 | 0.9172 | 0.9191 |
| 10,000 | 0.9255 | 0.9207 | 0.9280 | 0.9304 | 0.9294 | 0.9296 |

Test AUC 在低 support 時也都偏高，因此本軸主要看 PR-AUC。

| Support | LightGBM | XGBoost | CatBoost | HistGBT | Ensemble | TabPFN |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.9799 | 0.9724 | 0.9809 | 0.9747 | 0.9799 | 0.9819 |
| 500 | 0.9850 | 0.9824 | 0.9856 | 0.9842 | 0.9856 | 0.9851 |
| 1,000 | 0.9850 | 0.9860 | 0.9863 | 0.9877 | 0.9874 | 0.9877 |
| 2,000 | 0.9899 | 0.9883 | 0.9880 | 0.9893 | 0.9888 | 0.9907 |
| 5,000 | 0.9918 | 0.9925 | 0.9922 | 0.9919 | 0.9927 | 0.9936 |
| 10,000 | 0.9941 | 0.9938 | 0.9942 | 0.9946 | 0.9946 | 0.9947 |

Support `200` 的 confusion matrix 顯示，PR-AUC ranking 與 threshold `0.5`
classification 給出不同訊號。TabPFN 抓到 `235/250`，漏掉 `15`，false alarms 為
`185`；Ensemble 抓到 `239/250`，漏掉 `11`，false alarms 為 `210`。

![Label scarcity support 200 test confusion matrix, threshold 0.5](assets/m5/m5_confusion_label_scarcity_support_200_threshold_0_5.png)

---

## 5. Minimal Feature Engineering

Minimal-FE 比較 full 137 features 與 raw 17 features。

### 5.1 Full 137 Features

| Model | Fit-set AUC | Train AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 1.0000 | 0.9968 | 0.9941 | 0.9255 |
| XGBoost | 1.0000 | 0.9961 | 0.9938 | 0.9207 |
| CatBoost | 0.9995 | 0.9954 | 0.9942 | 0.9280 |
| HistGBT | 1.0000 | 0.9967 | 0.9946 | 0.9304 |
| Ensemble | 1.0000 | 0.9967 | 0.9946 | 0.9294 |
| TabPFN | 0.9999 | 0.9962 | 0.9946 | 0.9279 |

### 5.2 Raw 17 Features

| Model | Fit-set AUC | Train AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 0.9983 | 0.9866 | 0.9730 | 0.8558 |
| XGBoost | 1.0000 | 0.9868 | 0.9746 | 0.8543 |
| CatBoost | 0.9964 | 0.9838 | 0.9762 | 0.8535 |
| HistGBT | 0.9979 | 0.9848 | 0.9692 | 0.8457 |
| Ensemble | 0.9991 | 0.9867 | 0.9760 | 0.8644 |
| TabPFN | 1.0000 | 0.9903 | 0.9545 | 0.7815 |

Raw 17 features 下，ranking metric 與 fixed-threshold classification 給出不同訊號。
Tree models 的 test PR-AUC 較高；TabPFN 在 threshold 0.5 下 TN 最高、FP 最少，
且 FP+FN 總錯誤數最低。

![Raw 17-feature test confusion matrix, threshold 0.5](assets/m5/m5_confusion_minimal_fe_raw17_threshold_0_5.png)

---

## 6. Site-Transfer

本節列出 site-transfer split 下的 fit-set、train 與 test 分數。

| Model | Fit-set AUC | Train AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 1.0000 | 0.9970 | 0.9669 | 0.6224 |
| XGBoost | 1.0000 | 0.9961 | 0.9771 | 0.6248 |
| CatBoost | 0.9998 | 0.9958 | 0.9698 | 0.6094 |
| HistGBT | 1.0000 | 0.9969 | 0.9781 | 0.6546 |
| Ensemble | 1.0000 | 0.9967 | 0.9757 | 0.6337 |
| TabPFN | 1.0000 | 0.9954 | 0.9756 | 0.5479 |

Threshold `0.5` 下，TabPFN 抓到最多 anomalies，false alarms 也較高。

![Site-transfer test confusion matrix, threshold 0.5](assets/m5/m5_confusion_site_transfer_threshold_0_5.png)

Fixed recall `0.90` 下，六個模型都抓到 `123/136` anomalies。False alarms 最低的是
HistGBT（`148`），接著是 TabPFN（`156`）與 Ensemble（`166`）。

![Site-transfer test confusion matrix, fixed recall 0.90](assets/m5/m5_confusion_site_transfer_fixed_recall_0_90.png)

---

## 7. 數字與程式碼索引

| 項目 | 程式碼 | 輸出 |
|---|---|---|
| 50/50 six-model comparison | [scripts/run_m6_phaseD_50_50_full_models.py](../../scripts/run_m6_phaseD_50_50_full_models.py) | [data/processed/m6_phaseD_50_50_full_models.json](../../data/processed/m6_phaseD_50_50_full_models.json) |
| Confusion matrix figures | generated from [data/processed/m6_phaseD_50_50_full_models.json](../../data/processed/m6_phaseD_50_50_full_models.json) | [docs/reports/assets/m5/](assets/m5/) |
| Meter-aware multi-seed comparison | [scripts/run_m5_phaseD_foundation_vs_gbdt.py](../../scripts/run_m5_phaseD_foundation_vs_gbdt.py) | [data/processed/m6_phaseD_meter_aware.json](../../data/processed/m6_phaseD_meter_aware.json) |
| Comparison regression tests | [tests/test_m5_phaseD_comparison.py](../../tests/test_m5_phaseD_comparison.py) | local test gate |
| TabPFN feasibility spike | [scripts/run_m5_phaseC_tabpfn_spike.py](../../scripts/run_m5_phaseC_tabpfn_spike.py), [tests/test_m5_tabpfn_spike.py](../../tests/test_m5_tabpfn_spike.py) | `data/processed/` Phase C outputs |
| Frozen pipeline helpers | [src/lead/data.py](../../src/lead/data.py), [src/lead/features.py](../../src/lead/features.py), [src/lead/split.py](../../src/lead/split.py), [src/lead/sample.py](../../src/lead/sample.py), [src/lead/evaluate.py](../../src/lead/evaluate.py) | [tests/golden_metrics.json](../../tests/golden_metrics.json) |
