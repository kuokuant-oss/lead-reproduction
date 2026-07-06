# M5.1：TabPFN 調參敏感度與小樣本標註效率深入比較

- **資料來源**：ASHRAE GEPIII，來源：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)。
- **資料集**：`data/raw/m3/train.csv`、`data/raw/m3/bad_meter_readings.csv`、`data/raw/m3/building_metadata.csv`、`data/raw/m3/weather_train.csv`。
- **Anomaly labels**：來自 buds-lab `bad_meter_readings.zip`。
- **Feature basis**：`timestamp_merge` value-change features。
- **Split**：頂層使用 50/50 building held-out。Test half 為 `building_id % 2 == 1`；train half 內再切 fit buildings（`building_id % 4 == 0`）與 val buildings（`building_id % 4 == 2`）。
- **輸出**：[data/processed/m5_phaseD_deep_comparison.json](../../data/processed/m5_phaseD_deep_comparison.json)。
- **Handoff**：[docs/handoffs/m5-phaseD-deep-comparison.md](../handoffs/m5-phaseD-deep-comparison.md)。

---

## 1. 結論

本比較使用 TabPFN-3 local，評估其在 ASHRAE GEPIII anomaly detection 任務中，與 LightGBM、XGBoost、CatBoost、HistGBT 與 tree ensemble 的表現差異。整體結果與 TabPFN 主流論文的大方向一致：TabPFN 的主要優勢出現在小樣本、中低維特徵與低調參負擔的設定下，而不是在所有資料量或所有 tree tuning 強度下全面勝出。

在小樣本標註效率上，TabPFN-3 local 表現突出。Support 為 `100`、`150`、`300`、`500`、`1,000`、`2,000` 時，TabPFN 的 test PR-AUC 高於所有 tree models。Support 為 `20` 與 `50` 時，CatBoost 最高；這表示在極小 support 下，tree model 仍可能取得較高 PR-AUC，但 support 達 `100` 後 TabPFN 較能維持 anomaly detection 的排序品質。

在 feature 維度承載能力上，TabPFN-3 local 維持優勢。固定 fit rows 為 `500` 時，TabPFN 在 `17`、`50`、`137` features 三種設定下都取得最高 test PR-AUC。其中 `137` features 時，TabPFN test PR-AUC 達 `0.8520`，是本軸最高結果。

在完整 fit budget 與 tree tuning 後，TabPFN-3 local 不再全面領先。Tuned ensemble 的 test PR-AUC 為 `0.9109`，高於 TabPFN 的 `0.9024`；tuned XGBoost 為 `0.9100`，default LightGBM 為 `0.9086`，也高於 TabPFN。

穩定性方面，TabPFN-3 local 在六次 `DOWNSAMPLE_SEEDS x MODEL_SEEDS` 重跑中的 test PR-AUC mean 為 `0.8984`，高於所有 tree baselines，std 為 `0.0119`。同一批輸入重跑三次時，test PR-AUC std 為 `0.0020`，顯示 TabPFN 在固定輸入下的 run-to-run variation 很低。

總結，TabPFN-3 local 在本 anomaly detection 任務的小樣本與中低維特徵設定下具明顯競爭力，並呈現良好穩定性；但在完整 fit budget 或 tuned tree models 充分調參後，tree models 仍可能取得更高 test PR-AUC。本比較認為「TabPFN-3 local 是小樣本 tabular anomaly detection 的強基準模型」。

---

## 2. 方法

每個 cell 內，六個模型使用相同 fit rows、train scoring rows、val scoring rows 與 test scoring rows。JSON 內記錄 row-index fingerprint、prevalence、sample size 與 provenance。

| 項目 | 設定 |
|---|---|
| Models | LightGBM、XGBoost、CatBoost、HistGBT、Ensemble、TabPFN |
| Feature regime | `timestamp_merge` |
| Full feature count | 137 features：17 baseline + 120 value-change |
| Fit buildings | `building_id % 4 == 0` |
| Val buildings | `building_id % 4 == 2` |
| Test buildings | `building_id % 2 == 1` |
| Fit budget | default `10,000` balanced rows；small-n 與 support sweep 另列 |
| Scoring budget | 4,000 natural-prevalence train rows、4,000 val rows、4,000 test rows |
| Tuning rule | tuned trees 只用 val PR-AUC 選 config |
| Core metrics | ROC-AUC、PR-AUC |

Test 的 `threshold_0_5` 與 `fixed_recall_0_90` 是 post-hoc operating points。`fixed_recall_0_90` 的 threshold 由同一 split 的 labels 求得，包含 test summary 的 test labels。這些數字只描述該 scoring subsample。模型比較使用 threshold-free ROC-AUC 與 PR-AUC。

---

## 3. 調參敏感度與 out-of-sample ranking

本軸測量模型在 full feature table 上的 ranking 能力，以及 tree models 經 validation tuning 後能否改變排序。Tree tuning 用手動搜尋，所有 trial 在 fit rows 上訓練，在 val rows 上以 PR-AUC 選 config。Test half 只在選型完成後 scoring。本軸對應的能力是：在相同 fit rows、相同 scoring rows 下，default trees、tuned trees 與 TabPFN 對 anomaly rows 的排序品質。

### 3.1 Default Trees

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| LightGBM | 0.8763 | 0.9866 | 0.9086 |
| XGBoost | 0.8622 | 0.9837 | 0.8986 |
| CatBoost | 0.8682 | 0.9875 | 0.8791 |
| HistGBT | 0.8660 | 0.9832 | 0.8889 |
| Ensemble | 0.8722 | 0.9882 | 0.9011 |

### 3.2 Tuned Trees

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| LightGBM | 0.8839 | 0.9856 | 0.9006 |
| XGBoost | 0.8767 | 0.9869 | 0.9100 |
| CatBoost | 0.8797 | 0.9865 | 0.8891 |
| HistGBT | 0.8829 | 0.9852 | 0.9068 |
| Ensemble | 0.8789 | 0.9879 | 0.9109 |

### 3.3 TabPFN

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| TabPFN | 0.8858 | 0.9852 | 0.9024 |

Tuned ensemble 取得本軸最高 test PR-AUC。TabPFN 取得最高 val PR-AUC，test PR-AUC 接近 tuned HistGBT，但低於 tuned ensemble、tuned XGBoost 與 default LightGBM。

---

## 4. 小樣本標註效率

本軸測量模型在 labeled support 很少時維持 anomaly ranking 的能力。Support size 從 `20` 到 `2,000`，fit rows 以 balanced subsample 控制，scoring rows 固定為 natural-prevalence。表格列 test PR-AUC。

| Support | LightGBM | XGBoost | CatBoost | HistGBT | Ensemble | TabPFN |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.0622 | 0.1654 | 0.7567 | 0.0622 | 0.7388 | 0.7456 |
| 50 | 0.3994 | 0.5862 | 0.7525 | 0.4967 | 0.6176 | 0.7293 |
| 100 | 0.6516 | 0.6924 | 0.7223 | 0.6524 | 0.7047 | 0.7570 |
| 150 | 0.7075 | 0.6755 | 0.7316 | 0.6940 | 0.7228 | 0.7979 |
| 300 | 0.7551 | 0.7574 | 0.7172 | 0.7681 | 0.7442 | 0.8194 |
| 500 | 0.8000 | 0.7708 | 0.7350 | 0.7651 | 0.7550 | 0.8520 |
| 1,000 | 0.8020 | 0.8228 | 0.7638 | 0.8189 | 0.8053 | 0.8473 |
| 2,000 | 0.8487 | 0.8244 | 0.8535 | 0.8545 | 0.8651 | 0.8688 |

Best tree 高於 TabPFN 的第一個 support 為 `20`。該 cell 中 CatBoost test PR-AUC 為 `0.7567`，TabPFN 為 `0.7456`。Support 為 `100` 後，TabPFN 在所有列出的 support 上取得最高 test PR-AUC。

---

## 5. 小樣本下的 feature 維度承載能力

本軸測量模型在 fit rows 固定為 `500` 時，能否有效使用更多 feature dimensions。`17` features 是 raw baseline，`50` features 使用 baseline 17 欄加上前 33 個 value-change 欄位，`137` features 使用完整 value-change table。表格列 test PR-AUC。

| Feature Set | Features | LightGBM | XGBoost | CatBoost | HistGBT | Ensemble | TabPFN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw baseline | 17 | 0.6649 | 0.6623 | 0.7077 | 0.6776 | 0.7070 | 0.7580 |
| Baseline + first 33 value-change | 50 | 0.7791 | 0.7596 | 0.7519 | 0.7710 | 0.7772 | 0.8479 |
| Full value-change | 137 | 0.8000 | 0.7708 | 0.7350 | 0.7651 | 0.7550 | 0.8520 |

TabPFN 在三個 feature dimensions 都最高。`137` features 是本軸 TabPFN 的最高點。

---

## 6. Seed 穩定性與 TabPFN 同輸入變異

本軸測量兩種穩定性。第一種是 fit subsample 與 model seed 改變時，模型分數的 seed-to-seed variation。第二種是 TabPFN 在完全相同輸入下重跑時的 run-to-run variation。本軸固定 full 137 features 與 `10,000` fit rows，使用 `DOWNSAMPLE_SEEDS x MODEL_SEEDS`，共六次重跑。表格列 test ROC-AUC 與 test PR-AUC 的 mean/std。

| Model | Test AUC mean/std | Test PR-AUC mean/std |
|---|---:|---:|
| LightGBM | 0.9879 / 0.0019 | 0.8913 / 0.0131 |
| XGBoost | 0.9869 / 0.0024 | 0.8818 / 0.0138 |
| CatBoost | 0.9857 / 0.0030 | 0.8743 / 0.0129 |
| HistGBT | 0.9882 / 0.0020 | 0.8849 / 0.0138 |
| Ensemble | 0.9877 / 0.0021 | 0.8871 / 0.0155 |
| TabPFN | 0.9859 / 0.0018 | 0.8984 / 0.0119 |

TabPFN 的 six-run test PR-AUC mean 最高，std 低於 tree ensemble。

同一批輸入下，TabPFN 重跑三次的 test PR-AUC 為 `0.9040`、`0.9049`、`0.9003`。Mean 為 `0.9031`，std 為 `0.0020`。

---

## 7. 數字與程式碼索引

| 項目 | 程式碼 | 輸出 |
|---|---|---|
| M5.1 deep comparison | [scripts/run_m5_phaseD_deep_comparison.py](../../scripts/run_m5_phaseD_deep_comparison.py) | [data/processed/m5_phaseD_deep_comparison.json](../../data/processed/m5_phaseD_deep_comparison.json) |
| Handoff notes | [docs/handoffs/m5-phaseD-deep-comparison.md](../handoffs/m5-phaseD-deep-comparison.md) | local handoff |
| Reference M5 report | [docs/reports/m5-foundation-vs-gbdt.md](m5-foundation-vs-gbdt.md) | report style reference |
| Comparison atoms | [scripts/run_m6_phaseD_50_50_full_models.py](../../scripts/run_m6_phaseD_50_50_full_models.py) | reused metrics and model helpers |
| Frozen pipeline helpers | [src/lead/data.py](../../src/lead/data.py), [src/lead/features.py](../../src/lead/features.py), [src/lead/split.py](../../src/lead/split.py), [src/lead/sample.py](../../src/lead/sample.py) | public helper surface |
