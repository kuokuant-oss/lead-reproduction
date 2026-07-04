# M5.1：TabPFN 與 tree models 深度比較

- 資料來源：ASHRAE GEPIII，來源：[Kaggle ASHRAE Energy Prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)。
- 資料集：data/raw/m3/train.csv、data/raw/m3/bad_meter_readings.csv、data/raw/m3/building_metadata.csv、data/raw/m3/weather_train.csv。
- Anomaly labels：來自 buds-lab bad_meter_readings.zip。
- **Feature basis**：`timestamp_merge` value-change features。
- **Split**：50/50 building held-out；test half 為 `building_id % 2 == 1`，train half 再分成 fit buildings（`building_id % 4 == 0`）與 validation buildings（`building_id % 4 == 2`）。
- **Output**：[data/processed/m5_phaseD_deep_comparison.json](../../data/processed/m5_phaseD_deep_comparison.json)。
- **Handoff**：[docs/handoffs/m5-phaseD-deep-comparison.md](../handoffs/m5-phaseD-deep-comparison.md)。

---

## 1. 結論

本次重跑把 M5.1 deep comparison 接回 `debug_timestamp_merge` 的 canonical
pipeline。除 value-change regime 改為 `timestamp_merge` 外，fit rows、score rows、
scarcity sizes、tune trials、seed、split 規則與 model settings 均維持原實驗設定。

在 full 137-feature、10,000 fit rows 的主比較中，TabPFN 的 validation PR-AUC 最高
（`0.8858`），但 test PR-AUC 為 `0.9024`。timestamp_merge 下，tuned ensemble 的
test PR-AUC 最高（`0.9109`），tuned XGBoost 次高（`0.9100`），因此舊版
「tuned LightGBM 高於 TabPFN」的說法不再成立；新的結果是 tuned ensemble 與 tuned
XGBoost 在 test PR-AUC 上高於 TabPFN。

小樣本 sweep 顯示 TabPFN 的優勢從 support `100` 開始變得穩定。support `20` 與
`50` 時 CatBoost 的 test PR-AUC 最高；從 `100`、`150`、`300`、`500`、`1,000`
到 `2,000`，TabPFN 均為最高 test PR-AUC。JSON 的
`axes.sample_efficiency_fine.crossover_support` 記錄 support `20`：此時 best tree
為 CatBoost，test PR-AUC `0.7567`，TabPFN 為 `0.7456`。

feature dimensionality sweep 也和舊版不同：TabPFN 在 `17`、`50`、`137` features
三個維度都最高，最高點出現在 full `137` features，test PR-AUC `0.8520`；`50`
features 則為 `0.8479`。

seed stability 上，TabPFN 的六組 seed test PR-AUC mean 最高（`0.8984`），std 為
`0.0119`。同一輸入下 TabPFN 三次重跑 test PR-AUC 為 `0.9040`、`0.9049`、`0.9003`，
mean `0.9031`，std `0.0020`，顯示同輸入 run-to-run variation 小於跨抽樣/seed 變動。

---

## 2. 方法

每個 cell 都記錄 fit rows、train/val/test scoring rows、row-index fingerprint、
prevalence、sample size 與 provenance。核心表格以 threshold-free ROC-AUC 與 PR-AUC
為主；`threshold_0_5` 與 `fixed_recall_0_90` 只作為 post-hoc operating points。

| 項目 | 設定 |
|---|---|
| Models | LightGBM、XGBoost、CatBoost、HistGBT、Ensemble、TabPFN |
| Feature regime | `timestamp_merge` |
| Full feature count | 137 features（17 baseline + 120 value-change） |
| Fit buildings | `building_id % 4 == 0` |
| Val buildings | `building_id % 4 == 2` |
| Test buildings | `building_id % 2 == 1` |
| Fit budget | default `10,000` balanced rows；small-n axis 另跑 support sweep |
| Scoring budget | 4,000 natural-prevalence train rows、4,000 val rows、4,000 test rows |
| Tuning rule | tuned trees 以 validation PR-AUC 選 config |
| Core metrics | ROC-AUC、PR-AUC |

---

## 3. Full-Feature Out-Of-Sample Ranking

主比較使用 full `137` features 與 `10,000` balanced fit rows。Tree tuning 只看
validation PR-AUC；test half 保持 held-out，只用於最後評估。

### 3.1 Default Trees

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| LightGBM | 0.8763 | 0.9866 | 0.9086 |
| XGBoost | 0.8622 | 0.9837 | 0.8986 |
| CatBoost | 0.8682 | 0.9875 | 0.8791 |
| HistGBT | 0.8660 | 0.9832 | 0.8889 |
| Ensemble | 0.8722 | 0.9882 | 0.9011 |

Default tree 中，LightGBM 的 test PR-AUC 最高（`0.9086`），高於 default ensemble
（`0.9011`）。

### 3.2 Tuned Trees

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| LightGBM | 0.8839 | 0.9856 | 0.9006 |
| XGBoost | 0.8767 | 0.9869 | 0.9100 |
| CatBoost | 0.8797 | 0.9865 | 0.8891 |
| HistGBT | 0.8829 | 0.9852 | 0.9068 |
| Ensemble | 0.8789 | 0.9879 | 0.9109 |

Tuning 後，test PR-AUC 最高的是 tuned ensemble（`0.9109`），不是 tuned LightGBM。
Tuned XGBoost (`0.9100`) 也高於 TabPFN。

### 3.3 TabPFN

| Model | Val PR-AUC | Test AUC | Test PR-AUC |
|---|---:|---:|---:|
| TabPFN | 0.8858 | 0.9852 | 0.9024 |

TabPFN 的 validation PR-AUC 最高，但 test PR-AUC 低於 tuned ensemble、tuned XGBoost
與 default LightGBM。這表示 timestamp_merge 下，TabPFN 的 validation ranking 優勢沒有
完全轉成 full-feature test PR-AUC 第一名。

---

## 4. 小樣本標註效率

此軸固定 scoring rows 為 natural-prevalence，fit set 則使用 balanced support。表格列出
test PR-AUC。

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

CatBoost 在 support `20` 與 `50` 勝出；TabPFN 從 support `100` 起一路最高。
`crossover_support` 記錄 support `20`，best tree 為 CatBoost `0.7567`，TabPFN 為
`0.7456`。因此新的結論不是「TabPFN 在所有小樣本 support 都勝出」，而是
「極小 support 下 CatBoost 可領先；support 達 100 後 TabPFN 穩定領先」。

---

## 5. 小樣本下的 Feature Dimensionality

此軸固定 fit rows 為 `500`，比較 `17`、`50`、`137` features 的 test PR-AUC。

| Feature Set | Features | LightGBM | XGBoost | CatBoost | HistGBT | Ensemble | TabPFN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw baseline | 17 | 0.6649 | 0.6623 | 0.7077 | 0.6776 | 0.7070 | 0.7580 |
| Baseline + first 33 value-change | 50 | 0.7791 | 0.7596 | 0.7519 | 0.7710 | 0.7772 | 0.8479 |
| Full value-change | 137 | 0.8000 | 0.7708 | 0.7350 | 0.7651 | 0.7550 | 0.8520 |

TabPFN 在三個 feature dimensions 都是最高 test PR-AUC，最高點為 full `137`
features 的 `0.8520`。`50` features 仍是 TabPFN 的明顯高點（`0.8479`），但不再是
最高維度；full feature table 在 timestamp_merge 下略高。

---

## 6. Seed Stability 與 TabPFN 同輸入重跑

此軸固定 full `137` features 與 `10,000` fit rows，跑
`DOWNSAMPLE_SEEDS x MODEL_SEEDS` 六組組合。表格列出 test ROC-AUC 與 test PR-AUC 的
mean/std。

| Model | Test AUC mean/std | Test PR-AUC mean/std |
|---|---:|---:|
| LightGBM | 0.9879 / 0.0019 | 0.8913 / 0.0131 |
| XGBoost | 0.9869 / 0.0024 | 0.8818 / 0.0138 |
| CatBoost | 0.9857 / 0.0030 | 0.8743 / 0.0129 |
| HistGBT | 0.9882 / 0.0020 | 0.8849 / 0.0138 |
| Ensemble | 0.9877 / 0.0021 | 0.8871 / 0.0155 |
| TabPFN | 0.9859 / 0.0018 | 0.8984 / 0.0119 |

TabPFN 的 six-run test PR-AUC mean 最高（`0.8984`），但 test AUC mean 不是最高；
HistGBT 的 test AUC mean 為 `0.9882`。同一輸入下 TabPFN 三次重跑的 test PR-AUC 為
`0.9040`、`0.9049`、`0.9003`，mean `0.9031`，std `0.0020`。這表示同輸入重跑的
variation 小於跨 seed/subsample variation。

---

## 7. Traceability

| 項目 | 程式 | 輸出 |
|---|---|---|
| M5.1 deep comparison | [scripts/run_m5_phaseD_deep_comparison.py](../../scripts/run_m5_phaseD_deep_comparison.py) | [data/processed/m5_phaseD_deep_comparison.json](../../data/processed/m5_phaseD_deep_comparison.json) |
| Handoff notes | [docs/handoffs/m5-phaseD-deep-comparison.md](../handoffs/m5-phaseD-deep-comparison.md) | local handoff |
| Reference M5 report | [docs/reports/m5-foundation-vs-gbdt.md](m5-foundation-vs-gbdt.md) | report style reference |
| Comparison atoms | [scripts/run_m6_phaseD_50_50_full_models.py](../../scripts/run_m6_phaseD_50_50_full_models.py) | reused metrics and model helpers |
| Frozen pipeline helpers | [src/lead/data.py](../../src/lead/data.py), [src/lead/features.py](../../src/lead/features.py), [src/lead/split.py](../../src/lead/split.py), [src/lead/sample.py](../../src/lead/sample.py) | public helper surface |
