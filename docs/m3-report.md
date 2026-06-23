# M3 Report: Full ASHRAE GEPIII

**Status**: 完成
**Date**: 2026-06-22
**Task**: 使用完整 ASHRAE GEPIII train data，從 raw CSV 建立 features，並以
`bad_meter_readings.csv` anomaly label 做 binary classification。

M3 是 M2 LEAD reproduction 的延伸實驗。M2 驗證的是 LEAD competition subset
上的 paper reproduction；M3 驗證同一套方法論在完整 GEPIII train subset 上是否仍能
建立穩定 anomaly ranking model。M3 沒有 Kaggle leaderboard，因此所有結果皆為
building-held-out validation AUC。

---

# Ch1: 任務與評估設計

## 1.1 資料集

| 項目 | 值 |
|---|---:|
| Source | Kaggle `ashrae-energy-prediction` train data |
| Buildings | 1,449 |
| Rows | 20.2M |
| Meter types | electricity, chilled water, steam, hot water |
| Label source | buds-lab `bad_meter_readings.csv` |
| Label join | positional row-aligned join |
| Overall anomaly rate | 6.50% |

M3 使用本 repo 中的 GEPIII/Kaggle train subset。此結果應解讀為 GEPIII
anomaly-label reproduction，不外推到其他資料集或任務。

## 1.2 最終評估設計

最終報告採用 50/50 building split：

| Split | Train buildings | Validation buildings | Building overlap |
|---|---:|---:|---:|
| `building_id % 2` | 725 | 724 | 0 |

報告同時列出兩種 feature regime：

| Regime | Feature availability | Features | 解讀 |
|---|---|---:|---|
| Offline | Past + future value-change shifts | 137 | Batch labeling / retrospective analysis |
| Causal | Past-only value-change shifts | 77 | Online scoring with no future meter readings |

早期 80/20 experiments (`building_id % 5 == 4`, 1160/289 buildings) 只保留為
development 與 ablation evidence。它們用來在同一個 validation split 下比較
M3.1 到 M3.5，但不是最終報告 split。

---

# Ch2: 主要結果

最終模型是 four-model equal-weight ensemble：LightGBM, XGBoost, CatBoost,
HistGradientBoosting。Offline run 使用 M3.2 的 137-feature set；causal run
使用對應的 77-feature past-only set。

| Split | Regime | Features | Ensemble AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| 50/50 mod2 | Offline | 137 | **0.9921** | 0.7175 | 0.9387 | 0.8133 |
| 50/50 mod2 | Causal | 77 | **0.9911** | 0.7002 | 0.9311 | 0.7993 |

Machine-readable provenance: `docs/m3-50-50-ensemble.json`.

Offline score 是 50/50 protocol 下的 retrospective batch-labeling 結果。
Causal score 低 `0.0010`，量化了從 value-change features 中移除 future meter
readings 的成本。

M2 仍是 LEAD reproduction anchor：Kaggle Private `0.98616`，與原作者
`0.98661` 的 gap 為 `0.05%`。M3 不應與 M2 作直接 leaderboard score 比較，
因為 dataset、split、label 與 evaluation setup 都不同。

---

# Ch3: Ablation 結果

## 3.1 AUC Progression

下表整理 development path。80/20 rows 是 development evidence；50/50 rows 是
final protocol。

| Stage | Split | Regime | Features | AUC | 解讀 |
|---|---|---|---:|---:|---|
| M3.1 baseline | 80/20 | Offline | 17 | 0.9562 | Time, metadata, weather baseline |
| M3.2 value-change | 80/20 | Offline | 137 | 0.9920 | 主要 feature-engineering 跳躍 |
| M3.2 value-change | 80/20 | Causal | 77 | 0.9908 | Past-only value-change 仍然穩定 |
| M3.2 value-change | 50/50 | Offline | 137 | 0.9914 | 最終 split，single LightGBM |
| M3.2 value-change | 50/50 | Causal | 77 | 0.9903 | 最終 split，past-only LightGBM |
| M3.3 buds-lab alignment | 80/20 | Offline | 170 | 0.9913 | 沒有 robust AUC lift |
| M3.4 ensemble | 80/20 | Offline | 137 | 0.9928 | 小幅 ensemble lift |
| M3.4 ensemble | 50/50 | Offline | 137 | **0.9921** | 最終 offline result |
| M3.4 ensemble | 50/50 | Causal | 77 | **0.9911** | 最終 causal result |
| M3.5 post-processing | 80/20 | Offline | 137 | 0.9927 | Null result；rules 不轉移 |

## 3.2 Feature Engineering

M3.1 使用 17 個 baseline features：time features、building metadata、meter type、
meter reading、weather。M3.2 加入 120 個 value-change features，是主要
performance jump：

| Model | Features | AUC | Delta |
|---|---:|---:|---:|
| M3.1 baseline LightGBM | 17 | 0.9562 | - |
| M3.2 value-change LightGBM | 137 | 0.9920 | +0.0358 |

Value-change features 使用與 M2 相同的 shift family：`-24..-1`, `1..24`,
`-168..-48 step 24`, `48..168 step 24`，並同時使用 difference 與 ratio
forms；但 M3 的 diff sign 與 ratio orientation 和 M2 相反。M2 使用
`shift(n) - meter_reading` 與 `(shift(n)+1)/(meter_reading+1)`；M3 使用
`meter_reading - shift(n)` 與 `(meter_reading+1)/(shift(n)+1)`。這是 negation
and reciprocal 的 monotonic orientation difference，對 tree-based AUC 不改變；
見 ADR 0008。在 pandas 中，positive shifts 使用 past readings，negative shifts 使用
future readings。

## 3.3 Buds-lab Alignment

M3.3 補上 buds-lab GEPIII preprocessing 中優先級最高的 feature 類別：
cyclic time encodings、weather trailing lags and rolling means、holiday flags、
train-only `(site, meter)` target encoding、primary-use/meter interaction，
以及 site 0 meter 0 correction。

| Run | Features | AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| M3.2 reference | 137 | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| M3.3 buds-lab alignment | 170 | 0.9913 | 0.6668 | 0.9583 | 0.7864 |

Full buds-lab alignment 作為 validation/ablation step 有價值，但因為沒有改善
ranking AUC，所以不納入最終模型。Threshold-0.5 precision 與 F1 有改善，但
AUC 仍低於 M3.2 reference。

## 3.4 Ensemble

Ensemble 使用 M3.2 feature set，因為 M3.3 沒有改善 AUC。

| Model | 80/20 AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|
| LightGBM | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| XGBoost | 0.9909 | 0.6801 | 0.9559 | 0.7947 |
| CatBoost | 0.9891 | 0.7178 | 0.9579 | 0.8206 |
| HistGBT | 0.9915 | 0.6385 | 0.9650 | 0.7685 |
| Ensemble | **0.9928** | 0.6779 | 0.9664 | 0.7969 |

Ensemble lift 為正但幅度小：在 80/20 development split 上比 M3.2 LightGBM
高 `+0.00079`。定性結論仍是 value-change features 是主要貢獻，ensembling
是次要增益。

Threshold-0.5 precision 從 M3.2 的 `0.6409` 提升到 80/20 ensemble 的
`0.6779`。在最終 50/50 protocol 下，ensemble precision 為 offline `0.7175`、
causal `0.7002`。

## 3.5 Post-processing

M3.5 測試 M2/LEAD hard post-processing rules 是否能轉移到 raw GEPIII meter
readings。結果是不轉移。

| Rule | Trigger rows | Anomalies | ΔAUC vs pre |
|---|---:|---:|---:|
| Rule 1: `meter_reading == 1.0 -> 1` | 8 | 0 | -0.000002 |
| Rule 2a: Jan-1 start-point filter | 0 applied | 0 | 0.000000 |
| Rule 2b: `dayofyear > 366.9583 -> 0` | 478 | 13 | -0.000052 |
| Combined | - | - | -0.000054 |

Pre-rule ensemble AUC 是 `0.9927886`；combined post-processing AUC 是
`0.9927347`。這個 null result 表示 M2-specific post-processing patterns
不轉移到 raw GEPIII meter readings。

---

# Ch4: Validity Checks 與限制

Note: §4.1 validity checks and §4.2 generalization diagnostics are computed on
the 80/20 canonical development line (`building_id % 5 == 4`), not the final
50/50 split; read them as development evidence.

## 4.1 Leakage 與 split checks

沒有發現 train/validation building leakage 或 target-encoder leakage 的證據。
Future-shift features 是 non-causal，但在 offline batch-labeling setup 中與
past-shift features 呈現對稱訊號。

| Check | Result | 解讀 |
|---|---|---|
| Building overlap | 所有 reported splits 皆為 0 | Validation buildings 以 `building_id` held out。 |
| Past-only vs future-only | Past `0.9908`, future `0.9908`, full `0.9920` | Future shifts 是 non-causal，但不是 AUC 的唯一來源。 |
| Label shuffle, M3.2 seed 42 | 0.5669 | 高於 random；視為 residual structure/base-rate signal。 |
| Label shuffle, M3.5 seeds 42/123/999 | 0.5669 / 0.5669 / 0.4232, mean 0.5190 | Shuffle signal 不穩定，且遠低於 real-label result。 |
| Remove meter features | AUC drops to 0.8160 | Meter reading 與 value-change features 承載主要 anomaly signal。 |
| M3.3 target encoder ablation | Removing `gte_site_meter_anomaly` does not reduce shuffle AUC | Target encoding 不是 elevated shuffle result 的來源。 |

較穩妥的結論是：這些 checks 沒有顯示 split leakage，但 label-shuffle 結果顯示
dataset 中仍存在 metadata/base-rate structure。

## 4.2 Generalization diagnostics

| Diagnostic | Result | 解讀 |
|---|---:|---|
| Site-held-out ensemble AUC (`site_id % 5 == 4`) | 0.9774 | Cross-site validation 明顯比 building-held-out validation 更難。 |
| Per-meter AUC: electricity / chilled water / steam / hot water | 0.9991 / 0.9888 / 0.9553 / 0.9863 | Steam 是最弱 meter slice。 |
| Buildings with missing hours inside observed range | 945/1449 (65.2%) | Row-offset value-change shifts 是跨 timestamp gaps 的近似。 |

Value-change implementation 使用 `groupby().shift()`，因此 shifts 是 row-offset
features，而不是精確的 `timestamp + timedelta` joins。這是解讀 long-range
shifts 時的限制。

## 4.3 Primary-use slices

下表 AUC 由 `data/processed/m3_5_val_predictions.csv.gz` join
`data/raw/m3/building_metadata.csv` 計算而來。完整 machine-readable table 存在
`docs/m3-primary-use-auc.json`。

| Primary use | AUC | Rows | Anomalies | Buildings |
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

部分 primary-use categories 的 validation buildings 很少，因此不能把這些 slices
解讀成穩定的 building-type 結論。在較大的 slices 中，Education 最低
(`0.9894`)，Office 為 `0.9941`。

---

# Ch5: GEPIII 文獻對照

## 5.1 與 III1 對齊

Miller et al. (2020, GEPIII overview/results) 將 GEPIII 描述為大型
energy-prediction competition，評分指標為 RMSLE。表現最好的 workflows 是
LightGBM 等 GBDT large ensembles，而 preprocessing / feature engineering 是
關鍵差異。

M3 在 anomaly-label setting 中呈現相同大方向：

| III1 論點 | M3 發現 |
|---|---|
| Feature engineering 是核心 | Value-change features 讓 AUC 從 `0.9562` 提升到 `0.9920`。 |
| GBDT ensembles 表現強 | Four-model ensemble 在 80/20 development split 達 `0.9928`，在最終 50/50 offline split 達 `0.9921`。 |
| Electricity 相對容易 | Electricity anomaly AUC 為 `0.9991`，是最高的 meter slice。 |

## 5.2 與 III2 error analysis 的關係

Miller et al. (2022, GEPIII limitations/error analysis) 分析 top-50 competition
solutions 的 RMSLE prediction residuals。這與 M3 任務不同：III2 研究的是
energy-prediction residuals，M3 研究的是 anomaly-label ranking AUC。

這個任務差異解釋了 per-meter ordering 的不同：

| Meter type | III2 prediction-error pattern | M3 anomaly AUC |
|---|---|---:|
| Electricity | 最容易 / 預測最好 | 0.9991 |
| Chilled water | 比 electricity 困難 | 0.9888 |
| Steam | 比 electricity 困難 | 0.9553 |
| Hot water | III2 中最難，good fit 約四成 | 0.9863 |

在 RMSLE prediction-error framing 中，hot water 最難；但在 M3 anomaly-detection
framing 中，steam 是最弱 slice。這不是矛盾：prediction-error difficulty 與
anomaly-ranking difficulty 是不同任務。

III2 的 16-category error taxonomy、single-vs-multiple-building reach、以及
temporal-behavior categories 不直接採用，因為它們定義在 RMSLE prediction
residuals 上，而不是 binary anomaly-label ranking 上。

---

# Ch6: Summary

M3 已完成。最終 50/50 building-split ensemble 在 offline regime 達 `0.9921`
AUC，在 causal regime 達 `0.9911` AUC。主要 performance driver 是 M3.2
value-change feature engineering；buds-lab alignment 作為 ablation 有價值，
但不納入最終模型；ensemble 帶來小幅次要增益；hard-rule post-processing
沒有從 M2/LEAD 轉移到 raw GEPIII。

*Last updated: 2026-06-22 (M3 complete)*
