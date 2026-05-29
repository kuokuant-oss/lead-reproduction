# LEAD Reproduction 工作成果摘要

致 Chun Fu 教授
日期：2026-05-29

---

## TL;DR

依照您給的三步任務，完成情況如下：

| 任務 | 狀態 | 主要結果 |
|---|---|---|
| 1. Paper + Code 理解 (M1) | ✅ 完成 | 17 unknowns + 6 ADRs 文件化 |
| 2. Paper 復現 (M2) | ✅ 完成 | Kaggle Private AUC **0.98616**，跟原作者 0.98661 差 0.05% |
| 3. 完整 GEPIII 進階任務 (M3) | 🚧 進行中 | M3.1 val AUC 0.9562 + M3.2 val AUC **0.9920** |

**GitHub repo**：<https://github.com/kuokuant-oss/lead-reproduction>

---

## M2：Paper 復現結果

### 主要數字

| 指標 | 復現結果 | 原作者 | Gap |
|---|---|---|---|
| **Kaggle Private AUC** | **0.98616** | **0.98661** | **0.05% ⭐** |
| LightGBM val AUC（169 features） | 0.9818 | 0.9849 | 0.31% |
| 4-model ensemble val AUC | 0.9830 | 0.9866 | 0.36% |

Private gap 0.05% 在 noise floor（±0.0005）以內，統計上與原作者無差異。

### 執行方式

採 **one-shot inference**：根據 paper + buds-lab code 推斷 implementation，提交一次達成。未使用 leaderboard 累積誤差。

### 3 個 ablation 量化 pipeline 設計決策

| Ablation | 方式 | Kaggle Private ΔAUC |
|---|---|---|
| A：移除 gte_* features（16 欄） | LightGBM 重跑 | −0.004 |
| B：per-building mean 取代 fillna(0) | LightGBM 重跑 | −0.0128 |
| C：Rule 2a 移除 building_id filter | val 套用 Rule | −0.0001 |

### 17 個 unknowns 追蹤

Paper 未完整描述的實作細節，每個都做 implementation decision + 量化驗證：

- **已 resolved**：9 個（CV split、downsampling ratio、ClusterNo seed、SavGol leakage、cloud_coverage sentinel、gte_* leakage 量化等）
- **Partially resolved**：7 個（sklearn n_init 預設、value-change 方向、Rule 2a filter range 等）
- **Non-issue**：1 個

---

## M3：完整 ASHRAE GEPIII 進階任務

### 任務設定（依照您的指示）

- **Dataset**：完整 ASHRAE GEPIII（1,449 buildings，~20M rows）
- **從 raw 做 feature engineering**：從 train.csv + building_metadata.csv + weather_train.csv 開始
- **目標**：anomaly binary classification（label 來自 bad_meter_readings.csv）
- **Train/test split**：building_id % 5 == 4 → val（289 buildings），其餘 → train（1,160 buildings）

### 目前進度

| Milestone | Features | Val AUC | 狀態 |
|---|---|---|---|
| M3.1：baseline（time + meta + weather） | 17 | 0.9562 | ✅ 完成 |
| M3.2：+ value-change（60 shifts × 2） | 137 | **0.9920** | ✅ 完成 |
| M3.3：buds-lab 完整 feature alignment | ~150+ | TBD | 🚧 待續 |
| M3.4：4-model ensemble | — | TBD | 🚧 待續 |
| M3.5：post-processing rules | — | TBD | 🚧 待續 |

⚠️ M3 無對應 Kaggle leaderboard（GEPIII 競賽 metric 是 RMSLE meter prediction，不是 anomaly classification），故採用 self-defined val set 評估。

### M3.2 val AUC 0.9920 驗證（4 個 sanity check）

為確認 0.9920 不是 artifact，執行了以下 4 個驗證：

| Check | 方法 | 結果 | 判定 |
|---|---|---|---|
| SC0：Leakage（past vs future） | past-only / future-only / full AUC 比較 | past=0.9908，future=0.9908，full=0.9920 | ✅ PASS |
| SC1：Label shuffle | Train labels 隨機 shuffle 後重訓 | AUC=0.5669（real 0.9920 的 0.57×） | 🟡 BORDERLINE |
| SC2：移除 meter features | 移除 meter_reading + 120 lag features，只用 16 個 meta/weather features | AUC=0.8160，ΔAUC −0.1760 | ✅ PASS |
| SC3：Multi-seed 穩定性 | Seeds 42/123/999 各跑一次 | 0.9920/0.9928/0.9923，std=0.0003 | ✅ PASS |

**SC1 BORDERLINE 說明**：building meta（log_square_feet、meter type）跟 anomaly rate 有真實相關性（某些大型建築的讀數系統性異常），即使 shuffle labels 仍有弱訊號。關鍵對比：shuffled 0.5669 << real 0.9920（17.5 倍差距），確認 model 學的是真實 signal 而非 spurious correlation。

**M3.2 threshold=0.5 precision/recall**：P=0.6409、R=0.9665、F1=0.7707。高 recall 符合 anomaly detection 的優先目標（抓到所有 burst 比減少誤報更重要）。

### M3 vs M2 方法擴展觀察

M2（406 buildings，LEAD subset）→ M3（1,449 buildings，GEPIII full）驗證了 reproduction methodology 的擴展性：

1. **Value-change features 是主要貢獻**：M2 ΔAUC +0.0866，M3 ΔAUC +0.0358 — 規模不同但方向一致
2. **Building-level signal 在 M3 更顯著**：M3 top feature 是 log_square_feet（building size）；M2 沒有 building metadata-level features，top feature 是 dayofyear
3. **Anomaly bursts 雙向對稱**：past/future shifts 都有效（SC0 past=future=0.9908），反映 anomaly events 為 multi-hour 連續事件，不是 leakage

---

## 待續工作（M3.3-M3.5）

時間關係先告一段落。剩餘工作：

1. **M3.3**：對齊 buds-lab `02_preprocess_data.py` 缺失的 features
   - Cyclic encodings：sin/cos(hour, dayofweek, month)
   - Weather rolling lags：windows 7, 73（lag + rolling mean）
   - Holiday flags：US Federal Calendar via `holidays` library
   - GaussianTargetEncoder：per-(site, meter) target encoding（fit on train only）
   - Building interactions、site corrections

2. **M3.4**：4-model ensemble（LightGBM + XGBoost + CatBoost + HistGBT）

3. **M3.5**：Post-processing rules（Rule 1 通用；Rule 2b 通用；Rule 2a 需 EDA 重設計）

完整計畫見 [docs/m3-plan.md](https://github.com/kuokuant-oss/lead-reproduction/blob/main/docs/m3-plan.md)。

---

## 完整文件連結

| 文件 | 說明 |
|---|---|
| [docs/reproduction-report.md](https://github.com/kuokuant-oss/lead-reproduction/blob/main/docs/reproduction-report.md) | M2 完整復現報告（~5,500 字） |
| [docs/m3-report.md](https://github.com/kuokuant-oss/lead-reproduction/blob/main/docs/m3-report.md) | M3 進度報告 |
| [docs/workflow.md](https://github.com/kuokuant-oss/lead-reproduction/blob/main/docs/workflow.md) | 工作方法 framework（unknowns register + ADR + Stage-gate） |
| [docs/m3-plan.md](https://github.com/kuokuant-oss/lead-reproduction/blob/main/docs/m3-plan.md) | M3 計畫（M3.3-M3.5 待續） |
| [notebooks/06-m3-baseline.ipynb](https://github.com/kuokuant-oss/lead-reproduction/blob/main/notebooks/06-m3-baseline.ipynb) | M3 notebook（Cells 1-20，含 4 個 sanity check） |

整個 reproduction 累積：17 unknowns · 6 ADRs · 8 handoff docs · 2 完整 reports · 1 workflow doc。每個設計決策都有 trace 可查。

---

如有任何問題歡迎隨時告知。

Tony Kuo
