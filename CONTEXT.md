# Project Context

## What this project is
重現一篇論文(LEAD competition 第一名解法),並擴展到完整 ASHRAE 資料集。
論文連結:https://dl.acm.org/doi/abs/10.1145/3563357.3566147
原始 solution:https://github.com/buds-lab/LEAD-1st-solution

## Milestones
- M1:理解論文與方法
- M2:重現比賽結果(LEAD 資料集,~200 meters)
- M3:擴展到完整 ASHRAE 資料集(~2000 meters)

## 資料來源關係(working hypothesis,M1 階段 B 驗證)

LEAD 比賽資料的組成假設:

```
GEPIII raw data (Kaggle)
    + bad_meter_readings.zip (異常標注)
    → LEAD dataset (train/test CSV + train_features.csv 57 欄)
```

- **LEAD 提供的 `train_features.csv`(57 欄)可能來自** `buds-lab/ashrae-great-energy-predictor-3-solution-analysis` 的 `solutions/rank-1/scripts/02_preprocess_data.py`
- 此假設若成立,影響三件事:
  1. LEAD-1st-solution 的 Feature generator notebook 只負責 value-change features,不負責原始 57 欄的產生
  2. 解 169 vs ~175 特徵數差距時,必須同時參照 `02_preprocess_data.py`
  3. M3 不只是「換更大的資料」,而是「從頭跑完整 feature engineering pipeline(GEPIII raw → 57 欄 → value-change → 模型)」

驗證方式:比對 `02_preprocess_data.py` 的輸出欄位名稱與 `train_features.csv` 的 57 欄是否完全一致(見 `docs/unknowns.md` #7)。

## Glossary

**LEAD**:
Large-scale Energy Anomaly Detection,以 Kaggle 為平台的建築能源異常偵測社群比賽(2022)。資料來自 ASHRAE GEPIII 競賽的標注子集。
_Avoid_: 以 "LEAD" 單獨指涉資料時容易混淆 — 明確使用 "LEAD competition"(比賽本身)或 "LEAD dataset"(競賽使用的資料集)。

**ASHRAE**:
American Society of Heating, Refrigerating and Air-Conditioning Engineers。在本專案中以兩個脈絡出現:(1) ASHRAE 組織發起的 GEPIII 能源預測競賽;(2) GEPIII 產生的完整能源資料集(~1,413 meters),是本專案 M3 的目標資料集。
_Avoid_: 以 "ASHRAE dataset" 不加限定時無法區分 GEPIII 子集還是完整資料集 — 明確使用 "ASHRAE GEPIII dataset" 或 "full ASHRAE dataset"。

**GEPIII**:
Great Energy Predictor III,ASHRAE 在 Kaggle 舉辦的建築能源預測競賽(2020)。LEAD dataset 是 GEPIII 資料的人工標注子集;GEPIII 衍生的特徵(building meta、weather data 等)被直接沿用至 LEAD 解法中。

**Anomaly label**:
每筆 meter reading 的二元標籤:0(正常)或 1(異常)。在 LEAD dataset 中由人工標注,異常率約 5%。

**Meter**:
在本專案中指單一建築的能源電表,資料形式為以小時為單位的 `meter_reading` 時間序列(全年約 8,760 筆/meter)。
_Avoid_: "sensor" 或 "device" — meter 專指已彙總成每小時讀數的電表資料,不是原始感測器訊號。

**Point anomaly**:
時間序列中一個孤立的異常讀數,與鄰近點或整體時間序列相比明顯偏離。特徵:隨機、偶發、不連續。對應特徵:value-change 在相鄰 timestep 數值急遽改變。

**Sequential anomaly**:
連續多個 timestep 的異常讀數,代表某個持續性的異常事件(如設備故障導致讀數長時間 flatline 或持續偏低)。又稱 collective anomaly。
_Avoid_: "flatline anomaly" — flatline 是 sequential anomaly 的一個常見子類型,不是同義詞。

**Feature engineering**:
從原始 meter readings 及 metadata 衍生出模型特徵的過程。在本論文中包含六類:energy use、building meta、weather data、temporal、target encoding、value-change。

**Value-change feature**:
本論文的核心 feature innovation。基於 `meter_reading` 與 `t-s` 時刻讀數的差值 `X(t) - X(t-s)` 或比值 `(X(t)+1)/(X(t-s)+1)` 衍生的特徵,用以捕捉時間序列的變化幅度。論文 §2.2.2 說明 shift 設計邏輯(sub-day 和 multi-day 兩類),但舉例描述不精確;以代碼為 ground truth:實際 60 shifts × 2 types = 120 value-change features。用以偵測 point anomaly(急遽變化)和 sequential anomaly(零變化 / flatline)。
_Avoid_: "value-changing feature"(語法不正確)。

**Target encoding**:
將類別欄位(如 `building_id`)替換成對應的 anomaly label 平均值的技術。在本專案中若未在 CV fold 內部計算,會產生 data leakage,導致 validation score 虛高。

## Working principles

**三方不一致原則**: 當論文、README、程式碼描述不一致時,以程式碼為準(例:ADR 0005,imputation method)。發現任何不一致都記 ADR,不省略。

**Setup 完整性原則**: 使用者初始描述提供的 reference 連結(論文、程式碼、資料集)必須完整保留在 README + CONTEXT,即使當下覺得「之後再說」。setup 階段漏掉的 reference 會在後期反咬(例:commit 7e7bfa9 補入 M3 資源——`02_preprocess_data.py` 與 `bad_meter_readings.zip` 在 M1 階段 B 就需要)。

## Tech stack
- Python 3.11+
- uv(package manager)
- pandas, numpy, scikit-learn, lightgbm
- jupyter notebooks for exploration
- src/lead/ for reusable code

## Folder conventions
- \`data/raw/\`:從 Kaggle 直接下載的原始檔(不進 Git)
- \`data/interim/\`:中間處理結果
- \`data/processed/\`:餵給模型的最終格式
- \`notebooks/\`:探索用,檔名前綴編號(01-, 02-, ...)
- \`src/lead/\`:穩定的可重用程式碼
- \`docs/\`:筆記、決策紀錄
- \`reports/\`:圖表、結果輸出

---

Last reviewed: pending
