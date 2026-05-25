# Notebooks Map: buds-lab/LEAD-1st-solution

Source: https://github.com/buds-lab/LEAD-1st-solution
Cloned to: `.scratch/lead-1st-solution/` (gitignored)
Purpose: 為 Issues #4、#5、#6 提供索引,不需要每次重讀原始碼

---

## Repo 結構

```
lead-1st-solution/
├── README.md
├── Trimming outliers using trees (paper).pdf
├── Trimming outliers using trees (slides).pdf
└── notebooks/
    ├── Feature generator (training data).ipynb   ← 17 cells
    ├── Feature generator (test data).ipynb       ← 17 cells (幾乎與 train 版相同)
    └── Modeling and submission.ipynb              ← 16 cells
```

無 `src/`、無 `.py` 獨立檔案,整個 solution 全在這 3 個 notebooks 內。

---

## Notebook 說明

### `Feature generator (training data).ipynb` ★ Issue #4 主要來源

**做什麼**: 從競賽提供的 `train_features.csv`(57 欄)出發,加入 K-means cluster 標籤、填補 NaN、計算 value-change 差值與比值特徵、加入 Savitzky-Golay 殘差,輸出 `train_features.pickle`。

| Cell | 內容 |
|------|------|
| 0 | `!pip install kneed` |
| 1 | imports(numpy, pandas, sklearn, imblearn, lightgbm, xgboost 等) |
| 2–3 | 定義 MinMaxScaler / KMeans clustering 輔助函式 |
| 4–5 | 讀 train/test_features.csv;cloud_coverage 255→10 修正 |
| 5 | 合併 train+test,pivot to building×time,做 2 輪 ±10σ outlier removal |
| 6 | `log1p` transform(outlier removal 後) |
| 7–8 | K-means 分 10 群(`ClusterNo`),合入 train_features |
| 9 | merge ClusterNo 進 train_features |
| 10 | `impute_nulls()`:以**每棟建築的 mean** 填補 NaN |
| 11 | value-change **差值** features(60 個 shift) |
| 12 | value-change **比值** features(60 個 shift) |
| 13–14 | Savitzky-Golay 殘差(`Residual_savgol_w5p3`) |
| 15 | `train_features.to_pickle('train_features.pickle')` |

**⚠️ 注意**: README 寫 "median" 填補,但程式碼 Cell 10 用 **mean**。論文(§2.1)說 mean。**以程式碼為準。**

---

### `Feature generator (test data).ipynb` ★ Issue #4 參考(與 train 版幾乎相同)

**做什麼**: 與 training 版流程相同,輸出 `test_features.pickle`。結構 cell-by-cell 幾乎一致,差異在讀不同的 CSV 且不含 `anomaly` 欄。可省略單獨閱讀。

---

### `Modeling and submission.ipynb` ★ Issue #5 主要來源 / Issue #6 主要來源

**做什麼**: 讀取 pickle,下採樣、切分 train/val、正規化、訓練 4 個模型、ensemble、後處理、輸出 submission CSV。

| Cell | 內容 |
|------|------|
| 0 | imports |
| 1 | 讀 `train_features.pickle` / `test_features.pickle` |
| 2 | 新增 `dayofyear`(float = day + hour/24) |
| 3 | **Downsampling**:各取 2 份 normal(seed 10 / seed 20) + 2 份 anomaly 合併 |
| 4 | `list_variables`:drop anomaly/wind_direction/air_temperature_std_lag73,select float+int |
| 5 | 切出 features / target |
| 6 | **CV split**: `building_id % 5 < 4` → train;`building_id % 5 == 4` → val |
| 7 | `StandardScaler` 正規化(train fit,val/test transform) |
| 8 | XGBoost:`n_estimators=100`,其他全預設 |
| 9 | HistGradientBoosting:全預設 |
| 10 | CatBoost:全預設 |
| 11 | LightGBM:`n_estimators=100`,其他全預設 |
| 12 | Ensemble:(xgb+cat+lgb+hist)/4 |
| 13 | 用全部 train data(X_all)重新 fit 四個模型後 predict test |
| 14 | **Post-processing** + 輸出 CSV |

---

## README 重點摘要

- 競賽連結: https://www.kaggle.com/competitions/energy-anomaly-detection
- Imputation:**median**(README 說;但程式碼是 mean)
- Value-change:「varying shift steps from 1 hour to 168 hours」
- CV:split by building_id
- Downsampling:~5% anomalies → 平衡
- Ensemble:XGBoost + LightGBM + CatBoost + HistGB,weight 0.25 各

---

## 候選清單(按 issue 分類)

### Issue #4 — Feature engineering
- **主要**: `Feature generator (training data).ipynb` Cells 11–12(value-change shifts)
- **次要**: Cell 9(ClusterNo)、Cells 13–14(Savitzky-Golay)
- **模型端**: `Modeling and submission.ipynb` Cell 4(`list_variables` 決定最終特徵集)、Cell 2(dayofyear)

### Issue #5 — CV / Downsampling / Target encoding
- **主要**: `Modeling and submission.ipynb` Cells 3(downsampling)、6(CV split)
- **Target encoding**:在 `train_features.csv` 裡已預計算(gte_* 欄位),不在 notebook 內動態計算

### Issue #6 — Model hyperparameters + Post-processing
- **主要**: `Modeling and submission.ipynb` Cells 8–11(超參數)、Cell 14(post-processing)
- Post-processing Cell 14 的實際條件比論文描述**更具體**(見下方發現)

---

## 階段 A 的重要發現(給 Issue #4 / #5 / #6 預告)

以下是閱讀原始碼結構時看到的關鍵差異,**尚未深入驗證**,留給各 issue 深讀。

| 主題 | 論文/paper-notes 說 | 原始碼看起來 |
|------|---------------------|-------------|
| Value-change shift 數 | 8 個 shift 值 × 2 方向 = 16 shifts → 32 features | 60 個 shifts → **120 value-change features** |
| 總特徵數 | 169 | 初估 ~175(加上 ClusterNo、SavGol、dayofyear 後,需 Cell 4 實際執行確認) |
| CV split | 未說明 fold 數/seed | `building_id % 5`:確定性切分(無 seed),約 160 train / 40 val 棟 |
| Downsampling | 1 次全域 50:50 | 2 次 normal 取樣(seed 10 + seed 20)+ 2 份 anomaly → **資料量是 pos 的 4 倍總** |
| Post-processing Rule 2 | "start and end points" | 具體條件:`dayofyear==1` 特定建築 + `dayofyear>366.9583`(最後幾小時) |
| Imputation | mean(論文)/ median(README) | 程式碼:**mean per building**(論文正確,README 有誤) |
| 超參數 | 論文未揭露 | 看起來接近預設值(n_estimators=100),但需 Cell 8–11 深讀確認 |

---

Last reviewed: 2026-05-25
