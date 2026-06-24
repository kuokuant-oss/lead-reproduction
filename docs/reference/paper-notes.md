# Paper Notes: Trimming outliers using trees

**論文**: "Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition"
**作者**: Chun Fu, Pandarasamy Arjunan, Clayton Miller (NUS / BEARS-Berkeley)
**發表**: BuildSys '22, November 9–10, 2022, Boston, MA, USA
**DOI**: https://doi.org/10.1145/3563357.3566147
**原始碼**: https://github.com/buds-lab/LEAD-1st-solution

---

## Motivation

建築能源資料品質不良是預測模型精度下降的主因(§1.1):
- 設備故障、不當操作估計浪費商業建築 **15–30% 能源**
- 異常資料若未被識別,會產生偏差預測、污染後續能源管理模型
- HVAC 系統有豐富感測器訊號(溫度、風量、頻率等),可做 supervised anomaly detection;但**能源電表資料只有 meter reading**,缺乏輔助訊號,標注困難

LEAD 競賽的成立動機(§1.2):
1. 提供有標注的能源異常 benchmark 資料集
2. 透過 Kaggle 向多元背景參賽者徵集解法
3. 建立 supervised learning 在能源異常偵測的性能基準

---

## Problem

**任務**: 對每筆 hourly meter reading 做二元分類 — 正常(0)或異常(1)。

**資料規模**(§1.2):
- 訓練集: 200 棟建築,全年逐小時讀數 + anomaly label
- 測試集: 206 棟建築(未見過),輸出每筆讀數的異常機率
- 評估指標: **AUC-ROC**
- 類別不平衡: 異常率約 **5%**

**兩種異常類型**(§1.2):
- **Point anomaly**: 孤立的單一異常點,與鄰近點或整體時間序列相比明顯偏離。特徵:隨機、偶發、不連續。
- **Sequential anomaly** (collective anomaly): 連續多個異常點的集合,代表持續性異常事件(例如設備故障造成 flatline)。

**資料集來源**: ASHRAE Great Energy Predictor III (GEPIII) 資料集的標注子集。完整資料集有 1,413 meters;LEAD 競賽只使用其中 406 meters(200 訓練 + 206 測試),佔整體 ~29%。

---

## Method

Pipeline 七個階段(Figure 1, §2):

```
Training data
    ↓
Preprocessing (missing value imputation + feature normalization)
    ↓
Feature engineering (57 original → 169 total with value-change)
    ↓
Data downsampling (5% anomaly → 50:50 balance)
    ↓
Models × 4 (LightGBM, XGBoost, CatBoost, HistGradientBoosting)
    ↓
Weighted average ensemble (equal weights, 0.25 each)
    ↓
Post-processing (hard rules) → Final submission
```

### 1. Preprocessing (§2.1)

| 步驟 | 做法 | 備註 |
|------|------|------|
| Missing data imputation | 以每條時間序列的 **mean** 填補 NaN | NaN 佔全資料 6.2%;試過 forward fill / backward fill,效果均不如 mean |
| Feature normalization | 論文提及但未說明方法 | 對 LightGBM、XGBoost、CatBoost、HistGB 而言 scale 不影響 tree splits(單調變換不變性),此步驟對結果應無實際影響;**論文未說明,見原始碼** |

**注意**: 異常資料**不做清洗** — 因為競賽目標就是偵測這些異常,清洗會消除標注資訊。

### 2. Feature Engineering (§2.2)

競賽提供的原始特徵最多 57 個(含 GEPIII 衍生特徵),加上本論文新增的 value-change features,**最終使用 169 個特徵**(Table 3)。

⚠️ **Feature count 不符**: 57 原始 + 論文描述的 16 差值 + 16 比值 = 89,與 169 差距 80 個。來源不明。**論文未說明,見原始碼 buds-lab/LEAD-1st-solution。** 詳見 `docs/reference/unknowns.md`。

**六類特徵**(Table 1, §2.2):

| 類別 | 欄位 |
|------|------|
| Energy use | `meter_reading`(原始每小時電表讀值) |
| Building meta | `site_id`, `building_id`, `primary_use`, `square_feet`, `year_built`, `floor_count` |
| Weather data | `air_temperature`, `cloud_coverage`, `dew_temperature`, `precip_depth_1_hr`, `sea_level_pressure`, `wind_direction`, `wind_speed` |
| Temporal | `hour`, `weekday`, `day of year` |
| Target encoding | 依類別欄位(如 `building_id`)計算的 anomaly label 平均值;⚠️ 若未在 CV fold 內計算會產生 data leakage(論文未說明防護方式,見 unknowns) |
| Value-change (新增) | 差值與比值 — 詳見下方 |

#### Value-change features — 核心創新 (§2.2.1–2.2.3)

**差值 (Difference)**:

$$\text{diff}(t, s) = X(t) - X(t - s)$$

**比值 (Ratio)**:

$$\text{ratio}(t, s) = \frac{X(t) + 1}{X(t - s) + 1}$$

分子分母各加 1:防止 `meter_reading = 0` 時除以零。

**選用的 shift 值** (§2.2.2):
- 日內(完整枚舉): $s \in \{1, 2, 3, 23\}$
- 跨日(24h 間隔): $s \in \{24, 48, 72, 168\}$
- 競賽不需預測未來,**正負 shift 都納入**

| Shift | 捕捉的異常模式 |
|-------|--------------|
| ±1, ±2, ±3 | 相鄰小時的急遽變化 → point anomaly |
| ±23 | 跨越午夜的同時段對比 |
| ±24, ±48, ±72 | 日週期性偏離 |
| ±168 | 週週期性偏離(1 week = 168 hours) |

差值為零可偵測 flatline(sequential anomaly);比值相較差值對不同量級的時間序列更具一致性。

**兩者都保留的依據** (§2.2.1): 「差值和比值雖然相似,但實驗顯示同時包含兩種特徵能達到最佳預測性能。」— 對重現者的含義:勿貿然省略其中一種以節省特徵數,會影響結果。

嘗試過但效果不佳、未採用(§2.2.4):
- Savitzky-Golay filter 的平滑差值
- 時間序列的 K-means 分群

### 3. Data Splitting (§2.3.1)

- **Split by `building_id`**: 同一棟建築的所有讀數只進 train 或 validation,絕不跨集
- 效果: validation AUC 與 leaderboard 分數差距 **< 1%**,可靠地反映線上性能
- **論文未說明**: fold 數量、validation set 包含幾棟建築、random seed。**見原始碼。**

若用 random shuffle split:validation score 會高估(同建築的相鄰時間點互相洩漏),無法可靠指引 feature engineering 迭代。

### 4. Data Downsampling (§2.3.2)

- 原始異常率 ~5% → 嚴重 class imbalance,不處理則模型偏向預測 0
- 對**正常資料做 downsampling**,使 normal:abnormal = **50:50**
- 採用 downsampling(減少多數類),未使用 SMOTE 或其他 oversampling
- **論文未說明**: 是否按建築做、是否每個 fold 重新抽樣、random seed。**見原始碼。**

競賽對照(Table 3):未處理 imbalance 的解法私榜分數均低於 0.90。

### 5. Models (§2.3.3)

四個 GBDT 模型**各自獨立訓練**,超參數各別調整:

| 模型 | Train AUC | Test AUC | 備註 |
|------|-----------|----------|------|
| LightGBM | 0.9975 | 0.9849 | Feature engineering 實驗迭代主要用此模型(速度最快) |
| XGBoost | 0.9999 | 0.9840 | Train AUC 接近 1.0,有 overfitting 跡象 |
| CatBoost | 0.9999 | 0.9857 | 同上 |
| HistGradientBoosting | 0.9968 | 0.9839 | sklearn 內建實作 |

**超參數**: 論文**完全未揭露**。**見原始碼 buds-lab/LEAD-1st-solution。**

### 6. Ensemble (§2.3.4)

$$\text{prediction} = (\text{LightGBM} + \text{XGBoost} + \text{CatBoost} + \text{HistGB}) \times 0.25$$

- 等權平均,**未做 weight 優化**
- Ensemble AUC: **0.9866** vs 各模型平均 ~0.9846,提升 **+0.21%**

### 7. Post-processing (§2.4)

兩條 hard rule,依序套用於 ensemble 機率輸出之後:

**Rule 1**: `meter_reading == 1` → 強制預測為 **異常(1)**
來源:競賽討論區觀察,幾乎 100% 的 `meter_reading = 1` 都是異常(資料特有 artifact)。

**Rule 2**: 時間序列的 start/end points → 強制預測為 **正常(0)**
來源:視覺化觀察,各電表時間序列的起始和結束點幾乎都不是異常。
⚠️ **論文未定義 "start/end points" 的邊界**:首尾各幾筆?**見原始碼。**

---

## Key Results

**最終成績**(§3):
- Private leaderboard AUC-ROC: **0.9866**(遠超「優秀分類器」門檻 0.90)
- Precision: **98.7%** (預測為異常中有 98.7% 真的是異常)
- Recall: **81.9%** (所有真實異常中有 81.9% 被偵測到)
- F1 score: **~0.89**

**各技術的 AUC 貢獻**:
| 里程碑 | AUC |
|--------|-----|
| 加入 value-change features 前 | 0.9311 |
| 加入後 | 0.9849 |
| Ensemble 後 (final) | 0.9866 |

Feature engineering 貢獻 +5.8%;ensemble 再加 +0.21%。Feature engineering 是最大單一貢獻。

**Top 10 features** (Figure 5,從 LightGBM 的 feature importance 輸出):

| Rank | Feature | Category | Importance |
|------|---------|----------|-----------|
| 1 | `building_id` | Building meta | 219 |
| 2 | `value_chg_ratio_1` | Value-change (ratio, s=+1) | 144 |
| 3 | `value_chg_ratio_-1` | Value-change (ratio, s=-1) | 127 |
| 4 | `meter_reading` | Energy use | 98 |
| 5 | `dayofyear` | Temporal | 95 |
| 6 | `square_feet` | Building meta | 85 |
| 7 | `gte_building_id` | Target encoding | 63 |
| 8 | `value_chg_ratio_-168` | Value-change (ratio, s=-168) | 61 |
| 9 | `value_chg_ratio_2` | Value-change (ratio, s=+2) | 54 |
| 10 | `gte_meter_primary_use` | Target encoding | 54 |

**觀察**: Top 10 中的 4 個 value-change features 全是 **ratio 類型**(差值特徵未入選)。但論文仍建議保留差值 — 整體 AUC 在兩者並存時最高。Weekly ratio (s=±168) 的高 importance 支持了「能源資料有強烈週週期性」的假設。

**與其他解法的比較**(§3.3, Table 3):

| 解法 | Private AUC | 特徵數 | 關鍵差異 |
|------|------------|--------|---------|
| **本論文 (第一名)** | **0.9866** | **169** | value-change + downsampling + 4-model ensemble |
| Abhishek Maurya | 0.9237 | 31 | downsampling 但無 value-change |
| Abdallah El-Sawy | 0.8189 | 10 | 無 downsampling |
| FabioDalForno | 0.7566 | 6 | 有 value-change 但特徵極少 |
| Yoda | 0.7433 | 33 | 無 downsampling |

**結論**: 未處理 class imbalance 的解法全部低於 0.90;value-change features 是突破 0.93+ 的關鍵;ensemble 不是性能的主要來源(僅 +0.21%),但有穩定作用。

---

## Open Questions

*以下問題是重現時最可能踩坑的地方,詳細討論見 `docs/reference/unknowns.md`。*

1. **Feature count 169 的完整組成**:論文描述的 shifts 只能算出 89 個特徵,缺少的 ~80 個來源不明
2. **Validation split 實作細節**:fold 數、validation 建築數、seed
3. **Post-processing "start/end" 的邊界定義**:首尾各幾筆?
4. **Downsampling 的作用域與 seed**:全域或按建築?每 fold 重抽?
5. **Target encoding 的 leakage 防護方式**:論文未說明 CV 內計算的細節
6. **四個模型的超參數**:論文完全未揭露

---

Last reviewed: pending
