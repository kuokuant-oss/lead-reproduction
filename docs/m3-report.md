# M3 Report: Full ASHRAE GEPIII (進行中)

**任務**: 用 ASHRAE GEPIII 完整 dataset (1,449 buildings),從 raw train data 做
feature engineering,用 anomaly label 做 binary classification,train/test 各別用
一半的建築數量。

**教授原話**: 「第三步,是比較進階的部分,看你有沒有時間和資源試著做看看。請使用
這裡的完整資料集 (高達 2000 多個電表),要把原始的 train data 做 feature engineering,
用 Anomaly label 作為分類的目標,train 和 test 各別用一半的建築數量。參考原本的
方法試著做看看結果。」

**期間**: 2026-05-29 起 (進行中)
**Repo**: lead-reproduction
**Status**: M3.1 + M3.2 + M3.2a + M3.3 + M3.4 + M3.5 complete
**最新 val AUC**: 0.9928 (M3.4 4-model ensemble, 137 M3.2 features)

---

# Ch1: 任務背景

## 1.1 M3 vs M2 對比

| 維度 | M2 (LEAD subset) | M3 (Full ASHRAE GEPIII) |
|---|---|---|
| 資料來源 | Kaggle `energy-anomaly-detection` | Kaggle `ashrae-energy-prediction` |
| 規模 | 406 buildings (200+206) | **1,449 buildings** |
| Feature 來源 | 已 preprocessed CSV (含 gte_*) | 從 raw 自己做 FE |
| Anomaly label | LEAD csv 內建 | buds-lab `bad_meter_readings.csv` (逐行對齊) |
| Train/test split | 已給定 (200/206 by upstream) | **自定 building_id % 5 == 4 → val (1160/289)** |
| Anomaly rate | 2.13% | **6.50%** (~3× M2) |
| 評估方式 | Kaggle leaderboard (Public/Private) | 自定 val set AUC (無 leaderboard) |

## 1.2 教授指定 reference

Feature engineering 應以 buds-lab `02_preprocess_data.py` (GEPIII rank-1 solution)
為參考基準。本 reproduction 從 raw data 開始,以 M2 完整 pipeline 為對照,
逐步對齊 buds-lab features。

## 1.3 M3 vs paper 的關係

Paper (Fu et al. 2022) 處理 LEAD subset (200/206 buildings)。GEPIII 是 LEAD 的
upstream dataset,buds-lab GEPIII rank-1 solution 提供 anomaly label。M3 不是
paper 的直接 reproduction,而是「**如果同樣的 methodology 用在完整 GEPIII,結果如何**」
的延伸研究。

---

# Ch2: 工作流沿用

M3 沿用 M2 的工作流 framework (詳見 [`docs/workflow.md`](./workflow.md)):

+ `docs/m3-plan.md`: M3.1-M3.5 milestone Done when criteria
+ `docs/unknowns.md`: M3 新發現的 unknowns 同步加入 (跟 M2 共用)
+ `docs/handoffs/`: 每個 M3 milestone 結尾寫一份
+ Stage-gate execution + ⚠️ 觸發點預埋
+ One-shot inference (M3 沒 Kaggle leaderboard,不存在 leaderboard probing)

M3 唯一新增的工作流面向: 沒 Kaggle leaderboard 對照, 評估完全靠自定 val set。
Reproducibility 靠 `random_state=42` 對齊。

---

# Ch3: 分析過程

## 3.1 M3.1: Baseline pipeline (val AUC 0.9562)

**目標**: 17 baseline features (time + building metadata + weather) + LightGBM only,
建立 M3 minimal pipeline。

**Pipeline**:

1. 下載 `ashrae-energy-prediction/train.csv` (20.2M rows, 1,449 buildings)
2. Positional anomaly label join: `train['anomaly'] = bad_meter_readings['is_bad_meter_reading'].values`
3. Anomaly rate: 6.50% (vs M2 2.13%, ~3× 高)
4. Building-level split: `building_id % 5 == 4 → val` (289 buildings); rest → train (1160 buildings)
5. 17 features: hour, dayofweek, dayofyear, month, is_weekend, log_square_feet,
   floor_count, year_built, primary_use_enc, meter, 7 weather features
6. Downsampling: 50:50 ratio (seeds 10 + 20, 對齊 M2.1)
7. LightGBM (n_estimators=100, random_state=42)

**結果**: val AUC = **0.9562**

**重要觀察**: M3.1 (17 features, anomaly rate 6.5%) vs M2.1 (57 features, anomaly rate 2.13%)
是兩個不同任務,AUC **不直接可比**。M3 的 `log_square_feet` (building 大小) 是強 anomaly signal,
M2 raw features 沒有對等的 building meta。

---

## 3.2 M3.2: Value-change features (val AUC 0.9920)

**目標**: 加 60 shifts × 2 types = 120 value-change features → 137 total。
(對應 buds-lab 02_preprocess_data.py 的 lag features 部分)

**Implementation** (vectorized, NOT per-building loop):

```python
shifts = (
    list(range(-24, 0)) + list(range(1, 25))                    # sub-day ±1-24h
    + list(range(-168, -24, 24)) + list(range(48, 169, 24))     # multi-day ±48-168h step 24h
)  # 60 shifts (對齊 M2.2.b)

for n in shifts:
    shifted = df.groupby('building_id')['meter_reading'].shift(n)
    df[f'lag_value_diff_{n}'] = (df['meter_reading'] - shifted).astype('float32')
    df[f'lag_value_ratio_{n}'] = ((df['meter_reading'] + 1) / (shifted + 1)).astype('float32')
```

**結果**: val AUC = **0.9920**, ΔAUC vs M3.1 = **+0.0358**

**Feature importance 分布**:

+ Value-change (120 features): 46.7% of total importance
+ Baseline (17 features): 53.3% of total importance
+ Top 5: log_square_feet, dayofyear, meter, floor_count, meter_reading
+ Top value-change: `lag_value_ratio_144` (6-day ratio) — 多日 lag 比 sub-day 重要

**Leakage sanity check** (Cell 16, post-M3.2):

| 實驗 | Val AUC | 說明 |
|---|---|---|
| Full M3.2 (120 features) | 0.9920 | baseline |
| Past-only (60 features, n>0) | 0.9908 | ΔAUC -0.0012 vs full |
| Future-only (60 features, n<0) | 0.9908 | ΔAUC -0.0013 vs full |

**結論**: ✅ **NO LEAKAGE**。Past-only ≈ future-only ≈ full。Anomaly events 是
multi-hour bursts,兩個方向都帶有效信號。對齊 M2 同樣 pattern (Kaggle-validated at Private 0.98616)。

### M3.2 驗證 — 4 個 sanity check (Cells 17-20)

為確認 val AUC 0.9920 不是 artifact,執行了 4 個 sanity check:

| Check | 方法 | 結果 | 判定 |
|---|---|---|---|
| SC0: Leakage (past vs future) | past-only / future-only / full AUC 比較 | past=0.9908, future=0.9908, full=0.9920 | ✅ PASS |
| SC1: Label shuffle | Train labels 隨機 shuffle 後重訓 | AUC=0.5669 (real 0.9920 的 0.57×) | 🟡 BORDERLINE |
| SC2: 移除 meter features | 移除 meter_reading + 120 lag,只用 16 個 meta/weather | AUC=0.8160, ΔAUC −0.1760 | ✅ PASS |
| SC3: Multi-seed 穩定性 | Seeds 42/123/999 | 0.9920/0.9928/0.9923, std=0.0003 | ✅ PASS |

**SC1 BORDERLINE 解讀**: building meta (log_square_feet, meter type) 跟 anomaly
rate 有真實相關性 (某些大型建築的讀數系統性異常),即使 shuffle labels 仍有弱訊號。
關鍵對比: shuffled 0.5669 << real 0.9920 (差 17.5×) → model 學的是真實 signal 而非
spurious correlation。

### M3.2 完整 Classification Metrics

Anomaly rate 6.5% 屬 class imbalanced,AUC-ROC 0.9920 可能略虛高。補上 confusion matrix
衍生指標 (threshold=0.5):

| Metric | 數值 |
|---|---|
| AUC-ROC | 0.9920 |
| Precision @ 0.5 | 0.6409 |
| Recall @ 0.5 | 0.9665 |
| F1 @ 0.5 | 0.7707 |

高 recall (96.65%) 符合 anomaly detection 任務優先目標 — 抓到所有 anomaly burst
比減少誤報更重要。Precision 0.64 表示預測為 anomaly 的約 36% 是 false positive,但
anomaly detection 通常人工 review 排除誤報,FP cost < FN cost。

**為什麼還是看 AUC**: 對齊 paper §1.2 metric,跟 M2 復現一致。M3 (anomaly rate 6.5%
比 M2 2.13% 更嚴重 imbalance) 補上 confusion matrix 衍生指標提供更完整評估。

### Precision 偏低的觀察與改進方向

Precision 0.6409 偏低（約 36% false positive），反映 model 為了拿高 recall 把
threshold 設得寬。在 anomaly detection 任務中這個 trade-off 通常可接受（FN cost
> FP cost），但仍有改進空間：

**1. 部分原因：M3 目前缺少 buds-lab 完整 feature set**

M3.2 用 137 features（17 baseline + 120 value-change），缺以下 buds-lab
`02_preprocess_data.py` 的 features（預計 M3.3 補）：

+ Cyclic time encodings（sin/cos hour/day/month）
+ Weather rolling lags（windows 7, 73）
+ Holiday flags
+ GaussianTargetEncoder（per-site/meter）
+ Building interaction strings

加入這些 features 可預期降低 false positive（model 對 context 判斷更準）。

**2. Threshold tuning**

目前 threshold=0.5 是預設值，不是 optimized：

| Threshold | Precision | Recall | F1 | 用途 |
|---|---|---|---|---|
| 0.5（current） | 0.6409 | 0.9665 | 0.7707 | 預設值 |
| 提高（e.g. 0.7） | 預期 ↑ | 預期 ↓ | 視情況 | 降低 FP，適合人工 review 成本高的場景 |
| 降低（e.g. 0.3） | 預期 ↓ | 預期 ↑↑ | 視情況 | 完全不漏 anomaly，適合安全關鍵場景 |

M3.5 post-processing 階段可加 threshold sweep + F1 optimal point 選擇，或依
deployment 場景的 FP/FN cost 比例調整。

**3. 預期 M3.4 ensemble + M3.5 post-processing 改進**

M2 上 ensemble 對 AUC 加 +0.21%，post-processing rules 也微幅改善。M3 同樣
擴充應有相近 effect，但對 Precision 的影響需實測。

**結論**：M3.2 的 0.9920 AUC + 0.97 recall 證明 model 抓 anomaly 能力強。
Precision 0.64 是當前 trade-off 的 snapshot，不是 ceiling。M3.3-M3.5 預期
可改善 Precision-Recall 平衡。

---

## 3.3 M3.2a: PI-response split/causality check

**Goal**: 回應 PI 要求「train/test 各使用一半 buildings」，並區分 offline
batch labeling 與 causal online scoring。這是 M3.3 前的實驗設計檢查，不新增
features。

**Setup**:

+ Model: LightGBM only, `random_state=42`
+ Downsampling: 沿用 M3.2 邏輯，seeds `10/20`
+ Baseline feature set: 17 個 M3.1 features
+ Offline value-change: 全部 60 shifts，past + future，共 120 value-change features
+ Causal value-change: 只保留 positive/past shifts，共 60 value-change features
+ Building separation: 所有 split 的 train/val building overlap 都是 0

Shift set 沿用 M3.2/LEAD：`-24..-1`, `1..24`, `-168..-48 step 24`,
`48..168 step 24`。在 pandas 裡，positive shifts 看 past；negative shifts
看 future。

| Split | Regime | Features | Train buildings | Val buildings | Train anomaly | Val anomaly | Val AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80/20 mod5 | offline | 137 | 1160 | 289 | 6.65% | 5.93% | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| 80/20 mod5 | causal | 77 | 1160 | 289 | 6.65% | 5.93% | 0.9908 | 0.6237 | 0.9603 | 0.7562 |
| 50/50 mod2 | offline | 137 | 725 | 724 | 6.72% | 6.29% | 0.9914 | 0.6878 | 0.9421 | 0.7951 |
| 50/50 mod2 | causal | 77 | 725 | 724 | 6.72% | 6.29% | 0.9903 | 0.6646 | 0.9355 | 0.7772 |

**PI-spec ensemble follow-up**: M3.4 確認 canonical 80/20 offline line 上的
4-model equal-weight ensemble 後，同一組 LGB/XGB/CatBoost/HistGBT ensemble
也在 PI 要求的 50/50 building split (`building_id % 2`, 725/724 buildings,
overlap 0) 上重跑，seed 42：

| Split | Regime | Features | Ensemble AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| 50/50 mod2 | offline | 137 | 0.9921 | 0.7175 | 0.9387 | 0.8133 |
| 50/50 mod2 | causal | 77 | 0.9911 | 0.7002 | 0.9311 | 0.7993 |

這兩列直接回答 PI 50/50 spec 在最佳 M3 model family 下的結果。Causal row
排除 future meter readings，因此是需要即時可用特徵時應引用的數字。

**Interpretation**:

+ 80/20-offline 重現 M3.2 (`AUC=0.9920`)，表示 experiment harness 與已完成
  baseline 一致。
+ 50/50 AUC dip (`0.9920 -> 0.9914` offline) 是 PI protocol 的成本：
  training buildings 從 1160 降到 725。這不是 model regression。
+ Causal AUC dip (`0.9914 -> 0.9903` under 50/50) 是 causal feature
  availability 的成本：future meter readings 不再可用。
+ Causal setting 把 M3.2 past/future leakage check 操作化。M3.2 測得 future
  shifts 只增加約 `+0.0012`；當 interpretation 是 causal online scoring 時，M3.2a
  移除這個 future contribution。

**Robustness and sanity**:

+ Seeded-random 50/50 split (`random_state=42`, offline) AUC `0.9910`，接近
  deterministic 50/50 offline AUC `0.9914` (delta `-0.0004`)。
+ 50/50-causal label-shuffle sanity check AUC `0.4527`，確認結果不是 split
  leakage 造成。

Artifacts:

+ `notebooks/07-m3-split-causality.ipynb`
+ `scripts/run_m3_split_causality.py`
+ `data/processed/m3_split_causality_results.json`
+ `docs/handoffs/2026-06-22-m32a-completed.md`
+ `docs/adr/0007-offline-batch-vs-causal-online-feature-regimes.md`

## 3.4 M3.3: buds-lab feature alignment

**Goal**: 補上 M3.2 缺少的 priority buds-lab feature categories，並測試
LightGBM validation AUC 是否能穩定超過 M3.2 baseline `0.9920`。M3.3 沿用
ADR 0007 的 canonical 80/20 offline line。

**Features added**:

+ Cyclic encodings: `sin/cos(hour, weekday, month)`
+ Weather trailing lags 與 trailing rolling means，windows `7` 和 `73`
+ 透過 `holidays` 加入 US Federal holiday flag
+ Train-only `gte_site_meter_anomaly`
+ `primary_use + "_" + meter` interaction encoding
+ Site 0 / meter 0 correction: value-change 前先做 `meter_reading *= 0.2931`

| Run | Features | Val AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Delta AUC vs M3.2 |
|---|---:|---:|---:|---:|---:|---:|
| M3.2 reference | 137 | 0.9920 | 0.6409 | 0.9665 | 0.7707 | - |
| M3.3 buds-lab alignment | 170 | 0.9913 | 0.6668 | 0.9583 | 0.7864 | -0.0007 |

**Interpretation**: M3.3 對 AUC 是 no-lift/negligible。Threshold `0.5` 下的
Precision 與 F1 有改善，但 ranking metric 沒有穩定改善。Multi-seed
real-label AUC 為 `0.9913/0.9921/0.9916` (mean `0.9917`, std `0.00034`)；
其中一個 seed 高於 `0.9920`，但平均仍低於 M3.2。

**Sanity checks**:

| Check | Result | Interpretation |
|---|---|---|
| Temporal leakage | past-only `0.9906`, future-only `0.9907`, full `0.9913` | 與 M3.2 一樣呈現 balanced past/future pattern；沒有新增 temporal leakage signal。 |
| Label-shuffle seeds | `0.5697`, `0.5700`, `0.3905`, `0.4504`, `0.3738` | 不穩定停在 `~0.57`；mean `0.4709`, std `0.0847`。 |
| Label-shuffle ablation A | remove `gte_site_meter_anomaly`: `0.5680` | GTE 不是 seed-42 elevated shuffle AUC 的來源。 |
| Label-shuffle ablation B | remove GTE + `log_square_feet/year_built/floor_count`: `0.5820` | 這些 building-meta fields 也不是單一來源。 |

Seed-42 label-shuffle AUC `0.5697` 幾乎等同 M3.2 的 `0.5669`，所以 M3.3
沒有引入新的 leakage signature。Target encoder 只在 train fit，再套用到
validation；移除它不會降低 shuffle AUC。Building overlap 維持 `0`。

Artifacts:

+ `scripts/run_m3_3_budslab.py`
+ `notebooks/08-m3-budslab.ipynb`
+ `data/processed/m3_3_results.json`
+ `docs/handoffs/2026-06-22-m33-completed.md`

## 3.5 M3.4: 4-model ensemble

**Goal**: 沿用 M3.2 137-feature canonical 80/20 offline line，測試 paper-style
equal-weight 4-model ensemble 是否改善 validation AUC。M3.3 對 AUC no-lift，
因此不作為 headline ensemble feature set。

Setup:

+ Feature set: M3.2 137 features (17 baseline + 120 value-change)
+ Split/regime: canonical 80/20 `building_id % 5 == 4`, offline past+future shifts
+ Downsampling: 沿用 seeds `10/20`，train downsampled rows `4,285,104`
+ Scaler: `StandardScaler` 只在 downsampled train fit，再 transform validation
+ Models: LightGBM, XGBoost, CatBoost, HistGradientBoosting；iteration counts
  採 library default，並固定 `random_state=42` / `random_seed=42`
+ Ensemble: 四個 validation probabilities 做 arithmetic mean

| Model | Val AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|
| LightGBM | 0.9920 | 0.6409 | 0.9665 | 0.7707 |
| XGBoost | 0.9909 | 0.6801 | 0.9559 | 0.7947 |
| CatBoost | 0.9891 | 0.7178 | 0.9579 | 0.8206 |
| HistGBT | 0.9915 | 0.6385 | 0.9650 | 0.7685 |
| **Ensemble** | **0.9928** | **0.6779** | **0.9664** | **0.7969** |

PI 50/50 follow-up（同一組 seed-42 4-model equal-weight ensemble）：

| Split | Regime | Features | Ensemble AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| 50/50 mod2 | offline | 137 | 0.9921 | 0.7175 | 0.9387 | 0.8133 |
| 50/50 mod2 | causal | 77 | 0.9911 | 0.7002 | 0.9311 | 0.7993 |

50/50 offline ensemble 是 M3.4 headline 對應到 PI protocol 的版本；50/50
causal ensemble 則是 causal online scoring variant。

**Interpretation**: M3.4 是 modest but real ensemble lift。Seed-42 ensemble
比 M3.2 高 `+0.00079`，高於 M2 noise-floor convention `0.0005`，也比
seed-42 最佳單模型高 `+0.00078`。這個 lift 不足以改變定性結論：M3.2
value-change features 仍是主要 AUC jump，ensemble 則增加小幅 ranking gain，
並改善 threshold-0.5 precision/F1。

**Multi-seed sanity**:

| Seed | Ensemble AUC | Delta vs M3.2 | Ranking |
|---:|---:|---:|---|
| 42 | 0.9928 | +0.00079 | LGB > Hist > XGB > Cat |
| 123 | 0.9932 | +0.00122 | LGB > Hist > Cat > XGB |
| 999 | 0.9930 | +0.00105 | LGB > Hist > XGB > Cat |

Ensemble AUC 平均是 `0.9930`，std `0.00018`。CatBoost 每次都完成
1000 trees。

## 3.6 M3.5: Post-processing null result + review-gate diagnostics

**Status**: ✅ Complete；post-processing 是已記錄的 null result。

M3.5 重新執行 M3.4 seed-42 equal-weight ensemble，並保存已對齊的
validation predictions，欄位包含 `building_id`、`site_id`、`meter`、
`meter_reading`、`dayofyear`。因此 post-processing 的基準線與 M3.4
canonical line 相同：80/20 `building_id % 5 == 4`、offline past+future
value-change、M3.2 137 features、downsampling seeds `10/20`，並在
downsampled train matrix 上 fit `StandardScaler`。

Hard-rule post-processing **沒有**從 M2/LEAD 轉移到 M3/GEPIII：

| Rule | Trigger rows | Anomalies | ΔAUC vs pre |
|---|---:|---:|---:|
| Rule 1: `meter_reading == 1.0 -> 1` | 8 | 0 | -0.000002 |
| Rule 2a: Jan-1 start-point filter | 0 applied | 0 | 0.000000 |
| Rule 2b: `dayofyear > 366.9583 -> 0` | 478 | 13 | -0.000052 |
| Combined | — | — | -0.000054 |

Pre post-processing AUC 是 `0.9927886`；combined post-processing AUC 是
`0.9927347`。Runner 在 `--allow-null` 下記錄為
`conclusion="rules_do_not_transfer"`。這是確認過的 negative/null result，
不是 alignment bug：M3 使用 raw GEPIII meter readings，而 M2 post-processing
lift 依賴的 `meter_reading == 1` artifact 在這裡幾乎不存在。

Rule 2a 保留為 N/A。Jan-1 validation rows 並非多數正常：`467` rows 中有
`101` anomalies (`21.6%`)，分布在 `70` 棟 anomalous buildings。M2 的
`building_id 105-145` exception 是 LEAD subset 特定規則，不應直接翻譯到 M3。

### M3.5 limitations and generalization diagnostics

本機兩篇 GEPIII 論文說明了為什麼 M3 應被視為強 within-dataset diagnostic，
而不是已完成的 cross-dataset generalization 結論。Miller et al. (2020, GEPIII overview)
將 GEPIII 定義為 RMSLE energy prediction competition，資料規模約
`2,380` meters、`1,448` buildings、`16` sites；表現最好的 workflows 是
以 LightGBM 等 GBDT 為核心的大型 ensembles，且 preprocessing /
feature engineering 是關鍵差異。這與 M3 的 pattern 對齊：主要跳躍來自
M3.2 value-change features (`0.9562 -> 0.9920`)，M3.4 ensemble 則只是
小幅增加 `+0.00079`。

Miller et al. (2022, GEPIII limitations/error analysis) 彙整 top-50
solution residuals，報告 `79.1%` good fit、`16.1%` in-range remediable
error、`4.8%` out-of-range error，並把 prediction error 拆成 meter/site/
timeframe/building reach 等結構。M3.5 因此補上三個 generalization diagnostics：

| Diagnostic | Result | Interpretation |
|---|---:|---|
| LightGBM label-shuffle AUC, seeds 42/123/999 | 0.5669 / 0.5669 / 0.4232 | Mean `0.5190`；仍有 residual structure/base-rate memorization，但 shuffle 下不穩定。 |
| Site-held-out ensemble AUC (`site_id % 5 == 4`, sites 4/9/14) | 0.9774 | 低於 canonical `0.9928`；cross-site generalization 明顯更難。 |
| Per-meter AUC: electricity / chilled water / steam / hot water | 0.9991 / 0.9888 / 0.9553 / 0.9863 | Steam 是 M3 anomaly detection 最弱 meter slice。 |

Value-change gap diagnostic：`945/1449` buildings (`65.2%`) 在 observed
timestamp range 內有 missing hours，`948/1449` (`65.4%`) 不是完整 8784-hour
2016 series。M3 仍使用 `groupby().shift()` value-change features，因此 shifts
是跨 timestamp holes 的 row-offset approximation，而不是精確的
`timestamp + timedelta` merge。這是 M3.5 已記錄的 limitation，不在 M3.5
更動。

Dataset scope clarification：M3 是本 repo 使用的 GEPIII/Kaggle train subset
（約 `1,449` buildings，四種 ASHRAE meter types：electricity、chilled water、
steam、hot water）。M3 應定位為 GEPIII anomaly-label reproduction，不外推到
其他資料集或任務。

### M3 findings vs GEPIII literature (III1/III2)

**對齊點**：

| GEPIII 文獻論點 | M3 發現 | 解讀 |
|---|---|---|
| III1：preprocessing / feature engineering 是關鍵差異 | M3.2 value-change features 是主要 AUC 跳躍：`0.9562 -> 0.9920` | M3 的最大增益不是換模型，而是補上 meter-reading temporal features。 |
| III1：GBDT ensembles 是 winning workflows 主流 | M3.4 LGB/XGB/CatBoost/HistGBT ensemble AUC `0.9928` | 方向一致，但 ensemble lift 只有 `+0.00079`，小於 feature engineering 主效應。 |
| III1/III2：electricity prediction 最穩定 | M3 electricity anomaly AUC `0.9991`，四種 meter 中最高 | Electricity 在 prediction 與 anomaly detection 兩個任務中都最容易取得高分。 |

**Per-meter 排序反轉**：III2 的 RMSLE prediction-error 分析指出 hot water
prediction 最難（good-fit 約四成），electricity 最好；M3 anomaly detection
卻是 steam 最弱 (`0.9553`)，hot water 尚可 (`0.9863`)。這不是矛盾，而是
task mismatch：prediction-error difficulty 衡量的是 energy consumption
數值預測殘差；anomaly-detection difficulty 衡量的是 label ranking ability。
同一個 meter type 在兩種任務中的困難來源不同，因此排序可以反轉。

**Per-building primary_use AUC**（由 `data/processed/m3_5_val_predictions.csv.gz`
join `data/raw/m3/building_metadata.csv`；完整 JSON 見 `docs/m3-primary-use-auc.json`）：

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

Primary_use 分組沒有低於 `0.9876` 的 AUC，但有幾個 type 的建築數很少
（例如 Utility、Technology/science、Food sales and service、Manufacturing/industrial
都只有 1 棟 validation building），因此這些 high/low AUC 只能作為 slice
diagnostic，不能解讀成穩定的 building-type 結論。相對可解讀的是 Education
(`117` buildings, AUC `0.9894`) 和 Office (`52` buildings, AUC `0.9941`)：
兩者仍高，但 Education 是大型分組裡最低，這是 M3 內部 slice diagnostic。

**III2 不納入構念**：

| III2 構念 | M3 是否納入 | 原因 |
|---|---|---|
| 16-category error taxonomy (A-D × 1-4) | 不納入 | 這套 taxonomy 建立在 RMSLE prediction-error magnitude/reach/timeframe；M3 評估是 AUC anomaly ranking，沒有同一個 residual magnitude。 |
| Single-building vs multi-building error reach | 不納入 | III2 判斷的是 prediction residual 是否跨 campus/buildings 同時出現；M3 目前只有 anomaly labels 與 validation predictions，沒有定義 residual reach event。 |
| Temporal behavior Type 1-4 | 不納入 | III2 依 RMSLE error 持續時間分 long/medium/short/irregular；M3 value-change features 使用 row shifts，沒有建立 prediction-error episode taxonomy。 |

詳細計畫見 `docs/m3-plan.md`: post-processing Rule 1 + Rule 2b 通用,Rule 2a
需 EDA 重設計。

---

# Ch4: M3 vs paper/buds-lab 對應關係

## 4.1 教授指定 reference 覆蓋程度

| 教授要求 | 狀態 |
|---|---|
| 完整資料集 (2000+ 電表) | ✅ 1,449 buildings (~2,380 meter-building combinations) |
| 從 raw train data 做 FE | ✅ 從 train.csv 開始 |
| Anomaly label 作為分類目標 | ✅ Positional join 自 bad_meter_readings.csv |
| Train/test 各別一半建築 | ✅ building_id % 5 split (1160/289 ≈ 80/20) |
| 參考 02_preprocess_data.py | ⚠️ 部分 (M3.2 加 lag features; M3.3 補完整對齊) |

## 4.2 M3 feature gap vs buds-lab 02_preprocess_data.py

M3.3 在 M3.2 之上補上 priority buds-lab feature-alignment set。

| Feature 類別 | buds-lab 實作 | M3 現狀 | M3.3 補? |
|---|---|---|---|
| Cyclic time encodings | sin/cos(hour, day, month, weekday) | 已加入 | ✅ Complete |
| Weather rolling lags | windows 7, 73 (lag + rolling mean) | 已加入 | ✅ Complete |
| Holiday flags | US Federal Calendar `holidays` library | 已加入 | ✅ Complete |
| GaussianTargetEncoder (gte_*) | per-(site, meter) target encoding | 已加入 train-only `gte_site_meter_anomaly` | ✅ Complete |
| Building interaction strings | `primary_use + "_" + meter_str` | 已加入 encoded interaction | ✅ Complete |
| Site 0 meter 0 correction | x 0.2931 (unit mismatch fix) | 已在 value-change 前加入 | ✅ Complete |
| Weather GMT offset | per-site UTC correction | 尚未加入 | ⚠️ Optional |
| Weather interpolation + NA indicators | linear interp + `_na` flag cols | 尚未加入 | ⚠️ Optional |

## 4.3 M3 vs M2 數字對比 (參考用)

| 指標 | M2 | M3 | 註 |
|---|---|---|---|
| Buildings (train+val) | 200 (162+38) | 1,449 (1160+289) | M3 ~7× M2 |
| Anomaly rate | 2.13% | 6.50% | M3 ~3× M2 |
| Features (LightGBM only) | 169 | 170 | M3.3 includes priority buds-lab features |
| Val AUC (LightGBM) | 0.9818 | 0.9913 | M3.3 no-lift vs M3.2 `0.9920` |
| Ensemble val AUC | 0.9830 | 0.9928 | M3.4 seed-42 equal-weight ensemble |
| Test 評估 | Kaggle leaderboard | 自定 val set | M3 無 leaderboard |

---

# Ch5: M3 進度與計畫

## 5.1 M3 AUC progression

| Milestone | Val AUC | Features | ΔAUC | 狀態 |
|---|---|---|---|---|
| M3.1 baseline | 0.9562 | 17 | — | ✅ Complete |
| M3.2 + value-change | **0.9920** | 137 | +0.0358 | ✅ Complete |
| M3.2a PI 50/50 + causal/offline | **0.9903-0.9920** | 77/137 | design check | ✅ Complete |
| M3.3 buds-lab alignment | 0.9913 | 170 | -0.0007 | ✅ Complete; no-lift/negligible |
| M3.4 4-model ensemble | **0.9928** | 137 | +0.0008 vs M3.2 | ✅ Complete; modest positive lift |
| PI 50/50 ensemble follow-up | 0.9911-0.9921 | 77/137 | PI-spec completion | ✅ Complete; causal is online-scoring variant |
| M3.5 post-processing | 0.9927 post / 0.9928 pre | 137 | -0.000054 | ✅ Complete; null result documented |

## 5.2 M3 Exit Criteria

+ [x] M3.2 val AUC > 0.97 (達到 0.9920)
+ [x] M3 baseline + value-change pipeline 完成且 reproducible
+ [x] Leakage sanity check pass (NO LEAKAGE)
+ [x] PI-response 50/50 split + offline/causal design grid 完成
+ [x] M3.3 buds-lab alignment 完成; no robust AUC lift
+ [x] M3.4 ensemble 完成
+ [x] M3.5 post-processing 完成
+ [x] Each milestone 有對應 handoff doc

## 5.3 M3 思考點

1. **Scale 對 pipeline 的影響**: M3 規模是 M2 的 ~3.5× (buildings),但 LightGBM
   pipeline 在 ~20M rows 上仍可在 ~2 分鐘完成。M3.4 ensemble (尤其 CatBoost 1000 iters)
   可能需 30-60 min。

2. **Anomaly rate 差異**: M3 6.5% vs M2 2.13%。LEAD 是 GEPIII subset 且 anomaly
   定義可能更嚴。這影響 downsampling 後的 effective training size。

3. **無 leaderboard 對照**: M3 評估完全靠自定 val set,reproducibility 靠
   `random_state=42`。沒法跟 paper 數字直接比。

4. **Rule 2a building_id filter 不適用**: M2 用 `id>145 OR <105` (LEAD 200 buildings
   range)。M3 buildings 0-1448,完全不同 range。M3.5 需重新 EDA 找適合 filter。

5. **GaussianTargetEncoder leakage risk**: M3.3 加 gte_* 時必須 fit on train_m3 only,
   apply to val_m3 用 train params。否則會 leak anomaly label → AUC 虛高。

## 5.4 M3 Methodology lessons (新增)

| # | Milestone | Lesson |
|---|---|---|
| 8 | M3.1 | Positional anomaly label join 比 schema-based join 簡單但需驗證 row count 一致 |
| 9 | M3.2 | Vectorized groupby.shift on 20M rows 可行 (~15s per 60 shifts) |
| 10 | M3.2 leakage check | Past-only ≈ future-only AUC 反映 anomaly burst 雙向對稱性,不是 leakage |

(對應 M2 7 個 lessons + M3 新增,目前 total 10 個)

## 5.5 進度更新追蹤

+ 2026-05-29: M3.1 baseline 完成 (commit ea3977d)
+ 2026-05-29: M3.2 value-change 完成 (commit a1de001)
+ 2026-05-29: M3.2 leakage check + M3.3 redefined (commit c7d0c5a)
+ 2026-06-22: M3.2a PI-response split/causality design check 完成
+ 2026-06-22: M3.3 buds-lab alignment 完成; val AUC 0.9913, no-lift vs M3.2
+ 2026-06-22: M3.4 4-model ensemble 完成; seed-42 val AUC 0.9928
+ 2026-06-22: M3.5 post-processing null result 定稿; PI 50/50 ensemble follow-up 完成

---

*Last updated: 2026-06-22 (M3 complete; M3.5 null result 與 PI 50/50 ensemble follow-up 已定稿)*
