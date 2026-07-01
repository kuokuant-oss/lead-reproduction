# M5 Phase D — TabPFN（基礎模型）vs GBDT（樹模型）於 GEPIII

**Issue**: [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)
**資料**: 現有 M3 ASHRAE GEPIII frame（`20,216,100 × 21`），含標籤。
無 BDG2、無雲端、無資料外傳。TabPFN 僅以本地權重執行。
**Provenance**: 以 `data/processed/m5_phaseD_foundation_vs_gbdt.json`
（commit `8f4373b`，產生於 2026-06-26 UTC）為準。

## 設定

每個配對 cell 都重用**相同的 split、downsample、feature table 與固定的驗證
子樣本**，皆透過 frozen `src/lead` pipeline（`load_m3_frame`、
`add_value_change_features`、`split_mask` 式的 mask、`downsample_indices`、
`classification_metrics`）。一個配對 cell 中唯一的變因是模型本身。

+ **模型**：TabPFN-3 本地 checkpoint（`tabpfn==8.0.8`、RTX 4070 Laptop GPU、
  8 GB）vs LightGBM `LGBMClassifier(n_estimators=100)`。兩者吃同一份
  `StandardScaler` 轉換後的 table。
+ **Feature table**：137 features（17 baseline + 120 row-offset value-change），
  即 M3.2 line。
+ **Fit budget**：10,000 balanced rows（遠超 1,000-row 的 Phase C spike）。
  137 features ≤ 200 且 10,000 ≪ 1,000,000，因此整個執行**落在已記載的
  TabPFN-3 `1,000,000 × 200` 限制內** —— `ignore_pretraining_limits` 從未被設定。
  此 budget 受限於 8 GB laptop VRAM，而非那個已記載的上限。（完整 M3 downsample
  為 `4,285,104 × 137`，超過 `1,000,000 × 200` 的 row 上限，因此完整 table 無法
  餵給 TabPFN-3。）
+ **驗證（Validation）**：每軸固定 4,000-row 的自然盛行率（natural-prevalence）
  子樣本（anomaly rate ≈ 6%），由兩個模型以相同方式評分。
+ **Seeds**：fit-subsample 與模型 `random_state` 取 `{42, 123, 999}`；以
  mean ± std 回報。指標：ROC-AUC、PR-AUC（average precision）、0.5 門檻下的
  precision/recall/F1，以及 fit+predict 延遲。

以下所有指標除另註明外皆為 **3 個 seeds 的 mean ± std**。延遲為冷啟動的
in-process fit+predict（TabPFN 含 model init + fit + `predict_proba`）；TabPFN 的
`predict_proba` 每次呼叫都會對 in-context 訓練集重新計算。

---

## Axis 1 — In-domain（`80_20_mod5` building split）

| Model | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
| --- | --- | --- | --- | --- |
| GBDT (LightGBM, 10k fit) | 0.9877 ± 0.0012 | 0.9154 ± 0.0068 | 0.756 ± 0.013 | ~0.23 (warm) |
| TabPFN-3 (10k context) | **0.9925 ± 0.0005** | **0.9253 ± 0.0049** | 0.747 ± 0.007 | 26.8 ± 2.0 |

TabPFN 略勝 single-GBDT-at-10k 的 baseline（+0.0048 ROC、+0.010 PR-AUC），
但其 `predict_proba` 對 4,000 rows 需約 25.3 s（~6.3 ms/row），相對於 GBDT
的次秒級評分 —— 在推論上大約**慢兩個數量級**。

**脈絡**：已被接受的 M3.4 line 是*在完整資料上的 4-model ensemble*，ROC-AUC
為 `0.9928`，而 single-GBDT-at-10k 的 baseline 略低於它。TabPFN 在 10k context 下
的 in-domain ROC-AUC 接近該 ensemble，accuracy 已達強基準水準（延遲代價見上）。

---

## Axis 2 — Site transfer（PRIMARY，`site_id % 5 == 4` held out）

對 true cross-site 模型而言，held-out sites 在訓練時從未被看過。M3 ensemble
site-held-out anchor：ROC-AUC `0.9774`（完整資料的 4-model ensemble；與這些
single-model 10k cells 不直接可比）。

True cross-site（只用 source sites 訓練）：

| Condition | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
| --- | --- | --- | --- | --- |
| GBDT-retrain | 0.9797 ± 0.0008 | **0.8221 ± 0.0035** | 0.780 ± 0.013 | ~0.24 |
| TabPFN-in-context | **0.9833 ± 0.0009** | 0.8119 ± 0.0052 | **0.783 ± 0.003** | 26.5 ± 0.2 |

兩個 true cross-site 模型間，TabPFN-in-context 在 ROC-AUC（+0.0035）與 F1 勝出，
GBDT-retrain 在 PR-AUC 勝出（+0.010）。

附註（known-site building generalization，不納入跨站勝負）：GBDT-transfer-without-retrain
—— 即 in-domain 全 sites 模型直接套用 —— 得 ROC-AUC `0.9882`、PR-AUC `0.9023`，
但其 source buildings 橫跨所有 sites（含 held-out sites 的其他 buildings），衡量的
是已知 site 內新 building 的泛化，而非 cross-site transfer，因此不列入上表。

---

## Axis 3 — Label scarcity（`80_20_mod5`，固定 4k val）

ROC-AUC 與 PR-AUC（3 個 seeds 的 mean），隨著有標註的 support set 縮小：

| Support | GBDT ROC | TabPFN ROC | ΔROC | GBDT PR | TabPFN PR | ΔPR |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 0.9659 | 0.9806 | **+0.0148** | 0.6954 | 0.7953 | **+0.0999** |
| 500 | 0.9786 | 0.9829 | +0.0043 | 0.7669 | 0.8302 | +0.0634 |
| 1,000 | 0.9809 | 0.9834 | +0.0025 | 0.7815 | 0.8507 | +0.0692 |
| 2,000 | 0.9851 | 0.9863 | +0.0012 | 0.8635 | 0.8818 | +0.0183 |
| 5,000 | 0.9885 | 0.9899 | +0.0014 | 0.9086 | 0.9121 | +0.0035 |
| 10,000 | 0.9877 | 0.9925 | +0.0048 | 0.9154 | 0.9234 | +0.0080 |

**這是 TabPFN 最明確的勝場。** 在 200 labels 時，TabPFN 領先 +0.015 ROC 與
**+0.100 PR-AUC**；隨著標註增加，差距（在 PR-AUC 上）單調縮小。PR-AUC 視角
—— 對一個 ~6% 盛行率的 anomaly 任務而言是正確的觀察角度 —— 顯示這個基礎模型
在標註稀少時明顯更好，正是它被期待能幫上忙的地方。

---

## Axis 4 — Minimal feature engineering（`80_20_mod5`，10k fit、4k val）

| Feature set | GBDT ROC | TabPFN ROC | GBDT PR | TabPFN PR |
| --- | --- | --- | --- | --- |
| Raw baseline (17 feats) | **0.9587 ± 0.0042** | 0.9499 ± 0.0016 | **0.8305** | 0.7943 |
| Full value-change (137 feats) | 0.9877 | **0.9924** | 0.9154 | **0.9248** |
| **ROC drop 137 → 17** | **−0.0290** | −0.0424 | — | — |

**「降低特徵工程負擔」的假設在此並不成立。** 在 raw 17-feature 集上，GBDT
*勝過* TabPFN（0.9587 vs 0.9499 ROC；0.831 vs 0.794 PR-AUC），而移除工程化的
value-change lags 後，TabPFN 掉得*更多*（−0.042 vs GBDT 的 −0.029）。row-offset
value-change features 編碼了單一 raw row 無法表達的時間脈絡，TabPFN 的 in-context
learning 並未從 raw tabular rows 還原那種結構，因此它至少和 GBDT 一樣仰賴這些
工程化特徵。

---

## 結論（依 ADR 0015）

依 ADR 0015 的準則（以 transfer、label scarcity、minimal feature engineering
判斷，而非單一頭條 AUC）：

TabPFN-3 在 GEPIII 上不是 GBDT 的全面替代品，但在 label-scarce 與部分 cross-site
設定中提供了真實增益 —— 最明確是 label scarcity 的 PR-AUC，其次是 true cross-site
的 ROC-AUC。在 10k support 下，它的 in-domain accuracy 已接近完整 M3.4 GBDT
ensemble，但推論延遲高出數個數量級，限制了即時 FDD 使用。它也沒有降低 feature
engineering burden；value-change features 仍是關鍵。

**Decision**：keep GBDT as the production baseline；keep TabPFN as a label-scarce /
transfer candidate；do not claim real-time readiness；move cross-dataset validation
to Phase E (BDG2).

ADR 0025 now carries this Phase D result into BDG2 as a supervised-overlap M6
comparison frame. GBDT remains the real-time scanner candidate because it is
sub-second, while TabPFN was about `6.3 ms/row` (`~100x` slower) and remains
bounded by the TabPFN-3.0
research/internal-use license. Minimal feature engineering was not a TabPFN
win, either: raw-17-feature GBDT ROC-AUC `0.9587` exceeded TabPFN `0.9499`, so
value-change / meter-aware feature engineering remains necessary. In BDG2 FDD,
TabPFN is restricted to offline audit roles: second-stage candidate re-ranking,
model-disagreement diagnostics, active-learning audit-set selection, and
few-shot calibration after a small manual audit set. Its output is
triage/ranking utility, not BDG2 supervised performance or fault confirmation.

## 延後至 Phase E（BDG2）—— M5 的下一階段

+ 真正跨**資料集（cross-dataset）**轉移至 BDG2（不同 buildings、sites、meters），
  使用真實的 BDG2 資料、schema 與標籤 —— 而非已退役的合成 skeleton。
+ 對 BDG2 的無標註 / few-shot target-site 適應。
+ 任何 real-time FDD 延遲工程：TabPFN 的推論延遲必須降低數個數量級，且特徵必須
  是 `PAST_SHIFTS`-only（ADR 0007/0011）。
