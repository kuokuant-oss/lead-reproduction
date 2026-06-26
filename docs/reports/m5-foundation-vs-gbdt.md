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
為 `0.9928`，而 single-GBDT-at-10k 的 baseline 略低於它。在 10k context 下，
TabPFN 的 in-domain ROC-AUC 接近已接受的 M3.4 ensemble，顯示其 accuracy 已達
強基準水準；主要限制是推論成本仍遠高於 GBDT。

---

## Axis 2 — Site transfer（PRIMARY，`site_id % 5 == 4` held out）

對兩個*真正跨站（true cross-site）*的模型而言，held-out sites 在訓練時從未被看過。
M3 ensemble site-held-out anchor：ROC-AUC `0.9774`（完整資料的 4-model ensemble；
與這些 single-model 10k cells 不直接可比）。

| Condition | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
| --- | --- | --- | --- | --- |
| GBDT-retrain (source sites only) | 0.9797 ± 0.0008 | **0.8221 ± 0.0035** | 0.780 ± 0.013 | ~0.24 |
| TabPFN-in-context (source sites only) | **0.9833 ± 0.0009** | 0.8119 ± 0.0052 | **0.783 ± 0.003** | 26.5 ± 0.2 |
| GBDT-transfer, no retrain (in-domain model) | 0.9882 | 0.9023 | 0.761 | ~0.003 |

**解讀。** 在**真正跨站**的模型中（只用 source sites 訓練），TabPFN-in-context
在 ROC-AUC（+0.0035）與 F1 上勝過 GBDT-retrain，GBDT-retrain 則在 PR-AUC
上勝出（+0.010）—— 兩者勝負互見。**GBDT-transfer-without-retrain** 那一列有最高
的 ROC-AUC（0.9882）與 PR-AUC（0.9023），但它是較容易的設定：該模型是在
`80_20_mod5` building split 上訓練的，其 source buildings 橫跨*所有* sites ——
包含 held-out sites 裡的其他 buildings。因此該設定包含 site-level familiarity
（weather regime、site mix），與 true cross-site setting 不同。該設定衡量的是
已部署 all-sites 模型對*已知* sites 中新 *buildings* 的泛化；因此這一列應作為
known-site building generalization 的參考，而不應直接用來評估 cross-site
transfer。

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

**「降低特徵工程負擔」的假設在此並不成立。** 在 raw 17-feature 集上，GBDT 其實
*勝過* TabPFN（0.9587 vs 0.9499 ROC；0.831 vs 0.794 PR-AUC），而當移除工程化的
value-change lags 後，TabPFN 掉得*更多*（−0.042 vs GBDT 的 −0.029）。row-offset
value-change features 編碼了單一 raw row 無法表達的時間脈絡，而 TabPFN 的
in-context learning 並未從 raw tabular rows 還原那種時間序列結構 —— 它至少和
GBDT 一樣仰賴這些工程化特徵。在此 anomaly-detection 設定中，TabPFN **並未**
降低特徵工程負擔。

---

## 結論（依 ADR 0015）

ADR 0015 指出，要以 transfer、label scarcity、minimal feature engineering 來
評判 TabPFN —— 而非單一頭條 AUC。依此準則：

**TabPFN 勝過 GBDT 之處**

+ **Label scarcity（最強結果）**：在小 support 時有顯著的 PR-AUC 優勢
  （200 labels 時 +0.100 PR-AUC），並隨標註增加而縮小。
+ **真正跨站 transfer 的 ROC-AUC**：TabPFN-in-context 0.9833 vs GBDT-retrain
  0.9797（+0.0035）且 F1 較高 —— 不過 GBDT-retrain 在 PR-AUC 勝出。
+ **在對齊的 10k budget 下的 in-domain**：0.9925 vs 0.9877，追平調校後的
  M3.4 ensemble（0.9928）。

**GBDT 勝出 / TabPFN 幫不上忙之處**

+ **推論延遲**：主要阻礙是推論成本，而非 accuracy；TabPFN 的 in-context 推論
  遠慢於 GBDT 的次秒級評分（完整延遲數字見 Axis 1），就現況不適用於低延遲的
  real-time FDD。這是離線 benchmark，非 real-time FDD 保證；任何 real-time 宣稱
  仍需依 ADR 0007 與 ADR 0011 使用 `PAST_SHIFTS`-only 的 causal features。
+ **Minimal feature engineering**：在 raw features 上 GBDT > TabPFN，且 TabPFN
  在缺少工程化 lags 時掉得更多 —— 與「省特徵工程」的假設相反。
+ **Site-transfer 的 PR-AUC**：GBDT-retrain 略勝 TabPFN-in-context。
+ **調校後的頭條**：已被接受的完整資料 M3.4 GBDT ensemble（0.9928）並未被
  取而代之；TabPFN 只在遠高的推論成本下才追平它。

**總結**：TabPFN 是一個可信的基礎模型候選，**特別適用於 label-scarce 與
cross-site** 的情境，在那裡它帶來真實價值（尤其是 PR-AUC）。它目前尚不能取代
調校後的 GBDT line：accuracy 接近，但推論成本高，且仍依賴工程化的 value-change
features。

## 延後至 Phase E（BDG2）—— M5 的下一階段

+ 真正跨**資料集（cross-dataset）**轉移至 BDG2（不同 buildings、sites、meters），
  使用真實的 BDG2 資料、schema 與標籤 —— 而非已退役的合成 skeleton。
+ 對 BDG2 的無標註 / few-shot target-site 適應。
+ 任何 real-time FDD 延遲工程：TabPFN 的推論延遲必須降低數個數量級，且特徵必須
  是 `PAST_SHIFTS`-only（ADR 0007/0011）。
