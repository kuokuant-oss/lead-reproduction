# M5 Phase D：TabPFN（基礎模型）vs GBDT（樹模型）於 GEPIII

+ **Issue**：[#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)
+ **資料範圍**：既有 M3 ASHRAE GEPIII frame（`20,216,100 × 21`），含標籤。
+ **執行環境**：資料留在本地環境處理，TabPFN 使用本地權重執行。
+ **Provenance**：結果以 `data/processed/m5_phaseD_foundation_vs_gbdt.json`
  （commit `8f4373b`，產生於 2026-06-26 UTC）為準。

---

## 1. 任務與共同設定

M5 Phase D 比較 TabPFN-3 與 single LightGBM 在 GEPIII 上的模型行為。每個配對
cell 都重用相同的 split、downsample、feature table 與固定驗證子樣本，並透過 frozen
`src/lead` pipeline 執行：`load_m3_frame`、`add_value_change_features`、
`split_mask`、`downsample_indices`、`classification_metrics`。同一個配對 cell
中唯一變因是模型本身。

| 項目 | 設定 |
|---|---|
| GBDT | LightGBM `LGBMClassifier(n_estimators=100)` |
| TabPFN | TabPFN-3 本地 checkpoint（`tabpfn==8.0.8`、RTX 4070 Laptop GPU、8 GB） |
| 特徵表 | 137 features（17 baseline + 120 row-offset value-change），即 M3.2 line |
| 訓練預算 | 10,000 balanced rows |
| 驗證 | 每軸固定 4,000-row natural-prevalence 子樣本，anomaly rate 約 6% |
| Seeds | fit-subsample 與模型 `random_state` 取 `{42, 123, 999}` |
| 指標 | ROC-AUC、PR-AUC、precision/recall/F1@0.5、fit+predict 延遲 |

原始 headline 使用單一 validation 子樣本。INV-2 額外以 5 個 validation seeds
量測抽樣變異；in-domain GBDT ROC-AUC std 為 `0.0019`，site-transfer GBDT
ROC-AUC std 為 `0.0044`。下列小 delta 依此抽樣變異解讀。舊 fixture 未覆蓋，
新數據來源為 `data/processed/inv2_phaseD_val_variance.json`。

---

## 2. TabPFN 執行邊界

TabPFN 評估採用本地權重與 10,000 列平衡訓練子樣本。本次 10,000 列訓練預算主要由
本地 8 GB 顯示記憶體決定；該設定仍低於 TabPFN-3 已記載的 `1,000,000 × 200`
輸入限制，`ignore_pretraining_limits` 從未被設定。完整 M3 下採樣表格約為
`4,285,104 × 137`，超出可直接輸入範圍，因此本階段比較的是 10,000 列情境下的
模型行為。

TabPFN 的準確率提升需要搭配延遲解讀。本次 4,000 列驗證子樣本上，TabPFN 評分約需
25 至 27 秒，約為每列 6.3 毫秒；GBDT 則為次秒級。TabPFN 的機制會在預測時重新使用
上下文訓練集，後續 M6.3 銜接時需保留準確率、延遲與授權條件三類觀察量。

---

## 3. 四個比較軸

### 3.1 站內建物切分（`80_20_mod5`）

| 模型 | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
|---|---:|---:|---:|---:|
| GBDT (LightGBM, 10k fit) | 0.9877 ± 0.0012 | 0.9154 ± 0.0068 | 0.756 ± 0.013 | ~0.23 |
| TabPFN-3 (10k context) | **0.9925 ± 0.0005** | **0.9253 ± 0.0049** | 0.747 ± 0.007 | 26.8 ± 2.0 |

站內建物切分下，TabPFN 平均分數略高，但差距接近驗證抽樣變異，因此只能視為接近強
基準的結果。INV-2 計入 validation 抽樣變異後，TabPFN 減 GBDT 的 paired ROC-AUC
delta mean 為 `+0.00255`、std 為 `0.00238`，範圍為 `-0.0025` 至 `+0.0059`。

已被接受的 M3.4 line 是完整資料上的 4-model ensemble，ROC-AUC 為 `0.9928`。
TabPFN 在 10k context 下的 in-domain ROC-AUC 接近該 ensemble。

### 3.2 跨站轉移（`site_id % 5 == 4` held out）

真正跨站比較只採用訓練時未見目標站點的設定。M3 ensemble site-held-out anchor
為 ROC-AUC `0.9774`，此 anchor 是完整資料的 4-model ensemble，作為背景參照。

| 條件 | ROC-AUC | PR-AUC | F1@0.5 | fit+predict (s) |
|---|---:|---:|---:|---:|
| GBDT-retrain | 0.9797 ± 0.0008 | **0.8221 ± 0.0035** | 0.780 ± 0.013 | ~0.24 |
| TabPFN-in-context | **0.9833 ± 0.0009** | 0.8119 ± 0.0052 | **0.783 ± 0.003** | 26.5 ± 0.2 |

跨站設定中，TabPFN 的 ROC-AUC 優勢在配對比較下較一致；PR-AUC 則受驗證抽樣影響
較大，尚不能判定由哪個模型穩定領先。INV-2 計入 validation 抽樣變異後，TabPFN
減 GBDT-retrain 的 ROC-AUC delta mean 為 `+0.00438`、std 為 `0.00228`，所有
paired delta 為正；PR-AUC delta mean 為 `-0.00326`、std 為 `0.01045`，跨零。

另有一個已知站點內的新建物泛化結果：GBDT 直接套用站內模型可達 ROC-AUC `0.9882`、
PR-AUC `0.9023`。訓練資料已包含目標站點的其他建物，這項結果只作補充，不納入真正
跨站比較。

### 3.3 標註稀少（`80_20_mod5`，固定 4k validation）

| Support | GBDT ROC | TabPFN ROC | ΔROC | GBDT PR | TabPFN PR | ΔPR |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 0.9659 | 0.9806 | **+0.0148** | 0.6954 | 0.7953 | **+0.0999** |
| 500 | 0.9786 | 0.9829 | +0.0043 | 0.7669 | 0.8302 | +0.0634 |
| 1,000 | 0.9809 | 0.9834 | +0.0025 | 0.7815 | 0.8507 | +0.0692 |
| 2,000 | 0.9851 | 0.9863 | +0.0012 | 0.8635 | 0.8818 | +0.0183 |
| 5,000 | 0.9885 | 0.9899 | +0.0014 | 0.9086 | 0.9121 | +0.0035 |
| 10,000 | 0.9877 | 0.9925 | +0.0048 | 0.9154 | 0.9234 | +0.0080 |

標註稀少情境是本階段最支持 TabPFN 的證據。INV-2 的 paired bootstrap CI 顯示
support 200 的 PR-AUC delta mean 為 `+0.116`（CI `[0.095, 0.139]`），support
500 為 `+0.063`（CI `[0.049, 0.077]`），support 1,000 為 `+0.070`
（CI `[0.057, 0.083]`），皆不跨零。

固定 `val_seed=42` 下 support 5,000 的 GBDT ROC-AUC 高於 support 10,000 的
非單調，在 5 個 validation seeds 下消失；support 5,000 為 `0.9872 ± 0.0019`，
support 10,000 為 `0.9883 ± 0.0019`。此非單調屬 validation 抽樣雜訊。

### 3.4 最少特徵工程（`80_20_mod5`，10k fit、4k validation）

| 特徵集 | GBDT ROC | TabPFN ROC | GBDT PR | TabPFN PR |
|---|---:|---:|---:|---:|
| Raw baseline (17 feats) | **0.9587 ± 0.0042** | 0.9499 ± 0.0016 | **0.8305** | 0.7943 |
| Full value-change (137 feats) | 0.9877 | **0.9924** | 0.9154 | **0.9248** |
| ROC drop 137 → 17 | **−0.0290** | −0.0424 | — | — |

本階段未觀察到 TabPFN 可減少值變化特徵需求的證據。使用原始 17 特徵時，GBDT 的
ROC-AUC 與 PR-AUC 都高於 TabPFN；加入 137 個完整值變化特徵後，兩者分數才接近或
反轉。時間脈絡仍需要透過值變化或電表感知特徵提供。

---

## 4. M5 觀察摘要

GBDT 的優勢在於推論速度快、原始特徵條件下較強，且部署邊界清楚。TabPFN 的優勢
集中在標註稀少情境，並在真正跨站設定中呈現較穩定的 ROC-AUC 優勢。

M5 不支持「TabPFN 可降低值變化特徵需求」這個解讀。原始 17 特徵下，GBDT 優於
TabPFN；時間脈絡仍需要透過值變化或電表感知特徵提供。

---

## 5. 銜接 M6

M5 只提供 GEPIII 上的比較觀察；BDG2 上的正式判斷留給 M6。M6 應先完成標籤橋接
與完整性檢查，再於相同的 BDG2 overlap frame 上回報 GBDT 與 TabPFN 的準確率、
延遲、特徵需求與授權限制。BDG2-only、2017 年資料與其他電表範圍只作未標註
補充證據，不進入主要監督式評估分母。

M6.3 銜接時保留 TabPFN 約每列 6.3 毫秒的延遲、research/internal-use 授權邊界，
以及 real-time FDD claim 需使用 `PAST_SHIFTS`-only features（ADR 0007/0011）
這三個條件。

---

## 6. 數字來源與程式碼索引

| 報告內容 | 程式碼 | 數字輸出 |
|---|---|---|
| Phase C TabPFN feasibility spike | [scripts/run_m5_phaseC_tabpfn_spike.py](../../scripts/run_m5_phaseC_tabpfn_spike.py), [tests/test_m5_tabpfn_spike.py](../../tests/test_m5_tabpfn_spike.py) | `data/processed/` Phase C outputs |
| Phase D GBDT vs TabPFN comparison | [scripts/run_m5_phaseD_foundation_vs_gbdt.py](../../scripts/run_m5_phaseD_foundation_vs_gbdt.py), [tests/test_m5_phaseD_comparison.py](../../tests/test_m5_phaseD_comparison.py) | [data/processed/m5_phaseD_foundation_vs_gbdt.json](../../data/processed/m5_phaseD_foundation_vs_gbdt.json) |
| Validation 抽樣變異與 paired deltas | [scripts/run_m5_phaseD_foundation_vs_gbdt.py](../../scripts/run_m5_phaseD_foundation_vs_gbdt.py) | [data/processed/inv2_phaseD_val_variance.json](../../data/processed/inv2_phaseD_val_variance.json) |
| Frozen pipeline helpers | [src/lead/data.py](../../src/lead/data.py), [src/lead/features.py](../../src/lead/features.py), [src/lead/split.py](../../src/lead/split.py), [src/lead/sample.py](../../src/lead/sample.py), [src/lead/evaluate.py](../../src/lead/evaluate.py) | [tests/golden_metrics.json](../../tests/golden_metrics.json) |
