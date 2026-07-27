# M5 Plan：GEPIII 模型比較完成紀錄

**Status**: Complete
**Started**: 2026-06-25
**Primary issue**: [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)
**Report issue**: [#52](https://github.com/kuokuant-oss/lead-reproduction/issues/52)
**正式報告**: [docs/reports/m5-foundation-vs-gbdt.md](../reports/m5-foundation-vs-gbdt.md)

**Runtime provenance follow-up**: [#58](https://github.com/kuokuant-oss/lead-reproduction/issues/58)；runner 觀測層已完成，正式重跑須等待人工確認。

## 定位

M5 的範圍已收斂為 GEPIII 內部的 FDD 模型比較。它不是 BDG2 正式評估，也不宣稱 BDG2
transfer readiness；BDG2 的後續監督式評估已移到 M6，由 ADR 0025/0026 與
[bdg2-supervised-fdd-plan.md](./bdg2-supervised-fdd-plan.md) 定義。

M5 的可用結論來自正式報告與其輸出 JSON：

+ [docs/reports/m5-foundation-vs-gbdt.md](../reports/m5-foundation-vs-gbdt.md)
+ [data/processed/m6_phaseD_50_50_full_models.json](../../data/processed/m6_phaseD_50_50_full_models.json)

## 固定邊界

+ M3 numeric line 凍結：M3.2 LightGBM offline AUC `0.9920`、M3.4 ensemble AUC
  `0.9928`，以及既有 split、seed、downsampling、scaler path 不改。
+ M4 public API 凍結；M5 重用 `src/lead` data、feature、split、sample、evaluation
  helpers，不把實驗專用 helper 推成新的 public contract。
+ M5 評估只使用 ASHRAE GEPIII / Kaggle labels。BDG2 沒有 native per-row anomaly
  label；BDG2 label bridge 是 M6 問題，不回填成 M5 結論。
+ TabPFN 是候選模型，不是預設 production detector。GBDT 仍是可部署 scanner 的主要
  baseline；TabPFN 的 license 與 latency caveats 需隨後續比較一起呈現。

## 最終比較設計

正式比較使用同一組 feature matrix、scaler、split、fit rows、train scoring rows 與
test scoring rows；唯一變因是模型。

| 項目 | 設定 |
| --- | --- |
| Models | LightGBM、XGBoost、CatBoost、HistGBT、Ensemble、TabPFN |
| Feature regime | `row_offset_meter_aware` |
| Full features | 137 features：17 baseline + 120 value-change |
| Minimal-FE | 17 raw baseline features |
| Fit budget | 10,000 balanced rows；label-scarcity 另用 support sizes |
| Scoring budget | 4,000 natural-prevalence train rows 與 4,000 natural-prevalence test rows |
| In-domain split | `50_50_mod2`，依 `building_id % 2` 留出 |
| Site-transfer split | `site_id_mod2_50_50`，依 `site_id % 2` 留出 |
| Metrics | ROC-AUC、PR-AUC、threshold `0.5` confusion matrix、fixed recall `0.90` confusion matrix |

## 結果摘要

### In-domain full features

在 full 137 features 的 in-domain 50/50 test split 上，六個模型分數接近。Test
AUC 介於 `0.9938` 到 `0.9947`，test PR-AUC 介於 `0.9207` 到 `0.9304`。這一軸
沒有足夠證據宣稱單一模型明確勝出。

### Label scarcity

TabPFN 的主要優勢出現在低標註量 PR-AUC。Support `200` 時 TabPFN test PR-AUC
為 `0.7675`，高於其他模型；support `500` 時 TabPFN 為 `0.8220`，仍是最高。
Support 增加後，各模型差距快速收斂。

### Minimal feature engineering

Raw 17 features 下，ranking metric 與 threshold `0.5` classification 給出不同訊號。
Tree models 的 test PR-AUC 較高；TabPFN 在 fixed threshold 下 TN 最高、FP 最少，且
FP+FN 總錯誤數最低。因此 M5 不支持「TabPFN 可直接取代 value-change feature
engineering」這個較強假設。

### Site-transfer

Site-transfer split 下，TabPFN test AUC 仍高，但 test PR-AUC 是六個模型中最低
`0.5479`。以 anomaly ranking 來看，tree family 較強；這也是後續 BDG2 / M6 不能只
沿用 GEPIII in-domain verdict 的原因。

## 相關 commits 與 issue 對齊

## Additive follow-up: TabPFN 500K single context

The resource-guarded scaling line is specified in
[`m5-tabpfn-500k-single-context.md`](../reports/m5-tabpfn-500k-single-context.md)
and ADR 0028. It is additive: raw 17 features, unique balanced nested contexts,
isolated workers, and resumable 100K--500K budgets do not change accepted M5
metrics or golden fixtures. Formal GPU execution remains pending explicit
operator invocation.

## Additive follow-up: 訓練量 vs 性能曲線（context curve，執行中）

問的是「標註資料稀少時 TabPFN 是否有優勢」，因此**兩個模型都掃 N**，看
`TabPFN(N) − Trees(N)` 隨 N 的變化。協定見
[`m5-tabpfn-context-curve-runbook.md`](../reference/m5-tabpfn-context-curve-runbook.md)；
執行進度見
[2026-07-27 handoff](../handoffs/2026-07-27-m5-tabpfn-context-curve-gputw-run.md)。

2026-07-27 在 gputw.ai 的 RTX 5090 上跑了 3 小時，**8 個 cell 完成 3 個**：
10k/17、10k/137、20k/17，三個合併產物都通過識別關卡（各 10,137,155 列、
637,397 anomalies、與 100k 線 `raw_index`/`anomaly` 逐列相同）。GPU 已停機，
所有已計分結果都已拉回本機並驗證。既有的 100k 四張圖不受影響。

初步結果**兩條線方向相反**：17 維是 context 越小越好且三點單調
（ROC 0.9413 / 0.9398 / 0.9163 對 10k / 20k / 100k），137 維則相反
（10k 0.9902 → 100k 0.9919）。137 那組差距小且只有兩點，尚不足以稱為趨勢。

實測推翻了 §9 的估算：17 維 @20k 約 4,000 rows/s（估算值的 3.3 倍），瓶頸是
~1.35 MB/s 的上行且**並行無效**，因此是上傳受限。樹的 matched-N 手臂修好了一個
會破壞列身分對應的 bug，但**尚未產出任何結果**，所以 `TabPFN(N) − Trees(N)` 還算不出來。

| Commit | 用途 |
| --- | --- |
| `bfd6664` | 啟動 M5 TabPFN feasibility spike。 |
| `e0ecdf7` | 完成 Phase C local-checkpoint TabPFN feasibility。 |
| `e163bef` | 修正 Phase C threshold metric provenance 與 latency breakdown。 |
| `8f4373b` | 退回過早的 BDG2 ingestion skeleton，避免 M5 與 BDG2 scope 混在一起。 |
| `ebba62c` | 完成 M5 Phase D TabPFN-vs-GBDT comparison on GEPIII。 |
| `9e15709`、`68d2e80`、`407e4a7` | 收斂中文報告、FDD framing 與模型-track closeout。 |
| `04d7514`、`4142d5c` | 把 M5/M6 敘事改到 supervised BDG2 pivot；M5 保留為 GEPIII report。 |
| `c98f4ad` | 將 meter-aware M5 comparison line 實作成目前報告採用的結果面。 |

Issue 對齊：

+ [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)：M5 Phase D
  TabPFN-vs-GBDT GEPIII comparison，已完成。
+ [#52](https://github.com/kuokuant-oss/lead-reproduction/issues/52)：M5 報告與
  meter-aware comparison line 的收斂紀錄。
+ [#58](https://github.com/kuokuant-oss/lead-reproduction/issues/58)：在不改模型條件與
  預測邏輯下，補齊硬體與時間成本 provenance；runner 已就緒，正式重跑尚未開始。
+ [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27)、[#30](https://github.com/kuokuant-oss/lead-reproduction/issues/30)、[#32](https://github.com/kuokuant-oss/lead-reproduction/issues/32)
  是 M5 前置 planning、feasibility、metric audit 歷史；不再代表 active M5 待辦。
