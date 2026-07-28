# 2026-07-25 TabPFN 137-feature 全 holdout 完成 handoff

這份 handoff 記錄把 137-feature TabPFN（n=8、context 100k）補滿**整個 50/50 building holdout**
並加上 M3 四張圖第四條線的完整執行。承接
`docs/reference/m5-tabpfn-137feature-full-holdout-runbook.md` 與
`docs/handoffs/2026-07-25-tabpfn-estimator-sweep-and-137feature-results.md`。

## 1. 一句話

137-feature TabPFN 已跑完全部 10,137,155 列、通過四關身分驗證合併，並加上 M3 四張圖；
全 holdout pooled **ROC-AUC 0.9919 / PR-AUC 0.9314**。

## 2. 結果

| 線 | pooled ROC-AUC | pooled PR-AUC |
|---|---:|---:|
| 17-feature 樹 ensemble（gray） | 0.9663 | 0.8221 |
| 137-feature 樹 ensemble（blue） | 0.9918 | 0.9303 |
| 17-feature TabPFN（orange） | 0.9120 | 0.6639 |
| **137-feature TabPFN（violet，本次）** | **0.9919** | **0.9314** |

## 3. 產物

- 合併輸出（gitignored，本機）：`data/processed/m5_tabpfn_137_full_test_n8_predictions.npz`
  （10,137,155 列，637,397 anomalies，分數 finite [2.6e-5, 0.99999]，canonical 列序對齊）。
- 四張更新圖（第四條 violet 線）：
  - `docs/reports/assets/m3/m3_feature_engineering_roc_with_tabpfn.png`
  - `docs/reports/assets/m3/m3_feature_engineering_precision_recall_with_tabpfn.png`
  - `docs/reports/assets/m3/m3_tree_ensemble_by_site_roc_with_tabpfn.png`
  - `docs/reports/assets/m3/m3_tree_ensemble_by_site_precision_recall_with_tabpfn.png`

## 4. 執行方式：pool scheduler 取代 batch barrier

原 `run_m5_tabpfn_137_batches.ps1` 是**逐 batch 阻塞**（一批的 head/tail 都完成才進下一批），
任一 shard 落後就讓另一張 A100 空等、把延誤串成整體延誤。本次改用 **2-slot 貪婪 pool 排程**：
任一 slot 空出就啟動下一個 shard，兩張 A100 全程滿載到收尾。每個 shard 在第一個 chunk 就以
`site_id ⊆ 該批 site 集合` 驗證資料正確。

- sites 1/2/3 先前已完成（per-site shard）。
- batch 0–5（12 shard）本次跑完；合併涵蓋 sites 1/2/3 + 6 batch = 全 holdout。
- 收尾：0 個殘留 session、無 CU 乾燒。

## 5. 修掉的 bug（重要）

`scripts/supervise_m5_tabpfn_site_shard.ps1` 的 `Invoke-Recovery` **未把 `-ShardRootName`
傳給 deploy**，自癒時會 fallback 到 per-site 預設（`m5_tabpfn_site<Site>_...`），
把**已完成的 Site-1 資料**部署到 batch session、產出標成本批卻是別批的分數。曾在 b1/tail 踩到
（手動重部署與舊 buggy supervisor 相撞）。修法：`Invoke-Recovery` 明確 forward
`-ShardRootName $ShardRootName`。

**操作守則**：有 live supervisor 的 shard 不要手動重部署 —— 先殺/換掉 supervisor，
或讓（已修好的）supervisor 自己救。

## 6. 繪圖腳本改動

`scripts/plot_m3_figures.py` 與 `scripts/plot_m3_tree_ensemble_by_site.py` 各加一條
137-feature TabPFN 線（key `tabpfn_137`，色 `#7a51a8`，來源
`m5_tabpfn_137_full_test_n8_predictions.npz`），維持既有邏輯、只加線與圖例，仍寫入
既有 `*_with_tabpfn.png`。by-site 副標改為精簡的
`Pooled ROC-AUC — gray (0.9663) · blue (0.9918) · violet (0.9919)`（PR 同理）。

## 7. Decision

137-feature TabPFN 全 holdout 推論完成、驗證通過、四張圖到位，pooled AUC 與樹 ensemble 齊平。
執行改採 pool 排程避免 straggler 拖累整體；supervisor 的 wrong-ShardRootName 自癒 bug 已修。
剩餘可選工作：若要把 17-feature TabPFN（orange）也列入 by-site 副標可再加，但目前副標聚焦
gray/blue/violet 三條主線。
