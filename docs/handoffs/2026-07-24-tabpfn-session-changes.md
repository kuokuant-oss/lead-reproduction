# 2026-07-24 M5 TabPFN session 改動與待辦 handoff

這份 handoff 記錄 2026-07-24 這次 session 的所有改動、產物、發現與待更新事項，給接手者與未來的 AI agent。寫法沿用
`docs/reference/m5-tabpfn-colab-dual-shard-runbook.md`。

## 1. 這次做了什麼（一句話）

把暫停在 85.2% 的正式 17-feature TabPFN dual-shard run 恢復並跑完（10,137,155 rows），合併、產出四張含 TabPFN 疊線的 M3 圖；順手修好一個會殺掉 supervisor 的 resume race；接著完成 137-feature 的重新 fit，並用小樣本 probe 量測 `n_estimators` 對 TabPFN 的增益。

## 2. 已 commit（本分支 `codex/m6-tabpfn-context-curves`）

| commit | 內容 |
|---|---|
| `85e301d` | 暫停 dual-shard run 於 85.2%，強化 recovery supervisor |
| `2196b67` | 修 `launch_and_verify_colab` 的 resume 健康檢查：改成輪詢等待 heartbeat 追上 durable frontier，只有整段寬限期後仍落後才判 invariant，不再一次取樣就殺掉 supervisor；補 3 個測試 |
| `a6d90f8` | 把合併後的 TabPFN 分數疊到 M3 判別力四圖（by-site ROC/PR + pooled ROC/PR），另存 `*_with_tabpfn.png`，不覆蓋原圖；一併帶入 pooled 圖從 LightGBM 改為 Tree Ensemble 的既有未提交工作 |

## 3. 正式 17-feature run 結果（已完成、已驗證）

- head 253 chunks `[0, 5,060,000)`、tail 254 chunks `[5,060,000, 10,137,155)`，兩張 L4 皆已釋放。
- 合併產物 `data/processed/m5_tabpfn_distributed_context100000_predictions.npz`（gitignored），10,137,155 rows 逐 20k span 對齊 canonical `raw_index/anomaly/site_id/building_id`，另獨立驗證 raw_index 唯一、與 canonical 同集。
- pooled：TabPFN ROC-AUC **0.9120** / PR-AUC **0.6639**；對照 17-feature 樹 ensemble 0.9663 / 0.8221。TabPFN 在同 17 features 下低於樹 ensemble，主因是 in-context 100k vs 樹全量訓練的結構劣勢，非 bug（分數無飽和、47,815 相異值已驗證）。

## 4. 這次新增、尚未 commit 的腳本

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `scripts/fit_m5_tabpfn_137_context100000.py` | 137-feature 一次性 fit（100k context、seed 42、同 foundation） | 已跑成功，產物在 `data/processed/m5_tabpfn_137_full_test_context100000.work/` |
| `scripts/export_m5_tabpfn_137_shards.py` | 137-feature head/tail portable inputs（保留原 index、SHA 驗證） | 已寫、未執行 |
| `scripts/build_tabpfn_estimator_probe_inputs.py` | 建 estimator probe 的小 npz（scaled context + per-site 抽樣） | 已用於 probe |
| `scripts/run_tabpfn_estimator_probe_colab.py` | Colab estimator sweep worker（自包含、每 (site,n) 增量寫檔） | 已用於 probe |
| `scripts/probe_tabpfn_estimators.py` | 本機版 estimator probe（Colab 版的前身） | 未使用，可保留或刪 |

## 5. 兩個發現

- **137-feature fit 完成**：`m5_tabpfn_137_full_test_context100000.work/`（`model.tabpfn_fit` 30 MB、`scaler.joblib`、`fit_manifest.json`），feature_names = 17 baseline + 120 `lag_value_*`，context NaN fraction 2.18%（merge miss，合理）。**137-feature 全 test 推論尚未跑**（見 §7）。
- **estimator probe（僅方向參考，非結論）**：Site 1（n=1 幾乎亂猜 ROC 0.5447）在 50k 抽樣下 `n=4` 得 ROC 0.6267 / PR 0.2456，即 `n=1→n=4` ROC +0.082。這推翻了先前「estimator 只影響 <2%」的估計。完整量測計畫見
  `docs/handoffs/2026-07-24-tabpfn-17feature-estimator-sweep-plan.md`。

## 6. 需要更新的事項

- **runbook §1 與 137-feature**：`m5-tabpfn-colab-dual-shard-runbook.md` §1 仍寫「不可啟動 137-feature 實驗」。使用者已於 2026-07-24 核准 137-feature 為並存的第二條線（重新 fit、獨立產物、只並排比較不合併）。應在 runbook 另加一節說明，勿原地改 §1，以保 17-feature 契約可讀。
- **Colab 帳號**：新增第二個帳號 `tonykuo210100@gmail.com`，HOME `/home/tonykuo/.colab-tony`，已 OAuth 授權（`token.json` 存在）。登入須 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`，否則 Google 少給 `drive.file` scope 會導致寫檔前中止。此帳號有 Colab Pro。連不上時退回 `hank0503work@gmail.com`。
- **孤兒圖檔**：`docs/reports/assets/m3/confusion.png`、`lightgbm_pr.png`、`lightgbm_roc.png` 為被取代的殘留輸出，報告未引用，建議刪除，勿 commit。
- **既有未提交改動**：`docs/reports/m3-report.md`（表格 LightGBM→Tree Ensemble，與 a6d90f8 一致）、兩個 keepalive `.ps1`（+18 行 M5 keepalive 微調），非本 session 產生但屬同脈絡，可一併整理進 commit。

## 7. 兩個未執行的後續計畫

1. **137-feature 全 test 推論**：fit 已就緒，尚未建 portable inputs、尚未上 Colab。目標是把 137-feature TabPFN 當第四條線加到同樣四張圖。使用者要求 17-feature 全部收工後自動啟動、不等人；連不上新帳號就用 hank。
2. **17-feature estimator sweep（Site 1/2/3 全列、n ∈ {4,8}）**：見專屬 handoff。核心紀律：全列量測、不得用抽樣宣稱增益；因 ~103 小時遠超 L4 壽命，必須用 runbook 可續跑基建。

## 8. 現況快照

- `colab sessions` = `No active sessions`；probe 用的兩張 L4 已 terminate，CU 停止消耗。
- 本機無執行中的 fit / probe / worker process；兩個 recovery supervisor scheduled task 於正式 run 收尾後已無活動。
- 已保存 `data/processed/probe_results_a.json`（Site 1 n=4 抽樣結果）。

## 9. Decision

本 session 的正式 17-feature 交付（合併 + 四圖 + supervisor 修復）已完成並 commit（`a6d90f8` 等三筆）。137-feature fit 完成、estimator probe 取得方向性訊號，兩者的完整執行都已寫成 handoff 待命。接下來一步是**整理工作區並 commit/push**：收本 session 新腳本與兩份 handoff，一併處理 m3-report.md 與 keepalive 的既有改動，刪除三個孤兒 png。137-feature 全 test 與 estimator sweep 依各自 handoff 之後執行。
