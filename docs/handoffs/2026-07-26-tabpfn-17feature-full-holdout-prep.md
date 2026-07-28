# 2026-07-26 TabPFN 17-feature 全 holdout（n=8）準備完成 handoff

這份 handoff 記錄「把 M3 四張圖的橘色 17-feature TabPFN 線換成 n_estimators=8 全 holdout 版」
的**準備階段**。承接 `docs/reference/m5-tabpfn-137feature-full-holdout-runbook.md`、
`docs/handoffs/2026-07-25-tabpfn-137feature-full-holdout-complete.md` 與
`docs/handoffs/2026-07-24-tabpfn-17feature-estimator-sweep-plan.md`。

新的 reference：`docs/reference/m5-tabpfn-17feature-full-holdout-runbook.md`。

## 1. 一句話

Site 0 與 Site 4–15（7,401,924 列）的 17-feature n=8 推論**所有準備已完成並通過驗證**，
**尚未啟動任何 session**；跑完後併回 10,137,155 列即可換掉四張圖的橘線。

## 2. 為什麼做這件事：移除一個 confound

四張圖現在的橘線是 `n_estimators=1`，紫線的 137-feature TabPFN 是 `n_estimators=8`，
兩者同時差在特徵數（17→137）與 estimator 數（1→8）。本次把橘線也換成 n=8，
兩條 TabPFN 線就只差在特徵集合。

estimator sweep 的實測：Site 1 改善、Site 2 的 ROC 與 PR 同時退步、
Site 3 的 PR-AUC 由 0.167 升至 0.783 而 ROC 幾乎不動。

## 3. 已完成的準備

| 項目 | 產物 | 驗證 |
|---|---|---|
| 批次計畫 | `data/processed/m5_tabpfn_17_remaining_batch_plan.json` | 與 137 計畫 diff 後**批次幾何逐欄相同**（只差吞吐估算） |
| 12 個 shard | `data/processed/m5_tabpfn_f17_batch{0..5}_context100000_n8/{head,tail}/` | 見 §4 |
| pool 排程器 | `scripts/run_m5_tabpfn_shard_pool.ps1` | 以 stub launcher 實跑過完整排程序列 |
| 合併器 | `scripts/merge_m5_tabpfn_full_test.py`（改名 + `--line`） | 以 `--line 137` 回歸重現既有產物數字 |
| 繪圖接線 | `plot_m3_figures.py`、`plot_m3_tree_ensemble_by_site.py` | 見 §6 |

Site 1/2/3 的 17-feature n=8 六個 shard 已確認 durable 完整（2,735,231 列、六個 `result.json` 齊全），
所以合併只需要新跑的 13 個 site。2,735,231 + 7,401,924 = 10,137,155。

## 4. 匯出的驗證

新的 `scripts/export_m5_tabpfn_17_batch_shards.py` 用 137 線**同一組批次切點**，
從 M3 frame 取列、以 n=8 fit 的 context scaler 轉換（17-feature 沒有預先算好的全 test 矩陣可切，
但 frame 只載入一次就產完 12 個 shard）。

匯出後的獨立交叉驗證全部通過：

- 12 個 shard 合計 **7,401,924 列 / 514,681 anomalies**，與計畫相符。
- 12 個 shard 的 `raw_index_sha256` 與 `label_sha256` **全部等於 137 線同名 shard 的值**（12/12）。
  也就是說兩條線的每個 shard 覆蓋**完全相同的列、相同順序**。
- raw_index 無重複；與 Site 1/2/3 的聯集**恰好等於** 10,137,155 列的 holdout。
- feature 檔 **39–44 MB**（137 線是 310–350 MB），全部低於 64 MB 上傳門檻，**不需分段上傳**。

## 5. pool 排程器終於進版控

137 線的 handoff 說「本次改用 2-slot 貪婪 pool 排程」，但**那個排程器從未被 commit** ——
repo 裡只有被放棄的 batch-barrier runner（`run_m5_tabpfn_137_batches.ps1`，其 log 顯示只推進到 batch 1）。
實際完成 137 線的排法留在操作者的手動指令裡。

`scripts/run_m5_tabpfn_shard_pool.ps1` 把它固化：2-slot 貪婪，任一 slot 的 shard durable 完成就立刻接下一個；
以 plan / shard-root template / session template 參數化，兩條線共用。

對卡住的處理是**只報不修**：有 live supervisor 的 shard 不得手動重部署（137 線就是在這裡產出過錯批分數）。
唯一自動重排的情況是 queue script 等不到 A100 而放棄 —— 那時該 shard 沒有東西在跑，slot 是真的空的。

測試方式：把 `Start-Process` 換成 stub、其餘排程邏輯原封不動，跑一段「3 個已完成 + 3 個待跑」的序列，
確認它跳過已完成者、填滿兩個 slot、在同一輪回收完成者並立刻補位、佇列與 active 皆空時收工。

## 6. 圖的替換接線

兩支繪圖腳本的 `DEFAULT_TABPFN_PREDICTIONS` 已指向
`m5_tabpfn_17_full_test_n8_predictions.npz`，兩條 TabPFN 線的圖例都加上 `n=8`，明示受控比較。

- `plot_m3_tree_ensemble_by_site.py` 在檔案不存在時直接 FileNotFoundError（大聲失敗，正確）。
- `plot_m3_figures.py` 保留「檔案不存在就只畫兩線圖」的既有行為（fresh clone 需要它，npz 是 gitignored），
  但**新增 WARNING**：靜默略過會讓 `*_with_tabpfn.png` 留著上一次的內容，看起來和成功算圖一模一樣。

by-site 副標的三個 pooled 數字（gray 0.9663 / blue 0.9918 / violet 0.9919）**不受本次影響**，橘線不在副標裡。
跑完後若要把橘線也列入副標，用實測值補，不要事先填。

## 7. 待執行（尚未啟動）

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_m5_tabpfn_shard_pool.ps1
uv run python scripts/merge_m5_tabpfn_full_test.py --line 17
uv run python scripts/plot_m3_figures.py
uv run python scripts/plot_m3_tree_ensemble_by_site.py
~~~

排程器預設值就是這條線，不需參數。帳號 `tonykuo210100@gmail.com`（HOME `/home/tonykuo/.colab-tony`，
必帶 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`）、2 × A100。

預期 wall clock **約 2.6–3.0 小時**（7,401,924 列 ÷ ~430 rows/s ÷ 2 卡 ≈ 2.4 h 純推論，
加上明顯小於 137 線的部署開銷）。

## 8. 已知缺口

- **repo 沒有安裝 pytest**（`pyproject.toml` 的 dev group 只有 pre-commit 與 ruff），
  因此 `tests/` 未執行。已逐一確認受影響的兩個測試檔都不觸及本次改名的合併器或繪圖預設值：
  `test_tabpfn_portable_shard.py` 載入的是 `merge_m5_tabpfn_distributed_predictions`（另一支）。
  `pre-commit run`（含 ruff、ruff format、markdownlint）全數通過。
- 本次未開 GitHub issue、未 commit，依 `docs/reference/change-checklist.md` 這兩項在收斂時補。

## 9. Decision

準備階段完成且經獨立驗證，**未啟動任何 session、未消耗任何 CU**。
批次幾何與 137 線逐位相同，raw_index/label digest 證明兩條線覆蓋完全相同的列。
本次改動是把橘線的 n_estimators 由 1 換成 8；
