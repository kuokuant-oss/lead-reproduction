# 2026-07-25 TabPFN estimator sweep 與 137-feature 結果 handoff

這份 handoff 記錄 2026-07-24 深夜至 2026-07-25 的完整執行：17-feature estimator sweep、
137-feature（n=8）三個 site 完成、以及補滿整個 holdout 的批次作業啟動。
寫法沿用 `docs/reference/m5-tabpfn-colab-dual-shard-runbook.md`。

## 1. 一句話

在四張 A100 上跑完 12 格全列推論（17-feature 的 3 site × n∈{4,8}，137-feature 的 3 site × n=8），
並把補滿剩餘 7,401,924 列的批次作業備妥啟動。

## 2. 17-feature estimator sweep

全列量測、零抽樣、共 2,735,231 列。詳見
`docs/handoffs/2026-07-24-tabpfn-17feature-estimator-sweep-plan.md` §7.2。

| site | prevalence | ROC: n=1 → 4 → 8 | PR: n=1 → 4 → 8 |
|---|---:|---|---|
| Site 1 | 13.502% | 0.5447 → 0.6248 → 0.6647 | 0.2026 → 0.2458 → 0.2640 |
| Site 2 | 6.401% | 0.8435 → 0.8144 → 0.8177 | 0.2807 → 0.2262 → **0.2101** |
| Site 3 | 0.227% | 0.9762 → 0.9787 → 0.9800 | 0.1667 → **0.7900** → 0.7828 |

n=1 → n=8 的 PR 變化橫跨 −0.07（Site 2）到 +0.62（Site 3）。Site 2 的 PR 單調下滑，
n=8 未回升。三個 site 的 n=4 → n=8 變化都小於 n=1 → n=4。

## 3. 137-feature（n=8）三個 site

| site | 17f ROC → 137f | 17f PR → 137f | 樹 ensemble 137f |
|---|---|---|---|
| Site 1 | 0.6647 → **0.9972** | 0.2640 → **0.9886** | 0.997 / 0.986 |
| Site 2 | 0.8177 → **0.9910** | 0.2101 → **0.9015** | 0.991 / 0.900 |
| Site 3 | 0.9800 → **0.9987** | 0.7828 → **0.8586** | 0.999 / 0.886 |

Site 2 在 17 features 下 n=1→n=8 的 PR 由 0.2807 降至 0.2101；換 137 features 後為 0.9015。

## 4. 已備妥但**尚未啟動**：補滿整個 holdout

**狀態：準備工作 100% 完成，未啟動任何 Colab session。啟動與否由使用者決定。**

- 待跑 **7,401,924 列 / 514,681 anomalies**（sites 0、4–15），依 building_id 切成 6 批、12 個 shard，
  已排除完成的 Site 1/2/3。
- 12 個 shard **已全部匯出並通過驗證**（列數、anomaly 數、building 範圍逐項比對計畫，
  且確認不含已完成的 site），合計正好 7,401,924 列。
- 全部由既有的全 test 137 矩陣切片，與 M3 圖上三條線是**同一組 holdout 列**，可比性成立。
- 預定在 `tonykuo210100@gmail.com` 的兩張 A100 上依序執行（一次一批、head/tail 並行）。
- 完整說明見 `docs/reference/m5-tabpfn-137feature-full-holdout-runbook.md`。

**估時**：實測 A100 上 137 features / n=8 / microbatch 20000 為 **~330 rows/s**
（六個已完成 shard 落在 329–333，變異極小）。

| 項目 | 值 |
|---|---:|
| 純推論 | 6.2 GPU-hours |
| 兩卡並行 wall clock | 3.1 小時 |
| 每批部署開銷 × 6 | 36–60 分鐘 |
| **合計預期** | **約 3.7–4.2 小時** |

啟動指令：`powershell -File scripts\run_m5_tabpfn_137_batches.ps1`。
收工後以 `merge_m5_tabpfn_137_full_test.py` 併回 10,137,155 列，通過四關驗證後才能重繪 M3 四圖。

**流程備註**：本次曾誤將「準備完成」當成「可以開跑」而啟動了 batch 0，
約兩分鐘後（仍在上傳階段、未產出任何 chunk）依使用者指正停止並釋放兩張 A100，
`colab sessions` 已確認回到 `No active sessions`。準備與啟動應分開，啟動由使用者下令。

## 5. 效能與運維發現（已寫入 runbook §1.2）

1. **microbatch 是最大槓桿且幾乎不吃記憶體**：A100 上 1024 → 16384/20000 約 10 倍吞吐，
   torch 峰值保留僅由 3,578 MiB 升到 3,608 MiB（卡有 40 GB），因為記憶體由 100k context 主導。
   原計畫以 L4 估的 ~103 小時因此縮到 1–2 小時等級。
2. **n=8 不是 n=4 的兩倍成本**（~430 vs ~493 rows/s，17 features），瓶頸在 context 編碼而非 ensemble。
3. **137 features 吞吐穩定在 ~330 rows/s**（六個 shard 落在 329–333，變異極小）。
4. **單次上傳超過 64 MB 會失敗**：79 MB 檔在兩台不同 VM 上都失敗（500 → 400），
   切成 64 MB 分段後零重試成功。分段門檻已對齊 64 MB。

## 6. 修掉的缺陷（每一個都曾造成實際故障）

| 缺陷 | 後果 | 修法 |
|---|---|---|
| `--resume` 覆蓋命令列 microbatch | 以 16384 重啟卻仍跑 1024，慢 10 倍 | 命令列非預設值時以命令列為準 |
| remote root 未含 estimator / feature 數 | `--resume` 會把別條線的 checkpoint 當成已完成，**產出錯誤分數卻標對名稱** | root 加入 `_n<k>` / `_f<feat>` |
| 只掛 sync + keep-alive、沒有 supervisor | 掉線的 shard 靜止不動，assignment 空燒 CU | 每個 shard 三件齊備 |
| keep-alive 掛在上傳之後 | 上傳期間 session 被回收，部署仍「成功」，留下沒有 worker 的卡 | 固化順序：配置 → keep-alive → 部署 → 監控 |
| 以 session 名稱釋放 assignment | CLI 對失效 session 會自刪本機記錄，此後名稱式 stop 永遠失敗 | 改用 `state.client.unassign('<endpoint>')` |
| 孤兒判定過於激進 | **誤殺正在註冊的兄弟 session** | 連續多輪確認仍無名才回收 |
| launcher 寫死 `--n-features 17` | 137 那格啟動即被 worker 拒絕 | 從 manifest 讀取 |
| 背景程序未重導輸出 | 失敗完全靜默，「正在部署」與「已死」無法區分 | 一律寫 log 檔 |
| process 過濾未排除自己 | `Stop-Process` 殺掉自己的 shell | 加 `$_.ProcessId -ne $PID` |
| 上傳無重試 | 一次 500 就中止整段部署 | 5 次退避重試 |

## 7. 新增腳本

| 檔案 | 用途 |
|---|---|
| `fit_m5_tabpfn_17_context100000.py` | 17-feature 的 per-estimator fit（驗證 scaler 與正式 run 相同） |
| `export_m5_tabpfn_site_shards.py` | per-site portable shard（支援 reuse 與 slice） |
| `export_m5_tabpfn_137_batch_shards.py` | 依 building_id 批次切片匯出 |
| `plan_m5_tabpfn_137_remaining_batches.py` | 批次規劃與時間估算 |
| `evaluate_m5_tabpfn_site_sweep.py` | 單格評估，強制四項對齊證明 |
| `aggregate_m5_tabpfn_sweep_table.py` | n=1 → 4 → 8 增益表（基準不一致就拒絕產表） |
| `merge_m5_tabpfn_137_full_test.py` | 合併全 holdout 並驗證身分 |
| `queue_m5_tabpfn_site_shard.ps1` | 單 shard 上線，固化正確順序 |
| `deploy_m5_tabpfn_site_shard.ps1` | 通用部署（分段上傳、重試、診斷降級） |
| `supervise_m5_tabpfn_site_shard.ps1` | 自癒 supervisor（endpoint 釋放、孤兒確認） |
| `run_m5_tabpfn_137_batches.ps1` | 批次依序推進 |
| `reap_idle_m5_tabpfn_colab_sessions.ps1` | 閒置 session 自動關閉 |
| `calibrate_m5_tabpfn_microbatch.py` | microbatch 校準（detached、增量寫檔） |
| `verify_m5_tabpfn_137_shards.py` | 137 shard 結構驗證與本機 smoke |
| `build_m5_tabpfn_site_colab_scripts.py` | 由 manifest 產生遠端腳本 |

## 8. 待辦

1. 批次作業跑完 → 合併 → 重繪 M3 四圖（兩支繪圖腳本目前只吃單一 `--tabpfn-predictions`，
   需擴成可接受第二條 137-feature TabPFN 線）。
2. 既有三個測試失敗（`test_m5_tabpfn_recovery_supervisor`、`test_plot_m6_seen_vs_unseen_tabpfn`）
   為本分支既有問題，與本次改動無關（已用 `git diff HEAD` 確認相關檔案未被碰過），尚未處理。

## 9. Decision

17-feature estimator sweep 與 137-feature 三個 site 皆已完成。
補滿剩餘 7,401,924 列的批次作業**已完成全部準備但尚未啟動**，預期 wall clock 約 4 小時，等使用者下令再開跑。
所有踩過的坑都已寫入 runbook §1.2 與新的 137-feature runbook，避免重蹈。
