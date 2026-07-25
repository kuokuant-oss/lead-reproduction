# M5 TabPFN 17-feature 全 holdout（n_estimators=8）批次 runbook

這份 reference 描述**如何把 17-feature TabPFN（n=8）的分數補滿整個 50/50 building holdout**，
用來**取代** M3 四張圖上那條橘色的 17-feature TabPFN 線。給接手者、維運腳本與未來的 AI agent。

搭配閱讀：`m5-tabpfn-137feature-full-holdout-runbook.md`（批次切法與 pool 排程完全沿用它）、
`m5-tabpfn-colab-dual-shard-runbook.md`（§1.2 A100 實戰教訓）、
`docs/handoffs/2026-07-24-tabpfn-17feature-estimator-sweep-plan.md`（Site 1/2/3 的 n=4/8 結果）。

**狀態：所有準備工作已完成，尚未啟動任何 session。**

## 1. 為什麼要跑（這不只是「換個更好的數字」）

四張目標圖現在的橘線來自 `m5_tabpfn_distributed_context100000_predictions.npz`，那是
**`n_estimators=1`**；紫線的 137-feature TabPFN 是 **`n_estimators=8`**。

也就是說，圖上被當成「特徵不足」證據的橘 vs 紫落差，其實**同時混了兩個變因**：特徵數 17→137，
以及 estimator 1→8。137-feature handoff 的結論「TabPFN 先前的落後主要來自特徵不足」因此
**目前是被 confound 的**，不是乾淨的歸因。

本次把橘線也換成 n=8，兩條 TabPFN 線就只差在特徵集合，該歸因才成立（或被推翻）。

**不要預期橘線一定會變好。** estimator sweep 已證明 n 的效果**沒有單一方向**：

| site | n=1 → n=8 ROC | n=1 → n=8 PR |
|---|---|---|
| Site 1 | 0.5447 → 0.6647（+0.120） | 0.2026 → 0.2640（+0.061） |
| Site 2 | 0.8435 → 0.8177（**−0.026**） | 0.2807 → 0.2101（**−0.071**） |
| Site 3 | 0.9762 → 0.9800（+0.004） | 0.1667 → 0.7828（**+0.616**） |

pooled 的方向要跑完才知道，**不得事先在文件或圖說裡預告方向**。

## 2. 目標與可比性契約

四張目標圖：

- `docs/reports/assets/m3/m3_tree_ensemble_by_site_roc_with_tabpfn.png`
- `docs/reports/assets/m3/m3_tree_ensemble_by_site_precision_recall_with_tabpfn.png`
- `docs/reports/assets/m3/m3_feature_engineering_roc_with_tabpfn.png`
- `docs/reports/assets/m3/m3_feature_engineering_precision_recall_with_tabpfn.png`

評測母體是 `building_id % 2 == 1` 的 50/50 building holdout，共 **10,137,155 列**，
列序為 canonical `m5_tabpfn_distributed_context100000_predictions.npz`（依 building_id 遞增）。

不可改動：100k context（`context_sha256 = e9ffe0cf…d2688cbe`，與 137 線、與 n=1 正式 run 同一組）、
同一 foundation checkpoint（SHA-256 `d0d865d5…ea18f3988`）、17 個 `BASELINE_FEATURE_COLS`、
`n_estimators=8`、20,000-row checkpoint、microbatch 20,000、不可 CPU/TPU fallback。

## 3. 已完成與待跑的範圍

| 範圍 | 列數 | 狀態 |
|---|---:|---|
| Site 1 / 2 / 3 | 2,735,231 | ✅ 已完成（estimator sweep 的 n=8 格，6 shard 全 durable） |
| 其餘 13 個 site（0, 4–15） | **7,401,924** | 本 runbook 的對象 |
| 合計 | 10,137,155 | |

待跑部分含 514,681 個 anomalies。

## 4. 批次切法：與 137 線逐位相同

`m5_tabpfn_17_remaining_batch_plan.json` 由同一支 `plan_m5_tabpfn_137_remaining_batches.py`
產生（只改 `--out` 與 `--rows-per-second`）。因為輸入（canonical npz、`done_sites = 1,2,3`、
目標 120 萬列）完全相同，**批次幾何與 137 線的計畫逐欄相同**，已用 diff 驗證。

| batch | rows | buildings | sites | head / tail |
|---:|---:|---|---|---|
| 0 | 1,207,548 | 1–723 | 0, 4, 5 | 600,000 / 607,548 |
| 1 | 1,215,971 | 725–901 | 5, 6, 7, 8, 9 | 600,000 / 615,971 |
| 2 | 1,203,218 | 903–1015 | 9, 10 | 600,000 / 603,218 |
| 3 | 1,204,681 | 1017–1171 | 10, 11, 12, 13 | 600,000 / 604,681 |
| 4 | 1,216,672 | 1173–1289 | 13, 14 | 600,000 / 616,672 |
| 5 | 1,353,834 | 1291–1447 | 14, 15 | 680,000 / 673,834 |

沿用同一組切點不只是省事：**兩條 TabPFN 線的每個 shard 覆蓋完全相同的列、相同順序**，
已由 raw_index / label 的 SHA-256 逐 shard 比對證實（12/12 全中，見 §7）。

批次的 canonical 位置**不是連續的**，因為已完成的 Site 1/2/3 依 building_id 穿插其中。

## 5. 執行帳號

| 項目 | 值 |
|---|---|
| Google 帳號 | **`tonykuo210100@gmail.com`**（Colab Pro） |
| WSL HOME | `/home/tonykuo/.colab-tony` |
| 必要環境變數 | `OAUTHLIB_RELAX_TOKEN_SCOPE=1`（缺少時 Google 少給 `drive.file` scope，會在寫檔前中止） |
| GPU | 2 × A100 |

**極易誤讀**：兩個帳號的 HOME 都在 `/home/tonykuo/` 底下 —— `.colab-tony` 是
`tonykuo210100@gmail.com`，`.colab-hank` 是 `hank0503work@gmail.com`。路徑中的 `tonykuo`
是 WSL 使用者名稱，不是帳號。`token.json` 沒有 `id_token`、`account` 為空，**無法從本機檔案驗證信箱**；
行為特徵是 `.colab-tony` 每次呼叫都必須帶 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`。

若跑到一半想借 hank 帳號加開兩張卡，把 `-Batches` 拆兩半、各跑一個 pool 即可（兩個 pool 的
shard 集合不可重疊）。

## 6. 腳本與執行順序

| 步驟 | 腳本 | 狀態 |
|---|---|---|
| 1. 規劃 | `plan_m5_tabpfn_137_remaining_batches.py --out …_17_… --rows-per-second 430` | ✅ 已產出 |
| 2. 匯出 | `export_m5_tabpfn_17_batch_shards.py` | ✅ 12 個 shard 已產出並驗證 |
| 3. 執行 | `run_m5_tabpfn_shard_pool.ps1` | 待跑 |
| 4. 收尾 | `reap_idle_m5_tabpfn_colab_sessions.ps1` | 待跑 |
| 5. 合併 | `merge_m5_tabpfn_full_test.py --line 17` | 待跑 |
| 6. 重繪 | `plot_m3_figures.py` + `plot_m3_tree_ensemble_by_site.py` | 已改好接線，待資料 |

單一 shard 的上線由 `queue_m5_tabpfn_site_shard.ps1` 負責，固定順序為
**取得 A100 → 掛 keep-alive → 部署上傳 → 掛 sync + supervisor**。

### 6.1 執行指令

~~~powershell
# 準備（已完成；重跑匯出需加 --force）
uv run python scripts/plan_m5_tabpfn_137_remaining_batches.py `
  --out data/processed/m5_tabpfn_17_remaining_batch_plan.json --rows-per-second 430
uv run python scripts/export_m5_tabpfn_17_batch_shards.py

# 執行（tonykuo210100@gmail.com / HOME .colab-tony、兩張 A100）
# 三個都要掛，且都要重導 log —— 背景程序不接 log 會讓失敗完全靜默。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_m5_tabpfn_shard_pool.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\monitor_m5_tabpfn_run_progress.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reap_idle_m5_tabpfn_colab_sessions.ps1

# 看進度（唯一一個回答「整體還在不在動」的 log）
Get-Content data\processed\m5_tabpfn_17_run_progress.log -Tail 5

# 收尾
uv run python scripts/merge_m5_tabpfn_full_test.py --line 17
uv run python scripts/plot_m3_figures.py
uv run python scripts/plot_m3_tree_ensemble_by_site.py
~~~

`run_m5_tabpfn_shard_pool.ps1` 的預設值就是這條線（plan、shard root template
`m5_tabpfn_f17_batch{0}_context100000_n8`、session template `lead-tabpfn-b{0}-{1}-f17n8`），
因此不需要任何參數。

### 6.2 監控與 watchdog（開跑時必須全部掛上）

| 層級 | 腳本 | 作用 | 退出條件 |
|---|---|---|---|
| 每 shard | `monitor_m5_tabpfn_colab_keepalive.ps1` | 維持 keep-alive + 每 45 分 work-touch | **本 shard durable 完成**（新增） |
| 每 shard | `sync_m5_tabpfn_colab_tail.ps1` | 逐檔下載 chunk / 狀態檔；完成時 **stop session** | 完成後 exit 0 |
| 每 shard | `supervise_m5_tabpfn_site_shard.ps1` | session 掉了或進度停滯就重建 + `--resume` | 完成後 break |
| 全run | **`monitor_m5_tabpfn_run_progress.ps1`** | **持續看本機 chunk 進度**，算 rate / ETA，停滯就 ALERT | 12 shard 全完成 |
| 全run | `run_m5_tabpfn_shard_pool.ps1` | 2-slot 排程 + 停滯警告 | 佇列與 active 皆空 |
| 全run | `reap_idle_m5_tabpfn_colab_sessions.ps1` | 收掉閒置 / 未追蹤 session 與孤兒 assignment | 12 shard 全完成 |

前三者由 `queue_m5_tabpfn_site_shard.ps1` 自動掛上；後三者要各自啟動（見 §6.1）。

**`monitor_m5_tabpfn_run_progress.ps1` 是唯一回答「整個 run 還在不在動、還要多久」的東西**，
其餘 monitor 都只看單一 shard。它只觀察、不碰 session，因此可以和所有其他 monitor 併行。
判斷依據**只有本機 durable checkpoint**：遠端 heartbeat 與 session 列表在 worker 完全沒產出時
仍然一片健康，那正是它要抓的失效模式。

**輪詢 10 秒**（比照舊的 `monitor_m5_tabpfn_local_head.ps1`）。取樣間隔必須遠短於 chunk 間隔，
否則「剛剛有沒有產出 chunk」這個問題根本問不出來 —— 120 秒輪詢配上以分鐘計的門檻，
就是看得到總數卻看不到脈搏。

三種輸出：

- **`chunk b0/head 7/30 +47s median=45s`** —— 每個 chunk 一落地就印，附上與前一個的間隔與滾動中位數。
  這是「還在不在動」的即時脈搏。
- **`progress 13/373 chunks (3.5%) shards_done=0/12 rate=… eta=… | b0/head=7/30(quiet 0s) …`**
  —— 每 `HeartbeatSeconds`（預設 60 秒）一行。`quiet Ns` 欄位讓這一行**單獨**就能回答
  「此刻有沒有東西在產出」。
- **`ALERT shard_stalled` / `ALERT run_stalled`**。

停滯門檻**自我校準**，不寫死：以該 shard 自己觀測到的 chunk 間隔中位數為基準，
`limit = max(StallFloorSeconds, StallFactor × median)`（預設 300 秒 / 4 倍）。
chunk 間隔是批次與 GPU 的性質，不值得硬編一個數字；floor 則讓剛開跑、只有一個樣本的 shard
不會在正常暖機時誤報。全 run 任何地方都沒有新 chunk ≥ `RunStallMinutes`（預設 20 分）
→ `ALERT run_stalled`，代表 WSL / 認證 / 網路整體出事。

**log 必須能被 tail 而不擋住寫入。** `Add-Content` 每次取獨佔 handle，所以只要有人
`Get-Content -Wait` 或 `tail -F` 這個檔案，**寫入端就會開始失敗**——監控會剛好在有人盯著時
靜默停止記錄。改法：以 `FileShare::ReadWrite` 開一個常駐 `StreamWriter`（`AutoFlush`）。
一個「看了就會壞」的 log 比沒有 log 更糟。

這個坑在 2026-07-26 的正式 run 實際踩到，而且**誤導了診斷**：`run_m5_tabpfn_shard_pool.ps1`
的 log 凍結了 31 分鐘，看起來像排程器死了、batch 0→1 沒交棒；實際上 pool 全程正常，
交棒是**同一秒**完成的（`complete batch=0 shard=head` 與 `launched batch=1 shard=head`
時間戳相同）。真相在 stdout 鏡像檔裡，因為沒人鎖它。

由此得到兩條操作守則：

1. **每個背景程序都要有 stdout 鏡像**（`-RedirectStandardOutput`）。log 檔會被鎖，鏡像不會，
   兩者互為備援。診斷時先看鏡像。
2. **停掉 tail 要確認底層行程真的死了。** 殺掉外層 wrapper 不會殺掉 `tail.exe`；
   本次留下三個孤兒 `tail`，鎖持續存在，讓「我已經解鎖了」這個判斷是錯的。
   用 `Get-CimInstance Win32_Process -Filter "Name='tail.exe'"` 檢查命令列再逐一清掉。

**兩個開跑前修掉的坑（都會靜默毀掉這次 run）**：

1. `reap_idle_m5_tabpfn_colab_sessions.ps1` 的 tracked 清單原本**寫死**成 137 線的 site2/site3
   session。它對 `lead-tabpfn-*` 命名空間內**不在清單上的 session 一律 stop**，所以過期的清單
   不是「沒收乾淨」，而是**開跑幾分鐘內就把整批新 session 殺掉**。已改為從 plan 推導。
2. `monitor_m5_tabpfn_colab_keepalive.ps1` 原本沒有任何退出條件。sync monitor 會在 shard 完成時
   stop session，但 keep-alive 仍會永遠每輪 shell 進 WSL 去戳一個已不存在的 session；
   12 個 shard 依序跑完會累積 12 個殭屍 poller，而其中一個去 touch 一個已被重複使用的 session 名稱
   比單純的噪音更糟。已加上「本 shard durable 完成就退出」。

### 6.3 pool 排程取代 batch barrier

137 線原本用逐 batch 阻塞的 runner，一個 straggler 就讓另一張 A100 空等、把延誤串成整體延誤；
實跑到 batch 1 就被放棄、後面靠手動排。那個手動排法已固化為 `run_m5_tabpfn_shard_pool.ps1`：
**2-slot 貪婪排程，任一 slot 的 shard durable 完成就立刻接下一個**，兩張卡全程滿載到收尾。

排程器對卡住的處理是**只報不修**：有 live supervisor 的 shard 不得手動重部署（137 線就是在這裡
產出過錯批分數）。唯一會自動重排的情況是 queue script 等不到 A100 而放棄 —— 那時該 shard
沒有任何東西在跑，slot 是真的空的。

## 7. 驗證關卡（每一關都會擋下錯誤，不是事後檢查）

1. **匯出時**（已通過）：重算的列集合必須與計畫的列數、anomaly 數、building 範圍相符，
   且不得含已完成的 site；raw_index 必須仍對應到 canonical 的 anomaly / site_id / building_id。
2. **匯出後的交叉驗證**（已通過）：12 個 shard 的 `raw_index_sha256` 與 `label_sha256`
   **全部等於 137 線同名 shard 的值**，且 12 個 shard 的 raw_index 無重複、
   與 Site 1/2/3 的聯集**恰好等於** 10,137,155 列的 holdout。
3. **上傳後**：遠端 reassemble 比對四個 SHA-256（features、metadata、portable fit、foundation checkpoint）。
4. **worker 啟動時**：`--n-features 17`、`--n-estimators 8` 與 fitted state 不符就拒絕啟動。
5. **合併時**：union 必須恰好等於 holdout（無重複、無遺漏），labels/site/building 逐列與 canonical 相符，
   分數全為 finite。

## 8. 圖的替換契約

`plot_m3_figures.py` 與 `plot_m3_tree_ensemble_by_site.py` 的
`DEFAULT_TABPFN_PREDICTIONS` 已改指向 `m5_tabpfn_17_full_test_n8_predictions.npz`，
兩條 TabPFN 線的圖例都加上 `n=8`，明示這是受控比較。

**在合併產物出現之前**：`plot_m3_tree_ensemble_by_site.py` 會直接 FileNotFoundError（大聲失敗，正確）；
`plot_m3_figures.py` 保留既有的「檔案不存在就只畫兩線圖」行為（fresh clone 需要它，npz 是 gitignored），
但現在會印出 WARNING，說明 `*_with_tabpfn.png` 維持前一次的內容 —— 靜默略過看起來和成功算圖一模一樣。

by-site 圖的副標目前是 `Pooled ROC-AUC — gray (0.9663) · blue (0.9918) · violet (0.9919)`，
**三個數字都不受本次影響**（橘線不在副標裡）。跑完後若要把橘線也列入副標，再用實測值補上；
不要事先填。

## 9. 時間估算

17-feature、n=8、microbatch 20000 的實測吞吐是 **413–453 rows/s**（Site 1 head，A100-SXM4-40GB）。
取 430 rows/s：

| 項目 | 值 |
|---|---:|
| 待跑列數 | 7,401,924 |
| 單卡吞吐 | ~430 rows/s |
| 純推論 GPU 時數 | **~4.8 GPU-hours** |
| 兩張 A100 並行的 wall clock | **~2.4 小時** |
| 每批部署額外開銷 | 明顯低於 137 線 |
| **合計預期 wall clock** | **約 2.6–3.0 小時** |

部署開銷比 137 線低很多：17-feature 的 shard features 只有 **39–44 MB**（137 線是 310–350 MB），
全部低於 `deploy_m5_tabpfn_site_shard.ps1` 的 64 MB 門檻，**不需要分段上傳**。

若遇到 session 回收，supervisor 會自動重建並 `--resume`，只損失當前未完成的 20k chunk。

## 10. 注意事項（承接 137 runbook §6）

- **keep-alive 必須在上傳之前掛**，否則上傳期間 session 被回收，部署仍會「成功」跑完，
  留下沒有 worker 的 assignment 空燒 CU。
- **remote root 已含 batch 與 feature 數**（`/content/lead_tabpfn_b<N>_<shard>_f17_n8`），
  與 137 batch（`…_f137_n8`）及 17-feature per-site shard（`…_s<site>_<shard>_n8`）都不撞，
  避免 `--resume` 誤用別條線的 checkpoint。**這條線和 137 線覆蓋完全相同的列，撞目錄的後果是靜默取錯分數。**
- **判斷卡住要看遠端**：本機 chunk 數在部署期間必然是 0，「正在傳」與「已死」無法區分。
  查遠端 `work/` 是否有 `launcher.json` 與 `worker.log`。
- **孤兒 assignment 用 endpoint 回收**（`state.client.unassign('<endpoint>')`）；
  `colab stop -s <name>` 對已失去本機記錄的 session 無效。回收前需連續多輪確認仍為無名。
- **背景程序一律重導輸出到 log 檔**。

## 11. Decision

批次幾何與 137 線逐位相同、每批 head/tail 由 `run_m5_tabpfn_shard_pool.ps1` 的 2-slot 貪婪排程
推上 `tonykuo210100@gmail.com`（HOME `.colab-tony`）的兩張 A100，完成後以
`merge_m5_tabpfn_full_test.py --line 17` 併回 10,137,155 列，通過驗證才重繪 M3 四圖。
預期 wall clock 約 2.6–3.0 小時。

所有準備已就緒且已驗證：計畫 JSON、12 個 shard（含與 137 線逐 shard 的 raw_index/label digest 相符證明）、
pool 排程器、合併器、繪圖接線。**尚未啟動任何 session。**

本次的價值不在「換個更好的數字」，而在**移除橘 vs 紫之間的 estimator confound**：
跑完之後，兩條 TabPFN 線才真的只差在特徵集合，137-feature handoff 的歸因也才有資格被確認或推翻。
