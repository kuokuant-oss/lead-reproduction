# M5 TabPFN dual-L4 Colab runbook

這份 reference 給 Codex、其他 AI agent、維運腳本與未來接手者使用。它描述目前正式的 TabPFN 全 test split 推論，不是新的實驗設計。

## 1. 不可改動的正式契約

以下任一項改變，都會使輸出不能直接和既有 M3 曲線或其他 shard 合併：

- 模型：同一份已 fit 的 TabPFN fitted state；不可重新 fit。
- context：100000 rows。
- features：固定 17 個 baseline features；不可啟動 137-feature 實驗。
- test split：canonical test rows 共 10,137,155，必須維持原始 row order、labels、site、building identity。
- checkpoint：每 20,000 rows，一個 atomic .npz。
- microbatch：正式 worker 初始 1024，資源壓力時自動往下減，最低 64；不可用 CPU/TPU fallback。
- worker：不設模型 wall-time timeout。管理用 CLI timeout 只防止 upload/exec 控制程序卡死，不是模型執行期限。
- 本機正式 worker 必須保持停止；不要執行 scripts/run_m5_tabpfn_canonical_full_test.py。
- Colab 只允許兩個命名 session，不得建立第三個：
  - lead-tabpfn-tail
  - lead-tabpfn-tail-2

## 2. shard 與邊界

| shard | session | remote root | global rows | local rows | direction | expected chunks |
|---|---|---|---:|---:|---|---:|
| head | lead-tabpfn-tail-2 | /content/lead_tabpfn_head | [0, 5,060,000) | 5,060,000 | forward | 253 |
| tail | lead-tabpfn-tail | /content/lead_tabpfn_tail | [5,060,000, 10,137,155) | 5,077,155 | reverse | 254 |

head 最後一個 chunk 是 rows_005040000_005060000.npz。tail 從 canonical row 5,060,000 開始，最後一個 reverse chunk 是 rows_10120000_10137155.npz。邊界必須落在 20K checkpoint boundary，不能跨 boundary。

## 3. 本機資料與輸出位置

根目錄：

~~~text
C:\Users\tonykuo\projects\lead-reproduction
~~~

主要資料樹：

~~~text
data/processed/
  m5_tabpfn_canonical_full_test_context100000.work/
    model.tabpfn_fit       # 原始 fitted state
    scaler.joblib           # 與 fitted state 對應的 scaler
    fit_manifest.json       # context/hash/model contract
  m5_tabpfn_distributed_context100000/
    tail/                    # tail portable inputs + manifest
    head/                    # head portable inputs + manifest
    tail-results/
      chunks/rows_*.npz      # tail 本機 durable predictions
      progress.json
      heartbeat.json
      sync.log
      worker.log
      result.json             # shard 完成時才會有
    head-results/
      chunks/rows_*.npz      # head 本機 durable predictions
      progress.json
      heartbeat.json
      sync.log
      worker.log
      result.json
~~~

每個 checkpoint .npz 至少包含 raw_index、anomaly、score、site_id、building_id。寫入使用 temporary file + atomic rename；中斷時不能刪除完整 chunks，也不能以 partial file 覆蓋完整檔案。

## 4. 一次性準備 portable inputs

### 4.1 Tail

scripts/export_m5_tabpfn_colab_tail.py 從 canonical M3 frame 與已保存 fitted state 建立 tail portable shard。它會：

1. 讀 canonical row identity、labels、site、building。
2. 取 global [5,060,000, 10,137,155)。
3. 依 baseline 17 features 與 scaler.joblib 產生 float32 memmap。
4. 將 fitted archive 的 remote foundation model path 改為 /content/lead_tabpfn_tail/tabpfn-v3-classifier-v3_default.ckpt。
5. 寫 metadata.npz、model.portable.tabpfn_fit、manifest.json。
6. 對 features、metadata、fit state、foundation checkpoint 寫 SHA-256。

典型指令（只有在需要重建 inputs 時執行；不在模型運作中重跑）：

~~~powershell
.venv\Scripts\python.exe scripts\export_m5_tabpfn_colab_tail.py --shard tail --global-start 5060000 --global-end 10137155 --direction reverse --out-dir data\processed\m5_tabpfn_distributed_context100000\tail --remote-root /content/lead_tabpfn_tail
~~~

### 4.2 Head

scripts/prepare_m5_tabpfn_colab_head.py 將已驗證的 head checkpoints 轉成 portable rows_*.npz，並嚴格比對 raw_index、y/anomaly、site_id、building_id。它不重新預測。

~~~powershell
.venv\Scripts\python.exe scripts\prepare_m5_tabpfn_colab_head.py
~~~

### 4.3 大檔案切片

.scratch/split_large_file.ps1 預設每片 64 MiB。切片只為 Colab upload，不改變內容；原始 hash 必須在遠端 reassemble 後再次相同。

~~~powershell
powershell -NoProfile -File .scratch\split_large_file.ps1 -InputPath data\processed\m5_tabpfn_distributed_context100000\tail\features.float32.npy -OutputDirectory data\processed\m5_tabpfn_upload_parts
~~~

目前 upload parts 位於：

- data/processed/m5_tabpfn_upload_parts/：tail foundation checkpoint parts 與 tail features parts。
- data/processed/m5_tabpfn_head_upload_parts/：head features parts。

不要手動修改 parts、manifest 或 fitted archive；hash mismatch 時應停止該 recovery episode。

## 5. Colab 建立與部署流程

### 5.1 帳戶、硬體與 session

Windows launcher 設定：

~~~text
TABPFN_COLAB_HOME=/home/tonykuo/.colab-hank
TABPFN_COLAB_AUTH=oauth2
TABPFN_COLAB_ACCELERATOR=L4
~~~

這是 Colab CLI backend 認證，不等於已附加使用者 Chrome 分頁。CLI 目前只指定 accelerator，不能由這套 managed Colab CLI 指定 region/zone。

建立 session 的低階 helper 是 scripts/create_colab_session.py；它只接受 session、GPU、auth，並輸出 JSON 成功/錯誤資訊。正式情況不要直接建立第三個 session。

### 5.2 遠端目錄與上傳

supervisor 的 restore_colab_files() 依序執行：

1. create_*_dirs.py：重建 /content/lead_tabpfn_{head,tail}/work/chunks。
2. 上傳固定輸入：features parts、foundation parts、metadata.npz、model.portable.tabpfn_fit、manifest.json。
3. 上傳 scripts/run_m5_tabpfn_portable_shard.py。
4. 上傳所有本機 structurally valid、已完成的 rows_*.npz，使重建從 durable frontier 繼續。
5. 執行 reassemble_*_colab.py，重組並驗證四個 SHA-256：features、metadata、portable fit、foundation checkpoint。
6. 執行：
   - .scratch/install_colab_tabpfn.py：tabpfn==8.0.8
   - .scratch/install_colab_exact_runtime.py：scikit-learn==1.8.0、numpy==2.4.6、psutil==7.2.2
   - .scratch/inspect_colab_python_deps.py（只讀版本診斷；失敗會記錄 warning，不阻斷已完成安裝與恢復）

`inspect_colab_python_deps.py` 不是模型步驟，也不會改變 fitted state。若 Colab 的
remote exec 暫時回傳 503、session transport 短暫抖動或管理命令逾時，supervisor 會記錄
`remote_exec_failed=true`（含 return code 與去敏感化摘要），並把此診斷步驟降級為
`runtime_inspect_warning=true`；只有安裝腳本失敗才會阻止該次 restore。這避免因診斷命令
瞬斷而重傳所有 checkpoint。
7. 執行對應 worker launcher。

任何新 backend/session 都必須重做整套上傳；只拿到新 endpoint 不是恢復成功。

## 6. Remote worker 參數與可續跑行為

scripts/run_m5_tabpfn_portable_shard.py 是自包含、可上傳的正式 worker。它只在 worker entrypoint 內 import torch/TabPFN，並要求 CUDA 可用。

實際 launch command：

~~~text
python run_m5_tabpfn_portable_shard.py \
  --features /content/lead_tabpfn_{head,tail}/features.float32.npy \
  --metadata /content/lead_tabpfn_{head,tail}/metadata.npz \
  --fit-state /content/lead_tabpfn_{head,tail}/model.portable.tabpfn_fit \
  --work-dir /content/lead_tabpfn_{head,tail}/work \
  --context-rows 100000 \
  --query-microbatch-size 1024 \
  --min-query-microbatch-size 64 \
  --checkpoint-rows 20000 \
  --direction forward|reverse \
  --resume
~~~

resume 會逐一檢查既有 chunk 的 row identity 與 finite scores；已完成 chunk 直接跳過，只有缺少的 chunk 才重新 predict。每個 20K checkpoint 完成後立即寫入 progress.json，並釋放暫存 numpy/torch cache。遇到 GPU/RAM soft limit 時 microbatch 減半；到 hard limit 且已達 minimum 時才報錯。沒有 model wall-time timeout。

## 7. Supervisor、同步與 keep-alive

### 7.1 永久 supervisor

兩個入口：

~~~powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\ensure_m5_colab_recovery_supervisor.ps1
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\ensure_m5_colab_head_recovery_supervisor.ps1
~~~

對應 Windows tasks：

- CodexTabPFNColabRecoverySupervisor
- CodexTabPFNColabHeadRecoverySupervisor

它們使用 singleton lock：

- data/processed/m5_tabpfn_recovery_supervisor.lock
- data/processed/m5_tabpfn_colab_head_recovery_supervisor.lock

健康條件不是 PID alone，而是 worker heartbeat/checkpoint 持續前進。若 sync log 的最後有效事件顯示 remote work 消失，supervisor 會釋放該 shard 的 exact endpoint、建立同名新 L4、完整重傳、驗證 hash、啟動 --resume。舊 heartbeat 不能掩蓋最新的 remote-work failure。

### 7.2 Sync monitor

- tail：scripts/sync_m5_tabpfn_colab_tail.ps1
- head wrapper：scripts/sync_m5_tabpfn_colab_head.ps1

它每 60 秒列遠端 chunks，逐檔 atomic download 到本機，並同步 heartbeat.json、progress.json、result.json、worker.log、launcher.json。下載完整 expected chunks 且有 result.json 後，會停止該 shard 的 exact Colab session，釋放 GPU/CU。

### 7.3 keep-alive 與 45 分鐘 work touch

- tail：scripts/monitor_m5_tabpfn_colab_keepalive.ps1
- head wrapper：scripts/monitor_m5_tabpfn_colab_head_keepalive.ps1

monitor 會維持 Colab CLI keep-alive process；另外每 2700 秒（45 分鐘）對自己的 /content/.../work 做 read-only ls touch：

- tail：/content/lead_tabpfn_tail/work
- head：/content/lead_tabpfn_head/work

touch 不 import TabPFN、不 predict、不配置本機 GPU。成功會在 keep-alive log 出現 work_touch=true。這不是 Chrome 滑鼠 click；目前 Codex 沒有可附加使用者 Chrome 分頁的 controller，因此採用 backend work-root touch + CLI keep-alive。

## 8. 進度判讀與成功條件

本機最可信的是：

~~~powershell
Get-Content data\processed\m5_tabpfn_distributed_context100000\tail-results\progress.json
Get-Content data\processed\m5_tabpfn_distributed_context100000\head-results\progress.json
~~~

同時計算完整 chunks：

~~~powershell
(Get-ChildItem data\processed\m5_tabpfn_distributed_context100000\tail-results\chunks\rows_*.npz | Measure-Object).Count
(Get-ChildItem data\processed\m5_tabpfn_distributed_context100000\head-results\chunks\rows_*.npz | Measure-Object).Count
~~~

heartbeat_rows、session allocation、remote PID、colab_recovery_verified 都只能證明遠端 worker 活著，不能單獨宣稱 recovery success。recovery episode 成功必須同時滿足：

~~~text
current durable completed_rows > episode baseline_completed_rows
AND
current structurally valid chunk count > episode baseline_valid_chunks
~~~

Episode state：

- data/processed/m5_tabpfn_recovery_episode.json
- data/processed/m5_tabpfn_colab_head_recovery_episode.json

完整 shard 成功則必須看到 expected chunk count + result.json；此時 sync monitor 停 session，不能因 session 消失又重新建立。

## 9. 合併與下游輸出

兩個 shard 完成後，用 scripts/merge_m5_tabpfn_distributed_predictions.py 做 exact row identity checks，再產生：

~~~text
data/processed/m5_tabpfn_distributed_context100000_predictions.npz
~~~

合併器會對 canonical artifact 的每個 20K span 檢查 raw_index、anomaly、site_id、building_id；任何缺 chunk、跨 boundary、row drift 或 non-finite score 都會停止，不產生可信輸出。

~~~powershell
.venv\Scripts\python.exe scripts\merge_m5_tabpfn_distributed_predictions.py
~~~

只有合併成功後，才可將 tabpfn scores 交給報表/ROC/PR 產生器。不要用尚未完整合併的部分 chunks 畫全 test 曲線，也不要把 head/tail 的 local row index 當成 global row index。

## 10. 故障處理手冊

### A. Service Unavailable、no T4、ADC refresh

這些是可重試的 allocation/transport failure。保持本機 chunks，不刪 checkpoint；supervisor 會以 jittered exponential backoff 重試。若已有 stale named assignment，先釋放該 exact endpoint，再用同名 session 建新 L4。不能把 retrying 報成成功，也不能建立第三個 session。

### B. session 存在但 remote work/chunks 不見

backend work ID 不可假設可重連。依 sync.log 最新有效事件判斷；強制釋放該 shard assignment，建立同名新 session，重新建目錄、上傳固定輸入與所有 valid local chunks，再 SHA 驗證與 resume。

### C. upload 卡住或沒有錯誤細節

查看：

~~~text
data/processed/m5_tabpfn_recovery_status.json
data/processed/m5_tabpfn_colab_head_recovery_status.json
data/processed/m5_tabpfn_recovery_supervisor.log
data/processed/m5_tabpfn_colab_head_recovery_supervisor.log
~~~

phase 應沿著 restore_start → uploading → restore_sha256_verified → restore_complete → launch_start → health samples。只看到 colab_session_ready 不代表有重傳。

### D. heartbeat 有增加但本機 rows 沒增加

這不是 durable success。檢查 sync monitor PID、sync.log 最新行、本機 chunks count。不要刪本機 chunk，也不要用 remote heartbeat 取代本機 durable rows。

### E. 記憶體壓力/OOM

worker 會先降低 query microbatch，執行 gc.collect() 與 torch.cuda.empty_cache()；checkpoint 仍是 20K。不要為了短期速度修改 context、features、checkpoint boundary 或改用 CPU。

### F. 本機誤啟動

立即停止該本機 formal worker，保留所有已完成 checkpoints，並確認沒有 run_m5_tabpfn_canonical_full_test.py process。Colab recovery supervisor 必須是 --scope colab，不能使用 --scope local 或 --scope both。

## 11. AI agent 接手前檢查清單

1. 讀本文件與兩個 manifest；不要從 log 猜 boundary。
2. 確認兩個 session 名稱與兩個 lock owner，確認沒有第三個 formal session。
3. 讀兩個 progress.json、chunks count、episode baseline。
4. 確認 local formal process 為零；不要 import torch/TabPFN 做 health check。
5. 先執行兩個 exact ensure_*.ps1 entry points；受限 shell 遇 Task Scheduler Access Denied 時使用 host/escalated PowerShell。
6. 若健康，只報 durable progress；不要重啟、不重傳、不重新 fit。
7. 若 remote work 消失，保留 local chunks，依本文件完整重建；新 endpoint 本身不算成功。
8. 只有本機 rows/chunks 增加，或完整 shard 有 result.json + expected chunks，才能宣稱成功。
9. 全部完成後先 merge，再產生下游圖表；完成 shard 的 GPU session 必須被停止以免繼續消耗 CU。

## 12. 現況快照（寫作時）

這份 reference 寫作時兩個 shard 皆為 status=predicting，本機 durable outputs 分別位於 tail-results 與 head-results；實際數字以當下 progress.json 為準，不要把本節數字當成固定基線。

## 13. 2026-07-24 最新暫停快照與恢復規則

本次要求「跑完最後一個正在處理的 chunk 後暫停」。已先等待兩邊各自完成當前 in-flight chunk（head 到第 217、tail 到第 215）、確認本機 durable checkpoint 已落地，再依序停止兩個 supervisor（含 disable scheduled task）、sync monitor、keep-alive 與兩個精確的 Colab session；沒有在 predict 中途殺掉 worker。停止後已驗證 `colab sessions` 回報 `No active sessions`，兩張 L4 GPU 已釋放、CU 停止消耗。

（前一次暫停快照為 2026-07-23，durable 為 3,037,155 rows / 各 76 chunks；本節已更新為最新一次。）

### 13.1 暫停時的 durable 狀態

| shard | session / direction | durable rows | chunks | shard 範圍 | 狀態 |
|---|---|---:|---:|---|---|
| head | `lead-tabpfn-tail-2` / forward | 4,340,000 | 217 / 253 | `[0, 5,060,000)` | 已暫停 |
| tail | `lead-tabpfn-tail` / reverse | 4,297,155 | 215 / 254 | `[5,060,000, 10,137,155)` | 已暫停 |

兩邊合計已保存 8,637,155 / 10,137,155 rows（約 85.20%）。head 最新 durable chunk 為 `rows_04320000_04340000.npz`，forward frontier 在 canonical row 4,340,000，剩餘 `[4,340,000, 5,060,000)`（720,000 rows / 36 chunks）。tail 為 reverse（從 10,137,155 往下），已完成 canonical row 5,840,000 以上到頂，剩餘 `[5,060,000, 5,840,000)`（780,000 rows / 39 chunks）。兩邊皆尚無 `result.json`，屬未完成之刻意暫停。所有已完成 `.npz` chunks、`progress.json`、heartbeat 與 episode/manifest 檔案都保留在：

~~~text
data/processed/m5_tabpfn_distributed_context100000/head-results/
data/processed/m5_tabpfn_distributed_context100000/tail-results/
~~~

本機 formal worker process 為 0；這是刻意暫停，不代表資料遺失或需要重新 fit。恢復時各 shard 只需再跑約 36–39 個 chunk。

### 13.2 暫停與恢復操作順序

- 暫停前必須先確認兩個 `progress.json` 的 `completed_rows` 已跨過目前 chunk 邊界，且對應 chunk 已在本機；不能只看 remote heartbeat 或 PID。
- 暫停時兩個 scheduled task 已被 `Disable-ScheduledTask` 停用；恢復前必須先 `Enable-ScheduledTask`（或直接執行 ensure 腳本，它會重註冊），否則 supervisor 不會被排程帶起。
- 暫停後不要執行 `ensure_m5_colab_recovery_supervisor.ps1` 或 `ensure_m5_colab_head_recovery_supervisor.ps1`，否則 supervisor 會依設計嘗試恢復 GPU。
- 恢復時才用 host/escalated PowerShell 執行上述兩個 exact entry points；它們會重新使用有效 local chunks、驗證四個 SHA-256、上傳固定輸入並以既有 fitted state `--resume`，不 refit、不建立第三個 session。
- 恢復成功的判準仍是本機同步的 `completed_rows` 與 structurally valid chunk count 都高於該 episode 的 persisted baseline；新 session、PID、upload 完成或遠端 heartbeat 單獨都不算成功。
- 恢復重啟 supervisor（含長跑中途換新程式碼的重啟）前，判斷「worker 是否還在遠端產出」不能只看 `sync.log` 末行或本機 `heartbeat.json` 新鮮度——被回收的 session 會留下假新鮮 heartbeat，且 `sync.log` 的 `checkpoints=` 記的是本機數量、會蓋掉稍早的 `not found`。應改看 supervisor log 末尾是否已進入 `remote_work_missing_from_sync` / `restore_start`。

### 13.3 本版 supervisor / setup 注意事項

- `inspect_colab_python_deps.py` 是 best-effort 診斷；其 setup failure 會記錄 warning，不會阻止已驗證的 restore/resume 流程。
- 若 remote work ID 被 Google 回收，不能嘗試 reconnect 舊 ID；必須釋放該 shard 的 stale assignment、建立同名新 L4 session、重建 remote work/chunks、重傳所有本機有效 chunks 與固定輸入、完成四項 SHA-256 檢查後再 resume。
- 此帳號（`hank0503work@gmail.com`）的 L4 runtime 生命週期上限實測約 51–54 分鐘，之後 Google 會回收 remote work；這是預期行為，supervisor 會自動以同名 session 重建並 `--resume`。單一 shard 全程約經歷多次此類 recycle，屬正常。
- `rebuild_once_or_accept_durable_progress()` 會在「重建健康抽樣失敗但本機 durable rows/chunks 已前進」時結束該 episode，避免把已保存的進度誤判成永遠失敗。此版另加一道 90 秒 re-check：`launch_and_verify_colab()` 的 `second_health_sample` 常因該帳號 `colab exec` 瞬時逾時而失敗，但 worker 仍健康；接受條件不變（durable rows 與 valid chunk count 都須高於 baseline），只是在丟棄 runtime 前多等一個 sync 週期讓 in-flight chunk 落地。
- `_colab()` 的本機 subprocess timeout 已與傳給遠端的 `--timeout` 預算對齊（`_remote_exec`＝setup+120s、`_upload`＝900s、`_inspect_colab`＝120+120s）。舊版固定 180s 會在 64 MiB features/foundation part 上傳或 pip 安裝偶爾超時時，把健康步驟誤判成 `returncode=124` 失敗並觸發整輪重傳。
- 相關 recovery supervisor 測試目前全數通過（33 tests）。
- keep-alive 與 45 分鐘 work-root touch 只在正式運行時維持；完整 shard 出現 `result.json` 與所有 expected chunks 後，必須先停止該 shard 的 GPU session，避免繼續消耗 CU。
