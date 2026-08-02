# M5 E6 — GPUtw RTX PRO 6000 benchmark 就緒(PHASE A 完成)

PHASE A 全部完成,尚未發生任何 GPUtw SSH 連線,尚未產生任何費用。
現行 E6 完全未受影響。

## 準備 commit

分支 `m5-e6-gputw-concurrency-audit`,worktree
`../lead-reproduction-e6-gputw-probe`,base lineage 含 E6 protocol commit
`3a800a0`。未 push、未 merge、未開 PR。

## Bundle

| 項目 | 值 |
|---|---|
| 路徑 | `~/outputs/m5-e6-gputw-probe/m5-e6-gputw-probe-bundle.tar.zst` |
| SHA-256 | `1d6def7f6c3b01ee0582f20507eef18a479839e45eae11178818dd33a424838e` |
| 大小 | 106.3 MB |
| 檔案數 | 20 |
| 壓縮 | zstd |

**可重複建立已實測**:連續兩次以相同輸入重建,archive digest 完全相同
(`1d6def7f…`)。tar entry 的 mtime/uid/gid/mode 都固定,所以差異不會來自
封裝時間。

### 內容

三個代表 state 與其 context、scaler、200,000 列 non-holdout probe、352 列
sentinel、四支遠端腳本、no-fit guard、以及五份 manifest。

**刻意排除**:10,137,155 列 full feature matrix、full-holdout raw_index 清單、
full-holdout score、現行 E6 partial outputs、tree full-test outputs、
credentials。holdout 的列**不存在於 bundle 裡**,所以遠端不可能誤用 —— 這比
「上傳後再檢查」強。

## 輸入來源:本機重建,未動用 gpu-host

probe 與三個 context matrix 全部在本機從既有 manifest 與原始資料重算,
**完全沒有從正在執行 E6 的 gpu-host 讀取任何東西**。

驗證結果:

+ probe `raw_index` SHA-256 = `d4c6e4e76246a99b7a52268f11f1f4f6f78f85b25fb3b01bb6776caa9fc71e86`,
  與既有 audit artifact 記錄的完整值**完全相符**
+ probe 與 holdout 的 raw_index 交集 = **0**(用完整 `np.intersect1d`,非抽樣)
+ 200,000 列**全為 even-building**
+ hoist 建構與真正的 `build_feature_matrix` 在 400 列樣本上 **bit-exact**
  (`max|diff| = 0.0`,NaN pattern 相同)
+ 三個 state digest 相符、scaler 全部 exact

### 一個必須說明的 digest 差異

重建的 `.npz` **檔案** digest 與既有 artifact 記錄的
`a3cfd7cf…70a6b` **不同**,而陣列內容相同。原因已查明並實測確認:

`np.savez` 產生 zip,zip entry 記錄寫入平台(`create_system`:Windows/FAT = 0、
Unix = 3)。既有 artifact 在 Linux 寫,本機重建在 Windows 寫,所以那個 byte
不同 → 檔案 digest 不同 → 但沒有任何一個陣列受影響。

對照組證實了這個解釋:E6 的 full feature matrix 用 `open_memmap` 寫 `.npy`
(**沒有** zip 容器),兩台機器獨立重建得到**完全相同**的 digest。

因此 manifest 的內容權威值改為 **array-level digest**
(`x_sha256` + `raw_index_sha256`),而不是 `.npz` 檔案 digest。preflight 與
worker 都改用 array-level digest 驗內容,檔案 digest 只用來確認「遠端拿到的
就是本機送出的那一份」。用檔案 digest 做跨平台比對會把一份正確的輸入誤判成
錯的。

`x` 內容 digest = `1e200f3af98edc22c01930ddf941b22f22f738cf9f82d8c253fe699619fb97c8`

## 測試覆蓋

**40 項全部通過。** 覆蓋 prompt §八 列出的每一項:

+ bundle 不含 full holdout、odd-building 列、score 欄位、SSH key
+ probe 與 holdout 交集為 0
+ no-fit guard 在實際呼叫 `fit` 時拋出 `FitAttemptedError`
+ state digest 驗證、`effective n_estimators = 8`、scaler exact、float32 guard
+ checkpoint identity、20,000 列 microbatch 固定
+ single/dual worker schema、worker output roots 互斥
+ 兩 worker 不得寫同一路徑、同一 state 不得拆給兩 worker、不得第三 worker
+ 不得使用 full-holdout artifact、不得修改正式 E6 worktree
+ credential leak test、remote command dry run
+ collector 可拒絕 partial 或偽造輸出(短跑、竄改吞吐、偽造 speedup、
  重複 digest、同一 process 假裝兩 worker、宣告第三 worker)

collector 的驗證不是只比 digest:它從 `per_batch` **重新推導**每一個彙總值
(rows/s、sustained、speedup、投影時間)再比對,所以一份自己給自己蓋章的
結果不會通過。

## 需要人類提供的四個值

```
GPUTW_HOST      # 例:xxx.gputw.ai 或 IP
GPUTW_USER      # 遠端使用者名稱
GPUTW_PORT      # SSH port
GPUTW_SSH_KEY   # 本機 private key 檔案路徑,權限必須 0600
```

若 GPUtw 給的是一整條 SSH 指令,請提供該指令,我會安全解析出這四個值,
**不會把指令本身 commit**。

private key **不會**被複製進 repository、bundle、JSON、報告或 log。
host key 存進未追蹤的 `~/.gputw-probe-runtime/known_hosts`,已加入
`.gitignore`。連線一律 `BatchMode=yes`、`IdentitiesOnly=yes`、
`StrictHostKeyChecking=yes`,第一次連線會先顯示 fingerprint。

## 精確啟動命令

```bash
export GPUTW_HOST=...            # 由人類提供
export GPUTW_USER=...
export GPUTW_PORT=...
export GPUTW_SSH_KEY=/path/to/key
export GPUTW_BUNDLE="$HOME/outputs/m5-e6-gputw-probe/m5-e6-gputw-probe-bundle.tar.zst"
export GPUTW_BUNDLE_SHA256=1d6def7f6c3b01ee0582f20507eef18a479839e45eae11178818dd33a424838e
export GPUTW_LOCAL_OUT="$HOME/outputs/m5-e6-gputw-probe/results"

bash scripts/m5_e6_gputw_launch.sh
```

狀態:`bash scripts/m5_e6_gputw_status.sh`
中止:`bash scripts/m5_e6_gputw_abort.sh`

## 預估時間

| 階段 | 預估 |
|---|---|
| 上傳 106.3 MB bundle | 3–15 分鐘(視頻寬) |
| 建立固定環境(venv + 釘版套件 + torch cu130) | 10–25 分鐘 |
| preflight(含三個 state reload) | 2–4 分鐘 |
| compatibility sentinel(3 states × 8 repeats) | 1–2 分鐘 |
| 單 worker(3 states × 200,000 列) | 依吞吐,約 5–15 分鐘 |
| 雙 worker(3 輪 × 2 workers) | 約 10–25 分鐘 |
| **remote setup 硬性上限** | **45 分鐘** |
| **benchmark 總時間硬性上限** | **90 分鐘** |

## 硬性停止條件

+ remote setup 超過 45 分鐘未完成 → 停止
+ compatibility sentinel 失敗 → 立即停止,不進入 throughput 測試
+ benchmark 總時間超過 90 分鐘 → 停止
+ 任何 OOM、swap 持續增長、CUDA error、non-finite → 停止
+ 任一輪兩 worker 僵死超過 10 分鐘 → 終止該輪並停止
+ GPU 不是 RTX PRO 6000 → 立即停止
+ 不執行任何 full-holdout scoring、不建立正式 override、不接手任何正式 state

benchmark 結束後**不會**自行刪除或停止付費 instance —— 除非人類另有明確指示。
會明確提醒可關閉 instance 以停止計費。

## 現行 E6 狀態(只讀 heartbeat)

```
phase=running  states=0/24  quar=0  intr=0  procs=1  tmp=0
pf=ok  errn=0  ens_bad=0  gpu=96%  swap=0
```

正在執行 position 0(`seed42__cell11__frozen_reference`)。

**確認未觸碰現行 E6**:未 stop/kill/pause/attach `tmux e6`;未修改遠端 clone、
`.venv`、outputs 或 monitor;未在 gpu-host 啟動第二 worker 或執行 benchmark;
未讀取任何 scientific score;未從 gpu-host 讀取或打包任何資料。E6 worktree
仍在 `3a800a0`。

## 尚未開始的完整 seed block(供 PHASE B 後參考)

E4 的隨機化執行順序把三個 seed 交錯,窗口持續收斂:

| seed | positions | 狀態 |
|---|---|---|
| 42 | 0,1,10,11,14,15,20,21 | 已開始 |
| 123 | 2,3,4,5,12,13,16,17 | 完整未開始 |
| 999 | 6,7,8,9,18,19,22,23 | 完整未開始 |

## 尚未發生

+ GPUtw SSH 連線:**0 次**
+ GPUtw predictions:**0**
+ fits_performed:**0**
+ 費用:**NT$ 0**
