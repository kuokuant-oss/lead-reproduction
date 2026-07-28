# 2026-07-26 TabPFN 17-feature 全 holdout（n=8）完成 handoff

承接 `docs/handoffs/2026-07-26-tabpfn-17feature-full-holdout-prep.md` 與
`docs/reference/m5-tabpfn-17feature-full-holdout-runbook.md`。

## 1. 一句話

17-feature TabPFN（n=8）已跑完全部 10,137,155 列並通過合併驗證，M3 四張圖的橘線已換成 n=8；

## 2. 結果

| 線 | pooled ROC-AUC | pooled PR-AUC |
|---|---:|---:|
| 17-feature 樹 ensemble（gray） | 0.9663 | 0.8221 |
| 137-feature 樹 ensemble（blue） | 0.9918 | 0.9303 |
| TabPFN 17-feature **n=1**（舊橘線，已汰換） | 0.9120 | 0.6639 |
| **TabPFN 17-feature n=8（新橘線）** | **0.9163** | **0.6944** |
| TabPFN 137-feature n=8（violet） | 0.9919 | 0.9314 |

兩條 TabPFN 線同為 n=8、同一 100k context、同一組列，差異只在特徵集：

- estimator 1 → 8（固定 17 features）：+0.0043 ROC / +0.0305 PR
- 17 → 137 features（固定 n=8）：+0.0756 ROC / +0.2370 PR

## 3. 產物

- 合併輸出（gitignored）：`data/processed/m5_tabpfn_17_full_test_n8_predictions.npz`
  （10,137,155 列、637,397 anomalies、9,445,245 個相異分數、分數 finite [1.30e-5, 0.99999]）。
- 四張更新圖（橘線改為 n=8，圖例標示 `n=8`）：
  `m3_feature_engineering_{roc,precision_recall}_with_tabpfn.png`、
  `m3_tree_ensemble_by_site_{roc,precision_recall}_with_tabpfn.png`
- **新增** TabPFN-only 特徵對照圖：
  `m3_tabpfn_feature_contribution_{roc,precision_recall}.png`
  （只有兩條 TabPFN 線，版型沿用 feature-engineering 圖，副標改為 `TabPFN on the same ...`）
- **新增** 137-feature TabPFN 混淆矩陣：`m3_tabpfn_137_confusion_matrix.png`
  （原生 argmax 0.5，與樹 ensemble 那張同門檻可並排）

## 4. 執行實績

| 項目 | 值 |
|---|---|
| 待跑列數 | 7,401,924（sites 0、4–15） |
| shard / chunk | 12 shard、373 chunk，全部 durable |
| wall clock | 約 **2.9 小時**（1785 秒序：07392 → 07839） |
| 實測吞吐 | 單卡 ~400–432 rows/s，雙卡合計 ~834 rows/s |
| 部署耗時 | 每 shard 214–364 秒（39–44 MB 免分段上傳） |
| 批次交接 | 5 次，**全部同一秒完成**（`complete` 與 `launched` 時間戳相同） |
| 告警 | **0**（`ALERT` 全程未出現） |
| 收尾 | 兩帳號 0 session、0 孤兒 assignment、0 殘留行程 |

估計 2.6–3.0 小時，實際 2.9 小時，落在區間內。

## 5. 監控：修掉四個會靜默失效的問題

跑之前的監控有四個缺陷，全部會在「看起來正常」的情況下失效。

1. **reaper 的 tracked 清單寫死**成 137 線的 session。它對 `lead-tabpfn-*` 內不在清單上的
   session **一律 stop**，所以過期清單不是漏收，而是**開跑數分鐘內把整批新 session 殺光**。
   已改為從 plan 推導（`-Plan/-ShardRootTemplate/-SessionTemplate`）。
2. **keep-alive monitor 沒有退出條件**。sync 會在完成時停 session，但 keep-alive 會永遠
   每輪 shell 進 WSL 戳一個不存在的 session；12 個 shard 會累積 12 個殭屍 poller。
   已加「本 shard durable 完成即退出」——實測跑到一半時 keepalive 只剩 2 個而非 8 個，修正生效。
3. **log 檔會被讀取者鎖死。** `Add-Content` 每次取獨佔 handle，所以 `tail -F` / `Get-Content -Wait`
   會讓**寫入端**開始失敗。實際發生：pool log 凍結 31 分鐘，看起來像排程器死了、batch 0→1 沒交棒；
   實際上 pool 全程正常、交接是同一秒完成的。真相在 stdout 鏡像裡（沒人鎖它）。
   已改為 `FileShare::ReadWrite` 常駐 `StreamWriter`（pool 與 progress monitor 皆是）。
4. **sync monitor 在 session 被別人停掉後永遠不退出。** 迴圈先呼叫遠端 `ls`，完成檢查在 try 區塊
   最後；session 一旦消失，`ls` 拋例外 → catch → sleep → 重來，**永遠走不到完成檢查**。
   reaper 搶先停 session 時就會觸發（本次 b3/head、b4/tail 各中一次，留下兩個永久行程）。
   已把完成檢查移到迴圈開頭、先於任何遠端呼叫。

另外新增 `scripts/monitor_m5_tabpfn_run_progress.ps1`：唯一回答「整個 run 還在不在動、還要多久」
的監控。10 秒輪詢，每個 chunk 落地就印 `chunk b3/head 10/30 +70s median=35s`，
每 60 秒印一行含 `quiet Ns` 的心跳，停滯門檻以該 shard 自己的 chunk 間隔中位數自我校準
（`max(300s, 4×median)`），並涵蓋「被啟動但一個 chunk 都沒產出」的交接盲點（`ALERT no_first_chunk`）。

**兩條操作守則**：每個背景程序都要有 stdout 鏡像（log 會被鎖，鏡像不會，診斷先看鏡像）；
停掉 tail 要確認底層 `tail.exe` 真的死了（殺 wrapper 不會殺它，孤兒會繼續持鎖）。

## 7. Decision

17-feature TabPFN n=8 全 holdout 完成、驗證通過、四圖已換線，並新增 TabPFN-only 特徵對照圖
與 137-feature TabPFN 混淆矩陣。

執行面 5 次批次交接零延遲、全程零告警、收尾零殘留。監控層修掉四個靜默失效點
（reaper 誤殺、log 被讀者鎖死、sync 無法退出、背景程序未重導輸出）。
