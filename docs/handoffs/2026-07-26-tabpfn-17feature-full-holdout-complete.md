# 2026-07-26 TabPFN 17-feature 全 holdout（n=8）完成 handoff

承接 `docs/handoffs/2026-07-26-tabpfn-17feature-full-holdout-prep.md` 與
`docs/reference/m5-tabpfn-17feature-full-holdout-runbook.md`。

## 1. 一句話

17-feature TabPFN（n=8）已跑完全部 10,137,155 列並通過合併驗證，M3 四張圖的橘線已換成 n=8；
**estimator confound 移除後，「TabPFN 落後主要來自特徵不足」的歸因被證實而非推翻**。

## 2. 結果

| 線 | pooled ROC-AUC | pooled PR-AUC |
|---|---:|---:|
| 17-feature 樹 ensemble（gray） | 0.9663 | 0.8221 |
| 137-feature 樹 ensemble（blue） | 0.9918 | 0.9303 |
| TabPFN 17-feature **n=1**（舊橘線，已汰換） | 0.9120 | 0.6639 |
| **TabPFN 17-feature n=8（新橘線）** | **0.9163** | **0.6944** |
| TabPFN 137-feature n=8（violet） | 0.9919 | 0.9314 |

**歸因拆解（這是本次的核心產出）**：

- estimator 1 → 8（固定 17 features）：**+0.0043 ROC / +0.0305 PR**
- 17 → 137 features（固定 n=8）：**+0.0756 ROC / +0.2370 PR**

特徵集的貢獻約為 estimator 數的 **8 倍**（PR-AUC 上）。先前圖上橘（n=1）對紫（n=8）的落差
同時混了兩個變因，現在兩條 TabPFN 線都是 n=8、同一 100k context、同一組列，
唯一差異就是特徵集，歸因才成立。

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

**這一段是本次最有價值的部分**：跑之前的監控有四個缺陷，全部會在「看起來正常」的情況下失效。

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

## 6. 兩個分析上的釐清

**(a) AUC 打平不代表混淆矩陣會一樣。** 137-feature TabPFN 與 137-feature 樹 ensemble 的
AUC 幾乎相同（ROC 0.9919 vs 0.9918、PR 0.9314 vs 0.9303），但在 0.5 門檻下 TabPFN 的 FN
多 8,638 個。查證後確認**不是校準問題**（等警報預算下仍多 7,989 個 FN），而是**曲線形狀不同、中段交叉**：

| Recall | Tree Precision | TabPFN Precision | 勝方 |
|---:|---:|---:|:--|
| 0.50 / 0.60 | 0.9962 / 0.9777 | 0.9929 / 0.9766 | Tree |
| **0.70 – 0.85** | 0.9260 → 0.8240 | **0.9433 → 0.8308** | **TabPFN** |
| 0.90 – 0.96 | 0.7916 → 0.6920 | 0.7788 → 0.6522 | Tree |

0.5 門檻落在 recall ≈ 0.935，正好在樹模型佔優的高 recall 區。兩者優勢區互相抵消，
所以面積幾乎相等。**實務意涵**：警報預算在 recall 0.7–0.85 時 TabPFN 較好
（recall 0.75 少 12,140 個誤報）；要追到 recall 0.95 則樹模型明顯較省誤報。

**(b) 不要為了讓圖「符合預期」去調門檻。** 過程中一度把混淆矩陣改成 prevalence 對齊、
再改成 F1 最佳，想讓 TabPFN 看起來配得上它較高的 AUC——這是錯的：AUC 差 0.0001–0.0011
本來就是打平，打平的模型在任一工作點互有小幅高低是正常的；而且只調一方、另一方留在 0.5，
是拿兩套規則比較。最終回到原生 argmax 0.5，與樹 ensemble 同門檻。

## 7. Decision

17-feature TabPFN n=8 全 holdout 完成、驗證通過、四圖已換線，並新增 TabPFN-only 特徵對照圖
與 137-feature TabPFN 混淆矩陣。**特徵工程的貢獻是 estimator 數的約 8 倍**，
137-feature handoff 的歸因在移除 confound 後成立。

執行面 5 次批次交接零延遲、全程零告警、收尾零殘留。監控層修掉四個靜默失效點，
其中三個（reaper 誤殺、log 被讀者鎖死、sync 無法退出）都會讓「看起來正常」的 run 實際出錯，
下一次 run 直接受益。
