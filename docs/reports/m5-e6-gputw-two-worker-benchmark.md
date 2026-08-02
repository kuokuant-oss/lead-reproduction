# M5 E6 — GPUtw 兩 worker 併發 benchmark

## 狀態:NOT_EXECUTED

沒有任何 benchmark 被執行,因為沒有可用的 GPUtw instance。本輪不得自行儲值,
也不得在未有人類明確建立或授權的 instance 前呼叫付費 instance create API。

腳本、manifest、守衛與測試都已就緒,取得 instance 後可直接執行,不需要再寫
程式。

+ `compatibility_results.json` — `NOT_EXECUTED`
+ `single_worker_results.json` — `NOT_EXECUTED`
+ `two_worker_results.json` — `verdict: NOT_MEASURED`

## 為什麼 verdict 不是「依規格推斷」

規格已足以指出 RTX PRO 6000 是唯一值得測的候選(頻寬 2.00×、cores 2.69×),
以及 DGX Spark 在本工作型態上先天不利(頻寬 0.30×、cores 0.69×)。

但**規格只能排除候選,不能確認候選**。兩個 CUDA process 共用一張 GPU 時,
驅動做的是時間切片;aggregate 是否提升取決於單一 worker 是否真的讓 GPU 有
閒置餘裕,而那是實測問題。因此 verdict 維持 `NOT_MEASURED`,不因為推論方向
明確就填進去。

## 已固定的判定門檻

門檻在看到任何結果之前就固定,事後不得調整。

| verdict | 條件 |
|---|---|
| `TWO_WORKERS_BENEFICIAL` | aggregate rows/s ≥ 單 worker 的 **1.60×**,三輪都穩定,無 OOM / swap / non-finite / CUDA error,每個 worker 完成全部 probe,p95 batch latency 無不可接受停滯,sentinel 相容性通過,且每個完成 state 的成本低於單 worker |
| `TWO_WORKERS_MARGINAL` | aggregate speedup ≥ 1.20 且 < 1.60,或速度提升但成本/穩定性優勢不明確 |
| `TWO_WORKERS_HARMFUL` | aggregate speedup < 1.20,或任一 worker 頻繁 stall,或出現記憶體 / CPU / I/O / thermal contention |
| `INCOMPATIBLE` | sentinel 相容性未通過 —— 直接標記,不得進入兩 worker throughput 測試 |

## 執行設計

### 兩個 worker 必須是兩個真正的 process

用 `subprocess` 啟動兩個獨立的 `m5_e6_gputw_bench.py --mode worker`,各自持有
自己的 CUDA context、自己 reload 的 state、自己的 scaler 物件、自己的輸出檔。

刻意**不用** `ProcessPoolExecutor`:pool 會把例外吞掉並在背後重用 worker,那
正是本專案先前被咬過的地方,而且「兩個 worker」必須是可從 OS 層面查證的事實,
不是程式庫的內部安排。

刻意**不用** threads:同一個 process 內的兩條 thread 共用 CUDA context,量到
的不是本稽核要問的東西。

編排器在收到兩份輸出後,會檢查兩個 worker 回報的 `process_uuid` 不同 ——
若相同就直接失敗,因為那代表跑的不是兩個 process。

### 三輪對調

| Round | worker 0 | worker 1 |
|---|---|---|
| A | `seed42__cell00__cell_specific` | `seed42__cell01__cell_specific` |
| B | `seed42__cell01__cell_specific` | `seed42__cell00__cell_specific` |
| C | `seed42__cell11__frozen_reference` | `seed42__cell00__cell_specific` |

A 與 B 互換,所以「哪個 state 分到哪個 worker slot」不會被誤讀成效能差異。
C 換一個 scaler arm,確認 frozen_reference 路徑同樣穩定。

verdict 取三輪中**最差**的 speedup,不取平均 —— 併發的價值必須在最差情況下
仍然成立。

### 先單 worker,完全退出,再併發

單 worker 基準跑完後必須完全退出 process 並釋放 GPU,才進入兩 worker 測試。
否則第二階段量到的是殘留 context 的影響。

speedup 的分母不是「單 worker 吞吐 × 2」,而是同樣兩個 state **依序**單獨執行
的等效合計吞吐(調和平均),因為兩個 state 的單 worker 速度可能不同。

## 不可協商的執行約束

+ 一個 state 由一個 worker 完整跑完,**不得**把一個 state 拆給兩個 worker
+ 兩個 worker **不得**寫同一個輸出
+ **不得**共用同一個 model object
+ **不得**用 threads 假裝兩個 worker
+ **不得**使用現行 E6 的 full holdout
+ **不得**啟動第三個 worker
+ microbatch 上限維持 **20,000**,不測較大的 microbatch 作為正式推薦

最後一條需要說明理由:正式 E6 已凍結 516-microbatch 的 canonical batched
pass。若 GPUtw 後續分擔 state,必須保持相同的批次語義 —— 用更大的 batch 量出
更漂亮的數字,等於偷偷換掉推論過程。

## 相容性 sentinel(throughput 測試的前置關卡)

每個候選設備在 throughput 測試前,先對三個代表 state 各跑 352 列 sentinel,
每個 state fresh process、同 process 8 次重複。

**不要求跨 GPU bit-exact。** E3/E4/E5 都已確認同一 fitted state 上的重複推論
本來就不是 bitwise 可重現的(E4 192/192、E5 192/192 皆為相異 digest)。

但必須與 E4/E5 已知範圍比較:

+ 重複推論應有合理變異(預期 8/8 相異 digest)
+ 不得出現大量 non-finite
+ 不得出現方向反轉
+ 不得出現明顯超出 E4/E5 範圍的系統性位移
  (E5 co-primary half-width:AUC 0.000415–0.009790、margin 0.000334–0.006440)
+ `effective_n_estimators_` 必須是 8

任一項失敗 → 該設備直接標記 `INCOMPATIBLE`,不進入兩 worker 測試。

## 輸入的 non-holdout 保證

benchmark 只讀既有的 200,000 列 even-building probe,
digest `afe80b11…b46279`。

交集證明在**筆電端**完成:`m5_e6_gputw_guard.py` 用完整的
`np.intersect1d`(不是抽樣)證明 probe 與 10,137,155 列 holdout 的
raw_index 交集為 0,並把結果寫進 `probe_guard.json`。

GPUtw 端只驗證 probe 檔的 digest 與 guard 記錄相符,就繼承了整個證明。
**holdout 的 raw_index 清單因此完全不必上傳** —— 這比「上傳後再檢查」更強:
不存在的東西不會被誤用。

## 會記錄的量測項目

單 worker:reload 時間、scaler 時間、first-batch 時間、每 batch 時間、
median / p05 / p95 rows/s、sustained rows/s、aggregate rows/s、GPU 使用率、
peak VRAM 或 unified memory、peak RSS(讀 `VmHWM`,不取樣)、swap、
non-finite、wall-clock、推估 10,137,155 列 state 時間、推估 8 / 24 state 時間。

兩 worker:另加每 worker 相對單 worker 的 slowdown、aggregate speedup、
p95 batch latency、worker starvation、記憶體壓力、儲存爭用、thermal / power
throttling、兩個 process 是否都穩定完成。

peak RSS 一律讀 `/proc/self/status` 的 `VmHWM`,不用取樣 —— 取樣只會低估,
而且永遠往同一個方向低估,那正好是會讓人誤判「塞得下第二個 worker」的方向。

## Artifacts

+ `data/processed/m5_e6_gputw_audit/compatibility_results.json`
+ `data/processed/m5_e6_gputw_audit/single_worker_results.json`
+ `data/processed/m5_e6_gputw_audit/two_worker_results.json`
