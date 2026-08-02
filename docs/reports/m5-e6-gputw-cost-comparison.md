# M5 E6 — GPUtw 成本與完成時間比較

## 狀態:`PRICE_UNVERIFIED` + `THROUGHPUT_UNMEASURED`

成本模型的結構完整,但**所有金額欄位是 `null`,所有 GPUtw 時間欄位也是
`null`**,因為兩個必要輸入都還沒有:

| 輸入 | 狀態 | 為什麼不填 |
|---|---|---|
| GPUtw 時租 | 未取得 | 價格由前端 JavaScript 執行時載入,靜態抓取取不到。本輪明令必須用當下 API 或控制台實際價格,不得只用文件快照 |
| GPUtw 實測吞吐 | 未量測 | 沒有 instance,benchmark 未執行 |

用一個猜來的時租算出「省下多少錢」,比不算更糟 —— 它會產生一個看起來可以
決策、實際上沒有依據的數字。

## 唯一有把握的一邊:現行 gpu-host

| 項目 | 值 |
|---|---|
| 實測吞吐 | 1,420 rows/s |
| 單 state | 1.983 h |
| 剩餘 24 個 state | **47.6 h ≈ 1.98 天** |
| 邊際租金 | **NT$ 0** |

最後一行是整個成本比較的關鍵不對稱:**現行 gpu-host 是自有設備,續跑的邊際
租金為零。** GPUtw 要划算,必須用節省的時間換得足以抵過租金與 setup 的價值,
而不是只要「比較快」就成立。

而基準線本身只有 1.98 天。這讓可節省的絕對時間上限就很有限。

## 一次性 overhead(會計入每個候選)

| 項目 | 小時 |
|---|---|
| instance 啟動 | 0.10 |
| 環境建置 | 1.00 |
| 8 個 state artifact 傳輸 | 0.30 |
| probe 傳輸 | 0.10 |
| sentinel | 0.05 |
| preflight | 0.10 |
| archive 與下載 | 0.25 |
| **合計** | **1.90 h** |

DGX Spark 的「環境建置 1.00 h」很可能嚴重低估:若 aarch64 上必須改用 NGC
容器或自行編譯 torch,這一項會從小時級變成天級,而且結果是一個**與現行 E6
不同的 runtime**。

## 損益兩平

模型計算 `breakeven_state_count = setup_overhead / (baseline_state_hours −
device_state_hours)`。

+ 若設備比現行 gpu-host **慢**,breakeven 直接回傳 `"never"` —— 這正是 DGX
  Spark 依規格預期會落入的情況(頻寬 0.30×)。
+ 若設備更快,breakeven 是一個正數,代表至少要跑幾個 state 才值得。

以 RTX PRO 6000 為例,**假設**它達到現行的 2 倍(頻寬比暗示的上限,實際未測),
單 state 會是 0.99 h,每個 state 省 0.99 h,則 1.90 h 的 setup 需要約 **1.9 個
state** 才能打平。8 個 state 的 block 在時間上是划算的 —— 但這個「假設」正是
必須實測而非推論的部分,所以模型不把它寫進 artifact。

## Full feature matrix:建議重建,不建議傳輸

本輪**未實際傳輸** 5.56 GB 的 full feature matrix,只估算策略。

| 策略 | 評估 |
|---|---|
| 傳輸 5.56 GB | 若連線無法續傳,是高風險路徑 |
| **在設備上重建** | **建議** |

依據:gpu-host 上重建的 hoist 階段約 246 s、寫入約 10 s,而且與筆電產生
**逐位元相同**的 digest(`21d6b987…e44e41`,已於本次 E6 部署中實證)。若
GPUtw 端能重建出同一個 digest,就完全不必傳 5.56 GB。

但 DGX Spark 為 aarch64,重建是否得到同一個 digest **未經證實**,必須實測。
pandas / numpy 在不同架構上的浮點路徑若有任何差異,digest 就會不同 —— 那時
傳輸就變成唯一選項,而 5.56 GB 的傳輸風險又回來了。

## 傳輸量

| 項目 | GB |
|---|---|
| full feature matrix | 5.56 |
| probe matrix | 0.111 |
| 單一 state artifact | ~0.35 |
| 8 個 state | ~2.8 |

## 模型會計算什麼(取得價格與吞吐之後)

對每個候選 × 每個 worker 配置:

+ 單 worker 推估 state 小時數
+ 兩 worker 同時完成兩個 state 的 wall-clock
+ 8-state seed block 完成時間(單 / 雙 worker)
+ 16-state 完成時間
+ 租金(含 setup overhead)
+ break-even state count
+ 與讓現行 gpu-host 獨自跑完相比節省幾小時
+ 從「現在」開始的預估最終完成時間

## 解除封鎖需要什麼

1. **價格**:人類提供 GPUtw 控制台當下顯示的 NT$/hr(兩個型號各一)、最小
   計費單位、instance 啟停規則。或提供可讀取價格的 API token —— token **不得**
   commit 或寫入報告。
2. **吞吐**:人類手動建立一個 GPUtw instance 並提供連線資訊,才能執行
   sentinel、單 worker 與兩 worker benchmark。

兩者都拿到之後,`m5_e6_gputw_cost.py` 只要帶上參數即可產出完整成本模型,
不需要改程式。

## 一個必須先看清楚的時間現實

現行 gpu-host 獨自跑完只需 **1.98 天**,而 seed999 的「完整未開始」窗口只有
**11.9 小時**。取得 instance、建環境、跑 sentinel、跑單 worker、跑三輪兩
worker benchmark,再決定是否轉移 —— 這串前置作業本身就可能超過 11.9 小時。

也就是說,即使 RTX PRO 6000 的 benchmark 結果很好,**窗口也可能在結果出來
之前就關閉**。這不是反對做這次稽核(稽核結果對未來的 stage 仍然有效),
但它是決定「這一輪要不要真的動用 GPUtw」時最該先看的數字。

## Artifacts

`data/processed/m5_e6_gputw_audit/cost_model.json`
sha256 `0af19439cbbfa0b54b82f567e53c6fe0d6c192b417cb071cb6381edbfa9ae050`
