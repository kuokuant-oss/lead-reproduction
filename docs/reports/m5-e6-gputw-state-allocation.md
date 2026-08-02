# M5 E6 — GPUtw state 分配可行性

只依 state identity、seed、cell、scaler arm、凍結執行順序與 monitor heartbeat
的完成計數判斷。**未查看任何 scientific score value。**

## 最重要的一件事:窗口正在關閉

E4 的隨機化執行順序把三個 context seed **交錯**排列。seed block 不是「前 8
個、中 8 個、後 8 個」,而是散布在整個 24 state 的序列裡:

| seed | 在凍結執行順序中的 position | 第一個 position |
|---|---|---|
| 42 | 0, 1, 10, 11, 14, 15, 20, 21 | **0** |
| 123 | 2, 3, 4, 5, 12, 13, 16, 17 | **2** |
| 999 | 6, 7, 8, 9, 18, 19, 22, 23 | **6** |

現行 run 依 position 0 → 23 前進,每個 state 1.99 小時。所以:

| seed | 目前狀態 | 現行 run 還有多久碰到 |
|---|---|---|
| 42 | **已開始**(position 0 正在執行) | 0.0 h |
| 123 | 完整未開始 | **4.0 h** |
| 999 | 完整未開始 | **11.9 h** |

**「完整未開始的 seed block」是一個持續收斂的窗口,不是靜態的事實。**
在 4 小時後 seed123 就不再完整;在 11.9 小時後 seed999 也不再完整。

由於相容性 sentinel、單 worker 基準與兩 worker benchmark 都尚未執行(沒有
instance),而這些前置作業本身需要時間,**這兩個窗口很可能在取得測試結果之前
就已經關閉**。這一點必須先講,再談分配方案。

## 建議

`TRANSFER_ONE_COMPLETE_SEED_BLOCK` — **seed999**

seed999 是仍完整未開始、且現行 run 最晚才會碰到的 block,窗口最寬(11.9
小時)。

八個 state:

```
position  6  seed999__cell10__frozen_reference
position  7  seed999__cell10__cell_specific
position  8  seed999__cell00__frozen_reference
position  9  seed999__cell00__cell_specific
position 18  seed999__cell01__frozen_reference
position 19  seed999__cell01__cell_specific
position 22  seed999__cell11__cell_specific
position 23  seed999__cell11__frozen_reference
```

若要兩個 block,次選是 seed123 —— 但它的窗口只有 4.0 小時,實務上幾乎確定
來不及。

**沒有完整未開始的 seed block 時,規則是停止,不得提出任意 state 切割。**
目前尚有兩個,所以不觸發停止條件。

## 為什麼以完整 seed block 為單位

一個完整 seed block 讓該 seed 內的:

+ 四個 factorial cell
+ 兩個 scaler arm
+ negative-support contrast
+ positive × negative interaction
+ scaler 比較

全部留在同一個執行環境。也就是說,seed 內的對比不會被機器切開。

## 硬限制(全部強制執行並有測試覆蓋)

+ 一個 state 只能由一個 host、一個 worker、一個 process 完成
+ 不得跨 host 拼接 microbatches
+ 不得拆分同一個 state
+ 已在現行 gpu-host 開始的 state 不得轉移
+ 已完成的 state 不得重跑後擇優
+ 分配不得查看 scientific score values

## Worker queue 規則

固定、可重現、不看結果。**以 cell 為配對單位交替**:依 position 序取出各
cell,第 j 個 cell 的兩個 scaler arm 一起指派給 worker (j mod 2)。

### 為什麼不是逐一輪流

第一版寫的是「position 序第 i 個 state 給 worker (i mod 2)」。測試立刻抓到
它違規:E4 的凍結順序把同一個 cell 的兩個 scaler arm 排成**相鄰**,所以逐一
輪流會把 `cell_specific` 整個給 worker 0、`frozen_reference` 整個給 worker 1
—— 正是「不得把某一 scaler arm 全部固定給同一個 worker」所禁止的。

這個錯誤是被測試抓出來的,不是事後想到的。改成以 cell 配對為單位後,每個
worker 拿到兩個 cell 的完整 arm 對,兩個 worker 都同時見到兩個 arm。

seed999 的實際分派:

| worker | states |
|---|---|
| 0 | `cell10__frozen_reference`, `cell10__cell_specific`, `cell01__frozen_reference`, `cell01__cell_specific` |
| 1 | `cell00__frozen_reference`, `cell00__cell_specific`, `cell11__cell_specific`, `cell11__frozen_reference` |

4 / 4 平均,兩個 worker 各見兩個 arm、兩個 cell。

兩個 worker 在**同一台機器、同一張 GPU** 上,worker 只是排程 slot,不是不同
的執行環境,所以這個分派本身不引入任何 machine confounding;它要避免的只是
arm 與 slot 的系統性綁定。

### 其他 queue 約束

+ 每個 worker 同一時間只跑一個 state
+ 前一個 state 完成後才能取得下一個
+ 不使用 `ProcessPoolExecutor` 管理模型本身
+ 使用外層明確的 queue / worker slot
+ 每個 state 記錄 worker ID、GPU UUID 與 process UUID

## Execution-provenance 限制(必須在論文方法中出現)

以 seed block 分 host,會讓 **execution host 與 context seed 綁定**。

這不是原始的單 host protocol。該 seed 內的四個 factorial cell、兩個 scaler
arm、negative-support contrast、interaction 與 scaler 比較確實都留在同一個
執行環境,所以 **seed 內**的對比不被機器切開;但 **seed 之間**的比較就同時
混入了機器差異。

> 這只**削減**了 machine confounding,並未消除。
> **不得宣稱已完全消除 machine confounding。**

E5 已經證明這類環境差異是真實的而非假設:同一批 tree ensemble 在筆電與
gpu-host 之間平均差 8.1e−03。

此方案只能在事前的人類 execution override 下使用,且必須作為
execution-provenance limitation 報告。

## Draft override

`data/processed/m5_e6_gputw_audit/e6_gputw_execution_override.DRAFT.json`
sha256 `939c94831302a29d154583b8765a9c21c115bae4630470473de1244e8d20d9a5`

明確標記:

+ `launchable: false`
+ `human_authorisation_required: true`
+ `is_frozen: false`
+ `gputw_tree_outputs: forbidden`
+ `gputw_full_holdout_scoring_this_round: forbidden`
+ `no_state_is_split: true`
+ `no_cross_host_microbatch_splice: true`
+ `one_state_one_host_one_worker_one_process: true`
+ `endpoints_unchanged` / `decision_rules_unchanged` / `tree_outputs_unchanged: true`

**本輪未建立、也未凍結任何正式 override。**

## Artifacts

+ `data/processed/m5_e6_gputw_audit/state_allocation.json`
  sha256 `b5c39cd5ce1d328dc410cdebd178bb91f05ad8093703daa322b87a2e644d58e3`
+ `data/processed/m5_e6_gputw_audit/e6_gputw_execution_override.DRAFT.json`
