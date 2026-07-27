# M5 訓練量 vs 模型性能：TabPFN 在 scarce data 是否有優勢

**要回答的問題：在相同的標註資料量下，TabPFN 是否勝過樹集成，而且優勢是否隨資料量減少而擴大。**

做法是在同一個 50/50 building holdout 上掃訓練量 N，**兩個模型都訓在完全相同的那 N 列上**，
看 `TabPFN(N) − Trees(N)` 這個差距隨 N 的變化。TabPFN 的 N 是 context 列數，
樹的 N 是訓練列數；兩者都是「你手上有多少標註資料」，這正是 scarce-data 的問法。

TabPFN 帶著合成資料預訓練的先驗、樹沒有——這個不對稱**就是被檢驗的假說本身**
（先驗能不能替代資料），不是實驗瑕疵。

搭配閱讀：`m5-tabpfn-137feature-full-holdout-runbook.md`、`m5-tabpfn-17feature-full-holdout-runbook.md`
（批次切法、驗證關卡、portable worker 的行為完全沿用）。

**狀態：執行中（2026-07-27 起）。本文是計畫與協定；實際進度以
[docs/handoffs/2026-07-27-m5-tabpfn-context-curve-gputw-run.md](../handoffs/2026-07-27-m5-tabpfn-context-curve-gputw-run.md) 為準。**
fit / export 全部完成，已租 5090 並開始推論；17 維 @10k 已合併驗證。
§9 的成本估算已被實測取代（17 維 @20k 實測 ~4,000 rows/s，比估算快約 3.3 倍；
真正的瓶頸是 ~1.5–2.0 MiB/s 的上行頻寬，不是算力）。

## 1. 必須兩邊都掃 N，不能拿既有藍線當對照

**既有的樹集成藍線是用約 2.7M 列訓出來的**（`downsample_indices` 產生
`[negs1, pos, negs2, pos]`，正例重複一次）。拿 TabPFN@5k 去比它，比的是
「稀少 vs 充足」，回答不了 scarce-data 的問題。所以樹必須在每個 N 重訓一次。

巢狀前綴性質（§2）在這裡承擔雙重角色：既保證曲線上各點之間可比，
也保證同一個 N 上兩個模型**看到逐列相同的資料**。

三個要看的量：

- `TabPFN(N) − Trees(N)` 隨 N 的走向 —— 主結論。優勢若存在，應在小 N 最大。
- **交叉點** —— 樹追上 TabPFN 的資料量在哪裡。
- 17 維 vs 137 維的差距是否也隨 N 改變 —— 特徵工程買到的是「樣本效率」還是只有「終點高度」。
  這是同時跑兩個特徵寬度的理由。

**不要預期任何單一方向。** 17 維 n=1→n=8 的 estimator sweep 已證明這個系統對超參數的反應
沒有單一方向（Site 2 的 ROC 與 PR 都掉）。**不得事先在文件、圖說或 commit message 裡預告方向。**

## 2. 核心前提：小 context 是 100k 的精確前綴（已驗證）

`canonical_contract()` 以單一 budget 呼叫 `nested_balanced_indices()`（`run_m5_tabpfn_single_context_scaling.py:154`）。
該函式先用固定 seed 對完整的 positive / negative 陣列 `shuffle`，再各取前 `budget//2` 列交錯排列——
**shuffle 的結果與 budget 無關**，所以不同 budget 的分開呼叫會產生互為前綴的結果。

**已在真實資料上驗證通過**（`scripts/verify_m5_tabpfn_context_nesting.py`，
證據寫在 `data/processed/m5_tabpfn_context_nesting_proof.json`）：

```
candidate pool: 10,074,945 rows (676,817 positive), 4,000 validation rows held out
100k context digest matches the frozen run   (nesting reference: 1,353,634)

context          prefix_of_1,353,634  positive  disjoint  sha256
    5,000            True     2,500               True  da0772c6b3db44de
   10,000            True     5,000               True  1c89d75fc78506e5
   20,000            True    10,000               True  b436f94448d644d0
   50,000            True    25,000               True  06b9cbe9661c8adb
  100,000            True    50,000               True  e9ffe0cffd2e0cf3
  200,000            True   100,000               True  48b15c975d500913
  500,000            True   250,000               True  0a082294348f9a41
1,000,000            True   500,000               True  68970b305546c3ae
1,353,634            True   676,817               True  d915926498141cfd
```

兩件事同時成立才有意義：

- 九個 N 全都是最大者的精確前綴 —— 曲線各點之間、以及**同一點上兩個模型之間**都沒有抽樣變因。
- 重建的 100k digest `e9ffe0cf…d2688cbe` **等於既有 fit manifest 的凍結值** ——
  證明這條曲線是既有那條線的延伸，不只是「九個點彼此自洽」。

最後一列的 676,817 正好用光訓練半邊的全部正例，這就是 §4.1 的平衡上限。

比較的參考基準是最大的 N，不是 100k：樹要跑到 100k 以上，用 100k 當基準的話
那些點會因為「比較長」這個無關理由而全部判定失敗。（第一版就是這樣寫的，跑出來才發現。）

**驗證方式**：每個 context 的 `fit_manifest.json` 都會寫 `context_sha256`。開跑前逐一比對
「context_index 是否等於 100k context_index 的前 N 列」，不符即中止（見 §8 關卡 1）。

## 3. 已決定的設定（與 100k 逐項相同，只有 context 變）

| 項目 | 值 | 是否可變 |
|---|---|---|
| 評測母體 | `building_id % 2 == 1`，**10,137,155 列**，16 sites，724 buildings | ❌ 凍結 |
| 列序 | canonical `m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz`（依 building_id 遞增） | ❌ 凍結 |
| Foundation checkpoint | SHA-256 `d0d865d5…ea18f3988` | ❌ 凍結 |
| `n_estimators` | 8（關閉 auto-scale） | ❌ 凍結 |
| 固定驗證列 | 4,000（`--validation-rows`，與 context 不重疊） | ❌ 凍結 |
| Seed | 42 | ❌ 凍結 |
| checkpoint / microbatch | 20,000-row checkpoint；microbatch 上限 20,000 | ⚠️ 見 §6 |
| CPU / TPU fallback | 禁止 | ❌ 凍結 |
| **Context 列數** | **5,000 / 10,000 / 20,000 / 50,000 / 100,000** | ✅ 本次唯一變因 |

### 3.1 已定案的爭議點：StandardScaler 每個 context 重新擬合

100k 協定是「StandardScaler，以 context 擬合」。小 context 有兩種做法：

- **(A) 每個 context 各自重擬 scaler**（採用）
- (B) 全部沿用 100k 的 scaler

**採用 (A)。** 理由是決定性的：(B) 會把 100k 列的分布資訊**洩漏進 5k 條件**。
本曲線問的是「只有 N 列 context 時 TabPFN 有多強」，而在那個世界裡你也只有 N 列可以擬 scaler。
(B) 會系統性高估小 context 的表現，且高估幅度隨 context 縮小而變大——正好扭曲曲線最想看的那一段。

代價：每個 context 的全 test 特徵矩陣都不同（見 §5 的上傳量與 §6.3 的緩解做法）。

## 4. 範圍：N 的格點與兩邊的天花板

兩邊的可行上限不同，這本身就是結果的一部分，不是排程問題。

| N | 樹（本機 CPU） | TabPFN（GPU） | 說明 |
|---:|---|---|---|
| 5,000 | 待跑 | 待跑（本機 4070） | matched |
| 10,000 | 待跑 | 待跑（5090） | matched |
| 20,000 | 待跑 | 待跑（5090） | matched |
| 50,000 | 待跑 | 待跑（5090） | matched |
| 100,000 | 待跑 | ✅ 已完成 | matched；樹這格是新的 |
| 200,000 | 待跑 | ⬜ 視實測天花板 | |
| 500,000 | 待跑 | ⬜ 視實測天花板 | |
| 1,000,000 | 待跑 | ❌ 不可能 | |
| 1,353,634 | 待跑 | ❌ 不可能 | **平衡上限** |
| 2,735,231 | ✅ 既有藍線 | ❌ | 參考線，正例重複 |
| 10,074,945 | 待跑 | ❌ | 參考線，自然比例 6.7% |

每格都是兩個特徵寬度（17 / 137）。

### 4.1 為什麼樹停在 1,353,634

訓練半邊只有 **676,817 個正例**。50/50 平衡且不重複抽樣，總量就封頂在兩倍。
再往上只有兩條路，兩條都是換協定、都會多一個變因：

- **重複正例** —— 既有 M3 藍線就是這樣堆到 2,735,231 列。
- **放棄平衡** —— 全部 10,074,945 列、正例率 6.7%（`--natural-prevalence`）。

這兩點都要跑，但**在圖上必須標成協定不同的參考點，不能當成 matched 曲線的延伸**。
manifest 裡有 `natural_prevalence` 與 `matches_tabpfn_context` 兩個布林欄位供繪圖器判斷。

### 4.2 TabPFN 的天花板要實測

`m5_tabpfn_single_context_scaling.json` 記的是 `last_safe_budget: 100000`、
`headline_500k_success: false`——但那份 `resource_limits` 寫著 `total_mib: 8188`，
**是在本機 8GB 的 4070 上量的**。32GB 的 5090 上天花板未知，很可能到 200k–500k。

租到第一台 5090 時順手用 §7 的校準工具探（`--contexts 200000 500000`），成本是幾分鐘。
TabPFN 能推到多遠，直接決定曲線右端在哪裡收尾，是有報告價值的結果本身。

**10M 對 TabPFN 不是「還沒跑」而是「不可能」**：它對 context 做 attention，不是訓練。

### 4.3 全 holdout 推論次數

TabPFN 每個 N 一次全 holdout（10,137,155 列）× 2 寬度。樹同樣每格一次，
但樹在 CPU 上跑，不佔 GPU 預算。

**不能套用 137 runbook 的「sites 1/2/3 已完成」捷徑**——那 2,735,231 列是在 100k context 下算的，
換 N 就失效。批次計畫要以 `--done-sites`（不帶值）重新產生，涵蓋全部 16 個 site。

## 5. 命名規範（沿用既有慣例，插入 context 維度）

```
fit work dir     data/processed/m5_tabpfn_{17,137}_full_test_context{C}_n8.work
slice root       data/processed/m5_tabpfn_{17,137}_distributed_context{C}_n8
batch plan       data/processed/m5_tabpfn_{17,137}_context{C}_batch_plan.json
shard root       data/processed/m5_tabpfn_f{17,137}_batch{N}_context{C}_n8
merged out       data/processed/m5_tabpfn_{17,137}_full_test_context{C}_n8_predictions.npz
remote root      /workspace/lead_tabpfn_c{C}_b{N}_{head,tail}_f{17,137}_n8
```

**remote root 一定要含 context**。137 runbook §6 已經記載過撞目錄的後果：
`--resume` 會靜默取到別條線的 checkpoint。現在多了 context 維度，撞的機率更高，
而且**兩個 context 的分數都是 finite、列數都對，合併驗證抓不到**——只有 remote root 命名能防。

## 6. 程式改動（**已完成**）

原本六支腳本把 `context100000` 寫死在衍生路徑裡。那不是不方便而已：
用預設值跑 `--context-rows 5000`，5k 的 fit 會**寫進與 100k 完全相同的目錄、相同檔名**，
無聲覆蓋既有成果。這六處已全部改成隨 `--context-rows` 衍生。

### 6.1 已修好的寫死路徑

| 腳本 | 原本 | 現在 |
|---|---|---|
| `fit_m5_tabpfn_137_context100000.py` | work-dir 寫死 `…context100000{suffix}.work` | 隨 `--context-rows` 衍生 |
| `fit_m5_tabpfn_17_context100000.py` | work-dir 寫死 `…context100000_n{k}.work` | 同上 |
| `export_m5_tabpfn_137_shards.py` | `source_work_dir` / `out_root` 寫死 | 新增 `--context-rows`，並比對 fit manifest |
| `export_m5_tabpfn_137_batch_shards.py` | 輸出目錄與 slice 來源寫死 | 新增 `--context-rows`，並比對 fit manifest |
| `export_m5_tabpfn_17_batch_shards.py` | 輸出目錄與 remote root 寫死 | 新增 `--context-rows`、`--remote-prefix` |
| `merge_m5_tabpfn_full_test.py` | `LINE_ROOTS` / `LINE_OUT` 寫死 | 新增 `--context-rows`，roots 與 out 皆衍生 |

另外修掉一個會**擋死整條曲線**的斷言：`fit_m5_tabpfn_17_context100000.py` 的
`prove_scaler_matches_canonical` 要求 refit 出來的 scaler 與 canonical 100k 的
**逐位元相同**（`rtol=0, atol=0`）。那個要求對 estimator sweep 是對的（只有 n 改變，
scaler 漂移就是 bug），但對 context sweep 恰好相反——scaler **本來就該不同**。
已改成只在 `context_rows == 100_000` 時強制，其餘情況記錄 `False` 而不中止。
（跑 5k 的 17 維 fit 時實際撞到。）

**全部向後相容**：`--context-rows 100000`（預設值）產生的路徑與改動前的字面值**逐字相同**，
已用 10 條路徑逐一比對通過，既有的 100k 產物不會失聯。

兩個匯出器另外新增一道檢查：**context 以 fit manifest 為準，不從目錄名推斷**。
因為 `--source-work-dir` 可以覆寫路徑慣例，用錯 context 的矩陣會產出
「finite 分數、正確列、錯誤實驗」——下游每一道關卡都會放行。

### 6.2 新增的腳本

| 腳本 | 用途 |
|---|---|
| `verify_m5_tabpfn_context_nesting.py` | Gate 1：真實資料上證明巢狀性 + 比對凍結 digest（已跑過，9 個 N 全通過） |
| `run_m5_tree_ensemble_matched_context.py` | **樹的 matched-N 手臂**：訓在 TabPFN 的同一批列上，並比對 Gate 1 的 digest |
| `calibrate_m5_tabpfn_context_throughput.py` | §7 Step 0：量 rows/s 對 context 的縮放，解出 `a`、`b`，直接印出 GPU-hours |
| `export_m5_tabpfn_context_scaler.py` | 把 fit 的 `scaler.joblib` 匯成可攜的 `scaler.npz` |
| `gputw_tabpfn_shard.sh` | gputw.ai 的 push / run / status / pull（取代整套 Colab 機制） |

`plan_m5_tabpfn_137_remaining_batches.py` 不需改動：已有的 `--done-sites`（不帶值即空集合）
就能產生涵蓋全 16 site 的計畫。

### 6.3 上傳量優化（**已實作**，省 11.6 GiB 上行）

每個 context 的 shard 特徵矩陣是**已套用該 context scaler 的值**
（`export_m5_tabpfn_137_batch_shards.py` docstring 明說）。照現況，三個遠端 context 要上傳：

| 內容 | 每 context | ×3 context |
|---|---:|---:|
| 137 維特徵（10,137,155 × 137 × 4 B） | 5.17 GiB | 15.5 GiB |
| 17 維特徵（10,137,155 × 17 × 4 B） | 0.64 GiB | 1.9 GiB |
| **合計** | **5.81 GiB** | **17.4 GiB** |

但縮放是逐元素仿射轉換。**上傳未縮放矩陣一次、把 scaler 帶過去在遠端套用**，
上行量就從 17.4 GiB 降到 5.81 GiB（一份未縮放矩陣 + 每個 context 一個各 137 浮點數的
`scaler.npz`）。

已實作：`run_m5_tabpfn_portable_shard.py` 新增 `--scaler`，在既有的 microbatch 切片處套
`(X - mean) / scale`；features 本來就是 mmap 逐段讀，不增加記憶體。
`export_m5_tabpfn_context_scaler.py` 負責從 `scaler.joblib` 產出 `scaler.npz`。
worker 的結果 manifest 會記錄 `scaler_applied_at_predict` 與 `scaler_sha256`——
少了這兩欄，事後無法分辨一個 shard 是用預縮放矩陣還是未縮放矩陣算的。

**數值等價性已驗證**：拿真實的 100k 137 維 shard，反推回未縮放值再依 worker 的方式重套，
與原本匯出時縮放的結果最大差 `2.4e-07`（float32 往返誤差），平均差 `1.1e-10`。
兩條路徑算的是同一件事。

它同時省掉每個 context 重跑一次「quarter-hour 的 value-change 特徵建置」。

## 7. Step 0：吞吐量校準（**必做，且必須在租機報價前做完**）

現有的唯一實測錨點是 **A100-SXM4-40GB @ 100k context**：137 維 330 rows/s、17 維 430 rows/s。
把它外推到 5k–50k 需要知道**吞吐量如何隨 context 縮放**，而這件事目前 repo 裡沒有任何量測。

物理上 `t_per_row = a + b·C`：`b·C` 是 query 對 context 的 attention，`a` 是與 context 無關的
固定成本（特徵嵌入、MLP）。context 越小，`a` 越主導，**吞吐量會趨於平台而不是線性上升**。
所以「5k 比 100k 快 20 倍」一定是錯的，錯多少決定要租幾小時。

**Step 0 內容**（約 20–30 分鐘、一張卡）：在目標 5090 上，對同一批 20,000 列 query，
量 C ∈ {5k, 10k, 20k, 50k, 100k} × {17, 137} 的 rows/s，解出 `a` 與 `b`，
再用實測值重算 §9 的整張表。順帶確認 32GB VRAM 下 microbatch 20,000 是否仍可用
（50k context 是最吃緊的一格）。

**在 Step 0 完成前不要預付任何機時。** 下表的估算是有根據的猜測，不是量測。

## 8. 驗證關卡（每一關都擋錯，不是事後檢查）

沿用兩份 runbook 的關卡，**新增第 1 關**：

1. **【新增】context 巢狀性**：每個 context 的 `context_index` 必須**逐列等於** 100k
   `context_index` 的前 N 列；`context_sha256` 記入 manifest。不符即中止。
   這一關保證曲線的五個點之間沒有抽樣變因。
2. **匯出時**：重算列集合與計畫的列數 / anomaly 數 / building 範圍相符；
   raw_index 仍對應 canonical 的 anomaly / site_id / building_id。
3. **匯出後交叉驗證**：同一 context 下，17 與 137 兩條線的每個 shard `raw_index_sha256`
   與 `label_sha256` 必須相同；**且必須等於 100k 線同名 shard 的值**（列集合不隨 context 改變）。
   全部 shard 的 raw_index 聯集恰好等於 10,137,155 列、無重複。
4. **上傳後**：遠端比對四個 SHA-256（features、metadata、portable fit、foundation checkpoint）。
5. **worker 啟動時**：`--n-features`、`--n-estimators 8`、`--context-rows` 與 fitted state 不符就拒絕啟動。
6. **合併時**：union 恰好等於 holdout，labels / site / building 逐列與 canonical 相符，分數全 finite。

## 9. 成本與時程估算（**待 Step 0 取代**）

模型：`t_per_row = a + b·C`，假設 100k 時 80% 成本來自 context attention（即 `a` 佔 20%）。
5090 假設 ≈ A100 ±30%（GDDR7 1792 GB/s vs HBM2e 1555 GB/s；bf16 算力略低）。

### 9.1 遠端（gputw.ai RTX 5090，10k / 20k / 50k）

| 線 | context | 估 rows/s | GPU-hours |
|---|---:|---:|---:|
| 137 | 10,000 | ~1,180 | 2.4 |
| 137 | 20,000 | ~920 | 3.1 |
| 137 | 50,000 | ~550 | 5.1 |
| 17 | 10,000 | ~1,540 | 1.8 |
| 17 | 20,000 | ~1,200 | 2.4 |
| 17 | 50,000 | ~720 | 3.9 |
| | | **合計** | **~18.7** |

單張卡循序跑約 19 小時；**兩張 5090 並行約 9.5 小時**（head/tail 兩個 shard 各一張，
沿用既有 2-slot 排程的概念，但不需要 Colab 那套 session 管理）。

gputw.ai 公開頁面上我確認到的價格只有 RTX 3090 NT$7/hr 與 H100 NT$53/hr，
**5090 的單價需要登入 dashboard 確認**。以 NT$20–25/hr 估，18.7 GPU-hours ≈ **NT$375–470**；
兩張卡並行不改變總 GPU-hours，只縮短 wall clock。加上部署與下載的閒置計費，抓 NT$500–600。

### 9.2 本機 5k（RTX 4070 Laptop 8GB）

| 線 | 估 rows/s | wall clock |
|---|---:|---:|
| 137 @ 5k | ~275 | ~10.2 h |
| 17 @ 5k | ~360 | ~7.9 h |
| **合計** | | **~18 h** |

假設 4070 Laptop ≈ 0.2× A100（8GB GDDR6 ~256 GB/s，頻寬差約 6 倍，且 microbatch 會被迫調小）。
兩個晚上的背景長跑。

**建議的止損點：若 Step 0 量到本機 137 @ 5k 低於 400 rows/s，就把 5k 也搬到租來的 5090。**
多租 ~2.5 GPU-hours（約 NT$60）換掉 18 小時的本機佔用與失敗重跑風險，這筆保險便宜到不該猶豫。
本機跑仍是預設選項，但不值得為了省 NT$60 硬撐。

VRAM 方面 8GB 在 5k context 應可運作——portable worker 有自適應降 microbatch
（下限 256），但吞吐量會因此掉得比純頻寬推算更多，這也要靠 Step 0 確認。

## 10. gputw.ai 執行方式（與 Colab 路線的關鍵差異）

| 項目 | Colab（既有 runbook） | gputw.ai |
|---|---|---|
| 帳號 | `tonykuo210100@gmail.com`（HOME `.colab-tony`） | **`kuantingkuo@ntu.edu.tw`** |
| 存取 | `colab` CLI + Google Drive | **僅 SSH**（官網只提到 SSH） |
| 搶佔 | 會被回收，需 keep-alive / supervisor / reaper | **專用實例，不會被搶佔** |
| 上傳限制 | 單次 > 64 MB 失敗，需分段 | scp / rsync，無此限制 |
| 計費 | CU | 按秒計費，可隨時停 |

### 10.1 環境選擇：Linux，不要 Windows

映像選 **PyTorch 2.x + JupyterLab**（`gputw/pytorch:latest`）—— 有 SSH、有 CUDA、torch 預裝。
整套工具鏈是 rsync / ssh / tmux，**Windows (RDP) 只會讓每一步變難**。
退而求其次是 `gputw/cuda12-dev:latest`，但要自己裝 torch（約 2.5 GB）。

### 10.2 Blackwell kernel：上傳前必過的一關

RTX 5090 是 Blackwell，compute capability **12.0 (sm_120)**。
不含 sm_120 kernel 的 PyTorch build **import 正常、`torch.cuda.is_available()` 回 True、
`get_device_name` 也讀得到「RTX 5090」**，只有在真的發射 kernel 時才炸
（`no kernel image is available for execution on the device`）。

也就是說**「CUDA 可用」完全證明不了這台機器能跑**。開機第一件事跑：

```bash
ssh user@<ip> 'bash -s' < scripts/gputw_bootstrap.sh
```

`gputw_bootstrap.sh` 會查 GPU、印 capability、**實際做一次 fp32 與 fp16 matmul**、
確認 tabpfn（不在就裝 `tabpfn>=8,<9`）、並在 `/workspace` 留下 persistence 探針。
matmul 失敗就會直接給出 `--index-url .../cu128` 的修法。

**先過這一關再傳 5 GiB。** 傳完才發現 kernel 不對，等於白付一小時的上傳與租金。

**這是一次大幅簡化。** 下列 Colab 專用機制**全部不需要**：

- `monitor_m5_tabpfn_colab_keepalive.ps1`（無搶佔）
- `reap_idle_m5_tabpfn_colab_sessions.ps1`（無 session 概念）
- `supervise_m5_tabpfn_site_shard.ps1` 的 session 重建邏輯
- `deploy_m5_tabpfn_site_shard.ps1` 的 64 MB 分段
- 孤兒 assignment 回收

**保留**：`run_m5_tabpfn_portable_shard.py`（本來就設計成自足、可上傳、可 `--resume`）、
20,000 列 durable checkpoint、合併驗證。

**新增**：`scripts/gputw_tabpfn_shard.sh`（已寫好，四個子指令）取代整套 Colab 機制。

| 子指令 | 作用 |
|---|---|
| `push` | `rsync -avP -c` 推 shard + fit state + checkpoint + scaler，逐檔比對 SHA-256，不符即中止 |
| `run` | `tmux` + `nohup` 啟動 worker，stdout/stderr 一律進 `worker.log`；自動偵測 `scaler.npz` 決定要不要加 `--scaler` |
| `status` | 印 durable chunk 數、最新 chunk 的年齡（秒）、instance uptime、log 尾巴 |
| `pull` | 拉回 chunks，**比對遠端與本機 chunk 數**，數目不合就拒絕回報「可以關機」 |

`rsync` 帶 `-c`（checksum 而非 size+mtime）：中斷過的上傳留下的截斷檔案，
大小時間都可能看起來是對的。

判斷卡住**只看本機 durable checkpoint**，不看遠端 heartbeat——worker 完全沒產出時
遠端狀態仍然一片健康，那正是要抓的失效模式。`status` 印的「最新 chunk 年齡」
就是為了讓單獨一行就能回答「此刻還在不在動」。

**兩個 gputw.ai 特有的風險**：

- **官網未提及持久化儲存。** 停掉實例前必須確認結果已完整拉回本機（比對 chunk 數與 SHA）。
  **在確認「停機後 `/workspace` 是否保留」之前，不要停任何跑到一半的機器。** 這一項要在
  Step 0 租第一台機時當場測掉：寫一個檔案、停機、開機、看檔案還在不在。
- **按秒計費 + 無搶佔 = 忘記關機會一直燒錢。** Colab 的 session 回收雖然討厭，但也是個天然止損。
  這裡沒有，必須自己設定完成後自動停機，或至少掛一個到點提醒。

## 11. 執行順序（每個 context 一輪，context 之間可獨立）

### 11.0 一次性的前置（已完成 Gate 1，Step 0 待租機）

```powershell
# Gate 1：已跑過並通過，證據在 data/processed/m5_tabpfn_context_nesting_proof.json
uv run python scripts/verify_m5_tabpfn_context_nesting.py

# Step 0：本機基準（先跑這個，才知道 4070 撐不撐得起 5k）
uv run python scripts/calibrate_m5_tabpfn_context_throughput.py `
  --features data/processed/m5_tabpfn_137_distributed_context100000/head/features.float32.npy `
  --metadata data/processed/m5_tabpfn_137_distributed_context100000/head/metadata.npz `
  --out      data/processed/m5_tabpfn_context_throughput_local4070.json

# Step 0：租第一台 5090 之後，同一支腳本跑在遠端，兩份 JSON 直接對比
```

### 11.1 每個 context 一輪

```powershell
# --- C ∈ {5000, 10000, 20000, 50000}，L ∈ {17, 137} ---
$W = "data/processed/m5_tabpfn_${L}_full_test_context${C}_n8.work"

# 1. Fit（本機，數分鐘）。--work-dir 現在會自己帶上 context，不必再手動避讓。
uv run python scripts/fit_m5_tabpfn_${L}_context100000.py --context-rows $C --n-estimators 8

# 2. 匯出可攜 scaler（§6.3 的未縮放上傳路線要用）
uv run python scripts/export_m5_tabpfn_context_scaler.py --work-dir $W

# 3. 建全 test 矩陣 + 切 head/tail
uv run python scripts/export_m5_tabpfn_${L}_shards.py --context-rows $C --n-estimators 8

# 4. 批次計畫（全 16 site：--done-sites 不帶值）
uv run python scripts/plan_m5_tabpfn_137_remaining_batches.py --done-sites `
  --out data/processed/m5_tabpfn_${L}_context${C}_batch_plan.json

# 5. 匯出 12 個 shard
foreach ($b in 0..5) {
  uv run python scripts/export_m5_tabpfn_${L}_batch_shards.py --batch $b --context-rows $C `
    --plan data/processed/m5_tabpfn_${L}_context${C}_batch_plan.json `
    --remote-prefix /workspace
}

# 6. 推論：C=5000 本機；C ∈ {10k,20k,50k} 推 gputw.ai
#    （每個 batch 的 head/tail 各一台，或一台循序跑完 12 個 shard）

# 7. 合併（roots 與 out 都會自動帶上 context）
uv run python scripts/merge_m5_tabpfn_full_test.py --line $L --context-rows $C
```

### 11.2 gputw.ai 單一 shard

```bash
H=ubuntu@<instance-ip>
S=data/processed/m5_tabpfn_f137_batch0_context20000_n8/head

bash scripts/gputw_tabpfn_shard.sh push   $H $S 20000 137 0 head
bash scripts/gputw_tabpfn_shard.sh run    $H    20000 137 0 head
bash scripts/gputw_tabpfn_shard.sh status $H    20000 137 0 head   # 每隔幾分鐘看一次
bash scripts/gputw_tabpfn_shard.sh pull   $H    20000 137 0 head \
     data/processed/m5_tabpfn_f137_batch0_context20000_n8/head-results
# pull 會比對遠端與本機 chunk 數，數目不合就拒絕說「可以關機了」
```

建議順序：**Step 0 校準 → 10k 兩條線走通全流程 → 確認無誤後才批次推 20k / 50k / 5k**。
10k 是最便宜的完整一輪，拿它當 pipeline 的煙霧測試，比在 50k 上發現命名撞車便宜得多。

## 12. 圖：learning curve，不是水平參考線

**既有四張圖維持 100k 不動**，它們是「固定資料量下的模型比較」，仍然成立。
新增一個圖族，主角是 N 軸。

1. **主圖**：x = N（log 軸，5k → 10M），y = pooled ROC-AUC 與 PR-AUC。
   **四條實線**：TabPFN-17、TabPFN-137、Trees-17、Trees-137，全都隨 N 變動。
   TabPFN 兩條線在其天花板處**中斷**（不外插、不虛線延伸——那會暗示未測到的性能）。
   協定不同的兩個樹點（2.7M 重複正例、10M 自然比例）畫成獨立記號，不連進實線。

   **早期把樹畫成水平參考線是錯的**：那等於拿 2.7M 訓練量的樹去比 5k 的 TabPFN，
   量到的是資料量差異而不是模型差異。

2. **差距圖**：x = N，y = `TabPFN(N) − Trees(N)`，兩條線（17 維、137 維），零線標出。
   這張圖就是結論本身：曲線在小 N 為正且隨 N 下降 = scarce-data 優勢成立；
   穿越零線的位置 = 交叉點。

3. **副圖（可選）**：by-site 4×4 small multiples，每格畫 N 曲線，
   看哪些 site（100k 下落差最大的 Robin、Bear、Crow）對資料量最敏感。

副標的實測數字跑完再填，不要預先寫。

## 12.1 樹的 matched-N 執行

`scripts/run_m5_tree_ensemble_matched_context.py`（已寫好）。與既有 M3 runner 的兩處
**刻意**差異：

- **不用 `downsample_indices`。** 那個 helper 會複製正例（`[negs1, pos, negs2, pos]`），
  在 matched N 下等於讓樹拿到 TabPFN 兩倍的正例，卻掛同一個 N。
- **holdout 分批評分。** 14 格 × 10.1M 列 × 137 float32 在 32GB 機器上不能整塊放記憶體。

腳本有一道自動關卡：算出的 context digest 會**比對 Gate 1 的 proof JSON**，不符即中止。
matched-N 的整個主張就是「這是 TabPFN 的那批列」，這件事要用雜湊證明，不能寫在註解裡。

```powershell
# 每格約數分鐘訓練 + 數十分鐘推論，純 CPU，不佔 GPU 預算
foreach ($f in 17, 137) {
  foreach ($n in 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 1353634) {
    uv run python scripts/run_m5_tree_ensemble_matched_context.py --context-rows $n --features $f
  }
  # 協定不同的全資料參考點
  uv run python scripts/run_m5_tree_ensemble_matched_context.py --natural-prevalence --features $f
}
```

輸出 `m5_tree_ensemble_f{17,137}_context{N}_predictions.npz`，
欄位與 `m3_17_feature_ensemble_predictions.npz` 相同（canonical 列序、四個模型 + ensemble、
site_id / building_id），可直接餵給既有繪圖器。

**記憶體**：本機 31.6 GB。137 維在訓練半邊建 value-change 特徵是尖峰所在；
若不夠用就調小 `--predict-batch-rows`（預設 500,000），或先跑完 17 維再跑 137 維。

## 13. Decision

量的是**訓練量 vs 性能**，兩個模型都掃 N、都訓在逐列相同的資料上，
所以 `TabPFN(N) − Trees(N)` 這個差距可以直接歸因到模型而非資料量。

- **樹**：本機 CPU，N 從 5k 到 1,353,634（平衡上限），加兩個協定不同的參考點
  （2.7M 重複正例＝既有藍線、10.07M 自然比例）。不佔 GPU 預算。
- **TabPFN**：5k 在本機 RTX 4070，10k/20k/50k 在 gputw.ai 的 RTX 5090
  （帳號 `kuantingkuo@ntu.edu.tw`），100k 已完成。天花板要在 5090 上實測，
  可能到 200k–500k；**10M 對 TabPFN 是不可能，不是還沒跑**。
- 兩個特徵寬度（17 / 137）都跑，才看得出特徵工程買的是樣本效率還是終點高度。

九個 N 全是最大者的精確前綴（已驗證），scaler 每個 N 各自重擬，
以免把大 N 的分布洩漏進小 N 條件。

估計遠端 ~18.7 GPU-hours（兩卡並行約 9.5 小時 wall clock、約 NT$500–600），
本機 TabPFN 5k 約 18 小時；**但這是外推不是量測**，§7 的 Step 0 校準必須先做完才准預付機時。
上行量以 §6.3 的「未縮放矩陣 + 遠端套 scaler」計為 5.81 GiB（否則 17.4 GiB）。
樹的部分是純 CPU，不列入。

準備工作已全部完成並通過檢查：

- 六支腳本的寫死 context 路徑已修好，且在 100k 下與改動前逐字相同（10 條路徑比對通過）
- 兩個匯出器新增「context 以 fit manifest 為準」的檢查
- Gate 1 已在真實資料跑過並通過（9 個 N），重建的 100k digest 等於凍結值
- 樹的 matched-N runner 已寫好，內建 digest 比對與平衡上限守門
- `--scaler` 路徑已實作，數值等價性驗證到 2.4e-07
- 校準工具與 gputw.ai 部署腳本已寫好並通過語法／lint 檢查

**開跑前還剩三件事**：

1. §7 的吞吐量校準（本機與 5090 各跑一次）——在這之前不要預付機時。
2. §4.2 的 TabPFN 天花板實測（5090 上探 200k / 500k）——決定曲線右端在哪收尾。
3. §10 的「停機後 `/workspace` 是否保留」當場實測——寫檔、停機、開機、看檔案還在不在。

樹那一側不依賴以上任何一項，隨時可以先跑起來，而且它單獨就足以畫出
「樹的 learning curve」——TabPFN 到位之後直接疊上去。

**已於 2026-07-27 開跑。** 上面三件「開跑前還剩」的事，實際執行時的處理方式是：
吞吐量改為在正式 shard 上直接實測（見 handoff §2），`/vault` 的持久化問題由
「上傳一律落 `/vault`、再 hydrate 到 `/workspace`」解決。天花板實測仍未做。
進度、實測數字與三個傳輸失效模式一律見
[2026-07-27 handoff](../handoffs/2026-07-27-m5-tabpfn-context-curve-gputw-run.md)。
