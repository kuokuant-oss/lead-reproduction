# M5 TabPFN 17-feature estimator sweep（Site 1/2/3 全列）計畫 handoff

這份 handoff 給接手者、維運腳本與未來的 AI agent。它描述一個**尚未執行**的計畫：在既有 100k-context / 17-feature 設定上，量測 TabPFN `n_estimators` 從 1 提升到 4、8 對 Site 1、Site 2、Site 3 的判別力增益。寫法沿用
`docs/reference/m5-tabpfn-colab-dual-shard-runbook.md`。**此文件寫作時未啟動任何 session，也未 refit。**

> **2026-07-24 修訂（使用者指示）**：評測列明確限定為 `building_id % 2 == 1` 的 50/50 building holdout（見 §2.1）；
> Site 1 與 Site 2 改為**兩個帳號各兩張 A100、同時並行**，不再依序（見 §3.1）；每張 GPU 開跑前先做
> microbatch 校準（見 §3.2）；Site 3 保留，排在 Site 1、2 之後。時間估算（§4）連帶改寫。**仍未啟動任何 session。**

## 1. 目標與由來

正式全 test 的 17-feature TabPFN（`n_estimators=1`）分數低於 17-feature 樹 ensemble。已確認分數本身無 bug（無飽和、47,815 個相異值），落差來自兩點：`n_estimators=1` 關掉了 TabPFN 內部 ensemble，以及 in-context 100k vs 樹的全量訓練這個結構劣勢。

一次 50k 抽樣 probe 顯示 `n_estimators` 影響可能被低估：Site 1 的 `n=1 → n=4` 讓 ROC-AUC 從 0.5447 升到 0.6267（+0.082），比原先「<2%」的估計大。因此要用**完整資料**把這個增益量到底。

## 2. 不可改動的契約

以下任一項改變，結果就不能與既有基準或彼此比較：

- context：100000 rows，用 canonical `nested_balanced_indices`（seed 42）選出的同一組 context，與正式 17-feature run 完全相同。
- features：固定 17 個 baseline features（`BASELINE_FEATURE_COLS`）。不啟用 137-feature（那是另一條線，見
  [`m5-tabpfn-colab-dual-shard-runbook.md`] 與 137-feature handoff）。
- foundation model：`.tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt`，SHA-256 `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`。
- scaler：以 context 的 100k rows fit 的 `StandardScaler`，query 用同一 scaler transform。
- **評測資料：Site 1、Site 2、Site 3 在 50/50 building holdout 內的「全部 test 列」，禁止用 50k（或任何）抽樣充數。** 這是本計畫的核心要求：不得用抽樣結果宣稱「比以前好」。定義見 §2.1。
- microbatch：見 §3.2 的 A100 校準；校準值只能在資源壓力時往下減，最低 64；不可 CPU/TPU fallback。
- 唯一改變的變數是 `n_estimators ∈ {4, 8}`（`auto_scale_n_estimators=False`）。`n_estimators=1` 不需重跑，直接用既有正式全 test 分數當基準（見 §5）。

### 2.1 評測列的精確定義（2026-07-24 釘死）

「全部 test 列」= **`building_id % 2 == 1` 的 50/50 building holdout 中屬於該 site 的列**，也就是
`data/processed/m5_tabpfn_distributed_context100000_predictions.npz` 裡 `site_id == k` 的那些列，不是 BDG2 全表。

已在本機驗證該 npz：10,137,155 列、`building_id` 全為奇數（odd fraction = 1.0）、724 棟建物，與
`m3_figure_predictions_50_50.npz` 逐列對齊。這正是 M3 四張圖（含 `*_with_tabpfn.png`）的評測母體，所以只要 sweep 用同一組列，
新曲線就能直接疊到那些圖上（見 §7.1）。§3 的各 site 列數即取自這個 npz，不需另外重算。

## 3. 掃描維度與規模

| shard 維度 | 值 |
|---|---|
| sites | 1, 2, 3 |
| estimators | 4, 8 |
| 每格資料 | 該 site 全部 test 列 |

各 site 全列數與 prevalence：

| site | 全 test 列 | anomalies | prevalence |
|---|---:|---:|---:|
| Site 1 | 289,853 | 39,135 | 13.502% |
| Site 2 | 1,263,915 | 80,897 | 6.401% |
| Site 3 | 1,181,463 | 2,684 | 0.227% |
| 合計 | 2,735,231 | 122,716 | — |

Site 3 極稀有（2,684 正例），PR-AUC 會低且較不穩，但仍以全列量測，不做抽樣。

head / tail 切點（落在 20,000-row checkpoint boundary，取最接近中點者）：

| site | 全列 | head `[0, b)` | tail `[b, end)` |
|---|---:|---:|---:|
| Site 1 | 289,853 | 140,000 | 149,853 |
| Site 2 | 1,263,915 | 640,000 | 623,915 |
| Site 3 | 1,181,463 | 600,000 | 581,463 |

### 3.1 執行順序（2026-07-24 改寫）

Site 1 與 Site 2 綁在**不同 Google 帳號**上，因此兩者**同時並行**；單一 site 內部仍不同時跑兩個 estimator。

| site | 帳號 | GPU | HOME |
|---|---|---|---|
| Site 1 | `hank0503work@gmail.com`（Colab Pro） | 2 × A100（head + tail） | 既有 hank HOME `/home/tonykuo/.colab-hank` |
| Site 2 | `tonykuo210100@gmail.com`（Colab Pro） | 2 × A100（head + tail） | `/home/tonykuo/.colab-tony` |
| Site 3 | hank（Site 1 收工後直接接上） | 2 × A100（head + tail） | 同上 |

（2026-07-24 使用者指示其一：Site 1 與 Site 2 的帳號對調。較大的 Site 2 走 tonykuo。）

（2026-07-24 使用者指示其二：**Site 3 的啟動條件改為「Site 1 這一格 n=4 與 n=8 都跑完」，不必等 Site 2。**
因此兩個帳號各自成為一條獨立管線：hank 走 `Site 1 n=4 → Site 1 n=8 → Site 3 n=4 → Site 3 n=8`，
tonykuo 走 `Site 2 n=4 → Site 2 n=8`。Site 1 遠小於 Site 2，hank 會先空出來，這樣可避免該帳號的兩張 A100 閒置空燒 CU。）

launcher 環境變數要跟著換：runbook §5.1 的 `TABPFN_COLAB_ACCELERATOR=L4` 需改成 A100，`TABPFN_COLAB_HOME` 依帳號切換
（Site 1 用 hank 的既有 HOME，Site 2 用 `/home/tonykuo/.colab-tony`）。tonykuo 帳號登入時必須帶 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`，
否則 Google 少給 `drive.file` scope、會在寫檔前中止。

每個 site 的流程固定：

1. **切半**：依 §3 的切點切成 head / tail 兩半。head forward、tail reverse，比照正式 dual-shard。
2. **microbatch 校準**：兩張 A100 各自先跑 §3.2 的校準，取得該卡的 microbatch 與 rows/s。
3. **先 n=4**：兩張 A100 都以 `n_estimators=4` 跑，直到該 site 的 head + tail 兩半 n=4 全列 durable 完成。
4. **再 n=8**：同一 site、同樣兩張 A100，改 `n_estimators=8` 跑到兩半全列完成。n=8 的記憶體約 n=4 的兩倍，**重跑一次 §3.2 校準**再開跑。
5. **Site 3**：Site 1 與 Site 2 都收工後才開始，用先空出來的帳號，同樣 n=4 → n=8。

任何一輪未達「該 site 兩半全列 durable 完成」前，不得前進到下一個 estimator；Site 1、2 未全部完成前不得開 Site 3。

### 3.2 microbatch 校準（開跑前必做）

正式 17-feature run 的 1024 是 L4 22GB 的保守值。A100 40GB 記憶體近乎兩倍且頻寬更高，**每張卡在正式推論前都要先量到底能開多大**，
不要沿用 1024 就開跑。

- 候選值：1024 → 2048 → 4096 → 8192 → 16384，逐一往上試。
- **硬上界 20,000**：worker 的 `--checkpoint-rows` 是 20,000，且 `parse_args` 要求 `checkpoint_rows >= query_microbatch_size`；
  要超過 20,000 就得動 checkpoint 契約，本計畫不動。
- 每個候選值用該 shard 前若干個 microbatch 實測 rows/s 與 GPU 峰值佔用，取「不觸發降批、GPU 峰值低於
  `--gpu-soft-limit-fraction` 0.86」的最大值，記錄成 JSON（site、n_estimators、GPU 型號、microbatch、rows/s、峰值）。
- 校準值以 `--query-microbatch-size` 傳給 `run_m5_tabpfn_portable_shard.py`。worker 只會在壓力下往下減、不會自己往上加，
  所以校準是唯一能提速的入口。`--min-query-microbatch-size` 維持契約下限 64。
- n=4 與 n=8 各校準一次（記憶體量級不同）。校準本身**不算**正式分數，不得寫進成品 checkpoint 目錄。
- **校準的作用域是（GPU 型號 × n_estimators），不是每個 shard 各跑一次。** 四個 shard 都是 A100-SXM4-40GB、同一組 100k context、
  同樣 17 features、同一份 fitted state 大小，差別只在列數，因此同一個 n 的校準結果可套用到該 n 的四個 shard。
  這很重要：校準要實跑到最大候選值的一個完整 microbatch（16,384 列真實推論），單次約 20–30 分鐘；若每個 shard 都重跑，
  光校準就要數小時的 A100 時間。校準 JSON 需連同 GPU 型號一起留存當證據；換 GPU 型號就必須重跑。

### 3.3 A100 實測校準結果（2026-07-24，n=4）

`NVIDIA A100-SXM4-40GB`、17 features、100k context、`n_estimators=4`：

| microbatch | rows/s | torch peak reserved |
|---:|---:|---:|
| 1024（官方 run 值） | 55.2 | 3,578 MiB |
| 2048 | 108.7 | 3,580 MiB |
| 4096 | 213.2 | 3,590 MiB |
| 8192 | 410.1 | 3,608 MiB |

**兩個結論**：

1. 吞吐對 microbatch 近乎線性，而記憶體幾乎不變（40,441 MiB 的卡只用約 3.6 GB）。記憶體由 100k context 主導，與 query batch 大小幾乎無關，
   所以「1024 是安全值」在 A100 上等於白白慢十倍。正式採用 **16384**，實測 **~493 rows/s**（約為 1024 的 10 倍）。
2. 16384 沒有達到線性外推的 ~820 rows/s，因為 20,000 的 checkpoint 會被切成 16384 + 3616，尾巴那個小批很不划算。
   **n=8 應改用 microbatch 20000**（每個 checkpoint 剛好一批，無零頭），預期可再快一截；記憶體仍遠低於卡的容量。

### 3.3.1 n=8 實測：estimator 的成本非線性

Site 1 head、`n_estimators=8`、microbatch **20000**（剛好一個 checkpoint、無零頭批）實測 **413–453 rows/s**，
GPU 佔用仍只有 10%。對照 n=4 在 16384 的 ~493 rows/s，**n=8 幾乎沒有變慢，不是原估的一半**。

原計畫 §4 假設「estimator 對推論時間近似線性」（L4 上 n=4 22 rows/s → n=8 11 rows/s），據此估出 ~103 小時。
實測顯示在這個 batch 規模下瓶頸不在 estimator ensemble，而在 context 編碼，ensemble 成員在 GPU 上被有效平行化。
**因此 n=8 不需要為記憶體或速度做任何退讓，直接用 microbatch 20000。**

### 3.4 resume 會蓋掉 microbatch 的陷阱（已修）

`run_m5_tabpfn_portable_shard.py` 原本在 `--resume` 時從舊 `progress.json` 繼承 `effective_microbatch_size`，
**把命令列指定的新值直接丟掉**。原意是記住資源壓力造成的降批，副作用是重啟後永遠被釘在舊值上：
實際發生過一次——以 16384 重啟的 worker 仍以 1024 執行，吞吐停在 ~50 rows/s。
現在只要命令列給的值不等於預設 1024，就以命令列為準；真有壓力時 soft/hard limit 會在第一批內重新降批。

## 4. 時間量級與「必須可續跑」

實測吞吐（**L4**、17-feat、100k context、microbatch 1024）：`n=4 ≈ 22 rows/s`；estimator 對推論時間近似線性，故 `n=8 ≈ 11 rows/s`。
A100 40GB 加上 §3.2 校準後的較大 microbatch，預期是 L4 的 2–3 倍，但**在校準量到之前這只是估計**，以下表格用 L4 數字當保守上界，
校準完成後應以實測改寫本節。

每個 site 的兩半分在兩張 GPU 上，故該 site 的 wall clock ≈ 較大那半的時間：

| site | 全列 | n=4（L4 保守） | n=8（L4 保守） | 該 site 合計 | A100 樂觀（÷2.5） |
|---|---:|---:|---:|---:|---:|
| Site 1 | 289,853 | ~1.9 h | ~3.8 h | ~5.7 h | ~2.3 h |
| Site 2 | 1,263,915 | ~8.1 h | ~16.2 h | ~24.3 h | ~9.7 h |
| Site 3 | 1,181,463 | ~7.6 h | ~15.1 h | ~22.7 h | ~9.1 h |

Site 1 與 Site 2 並行，故前半段 wall clock 由 Site 2 決定（L4 保守 ~24 h、A100 樂觀 ~10 h）；Site 3 接在後面再加一段。
總 GPU 佔用時數不變（三 site × 兩 estimator 的純推論量級與原估的 ~103 L4-小時同級），只是攤到四張卡上。

**結論不變：此計畫遠超單一 GPU session 的壽命，不能用一次性 detached 腳本跑（那是 probe 的做法，只適合小樣本）。必須沿用 runbook 的可續跑基建**：每 20,000 rows 一個 atomic checkpoint、supervisor 在 GPU 被回收時以同名 session 重建並 `--resume`、sync monitor 逐檔下載、keep-alive + 45 分 work-touch。否則每次回收都從頭再來，永遠跑不完。A100 的 CU 燃燒率遠高於 L4（約 2.5×），因此**收工即釋放**比 L4 時代更重要。

## 5. 基準（n=1，已存在，不需重跑）

取自已 commit 的正式 17-feature 全 test 產物
`data/processed/m5_tabpfn_distributed_context100000_predictions.npz`（逐行與 canonical 對齊已驗證）：

| site | n=1 ROC-AUC | n=1 PR-AUC |
|---|---:|---:|
| Site 1 | 0.5447 | 0.2026 |
| Site 2 | 0.8435 | 0.2807 |
| Site 3 | 0.9762 | 0.1667 |

已有的唯一 n≥4 資料點（50k 抽樣 probe，僅供方向參考，**不作為結論**）：Site 1 `n=4` ROC-AUC 0.6267 / PR-AUC 0.2456，存於 `data/processed/probe_results_a.json`。全列版須重測覆蓋它。

## 6. 需要先建立或修改的東西

1. **estimator 參數化（✅ 2026-07-24 已完成）**：`create_real_model(model_path, seed, n_estimators=1)` 與
   `verify_fitted_context(model, rows, requested_estimators=1)` 都已可帶 estimator 數，預設值 1 使 17-feature 正式契約的檢查不變。
2. **per-estimator fit（尚未做）**：`n_estimators` 在 `TabPFNClassifier` 建構時決定，ensemble 成員在 `fit` 時建立，所以 n=4/8 各需一份
   17-feature fitted state（context 與 scaler 相同，只有 estimator 數不同）。fit 是 100k rows 的一次性動作，可在本機做，不算被禁止的本機全 test 推論。
   137-feature 線已用同一條路徑做出 n=1/4/8 三份 fit，可直接照抄做法。
3. **portable worker 的 estimator 參數化（尚未做）**：`scripts/run_m5_tabpfn_portable_shard.py` 目前載入的 fitted state 自帶 estimator 數，
   但 worker 的 preflight 應顯式記錄並驗證它等於本格要求的 n，確保 checkpoint / resume 行為不變。
4. **portable inputs（尚未做）**：只需 Site 1/2/3 的 17-feature scaled 全列 memmap + metadata（raw_index/anomaly/site_id/building_id），
   仿 `export_m5_tabpfn_colab_tail.py`，但範圍限這三個 site 在 §2.1 holdout 內的列，並依 §3 的切點分 head/tail。
5. **microbatch 校準腳本（尚未做）**：實作 §3.2，輸出 JSON 供 supervisor 與本 handoff 回填。
6. **記憶體**：`n=8` 的推論記憶體約是 `n=1` 的 8 倍。A100 40GB 下由 §3.2 校準決定 microbatch，壓力時自動下修（契約允許到 64）。

## 7. 成功判準

沿用 runbook §8：每格（site × n_estimators）唯有本機 durable checkpoint 覆蓋該 site 全列、且 score 全為 finite，才算完成。不得以遠端 heartbeat 或部分 chunk 宣稱完成，不得用抽樣分數代替全列分數。全部完成後產出一張
`n=1 → 4 → 8` 的 per-site ROC-AUC / PR-AUC 增益表。

### 7.1 與既有 M3 圖的可比性（2026-07-24 新增，使用者要求）

最終結果必須能和 `docs/reports/assets/m3/m3_tree_ensemble_by_site_*_with_tabpfn.png` 裡 Site 1、Site 2 的 TabPFN 曲線**直接並排**。
合併回本機後，先做這四項證明再畫圖：

1. 該 site 的 `raw_index` 集合與 `m5_tabpfn_distributed_context100000_predictions.npz` 中 `site_id == k` 的列**完全相同**（同集合、同數量、無重複）。
2. 依 `raw_index` 對齊後，`anomaly` / `site_id` / `building_id` 三者逐列相等。
3. 列數與 prevalence 命中 §3 的表（Site 1 289,853 / 13.502%、Site 2 1,263,915 / 6.401%、Site 3 1,181,463 / 0.227%）。
4. n=1 基準直接取自同一個 npz（§5），不重跑、不換算。

四項都通過，n=1/4/8 三條曲線才是同一組列上的差異，才可宣稱 estimator 增益，也才可疊到上述 M3 圖上。

## 7.2 已完成格的結果（2026-07-24 執行中回填）

每一格都通過 §7.1 的四項對齊證明。獨立算出的 n=1 基準與 §5 表格逐位相符（三個 site 六個數字全中），
這本身就是「確實在同一組列上評分」的旁證。

**六格全部完成（2026-07-24 定案）**，共 2,735,231 列全列量測、零抽樣：

| site | prevalence | 基準 n=1 ROC / PR | n=4 ROC / PR | n=8 ROC / PR |
|---|---:|---|---|---|
| Site 1 | 13.502% | 0.5447 / 0.2026 | 0.6248 (+0.0801) / 0.2458 (+0.0432) | 0.6647 (+0.1200) / 0.2640 (+0.0614) |
| Site 2 | 6.401% | 0.8435 / 0.2807 | 0.8144 (−0.0291) / 0.2262 (−0.0545) | 0.8177 (−0.0258) / 0.2101 (**−0.0706**) |
| Site 3 | 0.227% | 0.9762 / 0.1667 | 0.9787 (+0.0025) / 0.7900 (**+0.6233**) | 0.9800 (+0.0038) / 0.7828 (+0.6161) |

機器可讀版本：`data/processed/m5_tabpfn_estimator_sweep_table.json`
（由 `scripts/aggregate_m5_tabpfn_sweep_table.py` 產生；同一 site 的兩格若回報不同 n=1 基準會直接拒絕產表）。

**三個 site 的行為互不相同，沒有單一趨勢**：

- Site 1（基準接近亂猜 ROC 0.54）：兩個指標都明顯改善，且 n=4 → n=8 繼續上升。
- Site 2（基準已強 ROC 0.84）：**唯一負增益**，ROC 與 PR 同時退步。
- Site 3（極稀有，2,684 正例）：ROC 幾乎不動（+0.0025）但 PR-AUC 從 0.167 衝到 0.790。
  這個組合正是「整體排序沒怎麼變、但排序最頂端精確度大幅改善」的特徵；在 0.227% prevalence 下 PR-AUC 幾乎完全由頂端那一小段決定，
  單一 estimator 在該處噪音極大，ensemble 到 4 個即可抹平。

**實測**：

1. estimator 的影響橫跨 −0.07 到 +0.62。
2. Site 2 的 PR-AUC 隨 n 單調下滑（0.2807 → 0.2262 → 0.2101），
   n=8 並未回升，因此不是「n=4 恰好是低谷」。
3. **增益主要來自「開啟 ensemble」本身，而非把它加大**：三個 site 的 n=4 → n=8 變化都遠小於 n=1 → n=4，
   Site 3 的 n=8 甚至略低於 n=4。
4. 方向與基準強度似乎相關（Site 1 基準近亂猜→大幅改善；Site 2 基準已強→惡化；Site 3 極稀有→PR 巨幅改善而 ROC 幾乎不動），
   但**只有三個 site，不足以宣稱因果**，僅作為後續假設。
5. 操作結論：**不得用任何單一 site（更不用說單一 site 的 50k 抽樣）外推到其他 site**。
   若當初依 Site 1 的 probe 決策，會對 Site 2 給出完全相反的建議。

## 8. 現況快照（2026-07-24 寫作時）

- 未啟動任何 session；`colab sessions` 為 `No active sessions`，兩張 probe L4 已 terminate、CU 停止消耗。
- 本機無執行中的 fit / probe process。
- 已存在可重用資產：canonical 100k-context 17-feature fitted state
  （`data/processed/m5_tabpfn_canonical_full_test_context100000.work/`）、foundation checkpoint、`m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz`（canonical 身分來源）。
- probe 期間新增但尚未 commit 的腳本：`scripts/run_tabpfn_estimator_probe_colab.py`、`scripts/build_tabpfn_estimator_probe_inputs.py`、`scripts/probe_tabpfn_estimators.py`。全列正式版可能改寫或取代這些。
- 2026-07-24 修訂時追加：`create_real_model` 與 `verify_fitted_context` 已支援 `n_estimators`（預設 1，17-feature 契約不受影響）；
  17-feature 的 n=4/8 fit 仍未產出，Site 1/2/3 的 portable inputs 也仍未匯出。**依然未啟動任何 session。**

## 9. Decision

計畫已凍結但**尚未執行**：100k context、17 features、Site 1/2/3 在 `building_id % 2 == 1` holdout 內的全列、`n_estimators ∈ {4,8}`（n=16 已取消），n=1 用既有正式分數當基準。執行順序見 §3.1：**Site 1（tonykuo 帳號）與 Site 2（hank 帳號）各用兩張 A100 跑 head/尾兩半、兩個 site 同時並行；每張卡開跑前先做 §3.2 的 microbatch 校準；單一 site 內先跑完 n=4 再跑 n=8；Site 1、2 都收工後才用先空出來的帳號跑 Site 3。** 啟動前必須先完成 §6 剩餘項目（per-estimator fit、worker preflight 驗證、Site 1/2/3 portable inputs、校準腳本），並以 runbook 的可續跑基建（20k checkpoint + supervisor + keep-alive）執行，因為總量遠超單一 session 壽命。核心紀律：**全列量測，不得用抽樣結果宣稱增益**，且結果須通過 §7.1 的四項對齊證明才可與既有 M3 圖並排。
