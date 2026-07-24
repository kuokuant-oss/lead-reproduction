# M5 TabPFN 17-feature estimator sweep（Site 1/2/3 全列）計畫 handoff

這份 handoff 給接手者、維運腳本與未來的 AI agent。它描述一個**尚未執行**的計畫：在既有 100k-context / 17-feature 設定上，量測 TabPFN `n_estimators` 從 1 提升到 4、8 對 Site 1、Site 2、Site 3 的判別力增益。寫法沿用
`docs/reference/m5-tabpfn-colab-dual-shard-runbook.md`。**此文件寫作時未啟動任何 session，也未 refit。**

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
- **評測資料：Site 1、Site 2、Site 3 的「全部 test 列」，禁止用 50k（或任何）抽樣充數。** 這是本計畫的核心要求：不得用抽樣結果宣稱「比以前好」。
- microbatch：1024 起，資源壓力時往下減，最低 64；不可 CPU/TPU fallback。
- 唯一改變的變數是 `n_estimators ∈ {4, 8}`（`auto_scale_n_estimators=False`）。`n_estimators=1` 不需重跑，直接用既有正式全 test 分數當基準（見 §5）。

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

### 3.1 執行順序（固定）

不平行跑多個 site，也不同時跑多個 estimator。順序如下：

1. **切半**：把當前 site 的全列切成頭 / 尾兩半，切點落在 20,000-row checkpoint boundary 上（仿正式 dual-shard 的 head/tail）。
2. **兩張 L4 並行**：頭半在一張 L4、尾半在另一張 L4，方向可比照正式 run（head forward、tail reverse）。
3. **先 n=4**：兩張 L4 都以 `n_estimators=4` 跑，直到該 site 的頭 + 尾兩半 n=4 全列 durable 完成。
4. **再 n=8**：同一 site、同樣頭/尾兩張 L4，改 `n_estimators=8` 跑到兩半全列完成。
5. **換 site**：該 site 的 n=4 與 n=8 都完成後，才進入下一個 site。site 序為 **1 → 2 → 3**。

也就是每個 site 佔用兩張 L4、依序跑完 `{4, 8}` 兩輪，共走三個 site。任何一輪未達「該 site 兩半全列 durable 完成」前，不得前進到下一個 estimator 或下一個 site。

## 4. 時間量級與「必須可續跑」

實測吞吐（L4、17-feat、100k context）：`n=4 ≈ 22 rows/s`；estimator 對推論時間近似線性，故 `n=8 ≈ 11 rows/s`。

三個 site 全列 = 2,735,231 rows。單一 estimator 掃完三 site 的純推論估時：

| n_estimators | 約略吞吐 | 三 site 全列估時 |
|---|---:|---:|
| 4 | 22 rows/s | ~34 小時 |
| 8 | 11 rows/s | ~69 小時 |
| {4,8} 合計 | — | **~103 小時 ≈ 4.3 天 L4 算力** |

**結論：此計畫遠超單一 L4 的 ~52 分鐘壽命，不能用一次性 detached 腳本跑（那是 probe 的做法，只適合小樣本）。必須沿用 runbook 的可續跑基建**：每 20,000 rows 一個 atomic checkpoint、supervisor 在 L4 被回收時以同名 session 重建並 `--resume`、sync monitor 逐檔下載、keep-alive + 45 分 work-touch。否則每次回收都從頭再來，永遠跑不完。

## 5. 基準（n=1，已存在，不需重跑）

取自已 commit 的正式 17-feature 全 test 產物
`data/processed/m5_tabpfn_distributed_context100000_predictions.npz`（逐行與 canonical 對齊已驗證）：

| site | n=1 ROC-AUC | n=1 PR-AUC |
|---|---:|---:|
| Site 1 | 0.5447 | 0.2026 |
| Site 2 | 0.8435 | 0.2807 |
| Site 3 | 0.9762 | 0.1667 |

已有的唯一 n≥4 資料點（50k 抽樣 probe，僅供方向參考，**不作為結論**）：Site 1 `n=4` ROC-AUC 0.6267 / PR-AUC 0.2456，存於 `data/processed/probe_results_a.json`。全列版須重測覆蓋它。

## 6. 需要先建立或修改的東西（尚未做）

1. **per-estimator fit**：`n_estimators` 在 `TabPFNClassifier` 建構時決定並影響推論。n=4/8 各需一份對應 fitted state（context 與 scaler 相同，只有 estimator 數不同）。可仿 `scripts/run_m5_tabpfn_canonical_full_test.py` 的 `create_real_model` / `fit_or_load`，把 `n_estimators` 參數化。fit 是 100k rows 的一次性動作，可在本機做（與 17-feature fit 同路徑），不算被禁止的本機全 test 推論。
2. **portable worker 的 estimator 參數化**：`scripts/run_m5_tabpfn_portable_shard.py` 目前對應 `n_estimators=1` 的正式契約，需允許 4/8 並確保 checkpoint / resume 行為不變。
3. **portable inputs**：只需 Site 1/2/3 的 17-feature scaled 全列 memmap + metadata（raw_index/anomaly/site_id/building_id），仿 `export_m5_tabpfn_colab_tail.py`，但範圍限這三個 site 的 canonical 列。
4. **記憶體**：`n=8` 的推論記憶體約是 `n=1` 的 8 倍。L4 22GB 下需驗證 microbatch，必要時自動下修（契約允許到 64）。

## 7. 成功判準

沿用 runbook §8：每格（site × n_estimators）唯有本機 durable checkpoint 覆蓋該 site 全列、且 score 全為 finite，才算完成。不得以遠端 heartbeat 或部分 chunk 宣稱完成，不得用抽樣分數代替全列分數。全部完成後產出一張
`n=1 → 4 → 8` 的 per-site ROC-AUC / PR-AUC 增益表。

## 8. 現況快照（2026-07-24 寫作時）

- 未啟動任何 session；`colab sessions` 為 `No active sessions`，兩張 probe L4 已 terminate、CU 停止消耗。
- 本機無執行中的 fit / probe process。
- 已存在可重用資產：canonical 100k-context 17-feature fitted state
  （`data/processed/m5_tabpfn_canonical_full_test_context100000.work/`）、foundation checkpoint、`m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz`（canonical 身分來源）。
- probe 期間新增但尚未 commit 的腳本：`scripts/run_tabpfn_estimator_probe_colab.py`、`scripts/build_tabpfn_estimator_probe_inputs.py`、`scripts/probe_tabpfn_estimators.py`。全列正式版可能改寫或取代這些。

## 9. Decision

計畫已凍結但**尚未執行**：100k context、17 features、Site 1/2/3 全列、`n_estimators ∈ {4,8}`（n=16 已取消），n=1 用既有正式分數當基準。執行順序見 §3.1：**每個 site 切頭/尾兩半、分兩張 L4 並行，先跑完 n=4 再跑 n=8，該 site 兩者完成才換下一個 site，site 序 1→2→3。** 啟動前必須先完成 §6 的 per-estimator fit、worker 參數化與 Site 1/2/3 portable inputs，並以 runbook 的可續跑基建（20k checkpoint + supervisor + keep-alive）執行，因為 ~103 小時的總量遠超 L4 壽命。核心紀律：**全列量測，不得用抽樣結果宣稱增益。**
