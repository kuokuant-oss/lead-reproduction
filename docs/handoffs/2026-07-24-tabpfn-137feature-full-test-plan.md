# M5 TabPFN 137-feature 全 test 推論（n_estimators = 1 / 4 / 8）計畫 handoff

這份 handoff 給接手者、維運腳本與未來的 AI agent。它描述**已在本機備妥、但尚未啟動**的第二條 TabPFN 線：同樣的 100k context、
同樣的 50/50 building holdout，但用 137 features，並在 `n_estimators ∈ {1, 4, 8}` 各跑一次全 test 推論。寫法沿用
`docs/reference/m5-tabpfn-colab-dual-shard-runbook.md`。

**目前狀態：本機資產齊備並通過驗證，未上傳、未啟動任何 Colab session、未跑任何全 test 推論。**

## 1. 目標

使用者要的最終比較是四個模型放在同一組列上：

| 線 | features | 來源 | 狀態 |
|---|---|---|---|
| 17-feature Tree Ensemble | 17 | `m3_17_feature_ensemble_predictions.npz` | 已完成（圖上灰線） |
| 137-feature Tree Ensemble | 137 | `m3_figure_predictions_50_50.npz` | 已完成（圖上藍線） |
| TabPFN 17-feature, context 100k, n=1 | 17 | `m5_tabpfn_distributed_context100000_predictions.npz` | 已完成（圖上橘線） |
| **TabPFN 137-feature, context 100k, n=1/4/8** | 137 | 本計畫 | **未跑** |

成品必須能和下列四張既有圖做**完全相等**的對比（同一組列、同一組標籤）：

- `docs/reports/assets/m3/m3_feature_engineering_precision_recall_with_tabpfn.png`
- `docs/reports/assets/m3/m3_feature_engineering_roc_with_tabpfn.png`
- `docs/reports/assets/m3/m3_tree_ensemble_by_site_precision_recall_with_tabpfn.png`
- `docs/reports/assets/m3/m3_tree_ensemble_by_site_roc_with_tabpfn.png`

## 2. 不可改動的契約

- **評測列**：`building_id % 2 == 1` 的 50/50 building holdout 全 10,137,155 列，順序即 canonical
  `m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz` 的列序（依 building_id 遞增）。已驗證該母體 `building_id` 全為奇數、724 棟建物。
- **context**：100,000 列，canonical contract seed 42，與 17-feature 正式 run 同一組列
  （`context_sha256 = e9ffe0cf…d2688cbe`，三份 fit 相同）。
- **features**：17 baseline + 120 `lag_value_*`（`timestamp_merge` regime），順序取自 `fit_manifest.json`，與 M3 樹 ensemble 的 137 相同。
  context 的 NaN fraction 2.1827%（value-change merge miss，合理，TabPFN 與 scaler 都原生處理）。
- **foundation model**：`.tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt`，SHA-256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`。
- **scaler**：以 context 100k 列 fit 的 `StandardScaler`，query 用同一 scaler transform。三份 fit 的 scaler 逐值相同（匯出時已驗證）。
- **唯一變動**：`n_estimators ∈ {1, 4, 8}`（`auto_scale_n_estimators=False`）。
- microbatch：不可 CPU/TPU fallback；下限 64。上限依 GPU 校準，但不得超過 `--checkpoint-rows` 20,000。

## 3. 本機已備妥的資產

### 3.1 三份 fitted state（各 100k context、seed 42）

| n_estimators | work dir | 驗證 |
|---|---|---|
| 1 | `data/processed/m5_tabpfn_137_full_test_context100000.work/` | `effective_estimators = 1` |
| 4 | `data/processed/m5_tabpfn_137_full_test_context100000_n4.work/` | `effective_estimators = 4` |
| 8 | `data/processed/m5_tabpfn_137_full_test_context100000_n8.work/` | `effective_estimators = 8` |

每個 work dir 內含 `model.tabpfn_fit`、`scaler.joblib`、`fit_manifest.json`（含 `n_estimators` 欄位）。
n=1 的路徑刻意不加後綴，維持先前產物可定址。

### 3.2 三組 portable shard（head + tail）

| n_estimators | shard root |
|---|---|
| 1 | `data/processed/m5_tabpfn_137_distributed_context100000/` |
| 4 | `data/processed/m5_tabpfn_137_distributed_context100000_n4/` |
| 8 | `data/processed/m5_tabpfn_137_distributed_context100000_n8/` |

切點沿用 17-feature 正式 run 的 `BOUNDARY = 5,060,000`：head `[0, 5,060,000)` forward、tail `[5,060,000, 10,137,155)` reverse。
每個 shard 目錄含 `features.float32.npy`（float32、137 欄）、`metadata.npz`（raw_index / anomaly / site_id / building_id / global_position）、
`model.portable.tabpfn_fit`（init_params 只改 `model_path` 指向遠端 checkpoint）、`manifest.json`。

**特徵矩陣只算一次**：n=4 / n=8 的 root 以 `--reuse-features-from` 硬連結 n=1 的矩陣與 metadata，連結前會重算兩者的 SHA-256、
比對 context SHA 與 scaler 的 `mean_`/`scale_`。三個 root 的矩陣因此是同一份實體資料，不佔三倍磁碟，也不可能彼此漂移。

## 4. 這次改動的腳本

| 檔案 | 改動 |
|---|---|
| `scripts/run_m5_tabpfn_canonical_full_test.py` | `create_real_model(model_path, seed, n_estimators=1)` |
| `scripts/run_m5_tabpfn_single_context_scaling.py` | `verify_fitted_context(model, rows, requested_estimators=1)` |
| `scripts/fit_m5_tabpfn_137_context100000.py` | `--n-estimators`，work dir 自動加 `_n<k>` 後綴，manifest 記錄 estimator 數並在 fit 後斷言 |
| `scripts/export_m5_tabpfn_137_shards.py` | `--n-estimators`、`--reuse-features-from`（硬連結 + 雙重 digest 證明）、manifest 增 `n_estimators` 與 `source_work_dir` |
| `scripts/run_m5_tabpfn_portable_shard.py` | `--n-features`（預設 17）、`--n-estimators`（預設 1），並寫進 result.json |
| `scripts/verify_m5_tabpfn_137_shards.py` | 新增：列身分、矩陣 digest、portable archive 等價性、本機小量 smoke |
| `tests/test_tabpfn_portable_shard.py` | 補 5 個測試（worker 預設與覆寫、estimator 驗證、reuse 相容性、路徑命名） |

**所有預設值都維持 17-feature / n=1 的既有契約**，因此正式 17-feature run 的行為完全不變。

## 5. 尚未做的事（啟動前必須補齊）

1. **launcher / deploy / sync / supervisor 的 137 變體**：`launch_m5_tabpfn_colab_{head,tail}.py`、
   `deploy_m5_tabpfn_colab_head.ps1`、`sync_m5_tabpfn_colab_{head,tail}.ps1`、`supervise_m5_tabpfn_recovery.py`
   目前寫死 `/content/lead_tabpfn_{head,tail}` 與 17-feature 的參數。137 版需改遠端 root（匯出時已寫成
   `/content/lead_tabpfn_137_{head,tail}`）並在 worker 命令加 `--n-features 137 --n-estimators <k>`。
2. **microbatch 校準**：137 features 的每列成本高於 17，n=8 的記憶體又是 n=1 的 8 倍。上卡後先量能開多大再開跑，規則同
   estimator sweep handoff §3.2。
3. **上傳**：`features.float32.npy` head 約 2.6 GB、tail 約 2.6 GB，需照既有 upload-parts 流程切檔上傳。三個 estimator 共用同一份矩陣，
   同一台 VM 上只需上傳一次。
4. **繪圖**：`plot_m3_figures.py` 與 `plot_m3_tree_ensemble_by_site.py` 目前只吃單一 `--tabpfn-predictions`。要畫出第四條線，
   需擴成可接受多組 TabPFN 分數與圖例（17-feature n=1 vs 137-feature n=1/4/8）。
5. ~~runbook 說明~~：已完成。`docs/reference/m5-tabpfn-colab-dual-shard-runbook.md` 新增 §1.1「並行實驗線」，
   說明兩條實驗線與正式契約的關係；§1 保持原樣未動。

## 5.1 per-site 137-feature shard（2026-07-24 追加，使用者指示）

使用者要求在 17-feature 的 estimator sweep 收工後，加跑一格 **Site 1 / n_estimators=8 / 137 features**，
以便和同一組列的 17-feature 結果直接對照。已備妥、**排在 sweep 之後、不插隊**：

| 項目 | 值 |
|---|---|
| shard root | `data/processed/m5_tabpfn_site1_f137_context100000_n8/` |
| 列 | 289,853（head 140,000 + tail 149,853），與 17-feature 的 Site 1 **完全同一組列** |
| anomalies | 39,135（head 19,865 + tail 19,270） |
| features | 137 |
| fitted state | `m5_tabpfn_137_full_test_context100000_n8.work`（`effective_estimators=8` 已驗證） |
| remote root | `/content/lead_tabpfn_s1_{head,tail}_f137_n8` |

兩個實作重點：

1. **切片而非重算**：`export_m5_tabpfn_site_shards.py` 新增 `--slice-from`，直接依 `site_id` 從既有的全 test 137 矩陣挑出 Site 1 的列。
   那份矩陣已用同一個 context scaler 縮放過，所以切片時不再 transform；重算 value-change 特徵要十幾分鐘，切片是秒級。
2. **remote root 納入 feature 數**（`_f137`）：137 與 17 兩條線列數相同，若共用遠端目錄，`--resume` 會把對方的 checkpoint 當成
   已完成而跳過，**產出另一條線的分數卻標成本條線**。這與 n4/n8 需要隔離是同一類風險。

## 5.2 Site 1 結果（2026-07-24 完成）

同一組 289,853 列（Site 1 在 50/50 holdout 內的全部列），四項對齊證明通過：

| 模型 | ROC-AUC | PR-AUC |
|---|---:|---:|
| TabPFN 17 features, n=1（正式基準） | 0.5447 | 0.2026 |
| TabPFN 17 features, n=8 | 0.6647 | 0.2640 |
| **TabPFN 137 features, n=8** | **0.9972** | **0.9886** |
| （對照）M3 樹 ensemble 17 features | 0.830 | 0.471 |
| （對照）M3 樹 ensemble 137 features | 0.997 | 0.986 |

**兩個結論**：

1. **特徵工程的影響遠大於 estimator 數量**：n=1→8 只換來 ROC +0.12，17→137 features 換來 ROC +0.45，
   把一個近乎亂猜的模型（0.5447）變成近乎完美（0.9972）。
2. **修正先前的歸因**：M5 報告原本把「TabPFN 低於樹 ensemble」歸因於 in-context 100k 對上樹的全量訓練這個結構劣勢。
   但在同樣 137 features 下 TabPFN（0.9972 / 0.9886）已**基本追平**樹 ensemble（0.997 / 0.986），
   代表該落差主要來自**特徵不足**，而非 in-context learning 本身的限制。
   目前只有 Site 1 一個點，Site 2、Site 3 完成前不宣稱此結論普遍成立。

## 5.3 Site 2 結果（2026-07-25 完成）

同一組 1,263,915 列：

| 模型 | ROC-AUC | PR-AUC |
|---|---:|---:|
| TabPFN 17 features, n=1（正式基準） | 0.8435 | 0.2807 |
| TabPFN 17 features, n=8 | 0.8177 (−0.0258) | 0.2101 (−0.0706) |
| **TabPFN 137 features, n=8** | **0.9910 (+0.1475)** | **0.9015 (+0.6209)** |
| （對照）M3 樹 ensemble 137 features | 0.991 | 0.900 |

**這格解釋了 estimator sweep 裡唯一的負增益**：Site 2 在 17 features 下提高 n 會變差（PR 0.2807 → 0.2101），
但同樣 n=8 換成 137 features 後 PR 跳到 0.9015。所以那個退步是「在資訊不足的特徵空間裡，
ensemble 平均掉了單一 estimator 抓到的有限訊號」，**不是這個 site 本質難預測**。

TabPFN 137f（0.9910 / 0.9015）與樹 ensemble 137f（0.991 / 0.900）再次幾乎完全一致，
成為「落差來自特徵不足而非 in-context learning 限制」的第二個證據點。

## 5.4 三個 site 完整結果（2026-07-25）

全部 n_estimators=8，與 17-feature 線跑在**完全相同的列**上：

| site | prevalence | 17f ROC → 137f ROC | 17f PR → 137f PR | 樹 ensemble 137f (ROC / PR) |
|---|---:|---|---|---|
| Site 1 | 13.502% | 0.6647 → **0.9972** | 0.2640 → **0.9886** | 0.997 / 0.986 |
| Site 2 | 6.401% | 0.8177 → **0.9910** | 0.2101 → **0.9015** | 0.991 / 0.900 |
| Site 3 | 0.227% | 0.9800 → **0.9987** | 0.7828 → **0.8586** | 0.999 / 0.886 |

estimator 數的效果（同一 site、17 features、n=1→n=8）：Site 2 的 PR −0.0706。

per-cell 指標檔：`data/processed/m5_tabpfn_site{1,2,3}_f137_n8_sweep_metrics.json`

## 6. 成功判準

沿用 runbook §8：每個 (shard, n_estimators) 唯有本機 durable checkpoint 覆蓋全列、且 score 全為 finite 才算完成。
合併後必須逐 20k span 對齊 canonical `raw_index / anomaly / site_id / building_id`，並確認 raw_index 唯一且與 canonical 同集，
才可與 §1 的四張圖並排。不得以遠端 heartbeat 或部分 chunk 宣稱完成。

## 7. Decision

137-feature 線的**本機準備已完成並通過驗證**：三份 fitted state（n=1/4/8）、三組共用同一份特徵矩陣的 portable shard、
worker 的 feature/estimator 參數化、驗證腳本與測試。**依使用者指示，這裡停住，不上傳、不啟動、不跑全 test 推論。**
啟動時從 §5 的五件事開始，順序：先補 launcher/deploy 變體 → 上卡校準 microbatch → n=1 跑通 head/tail → 再 n=4 → 再 n=8。
