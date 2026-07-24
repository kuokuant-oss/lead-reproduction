# M5 TabPFN 137-feature 全 holdout（n_estimators=8）批次 runbook

這份 reference 描述**如何把 137-feature TabPFN 的分數補滿整個 50/50 building holdout**，
以便在 M3 的四張圖上加一條 137-feature TabPFN 線。給接手者、維運腳本與未來的 AI agent。

搭配閱讀：`m5-tabpfn-colab-dual-shard-runbook.md`（尤其 §1.1 並行實驗線、§1.2 A100 實戰教訓）。

## 1. 目標與可比性契約

四張目標圖：

- `docs/reports/assets/m3/m3_tree_ensemble_by_site_roc_with_tabpfn.png`
- `docs/reports/assets/m3/m3_tree_ensemble_by_site_precision_recall_with_tabpfn.png`
- `docs/reports/assets/m3/m3_feature_engineering_roc_with_tabpfn.png`
- `docs/reports/assets/m3/m3_feature_engineering_precision_recall_with_tabpfn.png`

它們的評測母體是 `building_id % 2 == 1` 的 50/50 building holdout，共 **10,137,155 列**，
順序為 canonical `m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz` 的列序（依 building_id 遞增）。

**可比性要求**：新線必須跑在**完全相同的列**上，否則不能與圖上既有的三條線並排。
本流程的每一步都以此為驗證標的（見 §5）。

不可改動：100k context（`context_sha256 = e9ffe0cf…d2688cbe`）、同一 foundation checkpoint
（SHA-256 `d0d865d5…ea18f3988`）、137 feature 順序取自 fit manifest、`n_estimators=8`、
20,000-row checkpoint、microbatch 上限 20,000、不可 CPU/TPU fallback。

## 2. 已完成與待跑的範圍

| 範圍 | 列數 | 狀態 |
|---|---:|---|
| Site 1 / 2 / 3 | 2,735,231 | ✅ 已完成（per-site shard） |
| 其餘 13 個 site（0, 4–15） | **7,401,924** | 本 runbook 的對象 |
| 合計 | 10,137,155 | |

待跑部分含 514,681 個 anomalies。

## 3. 批次切法（依 building_id）

`scripts/plan_m5_tabpfn_137_remaining_batches.py` 產生
`data/processed/m5_tabpfn_137_remaining_batch_plan.json`。

規則：

- **只切在 building 邊界**：一棟建物不會被拆到兩個 shard，因此每個批次都能獨立合併，
  per-site 曲線也不會因切割而失真。
- 目標每批約 120 萬列；**尾批若小於目標的一半就併入前一批**，避免為了十幾萬列付出一整輪
  上傳／reassemble／安裝的成本。
- 批內再切 head / tail，切點落在 20,000-row checkpoint 邊界。

實際結果（6 批、12 shard）：

| batch | rows | buildings | sites | head / tail |
|---:|---:|---|---|---|
| 0 | 1,207,548 | 1–723 | 0, 4, 5 | 600,000 / 607,548 |
| 1 | 1,215,971 | 725–901 | 5, 6, 7, 8, 9 | 600,000 / 615,971 |
| 2 | 1,203,218 | 903–1015 | 9, 10 | 600,000 / 603,218 |
| 3 | 1,204,681 | 1017–1171 | 10, 11, 12, 13 | 600,000 / 604,681 |
| 4 | 1,216,672 | 1173–1289 | 13, 14 | 600,000 / 616,672 |
| 5 | 1,353,834 | 1291–1447 | 14, 15 | 680,000 / 673,834 |

**注意**：批次的 canonical 位置**不是連續的**，因為已完成的 Site 1/2/3 依 building_id 穿插其中。
匯出器因此以「同一條規則重算列集合」再與計畫的列數／anomaly 數／building 範圍逐項比對，
不符就拒絕寫檔（`export_m5_tabpfn_137_batch_shards.py`）。

## 3.1 執行帳號（務必確認）

| 項目 | 值 |
|---|---|
| Google 帳號 | **`tonykuo210100@gmail.com`**（Colab Pro） |
| WSL HOME | `/home/tonykuo/.colab-tony` |
| 必要環境變數 | `OAUTHLIB_RELAX_TOKEN_SCOPE=1`（缺少時 Google 少給 `drive.file` scope，會在寫檔前中止） |
| GPU | 2 × A100 |

**極易誤讀的地方**：兩個帳號的 HOME 都位於 `/home/tonykuo/` 底下 —— `.colab-tony` 是
`tonykuo210100@gmail.com`，`.colab-hank` 是 `hank0503work@gmail.com`。**路徑中的 `tonykuo`
是 WSL 使用者名稱，不是帳號**；判斷帳號只能看 `.colab-tony` / `.colab-hank` 這一段。

`token.json` 內沒有 `id_token`、`account` 欄位為空，**因此無法從本機檔案直接驗證帳號信箱**。
目前的對應關係依據是 `docs/handoffs/2026-07-24-tabpfn-session-changes.md` 的記載，
加上行為特徵：`.colab-tony` 的每次呼叫都必須帶 `OAUTHLIB_RELAX_TOKEN_SCOPE=1` 才成功，`.colab-hank` 不需要。
若要更換帳號，請同時改 `run_m5_tabpfn_137_batches.ps1` 的 `-ColabHome` 預設值。

## 4. 腳本與執行順序

| 步驟 | 腳本 | 說明 |
|---|---|---|
| 1. 規劃 | `plan_m5_tabpfn_137_remaining_batches.py` | 產生批次計畫 JSON 與時間估算；不匯出、不啟動 |
| 2. 匯出 | `export_m5_tabpfn_137_batch_shards.py --batch N` | 從既有全 test 137 矩陣切片；四項一致性檢查後才寫檔 |
| 3. 執行 | `run_m5_tabpfn_137_batches.ps1` | 依序推進批次，每批 head/tail 並行於兩張 A100 |
| 4. 收尾 | `reap_idle_m5_tabpfn_colab_sessions.ps1` | 關閉閒置 session，避免空燒 CU |
| 5. 合併 | `merge_m5_tabpfn_137_full_test.py` | 併回 10,137,155 列並做完整身分驗證 |

單一 shard 的上線由 `queue_m5_tabpfn_site_shard.ps1` 負責，固定順序為
**取得 A100 → 掛 keep-alive → 部署上傳 → 掛 sync + supervisor**。

### 4.1 執行指令

~~~powershell
# 準備（已完成，重跑需加 --force）
uv run python scripts/plan_m5_tabpfn_137_remaining_batches.py
foreach ($b in 0..5) { uv run python scripts/export_m5_tabpfn_137_batch_shards.py --batch $b }

# 執行（tonykuo210100@gmail.com / HOME .colab-tony、兩張 A100）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_m5_tabpfn_137_batches.ps1

# 收尾
uv run python scripts/merge_m5_tabpfn_137_full_test.py
~~~

## 5. 驗證關卡（每一關都會擋下錯誤，不是事後檢查）

1. **匯出時**：重算的列集合必須與計畫的列數、anomaly 數、building 範圍相符，且不得含已完成的 site。
2. **上傳後**：遠端 reassemble 比對四個 SHA-256（features、metadata、portable fit、foundation checkpoint）。
3. **worker 啟動時**：`--n-features 137`、`--n-estimators 8` 與 fitted state 不符就拒絕啟動。
4. **合併時**：union 必須恰好等於 holdout（無重複、無遺漏），labels/site/building 逐列與 canonical 相符，分數全為 finite。

## 6. 注意事項（承接 dual-shard runbook §1.2）

- **單次上傳超過 64 MB 會失敗**。137 的 features 每個 shard 約 310–350 MB，一定要分段；
  `deploy_m5_tabpfn_site_shard.ps1` 的門檻已設為 64 MB，會自動切段，遠端 reassemble 再驗 SHA。
- **keep-alive 必須在上傳之前掛**，否則數分鐘的上傳期間 session 會被回收，而部署仍會「成功」跑完，
  留下沒有 worker 的 assignment 空燒 CU。
- **remote root 已含 batch 與 feature 數**（`/content/lead_tabpfn_b<N>_<shard>_f137_n8`），
  避免與 17-feature 或 per-site 的 137 shard 撞目錄而讓 `--resume` 誤用別條線的 checkpoint。
- **判斷卡住要看遠端**：本機 chunk 數在部署期間必然是 0，「正在傳」與「已死」無法區分。
  查遠端 `work/` 是否有 `launcher.json` 與 `worker.log`，以及 WSL 內是否真有 `colab upload` 行程。
- **孤兒 assignment 用 endpoint 回收**（`state.client.unassign('<endpoint>')`），
  `colab stop -s <name>` 對已失去本機記錄的 session 無效。回收前需連續多輪確認仍為無名，
  否則會誤殺正在註冊的 session。
- **背景程序一律重導輸出到 log 檔**，`Start-Process -WindowStyle Hidden` 不接 log 會讓失敗完全靜默。

## 7. 時間估算

實測吞吐（A100-SXM4-40GB、137 features、100k context、n=8、microbatch 20000）：
**約 330 rows/s，六個已完成的 137 shard 全部落在 329–333 rows/s，變異極小**。

| 項目 | 值 |
|---|---:|
| 待跑列數 | 7,401,924 |
| 單卡吞吐 | ~330 rows/s |
| 純推論 GPU 時數 | **~6.2 GPU-hours** |
| 兩張 A100 並行的 wall clock | **~3.1 小時** |
| 每批部署額外開銷（上傳 ~660 MB + reassemble + 安裝） | 約 6–10 分鐘 × 6 批 |
| **合計預期 wall clock** | **約 3.7–4.2 小時** |

若遇到 session 回收，supervisor 會自動重建並 `--resume`，只損失當前未完成的 20k chunk
（約 1 分鐘的運算），不會整批重跑。

## 8. Decision

批次以 building_id 切分、每批 head/tail 並行於 `tonykuo210100@gmail.com`（HOME `.colab-tony`）的兩張 A100、批次之間嚴格依序，
完成後合併回 10,137,155 列並通過四關驗證，才可用於重繪 M3 四圖。
預期 wall clock 約 4 小時。所有準備（計畫、12 個 shard、launcher、runner、合併器、收尾 reaper）
在開跑前已全部就緒。
