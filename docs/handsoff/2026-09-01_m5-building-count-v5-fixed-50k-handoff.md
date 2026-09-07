# M5 Building-Count V5 fixed-50K 監測與匯報交接（第二版）

最後唯讀核對：2026-09-01 18:50（Asia/Taipei）
前一版：`2026-08-31_m5-building-count-v5-fixed-50k-handoff.md`（設計與科學規格請以該版為準，本版不重複）

## 一頁摘要

- 正式排程仍在 WSL Ubuntu 正常執行，**不要停止、重啟、resume、重排或重跑任何 cell**。
- 進度：84 個模型 cell 中 **75 完成、9 待處理**；沒有 FAILED.json。
- 目前 cell：K=50、building seed=0、row seed=1、TabPFN，chunk 86 / 206。
- **K=725、400、200、100 全部完成**。剩下的 9 格全是 K=50 的 TabPFN。
- 預計全部完成：**2026-09-04 17:00 前後**（每格約 8.2 小時，逐格序列執行）。
- Git HEAD 仍為 `4397050376b135ffd1f14d856b6e696767d2588f`，worktree 乾淨。
- tmux session `m5-building-v5-fixed50k`，自 08-26 11:08 起未中斷。

## 你的任務範圍

1. 使用**繁體中文**回覆使用者。
2. 定期做**唯讀**健康檢查並匯報進度。
3. 有新的 COMPLETE.json 時，依使用者要求更新圖表與數據。
4. 除此之外不主動改動任何東西。

## 最快的上手方式

所有監測腳本已存成永久檔案，直接跑即可，不需要自己寫：

```bash
wsl.exe -d Ubuntu -u kuant_kuo -- bash -lc \
  'nice -n 15 /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
   "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5/outputs/scripts/status.py"'
```

輸出包含：supervisor 狀態、逐 K 的完成格數、FAILED.json 檢查、最近 4 筆事件、目前 cell 的 chunk 進度與心跳年齡，以及依實測 chunk 速率算出的完成時間預估。這支腳本只讀檔，安全。

判斷「還活著」的依據：心跳年齡 < 約 300 秒、chunk 數在增加、`nvidia-smi` 顯示 GPU 使用率高。三者其一異常才需要深入查，**但任何情況都不得介入**。

## 絕對禁止

- 停止、重啟、resume、kill tmux、重排 queue、重跑任何 cell。
- 修改 experiment script、frozen manifest、刪除 checkpoint。
- 在 run 中途建立任何 commit（包括空白 commit）— 會改變 HEAD，破壞後續 cell 的 provenance 一致性。
- 用 partial prediction chunks 計算任何指標。
- 對 `data/processed/...` 底下的實驗輸出做任何寫入。

看到 ETA 很長、GPU 使用率波動、或某格「看起來卡住」，都**不是**介入的理由。只能讀 heartbeat、chunk 增長、process 與 FAILED.json。

## 環境

WSL 發行版 Ubuntu，使用者 `kuant_kuo`。

```
正式程式庫  /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k
Python      /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python
輸出根目錄  <repo>/data/processed/m5_building_curve/v5_fixed_50k
```

每個 WSL bash 指令若要 import 專案模組須先 `export PYTHONPATH=src:scripts`（本文件列出的監測腳本不需要）。

Windows 端（2026-08-31 新建，供分析與繪圖用）：

```
C:\Users\User\projects\lead-analysis     uv 專案，Python 3.12.14
```

核心科學套件版本與 WSL 精確對齊（numpy 2.4.6 / pandas 3.0.3 / scipy 1.17.1 /
scikit-learn 1.8.0 等），已驗證兩邊算出的 PR-AUC、ROC-AUC 到 17 位有效數字相同。
Windows 也裝了 VS Code、MiKTeX（渲染 LaTeX 表格用）。

執行方式：

```powershell
uv --directory C:\Users\User\projects\lead-analysis run python <script>
```

## 目前進度

| Budget | Tree | TabPFN | 狀態 |
| --- | ---: | ---: | --- |
| K=725 | 2/2 | 2/2 | 完成 |
| K=400 | 10/10 | 10/10 | 完成 |
| K=200 | 10/10 | 10/10 | 完成 |
| K=100 | 10/10 | 10/10 | 完成（09-01 07:03 補齊） |
| K=50 | 10/10 | 1/10 | b0r0 完成、b0r1 執行中 |

TabPFN 每格穩定 8.15–8.30 小時（實測 17 格全距僅 515 秒）。Tree 每格約 6–9 分鐘。
剩餘 9 格 × 8.2 小時 ≈ 3.1 天。

## 匯報規則

- **只報 chilled water、steam、hot water**，不報 pooled、不報 macro。
- 聚合順序固定：同一 building seed 內先平均 row seeds → 再跨 building seeds 平均 →
  標準誤在 building-seed 層級計算（ddof=1）。K=725 沒有 building seed，誤差跨 5 個 row seed 算。
- 使用者已允許：某 building seed 只有一個 row seed 時可作為暫時估計，但**必須標明**。
- 比較任何一對 Tree/TabPFN 前，必須確認 metadata 與 predictions 陣列一致
  （`context_row_sha256`、`context_label_sha256`、`context_feature_matrix_sha256`、
  `holdout_row_sha256`、`holdout_rows`，以及 `validation_raw_index`、`anomaly`、`meter`
  逐元素相等）。現有腳本都已內建這道閘門。
- 預測欄位名稱不同：Tree 用 `ensemble`，TabPFN 用 `tabpfn`。

## 目前結果

### PR-AUC（平均 ± building-seed 標準誤）

| Meter | 模型 | K=50 | K=100 | K=200 | K=400 | K=725 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chilled water | Tree | 0.5926 ± 0.0190 | 0.5926 ± 0.0255 | 0.6466 ± 0.0299 | 0.7692 ± 0.0156 | 0.7871 ± 0.0036 |
| Chilled water | TabPFN | 待跑 | 0.6199 ± 0.0261 | 0.6749 ± 0.0186 | 0.7474 ± 0.0127 | 0.8138 ± 0.0030 |
| Steam | Tree | 0.5352 ± 0.0582 | 0.6160 ± 0.0499 | 0.6985 ± 0.0269 | 0.7438 ± 0.0053 | 0.7613 ± 0.0030 |
| Steam | TabPFN | 待跑 | 0.6251 ± 0.0173 | 0.7545 ± 0.0171 | 0.7732 ± 0.0152 | 0.7971 ± 0.0092 |
| Hot water | Tree | 0.5316 ± 0.0431 | 0.6687 ± 0.0354 | 0.7232 ± 0.0230 | 0.7998 ± 0.0096 | 0.8236 ± 0.0027 |
| Hot water | TabPFN | 待跑 | 0.6053 ± 0.0430 | 0.6860 ± 0.0215 | 0.7142 ± 0.0183 | 0.8157 ± 0.0024 |

### ROC-AUC

| Meter | 模型 | K=50 | K=100 | K=200 | K=400 | K=725 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chilled water | Tree | 0.9647 ± 0.0017 | 0.9691 ± 0.0021 | 0.9725 ± 0.0021 | 0.9792 ± 0.0013 | 0.9808 ± 0.0003 |
| Chilled water | TabPFN | 0.9648 (n=1) | 0.9679 ± 0.0037 | 0.9754 ± 0.0010 | 0.9783 ± 0.0007 | 0.9833 ± 0.0002 |
| Steam | Tree | 0.9404 ± 0.0024 | 0.9476 ± 0.0014 | 0.9639 ± 0.0027 | 0.9650 ± 0.0024 | 0.9670 ± 0.0009 |
| Steam | TabPFN | 0.9607 (n=1) | 0.9604 ± 0.0054 | 0.9787 ± 0.0016 | 0.9813 ± 0.0012 | 0.9844 ± 0.0005 |
| Hot water | Tree | 0.8922 ± 0.0096 | 0.9273 ± 0.0047 | 0.9328 ± 0.0054 | 0.9456 ± 0.0040 | 0.9494 ± 0.0008 |
| Hot water | TabPFN | 0.8936 (n=1) | 0.8908 ± 0.0112 | 0.9122 ± 0.0104 | 0.9199 ± 0.0009 | 0.9521 ± 0.0009 |

K=50 的 TabPFN 只有 b0r0 一格，**不可當結論**。該格 PR-AUC 為 chilled 0.6389、steam 0.6510、
hot 0.5686（Tree 對應 0.5280 / 0.6511 / 0.5319），chilled water 的 +0.111 是全實驗最大的
單一 context 優勢，但單一 seed 的變異在這份資料裡常達 0.1 以上。

## 產出位置

全部在 `C:\Users\User\Documents\Codex\2026-09-01\wsl-ubuntu-m5-building-count-v5\outputs\`：

| 檔案 | 內容 |
| --- | --- |
| `..._meter_pr_auc_..._2026-09-01.png` / `_data.json` | PR-AUC 三面板圖 |
| `..._meter_roc_auc_..._2026-09-01.png` / `_data.json` | ROC-AUC 三面板圖 |
| `m5_exp_b_building_count_v5_fixed50k_setup.{tex,pdf,png}` | 實驗設定表 |
| `m5_exp_b_building_count_meter_pr_auc_detail.{tex,pdf,png}` | K=50–725 詳細數據表 |
| `m5_exp_b_host_run_performance_cost.md` | 效能 vs 計算成本報告 |
| `scripts/` | 所有生成腳本 |
| `README.md` | 內容索引與重現指令 |

PR-AUC 圖的原始腳本與其 metric 快取留在舊資料夾（快取有 77 個檔，搬走會失效）：

```
C:\Users\User\Documents\Codex\2026-08-26\wsl-ubuntu-m5-building-count-v5\outputs\plot_m5_v5_fixed50k_building_scarcity.py
```

## 重現指令

新的 K=50 cell 完成後，依序重跑即可，全部只讀已發布的 COMPLETE.json：

```bash
# PR-AUC 圖（腳本在舊資料夾，輸出指向新資料夾）
/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
  /mnt/c/.../2026-08-26/.../plot_m5_v5_fixed50k_building_scarcity.py \
  --output /mnt/c/.../2026-09-01/.../m5_exp_b_..._pr_auc_..._2026-09-01.png \
  --summary /mnt/c/.../2026-09-01/.../m5_exp_b_..._pr_auc_..._2026-09-01_data.json

# ROC-AUC 圖
/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
  scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py --output ... --summary ...

# 成本報告
/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python scripts/make_report.py
```

詳細數據表（Windows 端）：

```powershell
uv --directory C:\Users\User\projects\lead-analysis run python scripts\make_meter_table.py
pdflatex -interaction=nonstopmode -output-directory=. m5_exp_b_building_count_meter_pr_auc_detail.tex
```

PDF 轉 PNG：`uvx --from pymupdf python -c "...get_pixmap(dpi=200)..."`。

兩支繪圖腳本都有各自的 metric 快取（`.m5_v5_fixed50k_plot_cache`、
`..._roc`），只會計算新出現的 cell，重跑很快。

## 已知地雷

**1. 圖表更新的時機。** 聚合以 building seed 為單位。K=50 b0r0 完成只代表 b0 有一個
row seed，b0r1 完成後才是完整的一個 seed。用單一 row seed 的點畫圖會誤導，更新前
先確認使用者要不要。

**2. PR-AUC 腳本的副標寫死了 "TabPFN K=50 pending"。** 在
`selection_caption()` 裡。現在 K=50 已有資料，直接重跑會出現「畫了點卻說 pending」
的矛盾。ROC 版已修成依實際選取動態生成，把同樣的修改套到 PR-AUC 版即可：

```python
parts = []
for budget in SCARCITY_BUDGETS:
    count = len(selected[budget])
    parts.append(f"K={budget} n={count}" if count else f"K={budget} pending")
fields = [tree_field, "TabPFN " + parts[0], *parts[1:]]
fields.append("K=725 both n=5 row seeds")
```

**3. Colab 與本機的 cell 混在一起。** 15 格 TabPFN 是在 Colab 跑的，時間資料不可與
本機比較。判別法：本機跑的有 `prediction_chunks` 目錄（206 個 chunk、約 29,400s），
Colab 的沒有該目錄（artifact span 約 2,510s）。所有 Tree cell 都在本機。做任何
**計算成本**分析時務必先排除 Colab 那批；**預測表現**則不受影響，不需排除。

**4. Bash 工具的變數展開問題。** 透過 `wsl.exe -d Ubuntu -- bash -lc '...'` 傳入的
shell 變數（`$VAR`）會被吃掉變成空字串，導致路徑錯誤且不易察覺。**一律使用完整字面
路徑**，或把邏輯寫成 Python 腳本再呼叫。

**5. uv 的 junction。** `uv python install` 在這台會因 Windows Redirection Guard 報
`os error 448`。Python 本體其實已裝好，刪掉 `%APPDATA%\uv\python\cpython-3.12-*`
那個 junction 即可，不需要提權或改系統設定。

## 待辦

1. **PR-AUC 圖尚未與 ROC-AUC 圖同步** — PR-AUC 圖還是 K=50 無 TabPFN 點的版本，
   且副標有地雷 2 的問題。使用者尚未決定是否要現在同步。
2. K=50 b0r1 預計 **09-01 23:29** 完成，屆時 K=50 有第一個完整 building seed。
3. 全部完成後（約 09-04 17:00），可用 `--final` 旗標重跑繪圖腳本產出正式版；
   該旗標會要求所有 K 都有五個 building seed 與兩個 row seed，通過才會輸出。
4. 使用者正在撰寫論文，先前討論過的兩個切入點：
   - steam 在 K=100→200 有 TabPFN 專屬的大幅躍升（+0.1295，5/5 seed 一致，t=9.4），
     Tree 同區間不顯著（t=1.35，僅 3/5 上升）。
   - 以 0.75 為堪用門檻時，steam 的 TabPFN 用 28% 建物池即達標，Tree 要用滿 725 棟。
   - K=50 的 TabPFN 補齊後才能判斷「高報酬區間的下界在哪」。
