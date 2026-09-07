# M5 Building-Count V5 fixed-50K 交接（第三版．排程已停止）

最後唯讀核對：2026-09-03 16:27（Asia/Taipei）
前一版：`2026-09-01_m5-building-count-v5-fixed-50k-handoff.md`
設計與科學規格以 `2026-08-31` 版為準，本版不重複。

## 一頁摘要

- **正式排程已於 2026-09-03 16:27 依使用者明確指示停止**，使用者要重開機。這**不是**故障。
- 停止時機是乾淨的邊界：K=50 b3r0 於 16:26 完成並通過 pair gate 之後、b3r1 產出任何
  prediction chunk 之前。
- 進度：84 個 cell 中 **81 完成、3 待處理**（K=50 的 b3r1、b4r0、b4r1）。
- 沒有 FAILED.json。Git HEAD 仍為 `4397050376b135ffd1f14d856b6e696767d2588f`，
  tracked worktree 乾淨。
- tmux session `m5-building-v5-fixed50k` 已 kill，scheduler（PID 463）與 cell process
  都已結束，GPU 回到 3% / 1,336 MiB。
- **K=725、400、200、100 全部完成。K=50 的 TabPFN 有 3 個完整 building seed（b0/b1/b2）
  加上 b3r0 單一 row seed。**
- 本次另完成一份 hot water 誤差結構分析，產物已存成永久檔案（見「分析產物」）。

## 現在的狀態：已停止

```
tmux ls                     → no server running
pgrep run_m5_building_count → 無
nvidia-smi                  → 3 %, 1336 MiB
formal root FAILED.json     → 0 個
COMPLETE.json 總數          → 81
git HEAD                    → 4397050376b135ffd1f14d856b6e696767d2588f
git status --porcelain -uno → 空（clean）
```

### b3r1 的殘留目錄（正常且可恢復）

`.../v5_fixed_50k/model_runs/building_seed3/row_seed1/tabpfn_k50_f137/` 裡有：

```
fit.json           39 B
model.tabpfn_fit   14,936,686 B
provenance.json    1,828 B
scaler.joblib      4,943 B
```

只有 context fit 階段的產物，**沒有 prediction chunk、沒有 heartbeat.json、沒有
COMPLETE.json**。cell 腳本帶 `--resume`、scheduler 有 `unit_reused`、繪圖腳本用
`is_complete()` 判斷，所以這個目錄不會被誤讀成完成，恢復時會從頭跑預測。

**不要手動刪除這個目錄**，除非使用者明確要求。

## 你的任務範圍

1. 使用**繁體中文**回覆使用者。
2. 目前沒有活的跑模程序，所以**沒有例行監看工作**。
3. 依使用者要求做唯讀分析、更新圖表與數據。
4. 除此之外不主動改動任何東西。

## 絕對禁止

- 未經使用者明確授權，不得重啟排程、resume 任何 cell、或啟動 Colab session。
- 修改 experiment script、frozen manifest、刪除 checkpoint 或 b3r1 的殘留目錄。
- 在 run 中途建立任何 commit（包括空白 commit）—— 會改變 HEAD，破壞 provenance 一致性。
- 用 partial prediction chunks 計算任何指標。
- 對 `data/processed/...` 底下的實驗輸出做任何寫入。

### Colab（重要）

`colab sessions` 目前列出兩個 A100：

```
gpu-a100-s-kkb-ass1c1-3c72afag08dtv | A100 | GPU
gpu-a100-s-kkb-usc1b2-2b1yo5g4tiuwz | A100 | GPU
```

**使用者已確認這兩個是他其他專案在跑的東西，與本實驗無關。不要停止、不要碰。**
本實驗的 Colab queue（K725 / Phase 3 / K200 3-cells）都已在 08 月結束，sessions 為 0。
前一版交接文件寫的「sessions 必須維持 0」指的是本實驗自己不得再開新 session，
仍然有效。

## 環境

WSL 發行版 Ubuntu，使用者 `kuant_kuo`。

```
正式程式庫  /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k
Python      /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python
輸出根目錄  <repo>/data/processed/m5_building_curve/v5_fixed_50k
raw 資料    /home/kuant_kuo/projects/lead-reproduction/data/raw/m3/
            （train.csv、building_metadata.csv）
```

要 import 專案模組的 WSL 指令須先 `export PYTHONPATH=src:scripts`。

狀態檢查腳本（唯讀，重開機後仍可用；現在會顯示 supervisor 的最後快照而非活程序）：

```bash
wsl.exe -d Ubuntu -u kuant_kuo -- bash -lc \
  'nice -n 15 /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
   "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5/outputs/scripts/status.py"'
```

## 目前進度

| Budget | Tree | TabPFN | 狀態 |
| --- | ---: | ---: | --- |
| K=725 | 2/2 | 2/2 | 完成（TabPFN 另有 r2–r4 在擴增 root，共 5 個 row seed）|
| K=400 | 10/10 | 10/10 | 完成 |
| K=200 | 10/10 | 10/10 | 完成 |
| K=100 | 10/10 | 10/10 | 完成 |
| K=50 | 10/10 | **7/10** | b0/b1/b2 完整、b3r0 完成、b3r1 已中止、b4 未開始 |

剩餘 3 格 × 約 8.2 小時 ≈ 24.6 小時。TabPFN 每格穩定 29,400–29,800 秒。

## 如何恢復排程（需使用者明確授權）

重開機後不會自動恢復。恢復時的原始指令是：

```bash
# 在 tmux session 內執行
cd /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k
export PYTHONPATH=src:scripts
/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
  scripts/run_m5_building_count_v5.py --mode formal --authorize-formal
```

恢復前必須先確認：HEAD 仍是 `4397050…`、worktree clean、`nvidia-smi` 正常、
沒有其他 process 佔用 GPU。scheduler 會自己跳過 81 個已完成的 unit（`unit_reused`），
從 b3r1 重新開始。**不要自行執行，等使用者說。**

## 目前結果

聚合規則：同一 building seed 內先平均 row seeds → 再跨 building seeds 平均 →
標準誤在 building-seed 層級（ddof=1）。K=725 沒有 building seed，誤差跨 5 個 row seed。

### PR-AUC

| Meter | 模型 | K=50 | K=100 | K=200 | K=400 | K=725 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chilled water | Tree | 0.5926 ± 0.0190 | 0.5926 ± 0.0255 | 0.6466 ± 0.0299 | 0.7692 ± 0.0156 | 0.7871 ± 0.0036 |
| Chilled water | TabPFN | **0.5813 ± 0.0337 (n=3)** | 0.6199 ± 0.0261 | 0.6749 ± 0.0186 | 0.7474 ± 0.0127 | 0.8138 ± 0.0030 |
| Steam | Tree | 0.5352 ± 0.0582 | 0.6160 ± 0.0499 | 0.6985 ± 0.0269 | 0.7438 ± 0.0053 | 0.7613 ± 0.0030 |
| Steam | TabPFN | **0.5686 ± 0.0657 (n=3)** | 0.6251 ± 0.0173 | 0.7545 ± 0.0171 | 0.7732 ± 0.0152 | 0.7971 ± 0.0092 |
| Hot water | Tree | 0.5316 ± 0.0431 | 0.6687 ± 0.0354 | 0.7232 ± 0.0230 | 0.7998 ± 0.0096 | 0.8236 ± 0.0027 |
| Hot water | TabPFN | **0.4839 ± 0.1081 (n=3)** | 0.6053 ± 0.0430 | 0.6860 ± 0.0215 | 0.7142 ± 0.0183 | 0.8157 ± 0.0024 |

### ROC-AUC

| Meter | 模型 | K=50 | K=100 | K=200 | K=400 | K=725 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chilled water | Tree | 0.9647 ± 0.0017 | 0.9691 ± 0.0021 | 0.9725 ± 0.0021 | 0.9792 ± 0.0013 | 0.9808 ± 0.0003 |
| Chilled water | TabPFN | **0.9554 ± 0.0098 (n=3)** | 0.9679 ± 0.0037 | 0.9754 ± 0.0010 | 0.9783 ± 0.0007 | 0.9833 ± 0.0002 |
| Steam | Tree | 0.9404 ± 0.0024 | 0.9476 ± 0.0014 | 0.9639 ± 0.0027 | 0.9650 ± 0.0024 | 0.9670 ± 0.0009 |
| Steam | TabPFN | **0.9344 ± 0.0244 (n=3)** | 0.9604 ± 0.0054 | 0.9787 ± 0.0016 | 0.9813 ± 0.0012 | 0.9844 ± 0.0005 |
| Hot water | Tree | 0.8922 ± 0.0096 | 0.9273 ± 0.0047 | 0.9328 ± 0.0054 | 0.9456 ± 0.0040 | 0.9494 ± 0.0008 |
| Hot water | TabPFN | **0.8379 ± 0.0541 (n=3)** | 0.8908 ± 0.0112 | 0.9122 ± 0.0104 | 0.9199 ± 0.0009 | 0.9521 ± 0.0009 |

K=50 的 Tree 是 n=5，TabPFN 是 n=3，兩者不同。**同 seed 集（b0/b1/b2）的對照**：

| Meter | 指標 | Tree (b0–b2) | TabPFN (b0–b2) | 差 |
| --- | --- | ---: | ---: | ---: |
| Chilled water | PR-AUC | 0.5823 ± 0.0305 | 0.5813 ± 0.0337 | −0.0010 |
| Steam | PR-AUC | 0.6020 ± 0.0578 | 0.5686 ± 0.0657 | −0.0334 |
| Hot water | PR-AUC | 0.5390 ± 0.0781 | 0.4839 ± 0.1081 | −0.0551 |
| Chilled water | ROC-AUC | 0.9647 ± 0.0021 | 0.9554 ± 0.0098 | −0.0093 |
| Steam | ROC-AUC | 0.9429 ± 0.0032 | 0.9344 ± 0.0244 | −0.0085 |
| Hot water | ROC-AUC | 0.8932 ± 0.0155 | 0.8379 ± 0.0541 | −0.0553 |

### K=50 逐 building seed 的 PR-AUC（TabPFN − Tree）

| Meter | b0 | b1 | b2 | b3（僅 r0）|
| --- | ---: | ---: | ---: | ---: |
| Chilled water | **+0.1128** | −0.0313 | −0.0846 | −0.0191 |
| Steam | +0.0116 | −0.0483 | −0.0635 | −0.0086 |
| Hot water | +0.0261 | −0.1342 | −0.0572 | −0.0812 |

**b0 是唯一 TabPFN 獲勝的 seed。** 09-01 曾記錄的「chilled water +0.113 是全實驗最大單一
context 優勢」在加入 b1/b2 後被抵銷到 −0.001。b3r0 也是三個 meter 全負。
目前的走向是：**K=50 時 TabPFN 在三個 meter 上全面落後，而 K≥100 時 chilled 與 steam
都是 TabPFN 領先**，也就是「TabPFN 優勢的下界約在 K=50–100 之間」。
b4 兩格跑完才能定稿。

## 本次新增：hot water 誤差結構分析

### 必須先知道的三個事實

1. **K=725 的 725 是「context 來源建築池」，不是評估集。** holdout 固定 4,102,084 列、
   **296 棟建築**；其中 hot water（meter 3）只有 **73 棟**（65 棟有異常）、636,121 列、
   90,691 個異常。chilled water 252 棟、steam 162 棟。
2. **訓練池與評估集完全不重疊**：296 棟 holdout 建築中，0 棟出現在 725 棟訓練池內。
   這是嚴格的 building-disjoint 設計。
3. **`building_id` 與 `site_id` 都不是特徵**（137 維特徵裡沒有，它們只是 join
   metadata 與天氣的鍵）。`year_built`、`log_square_feet`、`primary_use_enc`、
   `floor_count` 是特徵。所以任何「模型記住了某棟建築」的解釋都不成立；
   模型只能透過建築屬性泛化到沒看過的建築。

### K=725 hot water 分層 PR-AUC（總分打平，子群不同）

總計 Tree **0.8236 ± 0.0027**、TabPFN **0.8157 ± 0.0024**，差 −0.0079 ± 0.0048，
**五個 row seed 只有 3/5 同向**（總分既打平又不穩定，而多數子群方向穩定 5/5）。

每張表都涵蓋全部 73 棟 / 636,121 列 / 90,691 正樣本，缺值自成類別，
沒有異常的組別也列出（只提供負樣本，PR-AUC 未定義）。

**依 site**

| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | Tree | TabPFN | 差 | SE | 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Site 1 | 7 | 61,469 | 9.66% | 2,933 | 58,536 | 0.324 | 0.534 | +0.210 | 0.015 | 5/5 |
| Site 2 | 27 | 236,727 | 37.21% | 59,827 | 176,900 | 0.885 | 0.877 | −0.008 | 0.007 | 4/5 |
| Site 7 | 2 | 17,562 | 2.76% | 0 | 17,562 | — | — | — | — | — |
| Site 10 | 5 | 43,263 | 6.80% | 5,065 | 38,198 | 0.553 | 0.405 | −0.149 | 0.017 | 5/5 |
| Site 11 | 2 | 17,468 | 2.75% | 1 | 17,467 | 0.004 | 0.000 | −0.004 | 0.003 | 5/5 |
| Site 14 | 28 | 244,835 | 38.49% | 22,792 | 222,043 | 0.739 | 0.768 | +0.029 | 0.005 | 5/5 |
| Site 15 | 2 | 14,797 | 2.33% | 73 | 14,724 | 0.780 | 0.668 | −0.111 | 0.026 | 5/5 |
| 合計 | 73 | 636,121 | 100% | 90,691 | 545,430 | 0.824 | 0.816 | −0.008 | 0.005 | 3/5 |

**依 primary use**

| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | Tree | TabPFN | 差 | SE | 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Education | 37 | 322,635 | 50.72% | 48,733 | 273,902 | 0.870 | 0.843 | −0.027 | 0.008 | 5/5 |
| Entertainment/public assembly | 7 | 60,141 | 9.45% | 2,187 | 57,954 | 0.842 | 0.858 | +0.016 | 0.012 | 2/5 |
| Food sales and service | 2 | 17,564 | 2.76% | 192 | 17,372 | 0.808 | 0.702 | −0.106 | 0.019 | 5/5 |
| Healthcare | 2 | 16,582 | 2.61% | 5,523 | 11,059 | 0.494 | 0.675 | +0.180 | 0.018 | 5/5 |
| Lodging/residential | 6 | 52,385 | 8.24% | 2,978 | 49,407 | 0.795 | 0.927 | +0.132 | 0.024 | 5/5 |
| Office | 15 | 131,689 | 20.70% | 28,369 | 103,320 | 0.879 | 0.872 | −0.008 | 0.011 | 4/5 |
| Public services | 3 | 26,351 | 4.14% | 1,398 | 24,953 | 0.953 | 0.938 | −0.015 | 0.007 | 4/5 |
| Technology/science | 1 | 8,774 | 1.38% | 1,311 | 7,463 | 0.410 | 0.269 | −0.141 | 0.020 | 5/5 |
| 合計 | 73 | 636,121 | 100% | 90,691 | 545,430 | 0.824 | 0.816 | −0.008 | 0.005 | 3/5 |

**依 square feet 四分位**（零缺值）

| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | Tree | TabPFN | 差 | SE | 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9,703–49,662 sq ft | 18 | 157,906 | 24.82% | 36,180 | 121,726 | 0.917 | 0.865 | −0.053 | 0.009 | 5/5 |
| 49,662–87,673 sq ft | 18 | 156,919 | 24.67% | 19,945 | 136,974 | 0.816 | 0.847 | +0.031 | 0.011 | 5/5 |
| 87,673–152,559 sq ft | 18 | 157,442 | 24.75% | 23,919 | 133,523 | 0.841 | 0.842 | +0.001 | 0.008 | 2/5 |
| 152,559–387,500 sq ft | 19 | 163,854 | 25.76% | 10,647 | 153,207 | 0.468 | 0.502 | +0.034 | 0.003 | 5/5 |
| 合計 | 73 | 636,121 | 100% | 90,691 | 545,430 | 0.824 | 0.816 | −0.008 | 0.005 | 3/5 |

**依 year built**（缺值 49.42% 的列）

| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | Tree | TabPFN | 差 | SE | 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1900–1949 | 7 | 58,598 | 9.21% | 13,129 | 45,469 | 0.968 | 0.848 | −0.120 | 0.013 | 5/5 |
| 1950–1969 | 16 | 140,390 | 22.07% | 35,639 | 104,751 | 0.885 | 0.901 | +0.016 | 0.006 | 4/5 |
| 1970–1989 | 4 | 35,128 | 5.52% | 9,539 | 25,589 | 0.804 | 0.881 | +0.078 | 0.015 | 5/5 |
| 1990–2019 | 10 | 87,655 | 13.78% | 4,078 | 83,577 | 0.581 | 0.686 | +0.105 | 0.024 | 5/5 |
| 缺值 | 36 | 314,350 | 49.42% | 28,306 | 286,044 | 0.716 | 0.700 | −0.015 | 0.005 | 5/5 |
| 合計 | 73 | 636,121 | 100% | 90,691 | 545,430 | 0.824 | 0.816 | −0.008 | 0.005 | 3/5 |

**依 floor count**（缺值 80.78% 的列，實務上不可用）

| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | Tree | TabPFN | 差 | SE | 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2–4 層 | 3 | 25,793 | 4.05% | 3,329 | 22,464 | 0.652 | 0.476 | −0.176 | 0.023 | 5/5 |
| 4–6 層 | 3 | 26,264 | 4.13% | 1,670 | 24,594 | 0.311 | 0.428 | +0.117 | 0.032 | 5/5 |
| 6–9 層 | 8 | 70,237 | 11.04% | 2,999 | 67,238 | 0.312 | 0.369 | +0.057 | 0.013 | 5/5 |
| 缺值 | 59 | 513,827 | 80.78% | 82,693 | 431,134 | 0.848 | 0.851 | +0.003 | 0.005 | 3/5 |
| 合計 | 73 | 636,121 | 100% | 90,691 | 545,430 | 0.824 | 0.816 | −0.008 | 0.005 | 3/5 |

### 跨 budget 的 hot water 落差來源

hot water 的異常有 **89.1% 落在 `meter_reading = 0`**，但整體
`P(anomaly | reading = 0)` 只有 0.477（有 88,610 列是零讀數卻正常）。

用 AP 的可加分解（AP = 正樣本在其排名處 precision 的平均，故逐正樣本貢獻精確可加，
程式內對 sklearn 檢核到 1e-9），落差幾乎全部來自零讀數列：

| K | 總落差 | 零讀數列的貢獻 | 佔總落差 |
| ---: | ---: | ---: | ---: |
| 100 | −0.0634 ± 0.0501 | −0.0613 | 96.6% |
| 200 | −0.0372 ± 0.0270 | −0.0363 | 97.7% |
| 400 | −0.0856 ± 0.0243 | −0.0859 | 100.4% |
| 725 | −0.0079 ± 0.0048 | −0.0127 | （非零讀數轉正 +0.0048，故超過 100%）|

只有 K=400 的落差在 seed 層級顯著；但 K=400 的 10 格全部為負（符號檢定 p=0.002），
K=200 是 8/10、K=100 是 6/10。

其他佐證（cluster bootstrap 以建築為重抽單位、配對 Wilcoxon、一階隨機優越）
已存檔備審稿人詢問，數值在 `analysis_2026-09-03/data/standard_attribution.json`。

## 分析產物

永久位置：`docs/handsoff/analysis_2026-09-03/`

| 檔案 | 內容 |
| --- | --- |
| `scripts/k725_all_dimensions.py` | 上面五張分層表的產生腳本（最完整，優先用這支）|
| `scripts/k725_five_seed_hotwater.py` | K=725 五個 row seed 的 hot water 分析（含 validate_k725 閘門）|
| `scripts/k50_cell_readonly_metrics.py` | 單一 cell 的 PR/ROC 計算，用法 `<budget> <bseed> <rseed,逗號分隔>` |
| `scripts/precompute_positive_contrib.py` | 逐正樣本 AP 貢獻預算（換分箱不需重讀 predictions）|
| `scripts/m5_hot_water_gap_composition_pr_auc.py` | K=400 落差組成圖（依作圖規範 v0.3）|
| `scripts/standard_attribution*.py` | Shapley / dominance / cluster bootstrap / 隨機優越 |
| `data/hotwater_sideinfo.npz` | hot water 636,121 列的 y / building_id / site_id / meter_reading / hour / primary_use |
| `data/positive_contrib.npz` | K=400 十格的逐正樣本 AP 貢獻差 |
| `data/k725_hotwater_per_building.json` | K=725 逐建築的 Tree/TabPFN AP 與零讀數診斷 |
| `data/*.json` | 各 budget 的分組結果與歸因中間檔 |

**腳本裡的 `SCRATCH` 常數指向舊的 session 暫存目錄，重開機後不存在。**
執行前把該常數改成
`/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-03/data`
即可。

圖表與報告產物仍在
`C:\Users\User\Documents\Codex\2026-09-01\wsl-ubuntu-m5-building-count-v5\outputs\`
（清單見前一版交接文件）。

## 匯報規則

- **只報 chilled water、steam、hot water**，不報 pooled、不報 macro。
- 聚合順序：building seed 內平均 row seeds → 跨 building seeds 平均 →
  標準誤在 building-seed 層級（ddof=1）。K=725 誤差跨 5 個 row seed。
- 某 building seed 只有一個 row seed 時可作暫時估計，但**必須標明 n**。
- 比較任何 Tree/TabPFN 配對前，必須確認 `context_row_sha256`、`context_label_sha256`、
  `context_feature_matrix_sha256`、`holdout_row_sha256`、`holdout_rows`，以及
  `validation_raw_index`、`anomaly`、`meter` 逐元素相等。現有腳本都內建這道閘門。
- 預測欄位名稱：Tree 用 `ensemble`，TabPFN 用 `tabpfn`。
- 使用者要求**簡潔**。報 PR-AUC 比較就好，不要自行加上未被要求的衍生指標。

## 已知地雷

**1. 分層表必須涵蓋全部列，缺值必須自成類別。** PR-AUC 是正負樣本一起評的指標。
只數「有異常的建築」會漏掉只提供負樣本的建築（hot water 有 8 棟，含 Site 7 整個
兩棟、17,562 列、零異常），那些列仍然參與 pooled 排序並可能成為誤報。
metadata 缺值（year_built 缺 49.42% 的列、floor_count 缺 80.78%）要當成並列類別報出。

**2. 不要用「某組的異常集中在少數建築」去質疑該組 PR-AUC 的可靠性。** 那是把抽樣單位
（建築）與評估單位（列）混淆。holdout 固定不變，唯一的變異來源是 context 抽取，
所以五個 seed 的 SE 就是該量測的不確定性。建築集中度只影響**能不能用該組的標籤下
一般化結論**（例如 Healthcare 有 97.7% 的異常來自一棟，就不能叫它「用途類別的發現」），
不影響數字本身。

**3. 不要用 site 去「控制」year_built。** `site_id` 不是特徵，`year_built` 是。
兩者相關（sites 10/11/14 完全沒有 year_built），但年份是模型真的看得到的量。
可以報告年份分層，不能宣稱年份是成因。

**4. 不要對 AP 做 coalition 型的 Shapley 歸因。** `v(S)=AP(S)` 會改變評估母體的
盛行率，而 AP 對盛行率敏感，結果會與固定完整排序上的可加分解**方向相反**
（K=400 時 Shapley 把最大負值給 A 群零讀數 −0.0437，可加分解卻是 C 群 −0.0773）。
用固定排序上的可加分解。

**5. PR-AUC 圖腳本的副標寫死 "TabPFN K=50 pending"**（在 `selection_caption()`）。
K=50 已有資料，直接重跑會出現「畫了點卻說 pending」的矛盾。ROC 版已改成動態生成，
把同樣的修改套上即可。

**6. Colab 與本機的 cell 混在一起。** 15 格 TabPFN 在 Colab 跑，時間資料不可與本機比較。
判別法：本機的有 `prediction_chunks` 目錄（206 chunk、約 29,400–29,800s），Colab 的沒有。
所有 Tree cell 都在本機。做**計算成本**分析必須先排除 Colab 那批；**預測表現**不受影響。

**7. Bash 工具的變數展開問題。** 透過 `wsl.exe -d Ubuntu -- bash -lc '...'` 傳入的
`$VAR` 會變成空字串，路徑錯誤且不易察覺。**一律用完整字面路徑**，或寫成 Python 腳本。
另外 Git Bash 會把 `/mnt/c/...` 改寫成 `C:/Program Files/Git/mnt/c/...`，
所以呼叫 `wsl.exe` 要用 PowerShell 工具，不要用 Bash 工具。

**8. 作圖規範 v0.3。** 新圖一律依 `docs/policies` 的作圖規範（本次踩過的：主標題須為
名詞片語不可問句、subtitle 只有一行、字級 16/10.5/11.5、多列圖只有最底列顯示 x 刻度
且全圖只有一個共用軸名、只標少數需精確讀取的值、不得自建與模型色衝突的類別 palette
（只用 ink token）、bar 自 0 起算、手動 `subplots_adjust` 不依賴 `bbox_inches="tight"`）。
數值標籤放在座標軸**外**的固定欄，才能同時保住 8% headroom 又不互撞。

## 待辦

1. **重開機後恢復排程**（需使用者授權）—— 剩 K=50 的 b3r1、b4r0、b4r1，約 24.6 小時。
2. b4 兩格完成後，K=50 才有五個完整 building seed，可用 `--final` 旗標重跑繪圖腳本
   產出正式版（該旗標要求所有 K 都有 5 個 building seed 與 2 個 row seed）。
3. **PR-AUC 圖尚未與 ROC-AUC 圖同步**（地雷 5）。使用者尚未決定何時同步。
4. K=725 hot water 案例的圖還沒畫。使用者已定案：最多兩張圖、依作圖規範、
   用分層表那種完整口徑呈現，不要用單一建築當案例。
5. 論文的三個切入點：
   - steam 在 K=100→200 有 TabPFN 專屬的大幅躍升（+0.1295，5/5 seed，t=9.4），
     Tree 同區間不顯著（t=1.35，3/5 上升）。
   - 以 0.75 為堪用門檻時，steam 的 TabPFN 用 28% 建物池即達標，Tree 要用滿 725 棟。
   - K=50 目前顯示 TabPFN 三個 meter 全面落後（b0 是唯一例外），
     等 b4 補齊即可定調「TabPFN 優勢的下界」。
