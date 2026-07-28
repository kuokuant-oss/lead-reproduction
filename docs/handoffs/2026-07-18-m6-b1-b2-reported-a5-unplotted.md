# M6 — B1/B2 已入報告、A5 仍無圖、target-label 劑量軸未做

Written: 2026-07-18
Branch: `main`
HEAD at handoff: `4363ed0 Record B1/B2 results in the M6 report and correct A5 section 12.3(c)`
Execution environment: Windows PowerShell,repo venv at `.venv\Scripts\python.exe`

> **State.** A0 seen-curve 的 30 個 cell **全部跑完**(前一份 handoff 的工作已完成)。
> B1/B2 結果已寫入 `docs/reports/m6-site-transfer.md`,**並推翻了 §12.3(c) 的一個既有結論**。
> A5 是唯一「完整跑完、當主要結果寫進報告、卻一張圖都沒有」的實驗。
> 權威實驗協議仍是 [`docs/reports/m6-site-transfer.md`](../reports/m6-site-transfer.md)。
> Trust JSON `status` fields over prose,也 trust 本檔的 §6 over 任何記憶。

---

## 1. 本次做完的事

| 項目 | Commit |
|---|---|
| 48 張 M6 圖依實驗組別分四個資料夾;刪除 16 張過時圖 | `b99f18e` |
| B1/B2 結果入報告 §12.8–§12.10;修正 §12.3(c)、§12.4、§12.1 | `4363ed0` |

### 1.1 圖檔分類結果

```text
docs/reports/assets/m6/
  b1-training-meter-curves/  28  plot_m6_b1_curves.py
  seen-vs-unseen/            16  plot_m6_seen_vs_unseen.py
  site-structure/             3  plot_m6_site_structure.py
  b2-matched-support/         1  plot_m6_b2_matched_support.py
```

四個 plot 腳本的 `--out-dir` default 已同步改指向各自資料夾,**否則重跑會在根目錄重新生出散亂的圖**。四個腳本都實跑驗證過路徑可解析。

已刪除的 16 張 `m6_unseen_site_penalty_site_*.png` 是被否決的單折 A0 baseline(trap 19),歷史留在 `ec8a51f`。

### 1.2 資料完成度(逐一讀 JSON `status` 複核)

| Arm | Cells | 備註 |
|---|---|---|
| B1 `a1` / `a2` | 26 / 30 | **缺 seed 999 的 `m400`/`mall`,共 4 個** |
| B1 `a0odd` / `a0even` | 30 / 30 | 完整 3 seeds |
| B2 | 10 | `a1`/`a2`/`a0` × pos410394 × 3 seeds + 舊的 `a0_pos677077_seed42` |

> ⚠️ `.scratch\m6_autorun\m6_a0_curve.log` 最後一筆是 `04:28 ... exit=-1 status=missing`,**看起來像失敗但不是** —— 該 cell 的 JSON 現為 `completed`,是中斷後 resume 補完的。**永遠以 JSON status 為準,不要看 log 尾巴。**

---

## 2. 三個新結果(報告 §12.8–§12.10)

### 2.1 §12.8 — labelling density 未飽和

§150 的 plateau criterion **兩個方向都未達成**。a1 的增量序列 `+0.1107, +0.0154, +0.0129, +0.0342` —— 最後一段(400 → 1,033 meters)反而是第二大。

與 §12.5 併讀是本報告最可操作的一句:

> **Site coverage 已飽和(8→12 sites 值 +0.002),labelling density 沒有(50→1,033 meters 值 +0.17)。**

### 2.2 §12.9 — B2 排除了 anomaly support

關鍵契約事實:**A2 的 source anomalies 恰好等於共同下限 410,394,所以 matching 對 A2 是 no-op**(三個 seed 的 `fit_index_sha256` 完全相同)。真正被砍的只有 A1(904,080 → 410,394,−54.6%)。

**A1 的 macro-site PR 變動 `−0.0001`**,比 seed 散佈小一個數量級。anomaly 數量在 410k–904k 區間不是有效變因。

### 2.3 §12.10 — seen vs unseen

平均 penalty `+0.0485`,但極度集中:16 站有 15 站落在 `−0.04`~`+0.09`,**site 1 一站就 `+0.3218`**。列對齊 gate 通過(0 mismatch)。

---

## 3. 兩處被推翻的舊結論

### 3.1 §12.3(c) 更正

原文:sites 6、15「anomalies 不具 site 特異性,**不應投入 target-site labelling**,全域模型嚴格較優」。

A5 的 gap 是負的(`−0.1038` / `−0.1478`),但 §12.10 的 seen 臂在 **building 多樣性受控**下(725 vs 613 buildings)量到 `+0.0255` / `+0.0802`,**符號翻轉**。

原因:**A5 的 oracle 只用該 site 的 20–60 buildings,seen 臂保留全部 725 buildings 再加該 site。** 負號來自丟失 building 多樣性,不是 anomalies 缺乏 site 特異性。

> **正確陳述:把全域模型換成只用該 site 的模型會變差;在全域模型之上加該 site 的 labels 仍有小幅增益。**

原文已保留並加警示框,未刪除。

### 3.2 Site 2 降級

§12.3(a) 把 sites 1、2 並列為「可由 target labels 恢復」。但 site 2 從 A5 的 `+0.1632` 掉到受控設計的 `+0.0610`(掉近三分之二),**site 1 則兩個設計互相確認**(`+0.2622` / `+0.3218`)。

兩者差 5 倍,§12.11 已把 site 2 的授權改為中等證據、須同報兩個數值。

### 3.3 §12.4 的限制已解除

§12.4 原本聲明「要分離 site-shift 與 building-diversity 需另設 arm,不在 M6 範圍」。**a0 兩折的 seen 臂正是那個 arm。** 已在該節加註。

---

## 4. 可能的方向(依建議順序)

### 方向 A — B3:target-site label 劑量曲線(需新跑)

**問題:在訓練充分的全域模型上,加 k 個 target-site 電表,要多少才夠?**

現有資料**已強烈暗示「幾個就夠」**,但那是副產品不是實驗。我從 a0 折的 `site_allocation` 反推(a0 的 meter budget 跨 16 站分層,故 target site 也貢獻自己的電表):

| Budget | 模型真正看到的 site-1 電表 | 佔該站 | seen | unseen | gap | 佔最終 gap |
|---|---:|---:|---:|---:|---:|---:|
| 50 | **2** | 6.7% | 0.7262 | 0.5414 | +0.1847 | **57%** |
| 100 | 3 | 10% | 0.7631 | 0.5957 | +0.1674 | 52% |
| 400 | 10.5 | 35% | 0.9607 | 0.6080 | +0.3527 | 110% |
| all | 31.5 | 100% | 0.9722 | 0.6504 | +0.3218 | 100% |

7 個有實質 gap 的站,中位數是**用 2–6 個電表(約 4–6%)拿到 64% 的 gap**。

**三個必須說的保留:**

1. **是副產品**。budget 一動,16 站的電表同時變多,無法固定 source 只動 target。
2. **低劑量變異極大**。site 1 在 budget 50 的三個 seed 是 `0.856 / 0.602 / 0.720`,全距 0.25 **比 gap 本身還大**。抽到哪 2 個電表天差地遠。
3. **不單調**。site 8 在 budget 50 是 `−0.054`,到全量才 `+0.074`。

> ⚠️ **計算 target-site 電表數時,要用「單折」配額,不是兩折相加。** 每棟測試樓由**沒訓練過它**的那一折預測,所以模型看到的是該折的 `site_allocation`。我第一次算成 union 而高估了一倍。

**建議設計:**

```text
訓練 = A1 的 8 個 source sites 全部電表
     + 每個 test site 的 k 個電表(取自不列入測試的樓)
測試 = test sites 的其餘樓,k 之間固定同一批測試列
k ∈ {0, 1, 2, 5, 10, 20, all},3 seeds → 21 cells
```

k=0 是 unseen 基線,k=all 逼近 seen 上限。每個 cell 一次評估 8 個 test sites。**約 21 cells × 30–45 分鐘 ≈ 10–16 小時**。低劑量的高變異建議 k≤2 多跑幾個 seed。

**已查證:repo 裡沒有這個實驗。** 查過全部 `data/processed/`(含 `legacy/`)、44 個 scripts、`m6_site_transfer_protocol.py` 的所有 split(只有 a0/a1/a2/a3/a4/a5)、11 份 reports、62 份 handoffs、全部 branch、`notebooks/`、`.scratch/`。

最接近的三個(**都不是**同一件事):

- **M5.1 §4 小樣本標註效率**(`m5_phaseD_deep_comparison.json`,axis `sample_efficiency_fine`)—— 形狀最像,但 split 是 **building-level in-domain**,問的是「總共要多少標註」。**若要複用 scarcity 網格邏輯,從這裡抄。**
- **M5 §4 Label Scarcity**(support 200–10,000)—— 同軸粗網格,一樣 building split。
- **A5 in-site oracle** —— 確實用 target labels,但是該站**一半的樓**,且是**取代**而非疊加全域模型。

> 使用者表示記得做過類似實驗,但上述搜尋未找到。**下一手若要重找,請先問使用者關鍵字**,不要重跑一次同樣的搜尋。

### 方向 B — A5 補圖

A5 完整跑完、是 §12.3 的主要結果、**一張圖都沒有**。資料比預期豐富:每站有 5 模型的完整 ROC/PR curve arrays、per-meter slices、100-bin score histograms(normal/anomaly 分開)。

第一批(**四張全部只讀 JSON,無需重算,無對齊風險**):

1. **Oracle gap forest plot** —— §8 Figure 6,承諾過從未畫。§12.3 三個族群會自己浮出來。**5 個 unscorable sites 必須用不同標記。**
2. **§12.4 方法限制散佈圖** —— gap vs oracle support,標註 site 1(25 buildings/154,576 rows,`+0.2622`)對 site 6(22 buildings/152,016 rows,`−0.1038`)。守的是報告最易被質疑處。
3. **Support gate 圖** —— 顯示 gate 只用 support 決定、與 gap 無關(報告承認 gate 是 post-hoc 訂的)。會揭露一個巧合:**site 4 與 site 11 的 paired-eval anomalies 都恰好 197 個**,成因完全不同。
4. **五模型同號 heatmap** —— §12.3 反覆倚賴「5/5 正」「mixed」,目前只是文字。

第二批(**價值最高但要寫對齊邏輯**):

1. **Score 分布:cross-site vs oracle,同一批列** —— 解釋 **why**。site 1 的 cross-site PR 有 0.7206,但 threshold 0.5 的 recall 只有 `0.0415`、precision 卻有 `0.9365`:分數全擠在 0.5 稍下方,**排序對、位置錯**。Oracle 把同一批列推過門檻(recall 0.9284)。
   ⚠️ 兩側 `score_histograms` 都在,但 **cross-site 是整站、oracle 是 subset,不同列**。要從 NPZ 依 `paired_eval_key_sha256` 重算,**複用 `compare_m6_site_oracle.py` 既有配對邏輯,不要另寫一套**。

### 方向 C — 補 B1 的 4 個 cell(**最便宜**)

seed 999 的 `a1`/`a2` × `m400`/`mall`,約 60 分鐘。補完後 a1/a2 從 2 seeds 升到協議 §143 要求的 3 seeds,§150 的 95% interval 條件才能評估。**不影響 §12.8 的定性結論**(增量遠大於 seed 散佈)。

### 方向 D — 未報告的發現:ensemble 不是最佳模型

驗證於全站資料(B1 mall seed42,a1+a2 涵蓋 16 站):**ensemble 在 10/16 站不是最佳模型。** 多數情況差距 <0.01 無所謂,但:

- **Site 1:HistGBT `0.8035` vs ensemble `0.6504`(+0.153)** —— 偏偏是 penalty 最大的站
- Site 11:lightgbm `0.1871` vs ensemble `0.1485`
- Site 3 的模型分歧達 `0.463`(xgboost 0.3896 vs HistGBT 0.8524)

且**「最佳模型」不穩定**:site 1 在 A5 subset 上是 catboost 贏,全站上是 HistGBT 贏。

這對「要不要維持 frozen ensemble」是直接證據,**報告完全沒提**。但注意 §187:改模型要另開 model-development track,不能混入 frozen-pipeline 主表。**這是一個發現,不是一個改動授權。**

### 不建議

- **§8 Figure 1**(building vs site held-out)—— 已被 seen-vs-unseen 取代,後者列對齊更嚴謹
- **§8 Figure 2**(A3 四折分布)—— 只有 4 個數字,表格已足夠
- **§8 Figure 7/8**(confusion matrices)—— 被方向 B 的第 5 張涵蓋且可讀性遠勝
- **§8 Figure 10**(runtime)—— 無科學價值
- **A4 LOSO** —— §12.5 的飽和證據預測它會重現 A3,報告已建議不執行
- **by_meter slices** —— site 1 只有 2 個 meter 群組,太薄

---

## 5. 仍未解 / 不可解

- **A1/A2 方向比較無法定案**。B2 已排除 anomaly support,但 §12.6 的 test-side prevalence confound(3.601% vs 10.252%)**在 50/50 設計內無解**。要定案須改為兩方向共用 test 集的設計。這條 §306 措辭可能永遠拿不到授權。
- **Sites 13/10 的失敗層級**。§12.10 顯示「看過該 site」不是主因(penalty 僅 `+0.07`/`+0.03`),但 seen 臂的訓練仍來自其他 buildings,**無法排除困難出在 building 之間**。需 within-building 設計,不在 M6 範圍。
- **Aggregate 未產出**。`data/processed/m6_site_transfer_plot_data.json` 尚未由 `scripts/aggregate_m6_site_transfer.py` 產生。

---

## 6. Repo close-out(依 `docs/reference/change-checklist.md` 誠實記錄)

| 檢查項 | 狀態 |
|---|---|
| Slice 開始時開 GitHub issue | **未做**,依 backfill policy 誠實標記。 |
| `Closes #N` commit | **未做**,同上。 |
| README 更新 | **未做。** README 目前有使用者前次 M3 工作的未提交修改,**本 slice 刻意不觸碰以免糾纏**。B1/B2 完成確實影響 milestone 狀態,**這是一項已知未償債務**。 |
| `docs/plans/` 更新 | **未做**,同上理由(`docs/plans/m3-plan.md` 亦有未提交修改)。 |
| ADR | **未新增。** §12.3(c) 的推翻是結果更正,已記於報告;B1 擴充到 a0 折是協議變更,已記於 §4。皆非架構決策。 |
| Handoff | **本檔案。** |
| Result JSON 位置 | 全部在 `data/processed/`。`m6_seen_vs_unseen.json` 為 gitignored,可由 `plot_m6_seen_vs_unseen.py` 重建。 |
| CJK UTF-8 | 本檔以 UTF-8 寫入。 |
| 驗證 gate | `ruff` / `ruff format` 通過;**15 個 M6 測試全過**;markdownlint 通過。`pre-commit --all-files` **刻意未跑**,改為只跑本 slice 的檔案 —— 其 autofix hooks 會改寫 worktree 中他人的未提交工作。 |

### ⚠️ 未提交的他人工作

以下屬**使用者前次 M3 工作**,本 slice 完全未觸碰,仍在 worktree 中未提交:

```text
 M README.md
 M docs/plans/m3-plan.md
 M docs/reports/m3-report.md
?? docs/handoffs/2026-07-16-m3-figures-complete.md
?? docs/handoffs/2026-07-17-m6-a2-a3-a5-complete-b1-partial.md
?? docs/handoffs/2026-07-17-m6-site-transfer-rerun-ready.md
?? docs/metrics/m3-figures.json
?? docs/reports/assets/m3/
?? scripts/aggregate_m6_site_transfer.py
?? scripts/compare_m6_site_oracle.py
?? scripts/plot_m3_figures.py
?? scripts/run_m3_figure_observations.py
?? scripts/run_m3_full_site_transfer.py
?? scripts/run_m6_site_transfer_suite.ps1
?? tests/test_m3_figure_pipeline.py
?? tests/test_m3_full_site_transfer.py
?? tests/test_m6_powershell_suite.py
?? tests/test_m6_site_transfer.py
```

**注意 `scripts/aggregate_m6_site_transfer.py`、`compare_m6_site_oracle.py`、`tests/test_m6_site_transfer.py` 仍是 untracked** —— 報告 §11 已引用它們,且本 slice 跑的 15 個測試就來自那個 untracked 檔案。**這批應盡快提交**,否則報告索引指向 repo 裡不存在的檔案。
