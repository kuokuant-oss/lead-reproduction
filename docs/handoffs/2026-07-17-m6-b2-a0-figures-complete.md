# M6 — B2 與 A0 完成、48 張圖歸檔;B1 缺 4 cells

Written: 2026-07-17
Branch: `main`
HEAD at handoff: `ec8a51f Add M6 site-transfer report and figures`
Execution environment: Windows PowerShell,repo venv at `.venv\Scripts\python.exe`

> **State.** 62 個 completed cells。B2 全部 9 個完成,A0 full-budget reference 完成並**逐位重現 M3**。
> B1 停在 26/30(seed 999 缺 `m400`/`mall`,使用者決定跳過先跑 B2)。A4 仍 gated,未執行。
> **Aggregate 未跑**,`m6_site_transfer_plot_data.json` 不存在。
>
> 本檔接續 [`2026-07-17-m6-a2-a3-a5-complete-b1-partial.md`](2026-07-17-m6-a2-a3-a5-complete-b1-partial.md)。
> 該檔的 protocol 邊界與 traps 仍然有效,新增/修正見 §5。
> 權威實驗協議仍是 [`docs/reports/m6-site-transfer.md`](../reports/m6-site-transfer.md)。
> Trust JSON `status` fields over prose。

---

## 1. Disk 狀態

| Cell | 數量 | 說明 |
|---|---:|---|
| a2 | 2 | canonical + sourcecal |
| a3 | 8 | 四 folds × canonical/sourcecal |
| a5 | 16 | 每 site 一個 in-site oracle(+ 16 個 paired oracle 檔) |
| b1 | 26 | **缺 seed 999 的 `m400` / `mall`(各 a1/a2 共 4 cells)** |
| b2 | 9 | 3 seeds × {a0, a1, a2} @ `N_pos = 410,394` |
| **a0** | **1** | **`m6_site_transfer_b2_a0_pos677077_seed42`** — full building-held-out reference |
| a4 | 0 | gated,不執行(理由見 §3) |

重新確認:

```powershell
Set-Location C:\Users\tonykuo\projects\lead-reproduction
Get-ChildItem data\processed -File -Filter "m6_site_transfer_*.json" |
  Where-Object { $_.Name -notlike "*_manifest.json" -and $_.Name -ne "m6_site_transfer_plot_data.json" } |
  ForEach-Object { Get-Content -Raw -Encoding UTF8 -LiteralPath $_.FullName | ConvertFrom-Json } |
  Group-Object cell | Select-Object Name, Count
```

---

## 2. 本 slice 的主要發現

### 2.1 A0 逐位重現 M3(最重要的方法突破)

M3 的 `m3_figure_predictions_50_50.npz` **無法 join 回 site**:它的 `validation_raw_index` 與預測陣列順序不一致(逐列僅 88.35% 相符,per-site prevalence 全部塌成全域平均 6.29%)。`anomaly` 與 `ensemble` 彼此對齊,所以 **M3 報告的 pooled 數字有效**,但 per-site 無法還原。M3 是為 building-held-out 研究寫的,從不需要這個 join——**這不是 M3 的缺陷,是它沒被設計要做的事**。

解法是 additive 的,**不碰 M3 任何檔案**:M6 protocol 的 `a0_building_mod2` 與 M3 同一條分割規則。

```powershell
.\.venv\Scripts\python.exe scripts\run_m6_site_transfer.py `
  --experiment b2 --base-split a0 --positive-budget 677077 --selection-seed 42 `
  --manifest-in data\processed\m6_site_transfer_b2_a0_pos677077_seed42_manifest.json `
  --out data\processed\m6_site_transfer_b2_a0_pos677077_seed42.json `
  --predictions-out data\processed\m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz
```

驗證(五個數字**全部**逐位吻合):

| | M3 | A0 |
|---|---:|---:|
| PR-AUC | 0.9303 | 0.9303 |
| ROC-AUC | 0.9918 | 0.9918 |
| Precision@0.5 | 0.7317 | 0.7317 |
| Recall@0.5 | 0.9483 | 0.9483 |
| F1@0.5 | 0.8260 | 0.8260 |

分割亦逐項相同:725/724 buildings、10,078,945 / 10,137,155 rows、677,077 / 637,397 anomalies。耗時 16.2 min。

**A0 提供 M3 給不了的東西**:`site_overlap = 16 sites`、`slices.by_site_id` 涵蓋全部 16 站、含 `site_id` 的 NPZ。

### 2.2 A0 是報告 §12.4 說「M6 範圍外」的那個 arm——它一直在 §86 裡

§12.4 記錄了 A5 的限制:oracle 以「同 site 的 20–77 buildings」置換「他 site 的 1,031+ buildings」,site 身分與 building 多樣性無法分離,「要分離需另設 arm(cross-site 全資料 + target-site labels),不在 M6 範圍」。

**A0 就是那個 arm**(全部 16 站的偶數樓 = cross-site 全資料 + target-site labels),而且它的 test 列與 A5 的 paired oracle **逐位相同**(16 站全部驗證通過)。三臂比較因此合法:

| Site | cross | oracle | a0 | a0−cross | oracle 訓練樓數 |
|---|---:|---:|---:|---:|---:|
| Robin (1) | 0.721 | 0.983 | **0.986** | +0.265 | 25 |
| Bobcat (10) | 0.572 | 0.627 | **0.794** | +0.222 | 15 |
| Fox (2) | 0.719 | 0.882 | **0.900** | +0.181 | 68 |
| Moose (7) | 0.841 | 0.669 | **0.884** | +0.043 | 7 |
| Peacock (6) | 0.854 | 0.750 | **0.895** | +0.041 | 22 |
| Hog (13) | 0.619 | 0.669 | **0.637** | +0.018 | 77 |
| Cockatoo (15) | 0.906 | 0.759 | 0.891 | −0.016 | 62 |

**這推翻了報告 §12.3 的兩處分類(尚未寫入報告,見 §4):**

1. **Peacock / Moose 不是「anomalies 非 site-specific」。** a0 兩者都贏 cross;oracle 輸是因為只有 22 / 7 棟樓可訓練。**只有 Cockatoo 撐住原分類**(a0 ≈ cross)。
2. **Bobcat 不是「target labels 無法恢復」。** a0 到 0.794(+0.222),它只有 15 棟樓,oracle 同樣被多樣性餓死。

**站得住:Hog 確認本質困難。** 三臂全部 0.62–0.67,連目前能給的最大資訊量都不行。

### 2.3 B2:異常預算解釋不了方向差,而且方向差永遠不可識別

| | Pooled PR | 來源異常 | Test prev |
|---|---:|---:|---:|
| A1 canonical | 0.6447 | 904,080 | 3.60% |
| B2 A1 matched | **0.6514** | 410,394 | 3.60% |
| A2 canonical / matched | 0.8954 | 410,394 | 10.25% |
| B2 A0 | 0.9283 | 410,394 | 6.29% |

**砍掉 A1 55% 的異常,分數微升 +0.0067。** 方向差 +0.2507 → +0.2440,異常預算只解釋 2.7%。

**但 B2 無法救回 A1 vs A2 的比較,而且永遠不會。** §158 規定 test rows 維持 natural prevalence,所以測試側 3.60% vs 10.25% 的混淆原封不動,而 PR-AUC 的基線就等於 prevalence。**§306 的措辭是定案的不可用,不是待辦。**

內部驗證:A2 三個 seed 分數完全相同(spread 0.0000),因為 410,394 就是它的全部異常,抽樣是 no-op。

### 2.4 異常供給在約 10 萬就飽和——四個獨立證據

| 證據 | 動了什麼 | 買到 |
|---|---|---:|
| §12.5(A1→A3) | source sites 8→12,異常 +37% | +0.002 |
| B2 | 異常 −55% | +0.0067 |
| B1 曲線(a1) | **87k → 904k,異常 ×10** | **+0.002** |
| A0 | 410k → 677k | +0.0009 |

a1 在 **87k 異常 / 100 條電表**就飽和(0.5401 → 0.6430 → 之後 10 倍資料只 +0.002)。a2 更早,在 128k 見頂後**微降**(0.9058 → 0.8954)——論文 future work 問的「或開始下降」的實例。

**已收回的宣稱**:我曾從表格推論「異常數飽和但電表多樣性還有用(+0.033)」。畫成圖後看到那個 +0.033 **落在平原自己的抖動裡**(0.606–0.645,幅度 0.039),兩個變因分不開。

### 2.5 Crow 與 Bear 是 building 級退化案例

**Crow (site 11)**:5 棟樓,**building 1028 一棟佔全站 4,978 個異常的 92.2%**(異常率 19.599%),其餘四棟合計 389 個(其中兩棟各 3 個和 2 個)。96% 的異常在偶數樓 → A5/a0 的 train-even/test-odd 設計下,eval 側只剩 **197** 個異常。

**Bear (site 4)**:**91 棟樓**,但 top-1 佔 **87.2%**、86 棟幾乎乾淨、95.2% 異常在偶數樓、eval 側同樣只剩 **197** 個。**它有 91 棟樓,所以 site 級清單看不出來——這是為什麼原本被誤歸為「低 prevalence」。**

對照 **Rat (site 3)**:prevalence 更低(0.17%)但異常散在 44 棟樓,eval 側有 2,684 個 → **稀疏但不退化**,處方不同(多資料會改善,B1 顯示它 0.34 → 0.84)。

**cross-site 的測量(A1/A2/A3)對這兩站仍然有效**(測全站,4,978 / 4,087 個異常);退化只影響 within-site 設計(A5、a0)。

---

## 3. A4 建議:不執行

§9 Stage 5 的進入條件是「A3/A5 顯示仍需 finer site diagnosis,且成本可接受」。

- **§12.5 的飽和證據**預測 A4(15 source sites)將重現 A3 的 per-site 數值(8→12 只買到 +0.002)。
- **Crow 是唯一未解的站,而 A4 修不好它**:根本限制是 5 棟樓 + 92% 異常在單一 building,不是 source site 數量。
- **A0 已經填上 §12.4 的缺口**,先前主張 A4 的理由(唯一能診斷 Crow 的途徑)已不成立。

**Stage gate 未撤銷,由使用者決定。** 成本約 4h @ Normal priority。

---

## 4. 報告尚未寫入的內容(下一個 slice 的主要工作)

`docs/reports/m6-site-transfer.md` 的 §12 目前只涵蓋 A2/A3/A5。**以下全部未寫入:**

1. **§12.3 的分類推翻**(§2.2):Peacock / Moose 不是「非 site-specific」;Bobcat 不是「無法恢復」。目前報告仍是錯的。
2. **§12.4 的限制已被 A0 解除**——該節仍寫著「不在 M6 範圍」。
3. **A0 章節**:重現 M3 的驗證、三臂比較、M3 NPZ 的 join 限制。
4. **B2 章節**:§2.3 的全部內容。
5. **§12.6 / §12.8 的方向差裁定**:從「待 B2」改為「定案不可用」。
6. **§12.9**:移除 B2「尚未回答」;Crow 已於 §12.3.1 解決(已寫入)。
7. **飽和點更正**:從「410k 以下」改為「約 87k」,並收回「電表多樣性 +0.033」。
8. **Bear 併入 §12.3.1**:目前 §12.3(d) 仍把它誤歸為「低 prevalence 餓死 oracle」。
9. **Site 改名**:報告全篇仍用 `site 11`,圖已用 `Crow (site 11)`,兩者對不起來。

---

## 5. Known traps(沿用並新增)

沿用前檔 §5 的 trap 1–12,以下修正/新增:

- **Trap 13(新增)— M3 的 predictions NPZ 不可依 `validation_raw_index` join。** 順序與陣列不一致(逐列僅 88.35% 相符),join 會**靜默**給出垃圾(per-site prevalence 全部塌成 6.29%),不會報錯。要 per-site 就跑 A0,不要碰 M3。
- **Trap 14(新增)— site 級清單看不出 building 級退化。** Bear 有 91 棟樓卻 87% 異常在一棟。任何 within-site 設計前,先查 `data/processed/m6_site_structure.json` 的 `top_building_share` 與 `even_half_share`。
- **Trap 15(新增)— A0 與 B1 的 per-site 不可直接比。** A0 只測奇數樓,B1 測整站。同一 site 同一 budget 的 PR-AUC 差 0.06–0.13(Crow 甚至 0.1485 vs 0.0222)。`scripts/plot_m6_a0_vs_transfer.py` 用 NPZ 把 B1 重算到奇數樓並**硬性 assert 列數/異常數相符**,對不上就 raise。
- **Trap 16(修正前檔 trap 12)— Bear 與 Crow 的低支撐成因不同。** Crow/Bear 是異常集中在單一 building + 偶數偏斜;Rat/Wolf 是稀疏但分散。處方不同:稀疏型多給資料會改善,集中型不會。
- **Trap 17(新增)— B1 重算會吃記憶體。** 26 個 NPZ 共 5.6 GB,需用 `with np.load(...)` 逐一釋放,否則 OOM(已修)。
- **Trap 18(新增)— `--experiment b1` 只吃 `--direction a1|a2`**(§138),沒有 a0 的電表 sweep。A0 只有單點,圖上畫成水平參考線。

---

## 6. 圖與腳本(commit `ec8a51f`)

`docs/reports/assets/m6/` 共 **48 張**,依 `docs/reference/plot-style-rules.md` **v0.3**(該規範於 `930758d` 首次 commit,v0.3 新增「主標題是名詞片語圖名,不寫成問句;目的放 subtitle」)。

| 腳本 | 產出 |
|---|---|
| `scripts/plot_m6_b1_curves.py` | 12 張 B1 曲線(4 site 組 × 3 指標)+ 16 張 per-site 三指標 |
| `scripts/plot_m6_a0_vs_transfer.py` | 16 張 unseen-site penalty(transfer vs A0 baseline) |
| `scripts/plot_m6_site_structure.py` | 3 張 site 結構(集中度 / 累積 / vs building 數) |
| `scripts/plot_m6_b2_matched_support.py` | 1 張 B1+B2 合軸 |
| `scripts/m6_site_names.py` | GEPIII → BDG2 site 名稱,從 `data/raw/bdg2/metadata.csv` 讀並 assert 雙向 1:1 |

Site 名稱:0 Panther / 1 Robin / 2 Fox / 3 Rat / 4 Bear / 5 Lamb / 6 Peacock / 7 Moose / 8 Gator / 9 Bull / 10 Bobcat / 11 Crow / 12 Wolf / 13 Hog / 14 Eagle / 15 Cockatoo。

**Observation JSON 在 `data/processed/`(gitignored)**:`m6_site_structure.json`、`m6_a0_vs_transfer.json`。圖可由腳本重建,但來源 JSON 不在版控——這與 plot-style-rules §9.1 的意圖一致(數值放 `data/processed/`),只是本 repo 不追蹤該樹。

`scripts/plot_m6_a0_vs_transfer.py --reuse` 可跳過重算直接產圖。

---

## 7. 執行環境

前檔 §3.1 的 Task Scheduler / EcoQoS priority 陷阱仍然有效,但**該檔宣稱的 4 倍在 B1 上未複現**:同 budget 對照(`b1_a1_m50` BelowNormal 60.3 min vs `b1_a1_m50_s123` Normal 44.8 min)只有 **1.85 倍**。4 倍是從 A3 fold 量的,workload profile 不同。

實測耗時(Normal):B1 每 chunk 16–45 min;B2 每 seed 42–47 min;A0 單 cell 16.2 min。

**記憶體是真正的瓶頸**:pipeline 是單執行緒(實測只用 0.97 / 32 核心),而 frame 需 12–15 GB。曾觀察到 commit charge 38 GB > 實體 31.6 GB、Pages/sec 飆到 15,013、同樣工作慢 2.4 倍。**不要並行跑兩個 cell。**

Driver:`.scratch/m6_autorun/`(gitignored)的 `m6_resume.ps1`(B1)、`m6_b2.ps1`、`m6_a0.ps1`,皆為 cell 級 resume + 自我提權 Normal。排程 `M6Resume` 維持 **DISABLED**(使用者決定)。

---

## 8. Repo close-out(依 `docs/reference/change-checklist.md` 誠實記錄)

| 檢查項 | 狀態 |
|---|---|
| Slice 開始時開 GitHub issue | **未做。** 工作在 issue 開立前展開;依 backfill policy 誠實標記,不補造 retroactive issue history(使用者明確選擇)。 |
| `Closes #N` commit | **未做**,同上。 |
| README 更新 | **未做。** M6 仍未完成(B1 缺 4 cells、Aggregate 未跑),milestone 狀態未定案。 |
| `docs/plans/` 更新 | **未做**,同上。 |
| ADR | **未新增。** M3 NPZ 的 join 限制曾考慮記 ADR,但那是「拿工具做它沒被設計的事」,不是架構決策;記於本檔 §5 trap 13。 |
| Handoff | **本檔案。** |
| Result JSON 位置 | 合規:全部在 `data/processed/`。 |
| CJK UTF-8 | 本檔以 UTF-8 寫入。 |
| 驗證 gate | ruff / ruff format / markdownlint / pre-commit file hooks **通過**。**tests 未跑**:`pytest` 不是 `pyproject.toml` 宣告的相依(裝它需使用者確認)。`pre-commit --all-files` **刻意未跑**:其 autofix hooks 會改寫 worktree 中他人的未提交工作。 |

**Commits**:`930758d`(繪圖規範)、`ec8a51f`(報告 + 48 張圖 + 5 個腳本)。

`README.md`、`docs/plans/m3-plan.md`、`docs/reports/m3-report.md` 的既有修改與 M3 figure assets 屬於使用者前次工作,**本 slice 未觸碰**。

---

## 9. 下一步建議順序

1. **改報告**(§4 的 9 項)。這是最大的未償債務——**目前 §12.3 的分類是錯的**,而圖已經是對的。
2. 補跑 B1 seed 999 的 `m400` / `mall`(4 cells,約 60 min),圖即可從 2 seeds 升到 3 seeds(§143 要求至少 3)。
3. 跑 Aggregate 產生 `m6_site_transfer_plot_data.json`。
4. 由使用者決定 A4 stage gate(建議不執行,見 §3)。
5. 待上述完成後,一次更新 README / `docs/plans/m3-plan.md`,並依 checklist 補齊 issue/commit 決策。
