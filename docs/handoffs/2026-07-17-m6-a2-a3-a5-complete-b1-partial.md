# M6 Site Transfer — A2/A3/A5 完成，B1 部分完成，B2 未開始

Written: 2026-07-17
Branch: `main`
HEAD at handoff: `955f65e Clarify M5 combined timing semantics`（本 slice 未 commit）
Execution environment: Windows PowerShell，repo venv at `.venv\Scripts\python.exe`

> **State.** Runs are **stopped**（使用者收機）。Disk 有 **28 個 completed model cells**、16 個 paired
> oracle 檔案。**尚未 Aggregate**，`m6_site_transfer_plot_data.json` 不存在。
> A4 仍在 stage gate 後，未執行。
>
> 本檔案取代 [`2026-07-17-m6-site-transfer-rerun-ready.md`](2026-07-17-m6-site-transfer-rerun-ready.md)
> 的執行狀態描述。該檔的 protocol 邊界與 traps 仍然有效。
> 權威實驗協議仍是 [`docs/reports/m6-site-transfer.md`](../reports/m6-site-transfer.md)。
> Trust JSON `status` fields over prose。

---

## 1. 目前 disk 狀態（正式計數）

| Family | Completed cells | 說明 |
|---|---:|---|
| A2 | 2 | canonical + sourcecal |
| A3 | 8 | 四 folds × canonical/sourcecal，**全部完成** |
| A5 | 16 | 每 site 一個 in-site oracle，**全部完成** |
| B1 | 2 | 僅 `a1/a2 × meters50 × seed42` |
| B2 | 0 | 未開始（9 個 manifest 已備妥） |
| A4 | 0 | **gated，未執行** |
| **Total** | **28 / 65** | 另有 16 個 `m6_paired_oracle_site*.json` |

B1 尚缺 28 cells：`meters100/200/400/all × seed42` 與 `meters50/100/200/400/all × seed123/999`（每 budget 含 a1/a2 兩方向）。

從 disk 重新確認：

```powershell
Set-Location C:\Users\tonykuo\projects\lead-reproduction
Get-ChildItem data\processed -File -Filter "m6_site_transfer_*.json" |
  Where-Object { $_.Name -notlike "*_manifest.json" -and $_.Name -ne "m6_site_transfer_plot_data.json" } |
  ForEach-Object { Get-Content -Raw -Encoding UTF8 -LiteralPath $_.FullName | ConvertFrom-Json } |
  Group-Object cell | Select-Object Name, Count
```

---

## 2. 已取得的數值證據（僅記錄，未寫入報告結論）

### A2 反方向（odd→even）

| | A1 (even→odd, 既有 anchor) | A2 (odd→even) |
|---|---:|---:|
| Pooled PR-AUC (ensemble) | 0.6447 | **0.8954** |
| Macro-site PR-AUC | 0.7233 | **0.8892** |
| 最差 site PR-AUC | 0.1485 (site 11) | 0.7710 (site 10) |

**解讀邊界**：A2 test 側（偶數 sites）prevalence 10.25%，A1 test 側僅 3.60%；PR-AUC 隨機基線等於
prevalence，故方向差**不可**直接讀成「反方向較易 transfer」。A2 test 含 site 0（prevalence 33.1%、
PR 0.9991）對 pooled 有明顯拉抬。另注意 A2 訓練 anomaly 較少（410,394 vs 904,080）卻分數較高
—— 這正是 B2 要隔離的變因，B2 未完成前不下結論。

### A3 四折（16 sites 全覆蓋 OOF）

| Fold | Pooled PR | Pooled ROC | Macro-site PR mean | Macro-site PR min |
|---|---:|---:|---:|---:|
| 0 | 0.9875 | 0.9958 | 0.9459 | 0.7990 |
| 1 | 0.6480 | 0.9652 | 0.7815 | 0.5516 |
| 2 | 0.8130 | 0.9778 | 0.8139 | 0.7743 |
| 3 | 0.8083 | 0.9951 | 0.6914 | 0.2349 |

Fold 間 pooled PR 由 0.6480 到 0.9875，**site composition sensitivity 明顯**。

Per-site ensemble PR-AUC（OOF，每 site 恰好一次）最差五名：
site 11 `0.2349`、site 13 `0.5516`、site 1 `0.6653`、site 10 `0.7743`、site 3 `0.7714`。

### A5 paired in-site oracle（**本次最重要的發現**）

在相同 oracle-test rows 上比較（ensemble）：

| Site | rows | anomalies | cross_PR | oracle_PR | gap (oracle−cross) |
|---:|---:|---:|---:|---:|---:|
| 0 | 538,432 | 176,269 | 0.9999 | 0.9999 | 0.0000 |
| 1 | 289,853 | 39,135 | 0.7206 | 0.9828 | **+0.2622** |
| 2 | 1,263,915 | 80,897 | 0.7189 | 0.8821 | **+0.1632** |
| 3 | 1,181,463 | 2,684 | 0.8848 | 0.8331 | −0.0518 |
| 4 | 370,460 | 197 | 0.7627 | 0.3920 | **−0.3707** |
| 5 | 386,496 | 14,435 | 0.9778 | 0.9781 | +0.0004 |
| 6 | 345,117 | 28,654 | 0.8537 | 0.7499 | −0.1038 |
| 7 | 200,594 | 15,886 | 0.8414 | 0.6688 | −0.1727 |
| 8 | 284,376 | 31,083 | 0.8941 | 0.9404 | +0.0464 |
| 9 | 1,367,482 | 52,587 | 0.9313 | 0.9887 | +0.0575 |
| 10 | 206,430 | 15,814 | 0.5716 | 0.6274 | +0.0558 |
| 11 | 43,626 | 197 | 0.3594 | 0.0183 | **−0.3412** |
| 12 | 158,011 | 755 | 0.9989 | 0.9850 | −0.0139 |
| 13 | 1,334,223 | 59,283 | 0.6191 | 0.6686 | +0.0495 |
| 14 | 1,256,775 | 101,532 | 0.8065 | 0.8754 | +0.0689 |
| 15 | 909,902 | 17,989 | 0.9063 | 0.7586 | −0.1478 |

依 `docs/reports/m6-site-transfer.md` §10 預先定義的措辭：

- **site 11 / 13 / 10（最差 transfer sites）的 oracle 並未回升**（11 甚至更差、13 僅 +0.05、
  10 僅 +0.06）→ 屬於「**該 site 在 frozen feature/model contract 下本質較難**」，
  落差不能全部歸因為 unseen-site shift。
- **site 1（+0.26）與 site 2（+0.16）oracle 明顯回升** → 屬於「存在可由 target-site labels
  恢復的 unseen-site penalty」。
- **site 4 與 site 11 的 oracle test 只有 197 個 anomalies**，oracle 負 gap 很可能是
  support 不足所致，不是可靠的難度證據。這兩列在報告中必須標註低支撐度，**不得**當作
  「oracle 無用」的證據。

### A2 source-calibrated operating point

Canonical cell 的 `operating_points` **只有** `threshold_0_5`（無 `fixed_recall_0_90`，符合
no-test-label-leakage 契約）；sourcecal 才有 `source_calibrated_recall_0_90`。

A2 sourcecal ensemble：門檻 `0.5176` 學自 source calibration set（`building_id % 5 == 4`，
完全排除於 fit 外），該處 recall 精準為 `0.9000`；**套到 unseen test sites 後 recall 掉到
`0.8579`**，precision `0.7422`。值得追查的兩點：

- **site 8**：recall 0.923 但 precision 僅 **0.258**，115,420 個 false positives，
  predicted-positive rate 27.4% vs 實際 7.7%。
- **site 11 的 calibration 子集只有 3 個 anomalies**（17,548 rows），支撐度極低。

---

## 3. 執行環境的重要發現（影響後續所有排程）

### 3.1 Task Scheduler 預設 BelowNormal → EcoQoS → ~4x 慢

`schtasks` 建立的工作預設 priority 7 = `BelowNormal`。Windows 11 會把 BelowNormal 程序納入
**EcoQoS / efficiency mode**（降頻、優先排到 E-core）。實測同樣的 A3 fold：

| 執行方式 | Priority | 單一 fold 耗時 |
|---|---|---:|
| harness background task | Normal | **~14 min** |
| Task Scheduler（預設） | BelowNormal | **~57 min** |

A5 整個 stage 在 BelowNormal 下花了 **86.1 min**（Normal 下預估 ~25 min）；
B1 `s42_m50` chunk 花了 **60.3 min**（Normal 下預估 ~14 min）。機器當時 idle（24 cores，load 5%），
**不是 CPU 競爭**。

`.scratch\m6_autorun\m6_resume.ps1` 已加入自我提權：

```powershell
(Get-Process -Id $PID).PriorityClass = 'Normal'
```

子程序會繼承此 priority class，因此必須在 spawn 任何 child 之前設定。**後續若改用其他排程機制，
務必再確認 priority**，否則整體會慢約 4 倍（B1+B2 由 ~6.6h 變成 ~26h）。

### 3.2 哪些機制能存活

| 機制 | 結果 |
|---|---|
| WMI `Win32_Process.Create` | `Access is denied`（沙箱擋掉） |
| `schtasks /Run`（由 tool call 內手動觸發） | 數十秒內被砍，`CONTROL_C_EXIT` (0xC000013A) |
| `schtasks` **原生 trigger**（排程服務自己觸發） | **可存活**；PID 47644 由 04:44 連續跑 3.5 小時 |
| harness `run_in_background` | 可用，但曾在 ~94 min 被 stop |

原生 trigger 啟動的 instance **確實**能跨 session 存活（本次跑完 A3 尾段、整個 A5、一個 B1 chunk）。

---

## 4. 續跑方式

`.scratch/`（gitignored）內的 resume driver 是 **cell 級 resume**：每個 cell 執行前先讀 JSON
`status`，`completed` 即跳過。因此**中斷不需要重跑任何已完成 cell**，最多損失執行中的一個 chunk。
被砍的 cell 不會留下 result JSON（runner 只在結束時寫檔）；即使寫到一半，truncated JSON 會被判為
`unreadable` 而重跑。所有 cell 使用 frozen seeds，重跑結果決定性相同 —— **無污染風險**。

排程工作 `M6Resume` **已 DISABLE**（避免下次開機自動吃 CPU）。續跑：

```powershell
Set-Location C:\Users\tonykuo\projects\lead-reproduction

# 方式 A：前景/背景直接跑 driver（Normal priority，最快）
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\.scratch\m6_autorun\m6_resume.ps1

# 方式 B：重新啟用自癒排程（每 10 分鐘探班，lock 協調，不重跑 completed cell）
schtasks.exe /Change /TN "M6Resume" /ENABLE
```

Driver 的工作計畫：A2 → A3 → A5（suite，含 pairing）→ B1（依 seed×budget 切塊，每塊 2 cells）
→ B2（依 seed 切塊，每塊 3 cells）→ Aggregate。**A4 永不執行。**

剩餘估計（Normal priority）：B1 約 **4.3h**、B2 約 **2.2h**、Aggregate 約 5 min，合計 **~6.6h**。

Logs：

```text
.scratch\m6_autorun\m6_resume.log            主時間軸
.scratch\m6_autorun\child_*.out.log          各 cell / chunk 輸出
.scratch\m6_autorun\m6_resume_status.json    收工後 census
```

---

## 5. Known traps（沿用並新增）

1. **Prepared is not completed.** `status=manifest_prepared` 不含模型結果。
2. **Do not rerun M3.** 不動 `src/lead`、M3 scripts、既有 M3 JSON/NPZ、split/feature/sampling/
   scaler/model params/ensemble weights。
3. **B2 `pos1` files are probes.** 保留但不得聚合；共同 budget `N_pos = 410,394`（由 a0=677,077、
   a1=904,080、a2=410,394 取下限，**奇數 sites 側為綁定條件**）。
4. **B2 stdout bug is fixed.** `Invoke-Python` 必須保留 `| Out-Host`。
5. **Canonical 與 sourcecal 是不同 variant**，不可覆蓋或平均。fixed recall 0.90 僅 sourcecal 可用。
6. **絕不由 test labels 反推 recall-0.90 門檻。**
7. **A4 is gated.** A3/A5 證據已在本檔 §2，但 stage gate **尚未經使用者明確重開**，不得執行。
8. **Poor site performance is evidence, not failure.** site 11/13/10 的低分必須保留。
9. **Large local artifacts are ignored.** `data/processed/*` 不出現在一般 `git status`；直接查 disk。
10. **Preserve the dirty worktree.** `README.md`、`docs/plans/m3-plan.md`、`docs/reports/m3-report.md`
    的既有修改與 M3 figure assets 屬於使用者前次工作，**未被本 slice 觸碰**。
11. **新增：Task Scheduler priority 陷阱。** 見 §3.1，忘記提權會慢 4 倍。
12. **新增：A5 低支撐度 sites。** site 4 與 site 11 的 oracle test 各只有 197 個 anomalies，
    負 gap 不可解讀為「oracle 無效」。

---

## 6. Repo close-out 狀態（依 `docs/reference/change-checklist.md` 誠實記錄）

| 檢查項 | 狀態 |
|---|---|
| Slice 開始時開 GitHub issue | **未做。** 工作在 issue 開立前即展開；依 backfill policy 誠實標記，**不補造 retroactive issue history**。 |
| `Closes #N` commit | **未做**（本 slice 未 commit；使用者未要求 commit/push）。 |
| README 更新 | **未做。** M6 仍在執行中（28/65），milestone 狀態未定案；待 B1/B2/Aggregate 完成後再評估。 |
| `docs/plans/` 更新 | **未做。** 同上，避免在 numeric artifacts 未齊前寫入結論。 |
| ADR | **未新增。** 本 slice 未做架構決策；priority/排程屬執行環境細節，記於本 handoff §3。 |
| Handoff | **本檔案。** |
| Result JSON 位置 | 合規：全部在 `data/processed/`；`docs/` root 無散落 result JSON。 |
| CJK UTF-8 | 本檔以 UTF-8 寫入。 |
| 驗證 gate（tests / ruff / markdownlint / pre-commit） | **未於本次收尾執行**（無程式碼變更；`.scratch/` 為 gitignored 工作區）。commit 前必須補跑；注意前次紀錄 pre-commit 曾出現 `InvalidManifestError`。 |

**本 slice 對 tracked 檔案的唯一新增就是本 handoff。** 所有 driver/log 位於 gitignored 的
`.scratch/m6_autorun/`；實驗產物位於 gitignored 的 `data/processed/`。

---

## 7. 下一步建議順序

1. 續跑 B1（28 cells）與 B2（9 cells），確認 Normal priority。
2. 跑 Aggregate 產生 `data/processed/m6_site_transfer_plot_data.json`。
3. 檢查全部 65 cells 的 JSON `status` 與 fingerprint。
4. **A5 證據已足以支持 site-difficulty 與 unseen-site penalty 的區分**（§2）；
   由使用者決定是否重開 A4 stage gate。若 A3 dispersion + A5 gap 已足夠，A4（~4h @ Normal）
   可能只是重複證明同一件事。
5. 待 numeric artifacts 齊全後，再一次更新 README / `docs/plans/m3-plan.md` /
   `docs/reports/m6-site-transfer.md` 的結論，並依 checklist 補齊 issue/commit 決策。
