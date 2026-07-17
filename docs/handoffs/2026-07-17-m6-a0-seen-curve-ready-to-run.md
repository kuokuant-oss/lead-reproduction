# M6 — A0 seen-site 學習曲線:程式已就緒,未執行

Written: 2026-07-17
Branch: `main`
HEAD at handoff: `5dca1b1 Add handoff for M6 B2, A0 and the figure batch`
Execution environment: Windows PowerShell,repo venv at `.venv\Scripts\python.exe`

> **State.** 程式碼已寫好並驗證,**一個 cell 都還沒跑**(使用者決定延後)。
> 要跑的是 **30 個 cell**:A0 的 seen-site 學習曲線,2 個 building folds × 5 個 meter budgets × 3 seeds。
> 跑完直接產圖,腳本已就緒。
>
> 本檔只涵蓋這一件事。整體 M6 狀態見
> [`2026-07-17-m6-b2-a0-figures-complete.md`](2026-07-17-m6-b2-a0-figures-complete.md)。
> 權威實驗協議是 [`docs/reports/m6-site-transfer.md`](../reports/m6-site-transfer.md)。
> Trust JSON `status` fields over prose。

---

## 1. 這要回答什麼

**Seen 對 unseen,每個 site,整站。**

| 臂 | 訓練 | 測試 |
|---|---|---|
| **unseen** | 偶數 sites(a1)或奇數 sites(a2) | 整個 held-out site |
| **seen** | 每棟樓由「沒訓練過它」的那一折預測 | **整個 site** |

兩條線同一批列、同一條 x 軸(標註電表數),差別只剩「有沒有看過這個 site」。

### 為什麼 seen 需要兩折

要量 seen,模型必須看過這個 site,但不能拿訓練過的列去測(洩漏)。所以 seen 那一臂**只能測 site 的一部分**——這點繞不過去。

- **一折**(先前的 `b2_a0_pos677077_seed42`):只測奇數樓 → **B1 必須被切成奇數樓才能比,B1 的曲線就變了**。這是先前那批 `m6_unseen_site_penalty_site_*.png` 的做法,**使用者已否決**。
- **兩折**:`a0odd` 留出奇數樓、`a0even` 留出偶數樓。聯集後每棟樓都有 out-of-fold 預測 → seen 覆蓋整站 → **B1 一個字都不用動**。

---

## 2. 開跑

**由 Claude Code 以背景 task 啟動(`run_in_background: true`),不要用排程工作。**

- **排程工作在本專案反覆出事**,使用者已明確排除:`schtasks` 預設 BelowNormal → EcoQoS,以及 `schtasks /Run` 由 tool call 觸發會在數十秒內被砍(`CONTROL_C_EXIT`)。
- 背景 task 曾在 2026-07-17 被砍過兩次(約 94 分鐘、約 3.5 小時)。**這不是阻礙**:被砍時 agent 會收到 task-notification,直接重跑同一行指令即可;driver 是 cell 級 resume,已完成的 cell 會被 SKIP,最多損失執行中的那一個。只要 session 還在,這就是個自癒迴圈。

**先小規模驗證再放全量。** 本檔的所有 a0 驗證都只到 `--prepare-only`;**fit 階段從未在 a0 折上實跑過**(trap 24)。先跑這個,約 30–60 分鐘:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\.scratch\m6_autorun\m6_a0_curve.ps1 -Seeds 42 -MeterBudgets 50
```

log 出現兩行 `DONE ... status=completed` 才算通過。通過後放全量(30 cells,約 8–12 小時 @ Normal priority):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\.scratch\m6_autorun\m6_a0_curve.ps1
```

已完成的 cell 會被自動 SKIP,所以驗證用的那 2 個不會重跑。

**跑之前提醒使用者關掉 Chrome 等吃記憶體的程式**(見 trap 21)。

跑完直接產圖:

```powershell
$env:PYTHONUTF8="1"; $env:PYTHONPATH="$PWD\src;$PWD\scripts"
.\.venv\Scripts\python.exe scripts\plot_m6_seen_vs_unseen.py
```

Logs:

```text
.scratch\m6_autorun\m6_a0_curve.log            主時間軸
.scratch\m6_autorun\child_*_prep.out.log       各 cell 的 manifest 階段
.scratch\m6_autorun\child_*.out.log            各 cell 的 fit 階段
```

進度查詢:

```powershell
Get-Content -Tail 5 -Encoding UTF8 .\.scratch\m6_autorun\m6_a0_curve.log
Get-ChildItem data\processed -Filter "m6_site_transfer_b1_a0*_meters*_seed*.json" |
  Where-Object { $_.Name -notlike "*_manifest*" } | Measure-Object | Select-Object -ExpandProperty Count
```

---

## 3. 要跑的 30 個 cell

`2 folds × 5 budgets × 3 seeds`,stem 格式 `m6_site_transfer_b1_{fold}_meters{budget}_seed{seed}`:

| Fold | 留出 | 訓練 |
|---|---|---|
| `a0odd` | 奇數 buildings(10,137,155 列) | 偶數 buildings(10,078,945 列,全部 16 sites) |
| `a0even` | 偶數 buildings | 奇數 buildings |

Budgets `50 / 100 / 200 / 400 / all`,seeds `42 / 123 / 999`。

**耗時參考**(來自已完成的 B1/B2,Normal priority):每 cell 16–45 分鐘,隨 budget 上升。30 cells 約 **8–12 小時**。

---

## 4. 已完成的程式改動(全部 additive,已驗證)

| 檔案 | 改動 |
|---|---|
| `scripts/m6_site_transfer_protocol.py` | 新增 split `a0_building_mod2_inverse`(test = `building_id % 2 == 0`),是既有 `a0_building_mod2` 的互補 |
| `scripts/run_m6_site_transfer.py` | 新增 `B1_DIRECTION_SPLITS` 字典;`--direction` 從 `("a1","a2")` 擴充為 `("a1","a2","a0odd","a0even")`;`require_site_disjoint` 改為依 `unit_type` 決定 |
| `scripts/plot_m6_seen_vs_unseen.py` | 由 `plot_m6_a0_vs_transfer.py` 改寫並更名 |
| `.scratch/m6_autorun/m6_a0_curve.ps1` | 新 driver |

### 驗證紀錄

- **15 個既有 M6 測試全過**(`uv run --with pytest pytest tests/test_m6_site_transfer.py -q`)。
- **兩折互補且完整分割**:`fold1.test XOR fold2.test` 全 True、`fold1.test == fold2.train` 全 True。
- **既有 a1/a2 行為未變**:新舊 direction 映射對既有 cell 的 `split.rule` 字串逐字相符。
- **rule 字串刻意保持 byte-identical**(`source-site-stratified ...`),既有 B1 cell 重跑會重現原 manifest。曾一度改成 `source-stratified` 造成回歸,已修正並在程式碼註解記錄原因。
- `ruff` / `ruff format` 通過。
- **產圖腳本在 A0 未跑時可正常運作**:只畫 unseen 線,seen 標為 `Seen arm not yet run.`。已實測產出 16 張。

### 關鍵設計:`require_site_disjoint`

a0 的兩折按 **building** 切,所以每個 site 必然同時在 train 和 test 兩側。原本 B1 硬寫 `unit_type="site_id", require_site_disjoint=True`,對 a0 會 assert 失敗。已改為依 `base_manifest["unit_type"]` 決定——**a1/a2 仍然強制 site-disjoint**,只有 building 折豁免。

---

## 5. 產圖腳本的行為

`scripts/plot_m6_seen_vs_unseen.py` → `docs/reports/assets/m6/m6_seen_vs_unseen_site_{0..15}_pr_auc_f1_recall.png`(16 張,每張 3 個 panel:PR-AUC / F1@0.5 / Recall@0.5)。

**不變式:**

1. **unseen 線直接讀 B1 的 cell JSON `slices.by_site_id`——不重算、不重新評分、不取子集。** 那就是 B1 的曲線。
2. **seen 線把 `a0odd` 與 `a0even` 的 NPZ 預測聯集後才 per-site 評分。**
3. **列對齊 gate 是硬性的**:聯集後每站的 `n_rows` / `n_anomalies` 必須與 B1 的 slice 完全相同,否則 `raise SystemExit`,不畫圖。
4. `--reuse` 可跳過重算,直接由 `data/processed/m6_seen_vs_unseen.json` 產圖。

Observation JSON:`data/processed/m6_seen_vs_unseen.json`(gitignored,但圖可由腳本重建)。

---

## 6. Known traps

沿用 [`2026-07-17-m6-b2-a0-figures-complete.md`](2026-07-17-m6-b2-a0-figures-complete.md) §5 的 trap 1–18,以下新增:

- **Trap 19 — 不要用一折的 A0 當 baseline。** `b2_a0_pos677077_seed42`(單折)只覆蓋奇數樓。拿它當 baseline 會逼 B1 也切成奇數樓,B1 的數值就會移動 0.06–0.13(Robin 0.6504 → 0.7152、Crow 0.1485 → 0.0222)。**使用者已明確否決這個做法。** 那批 `m6_unseen_site_penalty_site_*.png`(16 張,commit `ec8a51f`)因此是**過時的**,新圖產出後應刪除。
- **Trap 20 — B1 沒有 building 分割。** B1 按 **site** 切;拿 building 奇偶去切它是把 A0 的設計強加上去,和 B1 的問題無關。seen 那一臂才是有限制的那個,要動就動它。
- **Trap 21 — 絕不並行跑 cell。** pipeline 是單執行緒(實測只用 0.97 / 32 核心)但 frame 需 12–15 GB。兩個 cell 同時跑會讓 commit charge 超過實體 31.6 GB,同樣工作慢 2.4 倍。driver 已寫成序列執行,不要「優化」它。
- **Trap 22 — B1 的 NPZ 很大。** 每個約 250 MB;`a0odd`+`a0even` 兩折聯集時必須用 `with np.load(...)` 逐一釋放,否則 OOM(已如此實作)。
- **Trap 23 — `pytest` 可跑,只是沒宣告。** `uv run --with pytest pytest tests/... -q` 可用。先前的 handoff 寫「tests 未跑」是因為我沒試這條路;**checklist 的 test gate 是跑得起來的,不要再跳過。**
- **Trap 24 — 只驗到 manifest,fit 階段未驗。** 本檔的所有 a0 驗證都是 `--prepare-only`。fit 階段理論上和 a1/a2 走同一條路,但先跑 §2 的小規模版確認,不要直接放整晚。
- **Trap 25 — 用背景 task 啟動,不要用排程。** 排程在本專案反覆出事(EcoQoS、`schtasks /Run` 被砍),使用者已排除。背景 task 被砍不是問題:重跑同一行指令,cell 級 resume 會 SKIP 已完成的。見 §2。

---

## 7. 跑完之後

1. **產圖**(§2 的指令),16 張 `m6_seen_vs_unseen_site_*.png`。
2. **刪除過時的 16 張** `m6_unseen_site_penalty_site_*.png`(見 trap 19)——需使用者確認(CLAUDE.md:file deletion 一律確認)。
3. **更新協議 §138**:B1 目前定義為「固定 A1 與 A2 的 location split」,已擴充到 a0 兩折,需要記錄。**這是協議變更,應在報告中明確標示,不可默默改。**
4. **報告 §12** 仍有 9 項未寫入的更正,見 [`2026-07-17-m6-b2-a0-figures-complete.md`](2026-07-17-m6-b2-a0-figures-complete.md) §4——**其中 §12.3 的分類目前是錯的**,那是最大的未償債務,優先於本實驗。
5. 補跑 B1 seed 999 的 `m400` / `mall`(4 cells,約 60 分鐘),圖即可從 2 seeds 升到協議要求的 3 seeds。

---

## 8. Repo close-out(依 `docs/reference/change-checklist.md` 誠實記錄)

| 檢查項 | 狀態 |
|---|---|
| Slice 開始時開 GitHub issue | **未做**,依 backfill policy 誠實標記(使用者明確選擇不開)。 |
| `Closes #N` commit | **未做**,同上。 |
| README 更新 | **未做。** 本 slice 未產生任何實驗結果。 |
| `docs/plans/` 更新 | **未做**,同上。 |
| ADR | **未新增。** B1 擴充到 a0 折是協議變更,應在報告 §138 記錄(見 §7.3),不是架構決策。 |
| Handoff | **本檔案。** |
| Result JSON 位置 | N/A(未產生結果)。 |
| CJK UTF-8 | 本檔以 UTF-8 寫入。 |
| 驗證 gate | `ruff`、`ruff format`、**15 個 M6 測試全過**、markdownlint 通過。`pre-commit --all-files` **刻意未跑**:其 autofix hooks 會改寫 worktree 中他人的未提交工作。 |

`README.md`、`docs/plans/m3-plan.md`、`docs/reports/m3-report.md` 的既有修改屬於使用者前次工作,**本 slice 未觸碰**。
