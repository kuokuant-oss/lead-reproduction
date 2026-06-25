# docs/reference/workflow.md — LEAD Reproduction 工作流全記錄

（本文是完整版工作流說明；reproduction report [Ch2](../reports/reproduction-report.md#ch2-工作流概覽) 是摘要版，並連結至此文各節。）

---

## §1 為什麼要記錄工作流

這份文件記錄我在 LEAD（Large-scale Energy Anomaly Detection）Kaggle 比賽
reproduction 工作中建立的工作流。Reproduction 目標是重現 Fu et al. 2022
「Trimming outliers using trees」的比賽方法，並在過程中理解每個設計決策背後
的理由。

工作流的記錄動機有三個：

**第一，跨 session 連續性。** AI-assisted coding 的最大痛點是 context window
重置。每次新 session，AI 工具重新從零開始。沒有系統性的文件，前一個 session
的判斷就消失了；有了結構化的 handoff docs + ADRs，新 session 只需讀指定文件
就能快速進入狀態。這不是理論設計——M2 milestone 跨 5+ 個 session、歷時 6 天，
靠的就是這套文件系統維持連續性。

**第二，防止 scope creep 和 framing drift。** 數字跑出來前，「這個算不算完成？」
很容易產生分歧。m2-plan.md 的「Done when」criteria 和量化指標，讓每個 milestone
的完成條件在動手前就說清楚，防止靠感覺判斷。

**第三，供未來 reproducer 理解設計邏輯。** Reproduction repo 不只是 code，
它是一份「為什麼這樣寫」的記錄。ADRs 記錄每個架構決策，`unknowns.md` 記錄
paper 未說清楚的地方。未來的 reproducer 不需要重新推敲這些判斷，直接讀文件。

---

## §2 文件生態系

### §2.1 ADR（Architectural Decision Records）

ADR 是本 repo 的骨幹文件。每個重要的架構或方法決策都記一份 ADR，格式為：
Status / Context / Decision / Rationale / Consequences。

目前有 6 個 ADR：

| ADR | 標題 |
|-----|------|
| 0001 | Split train/validation by building_id, not by time or random shuffle |
| 0002 | Downsample normal class to 50:50 ratio (not oversampling, not class weights) |
| 0003 | Include both difference and ratio value-change features |
| 0004 | Post-processing with two hard override rules |
| 0005 | Use mean imputation; README description appears incorrect |
| 0006 | Handling apparent paper-code discrepancies |

ADR 0006 是最重要的 meta-ADR：它定義「如何面對論文與代碼看起來不一致」的
分類框架（詳見 §3）。ADR 0005 記錄了 README 說 median 但代碼用 mean 的衝突，
並以代碼為準——這類「README vs 代碼」的 drift 在 open-source repo 很常見，
ADR 讓決策有明確記錄而不是靠記憶。

ADR 0001 記錄了 train/val split 的關鍵決策：按 building_id 切分（不是 time
或 random），讓同一棟建築的所有讀數只進 train 或 val，防止 data leakage。這個
決策讓 val 能真正反映模型在新建築上的泛化能力。

### §2.2 unknowns.md — 開放問題 living register

`unknowns.md` 是整個 M2 milestone 的核心文件，也是最常被讀取的文件。每次發現
paper 未說清楚的地方，就加進去；每個 milestone 結束時更新狀態。

文件格式：每個 unknown 有 ID（#1–#17）、state（open / partially-resolved /
resolved / out-of-scope）、discovery 時間點、和目前的理解。

這份文件的設計理念是：**不是 bug tracker，而是 epistemic state 的記錄**。
「不知道但知道自己不知道」比「不知道且以為已知」好得多。進入新 session 時，
讀 unknowns.md 能立刻看到哪些問題是已解決的、哪些是有待量化的、哪些是超出
本 milestone scope 的。

截至 M2.5 完成（2026-05-28），狀態：
- Resolved ✓：#5（SavGol 在 our pipeline 的 effect）、#15（Rule 2a building filter）
- Partially quantified：#10（imputation method — within-pipeline effect 已量化，
  但無法推論 paper 設計的全局 effect；詳見 §5 component interaction 說明）
- Open 或 out-of-scope：其餘 12 個（大多屬於 M3+ 的工作）

### §2.3 Handoff Docs

Handoff docs 讓跨 session 的工作可以不中斷。每個 milestone 結束時，我寫一份
handoff doc，記錄：本次完成了什麼、遇到什麼 blocking 問題、下一次 session
應該讀哪些文件、有哪些仍在進行的判斷。

目前有 4 份 handoff docs：

| 文件 | 對應 Milestone |
|------|---------------|
| `2026-05-26-m22-milestone-completed.md` | M2.2：pipeline + val score 達標 |
| `2026-05-26-m22a-completed.md` | M2.2a：ClusterNo 子里程碑（特徵解碼） |
| `2026-05-27-m23-completed.md` | M2.3：reproducibility fix（random_state=42） |
| `2026-05-28-m24-completed.md` | M2.4：post-processing + Kaggle submission |

M2.2a 是獨立的一份，因為 ClusterNo（building cluster feature）是 paper 中描述
不清楚的部分，解碼過程涉及多個 unknowns，值得獨立記錄。

每份 handoff doc 的格式標準化：本次完成摘要、關鍵數字、留下的 open questions、
下次 session 的 context requirements（讀哪 5-7 個文件）。這個格式讓任何 session
都能從 handoff 開始，而不需要靠記憶重建狀態。

### §2.4 m2-plan.md — 量化 milestone 計畫

m2-plan.md 記錄每個 stage 的「Done when」criteria 和量化指標。在每個 milestone
開始前，我先確認這份計畫，確保「完成」的定義是量化的而非主觀的。

典型的「Done when」：
- M2.2：val AUC ≥ 0.97（from M1 baseline 0.9536）
- M2.4：submit Kaggle + Private Score recorded
- M2.5：3 ablation CSVs on Kaggle + all unknowns updated

這個設計防止兩類失敗模式：一是跑出不錯的數字就停；二是一直調整但不確定算不算
完成。有了量化 criteria，「完成」就有清楚的判斷標準。

---

## §3 Verification 紀律 — ADR 0006

### 啟動事件

M2 早期，我在 commits 中寫了 3 個「論文 vs 代碼矛盾」。在跟論文作者討論前，
做了自我 verification（commit `9b3928d`）。結果 3 個都屬於 imprecise description
或 over-interpretation，沒有一個是 true contradiction。這次 calibration 避免了
在學術討論中說錯話，也讓後續 docs 措辭更為精確。

### 分類框架

ADR 0006 建立四層分類框架：

| 分級 | 定義 | 本 repo 案例 |
|------|------|------------|
| **True contradiction** | 論文寫 A，代碼寫 B，兩者 well-defined 且不可能同時為真 | (無真實案例) |
| **Imprecise description** | 論文舉例不完整，代碼是完整版 | Value-change shifts 舉例 8 個 vs 實際 60 個 |
| **Over-interpretation** | 把「will consider」讀成「did」 | §2.3.3 超參數 tuning 描述 |
| **File inconsistency** | 論文沒問題但 README 跟代碼不一致 | README 寫 median，論文/代碼是 mean |

**實際案例展開：**

*Value-change shifts（Imprecise description）*：Paper §2.2.1 舉例 8 個 shift，
代碼實際是 60 個（-24h 到 +24h 每小時 + 週間隔 24h × 6）。這不是矛盾，paper
只是用舉例的方式說明原則，代碼是完整實作。如果讀成「矛盾」，就會花時間查不
存在的問題。

*Imputation（File inconsistency）*：README 寫 median imputation，論文和代碼
都是 mean。以代碼為準（ADR 0005），記錄衝突來源（README drift），不需要跟
作者確認。

*SavGol（Imprecise description + ablation 量化）*：Paper §2.2.4 說「no apparent
positive effect」但 buds-lab code 仍包含此 feature。ADR 0006 框架讓我們標記為
「描述和代碼可以同時為真（no positive effect 不代表 remove it）」，留給 M2.5
Ablation A 量化（結果：val -0.001，Kaggle Private -0.004，在 noise floor 邊界）。

### 操作紀律

遇到疑似 paper-code 不一致時：
1. 先對照 ADR 0006 分類
2. True contradiction：記錄並標記 flag；Imprecise description：記錄正確理解；
   Over-interpretation：修正措辭；File inconsistency：以代碼為準
3. 不在 docs 裡寫「矛盾」、「錯誤」，除非確認是 true contradiction
4. 需要量化的不確定性 → 加進 unknowns.md，留給 ablation

---

## §4 Stage-gate 執行模式

每個 milestone 遵循固定的 stage-gate 模式，目的是讓「可能的誤解」在消耗大量
執行時間之前就浮出來。

**Step 1：讀 handoff doc 進入狀態**

新 session 開始時，必須讀指定的 5-7 個文件（handoff doc、m2-plan.md、
unknowns.md、相關 ADRs），報告理解後才動手。不跳過這一步，即使「感覺記得」
上次做到哪。Context window 重置後，「感覺記得」幾乎都是錯的。

**Step 2：確認 Done when criteria**

在 m2-plan.md 找到本次 milestone 的量化完成條件。如果條件不清楚，先釐清再
動手。這一步防止「跑完之後才發現做的不是要求的事」。

**Step 3：Pre-flight check**

動手前向 Tony 報告理解、計畫、和已知的 blocking issues。Tony 確認後才開始
執行。這一步的目的不是讓 Tony 審查所有細節，而是讓誤解在消耗計算時間之前
就浮出來。

**Step 4：執行 + 跑數字**

執行 notebook cells，記錄中間結果。遇到數字異常或邏輯衝突立即停下來報告，
不自行 reconcile。「任何衝突 → 停下來告訴我」這條原則讓 Tony 能 catch
framing 偏移，而不是在最後一步才發現方向錯了。

**Step 5：Commit（message 攜帶數字）**

每個 milestone 完成的 commit message 要包含量化指標，例如：
```
M2.4: Post-processing + Kaggle submission, Private Score 0.98616
```
這讓 `git log` 本身就是一份 progress log，不需要另外查文件。

**Step 6：寫 handoff doc**

記錄本次完成了什麼、留下什麼 open questions、下次的 context requirements。
這一步讓「工作的連續性」不依賴記憶。

---

## §5 One-shot Inference 哲學

Kaggle 競賽中常見的策略是反覆提交、觀察 leaderboard 分數、調整 pipeline。
Leaderboard probing 的問題是：public leaderboard 只有 30% 的 test data，
容易 overfit，且讓分數成為方向指引而不是 pipeline 設計的結果。

本 repo 採用相反的策略：**在提交前把不確定性記錄清楚，確定後一次提交**。

做法：
- Pipeline 設計的不確定性在 `unknowns.md` 記錄，而非靠試探分數解決
- 重要設計決策在 ADRs 記錄 rationale，讓決策可 audit
- 需要量化的 unknowns 留到 M2.5 ablation 解決（研究性質，不是分數最佳化）
- 只有在 pipeline 完整、unknowns 解決到滿意程度後，才提交

**結果（M2.4）**：6 天累積工作，單次提交，Private Score 0.98616，**Private gap 0.05%**（< noise floor，statistically indistinguishable）。

**M2.5 ablation 的定位**是研究，不是試探分數。3 個 ablation submissions
的目的：
- Ablation A：量化 SavGol 在我們 pipeline 中的 effect（unknown #5）
- Ablation B：量化 imputation method 在我們 pipeline 中的 effect（unknown #10）
- Ablation C：量化 Rule 2a building filter 的 effect（unknown #15）

**Component interaction 的重要教訓（Lesson #8）**：Ablation B 顯示在我們的
pipeline 裡，把 NaN drop 換成 per-building mean impute 會導致 Kaggle Private
-0.012。但這不能推論「paper 的 mean impute 設計是錯的」——因為 original
pipeline（with mean impute）的 Private Score 0.98661 > 我們的 0.98616。
Ablation 揭露的是 component 在我們 pipeline 裡的 interaction，不是 component
本身的全局 effect。

---

## §6 雙 AI 工作流

本 reproduction 使用兩個 Claude 實例分工，Tony 擔任 conductor：

**Claude Code（本地 repo）**

有 read/write 權限，負責讀 code、執行 notebook cells、寫/更新文件、commit。
每次開新 session 讀指定檔案重新進入狀態（context window 重置後靠 handoff docs
恢復）。不做 paper interpretation，只做 repo 操作和代碼分析。

**網頁版 Claude（paper + sanity check）**

負責解讀 PDF，確認 § 號、數字來源、方法論 edge cases。不接觸 repo，只提供
判斷。因為不接觸 repo，它不會被代碼「污染」，可以提供獨立的論文解讀。

**Tony 的 conductor 角色**

把兩個 Claude 的判斷互相對照，發現不一致時停下來確認。Repo + paper PDF +
buds-lab code 是 ground truth，兩個 Claude 都是 advisor/executor，不是 authority。

**實際發生的校準案例：**

*SavGol 矛盾 vs 描述不精確*：Paper §2.2.4 說「no apparent positive effect」，
但 buds-lab code 仍包含 SavGol feature。網頁版 Claude 最初讀成「矛盾」，Claude
Code 核對 ADR 0006 後標記為 imprecise description，留給 Ablation A 量化。
結果：val AUC 差異 -0.001，在 noise floor ±0.0005 邊界，屬於「保留無害，
移除也無害」的判斷。

*Framing 偏移*：M2.5 最初的 framing 是「Ablation B 顯示 paper 的 mean impute
方法有問題」。校準後：Ablation B 只是在我們的 pipeline 裡量化 swap imputation
的 effect，不能推論 paper 設計的全局 effect（component interaction 問題）。

*Confusion matrix 口徑差異*：Paper Fig 3 顯示的 confusion matrix 百分比與 §3
text 的 precision/recall 數字不直接對應；以 §3 text 為準（precision 98.7% /
recall 81.2%）。Fig 3 用 total prediction 正規化而非 row 正規化，直接代入
precision/recall 公式會得出錯誤結果。記錄供 future reproducer 參考。

雙 AI 模式的目的不是提高效率，而是提高判斷的 robustness：兩個沒有共享 context
的 AI 產生不一致的解讀時，就是需要人工判斷的訊號。Tony 每次見到不一致，都
是一次校準機會，而不是讓 AI 自行 reconcile。

---

## §7 Agent Skills 生態系

`.agents/skills/` 目錄包含 14 個 Matt Pocock 風格的 agent skills，提供結構化的
工作流觸發點。這些 skills 不是自動執行的，由 Tony 在對話中主動觸發。

**14 個 skills 完整列表：**
caveman、diagnose、grill-me、grill-with-docs、handoff、
improve-codebase-architecture、prototype、setup-matt-pocock-skills、tdd、
to-issues、to-prd、triage、write-a-skill、zoom-out

**本 reproduction 常用的 skills：**

| Skill | 用途 |
|-------|------|
| `handoff` | session 結束前產生 handoff doc，記錄 context、blocking questions、下一步 |
| `diagnose` | 分析問題根因，提供結構化診斷而非直接解法 |
| `zoom-out` | 退一步看整體架構，避免陷入細節；特別在 milestone 結束時確認 framing |
| `to-issues` | 把 conversation 中的 TODO 轉換成 GitHub Issues，讓行動有追蹤 |
| `triage` | 評估 issues 優先順序，決定哪些進 backlog、哪些立刻做 |
| `grill-me` | 對自己的方案進行 Socratic 質疑，找出隱含假設 |

Skills 的設計讓特定工作流步驟有標準觸發方式：
- 每次 milestone 完成 → 觸發 `handoff`
- 遇到架構方向疑問 → 觸發 `zoom-out`
- 要把討論轉成追蹤的 actions → 觸發 `to-issues`
- 對 ablation 設計不確定時 → 觸發 `grill-me`

---

## §8 M3 展望

M2 milestone 完成（M2.5 closed 2026-05-28）。M3 是下一個大工作：從 LEAD 的
406-building GEPIII subset，擴展到完整 ASHRAE GEPIII dataset。

**M3 vs M2 的關鍵差異：**

| 維度 | M2 (LEAD) | M3 (ASHRAE GEPIII) |
|------|-----------|-------------------|
| Buildings | 406 | 1449 |
| Total rows | ~1.8M test | ~20.2M train |
| Anomaly label | 競賽提供 `anomaly` column | `bad_meter_readings.csv`（row-aligned） |
| Anomaly rate | 2.13% | 6.5% |
| Train/val split | 競賽定義 | 自定義（按 building_id） |
| Weather data | 多 site | ASHRAE GEPIII weather |

**M3 的關鍵 finding（exploration 階段）**：

`bad_meter_readings.csv` 是 row-aligned 陣列（只有 1 欄 `is_bad_meter_reading`），
不是帶 join key 的 table。正確的 join 方式是 positional：
```python
train_raw['anomaly'] = bad_readings['is_bad_meter_reading'].values
```

Anomaly rate 從 M2 的 2.13% 升至 M3 的 6.5%（約 3×）。可能原因：LEAD 是
GEPIII subset，且 LEAD 的 anomaly 定義可能比 GEPIII 的 bad_meter_readings 更嚴。
這個差異會影響 M3 的 downsampling 策略和 threshold 設定。

M3 exploration 和 baseline 建立是後續工作的起點，預計沿用 M2 pipeline 結構
但需要調整 split 方式、重新評估 anomaly rate 對 downsampling ratio 的影響。

### Issue tracker map (M2)

| Milestone | GitHub Issue | Closed |
|---|---|---|
| M2.1 baseline pipeline | [#8](https://github.com/kuokuant-oss/lead-reproduction/issues/8) | 2026-05-26 |
| M2.2 value-change features | [#9](https://github.com/kuokuant-oss/lead-reproduction/issues/9) | 2026-05-26 |
| M2.3 4-model ensemble | [#10](https://github.com/kuokuant-oss/lead-reproduction/issues/10) | 2026-05-26 |
| M2.4 post-processing + refit | [#11](https://github.com/kuokuant-oss/lead-reproduction/issues/11) | 2026-05-29 |
| M2.5 ablation + closure | [#12](https://github.com/kuokuant-oss/lead-reproduction/issues/12) | 2026-05-29 |
| M2 unknowns research | [#5](https://github.com/kuokuant-oss/lead-reproduction/issues/5) | 2026-05-29 |

**Issue close 紀律**: Commit message 應包含 `Closes #N` 讓 GitHub 自動關閉 issue。
如果一個 milestone 對應多個 issue，寫 `Closes #N1, #N2`。Workflow §8 寫了這步，
但 M2 期間漏掉了 — 從 M3 開始嚴格執行。

---

*最後更新：2026-05-29（M2 issues closed，M3 plan 建立，M3.2 value-change 啟動）*

---

## Mandatory close-out checklist

Before any slice is committed, apply
[`docs/reference/change-checklist.md`](./change-checklist.md). This checklist is
the authoritative close-out gate for README freshness, plan tracker updates,
ADR status, issue closure, handoff writing, provenance placement, CJK UTF-8 diff
inspection, and local verification.
