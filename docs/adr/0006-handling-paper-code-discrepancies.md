# Handling apparent paper-code discrepancies

## Status

Accepted (2026-05-26)

## Context

在重現一篇論文(LEAD 1st-place solution, Fu et al. 2022)的過程中,我使用 AI 工具
(Claude Code)輔助讀論文 + 對照原始碼。發現多處「描述不一致」現象,並接受了 AI 的
強烈措辭(「論文嚴重低估」、「論文 vs 代碼矛盾」、「論文宣稱 X 但代碼無對應實作」),
將這些 claims 寫進專案 docs 並 commit。

準備跟教授討論之前,自我質疑「會不會是我們誤會」,做了一輪 verification:回到論文
PDF 原文,核對每個「矛盾」聲稱。結果:

- **V1 (32 vs 120 value-change features)**: 論文 §2.2.2 用 "i.e." 列舉
  sub-day {1,2,3,23} 和 multi-day {24,48,72,168} 兩組 shifts,但同段也說
  "shifts within one day were fully accounted for"。代碼實作完整 1-day shifts
  和更密集的 multi-day intervals。論文 intent 與代碼一致;只是 i.e. 列表給的
  例子不完整。
  → **判斷:過度解讀。應描述為「論文舉例不完整,代碼為 ground truth」,不是「論文錯」**

- **V2 (mean vs median imputation)**: 論文 §2.1 明確寫 "missing values were
  replaced with the **mean** value for each time series"。代碼也是 mean。
  README 寫 median。
  → **判斷:事實誤判。實為 README vs code 不一致,不是論文矛盾。論文跟代碼一致**

- **V3 (Hyperparameter tuning)**: 論文 §2.3.3:"different models **will be
  considered for** modeling and hyperparameter tuning"。"will be considered
  for" 描述未來/規劃意圖,並未承諾調出非預設值。代碼最終使用各 GBDT 庫預設值。
  → **判斷:過度解讀。論文措辭與代碼用預設值不衝突**

三個 apparent discrepancies 中:**1 個是事實誤判,2 個是過度解讀**。沒有任何
一個是真正的「論文錯誤」。

## Decision

未來在此 repo(以及任何重現論文任務)中,處理 apparent paper-code discrepancies
採用以下流程:

### 1. 先標記為 "apparent discrepancy",而非 "contradiction"

觀察階段用中性描述:

- 「Paper §X 描述 X,代碼實作 Y,描述方式不同」
- 不要直接寫「矛盾」、「論文錯」、「論文嚴重低估」等強烈措辭

強烈措辭僅在 verification 完成且確認屬於 true contradiction 後使用。

### 2. Verify against PDF 原文,不只信 AI 摘要

- AI 工具讀的可能是 `paper-notes.md` 二手資料或自己的記憶
- 重要 claim 必須回到 PDF,完整 quote 原文寫進 docs
- 完整 quote 讓未來的我能重新評估,並避免「轉述失真」

### 3. 將 findings 分級

| 分級 | 特徵 | 案例 |
|---|---|---|
| **True contradiction** | 論文寫 A,代碼寫 B,兩者都 well-defined,且不可能同時為真 | (本 repo 中尚無真實案例) |
| **Imprecise description** | 論文舉例不完整或措辭模糊,代碼是完整版本 | V1 value-change shifts |
| **Over-interpretation** | 讀者(或 AI)把論文的 "may" / "will consider" / "for example" 讀成 "did" 或 "must" | V3 hyperparameter tuning |
| **File inconsistency** | 論文本身沒問題,但 README 或其他文件跟代碼不一致 | V2 imputation (README 錯,論文/代碼一致) |

每類有不同的處理方式;只有第一類需要嚴肅看待為「論文錯誤」。

### 4. Soften wording in docs

| ❌ Assertive (避免) | ✅ Descriptive (採用) |
|---|---|
| "Paper says X but code does Y" | "Paper §X describes X; code implements Y; descriptions differ" |
| "Paper severely underestimates" | "Paper's i.e. list is incomplete; code is ground truth" |
| "Paper claims X with no corresponding implementation" | "Paper describes intent; code uses defaults; not in conflict" |

### 5. 跟教授討論前先 verify

特別是「論文寫錯」或「論文描述失準」這類聲稱:

- 教授可能有 context 解釋為什麼這樣寫(版面限制、寫作時的考量等)
- 帶著「verified discrepancies + 我的判斷分級」去討論,比帶著「論文錯誤清單」好十倍
- 真正的研究成熟度在於分級判斷,不在於找出多少「錯誤」

## Consequences

採用此原則後:

**正面**:

- 寫 docs 比較慢,但內容更紮實、可辯護
- 跟教授討論時不會說錯話導致尷尬
- 養成對 AI 結論的 critical thinking 習慣
- 真正的 true contradiction(如果有)反而會更被重視

**負面 / 取捨**:

- 「發現」的速度變慢(每個 claim 都要 verify)
- 部分初期措辭強烈的 commits 留在 git log 中(歷史紀錄)
  → 透過後續修正 commits 表達「我學到了」即可,不需 revert 歷史

## Lessons learned (個人反思)

AI 抓取長文時會挑關鍵字抓取,有時會忽略前後文意思或是過度解讀。所以應該在研究中
出現「矛盾」時,先確認原文原意,不要直接接受 AI 的強烈措辭。

同時,Claude Code 連結本地 + GitHub 留存工作紀錄檔等保存歷程的工作流,讓「AI 輔
助下出現決策偏誤」這件事是**可恢復的** — 有 commit history 和 docs 留存,
就有路徑可以循著根源導正。這個工作流本身就是 AI-assisted research 的安全網,
為合理的研究流程。

## References

- Verification commit: `9b3928d` (2026-05-26 housekeeping)
- 觸發此 ADR 的 verification 過程: 跟教授討論前的自我質疑
- 影響的其他 ADR: ADR 0005 (imputation;從「論文 vs code 矛盾」降級為「README 不準」)
- 影響的 unknowns: #1 (value-change features)、#6 (hyperparameters)
- 相關工作原則: CONTEXT.md「三方不一致原則」(本 ADR 是該原則的具體應用範例)
