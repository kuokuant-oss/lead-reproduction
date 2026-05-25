# M1 Plan: Understand the Paper and Method

## M1 完成標準

所有 6 個 B 類 unknown 已解決、4 個 ADR 已驗證、開發環境可執行,可以開始 M2 實作而不需猜測任何論文細節。

---

## 已完成事項

昨天完成的 7 個產出:

| 產出 | 說明 |
|------|------|
| `docs/paper-notes.md` | 論文全文精讀筆記,涵蓋 pipeline 七階段、公式、關鍵數字 |
| `docs/unknowns.md` | 6 個論文未說明的 B 類 unknown,附脈絡與重要性說明 |
| `docs/adr/0001-split-by-building-id.md` | 決策:以 building_id 切分 train/validation |
| `docs/adr/0002-downsample-majority-class.md` | 決策:downsample 正常類至 50:50 |
| `docs/adr/0003-value-change-both-diff-and-ratio.md` | 決策:差值與比值 features 都保留 |
| `docs/adr/0004-postprocessing-hard-rules.md` | 決策:兩條 hard override rules |
| `CONTEXT.md` | 專案約定、glossary、資料夾規範、tech stack |

---

## 剩下要做的事

共 4 張 issue。Issues 1–3 都是讀 buds-lab/LEAD-1st-solution 原始碼,但解決不同面向的 unknown;Issue 4 是環境設置。

---

### Issue 1: 釐清完整 169 個 features 的組成

**Title**: Resolve feature count gap: map all 169 features from buds-lab source code

**Why**:
論文描述的 shifts 最多只能算出 89 個特徵(57 原始 + 32 value-change),與 Table 3 的 169 差距 80 個。未知的 80 個特徵對模型 AUC 有貢獻,M2 實作如果只建出 89 個,輸出結果很可能無法重現 0.9866。

**What**:
- 閱讀 `buds-lab/LEAD-1st-solution` 中負責 feature engineering 的 notebook(s)
- 列出所有特徵名稱、類型(原始/衍生)、公式
- 確認 169 的來源:是否有額外的 shift 值、rolling statistics、或其他衍生欄位

**Done when**:
- [ ] 所有 169 個特徵有完整列表,附欄位名稱與公式
- [ ] 原本未知的 ~80 個特徵來源已確認
- [ ] `docs/unknowns.md` Unknown #1 標記為 resolved,附解答摘要
- [ ] `docs/paper-notes.md` Feature Engineering 章節補充更新(若有新發現)

**Out of scope**:
- 用 Python 實作 feature engineering(M2 的工作)
- 評估是否有更好的特徵(M3 的工作)

**Labels**: `type:research`
**Priority**: HIGH
**Depends on**: 無(可與 Issue 2 平行進行)

**Status (updated 2026-05-25)**: ✅ Complete
- 169 features composition resolved (B.1)
- Unknown #1 (feature gap) and #7 (CSV upstream) resolved
- Unknown #5 (target encoding) partially resolved — mechanism known,
  leakage impact deferred to M2
- B.2 (per-feature enumeration) skipped: generation rules suffice

---

### Issue 2: 釐清訓練 pipeline 的三個相依細節

**Title**: Clarify CV split, downsampling scope, and target encoding leakage protection from source

**Why**:
Unknowns #2、#4、#5 這三個問題同屬「訓練資料如何準備和評估」的同一層次:CV 切分、downsampling 作用域、target encoding 的 leakage 防護方式彼此有交互作用(例如 target encoding 是否在 fold 內計算,取決於 CV 的結構)。任何一個設計錯誤都會導致 local validation 分數不可靠,直接破壞 M2 的 feature engineering 迭代能力。

**What**:
- 讀 training loop 相關的 notebook(s)
- 確認:CV 是單次 split 還是 k-fold?fold 數?validation 建築數?seed?
- 確認:downsampling 是全域一次、按建築、還是每個 fold 重抽?seed?
- 確認:target encoding 是在 CV fold 外計算(有 leakage)還是 out-of-fold(無 leakage)?

**Done when**:
- [ ] CV 策略完整文件化(split 方式、fold 數或 validation 建築數、seed)
- [ ] Downsampling 的作用域與 seed 確認
- [ ] Target encoding 計算方式確認,leakage 防護有無說明
- [ ] `docs/unknowns.md` Unknowns #2、#4、#5 標記為 resolved
- [ ] 如有新的設計決策,補充 ADR

**Out of scope**:
- 實作 CV loop(M2)
- 比較不同 CV 策略的效果(M2)

**Labels**: `type:research`
**Priority**: HIGH
**Depends on**: 無(可與 Issue 1 平行進行)

---

### Issue 3: 文件化四個模型的超參數與 post-processing 邊界

**Title**: Document hyperparameters for all 4 GBDT models and Rule 2 start/end boundary

**Why**:
Unknown #6(超參數)和 Unknown #3(start/end boundary)解決難度低、風險低,但影響最終 AUC 的精確度。XGBoost 和 CatBoost 的 Train AUC 高達 0.9999,暗示特定超參數設定;若 M2 用預設值,各模型 AUC 分布可能和論文不一致。Rule 2 的邊界定義雖然影響有限,但是 rule-based 且需完全照原版。

**What**:
- 讀各模型的 training notebook/config
- 記錄 LightGBM、XGBoost、CatBoost、HistGradientBoosting 的所有非預設超參數
- 確認 Rule 2 "start/end points" 是首尾各幾筆(或其他邊界定義)

**Done when**:
- [ ] 四個模型的超參數各自以表格或 dict 形式記錄
- [ ] Rule 2 邊界定義確認,`docs/adr/0004` 補充更新
- [ ] `docs/unknowns.md` Unknowns #3、#6 標記為 resolved

**Out of scope**:
- 超參數調整實驗(M2)
- 新增第五個模型或替換模型(M3)

**Labels**: `type:research`
**Priority**: LOW
**Depends on**: 無;但建議在 Issues 1 & 2 之後進行,原始碼的閱讀脈絡還熱

---

### Issue 4: 設置開發環境與確認資料可用性

**Title**: Set up dev environment with uv and verify LEAD dataset access

**Why**:
M2 第一天需要立即跑程式。如果環境沒有在 M1 確認,M2 的第一個任務會卡在安裝依賴和 Kaggle 認證上,浪費時間。LEAD dataset 需要接受 Kaggle 競賽條款才能下載,這需要提前確認。

**What**:
- 初始化 `pyproject.toml`(uv),加入 pandas、numpy、scikit-learn、lightgbm、xgboost、catboost、jupyter 等依賴
- 確認 Python 版本符合 3.11+
- 從 Kaggle 下載 LEAD dataset,放到 `data/raw/`(不 commit)
- 確認訓練資料的 schema 與 `docs/paper-notes.md` 描述一致(欄位名稱、筆數大致符合)
- 更新 `.gitignore`(確保 `data/raw/` 已排除)

**Done when**:
- [ ] `uv sync` 可以在乾淨環境跑成功
- [ ] LEAD training/test CSV 在 `data/raw/` 可讀取
- [ ] `pandas.read_csv` 確認欄位:`meter_reading`、`building_id`、`anomaly`(train)等存在
- [ ] 訓練集約 200 棟建築、測試集約 206 棟,資料量與論文一致
- [ ] `.gitignore` 涵蓋 `data/raw/`

**Out of scope**:
- 資料清洗或 EDA(M2)
- 下載完整 ASHRAE GEPIII 資料集(M3)

**Labels**: `type:data`
**Priority**: HIGH
**Depends on**: 無(可最先或最後做,不阻塞 Issues 1–3)

---

## 建議執行順序

**現況 (2026-05-25)**:

| Issue | GitHub # | 狀態 |
|-------|----------|------|
| Issue 1: feature composition | #4 | ✅ done |
| Issue 2: CV / downsampling / target encoding | #5 | partially resolved — 剩下需 M2 實驗驗證,先不動 |
| Issue 3: hyperparameters + post-processing | #6 | still open (超參數尚未讀取) |
| Issue 4: dev environment | #7 | ✅ done (closed earlier) |

**現在只剩 Issue 3(GitHub #6)待處理**: 讀 Modeling notebook Cell 4 取出四個 GBDT 模型的超參數;Rule 2 邊界已在 B.1 中解決(Unknown #3 resolved)。

---

**原始規劃 (供參考)**:

**第一步(可平行)**: Issues 1、2、4

- Issues 1 和 2 都是讀 buds-lab 原始碼,但讀的是不同的 notebook 區塊(feature engineering vs. training loop),完全可以在同一個閱讀 session 中分頭記錄。
- Issue 4(環境)和閱讀原始碼完全獨立,可以同時進行。

**第二步(之後)**: Issue 3

- 超參數和 post-processing 邊界的優先序最低。在 Issues 1 和 2 完成之後,原始碼的閱讀脈絡還新鮮時補上,效率最高。
- 即使 Issue 3 延遲到 M2 開始後才完成也不致命 — 超參數在 M2 可以先用合理預設值跑通。

---

## 風險與已知阻塞

| 風險 | 說明 | 緩解方式 |
|------|------|---------|
| **原始碼不完整** | `buds-lab/LEAD-1st-solution` 的 notebooks 可能有硬編碼路徑、中途輸出或未更新的變數名稱,使部分 unknowns 難以直接從 code 回答 | 若確認後仍有歧義,在 `docs/unknowns.md` 標記為「partially resolved」並在 ADR 中記錄假設值 |
| **Kaggle 資料無法下載** | LEAD competition 若已關閉,資料可能需要特殊申請或找 ASHRAE GEPIII 的替代來源 | 提前確認 Kaggle 競賽頁面是否仍開放資料下載 |
| **Feature count 仍無法解釋** | 若讀完原始碼後 169 個特徵仍無法完整列出,Issue 1 無法完全 resolved | 記錄已知的特徵 + 不確定的部分,在 M2 以實驗方式驗證哪些特徵有效果 |
| **M1 範圍蔓延** | 讀完原始碼後可能發現論文還有其他未記載的設計決策 | 新發現的 unknowns 若屬 M2 必要,立即加 ADR;若屬可選優化,放入 `docs/unknowns.md` 標記 C 類 |

---

Last reviewed: 2026-05-25
