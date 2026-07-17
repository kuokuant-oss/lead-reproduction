---
type: convention
title: Plot Style Rules
version: 0.3
updated: 2026-07-17
scope: LEAD reproduction research figures, model comparisons, diagnostic plots, and EDA
status: canonical-for-new-figures
---

# 作圖規範 v0.3

本規範定義 `lead-reproduction` 的圖面語言。新圖一律依本檔；既有圖視為
legacy，不因本規範成立而自動重畫。若後續要改既有圖，必須連同產圖腳本、來源
JSON、報告引用與數字一致性一起處理。

本規範移植自另一研究專案的多面板圖原則，但已改成本專案的 anomaly detection、
模型比較、跨資料域與重現證據語境。另一專案的模型名稱、實驗代號、輸出目錄與
檔名詞彙不在本專案沿用。

## 1. 圖的任務

每張圖只回答一個主要問題。畫圖前先用一句話寫出問題，例如：

- support 增加時，各模型的 test PR-AUC 如何變化？
- 在相同評估列上，TabPFN 與 tree models 的等待成本相差多少？
- site-transfer 是否改變模型排序？
- BDG2-only 與 GEPIII-overlap 的資料分布是否不同？

一個 panel 以兩個位置維度（x、y）加一個系列維度為上限。若還要呈現 split、
meter、feature regime 或 operation point，優先固定條件或分面，不用顏色、大小、
透明度與線型同時堆疊。

圖面只放讀圖所需資訊。資料路徑、seed、row cap、split fingerprint、硬體、完整
方法備註與限制寫進來源結果 JSON 或報告正文，不塞進圖底 caption。

## 2. 標準圖面骨架

多面板圖由上到下固定為：

```text
主標題
一行 subtitle
N 個對齊面板
共用 y 軸標題（同單位時）
共用 x 軸標題
共用圖例
```

- 主標題是圖名，使用名詞片語，**不寫成問句**。§1 要求先寫下的那個問題與圖的目的放進 subtitle，不放進主標題。主標題使用自然語言，不使用 runner 名、milestone 內部代號或 JSON key。
- subtitle 承載目的與比較口徑，例如圖要回答什麼、資料域、split、metric 與是否共用尺度。
- 面板標題放在框上方、左對齊，使用人類可讀名稱。
- 圖例置於圖底、置中，單列優先，`frameon=False`。
- 圖底只留圖例；方法備註與結論寫在報告，不畫進圖。
- 不畫整張圖的外框；panel 移除上、右 spine。

單圖沿用同一結構，但不需要共用軸規則。只有一個系列且標題已說明其身分時，可
省略圖例。

## 3. 文字與版面層級

採三層文字層級，且層級必須一眼可辨：

| 層級 | 建議字級 | 樣式 |
|---|---:|---|
| 主標題 | 16 pt | 粗體、primary ink |
| subtitle | 10.5 pt | regular、secondary ink |
| 面板標題 | 11.5 pt | 粗體、primary ink |

字級表是起點，不是忽略畫布尺寸的絕對值。產圖後必須以報告實際嵌入寬度檢查感知
比例；大畫布不可留下巨大標題配小型內文，小畫布也不可讓長標題橫跨全寬。主標題
與 subtitle 之間要保留清楚的行距，subtitle 與第一個 panel 之間的空間則應小於無意義
的頁首留白。

- panel 標題以短名詞為主；公式、定義與方法條件改用較小的次級文字。
- 浮動備註必須有明確錨點；與主問題無關的 companion result、限制或版本說明移到報告。
- 最終尺寸下，最小的必要文字仍須可讀；不得靠放大整張圖掩蓋 panel 文字過小。

圖中文字預設使用英文，與 metric、模型名及報告既有術語一致；報告正文負責中文
解讀。不得在同一張圖混用 `HistGBT`、`HistGBDT`、
`HistGradientBoosting` 等多種顯示名稱。

Canonical 顯示名稱如下：

| token | 圖面名稱 |
|---|---|
| `lightgbm` | LightGBM |
| `xgboost` | XGBoost |
| `catboost` | CatBoost |
| `hist_gradient_boosting` | HistGBT |
| `ensemble` | Tree Ensemble |
| `tabpfn` | TabPFN |

`GBDT` 只表示 tree family，不是一個額外模型。若實作是 `LGBMClassifier`，圖面要寫
`LightGBM`，不能為了簡短改寫成 `GBDT`。

## 4. 共用軸與尺度

1. 同單位且目的為跨模型、跨 split 或跨 cohort 比大小時，必須共用軸與範圍。
2. 不同單位或數量級差距會壓平訊號時，可分面使用獨立尺度；subtitle 必須寫明
   `Independent y-scales`，且不得跨 panel 比高度或斜率。
3. 多列圖只有最底列顯示 x tick label；全圖只保留一個共用 x 軸標題。
4. 同單位多面板全圖只保留一個共用 y 軸標題。
5. 折線圖的 x 軸使用真實數值位置。`20, 50, 100, 500` 不得畫成等距，因為線的
   斜率具有變化率意義。
6. 柱狀圖的 x 軸是離散類別，使用等距位置；類別數值只是 label，不拿來決定間距。
7. 柱狀圖原則上從 0 起算。若局部差異需要放大，改用 dot plot、差值圖或明確標示
   的局部折線圖，不截斷 bar 軸製造差距。
8. log 軸必須在軸標題或面板標題標示 `(log scale)`；0、負值與 clipping 規則寫進
   報告或來源 JSON。
9. 水平 bar small multiples 也受共用尺度規則約束。若問題要求跨 panel 比較長度，
   必須使用同一 x 範圍；若保留獨立範圍，subtitle 必須寫 `Independent x-scales`。
10. 同一多面板重複出現的軸標題只保留一次；不能讓下排每個 panel 重複相同單位，
    上排卻完全沒有單位。

線性軸使用約 8% headroom。tick locator 優先使用 `1, 2, 5, 10` 步長；避免 2.5
步長配合整數 formatter 產生看似不等距的標籤。大數使用 `k`、`M`，不顯示
`1e6`；同一軸的精度一致。

## 5. Metric 與 anomaly detection 語意

- 類別高度不平衡時，PR-AUC 是模型排序的主要圖面 metric；ROC-AUC 作為次要或
  並列證據，不用 ROC-AUC 單獨支撐部署結論。
- ROC／PR panel 的 x、y 繪圖區使用 1:1 實體長寬，避免寬扁 panel 放大水平差異。
- 改成方形 panel 後必須重新校正 title pad、row spacing 與上界 headroom；不得沿用寬版
  layout 的間距，且曲線不可貼住 `1.0` 上界形成一條黏在 panel 標題下方的色線。
- 多模型曲線集中在邊界時，另用標明 `(zoomed)` 的局部 panel；zoom 的 x、y span
  優先保持一致，例如 ROC 使用 `FPR 0.00–0.08`、`TPR 0.92–1.00`。
- random-classifier 對角線只有在問題是「是否優於隨機」時才畫；模型都已遠離隨機線
  時省略，避免無資訊的灰色斜線占據圖面。
- train、validation、test 不得只靠顏色區分。固定用線型或分面作第二編碼，並在
  subtitle 說明 split。
- pooled 與 macro/per-unit 指標不得畫成同一條未註明的曲線。若並列，固定分面或
  在軸標題明示 aggregation。
- mean 必須搭配變異資訊（CI、標準差、range 或原始 seed 點）；單 seed 結果不得
  畫出不存在的誤差棒。
- null、underpowered、coverage 不足或 stochastic 結果，圖面不得用勝負色或
  `winner` 標籤過度解讀。限制與 stop rule 放在相鄰報告段落。
- 缺值使用 `NaN` 讓折線中斷，不跨缺漏條件補線。

### 5.1 Confusion matrix

- 每個模型使用標準 2×2 matrix；若需多模型比較，採 small multiples 並共用色階。
- cell 同時顯示 count 與明確分母的比例；標題或報告要說比例是 row-normalized、
  class-normalized 或 total-normalized。
- anomaly class 的 FN/TP 與 normal class 的 TN/FP 基數差異很大時，不用 raw count
  的單一色階暗示 TN 最重要。優先以 class-normalized 色階呈現，count 留在文字。
- 不同 operation point（例如 threshold 0.5 與 fixed recall 0.90）分圖或分面，不在
  同一 matrix 疊加。
- 四格預設使用一致字重且不加外框；除非圖說明確要求風險強調，不自行替 FN 加紅框、
  粗體或其他特殊視覺權重。

### 5.2 時間與成本

- `fit_predict_seconds` 必須標成合併等待時間，不得寫成 inference latency。
- fit、predict 與 preprocessing 只有在計時邊界分開量測時才可分段作圖。
- CPU 與 GPU 結果可比較使用者等待時間，但 subtitle 必須標示 backend；不得把它
  解讀成硬體無關的模型效率。
- 跨模型時間、RAM、VRAM 等同單位成本圖必須共用尺度。數量級差異過大時，使用
  log 軸或另出局部圖，不讓每個 panel 自訂尺度後都填滿畫面。

## 6. 線條、標記與格線

- 一般定量圖只留必要的水平 hairline 格線；ROC／PR 曲線圖預設不畫任何格線。
- 線寬、marker 與格線寬度按 figure 對角線縮放。以 9.4×7.6 inch 為基準：線寬
  1.0、marker 4.2、格線 0.7。
- 離散實驗點的模型比較固定使用不同 marker。ROC／PR 是連續門檻曲線，不加 marker，
  避免看起來像少量頂點相連的折線圖；模型以固定顏色與圖例辨識。
- 實線與 filled marker 表示主要／實際 arm；虛線與 hollow marker 表示對照／虛擬
  arm。模型顏色不因 arm、split、排名或是否被篩選而改變。
- 兩個以上系列必須有圖例；不在每個點標數字。只標決策轉折點、門檻或使用者需要
  精確讀取的少數值。
- 需要 marker 的其他曲線應依 x 軸位置取樣，而不是只按陣列索引固定間隔；點密度高度
  不均時，按索引取樣會在端點堆成雜點。
- 曲線貼近 0 或 1 時要保留少量 headroom，避免線與 marker 被座標邊界吃掉。
- 多面板共用系列身分時使用 figure-level 共用圖例；不得在每個 panel 重複放置相同
  圖例並遮蔽資料。
- 當各 panel 回答不同問題、需要不同 zoom 或各自圖例時，canonical 輸出應拆成獨立
  圖檔；不可為了維持固定張數而把可獨立閱讀的 ROC、PR 或 importance panel 硬擠成
  composite。
- 若某個 panel 的核心結論是 AUC、門檻或差值，將數值直接標在該 panel 的空白區域；
  不得只把關鍵數字放在可能被報告裁圖或截圖排除的全圖圖例。

## 6.1 留白與資訊密度

- 留白用來分組，不用來填滿固定畫布。標題區、panel 間距與圖底圖例區各自有明確用途。
- 上半部空曠、下半部擁擠代表結構需要重排，不應只縮小字體或加高畫布。
- 同一圖中的 peer panels 應具有相近的資料區面積、label 起點與視覺重量。
- 輸出後檢查資料墨水比例；若大量畫布沒有承載分組、閱讀順序或註解功能，就應裁減。

## 7. 顏色系統

### 7.1 基礎 token

| 用途 | hex |
|---|---|
| primary ink | `#0b0b0b` |
| secondary ink | `#52514e` |
| muted ink | `#898781` |
| grid | `#e1e0d9` |
| axis / baseline | `#c3c2b7` |
| surface | `#fcfcfb` |

文字一律使用 ink token，不用系列色寫字。圖面背景固定為 `surface`。

### 7.2 模型登記表

下表是新模型比較圖的唯一模型配色來源。顏色跟著模型，不跟著排名或圖種；只畫
其中幾個模型時也不得重新分配顏色。

| 模型 | hex | marker | 備註 |
|---|---|---|---|
| LightGBM | `#2a78d6` | circle `o` | blue |
| XGBoost | `#e07a00` | square `s` | orange |
| CatBoost | `#008b6d` | triangle `^` | green |
| HistGBT | `#8b65c2` | diamond `D` | purple |
| Tree Ensemble | `#0b0b0b` | pentagon `p` | ink；線寬可為 1.35× |
| TabPFN | `#d1498b` | X `X` | magenta |

離散點圖的配色不能單獨承載語意，marker 是必要的第二編碼；ROC／PR 連續曲線則依
前節規則省略 marker。新增模型時仍要先登記顏色與 marker，不得在單支腳本內即興
選色。正式採用新顏色前要以 light surface、實際線寬與適用的 marker 大小做 CVD／
對比驗證，並把驗證方式記在本節變更紀錄。

資料域、split、feature regime、tuned/default 等比較維度，優先使用分面或線型；
不要另外建立一套會與模型色衝突的 categorical palette。

## 8. 圖種選擇

| 問題 | 優先圖種 | 避免 |
|---|---|---|
| support／feature count 對 metric 的影響 | 真實數值 x 的折線或 dot-line | 等距數值類別折線 |
| 多模型單一 metric | 排序 dot plot | 類別過多的彩色 bar |
| metric 與等待成本 | 固定條件的 scatter | 每模型多條 history 軌跡 |
| seed 穩定性 | 原始點＋mean/interval | 只有 mean 的 bar |
| 分布比較 | ECDF、histogram 或 violin | 只報平均值 |
| 多模型 confusion matrix | 共色階 2×2 small multiples | model×cell 的 raw-count 熱圖 |
| 時序真值與預測 | 真實日期折線 | 等距 timestamp 或過度平滑 |

直方圖比較多個 cohort 時使用相同 bins。重疊會遮蔽形狀時優先改用 ECDF 或分面，
不要只提高透明度繼續疊圖。

### 8.1 Workflow 與方法示意圖

- 箭頭必須表達真實資料依賴，不能為了對稱畫出不存在的轉換。平行特徵分支合併時，
  使用明確 merge node 或共同輸出框，不畫成一個 feature set 產生另一個 feature set。
- fan-out／fan-in 優先採整齊、短且不交叉的連線；模型數增加時使用共用匯流排或分層
  節點，不讓多條長斜線在 ensemble 前交叉。
- 輸入、特徵、驗證、前處理、模型與輸出應有可辨識的層級；顏色只作輕量輔助，不能
  取代 stage label 與閱讀順序。
- 若某模型具有特殊缺值處理，可在模型框內簡短標示，但不得讓其他模型框因此失去對齊。

### 8.2 特徵重要性

- permutation importance 的變異資料必須保留在 observation JSON 或相鄰表格。若 SD
  誤差棒在最終輸出尺度無法清楚辨識，bar 圖不畫誤差棒，避免只留下無法解讀的小帽線。
- standalone permutation-importance 圖各自取該模型的 top 10，並依圖上實際編碼的
  importance 數值遞減；不得用另一個模型或 consensus 的排序套用到該圖。
- 若要做「跨模型共識」共同特徵列比較，必須另做明確標示的 aligned small multiples；
  不得把共同順序混入宣稱為 model-specific ranking 的獨立圖。
- 共識聚合與 ensemble permutation importance 是不同概念，必須使用不同標題並在
  subtitle 說明聚合口徑。

### 8.3 數值變化示意圖

- 示意圖必須直接標出被計算的 current／previous 點與代表性 Difference、Ratio 數值；
  只畫三條曲線和公式不算完成示意。
- Difference／Ratio 示意優先保留一條原始讀值時序，在同一 panel 選一組相鄰讀值：
  Difference 用兩讀值間的雙向垂直箭頭，Ratio 用從共同零基準到兩讀值的成對箭頭。
  公式與數值直接放在箭頭旁，不另外建立兩條完整衍生時序或多層說明框。
- exact timestamp merge 找不到配對時保留 `NaN` 斷線，並用淡色區段或短註解說明，
  不讓正確缺值看起來像繪圖錯誤。
- anomaly interval 使用淡色區段或稀疏標記；長段每點重複大型 marker 會遮蔽曲線。
- anomaly 顏色是資料語意色，不得挪用已登記的模型色，尤其不得與 TabPFN magenta
  混為一談。

### 8.4 Confusion matrix

- 2×2 matrix 的 colorbar 應保持窄且次要；色階只輔助比例，cell 文字才是精確讀值來源。
- 若圖說明確要求特別強調 FN，才使用一致的風險 accent、外框或註解；預設不強調。
- 長軸 tick label 優先拆成軸標題 `Ground truth`／`Model prediction` 加短類別名稱，
  避免每個 tick 重複 Actual／Predicted。

## 9. 輸出、來源與檔名

### 9.1 位置

- 報告專屬圖：`docs/reports/assets/<report-or-milestone>/`
- 跨報告使用或 EDA 圖：`docs/assets/<topic>/`
- 數值與 provenance JSON：`data/processed/`

`docs/` 下不得新增 loose result JSON。圖的來源 JSON、runner 與報告引用必須能互相
追溯；若一張圖由多個 artifact 組合，來源清單寫在報告相鄰段落或產圖腳本常數中。

### 9.2 檔名

模板為：

```text
<milestone>_<subject>_<view>_<metric>[_<operation_point>].png
```

- 全小寫 snake_case。
- 欄位由粗到細；所有會在同圖族中改變的資料維度都要入檔名。
- 使用 repo canonical token，不用臨時縮寫或自由 `suffix`。
- layout（例如 `grid`、`wide`）不是資料維度，不放進檔名。
- operation point 使用 `threshold_0_5`、`fixed_recall_0_90` 等可讀 token。
- 檔名中的 `m5` 等 milestone 只表示報告歸屬；圖面標題仍用自然語言。

例：

```text
m5_label_scarcity_test_pr_auc.png
m5_model_wait_time_fit_predict_seconds.png
m5_site_transfer_confusion_fixed_recall_0_90.png
bdg2_square_feet_ecdf.png
```

### 9.3 輸出格式

- 報告內嵌主格式為 PNG，預設 180 dpi。
- 需要投影片或印刷縮放時可另出 SVG/PDF，但不可讓不同格式使用不同資料或樣式。
- 使用手動 `subplots_adjust` 保留標題、軸名與圖例空間；不依賴
  `bbox_inches="tight"` 把圖外文字碰巧裁入。
- `savefig` 明確指定 `facecolor=surface`、`edgecolor="none"`；儲存後關閉 figure。

## 10. 產圖前後檢查

產圖前：

1. 用一句話寫出圖要回答的問題。
2. 指定來源 JSON、資料範圍、split、seed、metric 與 aggregation。
3. 確認模型名稱、顏色與 marker 來自本規範。
4. 判斷軸是否必須共用，x 是連續數值還是離散類別。
5. 檢查 underpowered、coverage、fallback、stochastic 與 timing boundary 限制。

產圖後：

1. 以最終輸出尺寸檢查，不只看 notebook inline preview。
2. 確認 title、subtitle、軸名、tick、圖例沒有裁切或互撞。
3. 確認黑白／低對比下仍可藉 marker、線型與分面辨識。
4. 對照來源 JSON 抽查極值、排序、缺值與 operation point。
5. 確認報告正文沒有把視覺差異寫成超出證據的結論。
6. 執行 repo 的 lint、test 與 `git diff --check`；CJK 文件另檢查 UTF-8 diff。
7. 檢查流程箭頭、缺值斷線、色階分母與 panel 尺度是否會造成錯誤語意；視覺 QA 不只
   是檢查文字有沒有碰撞。
8. 將圖縮到報告實際欄寬再檢查一次，確認 title 不壓過資料、註解仍可讀、marker 沒有
   聚成色塊。

## 11. Legacy 圖處理

目前 `scripts/run_bdg2_eda.py` 與
`scripts/run_m6_phaseD_50_50_full_models.py` 產出的既有圖早於本規範，可能使用預設
Matplotlib 配色、`tight_layout`、`bbox_inches="tight"` 或不同的 confusion matrix
結構。它們仍是已發表報告的有效 artifact，但不是新圖的樣式範本。

需要對齊 legacy 圖時，另開獨立 slice，至少同步處理：

- 產圖函式與測試；
- PNG artifact；
- 報告內嵌路徑與相關解讀；
- 來源 JSON／provenance 連結；
- 舊圖保留或 superseded 策略。

不得只手工修 PNG，也不得在不重跑來源驗證的情況下改圖中數字。

## 變更紀錄

- **v0.3 (2026-07-17)** — 主標題改為名詞片語圖名，禁止問句式標題；圖的目的與所回答的問題
  移入 subtitle。既有 legacy 圖不因此自動重畫（見 §11）。
- **v0.2 (2026-07-16)** — 納入 M3 報告圖的視覺 QA：補充標題間距與感知字級、
  留白密度、共用 x-scale、figure-level legend、依 x 位置抽樣 marker、workflow merge
  語意、permutation uncertainty、value-change 缺值與計算標註、confusion matrix 風險
  強調及 anomaly semantic color。
- **v0.1 (2026-07-15)** — 建立本專案 canonical 規範：多面板骨架、共用軸、
  anomaly detection metric 語意、模型顏色與 marker 登記表、confusion matrix、
  timing boundary、輸出位置、檔名與 legacy 遷移原則。
