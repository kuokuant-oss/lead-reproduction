# M6：Site Transfer 實驗設計

- **研究問題**：既有 supervised energy anomaly detection pipeline 能否泛化到訓練期間完全未見的 site？
- **論文來源**：Fu, Arjunan, and Miller (2022), *Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition*；[local PDF](../../.scratch/papers/lead-paper.pdf)。
- **資料來源**：ASHRAE GEPIII / LEAD labels；`data/raw/m3/`。
- **Frozen anchor**：[scripts/run_m3_full_site_transfer.py](../../scripts/run_m3_full_site_transfer.py)。
- **Anchor output**：[data/processed/m3_full_site_transfer.json](../../data/processed/m3_full_site_transfer.json)、`data/processed/m3_full_site_transfer_predictions.npz`。
- **狀態**：Protocol §1–§11 已凍結並據以執行。A1 anchor、A2、A3（四折）、A5（16 sites）已完成；B1 執行中；B2 未開始；A4 依 §9 stage gate 未執行。**執行結果與結論見 [§12](#12-執行結果)。**

Last updated: 2026-07-17

---

## 1. 研究定位

原論文的 competition validation 以 `building_id` 切分，確保 validation buildings 未出現在 training；competition test meters 則是不考慮 site 或 country 的隨機抽樣。論文在 Conclusion and Future Work 明確留下兩個後續問題：

1. 若 train/test 依 site 或 country 分開，模型能否預測 unseen locations 的 anomalies？
2. 改變有 anomaly labels 的 training meters 數量後，模型表現何時飽和或開始下降？

M6 將兩個問題拆成兩條獨立實驗軸：

- **Location-shift 軸**：完整保留 M3 pipeline，只改 train/test 的 site partition，測量 unseen-site generalization。
- **Training-scale 軸**：固定 location split，再控制 source-site training meters 數量，測量 learning curve。

本報告使用 **site-held-out internal generalization** 或 **zero-shot cross-site generalization** 描述主要實驗。Test site 不提供 labels 給模型訓練、選模或 threshold calibration。由於沒有 target-site fine-tuning，本實驗不把結果描述成 adaptation-based transfer learning；由於資料仍來自 GEPIII/LEAD，也不把它描述成 external-dataset transfer。

若未來取得可靠的 site-to-country mapping，才增加 country-held-out 軸。現階段結果只能回答 unseen site，不能回答 unseen country。

---

## 2. Frozen Pipeline 2：目前完整 site-transfer anchor

本報告將目前新完成的 full-data runner 稱為 **Pipeline 2 / A1 anchor**。它完整重用 M3/M4 凍結契約，唯一主要實驗變因是 split unit 從 `building_id` 改為 `site_id`：

| 項目 | A1 frozen contract |
|---|---|
| Split | train：偶數 sites；test：奇數 sites，`site_id % 2 == 1` 留出 |
| Train sites | `0, 2, 4, 6, 8, 10, 12, 14` |
| Test sites | `1, 3, 5, 7, 9, 11, 13, 15` |
| Site overlap | `0` |
| Building overlap | `0` |
| Train/test rows | `8,818,590 / 11,397,510` |
| Train/test buildings | `613 / 836` |
| Train/test anomaly prevalence | `10.2520% / 3.6007%` |
| Feature regime | `timestamp_merge`，17 baseline + 120 value-change，共 137 features |
| Fit sampling | 凍結 M3 downsampling；normal/anomaly 1:1 |
| Models | LightGBM、XGBoost、CatBoost、HistGradientBoosting、四模型等權 ensemble |
| Scaling / seed / hyperparameters | 沿用 frozen M3 contract |
| Scoring | 完整 held-out rows，不做 test score cap |

A1 的主要 threshold-free 結果如下。這些數字是後續所有 split robustness 實驗的 reference point，不因 M6 新實驗而覆寫：

| Model | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|
| LightGBM | 0.9584 | 0.6054 |
| XGBoost | 0.9386 | 0.6145 |
| CatBoost | 0.9324 | 0.6135 |
| HistGBT | 0.9772 | **0.6603** |
| Ensemble | 0.9744 | 0.6447 |

A1 是合理且可重現的第一個 paper-aligned site split，但不是完整的 site-transfer 證據。`site_id` 奇偶數只平衡 site 數量，沒有保證 rows、buildings、anomaly prevalence、氣候、國家或 building use 平衡。因此 M6 必須把 A1 當作單一固定 composition 的 anchor，而不是所有 unseen locations 的總結。

---

## 3. Pipeline 1 與 Pipeline 2 的比較邊界

為避免把不同問題混成同一個 leaderboard，M6 將比較分成兩層：

| 比較 | 目的 | 可回答的問題 | 不可直接聲稱 |
|---|---|---|---|
| Pipeline 1 building-held-out vs Pipeline 2 A1 site-held-out | 觀察 evaluation unit 從 building 改成 site 後的分數變化 | 同一模型族與 feature contract 在 in-domain building generalization 和 unseen-site generalization 的差異 | 分數差完全由 location shift 造成 |
| A1 vs A2/A3/A4 site splits | 測量結果對 site composition 的敏感度 | A1 是否是穩定、可重複的 cross-site 結果 | 任一單一 split 代表全部 sites |
| 相同 split 下不同 training-meter budgets | 在維持 source-site coverage 下隔離 labeled meter support 的影響 | performance plateau、sample efficiency | 不同 split 間的 learning curve 差異只由樣本數造成 |
| Cross-site prediction vs paired in-site oracle | 區分 site 本質難度與 unseen-location penalty | 同一批 site/building test rows 上，使用 target-site training data 可恢復多少表現 | Oracle 是可部署的 zero-shot transfer 結果 |
| Site-held-out vs future country-held-out | 增加 domain shift 強度 | unseen site 與 unseen country 的相對難度 | 在沒有 country mapping 時先宣稱 country transfer |

Pipeline 1 與 A1 雖然只更換 split 邏輯，但 train rows、buildings、anomaly prevalence 與 downsampled fit-set 數量會隨 split 改變。因此兩者的分數差是 **evaluation-unit change + site composition + training support distribution** 的合成結果，不解讀為 location shift 的純因果效果。

M6 不修改 Pipeline 1 或 A1 來強迫兩者具有相同資料量。若需要 isolating sensitivity，另建立 matched-anomaly-support 實驗並與 canonical full-data 結果並排呈現。

---

## 4. 實驗矩陣

### A0：Building-held-out reference

- 沿用既有 M3 building-level split 與已驗證結果。
- 只作 in-domain reference，不納入 site-fold aggregate。
- 不重跑、不覆寫既有 M3 artifacts，除非 provenance audit 顯示必要。

### A1：Even-to-odd full site-transfer anchor（已完成）

- Train：even sites；test：odd sites。
- Frozen full M3 pipeline，完整 held-out scoring。
- 後續所有 runner 必須能用 manifest/fingerprint 證明 A1 split 與 prediction rows 未漂移。
- A1 是主要可重現 anchor，不因其他 split 表現較好或較差而更換。

### A2：Reverse-direction site transfer

- Train：odd sites；test：even sites。
- 目的：測量 cross-site transfer 是否具有方向不對稱性。
- Pipeline、features、models、seeds、downsampling 與 scoring 契約全部和 A1 相同。
- A1/A2 分開報告，並另外計算兩方向 mean；不可只挑較好的方向。

### A3：Four-fold grouped-by-site evaluation（主要 robustness 實驗）

- 16 sites 依 `site_id % 4` 分為 4 個 frozen folds：`[0,4,8,12]`、`[1,5,9,13]`、`[2,6,10,14]`、`[3,7,11,15]`；每次 12 sites train、4 sites test。這是不用 labels 的 deterministic assignment，不宣稱氣候、國家或 prevalence 已平衡。
- 每個 site 恰好當一次 test site；任何 building 不得跨 fold。
- Fold assignment 在正式跑模前寫入 versioned manifest，記錄每 fold 的 sites、rows、buildings、anomalies 與 prevalence。
- Fold assignment 的目的不是把 test difficulty 調到一致，而是建立可重現 coverage；若用 row/anomaly counts 做平衡，規則與結果必須在跑模前凍結。
- 主 aggregate：四 folds pooled metrics 的 mean/std，以及合併全部 out-of-fold predictions 後的 global OOF metrics。
- Site macro metrics 由 16 個 held-out-site scores 計算，每 site 只出現一次，避免大型 sites 主導結論。

A3 是 M6 的主要確認性實驗。相較 A1 的 8-train/8-test，它使用 12 個 source sites，訓練資料較多；因此 A1 與 A3 不只差在 fold 數，也差在 source-domain coverage。兩者並排呈現，不把 A3 當作 A1 的直接替代。

### A4：Leave-One-Site-Out（LOSO，計算量允許後執行）

- 共 16 runs；每次 15 sites train、1 site test。
- 每個 run 產出單一 test site 的完整 predictions 與 metrics。
- 主要用途：找出最難 transfer 的 sites、估計 site-to-site dispersion，並檢查表現是否與 prevalence、rows、meter mix 或 metadata 有關。
- A4 不以 pooled row-weighted 分數作唯一排名；主要報 macro mean、median、IQR、min/max 與完整 per-site table。
- 若 A3 已顯示結論穩定且 A4 成本過高，可將 A4 降為 HistGBT + Ensemble 的確認性子集；此縮減必須在看到 A4 模型結果前決定。

### A5：Per-site in-site oracle（site 難度對照軸）

- 對每一個 target site，在 site 內部另做 frozen building-disjoint split；一半 buildings 作 oracle train，另一半 buildings 作 oracle test。
- 建議沿用 M3 的 deterministic building rule，並在正式跑模前凍結每個 site 的 oracle train/test building manifest。若某 site 的 building 數量或雙類別 support 不足，記為 unscorable，不為取得較好結果而重抽。
- Oracle model 使用和 A1–A4 相同的 features、downsampling、scaling、model parameters 與 seeds；唯一差異是允許使用同一 target site 內 oracle-train buildings 的 labels。
- 為建立真正 paired comparison，A1/A2/A3/A4 的 cross-site predictions 也要在同一批 oracle-test buildings/rows 上另外計分。主要比較量為 `in_site_oracle_PR_AUC - cross_site_PR_AUC`；兩側必須使用相同 `y_true` 與 eval-row fingerprint。
- A1/A2/A3/A4 原本對完整 held-out site 的 canonical metrics 保留不變；paired oracle subset 是 additive diagnostic，不覆寫 full-site score。
- A5 是 target-label oracle／per-site internal upper-bound reference，不是 zero-shot transfer 結果，也不得加入 A1–A4 的主 leaderboard。

A5 用來解釋 per-site dispersion。例如 site 11 cross-site PR-AUC `0.1485` 時：若 in-site oracle 同樣低，表示該 site 在 frozen feature/model contract 下本質較難；若 oracle 明顯回升，差距才可歸因為 unseen-site penalty。沒有這條對照軸，A3/A4 只能描述 site 間差異，無法判斷差異來源。

### B1：Training-meter learning curve

- 固定 A1 與 A2 的 location split，分別在 source sites 內控制 labeled training meters。
- 建議 budgets：`50, 100, 200, 400, all available meters`；若最小 budget 無法維持雙類別，再記錄 unscorable，而不是重抽到成功。
- Meter unit 定義為 source site 內唯一的 `(building_id, meter)` time series；不得把同一 meter 的 rows 拆成多個抽樣單位。
- 每個 budget 都必須跨全部 source sites 分層配置：先保證每個 source site 至少取得一個 meter，再依各 site 可用 meter 數按比例配置剩餘名額。整數尾差使用預先定義的 deterministic largest-remainder rule，不因模型結果調整。
- 每個方向、每個 seed 先在各 source site 內建立 frozen meter ordering；所有 budgets 從同一 ordering 取 prefix，使較大 budget 包含較小 budget，且每一層都維持 source-site coverage。
- 每個 budget 至少使用 3 個 frozen seeds；正式跑模前先寫出包含 split direction、seed、budget、site allocation 與 selected `(building_id, meter)` 的 versioned manifest。
- Test sites、test rows 與 scoring 完全固定；只有 source training meters 數量改變。
- Fit downsampling 邏輯不變，但每個 cell 記錄各 source site 被選到的 meters、rows、anomalies，以及 downsampling 前後的總 rows 與 anomaly support。
- 主要輸出：PR-AUC/ROC-AUC 對 labeled meters 的 learning curve、相鄰 budget gain，以及 plateau criterion。

這個 stratified nested manifest 是 B1 的必要條件。純隨機抽 50 個 meters 可能讓部分 source site 得到 0 個 meter，會把「labeled meters 變少」與「source-site coverage 變少」綁在一起，使 learning-curve 斜率無法歸因。

預先定義 plateau：連續兩個 budget 增量的 macro site PR-AUC 改善皆小於 `0.01`，且跨 seeds 的 95% interval 包含 0。此 criterion 是描述性 stage gate，不作統計顯著性宣稱。

### B2：Matched anomaly-support sensitivity

- 目的：輔助判斷 A0/A1/A2 的方向差有多少來自可用 anomaly labels 數量，而非只來自 unseen-site composition。
- Match 單位是 **anomaly rows 數量**，不是 balanced fit-set 的總 rows。A1 source prevalence 為 `10.2520%`，A2 source prevalence 對應原 A1 test side 的 `3.6007%`；只 match 總 fit rows 仍可能讓兩方向使用不同數量的 anomaly evidence。
- 正式跑模前，以所有待比較 splits 可提供的 anomaly rows 下限決定共同 `N_pos`，或選擇不超過該下限的 frozen anomaly budgets。這項決定不得參考模型分數。
- 每個 split、每個 seed 精確抽取相同 `N_pos` 個唯一 anomaly rows，再保留 frozen M3 的 `[negs1, pos, negs2, pos]` sampling shape：兩個 negative blocks 各抽 `N_pos`，同一 positive block 重複兩次。因此 fit-set 總數為 `4 * N_pos`、effective class balance 為 1:1；控制量仍是唯一 anomaly evidence `N_pos`，不是重複後的 fit-row 總數。
- 使用相同 seed grid，並記錄 sampled positive/negative row fingerprints、涵蓋的 sites/buildings/meters，以及抽樣前可用的 anomaly rows。Test rows 維持各自 natural prevalence。
- B2 是 sensitivity analysis，不覆蓋 full-data canonical results。
- 若 matched anomaly support 改變 model ranking 或 A1/A2 direction gap，報告必須同時呈現 canonical 與 matched-support 兩種結果。

### C1：Country-held-out extension（blocked until mapping exists）

- 需要可稽核的 `site_id -> country/location` mapping 與來源。
- 同一 country 的所有 sites 必須位於同一 split。
- 若 country 數過少，不做高變異的單次 50/50；優先 leave-one-country-out。
- Country test labels 不參與 feature selection、model selection 或 threshold calibration。
- 在 mapping 完成前，C1 保持 planned/blocked，不以 `site_id` 推測國家。

---

## 5. 所有實驗共同的 frozen fairness contract

除明確列為實驗變因的 split 或 training-meter budget 外，下列項目不得改變：

1. `load_m3_frame` 的資料、labels 與 merge contract。
2. `timestamp_merge` value-change 定義與 137-feature 順序。
3. Train/test 必須分開建 feature tables；不得為 test building/site 借用 train rows 建立 value-change features。
4. M3 downsampling、StandardScaler、model hyperparameters、model order、ensemble weights 與 random seeds。
5. 同一 cell 內所有 models 使用相同 train rows、test rows、features 與 labels。
6. A1–A4、B1/B2 與 C1 的 test labels 不得用於調參、選模、threshold calibration 或挑選「代表性」split。A5 明確是唯一允許使用 target-site oracle-train labels 的 diagnostic；oracle-test labels 仍只用於最後評估。
7. 每個 split 必須 assert site overlap = 0、building overlap = 0。
8. 每個 artifact 記錄 raw-row indices、split manifest、feature fingerprint、prediction fingerprint、environment 與 elapsed time。

A5 的 overlap assertion 與 transfer experiments 不同：同一 oracle run 的 train/test `site_id` 必然相同，但 building overlap 必須為 0。A5 artifacts 必須標記 `target_label_access = oracle_train_only`，避免被誤併入 zero-shot results。

若後續需要改善模型或調參，另開 model-development track；不能把 tuned model 數字混入 frozen-pipeline site-transfer 主表。

---

## 6. Metrics 與 operating points

### 6.1 Primary ranking metrics

| 層級 | Primary | Secondary | 用途 |
|---|---|---|---|
| Pooled rows | PR-AUC | ROC-AUC | 與既有 M3/M5 數字銜接，但會受大型 site 與 prevalence 主導 |
| Per-site macro | mean/median PR-AUC | mean/median ROC-AUC | M6 主要 generalization 結論 |
| Robustness | PR-AUC IQR、min/max、fold std | ROC-AUC IQR、min/max | 顯示 site composition sensitivity |
| Operational | precision、recall、F1、TP/FP/TN/FN | predicted-positive rate | 描述 threshold 下的實際 alarms |

PR-AUC 是主要 ranking metric，因 anomaly prevalence 低且各 sites 差異大。ROC-AUC 保留作為和論文及既有 milestone 的連續性指標，但不能單獨支持 operational usefulness。

每張 per-site 表必須同時列 `n_rows`、`n_anomalies` 與 `anomaly_rate`。若某 test site 只有單一類別，ROC/PR 記為 unscorable，不能以 0、1 或 pooled score代填。

### 6.2 Threshold contract

- `threshold = 0.5`：保留為模型原始 operating point。
- `fixed recall = 0.90`：threshold 只能從 source sites 的 calibration subset 估計，再原封不動套用到 unseen test sites。
- 禁止使用 test-site labels 反推 fixed-recall threshold；若引用既有 post-hoc test threshold，必須標成 descriptive oracle，不和 deployable threshold 混表。
- A3/A4 的每一 fold/run 都獨立從該 run 的 source sites calibration；不得使用跨 test folds labels 建共同 threshold。

建議 source calibration 從 training sites 內以 building-disjoint split 建立，且 calibration buildings 不參與 model fitting。若為嚴格維持 A1 的既有 pipeline 而沒有 calibration split，A1 只報 threshold-free metrics 與 threshold 0.5；source-calibrated fixed recall 作為 additive extension，不能回寫 A1 canonical artifact。

M6 additive runner 以 `--source-calibration` 建立獨立 variant：只在 source rows 內將 `building_id % 5 == 4` 留作 calibration，從 model fit 完全排除；每個 model 保存 calibration probabilities、calibration/test threshold metrics 與 per-site test confusion。未加此 flag 的 canonical cell 不產生 fixed-recall 0.90 圖，也不使用 test labels 補算。Canonical 與 source-calibrated variants 使用不同 artifact names，聚合時不可混為同一 cell。

---

## 7. A1 現況顯示必須保留 per-site 結果

A1 ensemble pooled PR-AUC 是 `0.6447`，但八個 held-out sites 的 PR-AUC 範圍是 `0.1485–0.9819`：

| Test site | Rows | Anomaly prevalence | Ensemble ROC-AUC | Ensemble PR-AUC |
|---:|---:|---:|---:|---:|
| 1 | 553,357 | 14.0558% | 0.9568 | 0.6504 |
| 3 | 2,370,097 | 0.1737% | 0.9976 | 0.8437 |
| 5 | 781,776 | 3.7831% | 0.9990 | 0.9819 |
| 7 | 366,681 | 7.8829% | 0.9877 | 0.9221 |
| 9 | 2,679,323 | 4.5969% | 0.9865 | 0.9331 |
| 11 | 119,459 | 4.1671% | 0.8906 | 0.1485 |
| 13 | 2,711,763 | 3.8965% | 0.9525 | 0.5225 |
| 15 | 1,815,054 | 1.9949% | 0.9941 | 0.7840 |

A1 ensemble macro site PR-AUC 是 `0.7233`，和 pooled PR-AUC `0.6447` 不同。後續報告不得只列 pooled score；否則大型 sites 與不同 prevalence 會掩蓋 site 11/13 的 transfer failure。

---

## 8. 預計圖表與資料契約

所有圖都由 tracked JSON/NPZ 中已記錄的 numeric data 重建；不允許只留下 PNG 而缺少 source arrays。

| 圖 | 內容 | 必要資料 |
|---|---|---|
| Figure 1 | Pipeline 1 building-held-out vs A1 site-held-out ROC/PR | split labels、完整 `y_true`、各 model probabilities、ROC/PR curve arrays |
| Figure 2 | A1/A2/A3 fold PR-AUC distribution | experiment/fold/model metrics、mean/std/IQR |
| Figure 3 | 16 sites per-site PR-AUC forest/dot plot | site metrics、rows、anomalies、prevalence、fold membership |
| Figure 4 | Site × model PR-AUC heatmap | 每 site、每 model ROC/PR-AUC；unscorable mask |
| Figure 5 | PR-AUC vs site anomaly prevalence | per-site prevalence、PR-AUC、rows；點大小可表示 rows |
| Figure 6 | Cross-site vs paired in-site oracle per site | 相同 oracle-test rows 的兩組 PR-AUC、oracle gap、eval-row fingerprint |
| Figure 7 | Threshold 0.5 confusion matrices | 每 model pooled與 per-site TP/FP/TN/FN |
| Figure 8 | Source-calibrated recall 0.90 confusion matrices | calibration threshold、calibration support、test confusion counts |
| Figure 9 | Training-meter learning curves | split direction、budget、seed、各 source-site meter allocation、pooled與macro site metrics |
| Figure 10 | Runtime/cost by experiment | feature time、fit time、predict time、total elapsed、peak memory（若可得） |
| Figure 11 | Country-held-out results（future） | country mapping provenance、country folds、per-country metrics |

每個 run 的 predictions NPZ 至少包含：

- `validation_raw_index`
- `site_id`
- `building_id`
- `meter`
- `anomaly`
- 每個 model 的 probability predictions
- `experiment_id` / `fold_id` 可由 companion JSON 無歧義還原

Companion JSON 至少包含：

- split manifest 與 overlap assertions
- B1 stratified meter manifest；B2 matched `N_pos` 與 positive/negative row fingerprints；A5 oracle building manifest 與 paired eval-row fingerprint
- train/test rows、buildings、meters、sites、anomalies、prevalence
- fit/downsample counts 與 seeds
- pooled、per-site、per-meter metrics
- ROC/PR curve arrays 與 threshold summaries
- feature names/order/hash、row-index hashes、prediction hashes
- model parameters、library versions、hardware、timing definitions
- feature、scaling、fit、test prediction、calibration prediction、metrics/curves 與 NPZ serialization 的分段時間；fit/test/calibration matrix shape、bytes 與 deterministic NaN sample
- pooled 與 macro-site mean/std/median/IQR/min/max、每 site support，以及 normal/anomaly score histograms
- plot-data contract 與 artifact paths

跨 cells 的繪圖不直接掃描各報告表格。`scripts/aggregate_m6_site_transfer.py` 將 completed JSONs 與 A5 paired-oracle JSONs 整理成單一 plot-data artifact，包含 `model_metrics`、`curves`、`site_metrics`、`threshold_0_5`、`fixed_recall_0_90`、`learning_curves`、`matched_anomaly_support`、`paired_oracle` 與 `runtime`。大型 exact probabilities 不重複寫入 aggregate；由 `cells[].predictions` 指向各 NPZ，可在需要重新計算 threshold、curve 或 slice 時讀取。

---

## 9. 執行順序與 stage gates

| Stage | 實驗 | 進入條件 | 完成條件 |
|---|---|---|---|
| 0 | A1 audit | 現有 JSON/NPZ 可讀 | split/row/prediction fingerprints、per-site metrics 與 plot contract 完整 |
| 1 | A2 reverse | Stage 0 通過 | 完整 full-data reverse predictions；A1/A2 公平性 asserts 通過 |
| 2 | A3 four-fold | Fold manifest 先凍結 | 四 folds 全部完成；每 site 恰好被測一次；OOF coverage 100% |
| 3 | A5 in-site oracle | 每 site oracle building manifest 先凍結 | 16 sites oracle 結果、paired cross-site subset 與 oracle-gap table 完整 |
| 4 | B1 learning curve | A1/A2 結果可重建，分層 meter manifests 已凍結 | 全 source-site coverage、nested meter budgets、至少 3 seeds、plateau summary 完整 |
| 5 | A4 LOSO | A3/A5 顯示仍需 finer site diagnosis，且成本可接受 | 16 sites per-site results 或預先宣告的模型子集完成；可與 A5 paired 比較 |
| 6 | B2 matched anomaly support | A0/A1/A2 anomaly support 差異影響解讀 | 相同 `N_pos`、row fingerprints、canonical 與 matched-support side-by-side 完成 |
| 7 | C1 country-held-out | country mapping 有可靠 provenance | country-disjoint asserts 與 leave-one-country-out 結果完成 |

任一 stage 若出現 artifact mismatch、overlap、test-label threshold leakage 或缺少完整 prediction rows，該 stage 不進入結果解讀，先修復 evidence contract。模型表現差不是停止條件；必須誠實保留 unfavorable site-transfer evidence。

---

## 10. 預先定義的結論邊界

M6 完成 A1+A2+A3 後，才可對 cross-site robustness 下主要結論；per-site dispersion 的成因解讀還必須加入 A5 paired oracle。建議依證據強度使用以下措辭：

- **只有 A1**：`在固定 even-to-odd site composition 下的 site-held-out internal generalization`。
- **A1+A2 一致**：`在兩個互補 50/50 site directions 下方向一致`。
- **A3 folds 一致且 site dispersion 可接受**：`在全部 16 sites 的 grouped out-of-fold evaluation 下具有穩定 cross-site generalization`。
- **部分 sites 明顯失敗，且 A5 oracle 同樣低**：`該 site 在 frozen feature/model contract 下本質較難，不能把全部落差歸因為 unseen-site shift`。
- **部分 sites 明顯失敗，但 A5 oracle 明顯回升**：`該 site 存在可由 target-site labels 恢復的 unseen-site penalty`；同時報 paired oracle gap，不以 pooled AUC 掩蓋。
- **B1 出現 plateau**：只描述在本資料與 frozen pipeline 下的 meter-budget plateau，不外推為其他 datasets 的 universal sample complexity。
- **C1 未完成**：不得使用 `cross-country transfer validated`。

本研究不預先把 A1 的高 ROC-AUC 解讀成部署成功。最終判斷以 PR-AUC、per-site dispersion、source-calibrated operating points，以及最差 sites 的 false alarms / missed anomalies 共同決定。

---

## 11. 程式碼與證據索引

| 項目 | 程式碼／文件 | 輸出／用途 |
|---|---|---|
| Paper future-work basis | [lead-paper.pdf](../../.scratch/papers/lead-paper.pdf) | Section 5, site/country split 與 training-meter scale 問題 |
| A1 full site-transfer runner | [scripts/run_m3_full_site_transfer.py](../../scripts/run_m3_full_site_transfer.py) | Frozen Pipeline 2 |
| A1 metrics | [data/processed/m3_full_site_transfer.json](../../data/processed/m3_full_site_transfer.json) | pooled/per-site metrics、curves、provenance |
| M6 split/sampling protocol | [scripts/m6_site_transfer_protocol.py](../../scripts/m6_site_transfer_protocol.py) | A2/A3/A4/A5 masks、B1 frozen meter order、B2 matched anomaly support |
| M6 additive runner | [scripts/run_m6_site_transfer.py](../../scripts/run_m6_site_transfer.py) | A2/A3/A4/A5/B1/B2；支援 `--prepare-only`。已執行 A2/A3/A5 與部分 B1（見 §12.1） |
| PowerShell suite orchestrator | [scripts/run_m6_site_transfer_suite.ps1](../../scripts/run_m6_site_transfer_suite.ps1) | Manifest-first 執行 A2/A3/A4/A5/B1/B2、source-calibrated variants、oracle pairing 與 plot-data aggregation |
| A5 paired comparator | [scripts/compare_m6_site_oracle.py](../../scripts/compare_m6_site_oracle.py) | 在相同 oracle-test rows 比較 cross-site 與 in-site oracle |
| Cross-cell plot-data aggregation | [scripts/aggregate_m6_site_transfer.py](../../scripts/aggregate_m6_site_transfer.py) | 將 metrics、curves、site slices、operating points、learning curves、oracle gaps 與 runtime 攤平為繪圖資料 |
| M6 protocol tests | [tests/test_m6_site_transfer.py](../../tests/test_m6_site_transfer.py) | split isolation、nested coverage、M3 sampling shape、paired row identity |
| M4 frozen interfaces | [m4-evaluation-report.md](m4-evaluation-report.md) | data/feature/split/evaluation contract |
| M5 bounded site comparison | [m5-foundation-vs-gbdt.md](m5-foundation-vs-gbdt.md) | 小樣本 site-transfer 參考，不取代 full-data A1 |
| Frozen helpers | [src/lead/](../../src/lead/) | M3/M4 public helper surface；M6 不改其既有邏輯 |

---

## 12. 執行結果

本節記錄 A2、A3、A5 的完整結果與可下的結論。§1–§11 的 protocol 在跑模前凍結，本節不回頭修改它們。

### 12.1 執行狀態與證據來源

| Stage | 狀態 | Cells |
|---|---|---|
| A1 anchor | 完成（早於本批） | 1 |
| A2 reverse | **完成** | 2（canonical + sourcecal） |
| A3 four-fold | **完成**，16 sites OOF coverage 100% | 8（四折 × canonical/sourcecal） |
| A5 in-site oracle | **完成** | 16 + 16 個 paired-oracle 檔案 |
| B1 learning curve | 執行中 | 9 / 30 |
| B2 matched support | 未開始 | 0 / 9 |
| A4 LOSO | **未執行**（§9 stage gate；見 §12.5 的建議） | 0 |

本節所有數值直接由 `data/processed/` 的 result JSON 讀出並複核，唯 A1 的 per-site 數字取自本報告 §7（A1 無獨立 M6 result JSON）。A5 的兩側比較使用相同 `paired_eval_key_sha256`。

### 12.2 資料集與 per-site 支撐度

全資料集：16 sites、1,449 buildings、2,380 meter series、20,216,100 rows、1,314,474 anomalies、overall prevalence **6.502%**（每 meter 平均 8,494 rows，對應整年小時級序列）。

依 §6.2，每張 per-site 表必須同列 rows / anomalies / anomaly rate：

| Site | Rows | Buildings | Meters | Meter types | Anomalies | Anomaly rate |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 1,076,662 | 105 | 129 | 0,1 | 356,496 | 33.111% |
| 1 | 553,357 | 51 | 63 | 0,3 | 77,779 | 14.056% |
| 2 | 2,530,312 | 135 | 289 | 0,1,3 | 172,476 | 6.816% |
| 3 | 2,370,097 | 274 | 274 | 0 | 4,117 | 0.174% |
| 4 | 746,746 | 91 | 91 | 0 | 4,087 | 0.547% |
| 5 | 781,776 | 89 | 89 | 0 | 29,575 | 3.783% |
| 6 | 668,133 | 44 | 80 | 0,1,2 | 66,658 | 9.977% |
| 7 | 366,681 | 15 | 42 | 0,1,2,3 | 28,905 | 7.883% |
| 8 | 567,915 | 70 | 70 | 0 | 43,504 | 7.660% |
| 9 | 2,679,323 | 124 | 306 | 0,1,2 | 123,167 | 4.597% |
| 10 | 411,407 | 30 | 50 | 0,1,3 | 42,499 | 10.330% |
| 11 | 119,459 | 5 | 14 | 0,1,3 | 4,978 | 4.167% |
| 12 | 315,909 | 36 | 36 | 0 | 1,479 | 0.468% |
| 13 | 2,711,763 | 154 | 309 | 0,1,2 | 105,665 | 3.897% |
| 14 | 2,501,506 | 102 | 288 | 0,1,2,3 | 216,881 | 8.670% |
| 15 | 1,815,054 | 124 | 250 | 0,1,2,3 | 36,208 | 1.995% |

Sites 高度不均質，這是後續所有解讀的前提：rows 相差 23 倍（site 11 的 119,459 對 site 13 的 2,711,763）、buildings 相差 55 倍（site 11 的 5 對 site 3 的 274）、prevalence 相差 190 倍（site 3 的 0.174% 對 site 0 的 33.111%）。Site 0 單站即佔全體 anomalies 的 27%。Site 11 只有 5 buildings，先天不足以支撐任何 within-site 設計。

### 12.3 A5：per-site 難度歸因（主要結果）

依 §128 的 unscorable 原則，先訂支撐度 gate：oracle fit rows ≥ 20,000、paired-eval anomalies ≥ 3,000、oracle train buildings ≥ 10。**此 gate 為 post-hoc 訂定（在看到 oracle gap 之後），但門檻只依 support 決定、不參考 gap 值**；未達標者記 unscorable，不重抽。排除 sites 3、4、7、11、12，其餘 11 sites 可解讀。

判讀主軸是 **in-site oracle 的絕對 PR-AUC**（「用該 site 自己的 labels 最好能到多少」），而非 gap——理由見 §12.4。

| Site | cross-site PR | **in-site oracle PR** | gap | 五模型同號 | oracle train buildings | oracle fit rows | paired-eval anomalies | 判讀 |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 0.9999 | **0.9999** | +0.0000 | mixed | 53 | 720,908 | 176,269 | transfer 已達上限 |
| 5 | 0.9778 | **0.9781** | +0.0004 | mixed | 45 | 60,560 | 14,435 | transfer 已達上限 |
| 12 | 0.9989 | 0.9850 | −0.0139 | mixed | 18 | 2,896 | 755 | unscorable；transfer 已可用 |
| 9 | 0.9313 | **0.9887** | +0.0575 | 5/5 正 | 62 | 282,320 | 52,587 | 可用；target labels 小幅增益 |
| 8 | 0.8941 | **0.9404** | +0.0464 | 5/5 正 | 35 | 49,684 | 31,083 | 可用；target labels 小幅增益 |
| 14 | 0.8065 | **0.8754** | +0.0689 | 5/5 正 | 51 | 461,396 | 101,532 | 可用；target labels 小幅增益 |
| 15 | 0.9063 | 0.7586 | −0.1478 | 5/5 負 | 62 | 72,876 | 17,989 | anomalies 非 site-specific（見下） |
| 6 | 0.8537 | 0.7499 | −0.1038 | 5/5 負 | 22 | 152,016 | 28,654 | anomalies 非 site-specific |
| 7 | 0.8414 | 0.6688 | −0.1727 | 5/5 負 | 7 | 52,076 | 15,886 | unscorable（buildings < 10） |
| 1 | 0.7206 | **0.9828** | **+0.2622** | 5/5 正 | 25 | 154,576 | 39,135 | **可由 target labels 恢復的 unseen-site penalty** |
| 2 | 0.7189 | **0.8821** | **+0.1632** | 5/5 正 | 68 | 366,316 | 80,897 | **可由 target labels 恢復的 unseen-site penalty** |
| 13 | 0.6191 | **0.6686** | +0.0495 | mixed | 77 | 185,528 | 59,283 | **target labels 無法恢復** |
| 10 | 0.5716 | **0.6274** | +0.0558 | mixed | 15 | 106,740 | 15,814 | **target labels 無法恢復** |
| 3 | 0.8848 | 0.8331 | −0.0518 | mixed | 137 | 5,732 | 2,684 | unscorable；transfer 已可用 |
| 4 | 0.7627 | 0.3920 | −0.3707 | 5/5 負 | 45 | 15,560 | 197 | unscorable（低 prevalence 使 eval 側僅 197 anomalies） |
| 11 | 0.3594 | 0.0183 | −0.3412 | 5/5 負 | 3 | 19,124 | 197 | **unscorable**；退化案例，見 §12.3.1 |

**（a）可由 target labels 恢復的 unseen-site penalty：sites 1、2。** In-site oracle 分別到 0.9828 / 0.8821，五個模型全部同號。Site 1 的 oracle 只用 25 buildings、154,576 fit rows，勝過用 1,031 buildings、3,913,152 fit rows 訓練的 cross-site 模型 `+0.2622`——25 倍資料劣勢下仍大勝，是最強的單點證據。

**（b）target labels 無法恢復：sites 13、10。** 判讀依據是 oracle 的絕對水準只有 0.6686 / 0.6274，為全部可解讀 sites 中最低兩名；其 gap 的模型間符號不穩定，本身即與 0 無異。Site 13 的 oracle 擁有 77 buildings、185,528 fit rows 且零 domain shift，仍停在 0.6686；cross-site 用 3,913,152 fit rows 停在 0.6191。兩個相差 21 倍的資料 regime 收斂到同一水準。內部對照：**site 1 的 oracle 資料比 site 13 更少**（154,576 vs 185,528 fit rows、25 vs 77 buildings）卻達 0.9828，故「資料量不足」無法解釋 sites 13/10。

**（c）Anomalies 非 site-specific：sites 15、6（7 因 buildings < 10 記 unscorable，但落在同一族）。** In-site oracle 在**五個模型上全部**輸給 cross-site，幅度 −0.10 至 −0.17，paired-eval anomalies 有 17,989 / 28,654，非噪音。解讀不是「target labels 有害」，而是 cross-site 模型的 1,000+ buildings 多樣性勝過該 site 自身的 20–60 buildings；這些 sites 的 anomalies 不具 site 特異性，這正是其 cross-site PR 達 0.85–0.91 的原因。**部署意涵：sites 6、15 不應投入 target-site labelling，全域模型嚴格較優。**

**（d）Unscorable：sites 3、4、12（低 prevalence 餓死 oracle）與 7（buildings < 10）。** 因 fit rows = 4 × train anomalies（§157 的 sampling shape），低 prevalence 的 sites 系統性地讓 oracle 挨餓：site 3（prevalence 0.174%）oracle 只有 5,732 fit rows、site 12（0.468%）只有 2,896。Sites 3、4、12 的 cross-site PR 為 0.88 / 0.76 / 0.999，**本無需此診斷**。Site 11 也記 unscorable，但成因與上述四者不同，見下節。

### 12.3.1 Site 11 是 building 級的退化案例，不是 site 級的難度

Site 11 在 M6 中每一個設計都墊底（A1 `0.1485`、A3 OOF `0.2349`、in-site oracle `0.0183`），且 B1 顯示它是唯一隨 source labelling 增加而**下降**的 site。成因不在任何模型結果裡，直接讀資料即可確定：

| Building | 奇偶 | Rows | Anomalies | Anomaly rate |
|---:|---|---:|---:|---:|
| **1028** | even | 23,414 | **4,589** | **19.599%** |
| 1029 | odd | 17,548 | 3 | 0.017% |
| 1030 | even | 26,073 | 190 | 0.729% |
| 1031 | odd | 26,078 | 194 | 0.744% |
| 1032 | even | 26,346 | 2 | 0.008% |

**Building 1028 一棟即佔 site 11 全部 4,978 個 anomalies 的 92.2%**，其 anomaly rate `19.599%` 是站級 `4.167%` 的 4.7 倍、全資料集 `6.502%` 的 3 倍。其餘四棟合計僅 389 個 anomalies，其中兩棟各只有 3 個與 2 個。**Site 11 在統計上不是一個 site，而是一棟 anomalous building 加四棟近乎乾淨的 building。**

這一次解釋 M6 對 site 11 的全部三種失敗：

- **A1/A3（test = 全 site 11）**：要找的 anomalies 有 92% 在未見過的 building 1028。
- **A5/a0（test = odd buildings 1029、1031）**：eval 側只有 `3 + 194 = 197` 個 anomalies，其中 building 1029 是 17,548 rows 中 3 個。所謂 test 實際上只有 building 1031 的 194 個。
- **A5 oracle（train = even buildings 1028、1030、1032）**：訓練 anomalies 的 96% 來自 building 1028，模型學到的是 1028 的特徵，再拿去預測 1029/1031。其 `0.0183` 是 **building 級異質性**，不是 site 級難度。

**因此不可依 §134 推論 site 11「在 frozen feature/model contract 下本質較難」**；分析單位錯了。對照 site 7（同為小 site，15 buildings）：其 anomalies 分布均勻（odd half 15,886 anomalies、`7.919%`），是一個統計上成立的 site，site 11 不是。

**建議：site 11 退出 site 級結論，或在每一處引用時標為退化案例。** 此判定不需要任何額外 run，且應在 §12.2 的 per-site 清單階段即可發現——本報告未能及早發現，誠實記錄於此。

### 12.4 A5 的方法限制：site 身分與 building 多樣性無法分離

In-site oracle 以「同 site 的 20–77 buildings」置換「他 site 的 1,031–1,147 buildings」，因此 gap 同時包含 site 身分的增益（+）與 building 多樣性的損失（−），兩者無法由本設計分離。實證支持此限制：

- gap 與所有 support 變數的相關性都很弱（|r| < 0.37；gap vs oracle fit rows 僅 **+0.137**，gap vs oracle train buildings 為 **−0.070**），故「負 gap 源自 oracle 資料不足」不成立。
- 決定性配對：**site 1（25 buildings / 154,576 fit rows）與 site 6（22 buildings / 152,016 fit rows）oracle 訓練規模幾乎相同，gap 卻為 `+0.2622` 與 `−0.1038`，且各自五個模型全部同號。**

因此 §3 表中「Cross-site prediction vs paired in-site oracle」一列的「可回答的問題」應理解為受此限制約束：A5 能回答「使用 target-site labels 可恢復多少表現」，但不能將 gap 拆解為 site-shift 與 building-diversity 兩項。要分離需另設 arm（cross-site 全資料 + target-site 少量 labels），不在 M6 範圍。

同時，A5 的 oracle 本身亦為 building-held-out 任務，故 sites 13/10 的結論僅能表述為「**目前測過的訓練配置（他 site 資料，或同 site 其他 buildings）皆無法使其超過 0.67**」，不可外推為「frozen contract 無法表達該 site 的 anomalies」——後者需要 within-building 設計，M6 未做。

### 12.5 Source-site coverage 已飽和

A1/A2 與 A3 對同一 site 使用**完全相同的 test rows**（逐 site 核對 `n_rows` 全等），且 A3 各 fold 的 source set 是 A1/A2 source set 的**嚴格超集**（例：fold 3 訓練 `0,1,2,4,5,6,8,9,10,12,13,14` ⊃ A1 的 8 個偶數 sites）。因此 `Δ = A3 OOF PR − 50/50 PR` 是「額外 4 個 source sites」的乾淨、prevalence-controlled 測量（sites +50%，anomalies 904,080 → 1,240,266，+37%）。

**16 sites 的平均 Δ = `+0.002`**，正負皆有（最大 `+0.086` 為 site 11，最小 `−0.074` 為 site 8），散佈約 ±0.07。Source sites 由 8 增至 12 實質買不到東西；per-site 難度是該 site 的穩定屬性，而非 fold composition 的產物。

**對 §9 Stage 5（A4 LOSO）的建議：不執行。** A4 的進入條件是「A3/A5 顯示仍需 finer site diagnosis，且成本可接受」。上述飽和證據預測 A4（15 source sites）將重現 A3 的 per-site 數值。針對唯一未解的 site 11，A4 亦無法解決其根本限制（5 buildings）：site 11 在 source sites 由 8 增至 12 時由 `0.1485` 升至 `0.2349`（全場最大增幅），再增 3 個 sites 不足以進入可用區間。此建議不撤銷 stage gate，由使用者決定。

### 12.6 Pooled PR-AUC 跨 cell 不可比

PR-AUC 的隨機基線等於 prevalence，而各 cell 的 test-side prevalence 差異極大，故跨 cell 的 pooled 比較無法識別：

| 比較 | Test prevalence | Raw pooled PR 的結論 | Lift（PR ÷ prevalence）的結論 |
|---|---|---|---|
| A1 vs A2 | 3.601% / 10.252% | A2 勝（0.8954 vs 0.6447） | A1 勝（17.9× vs 8.7×） |
| A3 四折排名 | 14.98% / 5.00% / 8.16% / 1.59% | fold 0 > 2 > 3 > 1 | fold 3 > 1 > 2 > 0 |

兩種正規化給出相反排序，故兩者皆不採用（lift 的上限為 1/prevalence，機械上偏袒低 prevalence 的 cell，同樣不是有效正規化）。A3 各折 pooled PR 由 `0.6480` 至 `0.9875` 的分散**主要來自 prevalence，不是 ranking quality 崩潰**：同組 ROC-AUC 僅在 `0.9220`–`1.0000` 的窄帶內。真正的訊號在 per-site 層（fold 3 的 macro-site PR min = `0.2349`）。此結果印證 §6.1 將 per-site macro 而非 pooled 列為主要結論依據的決定。

A3 四折：

| Fold | Test sites | Test prevalence | Pooled PR | Pooled ROC | Macro-site PR mean | Macro-site PR min |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0,4,8,12 | 14.98% | 0.9875 | 0.9958 | 0.9459 | 0.7990 |
| 1 | 1,5,9,13 | 5.00% | 0.6480 | 0.9652 | 0.7815 | 0.5516 |
| 2 | 2,6,10,14 | 8.16% | 0.8130 | 0.9778 | 0.8139 | 0.7743 |
| 3 | 3,7,11,15 | 1.59% | 0.8083 | 0.9951 | 0.6914 | 0.2349 |

### 12.7 A2 source-calibrated operating point

契約確認：A2 canonical cell 的 `operating_points` 只有 `threshold_0_5`，無 `fixed_recall_0_90`，符合 §210 的 no-test-label-leakage 要求；`source_calibrated_recall_0_90` 僅存在於 sourcecal variant。

門檻 `0.5175691547` 學自 source calibration set（`building_id % 5 == 4`，完全排除於 model fit 外），於該處 recall 為 `0.9000`。套用至 unseen test sites 後：recall `0.8579`、precision `0.7422`。**此 aggregate 數字掩蓋 per-site 校準崩潰**：

| Test site | Recall | Precision | Predicted-positive rate | Actual prevalence |
|---:|---:|---:|---:|---:|
| 0 | 0.992 | 0.993 | 33.10% | 33.11% |
| 12 | 0.999 | 0.785 | 0.60% | 0.47% |
| 4 | 0.988 | 0.617 | 0.88% | 0.55% |
| 8 | 0.923 | **0.258** | **27.40%** | 7.66% |
| 6 | 0.853 | 0.748 | 11.38% | 9.98% |
| 10 | 0.832 | 0.644 | 13.35% | 10.33% |
| 14 | 0.739 | 0.711 | 9.01% | 8.67% |
| 2 | 0.718 | 0.736 | 6.65% | 6.82% |

八個 test sites 中僅四個達到 0.90 recall 目標，其中 site 8 是以 3.6 倍過度預測換得（115,420 個 false positives，precision `0.258`）。兩個大型 sites（2、14）的 recall 僅 `0.718` / `0.739`。**單一 global threshold 由 source sites 校準後，無法保證任何個別 unseen site 達到目標 recall。** Site 11 的 calibration 子集只有 3 個 anomalies（17,548 rows），支撐度極低。

### 12.8 依 §10 的措辭裁定

**已獲授權：**

- Sites 1、2：`該 site 存在可由 target-site labels 恢復的 unseen-site penalty`（§309），並依該條同報 paired oracle gap。
- Sites 13、10：`該 site 在 frozen feature/model contract 下本質較難，不能把全部落差歸因為 unseen-site shift`（§308），但受 §12.4 的表述限制約束。

**未獲授權：**

- `在全部 16 sites 的 grouped out-of-fold evaluation 下具有穩定 cross-site generalization`（§307）——該條要求 site dispersion 可接受，而 fold 3 的 macro-site PR min 為 `0.2349`，不可接受。
- `在兩個互補 50/50 site directions 下方向一致`（§306）——A1/A2 的 prevalence confound（§12.6）未解，且 A2 的 source anomalies（410,394）僅為 A1（904,080）的 45%，該變因須待 B2 隔離。
- Sites 3、4、7、11、12 的任何 A5 本質性結論——unscorable。
- `cross-country transfer validated`（§311）——C1 未執行。

### 12.9 尚未回答

- **B1/B2 未完成**：B1 執行中（9/30），B2 未開始。A2 以 410,394 個 source anomalies 取得高於 A1（904,080）的 raw pooled PR，此變因至今未隔離，是 §12.6 中 A1/A2 方向比較無法定案的直接原因。
- ~~**Site 11 成因未知**~~ — **已解決**，見 §12.3.1：92.2% 的 anomalies 集中在單一 building 1028，屬 building 級退化案例，非 site 級難度。
- **Sites 13/10 的失敗層級未定**：§12.4 已說明 A5 無法區分 site-level 與 building-level 的困難；within-building 設計不在 M6 範圍。
- **Aggregate 未產出**：`data/processed/m6_site_transfer_plot_data.json` 待 B1/B2 完成後由 `scripts/aggregate_m6_site_transfer.py` 產生；§8 的 Figure 1–10 尚未繪製。
