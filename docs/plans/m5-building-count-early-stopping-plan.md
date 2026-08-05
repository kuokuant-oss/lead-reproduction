# M5 建築物數量曲線與 Tree Early Stopping 計畫

**Status**: Pipeline correction validated; formal rerun pending
**Started**: 2026-08-04
**Scope**: additive M5 comparison line；不改寫既有 M3/M5 已發布結果
**Formal run**: 使用者已於 2026-08-04 明確授權；由 gpu-host 脫離控制端連線執行

## 0. 已授權執行修正（取代下文衝突的舊草案）

1. 正式順序固定為：全偶數 building tree baseline 17 features、全偶數 building tree
   baseline 137 features；接著 K=10、20、50、100，每個 K 必須先完成 tree 137，
   再完成 TabPFN 137，兩者都完成後才進入下一個 K。K 實驗不跑 17 features。
2. K 的 building 集合為 deterministic strict prefixes；限制是當下 K 的平均每棟
   allocation 不超過 500 rows，所以 K=10/20/50/100 的總 context 上限依序為
   5K/10K/25K/50K，而不是限制每一棟都最多 500。每個新增 K block 依各棟實際
   available rows 做 deterministic proportional allocation，資料多的棟可高於 500，
   資料少的棟可低於 500。Row selection 只依 seed 42 與 raw row identity 的 stable
   hash，不看 anomaly label；每棟一旦加入後 quota 與 row prefix 固定，因此 building
   sets 與 row sets 都是 strict nested。不同實驗 context 分布不必相同。
3. Tree 以每五棟固定四棟 fit、一棟 external early-stop 的 building-disjoint 角色，
   fit rows 必須沿用凍結的 M3 `[negs1, pos, negs2, pos]` downsampling；external
   early-stop rows 保持未重抽樣的自然分布。先以固定 80/20 building roles 選 best
   iteration，再對同 K 的全部 available training source rows 套相同 M3 downsampling，
   以選定迭代數 final refit。全量 baseline 同樣先 ES 選迭代，再對所有偶數 building
   source rows 套 M3 downsampling 後 final refit。除 external early stopping 外不改動
   M3 tree training pipeline。
4. TabPFN K cells 沿用 matched-context 的 137-feature timestamp-merge、StandardScaler、
   seed 42、tabpfn 8.0.8 checkpoint 與 n_estimators=8；沒有 task-specific weight-update
   loop，因此不加入 early stopping。直接 baseline 使用 m5-matched-context-breakdown
   既有 50K/137 TabPFN prediction artifact，不重跑。
5. 每個 K manifest 必須逐棟揭露 position、site_id、building_id、primary_use、tree
   role、選取時主要 balance need、selection score、可用與實際選取 rows/anomalies/rate、
   meter types 與逐 meter rows/anomalies/rate；並揭露 K 總 site/meter 組成和 anomalous
   building-meter pair rate。
6. Tree 每個 component、TabPFN fitted state、每個 holdout prediction chunk 都是原子
   checkpoint。Supervisor 是可重入狀態機：驗證 COMPLETE marker、cell metadata 與
   prediction schema 後跳過完成工作；未完成工作永久重試。Windows Session-0 WMI
   watchdog 若發現 tmux 消失，會重新啟動 supervisor 並以 resume 繼續。
7. 發布關卡為每個 tree baseline 完成後，以及每個 K 的 tree+TabPFN 都完成後。
   每個關卡重建同一份 docs/reports/m5-building-count-experiment.md，更新 overall、
   per-meter、per-site PR-AUC/ROC-AUC 與 plot-ready ROC/PR curve artifacts，然後
   自動 commit 並 push 到 GitHub main。Push 失敗也會重試，不會靜默前進。

## 1. 目的

本工作新增一條以「標註資料來自多少棟建築物」為橫軸的 TabPFN-vs-tree
比較線，取代新實驗中原本只以 5K、10K、20K、50K、100K rows 表示資料量的方式。
同時修正 matched-context tree arm 沒有 early stopping 的方法學缺口，並把中間產物、
分組指標與 ROC/PR 曲線保存到足以重建 `m5-matched-context-breakdown` 同等或更詳細的程度。

本計畫回答五個問題：

1. 四個 tree component models 如何使用合理、無 building leakage 的 early stopping。
2. TabPFN 為何不使用傳統 early stopping，以及如何和需要 validation 的 trees 公平比較。
3. 10、20、50、100 等建築物 budget 如何形成確定、可重現、嚴格巢狀的集合。
4. 新實驗如何沿用 canonical M3/M5 feature、split、holdout 與預測 pipeline。
5. 如何保存 overall、per-meter、per-site 的 PR-AUC、ROC-AUC、預測與完整可重畫曲線資料。

## 2. 已確認的現況

### 2.1 Tree baseline

`run_m5_tree_ensemble_matched_context.py` 重用 `fit_frozen_models()`；目前四個模型皆為
固定迭代數：LightGBM 100、XGBoost 100、CatBoost 1000、HistGBT 100。fit 時沒有
傳入獨立 validation set，因此沒有 early stopping，也沒有 best iteration provenance。
這條歷史 frozen line 保留不動；新協定另建 early-stopped model contract，避免既有
報告數字被靜默重新定義。

### 2.2 TabPFN

專案鎖定 `tabpfn==8.0.8`。現有 runner 載入預訓練 TabPFN checkpoint，`fit()` 將
本任務 labeled examples 建成 inference context；它不對 foundation-model weights
執行 task-specific epoch/gradient optimization。因此傳統「監控 validation metric，
停止 weight updates」的 early stopping 不適用，也不應為了形式對稱而捏造。

若未來啟用 TabPFN fine-tuning、metric tuning、thinking mode 或任何會使用 validation
labels 做模型選擇的擴充，該 cell 必須換成另一個 protocol 名稱並將 validation labels
計入 budget；不可和本計畫的純 in-context TabPFN line 混報。

### 2.3 公平比較的主定義

公平的主軸是 **equal labeled-building acquisition budget**：在 K-building cell 中，
兩個 model families 可取得完全相同的 K 棟建物與其 labeled rows。TabPFN 因不需要
early stopping，可把 K 棟全部放入 context；trees 必須在同一 K 棟內切出 building-
disjoint fit/early-stop subsets。Early-stop buildings 的 labels 是 tree model selection
所使用的監督訊息，因此算在 K 的 budget 內，不能額外從 budget 外借一組 validation
建物。

結果必須同時回報：

+ `available_buildings=K`：主要 x 軸與共同 label acquisition budget。
+ `context_buildings/context_rows`：TabPFN 實際 context。
+ `tree_fit_buildings/tree_fit_rows`：tree 參數 fit 所用資料。
+ `tree_es_buildings/tree_es_rows`：tree early stopping 所用資料。

這個設計比較的是「給演算法相同標註來源成本後可達到的表現」，不是強迫兩種學習機制
具有相同 fit API。另輸出 sensitivity view，將 TabPFN 限制在 tree-fit buildings，
用來量化 trees 因 early stopping 保留建物所付出的 effective-fit-data cost；該 view
只作敏感度分析，不取代主比較。

## 3. 固定實驗邊界

+ 資料與 labels：沿用 M3 `load_m3_frame()` 與 buds-lab anomaly labels。
+ 外部 test：固定 `building_id % 2 == 1` 的 canonical 50/50 holdout；所有 K、模型、
  feature lines 都對 byte-identical test rows 預測。
+ 候選來源：只從 `building_id % 2 == 0` 的 training half 選建物。
+ Feature lines：17-feature baseline 與 137-feature `timestamp_merge` line 分開執行；
  scaler 與 feature construction 延續 canonical pipeline。
+ 正式 building budgets：預設 `10 20 50 100`，CLI 可加入更大 K，但必須是同一
  ladder 的 prefix。
+ Seeds：building order、role allocation、model 與任何 row cap 各自顯式記錄。
+ 既有 5K--100K row-context artifacts、M3 frozen model contract 與 golden metrics
  不重算、不覆寫。
 Tree training sampling：fit 與 final refit 固定使用 M3
  `[negs1, pos, negs2, pos]`（seeds 10/20）；scaler 只 fit 在 downsampled training
  matrix。external early-stop 與 odd-building test 不 downsample，以保留驗證與測試的
  原始 class distribution。

## 4. 建築物巢狀集合設計

### 4.1 建築物 profile

先由 training half 建立一列一建物的 profile CSV，至少包括：

+ `building_id`、`site_id`、`primary_use`。
+ 總 rows、normal rows、anomaly rows、building anomaly rate。
+ 各 meter 的 rows、anomalies、anomaly rate 與 meter-presence indicators。
+ meter 數量、異常 meter 數量、零異常指標。
+ `log1p(rows)` size bin 與 anomaly-rate bin。

profile 只由 training half labels 計算；不可查看 odd-building test labels 來決定順序。

### 4.2 Primary ladder：representative-balanced

以 deterministic greedy ordering 產生一次完整 building order，再令每個 K cell 使用
`order[:K]`。每一步從尚未選取的建物中選擇最能降低目前 prefix 與 training candidate
pool 目標分布差距者；目標包含：

1. site 建物占比；避免大型 site 在小 K 完全壟斷，同時不把極小 site 強行放大成
   1/16 的不自然權重。
2. 四類 meter 的 building-presence 與 row-share。
3. anomaly-rate bins（含 zero-anomaly、低、中、高）與 anomaly-bearing meter 數量。
4. building row-count bins，避免前綴只由最大建物或最小建物組成。

差距使用標準化 squared error；同分時以 seed-based stable hash、再以 building_id
決勝。演算法、權重、bins 與 target proportions 全部寫入 manifest。每個 requested K
都必須通過 `set(K_small) < set(K_large)`、唯一性、train-only 與 digest gates。

### 4.3 Sensitivity ladders

CLI 另支援下列 deterministic profile，但不和 primary line 平均：

+ `site_balanced`：site 目標改成均等，回答地理來源多樣性。
+ `meter_balanced`：meter-presence/row-share 權重提高，回答 meter mix。
+ `anomaly_balanced`：anomaly-rate bins 權重提高，檢查異常來源稀缺性。

報告先呈現 primary line；若三條 sensitivity ladder 的模型排序不同，必須明確報告
結論依賴 sampling profile，不得只挑有利的一條。

### 4.4 Tree fit / early-stop role

building order 在產生時同步分配固定角色：每 5 個位置含 4 個 fit 與 1 個 early-stop
building。正式 K 都是 10 的倍數，因此每個 cell 是精確 80/20 buildings；角色在較大
prefix 中不變，使 fit 與 early-stop subsets 也各自巢狀。挑選 early-stop 位置的建物時
同樣最小化 site、meter、anomaly-rate 與 size discrepancy，避免 validation 只來自
單一 site/meter。

每個 K 必須通過：fit/ES/test building-disjoint、fit/ES 皆有兩類 labels、四種 meter
coverage 可稽核，以及 ES composition deviation gate。若資料支持不足，cell 應明確
blocked，不得退回 row-random split。

### 4.5 Row policy

`row_policy` 定義所選 K 棟可提供給兩個 model families 的 labeled source rows；
`row_policy=all_rows` 表示 available pool 保留全部可用 rows，因此 K 是資料來源數與
自然資料量共同成長的軸。這不會覆蓋凍結的 M3 tree training sampling：tree fit 與
final refit 必須在各自 available source pool 上套 `[negs1, pos, negs2, pos]`，只有
TabPFN context 使用其允許的 available rows。manifest 與結果必須同時揭露 source
rows、downsampled effective fit rows、class prevalence、每棟 row contribution 與
最大建物占比。

為資源可行性保留顯式 `balanced_cap` sensitivity：先在每棟、每 label、每 meter 內以
stable row priority 抽樣，再套 per-building cap。它不得成為隱藏預設；使用時必須在
artifact 名稱、圖例與報告標明 cap，並保持相同 K 下 TabPFN/trees 的 available rows
一致。正式執行前 preflight 先列出各 K 的 row census，超出 TabPFN checkpoint/硬體
能力時由人工決定是否啟用此 sensitivity，pipeline 不自行改 protocol。

## 5. Tree early stopping contract

所有模型都只以 K cell 內的 ES buildings 監控 validation ranking，預設共同選擇指標為
ROC-AUC（與 M3 headline 一致且四套 library 定義一致）；PR-AUC 同步紀錄但不作停止
依據。可設定 `--early-stopping-metric pr_auc` 作預先聲明的 sensitivity，不可在看過
test 後切換。

| Model | 最大迭代 | Early stopping | Best-iteration scoring |
| --- | ---: | --- | --- |
| LightGBM | 5,000 trees | external ES set、patience 100、`min_delta=1e-5` | `best_iteration_` |
| XGBoost | 5,000 trees | external ES set、patience 100、`eval_metric=auc` | `best_iteration`/best model |
| CatBoost | 5,000 trees | external ES set、`eval_metric=AUC`、`od_wait=100`、`use_best_model=True` | `best_iteration_` |
| HistGBT | 1,000 iterations | explicit `X_val/y_val`、`early_stopping=True`、`n_iter_no_change=20`、`tol=1e-5` | `n_iter_` |

保留既有 structural hyperparameters 與 seed；提高的上限是 early stopping search ceiling，
不是宣稱每個模型都應跑滿。每個 component 必須保存 validation history、best iteration、
best validation ROC/PR、fit time、停止原因與 library version。Ensemble 仍是四模型等權平均，
不利用 test labels 調權。

若某 library 無法在目前鎖定版本使用 explicit external ES buildings，pipeline 應 fail
closed；不可改用其內部 row-random validation。當 K 太小或 ES labels 單一類別時亦
fail closed。

## 6. TabPFN contract

+ 使用與該 K primary cell 相同的 available buildings/rows、feature line、scaler
  protocol、seed 與 canonical test order。
+ 不建立 early-stop split、不更新 checkpoint weights、不使用 test 或 ES metric 選擇
  `n_estimators`。
+ `n_estimators`、checkpoint SHA256、TabPFN/library version、effective context rows、
  subsampling disabled status 與 fit-state digest 必須固定並記錄。
+ 若 `all_rows` 超出可驗證 context 能力，cell 標為 blocked_resource/protocol；不得
  靜默 subsample。

## 7. Pipeline 與 checkpoint layout

新增一條 additive building-curve pipeline，分為可獨立恢復的 bounded phases：

1. `profile`：產生 building profile、完整 nested ladder 與 composition audits。
2. `features`：按 feature line 建立/驗證 canonical feature artifacts。
3. `tree_fit`：unit = `(profile, K, features, component_model)`；原子保存模型、history
   與 fit manifest。
4. `tabpfn_fit`：unit = `(profile, K, features)`；沿用 isolated GPU worker 與 fitted-state
   checkpoint。
5. `predict`：unit = `(profile, K, features, model, holdout_chunk)`；每個 chunk p95 目標
   小於 10 分鐘。
6. `aggregate`：只在所有 expected units 完整、digest/provenance 相符時合併 predictions。
7. `report/plot`：由 prediction artifacts 重建 tidy metrics、curve points、圖與 Markdown。

每個 phase 都必須有：atomic temp-write/flush/fsync/rename、schema+digest validation、
deterministic resume、repo/source/input/row/group digests、JSONL progress、heartbeat、
completed/total、throughput/ETA、COMPLETE marker 與 missing-unit finalization refusal。
正式 run 不設 wall-clock timeout 或 auto-kill。

## 8. 詳細輸出契約

### 8.1 Composition/provenance

+ `building_profiles.csv`：一建物一列的 source profile。
+ `building_ladder.csv`：profile、order、role、首次納入 K、stable priority。
+ `building_ladder.json`：設定、targets、weights、bins、digests、每 K composition audit。
+ 每 cell manifest：available/tree-fit/tree-ES buildings 與 rows、labels、sites、meters、
  anomaly rates、row-concentration、feature/scaler/model/checkpoint provenance。

### 8.2 Predictions

每個完成 cell 的 canonical NPZ 至少保存：

+ `validation_raw_index`、`anomaly`、`building_id`、`site_id`、`meter`。
+ TabPFN、四個 tree components 與 tree ensemble 的 probability scores。
+ prediction arrays 與 row identity digests；合併前逐列檢查所有模型的 row/label order。

### 8.3 Metrics 與曲線

輸出 tidy tables，一列為 `(profile, K, features, model, grouping, group)`：

+ support：rows、buildings、anomalies、anomaly rate、sites、meters。
+ ROC-AUC、PR-AUC、threshold 0.5 precision/recall/F1/confusion matrix。
+ validation-derived fixed-recall threshold 及對應 test metrics（不得由 test 選 threshold）。
+ paired model differences；薄 slice 保留 paired bootstrap 欄位與 context-seed uncertainty
  的擴充位置。

保存 ROC `(fpr,tpr,threshold)` 與 PR `(recall,precision,threshold)` curve artifacts，範圍為：

+ overall 全部 canonical test rows。
+ 每個 meter（electricity/chilledwater/steam/hotwater）。
+ 每個 site 0--15。

原始 prediction artifacts 是可精確重畫的權威來源；壓縮 curve points 只供快速繪圖，
不可取代 predictions。

## 9. 圖表

至少產生：

1. building-count scaling：x=K，y=PR-AUC/ROC-AUC；overall、per-meter、per-site；
   TabPFN 與 tree ensemble 同圖，並標示實際 context rows。
2. prediction curves：可選 K/features/group，輸出 TabPFN、tree ensemble 與 components
   的 ROC 與 PR 曲線。
3. composition audit：各 K 的 site、meter、anomaly-rate bin、row contribution 與
   tree fit/ES role 分布。
4. early-stopping diagnostics：各 K/component 的 best iteration、validation curve 與
   ceiling-hit flag。

圖表遵守 `docs/reference/plot-style-rules.md`，圖與其 metadata/provenance 一起生成。

## 10. 執行步驟與驗收

### Slice A：協定與抽樣核心

1. 新增 ADR，記錄 equal labeled-building budget、TabPFN 不使用傳統 early stopping、
   tree ES labels 計入 K，以及 primary/sensitivity ladder。
2. 實作 building profile、deterministic greedy ladder、role allocation、manifest/digest。
3. 測試 exact K、strict nesting、determinism、seed sensitivity、train-only、role nesting、
   composition、single-class failure 與 synthetic reference equivalence。

### Slice B：Early-stopped tree arm

1. 新增 model contract 與四套 external-validation fit adapters，不改 frozen M3 helper。
2. 記錄 best iteration/history/stop reason，驗證 inference 使用 best model。
3. 測試所有 adapter 真的讀取 explicit ES set、ceiling-hit flag、無 row-random fallback，
   以及同 seed 重現。

### Slice C：TabPFN/building cell orchestration

1. 將 building manifest 轉成 TabPFN context 與 tree fit/ES indices。
2. 接上現有 TabPFN isolated worker、resource guard、fitted-state 與 prediction chunks。
3. 實作 unit census、provenance mismatch hard-fail、resume 與 bounded validation mode。

### Slice D：Aggregation、breakdown 與 plots

1. 合併 prediction chunks前執行 raw_index/label/site/building/meter identity gates。
2. 產生 overall/per-meter/per-site tidy metrics 與 ROC/PR curve artifacts。
3. 產生 building scaling、prediction curves、composition 與 early-stopping diagnostics。
4. 報告格式至少達到 `m5-matched-context-breakdown` 的 support、分組與 paired-difference
   詳細度。

### Slice E：驗證與交付

1. unit tests：sampling、model adapters、metrics/curves、atomic checkpoints、corrupt/temp
   rejection、resume、missing finalization、mode guard、bounded caps、heartbeat/log flush。
2. bounded non-scientific validation：synthetic/small deterministic buildings、每個 expensive
   phase 有明確 unit cap、輸出到隔離目錄並標示 NON_SCIENTIFIC_VALIDATION。
3. 執行 targeted tests、full tests、ruff、markdownlint、pre-commit；檢查 UTF-8 diff。
4. 更新 README、M5 plan tracker、ADR 與 handoff。
5. 提交 clean implementation 後等待人工明確授權 formal scientific run；此工作不自動
   啟動 10/20/50/100 全量 CPU/GPU 實驗。

## 11. 完成條件

+ Tree arm 四模型都有 building-disjoint external early stopping 與 best-iteration evidence。
+ TabPFN no-early-stopping 決策有官方設計與本地 API/provenance 證據，且沒有額外 model
  selection labels。
+ 10/20/50/100 是同一 deterministic building ladder 的 strict prefixes。
+ 同一 K 的兩個 model families 共享相同 available buildings、features 與 canonical test；
  差異與 effective-fit-data 均明確揭露。
+ pipeline 可 checkpoint/resume，bounded validation 通過，未經授權不啟動 formal run。
+ 每個完成 cell 可重建 overall/per-meter/per-site 的 PR-AUC、ROC-AUC、ROC/PR curves、
  composition 與 early-stopping diagnostics。

## 12. 2026-08-04 實作紀錄

+ Protocol：`scripts/m5_building_curve_protocol.py`、
  `scripts/prepare_m5_building_curve.py`。
+ Tree early stopping：`scripts/m5_tree_early_stopping.py`、
  `scripts/run_m5_building_curve_tree_cell.py`。
+ TabPFN building cell：`scripts/run_m5_building_curve_tabpfn_cell.py`；明確記錄
  `early_stopping=not_applicable_in_context_learning_no_weight_updates`。
+ Breakdown/curves：`scripts/report_m5_building_curve.py`、
  `scripts/plot_m5_building_curve.py`。
+ Tests：14 個新增 unittest 通過；涵蓋 even-building-only、strict nesting、role nesting、
  external ES adapters、single-class fail-closed、overall/meter/site metrics、ROC/PR curves、
  plan-mode no-launch 與 validation/formal cap guards。新增檔案通過 ruff。
+ 未執行 formal 10/20/50/100 cells；依 repository policy，需先 clean commit，再由 operator
  明確授權 CPU/GPU scientific run。
