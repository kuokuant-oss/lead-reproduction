# M5 building-count experiment V2 plan

## 狀態與範圍

本計畫是獨立的 V2 protocol。它不覆寫既有 M5 正式報告、V1 artifacts 或已封存的 representative-greedy run，也不啟動模型。

研究問題是：在 source-building 數量從 K=10 增加到 20、50、100 時，固定其他條件後，TabPFN 與 frozen tree ensemble 的 canonical odd-building holdout performance 如何變化，以及這個趨勢對不同 source-building draws 有多敏感。

V2 identity 為：

```text
(experiment_version, sampling_profile, building_seed, K, features, model,
 row_seed, model_seed)
```

本輪只使用 137-feature pipeline，building seeds 為 42、43、44、45、46；row seed 與 model seed 都固定為 42。

## Source-building sampling

實作與完整說明見 [m5-building-ladder-selection.md](../reference/m5-building-ladder-selection.md)，sampling audit 由 [audit_m5_building_candidate_sensitivity.py](../../scripts/audit_m5_building_candidate_sensitivity.py) 產生。

固定 protocol：

- Candidate pool 僅包含 even-ID training buildings；odd-ID buildings 完整保留為 canonical holdout。
- 每個 site 內以 `numpy.random.PCG64(building_seed, redraw_attempt)` 形成 seeded permutation，再依 candidate-pool site proportions 交錯組成一條 ladder。
- Sampling without replacement；K10、K20、K50、K100 是同一條 accepted ladder 的 strict nested prefixes。
- Meter 只作事先指定的 feasibility gate，不參與 ranking 或最佳化。K=10 每個 evaluation meter 至少有 2 棟 source buildings，且每次 K transition 每個 meter 至少新增 1 棟。
- 不符合 meter gate 時整條 ladder reject，以同一 building seed 的 deterministic redraw stream 重抽；不做 greedy correction、building swap 或 constraint relaxation。
- 最多 10,000 attempts，超過即明確失敗。
- Anomaly labels、anomaly rate、building size、meter row share 與其他 composition diagnostics 均不影響 building identity。

已完成的 sampling audit 使用 725 棟 even candidates。五個 seeds 的 accepted zero-based attempts 分別是 8、7、7、7、22；20 個 seed/K prefixes 都有不同 digest，所有 feasibility gates 通過。

## Row contract 與 matched context

每條 accepted ladder 的 row allocation 在 manifest 建立時一次完成：

- `row_policy=average_building_cap`。
- 每棟平均上限 500 rows，因此正式 K budgets 對應 5,000、10,000、25,000、50,000 allocated rows。
- `row_seed=42`，使用既有 stable raw-row priority；building seed 不改變已選 building 內的 row-priority policy。
- Tree 與 TabPFN 都直接使用 cell 的 `available_rows`，row order 與 row digest 必須 byte-identical。
- 使用 allocated rows 的 natural prevalence；V2 不新增 50:50 redraw，也不使用 M3 tree runner 的 anomaly duplication。
- Manifest 中每五個 position 的 role 仍保留作 audit metadata，但 V2 training 完全忽略 role。沒有 early-stop subset。

這裡參考 [m5-matched-context-breakdown.md](../reports/m5-matched-context-breakdown.md) 的兩個原則：兩個 model families 使用完全相同的 context rows，以及 tree 使用 frozen fixed-iteration contract 而非 early stopping。該舊實驗的 50:50 row draw 不移植到 V2，避免同時改動目前 building-count row contract。

## Models

Tree cell 由 [run_m5_building_count_v2_tree_cell.py](../../scripts/run_m5_building_count_v2_tree_cell.py) 執行，使用既有 test-guarded frozen contract：

| model | fixed fit budget |
|---|---:|
| LightGBM | 100 estimators |
| XGBoost | 100 estimators |
| CatBoost | 1,000 iterations |
| HistGradientBoosting | 100 iterations |

Tree 不建立 validation split、不傳 `eval_set`、不計算 best iteration，也不做 final refit。四個 model probabilities 的等權平均為 `ensemble`。

TabPFN 由 [run_m5_building_curve_tabpfn_cell.py](../../scripts/run_m5_building_curve_tabpfn_cell.py) 的 V2 identity mode 執行，固定 `n_estimators=8`。TabPFN 沒有 task-specific weight fitting，early stopping 不適用。

## Holdout 與 metrics

所有 40 個 cells 使用相同 canonical odd-building natural-prevalence holdout。Source selection 不讀取 odd-building data 或 labels；holdout 僅在 sampling 完成後用於 evaluation。

每個 cell 保存 raw predictions，並計算：

- overall PR-AUC 與 ROC-AUC；
- per-meter PR-AUC 與 ROC-AUC；
- per-site PR-AUC 與 ROC-AUC。

Raw per-seed results 必須保留。跨 building seeds 對每個 model/K 輸出 `n_building_seeds`、mean、sample SD、min、max；mean 不取代 raw results。

## Pipeline、artifacts 與 resume

Sweep orchestrator 是 [run_m5_building_count_v2.py](../../scripts/run_m5_building_count_v2.py)。預設 sweep 有：

```text
5 building seeds x 4 K budgets x 2 model families = 40 cells
```

獨立 artifact root：

```text
data/processed/m5_building_curve/v2/
  building_seed_sweep_42-43-44-45-46/
    model_runs/
      building_seed{seed}/
        tree_no_es_k{K}_f137/
        tabpfn_k{K}_f137/
    matched_context_gate.json
    aggregate/
      metrics.csv
      curves.csv
      building_seed_summary.csv
    sweep_status.json
```

每個 cell 都保存 provenance、checkpoint、heartbeat、prediction 與 COMPLETE marker。重新執行同一 command 會略過 identity 完整的 cells，並對未完成 cell 使用 `--resume`；若 result-affecting provenance 改變則拒絕沿用 checkpoint。V1 與 archived artifacts 不在這個 path 下。

## 必須通過的 gates

正式 aggregation 前全部成立：

1. Sampling audit status 為 passed，sampling profile 必須是 `site_stratified_random`。
2. 五條 ladders 均為 even-only、無 duplicate、strict nested，且 meter constraints 全通過。
3. Tree 與 TabPFN 的 `context_row_sha256`、row count、row policy、row seed、model seed 完全相同。
4. Tree 與 TabPFN 的 holdout digest 相同。
5. 實際 prediction artifacts 的 holdout raw index、label、building、site、meter arrays 完全相同。
6. Formal sweep 必須同時包含 tree 與 TabPFN，40 cells 全部完成才可更新 V2 report。
7. Cross-seed summary 的每個 model/K 必須有 5 個 raw building-seed results。

## 執行順序

只檢查 40-cell plan，不讀 frame、不 fit、不 predict：

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py --mode plan
```

小型 non-scientific end-to-end validation：

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py \
  --mode validation \
  --validation-context-rows 200 \
  --validation-holdout-rows 200
```

正式執行（需要通過 formal environment gate 與 TabPFN CUDA/model gate）：

```bash
.venv/bin/python scripts/run_m5_building_count_v2.py --mode formal
```

完成後 orchestrator 會先執行 matched-context gate，再 aggregate，最後以 [update_m5_building_count_v2_report.py](../../scripts/update_m5_building_count_v2_report.py) 重建 V2 report。正式執行前仍應以 clean commit 固定 code identity；本次準備工作本身不會啟動模型。
