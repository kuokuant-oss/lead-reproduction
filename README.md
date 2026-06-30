# LEAD Reproduction

本專案復現 Fu et al. 2022 BuildSys 論文
["Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition"](https://dl.acm.org/doi/abs/10.1145/3563357.3566147)，並把工作從 LEAD competition subset 延伸到 ASHRAE GEPIII raw dataset。

參考來源：

- 論文：Fu et al. 2022, BuildSys '22
- 原始解法：https://github.com/buds-lab/LEAD-1st-solution
- GEPIII reference：https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis

## 目前狀態

| Milestone | 範圍 | 狀態 | 主要結果 |
| --- | --- | --- | --- |
| **M1** | 閱讀 paper 與 buds-lab code，建立 unknowns register 與 ADR framework | Closed | 17 個 unknowns、ADR 0001-0006、169-feature 組成釐清 |
| **M2** | LEAD competition subset reproduction | Closed | Kaggle Private AUC `0.98616`，與原始解法 `0.98661` 的差距為 `0.05%` |
| **M3** | Full ASHRAE GEPIII reproduction | Complete | M3.4 ensemble AUC `0.9928`；PI 50/50 ensemble offline `0.9921` / causal `0.9911`；post-processing 為 null result |
| **M4** | Importable pipeline foundation | M4.0-M4.5 complete | `src/lead` public API frozen; M3.2/M3.4 regression gates pass; M4.2-M4.5 closed |
| **M5** | FDD on BDG2 | Model track (Phase A–D) complete；Phase E (BDG2) EDA stop point | FDD 選模階段完成：GEPIII 上 TabPFN-vs-GBDT 四軸比較（[報告](./docs/reports/m5-foundation-vs-gbdt.md)）— TabPFN 於 label scarcity（+0.100 PR-AUC @200 labels）與 true cross-site ROC-AUC（0.9833 vs 0.9797）勝出，GBDT 保有 inference 延遲與 minimal-FE 優勢，real-time 部署候選仍為 GBDT。Phase E 已完成 BDG2 read-only EDA，停等審查 |

Issue-level 進度見 GitHub [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones)。

## 主要文件

- **M2 復現報告**：[docs/reports/reproduction-report.md](./docs/reports/reproduction-report.md)
- **M3 完成報告**：[docs/reports/m3-report.md](./docs/reports/m3-report.md)
- **M4 評估報告**：[docs/reports/m4-evaluation-report.md](./docs/reports/m4-evaluation-report.md)
- **M5 FDD 選模報告**：[docs/reports/m5-foundation-vs-gbdt.md](./docs/reports/m5-foundation-vs-gbdt.md)
- **BDG2 EDA 報告**：[docs/reports/bdg2-eda.md](./docs/reports/bdg2-eda.md)
- **BDG2 data descriptor reference**：[docs/reference/papers/bdg2-miller-2020.md](./docs/reference/papers/bdg2-miller-2020.md)
- **工作方法**：[docs/reference/workflow.md](./docs/reference/workflow.md)
- **M4 計畫**：[docs/plans/m4-plan.md](./docs/plans/m4-plan.md)
- **M5 計畫**：[docs/plans/m5-plan.md](./docs/plans/m5-plan.md)

## Milestone 摘要

### M1 理解與決策框架

M1 不訓練模型，目標是把論文與原始碼中的關鍵決策變成可追蹤文件。

- `docs/reference/unknowns.md`：17 個 paper 或 code 未說清楚的地方。
- `docs/adr/`：目前共有 19 份 ADR；M1 產出 ADR 0001-0006。
- `docs/reference/paper-notes.md`：paper structured summary。
- `docs/reference/feature-engineering-rules.md`：feature 與 model 規則整理。

### M2 LEAD subset reproduction

M2 在 LEAD competition subset 上復現 169-feature pipeline、4-model ensemble 與 hard-rule post-processing。

主要結果：

- LightGBM 57-feature baseline validation AUC：`0.8952`
- 169-feature LightGBM validation AUC：`0.9818`
- 4-model ensemble validation AUC：`0.9830`
- Kaggle Private AUC：`0.98616`

詳細結果見 [docs/reports/reproduction-report.md](./docs/reports/reproduction-report.md)。

### M3 Full ASHRAE GEPIII reproduction

M3 從 ASHRAE GEPIII raw CSV 重建 feature engineering pipeline，使用 building-level validation split 驗證 M3.1-M3.5。

主要結果：

- M3.2 LightGBM offline AUC：`0.9920`
- M3.3 buds-lab alignment AUC：`0.9913`，判定為 no-lift。
- M3.4 4-model ensemble AUC：`0.9928`
- M3.5 hard-rule post-processing delta：`-0.000054`，判定為 null result。
- PI 50/50 ensemble offline AUC：`0.9921`
- PI 50/50 ensemble causal AUC：`0.9911`

詳細結果見 [docs/reports/m3-report.md](./docs/reports/m3-report.md)。Machine-readable provenance 放在 `docs/metrics/`。

### M4 Importable Pipeline Foundation

M4 把 notebook 與 script 中重複的 M3 helper 抽到 `src/lead`，先保留既有行為，再為後續語意修正建立 regression gates。

M4.0-M4.5 complete:

- `src/lead/data.py`：M3 data loading 與目前 positional label assignment。
- `src/lead/features.py`：value-change feature generation。
- `src/lead/split.py`：building-level split helpers。
- `src/lead/sample.py`：downsample index helper。
- `src/lead/evaluate.py`：AUC / precision / recall / F1 metrics。
- `src/lead/io.py`：JSON provenance helper。

M4.2 completed guarded positional label alignment; ADR 0010 is Accepted. M4.3 completed timestamp-merge value-change evaluation; ADR 0011 is Accepted. M4.4 preserved StandardScaler and positive-duplication sampling compatibility; ADR 0016 is Accepted. M4.5 completed the M5 readiness gate and froze the `src/lead` public API.

### M5 FDD on BDG2

M5 把工作從 reproduction 推進到 fault detection and diagnosis（FDD），最終目標是 BDG2。其 model track（Phase A–D）是 FDD 的選模階段：在現有 GEPIII 資料上以相同 split、downsample、feature table 與 seeds，對 TabPFN（tabular foundation model）與 GBDT 做四軸比較（in-domain、site transfer、label scarcity、minimal feature engineering），此階段已完成。

主要結果：

- TabPFN 在 label scarcity（200 labels 時 +0.100 PR-AUC）與 true cross-site ROC-AUC（`0.9833` vs GBDT-retrain `0.9797`）勝出。
- TabPFN 在 10k context 的 in-domain ROC-AUC（`0.9925`）接近 M3.4 ensemble（`0.9928`）。
- GBDT 保有推論延遲與 minimal feature engineering 的優勢，real-time 部署候選仍為 GBDT。

Phase E（BDG2）把選定的 FDD 模型轉移到 BDG2 corpus，但目前停在 pre-modeling EDA 審查點。模型選擇結果見 [docs/reports/m5-foundation-vs-gbdt.md](./docs/reports/m5-foundation-vs-gbdt.md)，Phase E 規劃見 [docs/plans/m5-plan.md](./docs/plans/m5-plan.md)。

Phase E Step 4 corrected the chilledwater BDG2 pilot gate and stopped before
full transfer: the pilot is underpowered because it has no powered
`bdg2_only__sufficient_obs` stratum. A pooled cross-site raw fallback was also
underpowered (`underpowered_even_pooled`, 3 BDG2-only sufficient-observation
buildings vs the 5-building minimum). Prior full/4b artifacts are quarantined
as diagnostics only, not accepted results. The finding is OOD-leaning rather
than missingness-only. BDG2 EDA has now reproduced the chilledwater sparsity
from the data side: 26 BDG2-only buildings have chilledwater columns, but only
3 satisfy the sufficient-observation rule. The EDA also quantifies OOD via
square_feet KS `0.2176`, sampled meter_reading KS `0.4549`, and primary_use
categorical PSI `1.415`. See
[docs/reports/phaseE-step4-bdg2-transfer.md](./docs/reports/phaseE-step4-bdg2-transfer.md)
and [docs/reports/bdg2-eda.md](./docs/reports/bdg2-eda.md).

## src/lead public API

M4.5 freezes `lead.__all__` as:

1. `ROOT`
2. `M3`
3. `PROC`
4. `RANDOM_STATE`
5. `DOWNSAMPLE_SEEDS`
6. `MODEL_SEEDS`
7. `SHUFFLE_SEEDS`
8. `BASELINE_FEATURE_COLS`
9. `BUILDING_META_FEATURE_COLS`
10. `CYCLIC_FEATURE_COLS`
11. `M3_3_EXTRA_FEATURE_COLS`
12. `WEATHER_LAG_BASE_COLS`
13. `WEATHER_WINDOWS`
14. `SHIFTS`
15. `PAST_SHIFTS`
16. `FUTURE_SHIFTS`
17. `load_m3_frame`
18. `load_bdg2_frame`
19. `add_value_change_features`
20. `split_mask`
21. `assert_no_building_overlap`
22. `leave_site_out_mask`
23. `downsample_indices`
24. `classification_metrics`
25. `write_json_with_provenance`

`add_value_change_features(df, shifts, value_change_regime=...)` supports `row_offset` and `timestamp_merge`. `row_offset` remains the M3 reproduction default.

## 專案結構

```text
docs/
├── plans/
│   ├── m1-plan.md
│   ├── m2-plan.md
│   ├── m3-plan.md
│   ├── m4-plan.md
│   └── m5-plan.md
├── reports/
│   ├── reproduction-report.md
│   ├── m3-report.md
│   ├── m4-evaluation-report.md
│   ├── m5-foundation-vs-gbdt.md
│   └── bdg2-eda.md
├── reference/
│   ├── workflow.md
│   ├── change-checklist.md
│   ├── unknowns.md
│   ├── paper-notes.md
│   ├── feature-engineering-rules.md
│   ├── papers/
│   │   └── bdg2-miller-2020.md
│   └── notebooks-map.md
├── metrics/
│   ├── m3-50-50-ensemble.json
│   └── m3-primary-use-auc.json
├── adr/
│   └── 0001-0019 decision records
├── handoffs/
│   └── historical session handoffs
├── agents/
│   └── agent workflow notes
└── assets/
    └── kaggle-final-score.png

notebooks/
├── 01-m2-baseline-pipeline.ipynb
├── 02-m2-clusterno.ipynb
├── 03-m2-value-change.ipynb
├── 04-m2-savgol-dayofyear.ipynb
├── 05-m2-integration.ipynb
├── 06-m3-baseline.ipynb
├── 07-m3-split-causality.ipynb
├── 08-m3-budslab.ipynb
├── 09-m3-ensemble.ipynb
└── 10-m3-postprocessing.ipynb

scripts/
├── run_m3_3_budslab.py
├── run_m3_4_ensemble.py
├── run_m3_5_postprocessing.py
├── run_m3_50_50_ensemble.py
└── run_m3_split_causality.py

src/lead/
├── data.py
├── features.py
├── split.py
├── sample.py
├── evaluate.py
└── io.py

tests/
├── golden_metrics.json
├── test_refactor_regression.py
├── test_call_arity.py
├── test_label_join_integrity.py
├── test_value_change_regimes.py
└── test_readme_freshness.py

data/
├── raw/        # gitignored
└── processed/  # gitignored
```

## 環境設定

需要 Python >= 3.11；本地驗證環境使用 Python 3.13 與 [uv](https://docs.astral.sh/uv/)。

Python version note: `pyproject.toml` requires `>=3.11`; Python 3.13 is the
local verified environment, not the minimum requirement.

Tracked code tree (`git ls-files src scripts tests`, summarized):

```text
scripts/
  diagnose_bdg2_timezone_alignment.py
  diagnose_phaseE_step3_smoke_attribution.py
  explore_bdg2.py
  phaseE_transfer.py
  run_m3_3_budslab.py
  run_m3_4_ensemble.py
  run_m3_5_postprocessing.py
  run_m3_50_50_ensemble.py
  run_m3_split_causality.py
  run_m4_3_timestamp_value_change.py
  run_m5_phaseC_tabpfn_spike.py
  run_m5_phaseD_foundation_vs_gbdt.py
  run_phaseE_step3_bdg2_transfer_smoke.py
  run_phaseE_step4a_bdg2_transfer.py
  run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py
  run_phaseE_step4c_pooled_powered_fallback.py

src/lead/
  __init__.py
  bdg2.py
  data.py
  evaluate.py
  features.py
  io.py
  sample.py
  split.py

tests/
  golden_metrics.json
  test_bdg2_loader.py
  test_call_arity.py
  test_label_join_integrity.py
  test_m5_phaseD_comparison.py
  test_m5_tabpfn_spike.py
  test_phaseE_step4_transfer.py
  test_public_api.py
  test_readme_freshness.py
  test_refactor_regression.py
  test_sampling_semantics.py
  test_split_helpers.py
  test_time_and_postprocessing_semantics.py
  test_value_change_regimes.py
```

```bash
git clone https://github.com/kuokuant-oss/lead-reproduction.git
cd lead-reproduction
uv sync
```

Tracked M3 scripts 會從 `lead` package 匯入共用 helper。若 shell 無法解析 `from lead import ...`，可使用：

```bash
uv pip install -e .
```

安裝 pre-commit hooks：

```bash
uv run pre-commit install
```

## 資料

資料不放入 repo。下載後放在 `data/raw/` 或 `data/raw/m3/`。

### M2 LEAD subset

來源：https://www.kaggle.com/competitions/energy-anomaly-detection/data

需要檔案：

- `data/raw/train_features.csv`
- `data/raw/test_features.csv`
- `data/raw/sample_submission.csv`

### M3 Full ASHRAE GEPIII

來源：https://www.kaggle.com/competitions/ashrae-energy-prediction/data

需要檔案：

- `data/raw/m3/train.csv`
- `data/raw/m3/bad_meter_readings.csv`
- `data/raw/m3/building_metadata.csv`
- `data/raw/m3/weather_train.csv`

Anomaly labels 來自 buds-lab `bad_meter_readings.zip`。

## 執行

M2 notebook pipeline：

```bash
uv run jupyter notebook notebooks/05-m2-integration.ipynb
```

M3 scripts：

```bash
uv run python scripts/run_m3_3_budslab.py
uv run python scripts/run_m3_4_ensemble.py
uv run python scripts/run_m3_5_postprocessing.py --allow-null
uv run python scripts/run_m3_50_50_ensemble.py
```

M4 regression fixtures：

```bash
uv run python -m unittest tests.test_refactor_regression
uv run python -m unittest tests.test_call_arity
uv run python -m unittest tests.test_readme_freshness
```

Golden regression values are tracked in [tests/golden_metrics.json](./tests/golden_metrics.json).

## 方法紀律

本復現遵守 one-shot inference：不做 Kaggle leaderboard probing，不用反覆提交測試集結果調參。設計決策記錄在 `docs/adr/`，未決問題記錄在 `docs/reference/unknowns.md`，歷史 handoff 記錄在 `docs/handoffs/`。

完整工作方法見 [docs/reference/workflow.md](./docs/reference/workflow.md)；每個 slice commit 前需套用 [docs/reference/change-checklist.md](./docs/reference/change-checklist.md)。
