# FDD on BDG2

本專案復現 Fu et al. 2022 BuildSys 論文
["Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition"](https://dl.acm.org/doi/abs/10.1145/3563357.3566147)，並把工作從 LEAD competition subset 延伸到 ASHRAE GEPIII raw dataset。復現完成後，本專案進一步延伸到 anomaly detection 的資料集整理、FDD 評估設計，以及 GBDT、ensemble、TabPFN 等模型表現比較。

參考來源：

- 論文：Fu et al. 2022, BuildSys '22
- 原始解法：[https://github.com/buds-lab/LEAD-1st-solution](https://github.com/buds-lab/LEAD-1st-solution)
- GEPIII reference：[https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis)

## 目前狀態

| Milestone | 範圍 | 狀態 | 主要結果 |
| --- | --- | --- | --- |
| **M1** | 閱讀 paper 與 buds-lab code，建立 unknowns register 與 ADR framework | Closed | 17 個 unknowns、ADR 0001-0006、169-feature 組成釐清 |
| **M2** | LEAD competition subset reproduction | Closed | Kaggle Private AUC `0.98616`，與原始解法 `0.98661` 的差距為 `0.05%` |
| **M3** | Full ASHRAE GEPIII reproduction | Complete | M3.4 ensemble AUC `0.9934`；PI 50/50 ensemble offline `0.9918` / causal `0.9913`；value-change canonical regime 為 `timestamp_merge`；post-processing 為 null result |
| **M4** | Importable pipeline foundation | M4.0-M4.5 complete | `src/lead` public API frozen; M3.2/M3.4 regression gates pass; M4.2-M4.5 closed |
| **M5** | GEPIII FDD model comparison | Complete | TabPFN 優勢集中在低標註量 PR-AUC；full-feature in-domain 六模型接近；site-transfer 與 M5.x per-unit 切分粒度下皆由 tree family 領先 |

Issue-level 進度見 GitHub [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones)。

## 主要文件

- **M2 復現報告**：[docs/reports/reproduction-report.md](./docs/reports/reproduction-report.md)
- **M3 完成報告**：[docs/reports/m3-report.md](./docs/reports/m3-report.md)
- **M4 評估報告**：[docs/reports/m4-evaluation-report.md](./docs/reports/m4-evaluation-report.md)
- **M5 GEPIII 模型比較報告**：[docs/reports/m5-foundation-vs-gbdt.md](./docs/reports/m5-foundation-vs-gbdt.md)
- **M5.1 TabPFN vs tree models 深入比較**：[docs/reports/m5-1-deep-comparison.md](./docs/reports/m5-1-deep-comparison.md)
- **M5.x 切分粒度比較**：[docs/reports/m5x-partition-granularity.md](./docs/reports/m5x-partition-granularity.md)
- **BDG2 data descriptor reference**：[docs/reference/papers/bdg2-miller-2020.md](./docs/reference/papers/bdg2-miller-2020.md)

## Milestone 摘要

### M1 理解與決策框架

M1 不訓練模型，目標是把論文與原始碼中的關鍵決策變成可追蹤文件。

- `docs/reference/unknowns.md`：17 個 paper 或 code 未說清楚的地方。
- `docs/adr/`：目前共有 27 份 ADR；M1 產出 ADR 0001-0006。
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

- M3.2 LightGBM offline AUC：`0.9925`
- M3.3 buds-lab alignment AUC：`0.9928`，未帶來穩定 lift。
- M3.4 4-model ensemble AUC：`0.9934`
- M3.5 hard-rule post-processing delta：`-0.000054`，判定為 null result。
- PI 50/50 ensemble offline AUC：`0.9918`
- PI 50/50 ensemble causal AUC：`0.9913`

上述 value-change 數字皆使用 `timestamp_merge`，也就是 buds-lab 原作 timestamp join 的忠實版；`row_offset` / `row_offset_meter_aware` 保留為歷史 ablation。詳細結果見 [docs/reports/m3-report.md](./docs/reports/m3-report.md)。Machine-readable provenance 放在 `docs/metrics/`。

### M4 Importable Pipeline Foundation

M4 把已驗證的 M3 reproduction pipeline 整理成可匯入的 `src/lead` package，並用 regression gates 鎖住 M3 numeric line。

M4.0-M4.5 complete。

主要結果：

- `src/lead` public API 凍結。
- M3.2 LightGBM golden AUC re-baseline 為 `0.9925`；M3.4 ensemble golden AUC re-baseline 為 `0.9934`。
- Label alignment 已由 ADR 0010 Accepted 鎖定；timestamp/value-change regime 已由 ADR 0011 Accepted 鎖定，canonical default 為 `timestamp_merge`。
- Sampling/scaler semantics 已由 ADR 0016 Accepted 鎖定。
- M5 可直接重用 data、feature、split、sample、evaluation helpers。

詳細結果見 [docs/reports/m4-evaluation-report.md](./docs/reports/m4-evaluation-report.md)。

### M5 GEPIII 模型比較

M5 已完成 GEPIII 上的 FDD 模型比較。比較線使用 `timestamp_merge`
features、50% training / 50% testing split，並列出 LightGBM、XGBoost、CatBoost、
HistGBT、Ensemble 與 TabPFN。

主要結果：

- TabPFN 的主要優勢出現在低標註量 PR-AUC；support 200 與 500 時 test PR-AUC 最高。
- Full features 的 in-domain 50/50 test split 上，六個模型分數接近，無明確模型排序。
- Raw 17 features 下，tree models 的 test PR-AUC 較高；TabPFN 在 threshold `0.5` 下 TN 最高、FP 最少，且 FP+FN 總錯誤數最低。
- Site-transfer test PR-AUC 由 tree family 較強。
- M5.1 深入比較（TabPFN-3 local vs tree models）：小樣本（support `100`-`2,000`）與固定 fit rows `500` 的中低維設定下，TabPFN test PR-AUC 最高（`137` features 達 `0.8520`），run-to-run 穩定性佳；但在完整 fit budget 與 tuned trees 充分調參後，tuned ensemble `0.9109` 反超 TabPFN `0.9024`。結論為 TabPFN-3 local 是小樣本 tabular anomaly detection 的強基準。
- M5.x per-unit 切分粒度（C1 最細到 C3 最粗）下，tree ensemble 在四個粒度的 pooled PR-AUC 皆領先 TabPFN；C2（`site_id, meter`）對 TabPFN 最友善（PR 差最佳 tree 約 `0.03`、coverage `99.75%`），最細的 C1 因 fallback rate `84.75%`、coverage 僅 `15.25%` 而不具實用價值。

詳細結果見 [docs/reports/m5-foundation-vs-gbdt.md](./docs/reports/m5-foundation-vs-gbdt.md)，TabPFN vs tree models 深入比較見 [docs/reports/m5-1-deep-comparison.md](./docs/reports/m5-1-deep-comparison.md)，per-unit 切分粒度見 [docs/reports/m5x-partition-granularity.md](./docs/reports/m5x-partition-granularity.md)。

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

`add_value_change_features(df, shifts, value_change_regime=...)` supports `row_offset`, `row_offset_meter_aware`, and `timestamp_merge`. `timestamp_merge` is the M3/M4/M5 canonical default; the row-offset regimes remain available for historical ablation and transfer-specific sensitivity checks.

## 專案結構

```text
docs/
├── plans/
│   ├── m1-plan.md
│   ├── m2-plan.md
│   ├── m3-plan.md
│   ├── m4-plan.md
│   ├── m5-plan.md
│   ├── phaseE-fdd-roadmap.md
│   └── bdg2-supervised-fdd-plan.md
├── reports/
│   ├── reproduction-report.md
│   ├── m3-report.md
│   ├── m4-evaluation-report.md
│   ├── m5-foundation-vs-gbdt.md
│   ├── m5-1-deep-comparison.md
│   ├── m5x-partition-granularity.md
│   ├── bdg2-eda.md
│   ├── bdg2-data-reality.md
│   ├── phaseE-step4-bdg2-transfer.md
│   └── assets/
│       └── m5/
│           └── confusion matrix PNGs
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
│   └── 0001-0027 decision records
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
├── diagnose_bdg2_timezone_alignment.py
├── diagnose_phaseE_step3_smoke_attribution.py
├── explore_bdg2.py
├── phaseE_transfer.py
├── run_bdg2_eda.py
├── run_gate_label_join_integrity.py
├── run_inv1_meter_aware_impact.py
├── run_inv3_scarcity_unique_support.py
├── run_inv4_shuffle_ablation.py
├── run_inv5_time_holdout.py
├── run_inv6_train_val_gap.py
├── run_inv7_per_building_distribution.py
├── run_inv8_sampling_fragility.py
├── run_m3_2_baseline.py
├── run_m3_3_budslab.py
├── run_m3_4_ensemble.py
├── run_m3_5_postprocessing.py
├── run_m3_50_50_ensemble.py
├── run_m3_split_causality.py
├── run_m4_3_timestamp_value_change.py
├── run_m5_phaseC_tabpfn_spike.py
├── run_m5_phaseD_deep_comparison.py
├── run_m5_phaseD_foundation_vs_gbdt.py
├── run_m5x_partition_granularity.py
├── run_m6_phaseD_50_50_full_models.py
├── run_phaseE_step3_bdg2_transfer_smoke.py
├── run_phaseE_step4a_bdg2_transfer.py
├── run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py
└── run_phaseE_step4c_pooled_powered_fallback.py

src/lead/
├── __init__.py
├── bdg2.py
├── data.py
├── features.py
├── split.py
├── sample.py
├── evaluate.py
└── io.py

tests/
├── golden_metrics.json
├── test_bdg2_loader.py
├── test_call_arity.py
├── test_label_join_integrity.py
├── test_m5_phaseD_comparison.py
├── test_m5_tabpfn_spike.py
├── test_m5_timestamp_merge_regime.py
├── test_m5x_partition_granularity.py
├── test_phaseE_step4_transfer.py
├── test_public_api.py
├── test_readme_freshness.py
├── test_refactor_regression.py
├── test_report_metric_consistency.py
├── test_sampling_semantics.py
├── test_split_helpers.py
├── test_time_and_postprocessing_semantics.py
└── test_value_change_regimes.py

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
  run_bdg2_eda.py
  run_gate_label_join_integrity.py
  run_inv1_meter_aware_impact.py
  run_inv3_scarcity_unique_support.py
  run_inv4_shuffle_ablation.py
  run_inv5_time_holdout.py
  run_inv6_train_val_gap.py
  run_inv7_per_building_distribution.py
  run_inv8_sampling_fragility.py
  run_m3_2_baseline.py
  run_m3_3_budslab.py
  run_m3_4_ensemble.py
  run_m3_5_postprocessing.py
  run_m3_50_50_ensemble.py
  run_m3_split_causality.py
  run_m4_3_timestamp_value_change.py
  run_m5_phaseC_tabpfn_spike.py
  run_m5_phaseD_deep_comparison.py
  run_m5_phaseD_foundation_vs_gbdt.py
  run_m5x_partition_granularity.py
  run_m6_phaseD_50_50_full_models.py
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
  test_m5_timestamp_merge_regime.py
  test_m5x_partition_granularity.py
  test_phaseE_step4_transfer.py
  test_public_api.py
  test_readme_freshness.py
  test_refactor_regression.py
  test_report_metric_consistency.py
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
uv run python scripts/run_m3_2_baseline.py
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

M5 GEPIII 模型比較：

```bash
uv run python scripts/run_m5_phaseC_tabpfn_spike.py
uv run python scripts/run_m5_phaseD_foundation_vs_gbdt.py --value-change-regime timestamp_merge --out data/processed/m6_phaseD_timestamp_merge_multiseed.json
uv run python scripts/run_m6_phaseD_50_50_full_models.py --out data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json
uv run python scripts/run_m5_phaseD_deep_comparison.py --out data/processed/m5_phaseD_deep_comparison.json
uv run python scripts/run_m5x_partition_granularity.py --out data/processed/m5x_partition_granularity.json
```

Golden regression values 記錄在 [tests/golden_metrics.json](./tests/golden_metrics.json)。

## 方法紀律

本復現遵守 one-shot inference：不做 Kaggle leaderboard probing，不用反覆提交測試集結果調參。設計決策記錄在 `docs/adr/`，未決問題記錄在 `docs/reference/unknowns.md`，歷史 handoff 記錄在 `docs/handoffs/`。

完整工作方法見 [docs/reference/workflow.md](./docs/reference/workflow.md)；每個 slice commit 前需套用 [docs/reference/change-checklist.md](./docs/reference/change-checklist.md)。
