# LEAD Reproduction

復現 [LEAD energy anomaly detection competition](https://www.kaggle.com/competitions/energy-anomaly-detection/overview) 第一名解法, 並延伸到完整 ASHRAE GEPIII dataset。

## 背景

- **論文**: Fu et al. 2022, ["Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition"](https://dl.acm.org/doi/abs/10.1145/3563357.3566147), BuildSys '22
- **原作者 code**: https://github.com/buds-lab/LEAD-1st-solution
- **GEPIII reference**: https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis

## 進度

| Milestone | 範圍 | 狀態 | 結果 |
|---|---|---|---|
| **M1** | Paper + buds-lab code 理解, unknowns register, ADR framework | ✅ Closed | 17 unknowns 整理, 6 ADRs, 169 features 完整解碼 |
| **M2** | LEAD reproduction (406 buildings) | ✅ Closed | Kaggle Private **0.98616** (vs 原作者 0.98661, gap 0.05%) |
| **M3** | Full ASHRAE GEPIII (1,449 buildings, 從 raw 做 FE) | ✅ Complete | M3.1-M3.5 done; PI 50/50 ensemble offline **0.9921** / causal **0.9911**; post-processing null result; 6 issues closed |
| **M4** | Importable pipeline foundation | M4.1 complete | Extracted `src/lead` package from notebooks/scripts; behavior-preserving M3.2/M3.4 regression gates pass; M4.2 key-aligned label join and M4.3 timestamp value-change remain Proposed |

issue-level 進度見 [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones)。

## 報告

- **M2 復現報告**: [docs/reproduction-report.md](./docs/reproduction-report.md) (~5500 字, 完整復現紀錄)
- **M3 完成報告**: [docs/m3-report.md](./docs/m3-report.md) (~2500 字, M3 complete)
- **工作方法**: [docs/workflow.md](./docs/workflow.md) (文件生態系 + Stage-gate + verification 紀律)

## 專案結構

```
docs/
├── reproduction-report.md   # M2 完整報告
├── m3-report.md             # M3 完成報告
├── workflow.md              # 工作方法
├── m2-plan.md               # M2 milestone plan (已關閉)
├── m3-plan.md               # M3 milestone plan (已關閉)
├── unknowns.md              # 17 unknowns register (M1 產出, paper 未說清的地方)
├── adr/                     # 6 個架構決策紀錄 (M1 產出)
└── handoffs/                # 跨 session context (每個 milestone 結尾寫一份)
notebooks/
├── 01-m2-baseline-pipeline.ipynb     # M2.1
├── 02-m2-clusterno.ipynb             # M2.2.a (ClusterNo)
├── 03-m2-value-change.ipynb          # M2.2.b
├── 04-m2-savgol-dayofyear.ipynb      # M2.2.c + M2.2.d
├── 05-m2-integration.ipynb           # M2.2.e → M2.5 (M2 主 notebook, 34 cells)
├── 06-m3-baseline.ipynb              # M3.1 + M3.2
├── 07-m3-split-causality.ipynb       # M3.2a PI-response split/causality check
├── 08-m3-budslab.ipynb               # M3.3 buds-lab alignment (no-lift)
├── 09-m3-ensemble.ipynb              # M3.4 4-model ensemble
└── 10-m3-postprocessing.ipynb        # M3.5 post-processing null result
scripts/
├── run_m3_3_budslab.py               # M3.3 buds-lab alignment runner
├── run_m3_4_ensemble.py              # M3.4 4-model ensemble runner
├── run_m3_5_postprocessing.py        # M3.5 post-processing + diagnostics runner
├── run_m3_50_50_ensemble.py          # PI-spec 50/50 ensemble follow-up
└── run_m3_split_causality.py         # M3.2a 80/20 vs 50/50, offline vs causal grid
src/lead/
├── data.py                  # M3 data loading, constants, current positional label assignment
├── features.py              # value-change feature generation
├── split.py                 # building-level split helpers and overlap assertions
├── sample.py                # current downsample index helper
├── evaluate.py              # AUC / precision / recall / F1 metrics
└── io.py                    # JSON provenance helper
data/
├── raw/                     # 下載的資料 (gitignored)
└── processed/               # 產生的輸出 (gitignored)
```

## M4 importable package

M4 moves the current pipeline foundation out of scattered notebook/script helper
copies and into the importable `src/lead` package so M5 can extend the work
toward FDD on BDG2. M4.1 is complete: the extraction is behavior-preserving, and
the M3.2 and M3.4 golden regression checks pass against
`tests/golden_metrics.json`.

The M4 planning and evaluation records are:

- [docs/m4-plan.md](./docs/m4-plan.md)
- [docs/m4-evaluation-report.md](./docs/m4-evaluation-report.md)

M4.2 and M4.3 are not executed yet. ADR 0010 proposes key-aligned label joins,
and ADR 0011 proposes timestamp-merge value-change semantics. Current M4.1 code
still preserves the M3 behavior: positional label assignment and row-offset
`groupby().shift()` value-change features.

`src/lead/__init__.py` exports the public API used by the tracked scripts:

- Constants and paths: `ROOT`, `M3`, `PROC`, `RANDOM_STATE`,
  `DOWNSAMPLE_SEEDS`, `MODEL_SEEDS`, `SHUFFLE_SEEDS`,
  `BASELINE_FEATURE_COLS`, `BUILDING_META_FEATURE_COLS`,
  `CYCLIC_FEATURE_COLS`, `WEATHER_LAG_BASE_COLS`, `WEATHER_WINDOWS`,
  `M3_3_EXTRA_FEATURE_COLS`, `SHIFTS`, `PAST_SHIFTS`, `FUTURE_SHIFTS`
- Data: `load_m3_frame`
- Features: `add_value_change_features`
- Splits: `split_mask`, `assert_no_building_overlap`
- Sampling: `downsample_indices`
- Evaluation: `classification_metrics`
- Provenance IO: `write_json_with_provenance`

## M1 產出 (基礎工作)

M1 不產生 notebook 或 model, 但是 M2/M3 的基礎:

- **`docs/unknowns.md`**: 17 個 paper 沒說清楚的地方 (e.g. 169 features 完整組成, downsampling seeds, Rule 2a building_id filter), 每個 unknown 隨 milestone 更新狀態
- **`docs/adr/`**: 6 個架構決策 — building-id split (0001), downsampling 50:50 (0002), value-change features 同時取差值跟比值 (0003), post-processing hard rules (0004), imputation method (0005), paper-code 不一致處理紀律 (0006)
- **`docs/workflow.md`**: 工作方法 framework, M2/M3 都沿用

M1 大約花 4 天, 累積的 docs 是 M2 設計決策的依據, 也是 M3 沿用的工作 framework。

## 環境設定

需要 Python 3.13, [uv](https://docs.astral.sh/uv/) 管理 dependencies, Git Bash on Windows (或任何 POSIX shell)。

```bash
git clone https://github.com/kuokuant-oss/lead-reproduction.git
cd lead-reproduction
uv sync
```

Tracked M3 scripts now import shared helpers from the `lead` package. From the
repo root this works after `uv sync`; if a shell cannot resolve `from lead
import ...`, either run with `PYTHONPATH=src` or install the package editable:

```bash
uv pip install -e .
```

Pre-commit hooks (ruff, markdownlint, large-file-check 500KB):

```bash
uv run pre-commit install
```

## 資料

資料不放 repo。從 Kaggle 下載放到 `data/raw/`。

### M2 (LEAD subset)

- https://www.kaggle.com/competitions/energy-anomaly-detection/data

放在 `data/raw/`:

- `train_features.csv` (LEAD preprocessed train)
- `test_features.csv` (LEAD preprocessed test)
- `sample_submission.csv`

### M3 (Full ASHRAE GEPIII)

- **Raw data**: https://www.kaggle.com/competitions/ashrae-energy-prediction/data (M3 只需 train.csv, building_metadata.csv, weather_train.csv, 不需 test)
- **Feature engineering reference**: [02_preprocess_data.py](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/scripts/02_preprocess_data.py)
- **Anomaly labels**: [bad_meter_readings.zip](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/input/bad_meter_readings.zip)

放在 `data/raw/m3/`:

- `train.csv` (648M, 20.2M rows, 1,449 buildings)
- `bad_meter_readings.csv` (39M, positional row-aligned with train.csv)
- `building_metadata.csv`
- `weather_train.csv`

## 執行

### M2 主 pipeline

```bash
uv run jupyter notebook notebooks/05-m2-integration.ipynb
```

輸出到 `data/processed/submission.csv` (1,800,567 rows)。

### M3 完成 pipeline

```bash
uv run jupyter notebook notebooks/06-m3-baseline.ipynb
```

Cells 1-11: M3.1 baseline。Cells 12-15: M3.2 value-change。Cells 16-20: 4 個 sanity check + 完整 metrics。

M3.3 buds-lab alignment:

```bash
uv run python scripts/run_m3_3_budslab.py
uv run jupyter notebook notebooks/08-m3-budslab.ipynb
```

M3.3 result: val AUC `0.9913` vs M3.2 `0.9920`; documented as no-lift/negligible.

M3.4 ensemble + M3.5 post-processing:

```bash
uv run python scripts/run_m3_4_ensemble.py
uv run jupyter notebook notebooks/09-m3-ensemble.ipynb
uv run python scripts/run_m3_5_postprocessing.py --allow-null
uv run jupyter notebook notebooks/10-m3-postprocessing.ipynb
```

PI-spec 50/50 ensemble follow-up:

```bash
uv run python scripts/run_m3_50_50_ensemble.py
```

These scripts import M3 data, feature, split, sampling, and evaluation helpers
from `src/lead`; experiment-specific orchestration remains in `scripts/`.

M3 final headline: M3.4 ensemble val AUC `0.9928`; PI 50/50 ensemble offline
`0.9921` / causal `0.9911`; post-processing closes as null result
(`combined ΔAUC -0.000054`).

Golden regression values for M4 are tracked in
[`tests/golden_metrics.json`](./tests/golden_metrics.json).

## 方法論

本復現遵守嚴格的 **one-shot inference** 紀律 — 不做 Kaggle leaderboard probing。所有設計決策在 `docs/adr/`、`docs/unknowns.md`、`docs/handoffs/` 紀錄完才提交。

M2 最終結果 (Kaggle Private 0.98616) 是 6 個工作天累積 paper + code 分析後第一次提交達成, gap 0.05% to 原作者 0.98661 (在 noise floor ±0.0005 內)。

完整方法論見 [docs/workflow.md](./docs/workflow.md)。
