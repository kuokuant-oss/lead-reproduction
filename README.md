# LEAD Reproduction

Reproducing the 1st-place solution of the [LEAD energy anomaly detection competition](https://www.kaggle.com/competitions/energy-anomaly-detection/overview), then extending to the full ASHRAE GEPIII dataset.

## Background

+ **Paper**: Fu et al. 2022, ["Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition"](https://dl.acm.org/doi/abs/10.1145/3563357.3566147), BuildSys '22
+ **Original solution**: https://github.com/buds-lab/LEAD-1st-solution
+ **GEPIII reference**: https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis

## Status

| Milestone | Scope | Status | Result |
|---|---|---|---|
| **M2** | LEAD reproduction (406 buildings) | ✅ **Closed** | Kaggle Private **0.98616** (vs 原作者 0.98661, gap 0.05%) |
| **M3** | Full ASHRAE GEPIII (1,449 buildings, raw FE) | 🚧 **In progress** | M3.1 ✅ + M3.2 ✅ (val AUC 0.9920); M3.3-M3.5 pending |

See [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones) for issue-level progress.

## Reports

+ **M2 Reproduction Report**: [docs/reproduction-report.md](./docs/reproduction-report.md) (~5500 字, 完整復現紀錄)
+ **M3 Progress Report**: [docs/m3-report.md](./docs/m3-report.md) (~2500 字, 進行中)
+ **Workflow**: [docs/workflow.md](./docs/workflow.md) (文件生態系 + Stage-gate + verification 紀律)

## Project structure

```
docs/
├── reproduction-report.md   # M2 final report
├── m3-report.md             # M3 progress report (in progress)
├── workflow.md              # Working methodology
├── m2-plan.md               # M2 milestone plan (closed)
├── m3-plan.md               # M3 milestone plan (M3.3+ pending)
├── unknowns.md              # 17 unknowns register (paper undocumented details)
├── adr/                     # 6 architecture decision records
└── handoffs/                # Cross-session context (per milestone)
notebooks/
├── 01-m2-baseline-pipeline.ipynb     # M2.1
├── 02-m2-clusterno.ipynb             # M2.2.a (ClusterNo)
├── 03-m2-value-change.ipynb          # M2.2.b
├── 04-m2-savgol-dayofyear.ipynb      # M2.2.c + M2.2.d
├── 05-m2-integration.ipynb           # M2.2.e → M2.5 (M2 main notebook, 34 cells)
└── 06-m3-baseline.ipynb              # M3.1 + M3.2 (in progress)
data/
├── raw/                     # Downloaded data (gitignored)
└── processed/               # Generated outputs (gitignored)
```

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) for dependency management, and Git Bash on Windows (or any POSIX shell).

```bash
git clone https://github.com/kuokuant-oss/lead-reproduction.git
cd lead-reproduction
uv sync
```

Pre-commit hooks (ruff, markdownlint, large-file-check 500KB):

```bash
uv run pre-commit install
```

## Data

Data is not committed to this repo. Download from Kaggle and place in `data/raw/`.

### M2 (LEAD subset)

https://www.kaggle.com/competitions/energy-anomaly-detection/data

Place under `data/raw/`:

+ `train_features.csv` (LEAD preprocessed train)
+ `test_features.csv` (LEAD preprocessed test)
+ `sample_submission.csv`

### M3 (Full ASHRAE GEPIII)

+ **Raw data**: https://www.kaggle.com/competitions/ashrae-energy-prediction/data (M3 只需 train.csv, building_metadata.csv, weather_train.csv, 不需 test)
+ **Feature engineering reference**: [02_preprocess_data.py](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/scripts/02_preprocess_data.py)
+ **Anomaly labels**: [bad_meter_readings.zip](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/input/bad_meter_readings.zip)

Place under `data/raw/m3/`:

+ `train.csv` (648M, 20.2M rows, 1,449 buildings)
+ `bad_meter_readings.csv` (39M, positional row-aligned with train.csv)
+ `building_metadata.csv`
+ `weather_train.csv`

## Running

### M2 main pipeline

```bash
uv run jupyter notebook notebooks/05-m2-integration.ipynb
```

Outputs to `data/processed/submission.csv` (1,800,567 rows).

### M3 in-progress pipeline

```bash
uv run jupyter notebook notebooks/06-m3-baseline.ipynb
```

Cells 1-11: M3.1 baseline. Cells 12-15: M3.2 value-change. Cell 16: leakage check.

## Methodology

This reproduction follows a strict **one-shot inference** discipline — no Kaggle leaderboard probing. All design decisions are documented in `docs/adr/`, `docs/unknowns.md`, and `docs/handoffs/` before submission.

Final M2 result (Kaggle Private 0.98616) was achieved on the first submission after 6 working days of cumulative paper + code analysis, with gap 0.05% to 原作者 0.98661 (within noise floor ±0.0005).

See [docs/workflow.md](./docs/workflow.md) for full methodology.

## License

MIT (TBD — confirm before publishing externally)
