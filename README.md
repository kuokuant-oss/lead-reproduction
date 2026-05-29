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
| **M3** | Full ASHRAE GEPIII (1,449 buildings, 從 raw 做 FE) | 🚧 進行中 | M3.1 ✅ + M3.2 ✅ (val AUC 0.9920, 4 個 sanity check 通過); M3.3-M3.5 待續 |

issue-level 進度見 [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones)。

## 報告

- **M2 復現報告**: [docs/reproduction-report.md](./docs/reproduction-report.md) (~5500 字, 完整復現紀錄)
- **M3 進度報告**: [docs/m3-report.md](./docs/m3-report.md) (~2500 字, 進行中)
- **工作方法**: [docs/workflow.md](./docs/workflow.md) (文件生態系 + Stage-gate + verification 紀律)

## 專案結構

```
docs/
├── reproduction-report.md   # M2 完整報告
├── m3-report.md             # M3 進度報告 (進行中)
├── workflow.md              # 工作方法
├── m2-plan.md               # M2 milestone plan (已關閉)
├── m3-plan.md               # M3 milestone plan (M3.3+ 待做)
├── unknowns.md              # 17 unknowns register (M1 產出, paper 未說清的地方)
├── adr/                     # 6 個架構決策紀錄 (M1 產出)
└── handoffs/                # 跨 session context (每個 milestone 結尾寫一份)
notebooks/
├── 01-m2-baseline-pipeline.ipynb     # M2.1
├── 02-m2-clusterno.ipynb             # M2.2.a (ClusterNo)
├── 03-m2-value-change.ipynb          # M2.2.b
├── 04-m2-savgol-dayofyear.ipynb      # M2.2.c + M2.2.d
├── 05-m2-integration.ipynb           # M2.2.e → M2.5 (M2 主 notebook, 34 cells)
└── 06-m3-baseline.ipynb              # M3.1 + M3.2 (進行中)
data/
├── raw/                     # 下載的資料 (gitignored)
└── processed/               # 產生的輸出 (gitignored)
```

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

### M3 進行中 pipeline

```bash
uv run jupyter notebook notebooks/06-m3-baseline.ipynb
```

Cells 1-11: M3.1 baseline。Cells 12-15: M3.2 value-change。Cells 16-20: 4 個 sanity check + 完整 metrics。

## 方法論

本復現遵守嚴格的 **one-shot inference** 紀律 — 不做 Kaggle leaderboard probing。所有設計決策在 `docs/adr/`、`docs/unknowns.md`、`docs/handoffs/` 紀錄完才提交。

M2 最終結果 (Kaggle Private 0.98616) 是 6 個工作天累積 paper + code 分析後第一次提交達成, gap 0.05% to 原作者 0.98661 (在 noise floor ±0.0005 內)。

完整方法論見 [docs/workflow.md](./docs/workflow.md)。
