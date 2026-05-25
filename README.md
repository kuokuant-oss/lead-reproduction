# LEAD Reproduction

Reproducing the 1st-place solution of the [LEAD energy anomaly detection competition](https://www.kaggle.com/competitions/energy-anomaly-detection/overview), then extending to the full ASHRAE dataset.

## Background
- Paper: https://dl.acm.org/doi/abs/10.1145/3563357.3566147
- Original solution: https://github.com/buds-lab/LEAD-1st-solution
- Full dataset: https://www.kaggle.com/competitions/ashrae-energy-prediction/data

## Status
🚧 Work in progress — see [milestones](https://github.com/kuokuant-oss/lead-reproduction/milestones).

## Project structure
See [CONTEXT.md](./CONTEXT.md) for full project context and conventions.

## Setup
TODO — will be filled in during M2.

## Data
Data is not committed to this repo. Download from Kaggle:

**LEAD dataset (M2)**
- https://www.kaggle.com/competitions/energy-anomaly-detection/data

**Full dataset (M3)**
- Raw data (Kaggle GEPIII): https://www.kaggle.com/competitions/ashrae-energy-prediction/data
  (M3 只需 train data,不需 test)
- Feature engineering reference:
  https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/scripts/02_preprocess_data.py
- Anomaly label source:
  https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/input/bad_meter_readings.zip

Place files in \`data/raw/\`.
