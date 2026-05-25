# Project Context

## What this project is
重現一篇論文(LEAD competition 第一名解法),並擴展到完整 ASHRAE 資料集。
論文連結:https://dl.acm.org/doi/abs/10.1145/3563357.3566147
原始 solution:https://github.com/buds-lab/LEAD-1st-solution

## Milestones
- M1:理解論文與方法
- M2:重現比賽結果(LEAD 資料集,~200 meters)
- M3:擴展到完整 ASHRAE 資料集(~2000 meters)

## Glossary
(隨著專案進行持續更新)

- **LEAD**:Large-scale Energy Anomaly Detection,本次 Kaggle 比賽名稱
- **ASHRAE**:American Society of Heating, Refrigerating and A/C Engineers;原始大資料集來源
- **Anomaly label**:每筆讀數是否為異常(0/1)
- **Meter**:電表/能源表讀數
- **Feature engineering**:從原始 meter readings 衍生出模型用的特徵

## Tech stack
- Python 3.11+
- uv(package manager)
- pandas, numpy, scikit-learn, lightgbm
- jupyter notebooks for exploration
- src/lead/ for reusable code

## Folder conventions
- \`data/raw/\`:從 Kaggle 直接下載的原始檔(不進 Git)
- \`data/interim/\`:中間處理結果
- \`data/processed/\`:餵給模型的最終格式
- \`notebooks/\`:探索用,檔名前綴編號(01-, 02-, ...)
- \`src/lead/\`:穩定的可重用程式碼
- \`docs/\`:筆記、決策紀錄
- \`reports/\`:圖表、結果輸出
