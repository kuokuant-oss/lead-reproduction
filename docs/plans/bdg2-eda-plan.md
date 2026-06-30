# BDG2 EDA Plan

**Stage**: Phase E pre-modeling EDA
**Status**: Completed (2026-06-30); stopped for review before any modeling or transfer follow-up
**GitHub Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)

## Purpose

對 BDG2 做任何完整資料集建模前，刻畫 BDG2 資料集本身，並定量描述它與 GEPIII 的分布差異（OOD），解釋 Phase E Step 4 的 OOD-leaning finding。

本 slice 是 read-only EDA：無建模、無 score、無偽造標籤、無 readiness/transfer 宣稱。

## Read Sources

嚴格區分「BDG2 側」與「BDG2-vs-GEPIII 對比側」，避免把唯讀對比誤寫成重建 GEPIII pipeline。

+ BDG2 側：只讀 `data/raw/bdg2`。
+ GEPIII 對比側：只讀既有、凍結、可追溯的 GEPIII 來源：`load_m3_frame` 與/或 `data/raw/m3/building_metadata.csv`。
+ GEPIII 對比只屬唯讀取數；不重建 GEPIII pipeline、不動 `src/lead`、不新增建模或 scoring、不動 M3 numeric line。

## 1. Per-Meter Structure EDA

Scope: 8 個 raw + 8 個 cleaned meter 檔案族，依 meter name 配對。

報告中清楚區分：

+ meter type / meter name。
+ raw vs cleaned 版本。
+ building 是否有該 meter。
+ 有 meter 但該 timestamp 缺值。
+ cleaned 相對 raw 被補、被刪、被平滑或 coverage 改變的程度。

## 2. Missingness Decomposition

不得只報 overall NaN rate；Phase E Step 4 已指出 uplift 不是缺值單獨解釋。

+ building-level meter availability：哪些 building 根本沒有該 meter（整欄全 NaN）。
+ timestamp coverage：有該 meter 的 building 覆蓋 17,544 小時範圍的比例。
+ observation missingness：應有觀測中缺了多少。
+ cleaned-vs-raw delta：cleaned 對缺值、零值、負值、常數段的影響。

## 3. Structural Data-Quality Indicators

使用中性詞彙；禁用 anomaly/fault/failure/abnormal rate。

+ zero-reading share。
+ negative-reading share（per meter 描述性；某些 meter 如 solar 淨計量負值可能合法，不預設為壞）。
+ constant-run / flatline share。
+ suspicious structural patterns。
+ data-quality indicators。

## 4. Temporal Coverage And Profiles

+ 各 meter 在 2016-2017 的覆蓋與缺洞型態。
+ 代表性 meter 的日內 profile（按 hour 平均）。
+ 代表性 meter 的季節 profile（按 month 平均）。

## 5. Metadata Distribution

Summarize:

+ `primaryspaceusage` / `primary_use`。
+ `sqft` / `square_feet` and `sqm`。
+ `yearbuilt` / `year_built`。
+ `numberoffloors` / `floor_count`。
+ `site_id` and `timezone`。
+ 每棟 meter 覆蓋。

## 6. BDG2-Only Vs GEPIII-Overlap

這是報告主線，不放附錄。使用 `is_gepiii_overlap`。

Compare 187 棟 BDG2-only vs 1,449 棟 GEPIII-overlap buildings on:

+ site 分布。
+ primary_use 組成。
+ square_feet。
+ meter coverage。
+ missingness / flatline / zero。
+ meter_reading magnitude / seasonality。

並從資料端重現 Phase E Step 4 的稀疏：

+ 有 chilledwater 的 BDG2-only 建築有幾棟。
+ 其中 sufficient_obs 建築有幾棟。
+ 解釋「為何只有約 3 棟」這個 underpowered 根因：是少有建築有該 meter，還是有但 observation missingness 太高。

## 7. BDG2 Vs GEPIII OOD Quantification

結論不靠肉眼看圖。

+ primary_use mapping coverage + unseen/unmapped rate（BDG2 類別不在 GEPIII 的比例）。
+ square_feet median / IQR / extreme tails。
+ meter_reading median / IQR / zero share / negative share。
+ coverage days / hourly completeness。
+ BDG2-only vs overlap 差異表。
+ BDG2 vs GEPIII 差異表。
+ 招牌 OOD 標量：對關鍵特徵（square_feet、meter_reading、primary_use coverage）各給一個分布距離標量（KS statistic 或 PSI）於 BDG2-only vs GEPIII 之間，讓「多 OOD」是一個可比的數。
+ ECDF / histogram overlay 只作輔助；結論落在定量表。

## 8. Deliverables

+ `docs/reports/bdg2-eda.md`：BDG2-only-vs-overlap 與 OOD 放主線。
+ `docs/assets/bdg2-eda/*.png`：每張圖小於 500 KB，通過 large-file gate。
+ provenance JSON summary：放在 `data/processed/` shard；遵守 gitignore，必要時只提交報告摘要。
+ `docs/handoffs/<date>-bdg2-eda.md`。

## Guardrails

+ Read-only: BDG2 from `data/raw/bdg2`; GEPIII from `load_m3_frame` / `building_metadata.csv` only.
+ No modeling, no scoring, no labels, no readiness/transfer claim.
+ 禁用 anomaly/fault/failure/abnormal rate；使用 zero/negative/constant-run share 與 data-quality indicators。
+ Wide meter 檔案很大（17,544 x 最多 1,578 欄）；逐 meter 載入、只聚合摘要、必要時抽樣。
+ 不對 BDG2 字串 `building_id` 用 `split_mask`。
+ 不修改 `src/lead`、M3 numeric line、BDG2 loader contract。
+ Run tests, Ruff, markdownlint, and `pre-commit run --all-files` before commit/push.
+ Keep `data/raw`, processed shards, and `.scratch` ignored; keep figures under the size limit.
