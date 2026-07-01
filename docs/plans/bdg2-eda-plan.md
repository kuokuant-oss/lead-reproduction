# BDG2 EDA Plan

**Stage**: Phase E pre-modeling EDA
**Status**: Completed (2026-06-30)；已在任何 modeling 或 transfer follow-up 前停下供 review
**GitHub Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)

**完成註記**: 這個 slice 產出 EDA report、小型 figures、gitignored provenance JSON shard 與 handoff。不產出 model、score、label、supervised BDG2 metric 或 transfer-readiness claim。

**2026-07-01 narrative note**: ADR 0025/0026 之後，這份 EDA 是 M6 的資料面 scope check。它說明哪些 buildings/meters 可以進入 GEPIII-overlap supervised evaluation，哪些 BDG2-only/2017/other-meter rows 必須留在 secondary pseudo-label 或 review branch。

## Purpose

本 EDA 的目標是刻畫 BDG2 的資料結構、meter coverage、missingness、raw-vs-cleaned delta，以及 BDG2-vs-GEPIII 的 reference distribution distance。它不把 zero/flatline/missingness 解讀成 anomaly label，也不宣稱 transfer readiness。

這份 plan 的產物支援 M6 scope 定義：先知道哪些 row 有 GEPIII overlap 與可橋接 label，哪些 row 只能當 secondary review context。

Interpretation rule：所有 EDA outputs 都是 descriptive data-quality 或 distribution-context evidence。Modeling、scoring、label bridging、supervised metrics 都從後續 M6 slices 開始。

Evidence levels：

+ L1 supervised metric evidence。
+ L2 bridge integrity evidence。
+ L3 descriptive data-quality evidence。
+ L4 secondary review / pseudo-label evidence。
+ Retired historical context。

這份 EDA 產出 L3 evidence；BDG2-only chilledwater coverage 只作 L4 context。

## 讀取來源

+ BDG2 local archive: `data/raw/bdg2`。
+ GEPIII comparison data：`load_m3_frame(verbose=False)` 與 `data/raw/m3/building_metadata.csv`。
+ GEPIII comparison 只作 distribution context；不重跑 model、不 score、不改 M3 numeric line。

## 1. Per-Meter Structure EDA

Scope：8 個 raw meter files 與 8 個 cleaned meter files。

報告：

+ meter type / meter name；
+ raw vs cleaned file availability；
+ building 是否有該 meter；
+ per-meter timestamp coverage；
+ cleaned 相對 raw 的 cells removed、cells newly present、observed value changes、coverage change。

## 2. Missingness Decomposition

不要只回報 overall NaN rate。要把 missingness 拆成：

+ building-level meter availability：building 是否有該 meter column；
+ timestamp coverage：building/meter 是否覆蓋 17,544 hourly timestamps；
+ observation missingness：有 column 且有 timestamp 後，cell 是否 missing；
+ cleaned-vs-raw delta：cleaned 是否額外移除 observations。

## 3. Structural Data-Quality Indicators

使用中性資料品質指標，不稱為 fault/anomaly rate：

+ zero-reading share；
+ negative-reading share；
+ constant-run / flatline share；
+ suspicious structural patterns；
+ data-quality indicators。

Flatline rule 必須明確記錄 run length、zero handling、missing handling、equality tolerance 與 aggregation denominator。

## 4. Temporal Coverage And Profiles

+ 每個 meter 的 2016-2017 coverage；
+ representative hourly profile；
+ representative monthly profile。

這些 profiles 只作 descriptive context，不是 model features、scores 或 readiness evidence。

## 5. Metadata Distribution

Summarize:

+ `primaryspaceusage` / `primary_use`；
+ `sqft` / `square_feet` and `sqm`；
+ `yearbuilt` / `year_built`；
+ `numberoffloors` / `floor_count`；
+ `site_id` and `timezone`；
+ meter availability by metadata。

Metadata summary 用於 scope 與 distribution context，不產生 headline model conclusion。

## 6. Report Narrative Order

`scripts/run_bdg2_eda.py` 產出的 report sections 依序為：

+ `Dataset Provenance And Cleaning`：連到 Miller et al. 2020 BDG2 data descriptor，說明 raw/cleaned release cleaning rules。
+ `BDG2 Data-Quality Inventory`：per-meter structure、missingness decomposition、zero / flatline、cleaned-vs-raw delta。
+ `BDG2-Only Sufficiency`：說明 BDG2-only rows 的 coverage 與 sufficient-observation context，但不把它當 active M6 supervised metric。
+ `GEPIII Comparison As Context`：meter coverage 與 reference distribution distances；這是 diagnostic lens，不是 transfer-readiness claim。

Tracked paper reference: `docs/reference/papers/bdg2-miller-2020.md`。Local PDF `docs/reference/papers/bdg2-miller-2020.pdf` 超過 500 KB large-file gate，維持 gitignored。

## 7. BDG2-Only Vs GEPIII-Overlap

使用 `is_gepiii_overlap` 把 BDG2-only 與 GEPIII-overlap buildings 分開比較：

+ site distribution；
+ primary_use distribution；
+ square_feet；
+ meter coverage；
+ missingness / flatline / zero；
+ meter_reading magnitude / seasonality。

BDG2-only chilledwater 的 coverage 可作 secondary branch context，但不是 M6 supervised headline。

## 8. Reference Distribution Distances

報告：

+ primary_use mapping coverage + unseen/unmapped rate；
+ square_feet median / IQR / extreme tails；
+ meter_reading median / IQR / zero share / negative share；
+ coverage days / hourly completeness；
+ BDG2-only vs overlap distance；
+ BDG2 vs GEPIII distance；
+ KS statistic 與 PSI，並標明 sampling basis。

`meter_reading` distance 必須附 caveat：BDG2 raw cells 與 GEPIII Kaggle-release cells 的差距包含 Miller et al. 2020 描述的 release-level differences、Kaggle unit-conversion errors、UTC-vs-local weather timestamps、meter mix 與 building heterogeneity。

## 9. 交付物

+ `docs/reports/bdg2-eda.md`：BDG2 data-quality inventory、BDG2-only sufficiency、GEPIII comparison context、reference distribution distances。
+ `docs/assets/bdg2-eda/*.png`：每張圖維持 500 KB 以下。
+ provenance JSON summary：寫到 `data/processed/` gitignored shard。
+ `docs/handoffs/<date>-bdg2-eda.md`。

## Guardrails

+ Read-only：BDG2 只讀 `data/raw/bdg2`；GEPIII 只讀 `load_m3_frame` / `building_metadata.csv`。
+ No modeling、no scoring、no labels、no readiness/transfer claim。
+ 不使用 anomaly/fault/failure/abnormal rate；只使用 zero/negative/constant-run share 等 data-quality indicators。
+ Wide meter tables 只作 EDA 讀取，不改 raw data。
+ 不改 `src/lead`、M3 numeric line、BDG2 loader contract。
+ Commit/push 前跑 tests、Ruff、markdownlint、`pre-commit run --all-files`。
+ `data/raw`、processed shards、`.scratch` 維持 ignored；figures 維持在 size limit 以下。
