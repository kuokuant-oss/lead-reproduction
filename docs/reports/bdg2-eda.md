# BDG2 EDA Report

**Date**: 2026-06-30
**Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)
**Plan**: [docs/plans/bdg2-eda-plan.md](../plans/bdg2-eda-plan.md)

## Scope 與 guardrails

這是一個 read-only、pre-modeling EDA slice。資料來源是 `data/raw/bdg2`，GEPIII comparison data 只來自 frozen GEPIII sources（`load_m3_frame` 與 `data/raw/m3/building_metadata.csv`）。本報告不建 model、不產生 score、不接 GEPIII overlap label bridge、不回報 supervised BDG2 metrics，也不宣稱 readiness/transfer。

ADR 0025/0026 之後，這份報告的用途是定義 M6 的資料面 scope：哪些 buildings/meters 可以進入 labeled overlap evaluation，哪些只能留在 secondary pseudo-label 或 review branch。報告使用中性資料品質詞彙：zero-reading share、negative-reading share、flatline share、missingness、coverage、distribution distance。

Interpretation rule：所有 EDA outputs 都是 descriptive data-quality 或 distribution-context evidence。它們不是 labels、model scores 或 supervised metrics。

Evidence levels：

+ L3 descriptive data-quality evidence：meter coverage、missingness、raw-vs-cleaned deltas、flatline/zero indicators。
+ L4 secondary review evidence：BDG2-only chilledwater coverage 與 raw/cleaned pseudo-label candidates。
+ Retired historical context：舊 Phase E chilledwater powered-gate interpretation。

## 主要發現

+ BDG2 有 1,636 buildings，但 meter availability 在各 meter 間高度不均。
+ Electricity coverage 最廣；chilledwater、steam、hotwater、gas、water、irrigation、solar 的 building coverage 明顯較窄。
+ 多個 meters 有高 zero-reading 或 flatline share，尤其是 irrigation、water、gas、hotwater。這些可能反映 Miller et al. 2020 描述的 operational off-periods，不等於 data faults。
+ Cleaned files 對每個 meter 都提高 null rate，反映 BDG2 release 自身的 outlier/zero removal rules。這是 data-quality delta，不是 label。
+ BDG2-only chilledwater 只作 secondary branch context：187 個 BDG2-only buildings 中，26 個有 chilledwater columns，但只有 3 個滿足 sufficient-observation rule（`missing_rate <= 0.50`）。
+ GEPIII comparison 只用來描述 coverage 與 distribution differences，不作 modeling 或 transfer-readiness claim。

## Dataset provenance 與 cleaning

BDG2 data descriptor 追蹤於 [docs/reference/papers/bdg2-miller-2020.md](../reference/papers/bdg2-miller-2020.md)。PDF 本地存於 `docs/reference/papers/bdg2-miller-2020.pdf`，因超過 repo 500 KB large-file gate 而維持 gitignored。

Miller et al. 2020 描述 raw release pipeline 包含 unit conversion、negative readings set to missing、移除超過 50% negative readings 的 meters、移除超過 100 consecutive days missing readings 的 meters、log plus three-standard-deviation outlier removal，以及 four-decimal rounding。Cleaned release 再套用 Twitter AnomalyDetection outlier removal、移除超過 24 小時的 zero-reading runs、移除 electricity zeros。這些 release-level rules 解釋了為什麼本 EDA 中 raw negative-reading share 為 0，以及為什麼每個 meter 的 cleaned null rate 都高於 raw null rate。

## BDG2 data-quality inventory

### Per-meter structure

| Meter | Buildings | BDG2-only buildings | Raw null | Cleaned null | Raw zero | Raw negative | Raw flatline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | 1578 | 151 | 0.04739 | 0.08929 | 0.04189 | 0 | 0.1539 |
| chilledwater | 555 | 26 | 0.06948 | 0.07766 | 0.1654 | 0 | 0.1967 |
| steam | 370 | 3 | 0.1126 | 0.1218 | 0.1284 | 0 | 0.177 |
| hotwater | 185 | 19 | 0.06201 | 0.07425 | 0.3294 | 0 | 0.4551 |
| gas | 177 | 52 | 0.03337 | 0.0477 | 0.3776 | 0 | 0.4609 |
| water | 146 | 25 | 0.05783 | 0.06938 | 0.481 | 0 | 0.5284 |
| irrigation | 37 | 20 | 0.107 | 0.1189 | 0.7662 | 0 | 0.8319 |
| solar | 5 | 0 | 0.2013 | 0.2172 | 0.2719 | 0 | 0.331 |

### Flatline definition

Flatline share 使用明確規則回報：minimum run length 為 `2`；zero-reading runs 計入；missing values 會中斷 run；equality 採 `exact`。分母是 adjacent non-missing building-meter-hour comparisons，aggregation 是 cell-weighted adjacent comparisons。Zero-reading share 另行回報，因此 zero prevalence 不會被藏在 flatline statistic 裡。

### Missingness decomposition

下表把 building-level meter availability 與 observation-level missingness 分開。`Absent buildings` 指 metadata 裡存在、但 wide meter file 沒有對應 column 的 buildings。

| Meter | Absent buildings | Median timestamp coverage | Raw observation missingness | Cleaned observation missingness |
| --- | --- | --- | --- | --- |
| electricity | 58 | 0.9985 | 0.04739 | 0.08929 |
| chilledwater | 1081 | 0.999 | 0.06948 | 0.07766 |
| steam | 1266 | 0.999 | 0.1126 | 0.1218 |
| hotwater | 1451 | 0.9998 | 0.06201 | 0.07425 |
| gas | 1459 | 1 | 0.03337 | 0.0477 |
| water | 1490 | 0.9968 | 0.05783 | 0.06938 |
| irrigation | 1599 | 0.9444 | 0.107 | 0.1189 |
| solar | 1631 | 0.9987 | 0.2013 | 0.2172 |

### Cleaned-vs-raw delta

| Meter | Null-rate delta | Raw present -> cleaned missing | Raw missing -> cleaned present | Changed observed cells |
| --- | --- | --- | --- | --- |
| electricity | 0.04189 | 0.04189 | 0 | 0.0004208 |
| chilledwater | 0.00818 | 0.00818 | 0 | 0.005495 |
| steam | 0.009229 | 0.009229 | 0 | 0.00359 |
| hotwater | 0.01224 | 0.01224 | 0 | 0.001415 |
| gas | 0.01433 | 0.01433 | 0 | 0.0005059 |
| water | 0.01155 | 0.01155 | 0 | 0.00391 |
| irrigation | 0.01195 | 0.01195 | 0 | 0.001761 |
| solar | 0.01595 | 0.01595 | 0 | 0 |

每個 meter 的 raw-to-cleaned missing 都是正值，raw missing-to-cleaned present 都是 0；這符合 cleaned files 移除額外 observations、而不是填補 raw gaps 的解讀。

### Metadata completeness

| Field | Source column | Usage | BDG2 non-null | BDG2 summary | BDG2-only summary | GEPIII-overlap summary |
| --- | --- | --- | --- | --- | --- | --- |
| primary_use | primaryspaceusage | headline_distance | 0.9872 | top Education (617) | top Education (68) | top Education (549) |
| square_feet | sqft | headline_distance | 1 | median 5.462e+04 | median 2.786e+04 | median 5.767e+04 |
| sqm | sqm | descriptive_only | 1 | median 5074 | median 2588 | median 5358 |
| year_built | yearbuilt | descriptive_only | 0.4994 | median 1971 | median 1976 | median 1970 |
| floor_count | numberoffloors | descriptive_only | 0.2696 | median 2 | median 2 | median 3 |
| site_id | site_id | descriptive_only | 1 | top Rat (305) | top Lamb (58) | top Rat (274) |
| timezone | timezone | descriptive_only | 1 | top US/Eastern (812) | top Europe/London (75) | top US/Eastern (739) |

## BDG2-only sufficiency

BDG2 有 187 個 BDG2-only buildings 與 1,449 個 GEPIII-overlap buildings。下表摘要 BDG2-only meter availability 與 sufficient-observation split。以 chilledwater 來看，26 個 BDG2-only buildings 有 meter columns，3 個符合 `missing_rate <= 0.50` 規則，23 個屬於 high-missing。這是有用的 BDG2-only context；active M6.1 bridge 則從 GEPIII-overlap rows 開始。

| Meter | BDG2-only with meter | Sufficient obs | High missing | Median missing rate |
| --- | --- | --- | --- | --- |
| electricity | 151 | 99 | 52 | 0 |
| chilledwater | 26 | 3 | 23 | 0.5024 |
| steam | 3 | 3 | 0 | 0 |
| hotwater | 19 | 2 | 17 | 0.5018 |
| gas | 52 | 50 | 2 | 0 |
| water | 25 | 23 | 2 | 0.01146 |
| irrigation | 20 | 20 | 0 | 0.05424 |
| solar | 0 | 0 | 0 | n/a |

### Chilledwater sufficient-observation threshold sensitivity

| Missing-rate threshold | Sufficient BDG2-only chilledwater buildings |
| --- | --- |
| 0.40 | 2 |
| 0.45 | 2 |
| 0.50 | 3 |
| 0.55 | 24 |
| 0.60 | 24 |

This is a knife-edge gate-sensitive result:

+ Threshold 從 `0.50` 放寬到 `0.55` 時，eligible count 會從 `3` 增加到 `24`。這個跳升幾乎都來自 top-site table 裡的 Swan：Swan 提供約 20 個 BDG2-only chilledwater columns，其 missingness 集中在 `0.5024` median 附近，略高於 `0.50` gate。
+ 在 `0.55` threshold 下，24 個 buildings 會超過舊 powered lower bound 的 5 buildings。因此 chilledwater BDG2-only finding 取決於 `0.50` cut point 與 Swan 的 missingness shape，而不是 BDG2-only chilledwater readings 普遍不存在。
+ 後續 chilledwater work 可檢查 Swan 約半缺值的 chilledwater coverage 是 structurally contiguous 還是 dispersed；此處尚未刻畫。若它是 structurally contiguous，within-Swan subwindow 可能支援後續 Level-3 weather-conditioned chilledwater review path。

### BDG2-only top-site contribution

| Site | BDG2-only buildings | BDG2-only chilledwater columns | BDG2-only chilledwater sufficient obs |
| --- | --- | --- | --- |
| Lamb | 58 | 0 | 0 |
| Panther | 31 | 0 | 0 |
| Rat | 31 | 0 | 0 |
| Swan | 21 | 20 | 0 |

## GEPIII comparison context

GEPIII comparison 是 coverage 與 distribution differences 的 diagnostic lens。它不是 modeling result、不是 transfer result，也不是 readiness claim。

### Meter coverage context

| Meter | All buildings marked yes | BDG2-only | GEPIII-overlap |
| --- | --- | --- | --- |
| electricity | 1578 | 151 | 1427 |
| chilledwater | 555 | 26 | 529 |
| steam | 370 | 3 | 367 |
| hotwater | 185 | 19 | 166 |
| gas | 177 | 52 | 125 |
| water | 146 | 25 | 121 |
| irrigation | 37 | 20 | 17 |
| solar | 5 | 0 | 5 |

BDG2-only vs GEPIII 的 primary-use unseen/unmapped rate 是 `0.1123`。Unseen or unmapped normalized categories 為 `(missing/unmapped)`。

Square-feet medians:

+ BDG2-only: `2.786e+04`.
+ GEPIII-overlap: `5.767e+04`.
+ GEPIII: `5.767e+04`.

BDG2-only buildings 集中在較少 sites，尤其是本地 archive 中的 Lamb、Panther、Rat、Swan。Meter availability 依 meter 差異很大。Electricity 最廣；solar 與 irrigation 很窄。Chilledwater 有足夠 overlap buildings 可支援 bridge baseline，但 BDG2-only sufficient-observation buildings 不足以支撐舊 Step 4 frame。

### Reference distribution distances

| Feature | KS | PSI | Basis |
| --- | --- | --- | --- |
| square_feet | 0.2176 | 0.2235 | BDG2-only vs GEPIII metadata |
| meter_reading | 0.4549 | 1.102 | sampled raw BDG2-only cells vs GEPIII `load_m3_frame` cells |
| primary_use coverage | n/a | 1.415 | categorical PSI; unseen/unmapped rate 0.1123 |

`meter_reading` distance 比較 sampled BDG2 raw cells 與 `load_m3_frame` 讀到的 GEPIII Kaggle-release cells。這個距離的一部分來自 Miller et al. 2020 描述的 release-level differences：meter-type mix、zero inflation、site composition、Kaggle unit-conversion errors，以及 BDG2 raw/cleaned 已修正但 Kaggle subset 保留的 UTC-vs-local weather timestamps。因此它不能被讀成純粹的 building behavior distance；後續 refinement 應優先使用 per-meter、log1p、zero-excluded distances。

### Per-meter reference distances

| Meter | Variant | KS | PSI | BDG2-only zero share | GEPIII zero share |
| --- | --- | --- | --- | --- | --- |
| electricity | raw_zero_included | 0.4447 | 1.017 | 0.2095 | 0.04376 |
| electricity | log1p_zero_included | 0.4447 | 1.017 | 0.2095 | 0.04376 |
| electricity | log1p_zero_excluded | 0.3799 | 0.8278 | 0.2095 | 0.04376 |
| chilledwater | raw_zero_included | 0.1177 | 0.08789 | 0.2543 | 0.1568 |
| chilledwater | log1p_zero_included | 0.1177 | 0.08789 | 0.2543 | 0.1568 |
| chilledwater | log1p_zero_excluded | 0.06005 | 0.07149 | 0.2543 | 0.1568 |
| steam | raw_zero_included | 0.5237 | 1.24 | 0.6224 | 0.1279 |
| steam | log1p_zero_included | 0.5237 | 1.24 | 0.6224 | 0.1279 |
| steam | log1p_zero_excluded | 0.1391 | 0.3773 | 0.6224 | 0.1279 |
| hotwater | raw_zero_included | 0.3487 | 1.66 | 0.4643 | 0.27 |
| hotwater | log1p_zero_included | 0.3487 | 1.66 | 0.4643 | 0.27 |
| hotwater | log1p_zero_excluded | 0.4582 | 2.322 | 0.4643 | 0.27 |

Chilledwater 在此表中的 per-meter distance 最低：raw KS `0.1177`，log1p-zero-excluded KS `0.06005`。這表示 chilledwater BDG2-only context 的主問題仍回到 coverage 與 missingness，尤其是 Swan，而不是 chilledwater reading magnitude 明顯離開 GEPIII reference distribution。較大的 pooled meter_reading distance 主要由 steam/electricity composition、zero inflation、release-regime differences 驅動。

Figures:

+ ![Square feet ECDF](../assets/bdg2-eda/square-feet-ecdf.png)
+ ![Meter reading histogram](../assets/bdg2-eda/meter-reading-hist.png)

## Temporal profiles

Provenance JSON 包含 representative electricity 與 chilledwater raw readings 的 hour/month mean profiles。這些只作 descriptive profiles，不是 model features、scores 或 readiness evidence。

+ Electricity 的最高 mean reading 約在 hour 14，最低約在 hour 3；按 month 看，8 月最高、12 月最低。
+ Chilledwater 的最高 mean reading 約在 hour 20，最低約在 hour 8；按 month 看，7 月最高、12 月最低。

## 方法 caveats 與 review notes

+ Released-raw negative-reading share 是在 released BDG2 raw files 上量測。這不代表原始 site-source feeds 從未有 negative readings；Miller et al. 2020 描述 release processing 會把 negative readings 設為 missing，並移除超過 50% negative readings 的 meters。
+ Cleaned null rate 高於 raw null rate 是 data-quality delta，不是 label。Miller et al. 2020 描述 cleaned files 會套用 Twitter AnomalyDetection outlier removal、移除超過 24 小時的 zero-reading runs、移除 electricity zeros。
+ Pooled meter_reading KS/PSI 只是 headline diagnostic。它混合 meter-type composition、zero inflation、site composition、以及已知 BDG2-vs-GEPIII release-regime differences；不能解讀成純 building behavior distance。

## Provenance

+ Machine-readable summary: `data/processed/bdg2_eda.json` (gitignored shard).
+ Script: `scripts/run_bdg2_eda.py`.
+ BDG2 source: `data/raw/bdg2`.
+ BDG2 paper reference: `docs/reference/papers/bdg2-miller-2020.md`.
+ GEPIII comparison source: `load_m3_frame(verbose=False)` and
  `data/raw/m3/building_metadata.csv`.
+ Distance scalar sampling: per-meter BDG2 sample
  `80000`, GEPIII sample
  `400000`, seed
  `42`.
