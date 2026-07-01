# BDG2 Data Reality Report

**Stage**: Phase E Stage 0, read-only inventory
**Generated**: 2026-06-29T15:47:55+08:00
**Input directory**: `C:\Users\tonykuo\projects\lead-reproduction\data\raw\bdg2`

## Provenance

- Local source：Kaggle archive `claytonmiller/buildingdatagenomeproject2`，由 `C:\Users\tonykuo\Downloads\archive` 複製。
- Official reference repo：https://github.com/buds-lab/building-data-genome-project-2
- Meter field reference：https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features
- Paper：Miller et al., Scientific Data 7, 368 (2020), DOI `10.1038/s41597-020-00712-x`。
- Download/copy date：2026-06-29。
- Upstream commit：不適用於本地 Kaggle CSV archive。先前 Git checkout 因為只有 Git LFS pointer、沒有真實 CSV，已捨棄。

## File Gate

- CSV files found：18。
- Git LFS pointer files：0。
- `electricity.csv`: 166.167 MB.
- `weather.csv`: 18.556 MB.
- `metadata.csv`: 0.259 MB.

## Metadata Reality

- `metadata.csv` shape：1636 rows x 32 columns。
- Building id 是 string `building_id` field，例如 `Panther_lodging_Dean`。
- Site count：19；building count：1636。
- Timezone column：`timezone`，共有 6 個 distinct values：Europe/Dublin、Europe/London、US/Central、US/Eastern、US/Mountain、US/Pacific。

### GEPIII-to-BDG2 Field Mapping

| GEPIII concept | BDG2 actual column |
| --- | --- |
| building_id | building_id |
| site_id | site_id |
| building_id_kaggle | building_id_kaggle |
| site_id_kaggle | site_id_kaggle |
| primary_use | primaryspaceusage |
| square_feet | sqft |
| year_built | yearbuilt |
| floor_count | numberoffloors |

### GEPIII Overlap Bridge

- `building_id_kaggle` non-empty buildings：1449。
- BDG2-only buildings：187。
- Loader contract：保留 `building_id_kaggle`、`site_id_kaggle`，並 derive `is_gepiii_overlap`。

Overlap site distribution:

| Site | GEPIII-overlap buildings |
| --- | --- |
| Bear | 91 |
| Bobcat | 30 |
| Bull | 124 |
| Cockatoo | 124 |
| Crow | 5 |
| Eagle | 102 |
| Fox | 135 |
| Gator | 70 |
| Hog | 154 |
| Lamb | 89 |
| Moose | 15 |
| Panther | 105 |
| Peacock | 44 |
| Rat | 274 |
| Robin | 51 |
| Wolf | 36 |

BDG2-only site distribution:

| Site | BDG2-only buildings |
| --- | --- |
| Bear | 1 |
| Bobcat | 6 |
| Eagle | 4 |
| Fox | 2 |
| Gator | 4 |
| Hog | 9 |
| Lamb | 58 |
| Mouse | 7 |
| Panther | 31 |
| Peacock | 3 |
| Rat | 31 |
| Robin | 1 |
| Shrew | 9 |
| Swan | 21 |

### Metadata Columns

`building_id, site_id, building_id_kaggle, site_id_kaggle, primaryspaceusage, sub_primaryspaceusage, sqm, sqft, lat, lng, timezone, electricity, hotwater, chilledwater, steam, water, irrigation, solar, gas, industry, subindustry, heatingtype, yearbuilt, date_opened, numberoffloors, occupants, energystarscore, eui, site_eui, source_eui, leed_level, rating`

### Meter Availability in Metadata

| Meter | Buildings marked Yes |
| --- | --- |
| electricity | 1578 |
| chilledwater | 555 |
| steam | 370 |
| hotwater | 185 |
| gas | 177 |
| water | 146 |
| irrigation | 37 |
| solar | 5 |

## Meter Files Reality

- Raw meter files: 8.
- Cleaned meter files: 8.
- Layout：每個 meter file 都是 wide table，包含 `timestamp` 與每個 building id 對應的一個 column。
- Long-table mapping for ingestion：把每個 meter file 的 building columns melt 成 `(building, meter_type, timestamp, reading)`；file stem 作為 `meter_type`，cell value 作為 `reading`。

| File | Variant | Meter | Rows | Building cols | Start | End | Null rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chilledwater.csv | raw | chilledwater | 17544 | 555 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.069479 |
| chilledwater_cleaned.csv | cleaned | chilledwater | 17544 | 555 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.077659 |
| electricity.csv | raw | electricity | 17544 | 1578 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.047395 |
| electricity_cleaned.csv | cleaned | electricity | 17544 | 1578 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.089287 |
| gas.csv | raw | gas | 17544 | 177 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.033372 |
| gas_cleaned.csv | cleaned | gas | 17544 | 177 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.047701 |
| hotwater.csv | raw | hotwater | 17544 | 185 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.062009 |
| hotwater_cleaned.csv | cleaned | hotwater | 17544 | 185 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.074247 |
| irrigation.csv | raw | irrigation | 17544 | 37 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.106967 |
| irrigation_cleaned.csv | cleaned | irrigation | 17544 | 37 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.118913 |
| solar.csv | raw | solar | 17544 | 5 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.201277 |
| solar_cleaned.csv | cleaned | solar | 17544 | 5 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.217225 |
| steam.csv | raw | steam | 17544 | 370 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.112592 |
| steam_cleaned.csv | cleaned | steam | 17544 | 370 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.121821 |
| water.csv | raw | water | 17544 | 146 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.05783 |
| water_cleaned.csv | cleaned | water | 17544 | 146 | 2016-01-01 00:00:00 | 2017-12-31 23:00:00 | 0.069383 |

## Weather Reality

- `weather.csv` shape：331166 rows x 10 columns。
- Weather key：`site_id`；site count：19。
- Timestamp range：2016-01-01 00:00:00 to 2017-12-31 23:00:00。
- Timezone column present：False。若需要 local-time interpretation，site timezone 必須從 metadata join。
- Columns：`timestamp, site_id, airTemperature, cloudCoverage, dewTemperature, precipDepth1HR, precipDepth6HR, seaLvlPressure, windDirection, windSpeed`
- Null rate：0.197133。
- Stage 1 timezone diagnostic note：`0.197133` 是整張 weather table across all weather fields 的 null rate。Meter/weather phase diagnostic 使用的 `airTemperature` column null rate 是 `0.000387`。

## Label Reality

- Per-row anomaly labels present：**False**。
- Label-like files found：[]。
- Label-like metadata/weather columns found：[]。

本地 BDG2 archive 沒有可與 GEPIII `bad_meter_readings.csv` 對等的 native per-row anomaly label。ADR 0025/0026 定義的現行策略是：把 rank-1 manual GEPIII annotations 透過 `building_id_kaggle`、meter code、timestamp 橋接到 BDG2 的 GEPIII-overlap、2016、meters-0-3 子集。這個 bridge 不標記 BDG2-only buildings、2017 rows 或 non-GEPIII meters。

Implication for M6:

- Supervised evaluation 需要 GEPIII/Kaggle bad-meter-reading bridge。
- Bridge 外的 rows 維持 unlabeled。
- Unlabeled rows 可支援 secondary review evidence，但不能進入 supervised metrics。

可行路線：

- Primary M6 path：ADR 0026 bridge integrity gate 通過後，在 GEPIII-overlap、2016、meters-0-3 子集做 supervised evaluation。
- Secondary M6.4 path：unlabeled BDG2 remainder 只作 pseudo-label 或 review evidence。
- Forecasting-residual labels 或 held-out temporal forecasts 產生的 anomaly scores，可作後續 secondary evidence。
- GEPIII-trained detector 可作 cross-dataset scoring baseline，但 metric scope 必須明確。
- Raw/cleaned difference pseudo-labels：raw present 但 cleaned 設為 `NaN` 的 cells 只能作 secondary branch 的候選 proxy。

## Raw, Cleaned, and Kaggle Variants

- 本地 archive 對 8 種 BDG2 meter types 都有 raw 與 cleaned files。
- 每個 measured meter type 的 cleaned null rate 都高於 raw null rate：

| Meter | cleaned null rate - raw null rate |
| --- | --- |
| chilledwater | 0.00818 |
| electricity | 0.041892 |
| gas | 0.014329 |
| hotwater | 0.012238 |
| irrigation | 0.011946 |
| solar | 0.015948 |
| steam | 0.009229 |
| water | 0.011553 |

- 這個 archive 沒有 separate GEPIII/Kaggle 2017 subset file；這符合 user-provided correction：`kaggle.csv` 不是本地 true-data download 的一部分。
- Stage 1 ADR 的 source-level note：BDG2 raw/cleaned files 是完整 2016+2017 BDG2 meter release；GEPIII/Kaggle subset 是 2017 local-time subset，unit-correction semantics 不同。Loader contract 不得假設 GEPIII site-0/meter-0 correction 適用於 BDG2 raw/cleaned files。

## Stage 0 decision boundary

本報告是 Stage 1 之後工作的 fact base。這個 inventory script 沒有 inspect 或改動 `src/lead` code，也不從量測到的 file schema 之外推論新的 BDG2 loader contract。
