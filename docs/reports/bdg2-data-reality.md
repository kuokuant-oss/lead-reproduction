# BDG2 Data Reality Report

**Stage**: Phase E Stage 0, read-only inventory
**Generated**: 2026-06-29T15:47:55+08:00
**Input directory**: `C:\Users\tonykuo\projects\lead-reproduction\data\raw\bdg2`

## Provenance

- Local source: Kaggle archive `claytonmiller/buildingdatagenomeproject2`, copied from `C:\Users\tonykuo\Downloads\archive`.
- Official reference repo: https://github.com/buds-lab/building-data-genome-project-2
- Meter field reference: https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features
- Paper: Miller et al., Scientific Data 7, 368 (2020), DOI `10.1038/s41597-020-00712-x`.
- Download/copy date: 2026-06-29.
- Upstream commit: not applicable to this local Kaggle CSV archive. The prior Git checkout was discarded because it contained Git LFS pointers rather than real CSV data.

## File Gate

- CSV files found: 18.
- Git LFS pointer files: 0.
- `electricity.csv`: 166.167 MB.
- `weather.csv`: 18.556 MB.
- `metadata.csv`: 0.259 MB.

## Metadata Reality

- `metadata.csv` shape: 1636 rows x 32 columns.
- Building id is the string `building_id` field, for example `Panther_lodging_Dean`.
- Site count: 19; building count: 1636.
- Timezone column: `timezone`, with 6 distinct values: Europe/Dublin, Europe/London, US/Central, US/Eastern, US/Mountain, US/Pacific.

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

- `building_id_kaggle` non-empty buildings: 1449.
- BDG2-only buildings: 187.
- Loader contract: retain `building_id_kaggle`, `site_id_kaggle`, and derive `is_gepiii_overlap`.

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
- Layout: each meter file is a wide table with `timestamp` plus one column per building id.
- Long-table mapping for ingestion: `(building, meter_type, timestamp, reading)` is obtained by melting each meter file's building columns, using the file stem as `meter_type` and the cell value as `reading`.

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

- `weather.csv` shape: 331166 rows x 10 columns.
- Weather key: `site_id`; site count: 19.
- Timestamp range: 2016-01-01 00:00:00 to 2017-12-31 23:00:00.
- Timezone column present: False. Site timezone must therefore be joined from metadata if local-time interpretation is needed.
- Columns: `timestamp, site_id, airTemperature, cloudCoverage, dewTemperature, precipDepth1HR, precipDepth6HR, seaLvlPressure, windDirection, windSpeed`
- Null rate: 0.197133.

## Label Reality

- Per-row anomaly labels present: **False**.
- Label-like files found: [].
- Label-like metadata/weather columns found: [].

BDG2, as present in this archive, does not provide a per-row anomaly label comparable to GEPIII `bad_meter_readings.csv`. Any Phase E supervised FDD claim must therefore choose and document a label strategy before training or evaluation.

Viable strategies:

- Unsupervised detection on BDG2 meter series.
- Forecasting-residual labels or anomaly scores from held-out temporal forecasts.
- Apply a GEPIII-trained detector as a cross-dataset scoring baseline.
- Raw/cleaned difference pseudo-labels: cells present in raw but set to `NaN` in cleaned are a candidate proxy for BDG2-cleaning-identified bad readings.

## Raw, Cleaned, and Kaggle Variants

- This local archive has raw and cleaned files for all eight BDG2 meter types.
- Cleaned null rates are higher than raw null rates for every measured meter type:

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

- The separate GEPIII/Kaggle 2017 subset file is not present in this archive; this matches the user-provided correction that `kaggle.csv` is not part of the local true-data download.
- Source-level note for Stage 1 ADR: BDG2 raw/cleaned files are the full 2016+2017 BDG2 meter release, while the GEPIII/Kaggle subset is a 2017 local-time subset with different unit-correction semantics. The loader contract must not assume the GEPIII site-0/meter-0 correction applies to BDG2 raw/cleaned files.

## Stage 0 Decision Boundary

This report is the fact base for Stage 1. No `src/lead` code was inspected or changed by this inventory script, and no BDG2 loader contract is inferred here beyond the measured file schema.
