# BDG2 Data Descriptor Reference

**Citation**: Miller, C., Kathirgamanathan, A., Picchetti, B., Arjunan, P.,
Park, J. Y., Nagy, Z., Raftery, P., Hobson, B. W., Shi, Z., & Meggers, F.
(2020). The Building Data Genome Project 2, energy meter data from the ASHRAE
Great Energy Predictor III competition. *Scientific Data*, 7, 368.

**DOI**: <https://doi.org/10.1038/s41597-020-00712-x>

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

**Official repository**:
<https://github.com/buds-lab/building-data-genome-project-2>

## 資料集定位

BDG2（Building Data Genome Project 2）是一個公開的 whole-building meter dataset。
Miller et al. 2020 將它定位為建築能源資料科學的基準資料集：資料來自 1,636 棟非住宅
建築、3,053 個能源或資源 meter，涵蓋 2016 與 2017 兩個完整年度，時間解析度為每小時
一筆。每個完整 meter 約有 17,544 個 hourly observations，整體約 5,360 萬筆 meter
readings。

資料涵蓋北美與歐洲 19 個 sites。建築層級 metadata、weather time series、raw meter
files 與 cleaned meter files 一起發布。ASHRAE Great Energy Predictor III（GEPIII）
competition 使用了 BDG2 的一個子集；因此 BDG2 與本 repo 的 M2/M3 GEPIII reproduction
有重疊，兩者屬於不同 release surface。

## Repo 內的實際資料形狀

本 repo 使用的本地 archive 是 Kaggle 發布的 flat CSV archive。
`docs/reports/bdg2-data-reality.md` 對本地檔案做過 read-only inventory，量到：

+ `metadata.csv`：`1636 x 32`。
+ raw meter files：8 個。
+ cleaned meter files：8 個。
+ weather file：`331166 x 10`。
+ 每個 meter file 是 wide table：第一欄 `timestamp`，其餘欄位是 building id。
+ 所有 meter files 的時間範圍都是 `2016-01-01 00:00:00` 到
  `2017-12-31 23:00:00`，共 17,544 小時。

Ingestion 時，wide meter file 需要 melt 成 long format：
`(building_id, meter_type, timestamp, meter_reading)`。File stem 是 meter type，
cell value 是 reading。這是 repo loader contract 的基礎。

## Metadata

BDG2 metadata 使用 string 型態的 `building_id`，例如 `Panther_lodging_Dean`。
GEPIII overlap 由 `building_id_kaggle` 與 `site_id_kaggle` 保留。

常用欄位對照：

| GEPIII concept | BDG2 actual column |
| --- | --- |
| building_id | `building_id` |
| site_id | `site_id` |
| building_id_kaggle | `building_id_kaggle` |
| site_id_kaggle | `site_id_kaggle` |
| primary_use | `primaryspaceusage` |
| square_feet | `sqft` |
| year_built | `yearbuilt` |
| floor_count | `numberoffloors` |

Metadata 中共有 19 個 sites、6 個 timezones。`primaryspaceusage`、`sqft`、`sqm`、
`timezone`、meter availability flags 等欄位可用於 coverage 與 distribution context。
`yearbuilt` 與 `numberoffloors` 的缺值較多，適合作為 descriptive context。

## Meter 類型與 coverage

BDG2 release 包含 8 種 meter type。各 meter 的 building coverage 差異很大：

| Meter | Buildings marked yes | Raw file building columns | Raw null rate |
| --- | ---:| ---:| ---:|
| electricity | 1578 | 1578 | 0.04739 |
| chilledwater | 555 | 555 | 0.06948 |
| steam | 370 | 370 | 0.11259 |
| hotwater | 185 | 185 | 0.06201 |
| gas | 177 | 177 | 0.03337 |
| water | 146 | 146 | 0.05783 |
| irrigation | 37 | 37 | 0.10697 |
| solar | 5 | 5 | 0.20128 |

Electricity 是 coverage 最廣的 meter。Chilledwater、steam、hotwater 仍有足夠
GEPIII-overlap coverage，可支援 M6 overlap bridge 的前段設計；但 BDG2-only
chilledwater coverage 很窄。

## Weather

`weather.csv` 使用 `site_id` 與 `timestamp` 作為主要 join key。它包含：

+ `airTemperature`
+ `cloudCoverage`
+ `dewTemperature`
+ `precipDepth1HR`
+ `precipDepth6HR`
+ `seaLvlPressure`
+ `windDirection`
+ `windSpeed`

Weather file 使用 site-level timestamp。Local-time interpretation 需要從 metadata
join site timezone。GEPIII/Kaggle subset 與 BDG2 release 的 timestamp、weather
correction semantics 有 release-regime 差異。

## Raw 與 cleaned release

BDG2 同時發布 raw 與 cleaned meter files。Miller et al. 2020 描述的 raw-data
processing 包含：

+ unit conversion；
+ negative readings set to missing；
+ 移除超過 50% negative readings 的 meters；
+ 移除超過 100 consecutive days missing readings 的 meters；
+ log plus three-standard-deviation outlier rule；
+ readings rounded to four decimals。

Cleaned files 進一步套用 outlier / zero-run cleaning，包括 Twitter AnomalyDetection
outlier removal、移除超過 24 小時的 zero-reading runs、移除 electricity zeros。

因此，本 repo 在 EDA 中看到的現象應該被解讀為 release-level data-quality delta：

+ released raw negative-reading share 為 0，反映 release processing 已處理 negative
  readings。
+ 每個 meter 的 cleaned null rate 都高於 raw null rate，代表 cleaned release 移除更多
  observations。
+ raw-present / cleaned-missing cells 可作 secondary pseudo-label 或 review evidence。

## 與 GEPIII 的關係

BDG2 和 GEPIII 的關係要分三層看：

+ BDG2 是完整的 2016+2017 building meter release。
+ GEPIII/Kaggle 是其中的競賽子集，且 release semantics 與 BDG2 raw/cleaned files
  有差異。
+ 本 repo 的 M2/M3 labels 來自 buds-lab / rank-1 GEPIII
  `bad_meter_readings.csv`。

本地 metadata 量到 1,449 棟 buildings 有 `building_id_kaggle`，屬於
GEPIII-overlap；187 棟是 BDG2-only。ADR 0025/0026 將 M6 supervised scope 限定在
GEPIII-overlap、2016、meters `electricity`、`chilledwater`、`steam`、`hotwater`，
並用 `(building_id_kaggle, meter code, timestamp)` 橋接 rank-1 GEPIII annotations。

## FDD 使用邊界

BDG2 支援下列 FDD 工作：

+ meter coverage、missingness、zero-reading、flatline、raw-vs-cleaned delta 的資料品質分析；
+ GEPIII-overlap subset 上的 supervised FDD evaluation；
+ BDG2-only / 2017 / non-GEPIII meters 的 secondary pseudo-label 或 review workflow；
+ building metadata、weather、meter profile 的 distribution-shift analysis。
