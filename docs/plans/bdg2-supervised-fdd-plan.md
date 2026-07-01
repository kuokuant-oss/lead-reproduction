# BDG2 Supervised FDD Plan

**Stage**: M6 pivot plan；基於 GEPIII overlap label bridge
**Status**: Draft for review; P0 documentation pivot
**決策紀錄**: [ADR 0025](../adr/0025-supervised-bdg2-fdd-overlap-evaluation.md), [ADR 0026](../adr/0026-bdg2-label-bridge-integrity.md)

## Scope

M6 在 BDG2 上做 FDD 評估，但 supervised scope 只限於可橋接真實 label 的 overlap 子集：GEPIII-overlap buildings、2016、meters `electricity`、`chilledwater`、`steam`、`hotwater`。

Label 來源是 `data/raw/m3/bad_meter_readings.csv`，也就是 M2/M3 已使用的 rank-1 manual GEPIII/Kaggle bad-reading annotations。橋接 key 是 `building_id_kaggle`、meter code、timestamp。這些 labels 的 provenance 是 GEPIII rank-1 annotation，不是 BDG2 native release label。

BDG2-only buildings、2017 rows、以及非 GEPIII meters 仍是 unlabeled。它們要被計數並排除在 supervised metrics 之外，直到後續 M6.4 secondary pseudo-label 或 review slice。

## M6 Supervised Scope Contract

M6 的所有 supervised BDG2 metrics 只能使用：

+ GEPIII-overlap buildings。
+ Year 2016。
+ Meters 0-3：`electricity`、`chilledwater`、`steam`、`hotwater`。
+ Labels bridged from rank-1 GEPIII/Kaggle `bad_meter_readings`。
+ Join key：`building_id_kaggle`、meter code、timestamp。

排除在 supervised denominators 之外：

+ BDG2-only buildings。
+ 2017 rows。
+ Non-GEPIII meters。
+ Raw/cleaned deltas。
+ Pseudo-label or review evidence。

本文件是 M6.1-M6.3 的 implementation contract；高階 roadmap 維護在 [phaseE-fdd-roadmap.md](./phaseE-fdd-roadmap.md)。

## Active / Secondary / Retired 路線

| Path | Status | Allowed outputs |
| --- | --- | --- |
| M6.1-M6.3 supervised overlap | Active | Bridge integrity、supervised metrics、model comparison |
| M6.4 unlabeled remainder | Secondary | Pseudo-labels、review evidence、descriptive quality signals |
| Old Phase E audit-yield route | Retired | Historical context only |

## Guardrails

+ M3 numeric line 凍結：M3.2 `0.9920`、M3.4 `0.9928`、seeds、default `row_offset`、StandardScaler path 都不動。
+ `lead.__all__` 凍結；後續若要加 public API，必須是 additive export，並有 tests 與 ADR coverage。
+ Raw BDG2 是 primary scoring surface；cleaned BDG2 是 companion sensitivity。
+ GEPIII-only `0.2931` correction 不進 BDG2 path。
+ Metrics 使用既有 helper，尤其是 `classification_metrics`。
+ 每個 supervised metric 都要標明 GEPIII-overlap、2016、meters 0-3、bridged rank-1 GEPIII annotations。

## M6 Ladder

### M6.1 Label Bridge And Integrity

建立 keyed label bridge，先驗證 integrity，再談 metrics。

完成條件：

+ eligible rows 可以建立 labeled overlap frame；
+ ADR 0026 integrity checks 通過；
+ coverage JSON 記錄 row counts、label hit rate、null-label rate、excluded rows；
+ 尚不回報 supervised accuracy metrics。

### M6.2 Supervised Transfer Accuracy

把 GEPIII-trained M3.4 ensemble score 到 BDG2 raw overlap rows，並按 meter 回報 ROC-AUC、PR-AUC、precision、recall、F1。Cleaned BDG2 作 companion sensitivity。

完成條件：

+ raw primary 與 cleaned companion supervised tables 都存在；
+ unknown #27 被量測為 source-vs-target / raw-vs-cleaned feature-regime delta；
+ 若 review 接受 evidence，ADR 0025 可由 Proposed 移到 Accepted。

### M6.3 GBDT 與 TabPFN 在 labeled BDG2 overlap 上比較

在同一個 labeled BDG2 overlap frame 上比較 GBDT 與 TabPFN，accuracy 與 latency 並列呈現。

完成條件：

+ comparison 沿用 M5 Phase D 的紀律；
+ TabPFN license 與約 `6.3 ms/row` latency caveats 隨結果一起呈現；
+ verdict 來自 labeled BDG2 overlap evidence：GBDT wins、TabPFN wins、或 hybrid workflow warranted。

### M6.4 Unlabeled Remainder

BDG2-only、2017、其他-meter rows 走 secondary pseudo-label 或 review screen。Raw-vs-cleaned 可以作資料品質訊號，但不是 ground truth。

完成條件：

+ 所有 unlabeled outputs 都標為 pseudo-label 或 review evidence；
+ 沒有任何 unlabeled remainder row 進入 supervised denominators。

### M6.5 Close-Out

更新 README、plans、ADR status、reports、handoff、validation evidence、issue state、provenance placement。

## Slice tracker

| Slice | Issue | Status | ADR |
| --- | --- | --- | --- |
| P0 docs-only pivot record | Not opened | In progress | ADR 0025/0026 Proposed |
| M6.1 label bridge + integrity | Not opened | Queued after P0 review | ADR 0026 |
| M6.2 supervised transfer accuracy | Not opened | Queued after M6.1 | ADR 0025 |
| M6.3 GBDT vs TabPFN supervised | Not opened | Queued after M6.2 | TBD |
| M6.4 unlabeled remainder | Not opened | Queued after M6.3 | TBD |
| M6.5 close-out | Not opened | Queued after M6.4 | TBD |

## Parked Contingency

LEAD1.0 dual-label electricity work 維持 parked。P0 不下載、不接線、不引用到 active M6 metrics。若後續要重啟，必須另開 approved issue 與 ADR。
