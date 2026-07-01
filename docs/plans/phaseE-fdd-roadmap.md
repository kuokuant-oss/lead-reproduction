# Phase E 到 M6 FDD Roadmap

## 目的與狀態

這份文件是 Phase E 到 M6 的現行 roadmap。Phase E Part A 已完成；現行 M6 主線是 BDG2 的 supervised FDD 評估，範圍限於 GEPIII-overlap subset，決策記錄為 [ADR 0025](../adr/0025-supervised-bdg2-fdd-overlap-evaluation.md) 與 [ADR 0026](../adr/0026-bdg2-label-bridge-integrity.md)。

本輪 P0 pivot 是 docs-only：不實作 label bridge、不跑 scoring、不回報 metric、不改 `src/lead`。

## 現行 M6 路線

M6 透過 GEPIII-overlap 的 supervised bridge 在 BDG2 上評估 FDD。現行路線是：

1. M6.1：建立 label bridge 並驗證 integrity。
2. M6.2：在 raw BDG2 overlap rows 上回報 supervised transfer metrics。
3. M6.3：在同一個 labeled overlap frame 上比較 GBDT 與 TabPFN。
4. M6.4：把 unlabeled remainder evidence 分開回報。
5. M6.5：收斂 documentation、provenance、validation 與 issue state。

所有 supervised metrics 都遵守 [bdg2-supervised-fdd-plan.md](./bdg2-supervised-fdd-plan.md) 的 M6 Supervised Scope Contract。

## Pivot fact

本地 BDG2 archive 沒有 native per-row anomaly labels。可用的監督式 label 來源是 M2/M3 已使用的 rank-1 manual GEPIII/Kaggle `bad_meter_readings.csv`。BDG2 metadata 保留 GEPIII-overlap buildings 的 `building_id_kaggle`，因此 M6 可以用 `(kaggle_building_id, meter_code, timestamp)` 把這批 labels 橋接到 BDG2 overlap rows。

合法的 supervised metrics 只屬於這個子集：GEPIII-overlap buildings、2016、meters 0-3，也就是 `electricity`、`chilledwater`、`steam`、`hotwater`。BDG2-only buildings、2017 rows、以及其他 meters 仍是 unlabeled。

## 固定邊界

+ M3 numeric line 凍結：`load_m3_frame` defaults、M3.2/M3.4 golden values、`+/- 0.0005` gate、downsampling semantics、StandardScaler fit path 都不動。
+ `lead.__all__` 凍結；除非後續 implementation slice 有 ADR 與 `test_public_api.py` coverage，否則不新增 public export。
+ Raw BDG2 是 primary scoring surface；cleaned BDG2 只作 companion sensitivity。
+ GEPIII-only `0.2931` unit correction 不進 BDG2 path。
+ 每個 supervised BDG2 metric 都必須明寫 scope：GEPIII-overlap、2016、meters 0-3、bridged rank-1 GEPIII annotations。
+ BDG2-only、2017、water、gas、solar、irrigation rows 要計數並排除在 supervised denominators 之外。
+ 每個 implementation slice 維持 one issue、one commit、stop for review，並跑完整 change checklist。

## M6 Ladder

### M6.1 Label Bridge And Integrity

建立 GEPIII labels 到 BDG2 overlap rows 的 keyed bridge，先證明橋接完整性，不回報 accuracy metrics。

完成條件：

+ eligible rows 可以建立 labeled overlap frame；
+ ADR 0026 guards 通過，包含 label-file schema、length、index、unique label keys、timestamp-grid sampling、null-label-rate checks；
+ coverage provenance 記錄 eligible rows、hit rates、positive counts、以及 excluded BDG2-only/2017/other-meter rows；
+ 不回報 supervised accuracy metrics。

### M6.2 Supervised Transfer Accuracy

把 GEPIII-trained M3.4 ensemble score 到 BDG2 raw overlap rows，並按 meter 回報 ROC-AUC、PR-AUC、precision、recall、F1。Cleaned BDG2 只作 companion sensitivity。

完成條件：

+ raw primary metrics 與 cleaned companion metrics 都存在；
+ unknown #27 量測為 raw-vs-source / BDG2-vs-Kaggle feature-regime delta；
+ 若 review 接受 evidence，ADR 0025 可由 Proposed 移到 Accepted。

### M6.3 GBDT Vs TabPFN Supervised Comparison

在同一個 labeled BDG2 overlap frame 上比較 GBDT 與 TabPFN。

完成條件：

+ accuracy 與 latency side by side；
+ TabPFN research/internal-use license 與約 `6.3 ms/row` latency caveats 隨結果一起呈現；
+ verdict 來自 labeled BDG2 overlap evidence，而不是預先指定 TabPFN 或 GBDT 的角色。

### M6.4 Unlabeled Remainder

BDG2-only buildings、2017 rows、其他 meters 走 secondary pseudo-label 或 review screen。這一支可以使用 raw-vs-cleaned 與資料品質訊號，但不能宣稱 ground truth。

完成條件：

+ 所有 outputs 都標為 pseudo-label 或 review evidence；
+ unlabeled remainder rows 不進 supervised metrics。

### M6.5 Close-Out

收斂 README、plan、ADR、handoff、provenance、validation、issue 與 CI 狀態。

## Unknown #27

Unknown #27 是 M6.2 要量測的 feature-regime delta：GEPIII/Kaggle source 與 BDG2 raw/cleaned target 在 weather timestamp、unit correction、meter reading distribution、coverage/missingness 上的差距。

GEPIII/Kaggle source 保留 UTC weather timestamps 與 unit-conversion errors；BDG2 raw/cleaned 使用 local-time weather 與 corrected units。GEPIII-only `0.2931` correction 不進 BDG2 path，避免 double conversion。

## 已退役或暫停的路線

| Route | Status | Why it is not active |
| --- | --- | --- |
| ADR 0019 BDG2 evaluation paradigm | Superseded | 已由 ADR 0025 supervised-overlap scope 取代 |
| ADR 0020 audit-yield evaluation | Historical context | 移到 M6.4 secondary review 背景 |
| ADR 0021 powered gate | Moot for primary M6 | Labeled-overlap denominator 不需要這個 gate |
| Chilledwater Step 4 / Swan path | Parked | Coverage-sensitive 且 unlabeled；最多只作 M6.4 背景 |

ADR 0022/0023/0024 仍是 active guardrails：electricity 是第一個 labeled supervised-evaluation meter，raw BDG2 是 primary scoring surface，value-change semantics 是未來 multi-meter guardrail。

## Slice tracker

| Slice | Issue | Status | ADR |
| --- | --- | --- | --- |
| P0 docs-only supervised pivot | Not opened | In progress | ADR 0025/0026 Proposed |
| M6.1 label bridge + integrity | Not opened | Queued after P0 review | ADR 0026 |
| M6.2 supervised transfer accuracy | Not opened | Queued after M6.1 | ADR 0025 |
| M6.3 GBDT vs TabPFN supervised | Not opened | Queued after M6.2 | TBD |
| M6.4 unlabeled remainder | Not opened | Queued after M6.3 | ADR 0020 historical context |
| M6.5 milestone close-out | Not opened | Queued after M6.4 | TBD |
