# M5 Plan: GEPIII 模型比較，銜接 M6 BDG2 評估

**Status**: M5 已完成 GEPIII 上的 FDD 模型比較；M6 BDG2 overlap 監督式評估已提出，尚未實作
**Started**: 2026-06-25
**GitHub Issue**: [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27)

## 定位

M5 是 GEPIII 上的 FDD 模型比較 slice。它在相同 split、seed、downsampling 與 feature table 下比較 GBDT 與 TabPFN，並把比較設計與觀察到的 tradeoffs 帶入 M6。

Observed M5 tradeoffs:

+ M5 evidence 顯示 TabPFN 在 label-scarce PR-AUC 與 GEPIII site-held-out ROC-AUC 有增益。
+ M5 evidence 顯示 GBDT 在 latency 與 raw-feature baseline behavior 上較強。
+ M6.3 會在同一個 labeled BDG2 overlap frame 上比較兩個模型，再做 BDG2 model-role decision。

M6 的標籤邊界由 ADR 0025/0026 定義：BDG2 archive 本身沒有 native per-row
anomaly labels；可監督評估的是 rank-1 GEPIII/Kaggle
`bad_meter_readings` annotations，透過 `building_id_kaggle`、meter code、timestamp
橋接到 BDG2 的 GEPIII-overlap、2016、meters 0-3 子集。BDG2-only、2017、其他
meter 不進入 supervised denominators。

## 固定邊界

+ M3 numeric line 凍結：`load_m3_frame` defaults、M3.2 `0.9920`、M3.4
  `0.9928`、downsample seeds、StandardScaler path、`±0.0005` gate 不動。
+ `lead.__all__` 凍結，除非後續 slice 明確用 additive API + ADR + test 覆蓋。
+ M3 site-held-out ensemble AUC `0.9774` 是 GEPIII internal generalization
  anchor，不是 BDG2 transfer result。
+ 任何 real-time FDD claim 必須使用 `PAST_SHIFTS`-only causal features
  per ADR 0007/0011。
+ 本計畫不下載資料、不接雲端、不新增 metric；執行 slice 才產出 provenance。

## M5 模型比較問題

M5 比較 TabPFN 與既有 GBDT line，使用相同 split、seed、downsample、feature table。
比較面向是：

1. In-domain：`80_20_mod5` split。
2. GEPIII site-held-out：`site_id % 5 == 4`，以 M3 `0.9774` 作內部參考。
3. Label scarcity：縮小 labeled support set，觀察模型在少標籤情境的退化。
4. Minimal feature engineering：比較 raw 17 features 與 137-feature value-change line。

TabPFN 是候選 FDD 工具，不是里程碑本身。GBDT 是既有可部署 baseline。

## Phase C: 本地 TabPFN feasibility

+ Optional dependency group: `m5`。
+ Full M3.2 downsampled training table 為 `4,285,104 x 137`，超過 TabPFN-3
  documented `1,000,000 x 200` limit。
+ Local feasibility table 降到 `1,000 x 137`。
+ TabPFN AUC `0.9904`，GBDT AUC `0.9870`。
+ TabPFN cold fit+predict `6.5070` seconds on RTX 4070 Laptop GPU。
+ Phase C metric audit #32 確認 threshold metrics 來自 TabPFN probabilities；
  `0.5` threshold confusion matrix 與 GBDT anchor 相同，但 AUC 不同。

## Phase D: TabPFN vs GBDT 正式比較

Harness: `scripts/run_m5_phaseD_foundation_vs_gbdt.py`
Result JSON: `data/processed/m5_phaseD_foundation_vs_gbdt.json`
Report: [docs/reports/m5-foundation-vs-gbdt.md](../reports/m5-foundation-vs-gbdt.md)

執行條件：

+ Fit budget: 10,000 balanced rows。
+ Validation: 4,000 fixed validation rows。
+ Seeds: `{42, 123, 999}`，回報 mean ± std。
+ Hardware: RTX 4070 Laptop GPU。
+ `tabpfn==8.0.8` local weights。
+ 不使用 `ignore_pretraining_limits`。

主要結果：

+ **In-domain (`80_20_mod5`)**：TabPFN ROC-AUC `0.9925`，GBDT `0.9877`。
  TabPFN 接近 M3.4 ensemble `0.9928`，但 inference 約 `~100x` slower。
+ **GEPIII site-held-out (`site_id % 5 == 4`)**：在 true site-held-out models
  中，TabPFN-in-context ROC-AUC `0.9833` > GBDT-retrain `0.9797`；
  GBDT 保有 PR-AUC 優勢。
+ **Label scarcity**：TabPFN 最明確的勝場，200 labels 時 PR-AUC 約
  `+0.100`。
+ **Minimal FE**：假設不成立。Raw 17 features 上 GBDT ROC-AUC `0.9587`
  高於 TabPFN `0.9499`，value-change features 仍必要。

結論：M5 只建立 GEPIII 上的比較基準。M6.3 要在同一個 labeled BDG2 overlap frame 上重新比較 GBDT 與 TabPFN 的 supervised accuracy、latency、feature requirements。TabPFN 的 `~6.3 ms/row` latency 與 TabPFN-3.0 research/internal-use license 必須跟著任何後續比較結果出現。

## Phase E / M6: BDG2 正式評估

M6 不再沿用「BDG2 全域 unlabeled transfer」作為主線。主線是：

1. **M6.1 label bridge + integrity**：只建立 keyed bridge 與 coverage gate，
   不回報 accuracy。
2. **M6.2 supervised transfer accuracy**：在 BDG2 raw overlap rows 上回報
   ROC-AUC、PR-AUC、precision、recall、F1；cleaned 只作 companion sensitivity。
3. **M6.3 GBDT vs TabPFN supervised comparison**：在同一 labeled overlap frame
   比 accuracy 與 latency。
4. **M6.4 unlabeled remainder**：BDG2-only、2017、其他 meter 只作 secondary
   pseudo-label / review evidence。
5. **M6.5 close-out**：README、plan、ADR、handoff、provenance、validation、issue、
   CI 狀態收斂。

## 舊 Phase E 設計如何閱讀

+ ADR 0019 / 0020 / 0021 已由 ADR 0025 supersede；0021 對 primary M6 path 為 moot。
+ Chilledwater Step 4、powered gate、Swan contiguity、`underpowered_even_pooled`
  都是舊 unlabeled route 的歷史背景，不是 active M6 headline。
+ BDG2 EDA 的數字仍有用：它說明 coverage、missingness、BDG2-vs-GEPIII
  distribution distance，以及哪些 row 可以或不能進入 M6 supervised scope。

## Issue Tracker Map

| Slice | GitHub issue | Status |
| --- | --- | --- |
| Phase B foundation-model planning | [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27) | Done |
| Phase C LEAD TabPFN feasibility spike | [#30](https://github.com/kuokuant-oss/lead-reproduction/issues/30) | Done |
| Phase C metric audit fix | [#32](https://github.com/kuokuant-oss/lead-reproduction/issues/32) | Done |
| Phase D retired skeleton | [#34](https://github.com/kuokuant-oss/lead-reproduction/issues/34) | Done |
| Phase D TabPFN-vs-GBDT GEPIII comparison | [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35) | Done |
| Phase E real BDG2 inventory / ingestion / assumption isolation | [#39](https://github.com/kuokuant-oss/lead-reproduction/issues/39) | Done as setup |
| BDG2 pre-modeling EDA | [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40) | Done |
| 舊 audit-yield 設計 | [#41](https://github.com/kuokuant-oss/lead-reproduction/issues/41) | 舊路線；已由 ADR 0025 取代 |
| Powered gate 降級 | [#42](https://github.com/kuokuant-oss/lead-reproduction/issues/42) | 歷史背景 |
| Electricity entry meter | [#44](https://github.com/kuokuant-oss/lead-reproduction/issues/44) | 已依 ADR 0025 重新定位 |
| Raw-first scoring | [#45](https://github.com/kuokuant-oss/lead-reproduction/issues/45) | 已依 ADR 0025 重新定位 |
| Value-change convergence | [#48](https://github.com/kuokuant-oss/lead-reproduction/issues/48) | 仍是 guardrail |
| Swan chilledwater off critical path | [#49](https://github.com/kuokuant-oss/lead-reproduction/issues/49) | 歷史背景 |
| M6.1 label bridge + integrity | _not opened_ | Queued |
