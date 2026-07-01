# M4 Report: Importable Pipeline Foundation

**Status**: 完成
**Date**: 2026-06-26
**Task**: 將 M3 reproduction pipeline 從 script/notebook-centered workflow
整理成可匯入的 `src/lead` package，並在不移動 M3 numeric line 的前提下，補上
M5 所需的資料、特徵、切分、抽樣、評估介面。

M4 是 foundation milestone：範圍是把 M3 reproduction pipeline 整理成可匯入的
`src/lead` package，並補上 M5 所需介面；BDG2 ingestion 與 FDD experiments 於 M5
展開。所有變更以 executable code 為 reproduction authority；若 paper、docs、code
不一致，除非 ADR 明確覆寫，否則以目前可執行路徑為準。

## 對應程式碼

+ Public API freeze：[src/lead init](../../src/lead/__init__.py)、[tests/test_public_api.py](../../tests/test_public_api.py)。
+ Data loading 與 label alignment：[src/lead/data.py](../../src/lead/data.py)、[tests/test_label_join_integrity.py](../../tests/test_label_join_integrity.py)。
+ Value-change features：[src/lead/features.py](../../src/lead/features.py)、[scripts/run_m4_3_timestamp_value_change.py](../../scripts/run_m4_3_timestamp_value_change.py)、[tests/test_value_change_regimes.py](../../tests/test_value_change_regimes.py)。
+ Split helpers：[src/lead/split.py](../../src/lead/split.py)、[tests/test_split_helpers.py](../../tests/test_split_helpers.py)。
+ Sampling semantics：[src/lead/sample.py](../../src/lead/sample.py)、[tests/test_sampling_semantics.py](../../tests/test_sampling_semantics.py)。
+ Evaluation / provenance helpers：[src/lead/evaluate.py](../../src/lead/evaluate.py)、[src/lead/io.py](../../src/lead/io.py)。
+ Regression gates：[tests/test_refactor_regression.py](../../tests/test_refactor_regression.py)、[tests/test_call_arity.py](../../tests/test_call_arity.py)、[tests/golden_metrics.json](../../tests/golden_metrics.json)。

---

# Ch1: Milestone Scope

## 1.1 核心目標

M4 的目標是把 M3 已驗證的行為鎖住，再抽出穩定 API。完成後，M5 可以直接 import
資料載入、特徵生成、split、sampling 與 evaluation helper，不需要讀 notebook cells
或複製 M3 script 內部函式。

| Slice | 狀態 | 主要產物 |
| --- | --- | --- |
| M4.0 baseline lock | Done | `tests/golden_metrics.json` |
| M4.1 extract `src/lead` | Done | `src/lead/{data,features,split,sample,evaluate,io}.py` |
| M4.2 label alignment guard | Done | ADR 0010, label integrity tests |
| M4.3 timestamp value-change regime | Done | ADR 0011, `m4_3_timestamp_value_change.json` |
| M4.4 sampling/scaler semantics | Done | ADR 0016, sampling semantics test |
| M4.5 M5 readiness gate | Done | frozen `lead.__all__`, readiness check JSON |

## 1.2 Regression Gate

M4 的 regression noise floor 固定為 AUC `0.0005`。M3.2 與 M3.4 是主要 gate；
50/50、site-held-out、steam meter metrics 作為 diagnostic inventory 保留。

| Result | Golden AUC | Source |
| --- | ---: | --- |
| M3.2 LightGBM 80/20 offline | 0.9920 | `tests/golden_metrics.json` |
| M3.4 4-model ensemble 80/20 offline | 0.9928 | `tests/golden_metrics.json` |
| 50/50 offline ensemble | 0.9921 | `docs/metrics/m3-50-50-ensemble.json` |
| 50/50 causal ensemble | 0.9911 | `docs/metrics/m3-50-50-ensemble.json` |
| Site-held-out ensemble | 0.9774 | `tests/golden_metrics.json` |
| Steam meter | 0.9553 | `tests/golden_metrics.json` |

---

# Ch2: Structural Findings and Resolutions

## 2.1 Helper duplication

M4 前，`load_m3_frame`、`add_value_change_features`、`downsample_indices`
等核心 helper 分散在 tracked M3 scripts。M4.1 將這些行為抽到 `src/lead`，並讓
M3 scripts 改為 import package helper。這是 behavior-preserving refactor；M3.2
與 M3.4 rerun 均維持在 regression gate 內。

## 2.2 Label alignment

M3 anomaly labels 來源為 buds-lab `bad_meter_readings.csv`。該檔案只有
`is_bad_meter_reading`，沒有可用於 true key join 的 row key。M4.2 因此沒有偽造
join key，而是保留 positional semantics，並加入 schema、length、index、row-key
guard。ADR 0010 記錄此決策與限制：若 label-side 缺 key，reorder-plus-index-reset
仍無法在 assignment time 被完全偵測。

## 2.3 Value-change semantics

M3 reproduction line 使用 `groupby("building_id").shift(n)`，其語意是 row offset，
不一定等於 timestamp `n` hours offset。M4.3 新增 explicit regime：
`row_offset` 與 `timestamp_merge`。`row_offset` 保持 default，以維持 M3
reproduction line；`timestamp_merge` 作為 opt-in semantic alternative。

M4.3 同一 harness 量測：

| Regime | AUC | Delta vs row-offset |
| --- | ---: | ---: |
| `row_offset` | 0.9920119520500562 | 0 |
| `timestamp_merge` | 0.9924831086743003 | +0.00047115662424412896 |

差異仍在 `0.0005` noise floor 內，因此 M4 不替換 default。

## 2.4 Sampling and scaler semantics

M4.4 review 後保留兩個 reproduction-compatibility 行為：

+ M3 scripts 在 tree models 前仍使用 `StandardScaler`，以保持原 script path 的
  numeric parity。
+ `downsample_indices` 保持 `[negs1, pos, negs2, pos]`，positive rows 會出現
  兩次，使 fit set 維持有效 50:50。

這些行為沒有改變 executable fit path，因此 expected AUC delta 為 `0`。
`tests/test_sampling_semantics.py` 已機器檢查 positive duplication 與 50:50 ratio。

---

# Ch3: Golden Gate Results

M4.5 重新確認 M3.2 與 M3.4 golden gates。結果存於
`data/processed/m4_5_readiness_check.json`。

| Gate | Rerun AUC | Golden AUC | Delta | 結論 |
| --- | ---: | ---: | ---: | --- |
| M3.2 LightGBM 80/20 offline | 0.9920119520500562 | 0.9920 | +0.000011952050056218688 | Pass |
| M3.4 ensemble 80/20 offline | 0.9927886432126508 | 0.9927886432126508 | +0.0 | Pass |

兩者皆在 `0.0005` AUC gate 內。M4 的 code extraction、label guard、
feature-regime addition、sampling/scaler documentation、API freeze 均未破壞
M3 accepted numeric line。

---

# Ch4: Public API

M4.5 凍結 `lead.__all__`，並由 `tests/test_public_api.py` 檢查 drift。Frozen
surface 如下：

1. `ROOT`
2. `M3`
3. `PROC`
4. `RANDOM_STATE`
5. `DOWNSAMPLE_SEEDS`
6. `MODEL_SEEDS`
7. `SHUFFLE_SEEDS`
8. `BASELINE_FEATURE_COLS`
9. `BUILDING_META_FEATURE_COLS`
10. `CYCLIC_FEATURE_COLS`
11. `M3_3_EXTRA_FEATURE_COLS`
12. `WEATHER_LAG_BASE_COLS`
13. `WEATHER_WINDOWS`
14. `SHIFTS`
15. `PAST_SHIFTS`
16. `FUTURE_SHIFTS`
17. `load_m3_frame`
18. `add_value_change_features`
19. `split_mask`
20. `assert_no_building_overlap`
21. `downsample_indices`
22. `classification_metrics`
23. `write_json_with_provenance`

`add_value_change_features(df, shifts, value_change_regime=...)` 支援
`row_offset` 與 `timestamp_merge`。M3 reproduction default 仍為 `row_offset`；
任何 real-time FDD claim 必須使用 `PAST_SHIFTS`-only causal features。

---

# Ch5: M5 Readiness

M4 完成後，M5 的 entry interfaces 已明確化：

+ Data interface: 以 `load_m3_frame` 風格 loader 產出 tabular frame。BDG2
  ingestion 於 M5 展開。
+ Label interface: BDG2 archive 本身沒有 native per-row anomaly labels；後續
  M6 只允許在 ADR 0025/0026 定義的 GEPIII-overlap、2016、meters 0-3 子集上
  透過 keyed bridge 接上 rank-1 GEPIII annotations。
+ Split interface: 以 `split_mask` 與 `assert_no_building_overlap` 支援
  building-level / site-held-out split。M3 site-held-out AUC `0.9774` 是
  GEPIII internal generalization anchor，不是 BDG2 transfer result。
+ Evaluation interface: 使用 `classification_metrics` 記錄 AUC、precision、
  recall、F1；任何 real-time FDD claim 必須遵守 ADR 0007/0011 的 causal
  feature discipline。

Remaining unknowns 19-21 屬於 M5 的 feasibility/experiment 範圍：TabPFN
row-feature fit、GPU/VRAM/license/local-vs-cloud execution path、以及 in-context
TabPFN real-time latency。M4 readiness gate 已獨立於這些項目完成，並將它們交給 M5
逐項驗證。

---

# Ch6: Conclusion

M4 已完成 importable pipeline foundation。M3 numeric line 維持在 regression gate
內，label alignment、value-change regime、sampling/scaler semantics 與 public API
surface 均已由 ADR 與 tests 鎖定。M5 可直接重用 data、feature、split、sample 與
evaluation helpers；GEPIII model selection、BDG2 supervised-overlap evaluation
與 real-time latency engineering 將在後續里程碑展開。

*Last updated: 2026-06-26 (M4 complete)*
