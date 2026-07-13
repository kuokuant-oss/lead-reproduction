# Handoff：M5 fit／predict 拆分計時重跑計畫

**狀態**：計畫已記錄，runner 尚未修改，正式模型尚未重跑。

## 目的

M5 與 M5.1 現有 `fit_predict_seconds` 合併模型 fit、模型初始化與多組 scoring-set
predictions。下一輪把研究執行等待成本與部署推論延遲拆開記錄，並固定 CPU
parallelism，使同一 cell 內的模型成本可重現比較。

## Runner 修改範圍

+ `scripts/run_m6_phaseD_50_50_full_models.py`
+ `scripts/run_m5_phaseD_deep_comparison.py`
+ 共用 timing／resource helper 與對應 regression tests

每個 model cell 新增：

+ `model_init_seconds`
+ `fit_seconds`
+ fit-set、train、val、test 各自的 `predict_seconds`
+ 每組 prediction 的 row count 與 `predict_us_per_row`
+ `predict_total_seconds`
+ 保留 `fit_predict_seconds` 作為相容欄位

TabPFN 每段 GPU 計時前後執行 `torch.cuda.synchronize()`。Tree models 的計時使用
同步 CPU call，不增加額外同步操作。

## CPU／GPU contract

+ 正式命令加入 `--cpu-threads 24`。
+ LightGBM 設定 `n_jobs=24`。
+ XGBoost 設定 `n_jobs=24`。
+ CatBoost 設定 `thread_count=24`。
+ HistGBT 透過 thread-pool／OpenMP limit 固定 24 threads。
+ TabPFN 固定 `device=cuda:0`，記錄 GPU device、driver 與 VRAM。
+ JSON 同時記錄 requested 與 effective thread settings。

## 正式重跑順序

先寫入 side-by-side outputs，通過 metric regression gate 後再取代 canonical files。

```powershell
$env:TABPFN_MODEL_CACHE_DIR="$PWD\.tabpfn-cache"
$env:TABPFN_NO_BROWSER="1"
$env:TABPFN_DISABLE_TELEMETRY="1"
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src;scripts"

.\.venv\Scripts\python.exe scripts/run_m6_phaseD_50_50_full_models.py `
  --out data/processed/m6_phaseD_50_50_full_models_timestamp_merge_timing_split.json `
  --fit-rows 10000 --score-rows 4000 `
  --scarcity-sizes 200 500 1000 2000 5000 10000 `
  --seed 42 --value-change-regime timestamp_merge `
  --cpu-threads 24 `
  --model-path .tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt

.\.venv\Scripts\python.exe scripts/run_m5_phaseD_deep_comparison.py `
  --out data/processed/m5_phaseD_deep_comparison_timing_split.json `
  --handoff docs/handoffs/m5-phaseD-deep-comparison-timing-split.md `
  --fit-rows 10000 --score-rows 4000 `
  --scarcity-sizes 20 50 100 150 300 500 1000 2000 `
  --tune-trials 12 --seed 42 --cpu-threads 24 `
  --model-path .tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt
```

## 驗收條件

+ Tree-model prediction `max_abs_diff=0`、classification mismatch count `=0`。
+ TabPFN 指標依既有 stochastic variation 報 mean／std，不以單次 bitwise equality
  作 gate。
+ 每個 completed cell 都有 fit、分組 predict、row count 與每列 latency。
+ `fit_predict_seconds` 等於 model init、fit 與各 prediction segments 的加總，容許
  timer resolution 造成的微小誤差。
+ 同一 test/query scoring rows 下並列 GBDT 與 TabPFN `predict_us_per_row`。
+ Ensemble 另記 probability combination time；完整 Ensemble cost 同時列出四個基模型
  dependency cost。
+ 報告分開呈現「fit」、「predict」、「µs/row」與「total」四個欄位。

## 報告更新規則

現有報告保留合併等待時間並標明其計時邊界。新 outputs 通過 gate 後，表格改列：

| Model | Fit (s) | Test predict (s) | Test µs/row | Total (s) |
|---|---:|---:|---:|---:|

M6.3／Phase E 的部署 latency 僅在 GBDT 與 TabPFN 使用相同 query rows 的 paired
measurement 下引用，兩個模型的每列時間並列呈現。
