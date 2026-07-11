# Handoff：M5 模型比較 runtime provenance 與重跑前 gate

**Issue**：[GitHub #58](https://github.com/kuokuant-oss/lead-reproduction/issues/58)

**狀態**：runner 觀測層已完成並通過 gate；正式模型尚未重跑，必須等待人工確認。

## 本 slice 完成內容

新增 `scripts/experiment_observability.py`，由四支既有 runner 共用。變更只增加
output provenance，不改資料、features、split、seed、downsampling、scaler、模型
超參數、模型執行順序或 prediction path。

接入的 runner：

+ `scripts/run_m5_phaseC_tabpfn_spike.py`
+ `scripts/run_m5_phaseD_foundation_vs_gbdt.py`
+ `scripts/run_m6_phaseD_50_50_full_models.py`
+ `scripts/run_m5_phaseD_deep_comparison.py`

每次完成 run 後，runner 仍透過原本的 `write_json_with_provenance(args.out, ...)`
寫到命令指定的同一份模型 JSON。硬體與時間資料不另存 detached artifact，因此不會
和 metrics 分離。

## 新增 output 欄位

`environment` 現在額外包含：

+ OS system/release/version/machine
+ CPU model、logical cores、processor architecture
+ total RAM bytes/GiB
+ Python version、implementation、executable
+ CatBoost、LightGBM、NumPy、pandas、scikit-learn、SciPy、TabPFN、Torch、XGBoost 版本
+ OMP/MKL/OpenBLAS/NumExpr thread-control environment variables
+ NVIDIA driver version

既有 GPU、VRAM、CUDA、Torch、TabPFN 與 local-checkpoint 欄位保持不變。

頂層新增 `timing_protocol`，定義 `time.perf_counter`、秒為單位、整體
`elapsed_seconds` 的涵蓋範圍、model-local `fit_predict_seconds` 的語義、JSON write
是否包含在內，以及未新增 CUDA synchronization。既有計時欄位名稱與行為保持不變。

## 已驗證 output binding

四支 runner 都把 `environment`、`elapsed_seconds` 與 `timing_protocol` 放入 results，
再把同一個 results 寫到 `args.out`。`tests/test_experiment_observability.py` 對四支 runner
做 regression gate，避免未來發生「模型跑完但 observability 沒寫進指定 JSON」。

正式重跑預定 canonical outputs：

| Runner | Output |
| --- | --- |
| Phase C spike | `data/processed/m5_phaseC_tabpfn_spike_timestamp_merge.json` |
| Phase D multi-seed | `data/processed/m6_phaseD_timestamp_merge_multiseed.json` |
| Six-model 50/50 | `data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json` |
| M5.1 deep comparison | `data/processed/m5_phaseD_deep_comparison.json` |

正式命令必須明確傳入以上 `--out`，不能依賴 default output 名稱。

## 本機環境快照

+ OS：Windows 11，build `10.0.26200`
+ CPU：13th Gen Intel Core i9-13980HX，32 logical cores
+ RAM：31.628 GiB
+ GPU：NVIDIA GeForce RTX 4070 Laptop GPU，約 8 GiB VRAM
+ NVIDIA driver：566.07
+ Python：CPython 3.13.13，repo `.venv`
+ TabPFN：8.0.8
+ Torch：2.7.1+cu128

正式 JSON 會在 run 當下重新讀取環境，不以本段靜態快照代替 provenance。

## 已通過 gate

+ `python -m unittest discover tests`：80 tests passed（新增 output-binding test 後需重跑）
+ Ruff：passed
+ markdownlint-cli2：passed
+ `pre-commit run --all-files`：passed
+ `git diff --check`：passed

## 重跑前準備與 stop rule

下一階段只允許：

1. 驗證四個 raw M3 inputs、local TabPFN checkpoint、CUDA/GPU、磁碟空間與 output parent。
2. 設定 `TABPFN_MODEL_CACHE_DIR`、`TABPFN_NO_BROWSER=1`、
   `TABPFN_DISABLE_TELEMETRY=1`，並使用 repo `.venv`。
3. 盤點並停止可明確認定為本 repo 遺留的模型 runner；不得任意終止其他使用者程序。
4. 記錄就緒狀態後停止。

在人工明確確認前，不得啟動任何正式模型 command，不得縮減 budget、skip TabPFN、改
seed、改 feature regime 或改 output path。

## 正式跑後驗收

每個 output 必須確認：

+ `provenance.command`、commit、generated timestamp 對應本次 run。
+ `environment` 具有 CPU/RAM/OS/dependencies/GPU/driver。
+ `timing_protocol` 與 `elapsed_seconds` 存在。
+ completed model cells 保有對應的 fit/predict timing。
+ canonical metrics 沒有無法解釋的 drift；若 drift，保留結果並先診斷，不直接更新報告。
