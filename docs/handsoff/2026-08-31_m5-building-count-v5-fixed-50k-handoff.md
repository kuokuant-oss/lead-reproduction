# M5 Building-Count V5 fixed-50K 正式實驗交接

最後唯讀核對：2026-08-31 09:33:08（Asia/Taipei）

## 一頁摘要

- 實驗名稱：m5_building_count_v5_fixed_50k
- 正式排程正在 WSL Ubuntu 正常執行；不要停止、重啟、resume、重排或重跑任何 cell。
- 排程進度：84 個模型 cell 中 71 個完成、13 個待處理；沒有 FAILED.json。
- 目前 cell：K=100、building seed=1、row seed=1、TabPFN。其 prediction chunk 心跳為 76 / 206（已完成 1,520,000 個 holdout rows）。
- 已完整可比較：K=400 與 K=200（各 5 個 building seed、各 seed 兩個 row seed）。
- K=100 尚未完整：b0、b3、b4 皆完成兩個 row seed；b1 僅 r0 完成、r1 正在跑；b2 尚未有 TabPFN 結果。
- K=50：全部 Tree cell 已完成，全部 TabPFN cell 尚待處理。
- Google Colab：0 個 active session；本正式實驗目前沒有使用 Colab。
- 目前原始碼身分：Git HEAD 4397050376b135ffd1f14d856b6e696767d2588f（4397050, Freeze V5 50K contexts and launch gate），tracked worktree 乾淨。

## 位置與執行環境

WSL 發行版 Ubuntu，使用者 kuant_kuo。

正式程式庫：

    /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k

實際 Python 環境（不可假設系統 python 存在）：

    /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python

每一個 WSL bash 指令都必須先設定：

    export PYTHONPATH=src:scripts

正式輸出根目錄：

    /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed/m5_building_curve/v5_fixed_50k

凍結 context、holdout、census 與 gate：

    /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/experiments/m5_building_count_v5_fixed_50k/audit

V4 的既有 nested building ladders：

    /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/experiments/m5_building_count_v4_fixed_10k/audit/building_ladder_seed{0..4}.json

最重要的狀態檔：

    data/processed/m5_building_curve/v5_fixed_50k/supervisor/status.json
    data/processed/m5_building_curve/v5_fixed_50k/supervisor/heartbeat.json
    data/processed/m5_building_curve/v5_fixed_50k/supervisor/events.jsonl
    experiments/m5_building_count_v5_fixed_50k/audit/heartbeat.json

目前的 tmux session 是 m5-building-v5-fixed50k；正式 scheduler 的命令為：

    /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python \
      scripts/run_m5_building_count_v5.py --mode formal --authorize-formal

它正在呼叫 run_m5_building_count_v5_tabpfn_cell.py，且目標輸出為：

    data/processed/m5_building_curve/v5_fixed_50k/model_runs/building_seed1/row_seed1/tabpfn_k100_f137

## 實驗設計（不可漂移）

- 使用既有且已驗證的五組 V4 nested building ladders；K 為 50、100、200、400，building draw seed 為 0–4。不可按 label 重抽、修補或替換 building。
- K=725 是含所有 725 個偶數 training buildings 的唯一 canonical full support；路徑使用 building_seed=725 僅為作業 sentinel，科學 building_draw_seed 必須為 null。
- 每個 context 固定恰好 50,000 筆唯一 training rows：25,000 anomaly 加 25,000 normal，均為不放回隨機抽樣。
- row draw seeds 是 0 與 1；V5 用新的、已記錄的 PCG64 seed domain，不能把 V4 10K context 延伸、混用或重建。
- 特徵數 137；model seed 42。
- 訓練資料自然可含四種 meter；固定 holdout 預先移除 electricity，評估只保留 chilled water、steam、hot water（meter IDs 1、2、3）。
- 同一 context 的 Tree 與 TabPFN 必須收到完全相同且順序相同的 training rows、labels、137-feature matrix，以及相同的 non-electric holdout。
- TabPFN 設定：8 estimators，sample subsampling disabled/null，模型檔案位於：

      /home/kuant_kuo/.cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt

- 成對順序為 Tree 後 TabPFN。全實驗是 42 matched contexts、84 model cells。scheduler 會對每個 pair 與每個 phase 寫 atomic gate，遇第一個失敗即阻斷，絕不可自動 retry。

## 正式排程與目前完成格

正式順序的設計是：

1. Phase 1：K=725 full support，r0/r1（2 contexts、4 cells）。
2. Phase 2：building seeds 0、1、2，依 K=400、200、100、50（每個 K 再依 seed、r0、r1）。
3. Phase 3：building seeds 3、4，同樣的 budget-major 次序。

截至本文件快照，模型 cell 的完成狀態：

| Budget | Tree | TabPFN | 狀態 |
| --- | ---: | ---: | --- |
| K=725 | 2/2 | 2/2 | 完成 |
| K=400 | 10/10 | 10/10 | 完成 |
| K=200 | 10/10 | 10/10 | 完成 |
| K=100 | 10/10 | 7/10 | b1/r1 正在跑；b2/r0、b2/r1 待處理 |
| K=50 | 10/10 | 0/10 | 待 TabPFN |

因此剩餘 13 個模型 cell：K100 的 3 個 TabPFN，以及 K50 的 10 個 TabPFN。

排程的 status.json 當時記錄 completed=71、pending=13、mean_completed_unit_seconds=23060.266057458837、eta_seconds=299783.45874696487。這是 scheduler 當下的原始估計，不應作為承諾完成時間。

## 檢查點、provenance 與安全操作

先閱讀並遵守：

    AGENTS.md
    docs/policies/long-running-research-execution.md
    docs/plans/m5-building-count-experiment-v5-fixed-50k-plan.md

核心原則是「NO CHECKPOINT, NO LAUNCH」。每個 cell 的 COMPLETE.json 代表該 cell 已完成、寫入、digest/schema/provenance 驗證並原子化發布；partial prediction chunks、tmp、corrupt 或不相容結果都不能視為完成。

對已完成 pair 做任何比較前，必須確認 Tree/TabPFN 的 cell metadata 一致，至少包括：

- context_row_sha256
- context_label_sha256
- context_feature_matrix_sha256
- holdout_row_sha256
- holdout_rows
- predictions.npz 的 validation_raw_index、anomaly、meter（逐元素完全一致）

預測欄位名稱不同：

- Tree 使用 ensemble。
- TabPFN 使用 tabpfn。

不得在 active scheduler 存在時執行 launch script、另開 model cell、手動 resume、kill/restart tmux、改動 experiment script、修改 frozen manifest、刪除 checkpoint，或為求快而重排 queue。可做的只有唯讀監控與對已存在 COMPLETE.json 的分析。

正式輸出 data/processed/... 已被 .gitignore 排除；不要嘗試把大型 output 或 checkpoints 加入 Git。

本次文件建立時，程式 Git worktree 已乾淨且 formal run 的 provenance 以目前 HEAD 啟動。不要在 run 中途為「記錄進度」建立空白 commit：這會改變 HEAD，可能讓後續 cell 的來源身分與既有結果不一致。若需要新的程式提交，應先取得明確授權，並在正式排程安全終止、provenance 影響已評估後處理。

## 唯讀監控範例

下列命令只讀資料，不會影響排程：

    wsl.exe -d Ubuntu -u kuant_kuo -- bash -lc \
      'export PYTHONPATH=src:scripts; cd /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k; \
       sed -n "1,220p" data/processed/m5_building_curve/v5_fixed_50k/supervisor/status.json'

    wsl.exe -d Ubuntu -u kuant_kuo -- bash -lc \
      'export PYTHONPATH=src:scripts; cd /home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k; \
       sed -n "1,260p" data/processed/m5_building_curve/v5_fixed_50k/model_runs/building_seed1/row_seed1/tabpfn_k100_f137/heartbeat.json'

檢查沒有失敗標記：

    find data/processed/m5_building_curve/v5_fixed_50k -name FAILED.json -type f -print

Colab 只可查詢 session；目前不得建立或使用 session：

    HOME=/home/kuant_kuo/.colab-tony OAUTHLIB_RELAX_TOKEN_SCOPE=1 \
      /home/kuant_kuo/.local/bin/colab sessions

## 報表規則與已知結果

使用者目前要求報告只呈現三個 non-electric meter；不要再列 pooled 指標。

聚合順序固定為：

1. 同一 building seed 內先平均其可用的 row seeds。
2. 再跨 building seeds 平均。
3. 標準誤在 building-seed 層級計算。

使用者已明確允許：若某 building seed 目前只有一個完整 row seed，可把該單一 row seed 作為該 seed 的暫時估計，但務必標明不完整，並絕不可讀取 active、partial cell。

以下數字的格式是「平均值 ± building-seed SE」，模型差異是 TabPFN 減 Tree。

### K=400（完整，n=5 building seeds）

| Meter | TabPFN PR-AUC | Tree PR-AUC | TabPFN ROC-AUC | Tree ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Chilled water | 0.7474 ± 0.0127 | 0.7692 ± 0.0156 | 0.9783 ± 0.0007 | 0.9792 ± 0.0013 |
| Steam | 0.7732 ± 0.0152 | 0.7438 ± 0.0053 | 0.9813 ± 0.0012 | 0.9650 ± 0.0024 |
| Hot water | 0.7142 ± 0.0183 | 0.7998 ± 0.0096 | 0.9199 ± 0.0009 | 0.9456 ± 0.0040 |

### K=200（完整，n=5 building seeds）

| Meter | TabPFN PR-AUC | Tree PR-AUC | TabPFN ROC-AUC | Tree ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Chilled water | 0.6749 ± 0.0186 | 0.6466 ± 0.0299 | 0.9754 ± 0.0010 | 0.9725 ± 0.0021 |
| Steam | 0.7545 ± 0.0171 | 0.6985 ± 0.0269 | 0.9787 ± 0.0016 | 0.9639 ± 0.0027 |
| Hot water | 0.6860 ± 0.0215 | 0.7232 ± 0.0230 | 0.9122 ± 0.0104 | 0.9328 ± 0.0054 |

### K=100（暫定，n=4 building seeds）

此為 b0、b3、b4 的 r0/r1 平均，加上 b1/r0 的單一完整 row seed；b1/r1 正在執行、b2 尚無 TabPFN COMPLETE.json，因此不可稱為完整比較。

| Meter | TabPFN PR-AUC | Tree PR-AUC | TabPFN ROC-AUC | Tree ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Chilled water | 0.6307 ± 0.0292 | 0.5703 ± 0.0198 | 0.9672 ± 0.0047 | 0.9682 ± 0.0022 |
| Steam | 0.6143 ± 0.0146 | 0.5885 ± 0.0533 | 0.9602 ± 0.0050 | 0.9468 ± 0.0012 |
| Hot water | 0.6075 ± 0.0551 | 0.6602 ± 0.0451 | 0.8936 ± 0.0143 | 0.9268 ± 0.0062 |

## 給下一位 agent 的工作邊界

- 使用繁體中文回覆使用者。
- 先做唯讀健康檢查，再報進度；每次新結果只由兩個同 context 的 COMPLETE.json 產生。
- 不得因 ETA、長時間未完成、GPU 使用率或想加速而中止或重啟 job；若懷疑卡住，只能讀 heartbeat、chunk 增長、process 與 FAILED.json。
- 不得以 partial prediction chunks 計算正式或暫定 metric。
- 不得報 pooled；只報 chilled water、steam、hot water。
- K=200/K=400 已可做完整兩模型比較。K=100 與 K=50 必須等對應 TabPFN COMPLETE.json 出現並通過 matched-identity checks 才可擴充結果。
- 本文件是 snapshot；取得後續進度時，以 supervisor/status.json、active cell heartbeat.json 與 COMPLETE.json 為準。
