# M5 E6 — GPUtw.ai 候選設備稽核

本輪不接觸現行 E6。base 為 protocol commit `3a800a0`,在獨立分支
`m5-e6-gputw-concurrency-audit` 與獨立 worktree 上進行。沒有執行任何
benchmark、沒有評分任何 holdout 列、沒有任何 fit。

## 結論先行

| | DGX Spark GB10 | RTX PRO 6000 WS |
|---|---|---|
| 相容性 | **ARM64_TOOLCHAIN_RISK** | LIKELY_COMPATIBLE(待實測) |
| 記憶體頻寬 vs 現行機器 | **0.30×** | 2.00× |
| CUDA cores vs 現行機器 | 0.69× | 2.69× |
| 是否值得進兩 worker 測試 | 先解 ARM64 才有意義 | 是,唯一候選 |

**唯一推薦:RTX PRO 6000 WS。** DGX Spark 不是因為容量不足被排除,而是因為
它在本工作型態上的兩個決定性數字都比現行機器差,而且 ARM64 工具鏈與凍結的
`torch 2.12.1 + CUDA 13.0` 之間有未解的落差。

## 對照基準:現行 gpu-host

一切比較都以正在執行 E6 的機器為基準,這些數字是本專案自己量到的:

| 項目 | 值 |
|---|---|
| GPU | RTX 5070 Ti (Blackwell GB203) |
| 架構 / compute capability | x86_64 / 12.0 (sm_120) |
| VRAM | 16 GB GDDR7 |
| 記憶體頻寬 | 896 GB/s |
| CUDA cores | 8,960 |
| torch / CUDA / TabPFN / Python | 2.12.1 / 13.0 / 8.0.8 / 3.12.13 |
| 實測持續吞吐 | 1,414 rows/s(probe)、1,420 rows/s(正式 run) |
| 單 state 時間 | 1.99 h |
| 單 worker peak VRAM | 1.87 GB |
| 單 worker peak RSS | 3.25 GB |

## 候選一:DGX Spark GB10

| 項目 | 值 | 來源 |
|---|---|---|
| SoC | GB10 Grace Blackwell Superchip | 廠商規格 |
| CPU | 20 核 Arm(10× Cortex-X925 + 10× Cortex-A725) | 廠商規格 |
| **CPU 架構** | **aarch64** | 廠商規格 |
| compute capability | 12.1 (sm_121) | CUDA 13.0 release notes |
| CUDA cores / SM | 6,144 / 48 | 廠商規格 |
| 記憶體 | 128 GB LPDDR5x(實測可見約 122 GB) | 廠商規格 + 第三方 |
| **記憶體頻寬** | **273 GB/s** | 第三方多方實測 |
| 儲存 | 4 TB NVMe | 廠商規格 |

### 128 GB 不是 128 GB VRAM

這是本報告最需要講清楚的一句話。

> **128 GB 是 CPU 與 GPU 透過 NVLink-C2C 以 ATS 共享的一致性統一系統記憶體
> (coherent unified system memory),不是 128 GB 獨立 VRAM 的同義詞。**

沒有可供搬入的獨立 device VRAM;CPU 與 GPU 競爭同一個 273 GB/s 的記憶體池,
而現行機器的 GPU 獨佔 896 GB/s。把它讀成「等同 128 GB 顯卡」,會在兩 worker
的判斷上得到完全相反的結論 —— 因為那會讓人以為容量寬裕就代表可以塞兩個
worker,而容量從來不是這裡的瓶頸。

### ARM64:決定性風險

| 項目 | 狀態 | 依據 |
|---|---|---|
| tabpfn 8.0.8 套件 | **COMPATIBLE** | 只發佈 `tabpfn-8.0.8-py3-none-any.whl`,純 Python,無平台專屬 wheel |
| Python 3.12 | COMPATIBLE | `requires_python >=3.10`,支援到 3.14 |
| **torch 2.12.1 aarch64 + CUDA 13 + sm_121** | **BLOCKING_RISK** | 官方 aarch64 + CUDA 13 wheel 仍未成熟;主要途徑是 NVIDIA NGC 容器(PyTorch 2.10)或自行編譯 |
| joblib pickle 跨架構 reload | UNVERIFIED_RISK | 兩者皆 little-endian,numpy 陣列通常可跨架構,但 pickle 內若含架構相依物件則不保證 |
| sklearn / scipy / numpy / pandas | LIKELY_COMPATIBLE | 皆有 manylinux aarch64 wheel,但版本必須逐一釘死 |

**不能只因為 CUDA 可見就判定相容。** 真正的問題是:現行 E6 凍結 torch 2.12.1
+ CUDA 13.0;若 DGX Spark 上只能拿到 NGC 的 torch 2.10,GPUtw 上跑的 state
與 gpu-host 上跑的 state 就處在不同的 runtime。E6 是確認階段,環境連續性不是
可選項目。

persisted E4 state 是 zip 內含三個 joblib pickle。跨架構 reload 必須實測,
不能因為「都是 little-endian」就假定成立。

## 候選二:RTX PRO 6000 Blackwell WS

| 項目 | 值 | 來源 |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell (GB202) | 廠商規格 |
| CPU 架構 | x86_64 | PCIe 5.0 x16 工作站卡 |
| compute capability | 12.0 (sm_120) | 廠商規格 |
| CUDA cores / Tensor / RT | 24,064 / 752 / 188 | 廠商規格 |
| 記憶體 | 96 GB GDDR7 with ECC,512-bit | 廠商規格 |
| **記憶體頻寬** | **1,792 GB/s** | 廠商規格 |
| FP32 | 125 TFLOPS | 廠商規格 |

sm_120 與現行 gpu-host 相同,x86_64 與現行相同,torch 2.12.1 + CUDA 13.0 是
PyTorch 的主線發佈目標。**與現行 E6 的環境落差最小**,這正是確認階段最需要的
性質。joblib pickle 跨機器 reload 風險低(同架構)。

## 為什麼頻寬是決定性的

TabPFN 推論是對 context 做 attention。context 20,000 × 137 特徵在每個
microbatch 都要重新走一遍,屬於記憶體頻寬受限而非算力受限的型態。

第三方 LLM decode 實測支持這個方向:同一個 GPT-OSS 20B,GB10 得到 49.7
tokens/s,RTX PRO 6000 得到 215 tokens/s,約 **4.3 倍**,與兩者 273 對 1,792
GB/s 的頻寬比同一量級。多份獨立評測都指出 GB10 的 LPDDR5x 頻寬是其推論效能的
主要限制。

依此推論,GB10 的**單 worker 就預期比現行 gpu-host 慢**,兩 worker 更難補回。
RTX PRO 6000 是唯一有機會讓兩 worker 產生正效益的候選。

**這是先驗推論,不是實測結果,因此不寫成 verdict。** 規格能排除候選,不能確認
候選 —— 兩 worker 是否划算是實測問題。

## 容量與效益必須分開講

現行單 worker 只用 1.87 GB VRAM、3.25 GB RSS。96 GB 可容納約 51 個這樣的
worker,128 GB 統一記憶體更多。所以「放不放得下」在兩個候選上都不是問題。

| 面向 | DGX Spark | RTX PRO 6000 |
|---|---|---|
| 容量是否足夠 | 足夠(遠超所需) | 足夠(遠超所需) |
| 計算資源是否足夠 | 6,144 cores,0.69× 現行 | 24,064 cores,2.69× 現行 |
| 記憶體頻寬是否足夠 | **273 GB/s,0.30× 現行** | 1,792 GB/s,2.00× 現行 |
| CPU 與 I/O 是否足夠 | 20 核 Arm;與 GPU 共用同一記憶體池 | 未知,取決於 GPUtw 主機配置 |
| 實際 aggregate throughput 是否提高 | **未實測** | **未實測** |

用「96 GB 或 128 GB 放得下」來論證兩 worker 划算,是本稽核明確拒絕的推論。

## MPS / MIG

+ GB10 為單一 SoC 上的整合 GPU,不支援 MIG;MPS 可用性未經證實。
+ RTX PRO 級不支援 MIG;MPS 通常可用,但未經證實。
+ 兩個獨立 CUDA process 不需要 MPS 也能共用一張 GPU,由驅動做時間切片 ——
  這正是本稽核要實測的行為。

## 價格:未取得

`PRICE_UNVERIFIED`。GPUtw 的價格與供應狀態由前端 JavaScript 於執行時載入,
靜態抓取取不到數字。本輪明令成本必須用當下 API 或控制台的實際價格,不得只用
文件快照,因此**沒有填入任何猜測價格**,成本模型中所有金額欄位維持 `null`。

平台本身標示:台灣在地 GPU 雲、按秒計費、資料留在台灣、對標市場價並以低於
市場 30% 為目標。硬體購買價(DGX Spark 約 US$3,999 / 台灣麗臺版 NT$135,345、
RTX PRO 6000 約 US$9,000)僅供參考,**不得用來推導時租**。

解除此狀態需要:人類提供控制台當下的 NT$/hr(兩個型號各一),或提供可讀取
價格的 API token(不得寫入 repo 或報告)。

## Artifacts

`data/processed/m5_e6_gputw_audit/device_inventory.json`
sha256 `3b793c2b693f6f8592c3ac4d49a800d277217417b285470f55a3ebe07a3ccd98`

## 來源

+ [NVIDIA DGX Spark 產品頁](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
+ [DGX Spark Unpacked: GB10, Unified Memory, sm_121](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists)
+ [NVIDIA DGX Spark In-Depth Review — LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
+ [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
+ [PyTorch: Enable CUDA 13.0 binaries](https://github.com/pytorch/pytorch/issues/159779)
+ [DGX Spark GB10 CUDA 13.0 Python 3.12 SM_121 — PyTorch Forums](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
+ [GPUtw.ai](https://gputw.ai/pricing)
