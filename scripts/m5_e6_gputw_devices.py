"""GPUtw.ai 候選設備清查:規格、相容性風險與兩 worker 可行性的先驗判斷。

本模組只記錄可查證的設備事實,不執行任何 benchmark,也不接觸現行 E6。

每一筆數值都標註來源類別:
  vendor_spec   廠商或規格表數字
  third_party   第三方實測
  measured      本專案自己在現行 gpu-host 上量到的
  unverified    尚未取得(例如需要登入控制台的即時價格)

刻意不把「規格推導的預期」寫成 benchmark 結果。設備能否跑兩個 worker
是實測問題;規格只能排除明顯不可行的候選,不能確認可行的候選。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# 現行 E6 正式執行機器,作為一切比較的基準。
BASELINE = {
    "role": "現行 E6 正式執行機器,本輪絕對不得干擾",
    "host": "gpu-host",
    "gpu": "NVIDIA GeForce RTX 5070 Ti",
    "gpu_source": "measured (torch.cuda.get_device_name on the live run)",
    "architecture": "Blackwell GB203",
    "cpu_architecture": "x86_64",
    "compute_capability": "12.0 (sm_120)",
    "vram_gb": 16,
    "vram_type": "GDDR7",
    "memory_bandwidth_gb_s": 896,
    "memory_bandwidth_source": "vendor_spec (256-bit GDDR7 @ 1.75 GHz)",
    "cuda_cores": 8960,
    "torch": "2.12.1",
    "cuda_runtime": "13.0",
    "tabpfn": "8.0.8",
    "python": "3.12.13",
    "wsl_memory_ceiling_gb": 24,
    "measured_sustained_rows_per_second": 1414,
    "measured_median_rows_per_second": 1422,
    "measured_live_run_rows_per_second": 1420,
    "measured_source": "e6_throughput_probe.json (afe80b11...b46279) 與現行 run log",
    "measured_peak_vram_gb": 1.87,
    "measured_peak_rss_gb": 3.25,
    "state_hours": 1.99,
    "workers": 1,
}

CANDIDATES = {
    "dgx_spark_gb10": {
        "display_name": "NVIDIA DGX Spark (GB10 Grace Blackwell Superchip), 4TB",
        "soc": "GB10 Grace Blackwell Superchip",
        "gpu": "Blackwell GPU integrated in GB10 SoC",
        "gpu_source": "vendor_spec",
        "cpu": "20-core Arm (10x Cortex-X925 + 10x Cortex-A725)",
        "cpu_architecture": "aarch64",
        "cpu_architecture_source": "vendor_spec",
        "cpu_cores": 20,
        "compute_capability": "12.1 (sm_121)",
        "compute_capability_source": "vendor_spec / CUDA 13.0 release notes",
        "cuda_cores": 6144,
        "streaming_multiprocessors": 48,
        "max_clock_ghz": 2.42,
        "memory_gb": 128,
        "memory_type": "LPDDR5x",
        "memory_model": "coherent unified system memory",
        "memory_bandwidth_gb_s": 273,
        "memory_bandwidth_source": "third_party (multiple independent reviews)",
        "usable_memory_note": "廠牌標示 128 GB,實測可見約 122 GB",
        "storage": "4 TB NVMe SSD",
        "MEMORY_MODEL_WARNING": (
            "128 GB 是 CPU 與 GPU 透過 NVLink-C2C 以 ATS 共享的一致性統一系統"
            "記憶體 (coherent unified system memory),不是 128 GB 獨立 VRAM 的"
            "同義詞。沒有可供搬入的獨立 device VRAM;CPU 與 GPU 競爭同一個 "
            "273 GB/s 記憶體池,而現行機器的 GPU 獨佔 896 GB/s。把它讀成"
            "「等同 128 GB 顯卡」會在兩 worker 判斷上得到完全相反的結論。"
        ),
        "bandwidth_ratio_vs_baseline": 273 / 896,
        "cuda_core_ratio_vs_baseline": 6144 / 8960,
    },
    "rtx_pro_6000_ws": {
        "display_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "gpu": "RTX PRO 6000 Blackwell (GB202)",
        "gpu_source": "vendor_spec",
        "cpu": "取決於 GPUtw 的主機配置,尚未取得",
        "cpu_architecture": "x86_64",
        "cpu_architecture_source": "vendor_spec (PCIe 5.0 x16 workstation card)",
        "cpu_cores": None,
        "compute_capability": "12.0 (sm_120)",
        "compute_capability_source": "vendor_spec",
        "cuda_cores": 24064,
        "tensor_cores": 752,
        "rt_cores": 188,
        "memory_gb": 96,
        "memory_type": "GDDR7 with ECC",
        "memory_model": "dedicated device VRAM, 512-bit bus",
        "memory_bandwidth_gb_s": 1792,
        "memory_bandwidth_source": "vendor_spec (512-bit GDDR7)",
        "fp32_tflops": 125,
        "interconnect": "PCI Express 5.0 x16",
        "storage": "取決於 GPUtw 的主機配置,尚未取得",
        "bandwidth_ratio_vs_baseline": 1792 / 896,
        "cuda_core_ratio_vs_baseline": 24064 / 8960,
    },
}

# 相容性風險,依證據強度排序。ARM64 是本次稽核的決定性項目。
COMPATIBILITY = {
    "dgx_spark_gb10": {
        "tabpfn_8_0_8_package": {
            "status": "COMPATIBLE",
            "evidence": "tabpfn 8.0.8 只發佈 tabpfn-8.0.8-py3-none-any.whl,是"
            "純 Python 套件,沒有平台專屬 wheel,因此套件本身不阻擋 aarch64。",
            "source": "PyPI JSON API",
        },
        "python_3_12": {
            "status": "COMPATIBLE",
            "evidence": "tabpfn 8.0.8 requires_python >=3.10,支援到 3.14。",
            "source": "PyPI JSON API",
        },
        "torch_2_12_1_aarch64_cuda13": {
            "status": "BLOCKING_RISK",
            "evidence": (
                "現行 E6 凍結 torch 2.12.1 + CUDA 13.0。官方 aarch64 + CUDA 13 "
                "+ sm_121 wheel 在公開管道仍未成熟:主要可用途徑是 NVIDIA NGC "
                "容器(PyTorch 2.10)或自行從原始碼編譯。取得與現行 E6 完全相同"
                "的 torch 2.12.1 aarch64 build 未經證實。"
            ),
            "source": "third_party (PyTorch issue tracker, NVIDIA forums)",
            "consequence": (
                "若只能用 torch 2.10,GPUtw 上跑的 state 與 gpu-host 上跑的 state "
                "就處在不同的 runtime。E6 是確認階段,環境連續性不是可選項目。"
            ),
        },
        "joblib_pickle_cross_architecture": {
            "status": "UNVERIFIED_RISK",
            "evidence": (
                "persisted E4 state 是 zip 內含 joblib pickle(init_params.json、"
                "fitted_attrs.joblib、executor_state.joblib)。x86_64 與 aarch64 "
                "都是 little-endian,numpy 陣列通常可跨架構反序列化,但 pickle "
                "內若含架構相依物件則不保證。"
            ),
            "consequence": "必須實測 reload,不能因為「都是 little-endian」就假定成立。",
        },
        "sklearn_scipy_numpy_pandas_aarch64": {
            "status": "LIKELY_COMPATIBLE",
            "evidence": "這些套件都提供 manylinux aarch64 wheel。",
            "consequence": "版本必須逐一釘到與現行環境相同,否則樹側與 scaler 行為可能位移。",
        },
        "overall": "ARM64_TOOLCHAIN_RISK",
    },
    "rtx_pro_6000_ws": {
        "tabpfn_8_0_8_package": {
            "status": "COMPATIBLE",
            "evidence": "同上,純 Python。",
        },
        "python_3_12": {"status": "COMPATIBLE", "evidence": "同上。"},
        "torch_2_12_1_x86_64_cuda13": {
            "status": "LIKELY_COMPATIBLE",
            "evidence": (
                "x86_64 + CUDA 13.0 是 PyTorch 的主線發佈目標,且現行 gpu-host "
                "已在 sm_120 上以 torch 2.12.1 執行同一份 TabPFN 8.0.8。"
                "RTX PRO 6000 同為 sm_120。"
            ),
            "consequence": "與現行 E6 的環境落差最小,但仍須實測確認。",
        },
        "joblib_pickle_cross_architecture": {
            "status": "LOW_RISK",
            "evidence": "同為 x86_64,與現行 gpu-host 相同架構。",
        },
        "overall": "LIKELY_COMPATIBLE_PENDING_MEASUREMENT",
    },
}

# 併發相關,規格層面能說什麼、不能說什麼。
CONCURRENCY = {
    "measured_single_worker_footprint": {
        "peak_vram_gb": 1.87,
        "peak_rss_gb": 3.25,
        "source": "e6_throughput_probe.json,現行 gpu-host,context 20000",
    },
    "capacity_is_not_the_question": (
        "現行單 worker 只用 1.87 GB VRAM。96 GB 可容納約 51 個這樣的 worker,"
        "128 GB 統一記憶體更多。所以「放不放得下」在兩個候選上都不是問題,"
        "而兩 worker 是否划算與容量無關 —— 決定因素是記憶體頻寬、計算單元與 "
        "CPU/IO 是否有閒置餘裕。用容量論證併發效益是本稽核明確拒絕的推論。"
    ),
    "why_bandwidth_dominates": (
        "TabPFN 推論是對 context 做 attention。context 20000 x 137 特徵在每個 "
        "microbatch 都要重新走一遍,屬於記憶體頻寬受限而非算力受限的型態。"
        "第三方 LLM decode 實測支持這個方向:GB10 對 RTX PRO 6000 在 "
        "GPT-OSS 20B 上是 49.7 對 215 tokens/s,約 4.3 倍,與兩者 273 對 "
        "1792 GB/s 的頻寬比同一量級。"
    ),
    "mps_mig": {
        "dgx_spark_gb10": "GB10 為單一 SoC 上的整合 GPU,不支援 MIG;MPS 可用性未經證實。",
        "rtx_pro_6000_ws": "GeForce/RTX PRO 級不支援 MIG;MPS 通常可用,但未經證實。",
        "note": "兩個獨立 CUDA process 不需要 MPS 也能共用一張 GPU,"
        "只是由驅動做時間切片,這正是本稽核要實測的行為。",
    },
    "prior_expectation_not_a_result": (
        "依規格,GB10 的頻寬是現行機器的 0.30 倍、CUDA cores 0.69 倍,因此"
        "單 worker 預期就比現行 gpu-host 慢,兩 worker 更難補回。"
        "RTX PRO 6000 的頻寬 2.0 倍、cores 2.69 倍,是唯一有機會讓兩 worker "
        "產生正效益的候選。這是先驗推論,不是實測結果,不得寫成 verdict。"
    ),
}

PRICING = {
    "vendor": "GPUtw.ai",
    "platform_note": "台灣在地 GPU 雲平台,標示按秒計費、資料留在台灣。",
    "pricing_page_states": "對標市場競爭價格,目標低於市場價 30%,NT$/hr,依即時供應調整",
    "status": "PRICE_UNVERIFIED",
    "reason": (
        "GPUtw 的價格與供應狀態由前端 JavaScript 於執行時載入,靜態抓取"
        "取不到數字。本 prompt 明令成本必須用當下 API 或控制台實際價格,"
        "不得只用文件快照,因此此處不填任何猜測價格。"
    ),
    "required_to_resolve": [
        "人類提供 GPUtw 控制台當下顯示的 NT$/hr(兩個型號各一)",
        "或提供可讀取價格的 API token(不得寫入 repo 或報告)",
    ],
    "dgx_spark_hourly_ntd": None,
    "rtx_pro_6000_hourly_ntd": None,
    "minimum_billing_unit": None,
    "startup_shutdown_rules": None,
    "hardware_reference_only": {
        "dgx_spark_msrp_usd": 3999,
        "dgx_spark_taiwan_leadtek_ntd": 135345,
        "rtx_pro_6000_street_usd": 9000,
        "note": "這是購買價,不是租金,不得用來推導時租。",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def inventory() -> dict:
    return {
        "schema": "m5_e6_gputw_device_inventory_v1",
        "generated": time.time(),
        "purpose": "GPUtw.ai 兩候選設備的規格、相容性與兩 worker 可行性稽核",
        "base_protocol_commit": "3a800a0",
        "current_e6_untouched": True,
        "benchmarks_executed": 0,
        "holdout_rows_scored": 0,
        "fits_performed": 0,
        "baseline_host": BASELINE,
        "candidates": CANDIDATES,
        "compatibility": COMPATIBILITY,
        "concurrency": CONCURRENCY,
        "pricing": PRICING,
        "evidence_grades": {
            "vendor_spec": "廠商規格表",
            "third_party": "第三方公開實測",
            "measured": "本專案在現行 gpu-host 上實測",
            "unverified": "尚未取得",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    digest = atomic_json(args.out / "device_inventory.json", inventory())
    print(f"device_inventory.json sha256 = {digest}")
    for key, c in CANDIDATES.items():
        print(
            f"  {c['display_name']}\n"
            f"    arch={c['cpu_architecture']}  cc={c['compute_capability']}  "
            f"cores={c['cuda_cores']:,}  mem={c['memory_gb']} GB "
            f"{c['memory_type']}  bw={c['memory_bandwidth_gb_s']:,} GB/s "
            f"({c['bandwidth_ratio_vs_baseline']:.2f}x baseline)\n"
            f"    compatibility={COMPATIBILITY[key]['overall']}"
        )
    print(f"  pricing status = {PRICING['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
