"""GPUtw 成本與完成時間模型。

價格是必填的外部輸入,不是這裡的預設值。GPUtw 的價格頁由前端在執行時載入,
靜態抓取取不到數字,而本輪明令成本必須用當下 API 或控制台的實際價格。因此
沒有給價格時,模型仍會產出完整的時間結構,但每一個金額欄位都是 null,並在
產物裡標記 PRICE_UNVERIFIED。用一個猜來的時租算出「省下多少錢」比不算更糟。

吞吐同樣是外部輸入。沒有量到之前,單 worker 與兩 worker 的 rows/s 都是 null,
時間欄位也跟著是 null;規格只能排除候選,不能替代實測。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

FULL_HOLDOUT_ROWS = 10_137_155
BASELINE_ROWS_PER_SECOND = 1420.0  # 現行 gpu-host 實測
BASELINE_STATE_HOURS = FULL_HOLDOUT_ROWS / BASELINE_ROWS_PER_SECOND / 3600
STATES_PER_SEED_BLOCK = 8
FEATURE_MATRIX_GB = 5.56
PROBE_MATRIX_GB = 0.111
STATE_ARTIFACT_GB = 0.35  # 每個 persisted E4 state 約略大小

# 一次性 overhead,以現行專案的實測與經驗為準,標明來源。
SETUP_OVERHEAD_HOURS = {
    "instance_start": 0.10,
    "environment_build": 1.00,
    "state_transfer_8_states": 0.30,
    "probe_transfer": 0.10,
    "sentinel": 0.05,
    "preflight": 0.10,
    "archive_and_download": 0.25,
}
FEATURE_MATRIX_STRATEGY = {
    "transfer": {
        "size_gb": FEATURE_MATRIX_GB,
        "note": "本輪不得實際傳輸,只估算。若連線無法續傳,5.56 GB 是高風險路徑。",
    },
    "rebuild_on_device": {
        "measured_on_gpu_host_seconds": 256,
        "note": "gpu-host 上重建 hoist 階段約 246 s、寫入約 10 s,且與筆電"
        "產生逐位元相同的 digest。若 GPUtw 端能重建出同一個 digest,"
        "就不必傳 5.56 GB。DGX Spark 為 aarch64,重建是否得到同一 digest "
        "未經證實,必須實測。",
        "recommended": True,
    },
}


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def state_hours(rows_per_second: float | None) -> float | None:
    if not rows_per_second:
        return None
    return FULL_HOLDOUT_ROWS / rows_per_second / 3600


def device_model(
    name: str,
    single_rps: float | None,
    two_worker_aggregate_rps: float | None,
    hourly_ntd: float | None,
) -> dict:
    one = state_hours(single_rps)
    setup = sum(SETUP_OVERHEAD_HOURS.values())

    single_block = one * STATES_PER_SEED_BLOCK if one else None
    # 兩 worker:8 個 state 由兩個 slot 平行消化,合計吞吐決定牆鐘時間。
    two_block = (
        STATES_PER_SEED_BLOCK * FULL_HOLDOUT_ROWS / two_worker_aggregate_rps / 3600
        if two_worker_aggregate_rps
        else None
    )
    two_pair_wall = (
        2 * FULL_HOLDOUT_ROWS / two_worker_aggregate_rps / 3600
        if two_worker_aggregate_rps
        else None
    )

    def cost(hours: float | None) -> float | None:
        if hours is None or hourly_ntd is None:
            return None
        return (hours + setup) * hourly_ntd

    # 損益兩平:要跑幾個 state,節省的時間才抵得過一次性 setup。
    breakeven = None
    if one is not None:
        saving_per_state = BASELINE_STATE_HOURS - one
        if saving_per_state > 0:
            breakeven = setup / saving_per_state
        else:
            breakeven = "never (單 state 不比現行 gpu-host 快)"

    return {
        "device": name,
        "single_worker_rows_per_second": single_rps,
        "single_worker_state_hours": one,
        "single_worker_vs_baseline_ratio": (
            single_rps / BASELINE_ROWS_PER_SECOND if single_rps else None
        ),
        "two_worker_aggregate_rows_per_second": two_worker_aggregate_rps,
        "two_worker_pair_wall_hours": two_pair_wall,
        "setup_overhead_hours": setup,
        "setup_overhead_breakdown": SETUP_OVERHEAD_HOURS,
        "eight_state_block_hours_single_worker": single_block,
        "eight_state_block_hours_two_worker": two_block,
        "sixteen_state_hours_single_worker": one * 16 if one else None,
        "sixteen_state_hours_two_worker": (
            16 * FULL_HOLDOUT_ROWS / two_worker_aggregate_rps / 3600
            if two_worker_aggregate_rps
            else None
        ),
        "hourly_ntd": hourly_ntd,
        "eight_state_block_cost_ntd_single_worker": cost(single_block),
        "eight_state_block_cost_ntd_two_worker": cost(two_block),
        "breakeven_state_count": breakeven,
        "hours_saved_vs_gpu_host_alone_for_8_states": (
            STATES_PER_SEED_BLOCK * BASELINE_STATE_HOURS - (two_block or single_block)
            if (two_block or single_block)
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--remaining-states", type=int, required=True)
    ap.add_argument("--dgx-single-rps", type=float)
    ap.add_argument("--dgx-two-worker-rps", type=float)
    ap.add_argument("--dgx-hourly-ntd", type=float)
    ap.add_argument("--rtx-single-rps", type=float)
    ap.add_argument("--rtx-two-worker-rps", type=float)
    ap.add_argument("--rtx-hourly-ntd", type=float)
    args = ap.parse_args()

    devices = {
        "dgx_spark_gb10": device_model(
            "NVIDIA DGX Spark (GB10)",
            args.dgx_single_rps,
            args.dgx_two_worker_rps,
            args.dgx_hourly_ntd,
        ),
        "rtx_pro_6000_ws": device_model(
            "NVIDIA RTX PRO 6000 Blackwell WS",
            args.rtx_single_rps,
            args.rtx_two_worker_rps,
            args.rtx_hourly_ntd,
        ),
    }
    gpu_host_alone = args.remaining_states * BASELINE_STATE_HOURS
    priced = any(d["hourly_ntd"] is not None for d in devices.values())
    measured = any(d["single_worker_rows_per_second"] for d in devices.values())

    payload = {
        "schema": "m5_e6_gputw_cost_model_v1",
        "generated": time.time(),
        "price_status": "PRICED" if priced else "PRICE_UNVERIFIED",
        "throughput_status": "MEASURED" if measured else "THROUGHPUT_UNMEASURED",
        "price_source_requirement": (
            "必須使用 GPUtw 當下 API 或控制台的實際價格,不得只用文件快照。"
            "GPUtw 的價格由前端執行時載入,靜態抓取取不到,因此在人類提供"
            "價格之前,所有金額欄位保持 null。"
        ),
        "baseline": {
            "host": "gpu-host RTX 5070 Ti",
            "measured_rows_per_second": BASELINE_ROWS_PER_SECOND,
            "state_hours": BASELINE_STATE_HOURS,
            "remaining_states": args.remaining_states,
            "gpu_host_alone_hours": gpu_host_alone,
            "gpu_host_alone_days": gpu_host_alone / 24,
            "marginal_cost_ntd": 0,
            "note": "現行 gpu-host 是自有設備,續跑的邊際租金為零。"
            "GPUtw 要划算,必須用節省的時間換得足以抵過租金與 setup 的價值。",
        },
        "devices": devices,
        "feature_matrix_strategy": FEATURE_MATRIX_STRATEGY,
        "transfer_sizes_gb": {
            "feature_matrix": FEATURE_MATRIX_GB,
            "probe_matrix": PROBE_MATRIX_GB,
            "one_state": STATE_ARTIFACT_GB,
            "eight_states": STATE_ARTIFACT_GB * 8,
        },
        "full_feature_matrix_transferred_this_round": False,
    }
    digest = atomic_json(args.out / "cost_model.json", payload)
    print(f"cost_model.json sha256 = {digest}")
    print(f"  price status      = {payload['price_status']}")
    print(f"  throughput status = {payload['throughput_status']}")
    print(
        f"  現行 gpu-host 獨自跑完剩下 {args.remaining_states} 個 state:"
        f"{gpu_host_alone:.1f} h ({gpu_host_alone / 24:.2f} 天)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
