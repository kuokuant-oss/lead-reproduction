"""雙 worker 併發 benchmark:三輪對調,兩個真正獨立的 CUDA process。

用 subprocess 啟動兩個 `m5_e6_gputw_single_worker.py`,各自持有自己的 CUDA
context、自己 reload 的 state、自己的 scaler 物件、自己的輸出檔。刻意不用
threads(共用 CUDA context,量到的不是這裡要問的東西),也不用
ProcessPoolExecutor(會吞例外並在背後重用 worker,而且「兩個 worker」必須是
OS 層面可查證的事實)。

speedup 的分母不是「單 worker 吞吐 × 2」,而是同樣兩個 state 依序單獨執行的
等效合計吞吐(調和平均),因為兩個 state 的單 worker 速度不一定相同。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e6_gputw_single_worker import atomic_json, read_json  # noqa: E402

FULL_HOLDOUT_ROWS = 10_137_155
STATES_PER_SEED_BLOCK = 8
BENEFICIAL = 1.60
MARGINAL = 1.20
STALL_LIMIT_SECONDS = 600  # 任一輪僵死超過 10 分鐘就停止


def launch(
    script: Path, args, unit_id: str, worker_id: int, rnd: str
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--bundle-root",
        str(args.bundle_root),
        "--out",
        str(args.out),
        "--unit-id",
        unit_id,
        "--worker-id",
        str(worker_id),
        "--round",
        rnd,
        "--device",
        args.device,
    ]
    return subprocess.Popen(cmd, env=dict(os.environ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--single-results", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    plan = read_json(args.bundle_root / "benchmark_plan.json")
    rounds_plan = plan["dual_worker_rounds"]
    if plan["max_workers"] != 2:
        raise SystemExit("benchmark plan 的 max_workers 不是 2")

    single = read_json(args.single_results)
    single_aggregate = {
        r["unit_id"]: r["aggregate_rows_per_second"] for r in single["results"]
    }
    single_sustained = {
        r["unit_id"]: r["sustained_rows_per_second"] for r in single["results"]
    }
    print("單 worker 基準:")
    for u, v in single_aggregate.items():
        print(
            f"  {u:<40} aggregate={v:>9,.0f} r/s  sustained={single_sustained[u]:>9,.0f} r/s"
        )

    script = Path(__file__).resolve().parent / "m5_e6_gputw_single_worker.py"
    rounds, hard_stop = {}, None

    for name in ("A", "B", "C"):
        u0, u1 = rounds_plan[name]
        print(f"\n=== Round {name}: w0={u0}  w1={u1} ===", flush=True)
        t0 = time.perf_counter()
        p0 = launch(script, args, u0, 0, name)
        p1 = launch(script, args, u1, 1, name)
        try:
            rc0 = p0.wait(timeout=STALL_LIMIT_SECONDS)
            rc1 = p1.wait(timeout=STALL_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            p0.kill()
            p1.kill()
            hard_stop = f"Round {name}: worker 僵死超過 {STALL_LIMIT_SECONDS}s,已終止"
            print(hard_stop)
            break
        wall = time.perf_counter() - t0
        if rc0 or rc1:
            hard_stop = f"Round {name}: worker 退出碼 {rc0}, {rc1}"
            print(hard_stop)
            break

        w0 = read_json(args.out / f"worker_{name}_w0.json")
        w1 = read_json(args.out / f"worker_{name}_w1.json")
        r0, r1 = w0["results"][0], w1["results"][0]
        if r0["process_uuid"] == r1["process_uuid"]:
            raise SystemExit("兩個 worker 回報同一個 process UUID;不是兩個 process")
        if r0["pid"] == r1["pid"]:
            raise SystemExit("兩個 worker 回報同一個 PID")

        a0 = r0["aggregate_rows_per_second"]
        a1 = r1["aggregate_rows_per_second"]
        aggregate = a0 + a1
        sequential = 2 / (1 / single_aggregate[u0] + 1 / single_aggregate[u1])
        speedup = aggregate / sequential
        swap = r0["system_telemetry"].get("swap_used_bytes", 0) + r1[
            "system_telemetry"
        ].get("swap_used_bytes", 0)
        stable = r0["microbatches"] == 10 and r1["microbatches"] == 10
        clean = (
            swap == 0
            and r0["scores_retained"] == 0
            and r1["scores_retained"] == 0
            and r0["fits_performed"] == 0
            and r1["fits_performed"] == 0
        )

        rounds[name] = {
            "worker0": {
                "unit_id": u0,
                "pid": r0["pid"],
                "process_uuid": r0["process_uuid"],
                "aggregate_rows_per_second": a0,
                "sustained_rows_per_second": r0["sustained_rows_per_second"],
                "p95_batch_seconds": r0["p95_batch_seconds"],
                "peak_vram_gb": r0["peak_vram_gb"],
                "peak_rss_gb": r0["peak_rss_gb"],
                "slowdown_vs_single": a0 / single_aggregate[u0],
                "gpu_telemetry_samples": r0["gpu_telemetry_samples"],
                "system_telemetry": r0["system_telemetry"],
            },
            "worker1": {
                "unit_id": u1,
                "pid": r1["pid"],
                "process_uuid": r1["process_uuid"],
                "aggregate_rows_per_second": a1,
                "sustained_rows_per_second": r1["sustained_rows_per_second"],
                "p95_batch_seconds": r1["p95_batch_seconds"],
                "peak_vram_gb": r1["peak_vram_gb"],
                "peak_rss_gb": r1["peak_rss_gb"],
                "slowdown_vs_single": a1 / single_aggregate[u1],
                "gpu_telemetry_samples": r1["gpu_telemetry_samples"],
                "system_telemetry": r1["system_telemetry"],
            },
            "two_worker_aggregate_rows_per_second": aggregate,
            "sequential_equivalent_rows_per_second": sequential,
            "aggregate_speedup": speedup,
            "wall_seconds": wall,
            "combined_peak_vram_gb": r0["peak_vram_gb"] + r1["peak_vram_gb"],
            "combined_peak_rss_gb": r0["peak_rss_gb"] + r1["peak_rss_gb"],
            "swap_used_bytes": swap,
            "distinct_process_uuids": True,
            "distinct_pids": True,
            "stable": stable,
            "clean": clean,
        }
        print(
            f"  w0 {a0:>9,.0f}   w1 {a1:>9,.0f}   aggregate {aggregate:>9,.0f}   "
            f"speedup {speedup:.3f}x   VRAM {rounds[name]['combined_peak_vram_gb']:.2f} GB"
        )

    if not rounds:
        verdict = "EXECUTION_INCOMPLETE"
        worst = None
    else:
        speedups = [r["aggregate_speedup"] for r in rounds.values()]
        worst = min(speedups)
        all_clean = all(r["clean"] for r in rounds.values())
        all_stable = all(r["stable"] for r in rounds.values())
        complete = len(rounds) == 3 and hard_stop is None
        if not complete:
            verdict = "EXECUTION_INCOMPLETE"
        elif not all_clean:
            verdict = "TWO_WORKERS_HARMFUL"
        elif worst >= BENEFICIAL and all_stable:
            verdict = "TWO_WORKERS_BENEFICIAL"
        elif worst >= MARGINAL:
            verdict = "TWO_WORKERS_MARGINAL"
        else:
            verdict = "TWO_WORKERS_HARMFUL"

    best_single = max(single_sustained.values()) if single_sustained else None
    payload = {
        "schema": "m5_e6_gputw_dual_worker_v1",
        "generated": time.time(),
        "verdict": verdict,
        "hard_stop": hard_stop,
        "thresholds": {
            "beneficial_aggregate_speedup": BENEFICIAL,
            "marginal_aggregate_speedup": MARGINAL,
            "fixed_before_measurement": True,
            "capacity_alone_never_justifies_two_workers": True,
        },
        "single_worker_aggregate_rows_per_second": single_aggregate,
        "single_worker_sustained_rows_per_second": single_sustained,
        "rounds": rounds,
        "worst_round_speedup": worst,
        "rounds_completed": len(rounds),
        "single_worker_projected_state_hours": (
            FULL_HOLDOUT_ROWS / best_single / 3600 if best_single else None
        ),
        "single_worker_projected_block_hours": (
            STATES_PER_SEED_BLOCK * FULL_HOLDOUT_ROWS / best_single / 3600
            if best_single
            else None
        ),
        "two_worker_projected_block_hours": (
            STATES_PER_SEED_BLOCK
            * FULL_HOLDOUT_ROWS
            / rounds["A"]["two_worker_aggregate_rows_per_second"]
            / 3600
            if "A" in rounds
            else None
        ),
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "third_worker_started": False,
    }
    digest = atomic_json(args.out / "dual_worker_results.json", payload)
    print(
        f"\nverdict = {verdict}" + (f"   worst speedup = {worst:.3f}x" if worst else "")
    )
    print(f"dual_worker_results.json sha256 = {digest}")
    return 0 if verdict != "EXECUTION_INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
