"""兩 worker 併發 benchmark 的編排與判定。

兩個 worker 是兩個真正獨立的 OS process,各自持有自己的 CUDA context、自己
reload 的 state、自己的 scaler 物件、自己的輸出檔。刻意用 subprocess 而不是
ProcessPoolExecutor:pool 會把例外吞掉並在背後重用 worker,那正是本專案先前
被咬過的地方,而且「兩個 worker」必須是可從 OS 層面查證的事實,不是程式庫的
內部安排。也刻意不用 threads —— 同一個 process 內的兩條 thread 共用 CUDA
context,量到的不是本稽核要問的東西。

先做單 worker 基準,完全退出並釋放 GPU,再做併發。三輪對調(A/B 互換、
C 換一個 arm)是為了讓「哪個 state 分到哪個 worker」不會被誤讀成效能差異。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e6_gputw_bench import atomic_json  # noqa: E402

FULL_HOLDOUT_ROWS = 10_137_155

# 三輪。A 與 B 互換,所以 worker slot 與 state 的配對不會與結果混淆。
ROUNDS = {
    "A": ("seed42__cell00__cell_specific", "seed42__cell01__cell_specific"),
    "B": ("seed42__cell01__cell_specific", "seed42__cell00__cell_specific"),
    "C": ("seed42__cell11__frozen_reference", "seed42__cell00__cell_specific"),
}

BENEFICIAL_THRESHOLD = 1.60
MARGINAL_THRESHOLD = 1.20


def launch(args, unit_id: str, worker_id: int, round_name: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "m5_e6_gputw_bench.py"),
        "--mode",
        "worker",
        "--unit-id",
        unit_id,
        "--worker-id",
        str(worker_id),
        "--round",
        round_name,
        "--state-manifest",
        str(args.state_manifest),
        "--probe-npz",
        str(args.probe_npz),
        "--probe-guard",
        str(args.probe_guard),
        "--repo-root",
        str(args.repo_root),
        "--cache-root",
        str(args.cache_root),
        "--out",
        str(args.out),
        "--device",
        args.device,
    ]
    env = dict(os.environ, PYTHONPATH=f"src{os.pathsep}scripts")
    return subprocess.Popen(cmd, env=env)


def read_worker(out: Path, round_name: str, worker_id: int) -> dict:
    path = out / f"worker_{round_name}_w{worker_id}.json"
    if not path.exists():
        raise SystemExit(f"缺少 worker 輸出:{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(speedup: float, stable: bool, clean: bool) -> str:
    if not clean:
        return "TWO_WORKERS_HARMFUL"
    if speedup >= BENEFICIAL_THRESHOLD and stable:
        return "TWO_WORKERS_BENEFICIAL"
    if speedup >= MARGINAL_THRESHOLD:
        return "TWO_WORKERS_MARGINAL"
    return "TWO_WORKERS_HARMFUL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--probe-npz", type=Path, required=True)
    ap.add_argument("--probe-guard", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--single-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    single = json.loads(args.single_results.read_text(encoding="utf-8"))
    single_by_unit = {
        r["unit_id"]: r["probe"]["aggregate_rows_per_second"] for r in single["results"]
    }
    single_sustained = {
        r["unit_id"]: r["probe"]["sustained_rows_per_second"] for r in single["results"]
    }
    print("單 worker 基準 (aggregate rows/s):")
    for u, v in single_by_unit.items():
        print(f"  {u:<40} {v:>9,.0f}")

    rounds = {}
    for name, (u0, u1) in ROUNDS.items():
        print(f"\n=== Round {name}: w0={u0}  w1={u1} ===", flush=True)
        t0 = time.perf_counter()
        p0 = launch(args, u0, 0, name)
        p1 = launch(args, u1, 1, name)
        rc0, rc1 = p0.wait(), p1.wait()
        wall = time.perf_counter() - t0
        if rc0 or rc1:
            raise SystemExit(f"Round {name}: worker 退出碼 {rc0}, {rc1}")

        w0 = read_worker(args.out, name, 0)
        w1 = read_worker(args.out, name, 1)
        r0, r1 = w0["results"][0], w1["results"][0]
        if r0["process_uuid"] == r1["process_uuid"]:
            raise SystemExit(
                "兩個 worker 回報同一個 process UUID;不是真正的兩個 process"
            )

        a0 = r0["probe"]["aggregate_rows_per_second"]
        a1 = r1["probe"]["aggregate_rows_per_second"]
        aggregate = a0 + a1
        # 對照組:同樣兩個 state 若依序單獨執行的合計吞吐。
        seq = 2 / (1 / single_by_unit[u0] + 1 / single_by_unit[u1])
        speedup = aggregate / seq
        clean = (
            w0["swap_used_bytes"] == 0
            and w1["swap_used_bytes"] == 0
            and r0["probe"]["scores_retained"] == 0
            and r1["probe"]["scores_retained"] == 0
        )
        stable = r0["probe"]["microbatches"] == 10 and r1["probe"]["microbatches"] == 10
        rounds[name] = {
            "worker0": {
                "unit_id": u0,
                "process_uuid": r0["process_uuid"],
                "aggregate_rows_per_second": a0,
                "sustained_rows_per_second": r0["probe"]["sustained_rows_per_second"],
                "p95_batch_seconds": r0["probe"]["p95_batch_seconds"],
                "peak_vram_gb": r0["probe"]["peak_vram_gb"],
                "peak_rss_gb": r0["probe"]["peak_rss_gb"],
                "slowdown_vs_single": a0 / single_by_unit[u0],
            },
            "worker1": {
                "unit_id": u1,
                "process_uuid": r1["process_uuid"],
                "aggregate_rows_per_second": a1,
                "sustained_rows_per_second": r1["probe"]["sustained_rows_per_second"],
                "p95_batch_seconds": r1["probe"]["p95_batch_seconds"],
                "peak_vram_gb": r1["probe"]["peak_vram_gb"],
                "peak_rss_gb": r1["probe"]["peak_rss_gb"],
                "slowdown_vs_single": a1 / single_by_unit[u1],
            },
            "two_worker_aggregate_rows_per_second": aggregate,
            "sequential_equivalent_rows_per_second": seq,
            "aggregate_speedup": speedup,
            "wall_seconds": wall,
            "swap_used_bytes": w0["swap_used_bytes"] + w1["swap_used_bytes"],
            "distinct_process_uuids": True,
            "clean": clean,
            "stable": stable,
        }
        print(
            f"  w0 {a0:>9,.0f} r/s   w1 {a1:>9,.0f} r/s   "
            f"aggregate {aggregate:>9,.0f}   speedup {speedup:.3f}x"
        )

    speedups = [r["aggregate_speedup"] for r in rounds.values()]
    all_clean = all(r["clean"] for r in rounds.values())
    all_stable = all(r["stable"] for r in rounds.values())
    worst = min(speedups)
    v = verdict(worst, all_stable, all_clean)

    best_single = max(single_sustained.values())
    payload = {
        "schema": "m5_e6_gputw_two_worker_v1",
        "generated": time.time(),
        "scientific": False,
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "environment": single["environment"],
        "thresholds": {
            "beneficial_aggregate_speedup": BENEFICIAL_THRESHOLD,
            "marginal_aggregate_speedup": MARGINAL_THRESHOLD,
            "note": "門檻在看到結果之前就固定,事後不得調整",
        },
        "single_worker_aggregate_rows_per_second": single_by_unit,
        "single_worker_sustained_rows_per_second": single_sustained,
        "rounds": rounds,
        "worst_round_speedup": worst,
        "best_round_speedup": max(speedups),
        "mean_speedup": sum(speedups) / len(speedups),
        "all_rounds_clean": all_clean,
        "all_rounds_stable": all_stable,
        "verdict": v,
        "single_worker_projected_state_hours": FULL_HOLDOUT_ROWS / best_single / 3600,
        "two_worker_projected_pair_hours": (
            2
            * FULL_HOLDOUT_ROWS
            / (rounds["A"]["two_worker_aggregate_rows_per_second"])
            / 3600
        ),
    }
    digest = atomic_json(args.out / "two_worker_results.json", payload)
    print(f"\nverdict = {v}   worst speedup = {worst:.3f}x")
    print(f"two_worker_results.json sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
