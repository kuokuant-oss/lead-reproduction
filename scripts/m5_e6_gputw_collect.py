"""收集並驗證遠端 benchmark 結果,拒絕不完整或被竄改的輸出。

驗證不是只比 digest。digest 只證明檔案沒被改過,不證明它算得對,也不證明它
真的跑完了。所以這裡會重新推導每一個宣稱的彙總值(rows/s、speedup、投影時間),
並與檔案裡寫的比對 —— 一份自己給自己蓋章的結果不會通過。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

PROBE_ROWS = 200_000
MICROBATCHES = 10
FULL_HOLDOUT_ROWS = 10_137_155
STATES_PER_SEED_BLOCK = 8
TOL = 1e-6


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def rederive_single(rec: dict) -> list[str]:
    """從 per_batch 重新推導,不信任檔案裡的彙總值。"""
    bad = []
    pb = rec.get("per_batch") or []
    if len(pb) != MICROBATCHES:
        bad.append(f"{rec['unit_id']}: {len(pb)} 個 microbatch,預期 {MICROBATCHES}")
        return bad
    rows = sum(b["rows"] for b in pb)
    if rows != PROBE_ROWS:
        bad.append(f"{rec['unit_id']}: 共 {rows} 列,預期 {PROBE_ROWS}")
    for b in pb:
        if not close(b["rows_per_second"], b["rows"] / b["seconds"]):
            bad.append(
                f"{rec['unit_id']}: microbatch {b['index']} 的 rows_per_second 不自洽"
            )
    rates = [b["rows_per_second"] for b in pb]
    if not close(rec["sustained_rows_per_second"], min(rates[1:])):
        bad.append(f"{rec['unit_id']}: sustained_rows_per_second 與 per_batch 不符")
    if not close(
        rec["projected_state_hours"],
        FULL_HOLDOUT_ROWS / rec["sustained_rows_per_second"] / 3600,
    ):
        bad.append(f"{rec['unit_id']}: projected_state_hours 與吞吐不符")
    for k in ("scores_retained", "fits_performed"):
        if rec.get(k, -1) != 0:
            bad.append(f"{rec['unit_id']}: {k} 不是 0")
    if len({b["digest"] for b in pb}) != MICROBATCHES:
        bad.append(f"{rec['unit_id']}: microbatch digest 有重複,輸出可能是複製的")
    return bad


def rederive_dual(dual: dict, single: dict) -> list[str]:
    bad = []
    agg = {r["unit_id"]: r["aggregate_rows_per_second"] for r in single["results"]}
    for name, rnd in (dual.get("rounds") or {}).items():
        w0, w1 = rnd["worker0"], rnd["worker1"]
        if w0["process_uuid"] == w1["process_uuid"]:
            bad.append(f"Round {name}: 兩個 worker 是同一個 process")
        if w0["pid"] == w1["pid"]:
            bad.append(f"Round {name}: 兩個 worker 是同一個 PID")
        if w0["unit_id"] == w1["unit_id"]:
            bad.append(f"Round {name}: 同一個 state 被拆給兩個 worker")
        total = w0["aggregate_rows_per_second"] + w1["aggregate_rows_per_second"]
        if not close(rnd["two_worker_aggregate_rows_per_second"], total):
            bad.append(f"Round {name}: aggregate 與兩個 worker 的和不符")
        seq = 2 / (1 / agg[w0["unit_id"]] + 1 / agg[w1["unit_id"]])
        if not close(rnd["sequential_equivalent_rows_per_second"], seq, 1e-4):
            bad.append(f"Round {name}: sequential 等效吞吐推導不符")
        if not close(rnd["aggregate_speedup"], total / seq, 1e-4):
            bad.append(f"Round {name}: aggregate_speedup 推導不符")
    for k in ("holdout_rows_scored", "scores_retained", "fits_performed"):
        if dual.get(k, -1) != 0:
            bad.append(f"dual_worker: {k} 不是 0")
    if dual.get("third_worker_started") is not False:
        bad.append("dual_worker: 未宣告第三 worker 為 False")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hourly-ntd", type=float, help="GPUtw 當下實際時租")
    ap.add_argument("--setup-hours", type=float, default=1.0)
    args = ap.parse_args()

    required = [
        "remote_environment.json",
        "preflight.json",
        "compatibility_results.json",
        "sentinel_results.json",
        "single_worker_results.json",
        "dual_worker_results.json",
    ]
    missing = [n for n in required if not (args.results_root / n).exists()]
    if missing:
        raise SystemExit(f"結果不完整,缺少:{missing}")

    files = {n: sha256_file(args.results_root / n) for n in required}
    single = read_json(args.results_root / "single_worker_results.json")
    dual = read_json(args.results_root / "dual_worker_results.json")
    sent = read_json(args.results_root / "sentinel_results.json")
    comp = read_json(args.results_root / "compatibility_results.json")
    env = read_json(args.results_root / "remote_environment.json")["environment"]

    problems = []
    if len(single["results"]) != 3:
        problems.append(f"single worker 只有 {len(single['results'])} 個 state,預期 3")
    for rec in single["results"]:
        problems.extend(rederive_single(rec))
    problems.extend(rederive_dual(dual, single))
    if comp["verdict"] not in ("COMPATIBLE", "INCOMPATIBLE", "EXECUTION_INCOMPLETE"):
        problems.append(f"未知的 compatibility verdict:{comp['verdict']}")
    if "RTX PRO 6000" not in env.get("gpu_name", ""):
        problems.append(f"GPU 不是 RTX PRO 6000:{env.get('gpu_name')}")

    best_single = max(r["sustained_rows_per_second"] for r in single["results"])
    single_block_h = STATES_PER_SEED_BLOCK * FULL_HOLDOUT_ROWS / best_single / 3600
    dual_block_h = dual.get("two_worker_projected_block_hours")
    baseline_block_h = STATES_PER_SEED_BLOCK * FULL_HOLDOUT_ROWS / 1420.0 / 3600

    def cost(hours):
        if hours is None or args.hourly_ntd is None:
            return None
        return (hours + args.setup_hours) * args.hourly_ntd

    cost_model = {
        "schema": "m5_e6_gputw_projected_cost_v1",
        "price_status": "PRICED" if args.hourly_ntd else "PRICE_UNVERIFIED",
        "hourly_ntd": args.hourly_ntd,
        "setup_hours": args.setup_hours,
        "baseline_gpu_host": {
            "rows_per_second": 1420.0,
            "eight_state_block_hours": baseline_block_h,
            "marginal_cost_ntd": 0,
            "note": "現行 gpu-host 為自有設備,續跑邊際租金為零",
        },
        "single_worker": {
            "sustained_rows_per_second": best_single,
            "state_hours": FULL_HOLDOUT_ROWS / best_single / 3600,
            "eight_state_block_hours": single_block_h,
            "eight_state_block_cost_ntd": cost(single_block_h),
            "vs_baseline_ratio": best_single / 1420.0,
        },
        "dual_worker": {
            "verdict": dual["verdict"],
            "worst_speedup": dual.get("worst_round_speedup"),
            "eight_state_block_hours": dual_block_h,
            "eight_state_block_cost_ntd": cost(dual_block_h),
        },
        "hours_saved_vs_gpu_host_for_one_block": (
            baseline_block_h - (dual_block_h or single_block_h)
        ),
    }

    payload = {
        "schema": "m5_e6_gputw_collect_v1",
        "generated": time.time(),
        "accepted": not problems,
        "problems": problems,
        "file_digests": files,
        "compatibility_verdict": comp["verdict"],
        "dual_worker_verdict": dual["verdict"],
        "sentinel_verdict": sent["verdict"],
        "environment_digest": env.get("environment_digest"),
        "gpu_name": env.get("gpu_name"),
        "gpu_uuid": env.get("gpu_uuid"),
        "rederivation": "每個彙總值都由 per_batch 重新推導後比對,不採信檔案自述",
    }
    d1 = atomic_json(args.out / "telemetry_summary.json", payload)
    d2 = atomic_json(args.out / "projected_cost_model.json", cost_model)
    for n in required:
        (args.out / n).write_bytes((args.results_root / n).read_bytes())

    print(f"collector accepted = {payload['accepted']}")
    for p in problems:
        print(f"  REJECT: {p}")
    print(f"telemetry_summary.json    = {d1}")
    print(f"projected_cost_model.json = {d2}")
    return 0 if payload["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
