"""GPUtw 單 worker benchmark 與相容性 sentinel。

一個 process 一個 state,與 E6 正式 runner 同樣的語義:fresh reload、
契約驗證、scaler 精確重現、float32 全程、no-fit guard、20,000 列 microbatch。

不同之處只有兩點,而且都是為了讓這支腳本不可能污染科學結果:
輸入固定是 200,000 列 even-building probe,永遠不碰 holdout;輸出只保留
timing 與 digest,score 向量在每個 microbatch 結束後立刻丟棄。

microbatch 上限刻意維持 20,000。正式 E6 已凍結 516-microbatch 的 canonical
batched pass,若 GPUtw 後續分擔 state,必須保持相同的批次語義 —— 用更大的
batch 量出更漂亮的數字,等於偷偷換掉推論過程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_ROWS = 200_000
MICROBATCH = 20_000
FEATURES = 137
SENTINEL_ROWS = 352
SENTINEL_REPEATS = 8
REQUIRED_N_ESTIMATORS = 8
FULL_HOLDOUT_ROWS = 10_137_155

REPRESENTATIVE_STATES = (
    ("seed42", "00", "cell_specific"),
    ("seed42", "01", "cell_specific"),
    ("seed42", "11", "frozen_reference"),
)
CELL_DIR = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}

# E4/E5 已知的重複推論變異範圍。sentinel 只跟這個比,不要求跨 GPU bit-exact。
E4_E5_REFERENCE = {
    "steam_auc_half_width_range": [0.000415, 0.009790],
    "steam_margin_half_width_range": [0.000334, 0.006440],
    "distinct_digests_expected": "8/8 (重複推論不是 bitwise 可重現的)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def peak_rss_gb() -> float:
    """VmHWM,核心自己的高水位;取樣只會低估,而且永遠往同一方向低估。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    import psutil

    return psutil.Process().memory_info().rss / 1e9


def environment() -> dict:
    import torch

    import tabpfn

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "tabpfn": tabpfn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cpu_count": os.cpu_count(),
    }


def verify_probe(probe_npz: Path, guard: Path) -> np.ndarray:
    """只讀 even-building probe,且必須符合筆電端寫下的交集證明。"""
    g = json.loads(guard.read_text(encoding="utf-8"))
    if g["gputw_may_score_holdout"] is not False:
        raise SystemExit("probe_guard 未宣告禁止 holdout scoring")
    if g["disjoint_proof"]["intersection_size"] != 0:
        raise SystemExit("probe_guard 記錄的交集不是 0")
    got = sha256_file(probe_npz)
    if got != g["probe_npz_sha256"]:
        raise SystemExit(
            f"HARD FAILURE probe digest {got[:16]} 與 guard 記錄不符;"
            "無法繼承交集證明,拒絕執行"
        )
    with np.load(probe_npz) as z:
        x = np.asarray(z["x"])
        raw = np.asarray(z["raw_index"], dtype="int64")
    if (
        hashlib.sha256(raw.tobytes()).hexdigest()
        != g["disjoint_proof"]["probe_raw_index_sha256"]
    ):
        raise SystemExit("probe raw_index digest 與 guard 記錄不符")
    if x.shape != (PROBE_ROWS, FEATURES):
        raise SystemExit(f"probe 形狀 {x.shape}")
    if x.dtype != np.float32:
        raise SystemExit(f"probe dtype {x.dtype};全程必須是 float32")
    return x


def load_state(spec: dict, repo_root: Path, cache_root: Path, device: str):
    """reload 一個 persisted E4 state,驗證契約與 scaler,不做任何 fit。"""
    import joblib
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler

    seed_tag, cell, arm = spec["seed_tag"], spec["cell"], spec["arm"]
    state_path = repo_root / spec["state_path"]
    state_sha = sha256_file(state_path)
    if state_sha != spec["state_sha256"]:
        raise SystemExit(f"{spec['unit_id']}: state digest 漂移")

    with np.load(cache_root / f"{seed_tag}__cell{cell}.npz") as z:
        raw_ctx = np.asarray(z["x"])
    if raw_ctx.dtype != np.float32:
        raise SystemExit(f"context dtype {raw_ctx.dtype};E4 縮放的是 float32")

    if arm == "frozen_reference":
        scaler = joblib.load(repo_root / spec["scaler_path"])
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(raw_ctx)
    scaler_check = verify_scaler(scaler, raw_ctx, state_path)

    t0 = time.perf_counter()
    model = load_fitted_tabpfn_model(state_path, device=device)
    reload_seconds = time.perf_counter() - t0
    ensemble = verify_ensemble(model)
    if ensemble["effective_n_estimators_"] != REQUIRED_N_ESTIMATORS:
        raise SystemExit("effective n_estimators_ 不是 8")
    del torch
    return (
        model,
        scaler,
        {
            "state_sha256": state_sha,
            "reload_seconds": reload_seconds,
            "ensemble": ensemble,
            "scaler_verification": scaler_check,
        },
    )


def run_sentinel(model, scaler, sentinel_x: np.ndarray) -> dict:
    """352 列 sentinel,同 process 8 次重複。不要求跨 GPU bit-exact。"""
    q = scaler.transform(sentinel_x).astype("float32")
    if q.dtype != np.float32:
        raise SystemExit("sentinel query 被上轉型")
    runs = []
    for r in range(SENTINEL_REPEATS):
        t0 = time.perf_counter()
        out = model.predict_proba(q)[:, 1].astype("float64")
        if not np.all(np.isfinite(out)):
            raise SystemExit(f"sentinel repeat {r} 出現 non-finite")
        runs.append(
            {
                "repeat": r,
                "seconds": time.perf_counter() - t0,
                "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
                "mean": float(out.mean()),
                "min": float(out.min()),
                "max": float(out.max()),
            }
        )
    means = np.array([r["mean"] for r in runs])
    n = means.size
    half = (
        float(2.364 * means.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    )  # two-sided 95% Student-t, df=7
    digests = {r["sha256"] for r in runs}
    return {
        "repeats": SENTINEL_REPEATS,
        "rows": int(sentinel_x.shape[0]),
        "distinct_digests": len(digests),
        "bitwise_identical": len(digests) == 1,
        "endpoint_mean": float(means.mean()),
        "endpoint_half_width": half,
        "all_finite": True,
        "e4_e5_reference": E4_E5_REFERENCE,
        "runs": runs,
    }


def run_probe(model, scaler, x: np.ndarray, label: str) -> dict:
    """200,000 列,10 個 20,000 列 microbatch。不保存任何 score 向量。"""
    import torch

    t_scale = time.perf_counter()
    q = scaler.transform(x).astype("float32")
    scale_seconds = time.perf_counter() - t_scale
    if q.dtype != np.float32:
        raise SystemExit("probe query 被上轉型")

    per_batch = []
    t_all = time.perf_counter()
    for start in range(0, PROBE_ROWS, MICROBATCH):
        chunk = q[start : start + MICROBATCH]
        t0 = time.perf_counter()
        out = model.predict_proba(chunk)[:, 1]
        dt = time.perf_counter() - t0
        if out.shape[0] != chunk.shape[0] or not np.all(np.isfinite(out)):
            raise SystemExit(f"{label}: microbatch {start} 輸出無效")
        per_batch.append(
            {
                "rows": int(chunk.shape[0]),
                "seconds": dt,
                "rows_per_second": chunk.shape[0] / dt,
            }
        )
        del out  # 不保存 scientific score
    wall = time.perf_counter() - t_all
    rates = [b["rows_per_second"] for b in per_batch]
    tail = rates[1:] or rates
    return {
        "label": label,
        "scale_seconds": scale_seconds,
        "microbatches": len(per_batch),
        "microbatch_rows": MICROBATCH,
        "first_batch_rows_per_second": rates[0],
        "median_rows_per_second": statistics.median(rates),
        "p05_rows_per_second": float(np.percentile(rates, 5)),
        "p95_rows_per_second": float(np.percentile(rates, 95)),
        "p95_batch_seconds": float(
            np.percentile([b["seconds"] for b in per_batch], 95)
        ),
        "sustained_rows_per_second": min(tail),
        "aggregate_rows_per_second": PROBE_ROWS / wall,
        "wall_seconds": wall,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_rss_gb": peak_rss_gb(),
        "projected_state_hours": FULL_HOLDOUT_ROWS / min(tail) / 3600,
        "per_batch": per_batch,
        "scores_retained": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sentinel", "single", "worker"), required=True)
    ap.add_argument("--unit-id", help="worker 模式:要跑的單一 state")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--round", default="")
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--probe-npz", type=Path, required=True)
    ap.add_argument("--probe-guard", type=Path, required=True)
    ap.add_argument("--sentinel-npz", type=Path)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run_uuid = str(uuid.uuid4())

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用;禁止 CPU fallback")
    env = environment()

    x = verify_probe(args.probe_npz, args.probe_guard)
    specs = json.loads(args.state_manifest.read_text(encoding="utf-8"))["states"]
    by_id = {s["unit_id"]: s for s in specs}

    if args.mode == "worker":
        if not args.unit_id:
            raise SystemExit("worker 模式需要 --unit-id")
        targets = [by_id[args.unit_id]]
    else:
        targets = [by_id[f"{s}__cell{c}__{a}"] for s, c, a in REPRESENTATIVE_STATES]

    sentinel_x = None
    if args.mode == "sentinel":
        if not args.sentinel_npz:
            raise SystemExit("sentinel 模式需要 --sentinel-npz")
        with np.load(args.sentinel_npz) as z:
            sentinel_x = np.asarray(z["x"])
        if sentinel_x.shape[0] != SENTINEL_ROWS or sentinel_x.dtype != np.float32:
            raise SystemExit(f"sentinel 矩陣 {sentinel_x.shape} {sentinel_x.dtype}")

    results = []
    for spec in targets:
        torch.cuda.reset_peak_memory_stats()
        model, scaler, meta = load_state(
            spec, args.repo_root, args.cache_root, args.device
        )
        rec = {
            "unit_id": spec["unit_id"],
            "process_uuid": run_uuid,
            "worker_id": args.worker_id,
            "round": args.round,
            "pid": os.getpid(),
            "gpu_uuid": torch.cuda.get_device_properties(0).uuid.hex
            if hasattr(torch.cuda.get_device_properties(0), "uuid")
            else None,
            **meta,
        }
        if args.mode == "sentinel":
            rec["sentinel"] = run_sentinel(model, scaler, sentinel_x)
        else:
            rec["probe"] = run_probe(model, scaler, x, spec["unit_id"])
        results.append(rec)
        print(
            f"  {spec['unit_id']:<40} "
            + (
                f"sentinel digests={rec['sentinel']['distinct_digests']}/8 "
                f"half_width={rec['sentinel']['endpoint_half_width']:.6f}"
                if args.mode == "sentinel"
                else f"{rec['probe']['sustained_rows_per_second']:>8,.0f} r/s "
                f"state={rec['probe']['projected_state_hours']:.2f} h "
                f"VRAM={rec['probe']['peak_vram_gb']:.2f} GB"
            ),
            flush=True,
        )
        del model, scaler
        torch.cuda.empty_cache()

    import psutil

    swap = psutil.swap_memory()
    payload = {
        "schema": f"m5_e6_gputw_{args.mode}_v1",
        "generated": time.time(),
        "scientific": False,
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "no_fit_guard_blocked": blocked,
        "environment": env,
        "swap_used_bytes": int(swap.used),
        "process_uuid": run_uuid,
        "worker_id": args.worker_id,
        "round": args.round,
        "results": results,
    }
    name = (
        f"{args.mode}_{args.round}_w{args.worker_id}.json"
        if args.mode == "worker"
        else f"{args.mode}_results.json"
    )
    digest = atomic_json(args.out / name, payload)
    print(f"{name} sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
