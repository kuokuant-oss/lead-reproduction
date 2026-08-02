"""單 worker benchmark,同時作為雙 worker 的 worker 執行體。

一次一個 state,fresh process,200,000 列 non-holdout probe,10 個 20,000 列
microbatch。不保存任何 score 向量 —— 每個 microbatch 的輸出在算完 digest 與
時間後立刻丟棄。

`--worker-id` 與 `--round` 只影響輸出檔名與 telemetry 標記;雙 worker 就是兩個
這支腳本的獨立 process,所以「兩個 worker」是 OS 層面可查證的事實,而不是
程式庫內部的安排。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_ROWS = 200_000
MICROBATCH = 20_000
FEATURES = 137
REQUIRED_N_ESTIMATORS = 8
FULL_HOLDOUT_ROWS = 10_137_155
STATES_PER_SEED_BLOCK = 8


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def peak_rss_gb() -> float:
    """VmHWM,核心自己的高水位。取樣只會低估,而且永遠往同一方向低估。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return -1.0


def gpu_telemetry() -> dict:
    q = (
        "utilization.gpu,utilization.memory,memory.used,power.draw,"
        "temperature.gpu,clocks_throttle_reasons.active"
    )
    try:
        line = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        parts = [p.strip() for p in line.split(",")]
        return {
            "gpu_utilisation_pct": float(parts[0]),
            "memory_controller_utilisation_pct": float(parts[1]),
            "memory_used_mib": float(parts[2]),
            "power_draw_w": float(parts[3]),
            "temperature_c": float(parts[4]),
            "throttle_reasons": parts[5],
        }
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return {}


def system_telemetry() -> dict:
    out = {}
    try:
        import psutil

        out["cpu_percent_per_core"] = psutil.cpu_percent(interval=0.3, percpu=True)
        sw = psutil.swap_memory()
        out["swap_used_bytes"] = int(sw.used)
        vm = psutil.virtual_memory()
        out["ram_available_gb"] = vm.available / 1e9
        io = psutil.disk_io_counters()
        if io:
            out["disk_read_bytes"] = int(io.read_bytes)
    except Exception:  # noqa: BLE001 - telemetry 缺失不該讓 benchmark 失敗
        pass
    return out


def load_unit(bundle: Path, spec: dict, device: str):
    import joblib
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler

    sp = bundle / spec["state_file"]
    if sha256_file(sp) != spec["state_sha256"]:
        raise SystemExit(f"{spec['unit_id']}: state digest 漂移")
    with np.load(bundle / spec["context_file"]) as z:
        raw_ctx = np.asarray(z["x"])
    if raw_ctx.dtype != np.float32:
        raise SystemExit(f"{spec['unit_id']}: context 不是 float32")
    if spec["scaler"]["kind"] == "persisted":
        scaler = joblib.load(bundle / spec["scaler"]["file"])
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(raw_ctx)
    t0 = time.perf_counter()
    model = load_fitted_tabpfn_model(sp, device=device)
    reload_seconds = time.perf_counter() - t0
    sc = verify_scaler(scaler, raw_ctx, sp)
    if not sc["exact"]:
        raise SystemExit(f"{spec['unit_id']}: scaler 未精確重現")
    ens = verify_ensemble(model)
    if ens["effective_n_estimators_"] != REQUIRED_N_ESTIMATORS:
        raise SystemExit(f"{spec['unit_id']}: effective n_estimators_ 不是 8")
    return model, scaler, reload_seconds, sc, ens


def run_probe(model, scaler, x: np.ndarray, unit_id: str) -> dict:
    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    q = scaler.transform(x).astype("float32")
    scale_seconds = time.perf_counter() - t0
    if q.dtype != np.float32:
        raise SystemExit("probe query 被上轉型")

    per_batch, tele = [], []
    t_all = time.perf_counter()
    for start in range(0, PROBE_ROWS, MICROBATCH):
        chunk = q[start : start + MICROBATCH]
        tb = time.perf_counter()
        out = model.predict_proba(chunk)[:, 1]
        dt = time.perf_counter() - tb
        if out.shape[0] != chunk.shape[0]:
            raise SystemExit(f"{unit_id}: microbatch {start} 列數不符")
        if not np.all(np.isfinite(out)):
            raise SystemExit(f"{unit_id}: microbatch {start} 出現 non-finite")
        per_batch.append(
            {
                "index": start // MICROBATCH,
                "rows": int(chunk.shape[0]),
                "seconds": dt,
                "rows_per_second": chunk.shape[0] / dt,
                "digest": hashlib.sha256(
                    np.asarray(out, dtype="float64").tobytes()
                ).hexdigest(),
            }
        )
        del out  # 不保存 scientific score
        if start // MICROBATCH in (2, 6):
            tele.append(gpu_telemetry())
    wall = time.perf_counter() - t_all

    rates = [b["rows_per_second"] for b in per_batch]
    tail = rates[1:] or rates
    return {
        "unit_id": unit_id,
        "scale_seconds": scale_seconds,
        "microbatches": len(per_batch),
        "microbatch_rows": MICROBATCH,
        "first_batch_seconds": per_batch[0]["seconds"],
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
        "gpu_telemetry_samples": tele,
        "system_telemetry": system_telemetry(),
        "projected_state_hours": FULL_HOLDOUT_ROWS / min(tail) / 3600,
        "projected_eight_state_block_hours": STATES_PER_SEED_BLOCK
        * FULL_HOLDOUT_ROWS
        / min(tail)
        / 3600,
        "per_batch": per_batch,
        "scores_retained": 0,
        "fits_performed": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unit-id", help="只跑這一個 state(雙 worker 模式)")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--round", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.worker_id not in (0, 1):
        raise SystemExit(
            f"worker-id 只能是 0 或 1,收到 {args.worker_id};禁止第三 worker"
        )

    run_uuid = str(uuid.uuid4())

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用;禁止 CPU fallback")

    sm = read_json(args.bundle_root / "state_manifest.json")
    pm = read_json(args.bundle_root / "probe_manifest.json")
    probe_npz = args.bundle_root / pm["probe"]["path"]
    got = sha256_file(probe_npz)
    if got != pm["probe"]["sha256"]:
        raise SystemExit("probe 檔案 digest 與 bundle manifest 不符;拒絕執行")
    with np.load(probe_npz) as z:
        x = np.asarray(z["x"])
        raw = np.asarray(z["raw_index"])
    # 內容權威值是 array-level digest,見 probe_manifest 的 npz_difference_cause。
    if (
        hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        != pm["probe"]["x_sha256"]
    ):
        raise SystemExit("probe x 內容 digest 不符;拒絕執行")
    if (
        hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        != pm["probe"]["existing_artifact_raw_index_sha256"]
    ):
        raise SystemExit("probe raw_index 與既有 audit artifact 不符;拒絕執行")
    if x.shape != (PROBE_ROWS, FEATURES) or x.dtype != np.float32:
        raise SystemExit(f"probe {x.shape} {x.dtype}")

    specs = sm["states"]
    if args.unit_id:
        specs = [s for s in specs if s["unit_id"] == args.unit_id]
        if len(specs) != 1:
            raise SystemExit(f"找不到 state {args.unit_id}")

    gpu_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    gpu_uuid = gpu_uuid.hex if gpu_uuid is not None else ""

    results = []
    for spec in specs:
        model, scaler, reload_seconds, sc, ens = load_unit(
            args.bundle_root, spec, args.device
        )
        rec = run_probe(model, scaler, x, spec["unit_id"])
        rec.update(
            {
                "process_uuid": run_uuid,
                "pid": os.getpid(),
                "worker_id": args.worker_id,
                "round": args.round,
                "gpu_uuid": gpu_uuid,
                "reload_seconds": reload_seconds,
                "state_sha256": spec["state_sha256"],
                "scaler_verification": sc,
                "ensemble": ens,
            }
        )
        results.append(rec)
        print(
            f"  {spec['unit_id']:<40} sustained={rec['sustained_rows_per_second']:>8,.0f} r/s "
            f"aggregate={rec['aggregate_rows_per_second']:>8,.0f} r/s "
            f"state={rec['projected_state_hours']:.2f} h "
            f"VRAM={rec['peak_vram_gb']:.2f} GB RSS={rec['peak_rss_gb']:.2f} GB",
            flush=True,
        )
        del model, scaler
        torch.cuda.empty_cache()

    payload = {
        "schema": "m5_e6_gputw_single_worker_v1",
        "generated": time.time(),
        "process_uuid": run_uuid,
        "worker_id": args.worker_id,
        "round": args.round,
        "gpu_uuid": gpu_uuid,
        "no_fit_guard_blocked": blocked,
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "results": results,
    }
    name = (
        f"worker_{args.round}_w{args.worker_id}.json"
        if args.unit_id
        else "single_worker_results.json"
    )
    digest = atomic_json(args.out / name, payload)
    print(f"{name} sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
