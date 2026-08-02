"""Non-scientific throughput probe on even-building (non-holdout) rows.

This exists to answer one question before E6 is frozen: how fast does the
context-20000 configuration actually stream? The design audit could only anchor
on a context-100000 measurement, which bounds the time from above but does not
pin it.

Nothing here touches the 10,137,155-row holdout, reads any existing score
column, or keeps a scientific score. It builds a fixed, digest-pinned 200,000-row
probe matrix from the **even-building** half, runs the same streaming code E6
will run, and records only throughput and resource telemetry.
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_ROWS = 200_000
MICROBATCH = 20_000
FEATURES = 137
SEED = 20260802

PROBE_STATES = (
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


def build_probe_matrix(out: Path) -> dict:
    """A fixed 200,000-row matrix drawn from the even-building (fit) half."""
    from lead import load_m3_frame
    from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix

    npz = out / "e6_probe_matrix.npz"
    meta_path = out / "e6_probe_matrix.json"
    if npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha256_file(npz) == meta["npz_sha256"]:
            print(f"probe matrix already built: {meta['npz_sha256'][:16]}")
            return meta

    frame = load_m3_frame(verbose=False)
    fit_frame = frame.loc[frame["building_id"] % 2 == 0]
    pool = fit_frame.index.to_numpy(dtype="int64")
    rng = np.random.default_rng(SEED)
    take = np.sort(rng.choice(pool.size, size=PROBE_ROWS, replace=False))
    raw = pool[take]

    t0 = time.perf_counter()
    x = build_feature_matrix(fit_frame, raw, "F4", full_frame=fit_frame)
    validate_feature_matrix(x, matrix_name="e6 throughput probe")
    build_seconds = time.perf_counter() - t0
    if x.shape != (PROBE_ROWS, FEATURES) or x.dtype != np.float32:
        raise SystemExit(f"probe matrix {x.shape} {x.dtype}")

    tmp = npz.with_name(f".{npz.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, x=x, raw_index=raw)
    os.replace(tmp, npz)
    meta = {
        "rows": PROBE_ROWS,
        "features": FEATURES,
        "dtype": "float32",
        "source_half": "even buildings (the fit half); the holdout is untouched",
        "selection_seed": SEED,
        "raw_index_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "npz_sha256": sha256_file(npz),
        "build_seconds": build_seconds,
        "all_even_buildings": True,
    }
    atomic_json(meta_path, meta)
    print(f"probe matrix built in {build_seconds:,.0f}s  sha256={meta['npz_sha256']}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--e4-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    args = ap.parse_args()

    import joblib
    import psutil

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import tabpfn
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler

    if tabpfn.__version__ != "8.0.8":
        raise SystemExit(f"TabPFN {tabpfn.__version__} != 8.0.8")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU fallback is prohibited")

    meta = build_probe_matrix(args.out)
    with np.load(args.out / "e6_probe_matrix.npz") as z:
        raw_probe = np.asarray(z["x"])  # stored dtype, never upcast
    if raw_probe.dtype != np.float32:
        raise SystemExit(f"probe dtype {raw_probe.dtype}")

    proc = psutil.Process()
    results = []
    for seed_tag, cell, arm_name in PROBE_STATES:
        uid = f"{seed_tag}__cell{cell}__{arm_name}"
        state = args.e4_root / uid / "model.tabpfn_fit"
        state_sha = sha256_file(state)

        with np.load(args.cache_root / f"{seed_tag}__cell{cell}.npz") as z:
            raw_ctx = np.asarray(z["x"])
        if arm_name == "frozen_reference":
            sp = (
                args.repo_root
                / "data/processed/m5_hotwater_label_factorial/scalers"
                / f"{seed_tag}_pooled_reference.joblib"
            )
            scaler = joblib.load(sp)
        else:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(raw_ctx)
        scaler_check = verify_scaler(scaler, raw_ctx, state)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = load_fitted_tabpfn_model(state, device="cuda:0")
        cold_reload = time.perf_counter() - t0
        ensemble = verify_ensemble(model)

        q = scaler.transform(raw_probe).astype("float32")
        per_batch = []
        for start in range(0, PROBE_ROWS, MICROBATCH):
            chunk = q[start : start + MICROBATCH]
            tb = time.perf_counter()
            out = model.predict_proba(chunk)[:, 1]
            dt = time.perf_counter() - tb
            if out.shape[0] != chunk.shape[0] or not np.all(np.isfinite(out)):
                raise SystemExit(f"{uid}: invalid probe output")
            per_batch.append(
                {
                    "rows": int(chunk.shape[0]),
                    "seconds": dt,
                    "rows_per_second": chunk.shape[0] / dt,
                }
            )
            del out  # no scientific score is kept

        rates = [b["rows_per_second"] for b in per_batch]
        rec = {
            "unit_id": uid,
            "state_sha256": state_sha,
            "cold_reload_seconds": cold_reload,
            "scaler_verification": scaler_check,
            "ensemble": ensemble,
            "microbatches": len(per_batch),
            "first_batch_rows_per_second": rates[0],
            "first_batch_penalty_ratio": statistics.median(rates[1:]) / rates[0]
            if len(rates) > 1
            else 1.0,
            "median_rows_per_second": statistics.median(rates),
            "min_sustained_rows_per_second": min(rates[1:])
            if len(rates) > 1
            else rates[0],
            "max_rows_per_second": max(rates),
            "peak_rss_gb": proc.memory_info().rss / 1e9,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "per_batch": per_batch,
        }
        results.append(rec)
        print(
            f"  {uid:<36} reload={cold_reload:5.1f}s "
            f"median={rec['median_rows_per_second']:8,.0f} r/s "
            f"min_sustained={rec['min_sustained_rows_per_second']:8,.0f} r/s "
            f"VRAM={rec['peak_vram_gb']:.2f}GB RSS={rec['peak_rss_gb']:.2f}GB",
            flush=True,
        )
        del model, q
        torch.cuda.empty_cache()

    swap = psutil.swap_memory()
    worst = min(r["min_sustained_rows_per_second"] for r in results)
    rows = 10_137_155
    one_state_h = rows / worst / 3600
    total_h = 24 * one_state_h
    total_with_overhead = total_h * 1.20

    payload = {
        "schema": "m5_e6_throughput_probe_v1",
        "generated": time.time(),
        "scientific": False,
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "probe_matrix": meta,
        "no_fit_guard_blocked": blocked,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "swap_used_bytes": int(swap.used),
        "states": results,
        "gate": {
            "worst_sustained_rows_per_second": worst,
            "one_state_hours": one_state_h,
            "twenty_four_state_hours": total_h,
            "overhead_factor": 1.20,
            "projected_total_hours": total_with_overhead,
            "limit_hours": 336,
            "passed": total_with_overhead <= 336,
            "basis": "lowest sustained throughput across the three probe states, "
            "plus 20% engineering overhead; a facility-reasonableness gate, not a "
            "scientific one",
        },
    }
    digest = atomic_json(args.out / "e6_throughput_probe.json", payload)
    g = payload["gate"]
    print(f"\nprobe artifact sha256 = {digest}")
    print(
        f"  worst sustained        : {g['worst_sustained_rows_per_second']:,.0f} rows/s"
    )
    print(f"  one state              : {g['one_state_hours']:.2f} h")
    print(f"  24 states              : {g['twenty_four_state_hours']:.1f} h")
    print(f"  +20% overhead          : {g['projected_total_hours']:.1f} h")
    print(f"  limit                  : {g['limit_hours']} h")
    print(f"\nGATE {'PASSED' if g['passed'] else 'FAILED'}")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
