"""352 列相容性 sentinel。throughput 測試的前置關卡。

不要求與 RTX 5070 Ti 逐位元相同 —— E3/E4/E5/E6 都已確認同一 fitted state 上的
重複推論本來就不是 bitwise 可重現的。要檢查的是別的東西:factorial 方向有沒有
反轉、endpoint 有沒有明顯系統性位移、重複的離散程度有沒有異常超出歷史範圍。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SENTINEL_ROWS = 352
REPEATS = 8
REQUIRED_N_ESTIMATORS = 8

# E4/E5/E6 的歷史範圍。sentinel 只跟這個比。
HISTORY = {
    "e5_steam_auc_half_width_range": [0.000415, 0.009790],
    "e5_steam_margin_half_width_range": [0.000334, 0.006440],
    "distinct_digests_expected": 8,
    "e6_gpu_host_sentinel_distinct_digests": 8,
    "half_width_alarm_multiplier": 10.0,
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    sc = verify_scaler(scaler, raw_ctx, sp)
    if not sc["exact"]:
        raise SystemExit(f"{spec['unit_id']}: scaler 未精確重現")
    model = load_fitted_tabpfn_model(sp, device=device)
    ens = verify_ensemble(model)
    if ens["effective_n_estimators_"] != REQUIRED_N_ESTIMATORS:
        raise SystemExit(f"{spec['unit_id']}: effective n_estimators_ 不是 8")
    return model, scaler, sc, ens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用;禁止 CPU fallback")

    sm = read_json(args.bundle_root / "state_manifest.json")
    pm = read_json(args.bundle_root / "probe_manifest.json")
    sent_path = args.bundle_root / pm["sentinel"]["path"]
    if sha256_file(sent_path) != pm["sentinel"]["sha256"]:
        raise SystemExit("sentinel digest 不符")
    with np.load(sent_path) as z:
        sent_x = np.asarray(z["x"])
    if sent_x.shape[0] != SENTINEL_ROWS or sent_x.dtype != np.float32:
        raise SystemExit(f"sentinel {sent_x.shape} {sent_x.dtype}")

    gpu_uuid = (
        torch.cuda.get_device_properties(0).uuid.hex
        if hasattr(torch.cuda.get_device_properties(0), "uuid")
        else ""
    )

    results, failures = [], []
    for spec in sm["states"]:
        run_uuid = str(uuid.uuid4())
        model, scaler, sc, ens = load_unit(args.bundle_root, spec, args.device)
        q = scaler.transform(sent_x).astype("float32")
        if q.dtype != np.float32:
            raise SystemExit("sentinel query 被上轉型")

        runs = []
        for r in range(REPEATS):
            t0 = time.perf_counter()
            out = model.predict_proba(q)[:, 1].astype("float64")
            if not np.all(np.isfinite(out)):
                failures.append(f"{spec['unit_id']}: repeat {r} 出現 non-finite")
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
        half = float(2.364 * means.std(ddof=1) / np.sqrt(means.size))  # 95% t, df=7
        digests = {r["sha256"] for r in runs}

        rec = {
            "unit_id": spec["unit_id"],
            "cell": spec["cell"],
            "scaler_arm": spec["scaler_arm"],
            "process_uuid": run_uuid,
            "gpu_uuid": gpu_uuid,
            "rows": SENTINEL_ROWS,
            "repeats": REPEATS,
            "distinct_digests": len(digests),
            "bitwise_identical": len(digests) == 1,
            "endpoint_mean": float(means.mean()),
            "endpoint_min": float(means.min()),
            "endpoint_max": float(means.max()),
            "endpoint_half_width": half,
            "all_finite": all(np.isfinite(means)),
            "scaler_verification": sc,
            "ensemble": ens,
            "fits_performed": 0,
            "runs": runs,
        }
        # 離散程度異常放大是相容性問題;比歷史上限大一個量級才報警。
        ceiling = (
            HISTORY["e5_steam_auc_half_width_range"][1]
            * HISTORY["half_width_alarm_multiplier"]
        )
        if half > ceiling:
            failures.append(
                f"{spec['unit_id']}: half-width {half:.6f} 超出歷史上限 {ceiling:.6f}"
            )
        if ens["effective_n_estimators_"] != REQUIRED_N_ESTIMATORS:
            failures.append(f"{spec['unit_id']}: ensemble contract 失敗")
        results.append(rec)
        print(
            f"  {spec['unit_id']:<40} digests={len(digests)}/8 "
            f"mean={rec['endpoint_mean']:.6f} half={half:.6f}",
            flush=True,
        )
        del model, scaler, q
        torch.cuda.empty_cache()

    # factorial 方向:cell01 的 steam 傾向應高於 cell00(negative support 存在時)
    by_cell = {r["cell"]: r["endpoint_mean"] for r in results}
    direction_ok = None
    if "00" in by_cell and "01" in by_cell:
        direction_ok = by_cell["01"] > by_cell["00"]
        if not direction_ok:
            failures.append(
                f"factorial 方向反轉:cell01 mean {by_cell['01']:.6f} "
                f"未高於 cell00 {by_cell['00']:.6f}"
            )

    verdict = "COMPATIBLE" if not failures else "INCOMPATIBLE"
    payload = {
        "schema": "m5_e6_gputw_sentinel_v1",
        "generated": time.time(),
        "verdict": verdict,
        "failures": failures,
        "bitwise_cross_gpu_equality_required": False,
        "history_reference": HISTORY,
        "factorial_direction_cell01_gt_cell00": direction_ok,
        "no_fit_guard_blocked": blocked,
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "results": results,
    }
    d1 = atomic_json(args.out / "sentinel_results.json", payload)
    atomic_json(
        args.out / "compatibility_results.json",
        {
            "schema": "m5_e6_gputw_compatibility_v1",
            "verdict": verdict,
            "failures": failures,
            "states_reloaded": len(results),
            "states_expected": len(sm["states"]),
            "sentinel_results_sha256": d1,
            "fits_performed": 0,
        },
    )
    print(f"\ncompatibility verdict = {verdict}")
    for f in failures:
        print(f"  FAIL: {f}")
    return 0 if verdict == "COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
