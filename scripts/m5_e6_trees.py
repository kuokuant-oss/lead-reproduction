"""M5 E6 fixed-tree comparators over the full odd-building holdout.

Runs only in the environment that fitted them. E5 measured the cost of ignoring
that: the same ensembles differ by a mean of 8.1e-03 between this laptop and
gpu-host, so a gpu-host tree output is not the same comparator wearing a
different hat. No refit, no artificial replicates, no gpu-host tree output.

The 24-unit bit-exact identity gate against E4's frozen 352-row comparator runs
twice -- once before any holdout row is scored and once after the last one -- so
a mid-run environment change cannot pass unnoticed. `max_abs_diff` must be
exactly 0 and all 352 rows exactly equal; there is no tolerance and no sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROWS = 10_137_155
FEATURES = 137
SCREENING_ROWS = 352
CHUNK = 500_000


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


def atomic_npy(path: Path, arr: np.ndarray) -> str:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def environment() -> dict:
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "host_role": "original fixed-tree fit environment",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "joblib": joblib.__version__,
    }


def ensemble_predict(runner, saved: dict, x: np.ndarray) -> np.ndarray:
    return np.mean(
        [
            runner.predict_probability(name, saved["models"][name], x)
            for name in saved["model_order"]
        ],
        axis=0,
    ).astype("float32")


def run_gate(
    runner, specs, repo_root: Path, x352: np.ndarray, phase: str
) -> list[dict]:
    gate = []
    print(f"\ngate ({phase}): 24 units must reproduce E4's 352-row comparator exactly")
    for spec in specs:
        uid = spec["unit_id"]
        saved = joblib.load(repo_root / spec["tree_ensemble"])
        scaler = joblib.load(repo_root / spec["tree_scaler"])
        canonical = repo_root / spec["e4_comparator"]
        with np.load(canonical, allow_pickle=True) as z:
            want = np.asarray(z["score"], dtype="float64")
        got = np.asarray(
            ensemble_predict(runner, saved, scaler.transform(x352).astype("float32")),
            dtype="float64",
        )
        diff = np.abs(got - want)
        exact_rows = int((diff == 0).sum())
        max_abs = float(diff.max())
        ok = max_abs == 0.0 and exact_rows == SCREENING_ROWS
        gate.append(
            {
                "phase": phase,
                "unit_id": uid,
                "max_abs_diff": max_abs,
                "exact_rows": exact_rows,
                "rows": SCREENING_ROWS,
                "bit_exact": ok,
            }
        )
        print(
            f"  {'PASS' if ok else 'FAIL'} {uid:<40} max|d|={max_abs:.3e} "
            f"exact={exact_rows}/{SCREENING_ROWS}"
        )
        if not ok:
            raise SystemExit(
                f"HARD FAILURE {uid} is not bit-exact against E4's comparator in "
                f"the {phase} gate; stopping rather than applying a tolerance"
            )
    return gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    from run_m5_story_ae_probe import build_feature_matrix, load_tree_runner

    from lead import load_m3_frame

    tm = json.loads(
        (args.protocol_root / "e6_tree_manifest.json").read_text(encoding="utf-8")
    )
    feat = json.loads(
        (args.feature_root / "e6_feature_manifest.json").read_text(encoding="utf-8")
    )
    specs = tm["units"]

    env = environment()
    contract = tm["environment_contract"]
    drift = {
        k: (contract[k], env[k]) for k in contract if k in env and contract[k] != env[k]
    }
    if drift:
        raise SystemExit(f"HARD FAILURE tree environment drift: {drift}")

    for spec in specs:
        for key, sha in (
            ("tree_ensemble", "tree_ensemble_sha256"),
            ("tree_scaler", "tree_scaler_sha256"),
        ):
            if sha256_file(args.repo_root / spec[key]) != spec[sha]:
                raise SystemExit(f"{spec['unit_id']}: {key} digest drifted")

    x = np.load(args.feature_root / feat["path"], mmap_mode="r")
    if x.shape != (ROWS, FEATURES) or x.dtype != np.float32:
        raise SystemExit(f"feature matrix {x.shape} {x.dtype}")

    sq = np.load(args.feature_root / "e6_sentinel_query.npz")
    raw352 = np.asarray(sq["raw_index"], dtype="int64")
    frame = load_m3_frame(verbose=False)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    x352 = build_feature_matrix(holdout, raw352, "F4", full_frame=holdout)
    del frame

    runner = load_tree_runner()
    gate_before = run_gate(runner, specs, args.repo_root, x352, "before")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\ngate passed 24/24 -- scoring {ROWS:,} holdout rows per unit\n")
    records = []
    for spec in specs:
        uid = spec["unit_id"]
        out_path = args.out / uid / "tree_scores.float32.npy"
        if (args.out / uid / "TREE_COMPLETE.json").exists():
            print(f"  {uid}: already complete, skipping")
            records.append(
                json.loads(
                    (args.out / uid / "TREE_COMPLETE.json").read_text(encoding="utf-8")
                )
            )
            continue
        saved = joblib.load(args.repo_root / spec["tree_ensemble"])
        scaler = joblib.load(args.repo_root / spec["tree_scaler"])
        scores = np.empty(ROWS, dtype="float32")
        t0 = time.perf_counter()
        for start in range(0, ROWS, CHUNK):
            stop = min(start + CHUNK, ROWS)
            q = scaler.transform(np.asarray(x[start:stop])).astype("float32")
            if q.dtype != np.float32:
                raise SystemExit(f"{uid}: float64 upcast at row {start}")
            scores[start:stop] = ensemble_predict(runner, saved, q)
            del q
        seconds = time.perf_counter() - t0
        if not np.all(np.isfinite(scores)):
            raise SystemExit(f"{uid}: non-finite tree scores")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sha = atomic_npy(out_path, scores)
        rec = {
            "unit_id": uid,
            "rows_scored": int(ROWS),
            "scores_path": out_path.name,
            "scores_sha256": sha,
            "scores_mean": float(scores.mean()),
            "scores_min": float(scores.min()),
            "scores_max": float(scores.max()),
            "all_finite": True,
            "seconds": seconds,
            "rows_per_second": ROWS / seconds,
            "refit": False,
            "environment": env,
        }
        atomic_json(args.out / uid / "TREE_COMPLETE.json", rec)
        records.append(rec)
        print(
            f"  {uid:<40} {ROWS / seconds:>9,.0f} r/s  {seconds / 60:5.1f} min  "
            f"{sha[:16]}",
            flush=True,
        )
        del scores

    gate_after = run_gate(runner, specs, args.repo_root, x352, "after")

    digest = atomic_json(
        args.out / "e6_tree_results.json",
        {
            "schema": "m5_e6_tree_results_v1",
            "generated": time.time(),
            "environment": env,
            "environment_contract_verified": True,
            "no_fit_guard_blocked": blocked,
            "refit": False,
            "gpu_host_tree_outputs": False,
            "feature_sha256": feat["sha256"],
            "gate_before": gate_before,
            "gate_after": gate_after,
            "gate_before_passed": sum(1 for g in gate_before if g["bit_exact"]),
            "gate_after_passed": sum(1 for g in gate_after if g["bit_exact"]),
            "units": records,
        },
    )
    print(f"\ntree results sha256 = {digest}")
    print(f"  gate before {sum(1 for g in gate_before if g['bit_exact'])}/24 bit-exact")
    print(f"  gate after  {sum(1 for g in gate_after if g['bit_exact'])}/24 bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
