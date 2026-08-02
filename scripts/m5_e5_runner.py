"""M5 E5 runner: reload one E4 state, score the 192-row query 8 times, stop.

One state per invocation, in its own process. Nothing is fitted: the no-fit
guard replaces every fit entry point with a raising stub before TabPFN is
imported for use, so a future edit that reintroduces a fit fails at the first
call instead of quietly producing numbers that are not a replication.

The scaler is not re-chosen. Each unit inherits E4's arm, and the scaler is
verified against the scaled context matrix stored inside the E4 state before any
query row is scored. In cell 00 TabPFN classifies `meter` as categorical and
re-encodes it after our scaling, so that column is excluded from the comparison
and the exclusion is recorded rather than assumed away.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPEATS = 8
QUERY_ROWS = 192
FEATURES = 137
REQUIRED_N_ESTIMATORS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def state_context(state_path: Path) -> tuple[np.ndarray, list[bool]]:
    """The scaled context matrix E4 fitted on, and which columns stayed numeric.

    TabPFN stores the training matrix it received, then re-encodes any column it
    classified as categorical. Only the numeric columns still carry our scaling,
    so only those can verify a scaler.
    """
    import joblib

    with zipfile.ZipFile(state_path) as z:
        attrs = joblib.load(io.BytesIO(z.read("fitted_attrs.joblib")))
        ex = joblib.load(io.BytesIO(z.read("executor_state.joblib")))
    schema = attrs["inferred_feature_schema_"]
    numeric = [
        "CATEG" not in str(getattr(f, "modality", "")).upper() for f in schema.features
    ]
    return np.asarray(ex.X_train, dtype="float64"), numeric


def verify_scaler(scaler: Any, raw_context: np.ndarray, state_path: Path) -> dict:
    """Reject the scaler unless it reproduces E4's scaled context exactly.

    `raw_context` must carry the dtype E4 fed the scaler. E4 read the cached
    float32 matrix and transformed it directly; upcasting to float64 first
    changes the arithmetic and the comparison fails on the last bit for reasons
    that have nothing to do with the scaler being right.
    """
    if raw_context.dtype != np.float32:
        raise AssertionError(f"context dtype is {raw_context.dtype}; E4 scaled float32")
    x_train, numeric = state_context(state_path)
    if raw_context.shape != x_train.shape:
        raise AssertionError(
            f"context shape {raw_context.shape} != state {x_train.shape}"
        )
    rebuilt = scaler.transform(raw_context).astype("float32").astype("float64")
    cols = np.flatnonzero(numeric)
    a, b = rebuilt[:, cols], x_train[:, cols]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise AssertionError("NaN pattern differs between the rebuild and the state")
    max_diff = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
    if max_diff != 0.0:
        raise AssertionError(
            f"HARD FAILURE the scaler does not reproduce E4's scaled context "
            f"(max |difference| = {max_diff:.3e} over {cols.size} numeric columns)"
        )
    return {
        "numeric_columns_compared": int(cols.size),
        "categorical_columns_excluded": int(FEATURES - cols.size),
        "max_abs_difference": max_diff,
        "exact": True,
    }


def verify_ensemble(model: Any) -> dict:
    """Hard-fail unless the realised ensemble is exactly 8 members."""
    effective = int(getattr(model, "n_estimators_", -1))
    configs = getattr(model, "ensemble_configs_", None)
    n_configs = len(configs) if configs is not None else -1
    pre = getattr(getattr(model, "executor_", None), "ensemble_preprocessor", None)
    runtime = {
        name: (len(c) if (c := getattr(pre, name, None)) is not None else -1)
        for name in (
            "configs",
            "pipelines",
            "pipeline_seeds",
            "subsample_feature_indices",
        )
    }
    observed = {
        "requested_n_estimators": int(model.n_estimators),
        "auto_scale_n_estimators": bool(model.auto_scale_n_estimators),
        "effective_n_estimators_": effective,
        "len_ensemble_configs_": n_configs,
        "runtime_ensemble_containers": runtime,
    }
    bad = [f"n_estimators_={effective}"] if effective != REQUIRED_N_ESTIMATORS else []
    if n_configs != REQUIRED_N_ESTIMATORS:
        bad.append(f"ensemble_configs_={n_configs}")
    bad += [f"{k}={v}" for k, v in runtime.items() if v != REQUIRED_N_ESTIMATORS]
    if model.auto_scale_n_estimators:
        bad.append("auto_scale_n_estimators=True")
    if int(model.n_estimators) != REQUIRED_N_ESTIMATORS:
        bad.append(f"requested={model.n_estimators}")
    if bad:
        raise AssertionError(
            f"HARD FAILURE effective ensemble is not {REQUIRED_N_ESTIMATORS}: "
            + ", ".join(bad)
        )
    return observed


def load_inputs(spec: dict, proto: dict, root: Path, cache_root: Path) -> dict:
    """The raw context matrix, the raw 192-row query matrix, and the scaler."""
    import joblib

    seed, cell, arm = spec["context_seed"], spec["cell"], spec["scaler_arm"]

    ctx_npz = cache_root / f"seed{seed}__cell{cell}.npz"
    ctx_meta = ctx_npz.with_suffix(".json")
    meta = read_json(ctx_meta)
    if sha256_file(ctx_npz) != meta["npz_sha256"]:
        raise AssertionError(f"{ctx_npz.name}: digest does not match its manifest")
    if meta["context_manifest_sha256"] != spec["context_manifest_sha256"]:
        raise AssertionError(f"{ctx_npz.name}: built from a different context manifest")
    with np.load(ctx_npz) as z:
        # Stored dtype, not upcast: E4 passed the cached float32 array straight
        # to the scaler, and transforming a float64 copy changes the last bit.
        raw_context = np.asarray(z["x"])
    if raw_context.shape != (20_000, FEATURES):
        raise AssertionError(f"context shape {raw_context.shape}")

    q_npz = cache_root / "query192.npz"
    q_meta = read_json(q_npz.with_suffix(".json"))
    if sha256_file(q_npz) != q_meta["npz_sha256"]:
        raise AssertionError("query192 cache digest does not match its manifest")
    if q_meta["raw_index_sha256"] != proto["query"]["raw_index_sha256"]:
        raise AssertionError("query192 cache was built from a different query")
    with np.load(q_npz) as z:
        raw_query = np.asarray(z["q"])
        q_raw_index = np.asarray(z["raw_index"], dtype="int64")
        q_meter = np.asarray(z["meter"], dtype="int8")
        q_anom = np.asarray(z["anomaly"], dtype="int8")
    if raw_query.shape != (QUERY_ROWS, FEATURES):
        raise AssertionError(f"query shape {raw_query.shape}")
    if (
        hashlib.sha256(q_raw_index.tobytes()).hexdigest()
        != proto["query"]["raw_index_sha256"]
    ):
        raise AssertionError("query raw_index order drifted")

    if arm == "frozen_reference":
        spath = root / spec["scaler_source"]["path"]
        if sha256_file(spath) != spec["scaler_source"]["sha256"]:
            raise AssertionError("frozen scaler digest drifted")
        from sklearn.preprocessing import StandardScaler  # noqa: F401

        scaler = joblib.load(spath)
        origin = {"kind": "persisted", "path": spec["scaler_source"]["path"]}
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(raw_context)
        origin = {"kind": "rebuilt", "rule": "StandardScaler().fit(context)"}

    return {
        "raw_context": raw_context,
        "raw_query": raw_query,
        "scaler": scaler,
        "scaler_origin": origin,
        "q_meter": q_meter,
        "q_anom": q_anom,
        "query_sha256": proto["query"]["raw_index_sha256"],
        "context_npz_sha256": meta["npz_sha256"],
        "query_npz_sha256": q_meta["npz_sha256"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    args = ap.parse_args()

    proto = read_json(args.protocol)["protocol"]
    specs = {s["unit_id"]: s for s in read_json(args.state_manifest)["states"]}
    if args.unit_id not in specs:
        raise SystemExit(f"{args.unit_id} is not in the frozen state manifest")
    spec = specs[args.unit_id]

    root = args.run_root / args.unit_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / "UNIT_COMPLETE.json").exists():
        print(f"{args.unit_id} already complete")
        return 0
    if (root / "INTERRUPTED_INCOMPLETE.json").exists():
        raise SystemExit(
            f"{args.unit_id} is INTERRUPTED_INCOMPLETE; refusing to resume"
        )

    process_uuid = str(uuid.uuid4())

    # Arm the guard before TabPFN is used for anything.
    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import tabpfn
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e4_endpoints import endpoints

    required_version = proto["inherited_from_e4"]["scientific_tabpfn_version"]
    if tabpfn.__version__ != required_version:
        raise SystemExit(f"TabPFN {tabpfn.__version__} != {required_version}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU fallback is prohibited")

    state_path = args.repo_root / spec["state_path"]
    state_sha = sha256_file(state_path)
    if state_sha != spec["state_sha256"]:
        raise SystemExit(f"HARD FAILURE state digest drifted for {args.unit_id}")

    data = load_inputs(spec, proto, args.repo_root, args.cache_root)
    scaler_check = verify_scaler(data["scaler"], data["raw_context"], state_path)

    q = data["scaler"].transform(data["raw_query"]).astype("float32")
    if q.shape != (QUERY_ROWS, FEATURES):
        raise SystemExit(f"transformed query shape {q.shape}")

    atomic_json(
        root / "reload_start.json",
        {
            "unit_id": args.unit_id,
            "process_uuid": process_uuid,
            "started": time.time(),
            "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name(0),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "state_sha256": state_sha,
            "query_sha256": data["query_sha256"],
            "context_npz_sha256": data["context_npz_sha256"],
            "query_npz_sha256": data["query_npz_sha256"],
            "scaler_origin": data["scaler_origin"],
            "scaler_verification": scaler_check,
            "transformed_query_shape": list(q.shape),
            "transformed_query_dtype": str(q.dtype),
            "no_fit_guard_blocked": blocked,
        },
    )

    interrupted = root / "INTERRUPTED_INCOMPLETE.json"
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = load_fitted_tabpfn_model(state_path, device="cuda:0")
        load_seconds = time.perf_counter() - t0
        ensemble = verify_ensemble(model)

        records = []
        for r in range(REPEATS):
            rt0 = time.perf_counter()
            score = np.asarray(model.predict_proba(q)[:, 1], dtype="float64")
            secs = time.perf_counter() - rt0
            if score.size != QUERY_ROWS:
                raise AssertionError(f"repeat {r}: score length {score.size}")
            if not np.all(np.isfinite(score)):
                raise AssertionError(f"repeat {r}: non-finite scores")
            rec = {
                "unit_id": args.unit_id,
                "context_seed": spec["context_seed"],
                "cell": spec["cell"],
                "scaler_arm": spec["scaler_arm"],
                "repeat": r,
                "mode": "same_process_inference_repeat_after_fresh_reload",
                "endpoints": endpoints(score, data["q_meter"], data["q_anom"]),
                "score": score.tolist(),
                "score_sha256": hashlib.sha256(score.tobytes()).hexdigest(),
                "seconds": secs,
                "state_sha256": state_sha,
                "process_uuid": process_uuid,
                "timestamp": time.time(),
            }
            atomic_json(root / "repeats" / f"repeat_{r:03d}.json", rec)
            records.append(rec)
            print(
                f"  {args.unit_id} repeat {r + 1}/{REPEATS} ({secs:,.1f}s)", flush=True
            )
    except BaseException as exc:
        atomic_json(
            interrupted,
            {
                "unit_id": args.unit_id,
                "process_uuid": process_uuid,
                "reason": f"{type(exc).__name__}: {exc}",
                "repeats_written": len(list((root / "repeats").glob("*.json"))),
                "reload_backfill": "forbidden",
                "timestamp": time.time(),
            },
        )
        raise

    atomic_json(
        root / "UNIT_COMPLETE.json",
        {
            "unit_id": args.unit_id,
            "context_seed": spec["context_seed"],
            "cell": spec["cell"],
            "scaler_arm": spec["scaler_arm"],
            "process_uuid": process_uuid,
            "status": "COMPLETE",
            "fits_performed": 0,
            "reloads": 1,
            "repeats": len(records),
            "score_vector_length": QUERY_ROWS,
            "load_seconds": load_seconds,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "state_sha256": state_sha,
            "ensemble": ensemble,
            "scaler_verification": scaler_check,
            "distinct_score_digests": len({r["score_sha256"] for r in records}),
            "completed": time.time(),
        },
    )
    print(
        f"{args.unit_id} COMPLETE: reload={load_seconds:,.1f}s repeats={len(records)} "
        f"effective_n_estimators_={ensemble['effective_n_estimators_']} "
        f"scaler_exact={scaler_check['exact']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
