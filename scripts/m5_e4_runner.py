"""M5 E4 formal Path A runner: one unit = one fit + 8 same-process repeats.

A unit is one (context seed, cell, scaler arm). Each invocation runs exactly one
unit to completion: build or reuse the cached raw feature matrix, apply that
unit's scaler, fit once, verify the effective ensemble size, persist the state,
then perform 8 same-process inference repeats in the same process.

If the process dies before the repeats finish, the unit is marked
INTERRUPTED_INCOMPLETE. Missing repeats are never backfilled from a reloaded
state, because a reloaded state is not the same process.

Row probabilities are never averaged before scoring. The 8 internal ensemble
members of `n_estimators=8` are combined inside each `predict_proba` call and
are not replicates of any kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import ROOT, array_sha256, load_m3_frame, validate_context_manifest
from lead.m5_context import query_paths
from m5_e4_endpoints import endpoints
from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix

CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
FACTORIAL_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
QUERY_ROOT = ROOT / "data" / "processed" / "m5_context_stories"
REQUIRED_EFFECTIVE_N_ESTIMATORS = 8
REPEATS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Atomic write with platform-independent bytes.

    `newline=""` keeps the artifact byte-identical whichever platform writes it,
    so a digest taken here still matches after the results cross a machine
    boundary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def unit_id(seed: int, cell: str, arm: str) -> str:
    return f"seed{seed}__cell{cell}__{arm}"


def verify_ensemble(model: Any) -> dict:
    """Hard-fail unless the realised ensemble is exactly 8 members.

    The requested constructor value is not evidence. `n_estimators_` is the
    value TabPFN actually resolved, and in low-memory mode the containers the
    executor iterates are what a prediction really uses -- all of them are
    checked, and any mismatch stops the unit.
    """
    effective = int(getattr(model, "n_estimators_", -1))
    configs = getattr(model, "ensemble_configs_", None)
    n_configs = len(configs) if configs is not None else -1
    pre = getattr(getattr(model, "executor_", None), "ensemble_preprocessor", None)
    runtime = {}
    for name in ("configs", "pipelines", "pipeline_seeds", "subsample_feature_indices"):
        container = getattr(pre, name, None)
        runtime[name] = len(container) if container is not None else -1

    observed = {
        "requested_n_estimators": int(model.n_estimators),
        "auto_scale_n_estimators": bool(model.auto_scale_n_estimators),
        "effective_n_estimators_": effective,
        "len_ensemble_configs_": n_configs,
        "runtime_ensemble_containers": runtime,
    }
    bad = [
        f"n_estimators_={effective}"
        if effective != REQUIRED_EFFECTIVE_N_ESTIMATORS
        else "",
        f"ensemble_configs_={n_configs}"
        if n_configs != REQUIRED_EFFECTIVE_N_ESTIMATORS
        else "",
        *[
            f"{k}={v}"
            for k, v in runtime.items()
            if v != REQUIRED_EFFECTIVE_N_ESTIMATORS
        ],
    ]
    bad = [b for b in bad if b]
    if model.auto_scale_n_estimators:
        bad.append("auto_scale_n_estimators=True")
    if bad:
        raise AssertionError(
            "HARD FAILURE effective ensemble size is not "
            f"{REQUIRED_EFFECTIVE_N_ESTIMATORS}: {', '.join(bad)}"
        )
    return observed


def cache_paths(cache_root: Path, seed: int, cell: str) -> tuple[Path, Path]:
    base = cache_root / f"seed{seed}__cell{cell}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def load_unit(
    seed: int, cell: str, arm: str, model_path: Path, cache_root: Path | None
) -> dict:
    """Raw feature matrices, then this unit's scaler.

    The raw matrices depend only on (context seed, cell), so they are cached
    under exactly that key and verified by digest before reuse. The scaler is
    applied after the cache, so the two arms share a build without sharing a
    transform.
    """
    qm_path, q_path = query_paths(QUERY_ROOT, "screening")
    qm = json.loads(qm_path.read_text(encoding="utf-8"))
    with np.load(q_path) as payload:
        q_raw = np.asarray(payload["raw_index"], dtype="int64")
        q_meter = np.asarray(payload["meter"], dtype="int8")
        q_anom = np.asarray(payload["anomaly"], dtype="int8")
    if array_sha256(q_raw) != qm["raw_index_sha256"]:
        raise AssertionError("fixed 352-row query digest drifted")

    mpath = FACTORIAL_ROOT / "manifests" / f"seed{seed}" / f"{CELLS[cell]}.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if int(manifest["context_seed"]) != seed or int(manifest["model_seed"]) != 42:
        raise AssertionError(f"{mpath}: seed contract violated")

    npz, meta = cache_paths(cache_root, seed, cell) if cache_root else (None, None)
    cached = None
    if npz is not None and npz.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        if (
            info["context_seed"] == seed
            and info["cell"] == cell
            and info["context_manifest_sha256"] == sha256_file(mpath)
            and info["query_sha256"] == qm["raw_index_sha256"]
            and info["npz_sha256"] == sha256_file(npz)
        ):
            with np.load(npz) as z:
                cached = (z["x"], z["y"], z["q"])
            print(f"cache hit: {npz.name}", flush=True)

    if cached is None:
        frame = load_m3_frame(verbose=False)
        validate_context_manifest(frame, manifest)
        raw = np.asarray(manifest["raw_index"], dtype="int64")
        fit_frame = frame.loc[frame["building_id"] % 2 == 0]
        holdout_frame = frame.loc[frame["building_id"] % 2 == 1]
        x = build_feature_matrix(fit_frame, raw, "F4", full_frame=fit_frame)
        q = build_feature_matrix(holdout_frame, q_raw, "F4", full_frame=holdout_frame)
        validate_feature_matrix(q, matrix_name="e4 query")
        y = frame.iloc[raw]["anomaly"].to_numpy(dtype="int8")
        if x.shape != (20_000, 137) or int(y.sum()) != 10_000:
            raise AssertionError("E4 F4 or label-balance contract failed")
        if npz is not None:
            npz.parent.mkdir(parents=True, exist_ok=True)
            tmp = npz.with_suffix(".npz.tmp")
            np.savez(tmp, x=x, y=y, q=q)
            os.replace(tmp, npz)
            atomic_json(
                meta,
                {
                    "context_seed": seed,
                    "cell": cell,
                    "context_manifest_sha256": sha256_file(mpath),
                    "query_sha256": qm["raw_index_sha256"],
                    "npz_sha256": sha256_file(npz),
                    "x_shape": list(x.shape),
                    "q_shape": list(q.shape),
                    "built": time.time(),
                },
            )
    else:
        x, y, q = cached

    if arm == "cell_specific":
        scaler = StandardScaler().fit(x)
        scaler_source = "this cell's own 20,000 context rows"
        scaler_sha = None
    else:
        import joblib

        spath = FACTORIAL_ROOT / "scalers" / f"seed{seed}_pooled_reference.joblib"
        scaler = joblib.load(spath)
        scaler_source = str(spath.relative_to(ROOT))
        scaler_sha = sha256_file(spath)

    return {
        "x": scaler.transform(x).astype("float32"),
        "y": y,
        "q": scaler.transform(q).astype("float32"),
        "q_meter": q_meter,
        "q_anom": q_anom,
        "manifest_sha256": sha256_file(mpath),
        "context_raw_index_sha256": manifest["raw_index_sha256"],
        "query_sha256": qm["raw_index_sha256"],
        "model_path_sha256": sha256_file(model_path),
        "scaler_source": scaler_source,
        "scaler_sha256": scaler_sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-seed", type=int, required=True, choices=(42, 123, 999))
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument(
        "--scaler-arm", required=True, choices=("cell_specific", "frozen_reference")
    )
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proto = json.loads(args.protocol.read_text(encoding="utf-8"))["protocol"]
    contract = proto["ensemble"]
    if contract["required_effective_n_estimators_"] != REQUIRED_EFFECTIVE_N_ESTIMATORS:
        raise SystemExit("protocol and runner disagree on the ensemble size")

    uid = unit_id(args.context_seed, args.cell, args.scaler_arm)
    root = args.run_root / uid
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "FIT_COMPLETE.json"
    if marker.exists():
        print(f"{uid} already complete")
        return 0
    if (root / "INTERRUPTED_INCOMPLETE.json").exists():
        raise SystemExit(f"{uid} is marked INTERRUPTED_INCOMPLETE; refusing to resume")

    process_uuid = str(uuid.uuid4())

    if args.dry_run:
        data = load_unit(
            args.context_seed,
            args.cell,
            args.scaler_arm,
            args.model_path,
            args.cache_root,
        )
        print(f"dry-run ok {uid}: x={data['x'].shape} q={data['q'].shape}", flush=True)
        return 0

    import tabpfn
    import torch
    from tabpfn import TabPFNClassifier

    if tabpfn.__version__ != proto["inherited"]["scientific_tabpfn_version"]:
        raise SystemExit(
            f"TabPFN {tabpfn.__version__} != "
            f"{proto['inherited']['scientific_tabpfn_version']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU fallback is prohibited")

    data = load_unit(
        args.context_seed, args.cell, args.scaler_arm, args.model_path, args.cache_root
    )
    atomic_json(
        root / "fit_start.json",
        {
            "unit_id": uid,
            "context_seed": args.context_seed,
            "cell": args.cell,
            "scaler_arm": args.scaler_arm,
            "process_uuid": process_uuid,
            "started": time.time(),
            "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name(0),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "manifest_sha256": data["manifest_sha256"],
            "context_raw_index_sha256": data["context_raw_index_sha256"],
            "query_sha256": data["query_sha256"],
            "model_path_sha256": data["model_path_sha256"],
            "scaler_source": data["scaler_source"],
            "scaler_sha256": data["scaler_sha256"],
        },
    )

    interrupted = root / "INTERRUPTED_INCOMPLETE.json"
    try:
        model = TabPFNClassifier(
            n_estimators=contract["requested_n_estimators"],
            auto_scale_n_estimators=contract["auto_scale_n_estimators"],
            model_path=str(args.model_path),
            device="cuda",
            random_state=proto["inherited"]["model_seed"],
            fit_mode="low_memory",
            memory_saving_mode=True,
            keep_cache_on_device=False,
            ignore_pretraining_limits=True,
            n_preprocessing_jobs=1,
            inference_config={"SUBSAMPLE_SAMPLES": None},
            show_progress_bar=False,
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model.fit(data["x"], data["y"])
        fit_seconds = time.perf_counter() - t0
        if int(getattr(model, "n_train_samples_", -1)) != len(data["y"]):
            raise AssertionError("TabPFN fitted an unexpected row count")

        ensemble = verify_ensemble(model)

        state_path = root / "model.tabpfn_fit"
        model.save_fit_state(state_path)
        state_sha = sha256_file(state_path)

        records = []
        for r in range(REPEATS):
            rt0 = time.perf_counter()
            score = np.asarray(model.predict_proba(data["q"])[:, 1], dtype="float64")
            secs = time.perf_counter() - rt0
            if not np.all(np.isfinite(score)):
                raise AssertionError(f"non-finite scores in repeat {r}")
            rec = {
                "unit_id": uid,
                "context_seed": args.context_seed,
                "cell": args.cell,
                "scaler_arm": args.scaler_arm,
                "repeat": r,
                "mode": "same_process_inference_repeat",
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
            print(f"  {uid} repeat {r + 1}/{REPEATS} ({secs:,.1f}s)", flush=True)
    except BaseException as exc:
        atomic_json(
            interrupted,
            {
                "unit_id": uid,
                "process_uuid": process_uuid,
                "reason": f"{type(exc).__name__}: {exc}",
                "repeats_written": len(list((root / "repeats").glob("*.json"))),
                "reload_backfill": "forbidden",
                "timestamp": time.time(),
            },
        )
        raise

    atomic_json(
        marker,
        {
            "unit_id": uid,
            "context_seed": args.context_seed,
            "cell": args.cell,
            "scaler_arm": args.scaler_arm,
            "process_uuid": process_uuid,
            "status": "COMPLETE",
            "fits": 1,
            "repeats": len(records),
            "fit_seconds": fit_seconds,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "state_sha256": state_sha,
            "state_bytes": state_path.stat().st_size,
            "ensemble": ensemble,
            "distinct_score_digests": len({r["score_sha256"] for r in records}),
            "completed": time.time(),
        },
    )
    print(
        f"{uid} COMPLETE: fit={fit_seconds:,.1f}s repeats={len(records)} "
        f"effective_n_estimators_={ensemble['effective_n_estimators_']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
