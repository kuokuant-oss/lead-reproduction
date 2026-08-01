"""M5 E3 variance pilot runner: one cell = one fit + same-process repeats.

Executes exactly one factorial cell per invocation, in the frozen order. The
same process that performs the fit performs every same-process repeat; if that
process dies before the repeats finish, the cell is marked INTERRUPTED and the
runner refuses to backfill missing repeats from a reloaded state.

Each repeat is scored independently and checkpointed atomically. Row
probabilities are never averaged before scoring.

No E4, no Path B, no 192-row query, no tree refit.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import ROOT, array_sha256, load_m3_frame, validate_context_manifest
from lead.m5_context import query_paths
from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix

METER = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}
CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
FACTORIAL_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
QUERY_ROOT = ROOT / "data" / "processed" / "m5_context_stories"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(tmp, path)


def endpoints(score: np.ndarray, meter: np.ndarray, anomaly: np.ndarray) -> dict:
    """All readouts for one scored query pass. Score-based and rank-based are
    computed and reported separately and never pooled."""

    def group(m: str, a: int) -> np.ndarray:
        return score[(meter == METER[m]) & (anomaly == a)]

    steam_pos = group("steam", 1)
    hw_neg = group("hotwater", 0)
    cw_pos = group("chilledwater", 1)
    cw_neg = group("chilledwater", 0)

    def pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
        y = np.concatenate([np.ones(pos.size, "int8"), np.zeros(neg.size, "int8")])
        return float(roc_auc_score(y, np.concatenate([pos, neg])))

    # within-meter rank of chilledwater positives among all chilledwater rows
    cw_all = score[meter == METER["chilledwater"]]
    cw_lab = anomaly[meter == METER["chilledwater"]]
    cw_rank = pd.Series(cw_all).rank(method="average", pct=True).to_numpy()
    global_rank = pd.Series(score).rank(method="average", pct=True).to_numpy()

    y_cw = np.concatenate([np.ones(cw_pos.size, "int8"), np.zeros(cw_neg.size, "int8")])
    s_cw = np.concatenate([cw_pos, cw_neg])

    return {
        # --- steam principal (the only precision-gating endpoints) ---
        "steam_positive_vs_hotwater_negative_pairwise_auc": pairwise_auc(
            steam_pos, hw_neg
        ),
        "steam_positive_minus_hotwater_negative_score_margin": float(
            steam_pos.mean() - hw_neg.mean()
        ),
        # --- steam rank-based, reported separately, non-gating ---
        "steam_positive_global_rank": float(
            global_rank[(meter == METER["steam"]) & (anomaly == 1)].mean()
        ),
        "steam_positive_within_meter_rank": float(
            pd.Series(score[meter == METER["steam"]])
            .rank(method="average", pct=True)
            .to_numpy()[anomaly[meter == METER["steam"]] == 1]
            .mean()
        ),
        # --- chilledwater within-meter secondary (non-gating) ---
        "chilledwater_positive_vs_chilledwater_negative_pairwise_auc": pairwise_auc(
            cw_pos, cw_neg
        ),
        "chilledwater_positive_minus_chilledwater_negative_score_margin": float(
            cw_pos.mean() - cw_neg.mean()
        ),
        "chilledwater_within_meter_pr_auc": float(average_precision_score(y_cw, s_cw)),
        "chilledwater_within_meter_roc_auc": float(roc_auc_score(y_cw, s_cw)),
        "chilledwater_positive_within_meter_rank": float(cw_rank[cw_lab == 1].mean()),
        "chilledwater_positive_global_rank": float(
            global_rank[(meter == METER["chilledwater"]) & (anomaly == 1)].mean()
        ),
        # --- resolution-limited diagnostic only, never mechanism-bearing ---
        "RESOLUTION_LIMITED_DIAGNOSTIC_chilledwater_positive_vs_hotwater_negative_pairwise_auc": pairwise_auc(
            cw_pos, hw_neg
        ),
    }


def half_width(values: list[float]) -> float:
    """Two-sided 95% Student-t half-width on the repeat-level mean."""
    from scipy import stats

    n = len(values)
    if n < 2:
        return float("inf")
    return float(stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))


def evaluate_gate(
    records: list[dict], bounded_target: float, margin_target: float
) -> dict:
    """Only the two steam endpoints gate continuation."""
    auc = [
        r["endpoints"]["steam_positive_vs_hotwater_negative_pairwise_auc"]
        for r in records
    ]
    margin = [
        r["endpoints"]["steam_positive_minus_hotwater_negative_score_margin"]
        for r in records
    ]
    hw_auc, hw_margin = half_width(auc), half_width(margin)
    return {
        "n": len(records),
        "auc_mean": float(np.mean(auc)),
        "auc_sd": float(np.std(auc, ddof=1)),
        "auc_half_width": hw_auc,
        "auc_target": bounded_target,
        "auc_pass": bool(hw_auc <= bounded_target),
        "margin_mean": float(np.mean(margin)),
        "margin_sd": float(np.std(margin, ddof=1)),
        "margin_half_width": hw_margin,
        "margin_target": margin_target,
        "margin_pass": bool(hw_margin <= margin_target),
        "both_pass": bool(hw_auc <= bounded_target and hw_margin <= margin_target),
    }


def load_cell(cell: str, model_path: Path) -> dict:
    frame = load_m3_frame(verbose=False)
    qm_path, q_path = query_paths(QUERY_ROOT, "screening")
    qm = json.loads(qm_path.read_text(encoding="utf-8"))
    with np.load(q_path) as payload:
        q_raw = np.asarray(payload["raw_index"], dtype="int64")
        q_meter = np.asarray(payload["meter"], dtype="int8")
        q_anom = np.asarray(payload["anomaly"], dtype="int8")
    if array_sha256(q_raw) != qm["raw_index_sha256"]:
        raise AssertionError("fixed 352-row query digest drifted")

    mpath = FACTORIAL_ROOT / "manifests" / "seed42" / f"{CELLS[cell]}.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    validate_context_manifest(frame, manifest)
    raw = np.asarray(manifest["raw_index"], dtype="int64")

    fit_frame = frame.loc[frame["building_id"] % 2 == 0]
    holdout_frame = frame.loc[frame["building_id"] % 2 == 1]
    x = build_feature_matrix(fit_frame, raw, "F4", full_frame=fit_frame)
    q = build_feature_matrix(holdout_frame, q_raw, "F4", full_frame=holdout_frame)
    validate_feature_matrix(q, matrix_name="e3 query")
    y = frame.iloc[raw]["anomaly"].to_numpy(dtype="int8")
    if x.shape != (20_000, 137) or int(y.sum()) != 10_000:
        raise AssertionError("E3 F4 or label-balance contract failed")

    scaler = StandardScaler().fit(x)  # cell-specific scaler
    return {
        "x": scaler.transform(x).astype("float32"),
        "y": y,
        "q": scaler.transform(q).astype("float32"),
        "q_meter": q_meter,
        "q_anom": q_anom,
        "manifest": manifest,
        "manifest_sha256": sha256_file(mpath),
        "query_sha256": qm["raw_index_sha256"],
        "model_path_sha256": sha256_file(model_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--max-repeats", type=int, default=40)
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proto = json.loads(args.protocol.read_text(encoding="utf-8"))["protocol"]
    order = proto["supplementary_decisions"]["1_schedule_seed"]["realised_cell_order"]
    if args.cell not in order:
        raise SystemExit(f"cell {args.cell} not in frozen order {order}")
    batches = proto["supplementary_decisions"]["3_repeat_batches"]["batches"]
    if args.max_repeats not in batches:
        raise SystemExit(f"--max-repeats must be one of the frozen batches {batches}")
    bounded_target = (
        proto["supplementary_decisions"]["5_ci_half_width"]["bounded_metric_pass"]
        .rsplit("<=", 1)[1]
        .strip()
    )
    bounded_target = float(bounded_target)
    margin_target = float(
        proto["reference_iqr"]["per_cell"][args.cell]["margin_half_width_target"]
    )

    root = args.run_root / f"cell_{args.cell}"
    repeats_dir = root / "repeats"
    repeats_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(repeats_dir.glob("repeat_*.json"))
    fit_marker = root / "fit_complete.json"

    # ---- resume guard: same-process repeats may not be backfilled ----
    process_uuid = str(uuid.uuid4())
    if fit_marker.exists() and len(done) < args.max_repeats:
        prior = json.loads(fit_marker.read_text(encoding="utf-8"))
        atomic_json(
            root / "INTERRUPTED.json",
            {
                "cell": args.cell,
                "status": "INTERRUPTED_INCOMPLETE",
                "fit_process_uuid": prior.get("process_uuid"),
                "current_process_uuid": process_uuid,
                "completed_repeats": len(done),
                "target_repeats": args.max_repeats,
                "reason": (
                    "the process that produced this fit is gone; missing "
                    "same-process repeats must not be backfilled from a "
                    "reloaded state, and the cell must not be silently refit"
                ),
                "timestamp": time.time(),
            },
        )
        print(
            "INTERRUPTED: same-process repeats incomplete; stopping for human decision",
            flush=True,
        )
        return 3
    if fit_marker.exists() and len(done) >= args.max_repeats:
        print(f"cell {args.cell} already has {len(done)} repeats", flush=True)
        return 0

    if args.dry_run:
        data = load_cell(args.cell, args.model_path)
        atomic_json(
            root / "dry_run.json",
            {
                "cell": args.cell,
                "x_shape": list(data["x"].shape),
                "query_rows": int(data["q"].shape[0]),
                "manifest_sha256": data["manifest_sha256"],
                "query_sha256": data["query_sha256"],
                "no_fit_performed": True,
            },
        )
        print(f"dry-run ok: x={data['x'].shape} query={data['q'].shape}", flush=True)
        return 0

    import torch
    from tabpfn import TabPFNClassifier
    import tabpfn

    if tabpfn.__version__ != proto["inherited"]["scientific_tabpfn_version"]:
        raise SystemExit(
            f"TabPFN {tabpfn.__version__} != required "
            f"{proto['inherited']['scientific_tabpfn_version']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU fallback is prohibited")

    data = load_cell(args.cell, args.model_path)
    atomic_json(
        root / "fit_start.json",
        {
            "cell": args.cell,
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
            "query_sha256": data["query_sha256"],
            "model_path_sha256": data["model_path_sha256"],
        },
    )

    model = TabPFNClassifier(
        n_estimators=args.n_estimators,
        auto_scale_n_estimators=False,
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

    state_path = root / "model.tabpfn_fit"
    model.save_fit_state(state_path)
    state_sha = sha256_file(state_path)
    atomic_json(
        fit_marker,
        {
            "cell": args.cell,
            "process_uuid": process_uuid,
            "fit_seconds": fit_seconds,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "state_path": str(state_path),
            "state_sha256": state_sha,
            "state_bytes": state_path.stat().st_size,
            "n_estimators": args.n_estimators,
            "completed": time.time(),
        },
    )
    print(
        f"fit {args.cell}: {fit_seconds:,.1f}s peak_gpu="
        f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
        flush=True,
    )

    records: list[dict] = []
    gate_log: list[dict] = []
    final_gate: dict | None = None
    stopped_at = None
    for target in [b for b in batches if b <= args.max_repeats]:
        for r in range(len(records), target):
            rt0 = time.perf_counter()
            score = np.asarray(model.predict_proba(data["q"])[:, 1], dtype="float64")
            secs = time.perf_counter() - rt0
            if not np.all(np.isfinite(score)):
                raise AssertionError(f"non-finite scores in repeat {r}")
            ep = endpoints(score, data["q_meter"], data["q_anom"])
            rec = {
                "cell": args.cell,
                "repeat": r,
                "mode": "same_process",
                "process_uuid": process_uuid,
                "state_sha256": state_sha,
                "seconds": secs,
                "endpoints": ep,
                "score_sha256": hashlib.sha256(score.tobytes()).hexdigest(),
                "timestamp": time.time(),
            }
            atomic_json(repeats_dir / f"repeat_{r:03d}.json", rec)
            records.append(rec)
            atomic_json(
                root / "heartbeat.json",
                {
                    "cell": args.cell,
                    "status": "running",
                    "completed_repeats": r + 1,
                    "current_batch_target": target,
                    "max_repeats": args.max_repeats,
                    "process_uuid": process_uuid,
                    "last_repeat_seconds": secs,
                    "timestamp": time.time(),
                },
            )
            print(f"  repeat {r + 1}/{target} ({secs:,.1f}s)", flush=True)

        # Precision gate: only the two steam endpoints decide continuation, and
        # the escalation happens inside this same process so the repeats stay
        # same-process by construction.
        final_gate = evaluate_gate(records, bounded_target, margin_target)
        final_gate["batch_target"] = target
        gate_log.append(final_gate)
        atomic_json(root / "gate_log.json", gate_log)
        print(
            f"  gate @n={target}: auc_hw={final_gate['auc_half_width']:.6f}"
            f"<={bounded_target} {final_gate['auc_pass']} | "
            f"margin_hw={final_gate['margin_half_width']:.6f}"
            f"<={margin_target:.6f} {final_gate['margin_pass']}",
            flush=True,
        )
        if final_gate["both_pass"]:
            stopped_at = target
            break

    cell_status = (
        "COMPLETE_GATE_PASSED"
        if final_gate and final_gate["both_pass"]
        else "MEASUREMENT_UNSTABLE_AT_CAP"
    )
    atomic_json(
        root / "CELL_COMPLETE.json",
        {
            "cell": args.cell,
            "status": cell_status,
            "repeats": len(records),
            "stopped_at_batch": stopped_at,
            "gate": final_gate,
            "gate_log": gate_log,
            "process_uuid": process_uuid,
            "state_sha256": state_sha,
            "completed": time.time(),
        },
    )
    atomic_json(
        root / "heartbeat.json",
        {
            "cell": args.cell,
            "status": "completed",
            "cell_status": cell_status,
            "completed_repeats": len(records),
            "max_repeats": args.max_repeats,
            "process_uuid": process_uuid,
            "timestamp": time.time(),
        },
    )
    print(
        f"cell {args.cell} {cell_status}: {len(records)} same-process repeats",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
