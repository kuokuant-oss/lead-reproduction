"""M5 E6 runner: one state, one canonical single-process batched pass.

State-major by design. One invocation loads exactly one persisted E4 state in a
fresh process, verifies the ensemble contract and the scaler, runs the 352-row
sentinel eight times, then streams the entire 10,137,155-row holdout through one
fixed microbatch partition in one fixed order, writing every microbatch
atomically and stamping each with this process's UUID.

Nothing is fitted. The E5 no-fit guard is armed before TabPFN is imported for
use, so a reintroduced fit raises at the first call instead of quietly producing
numbers that are not a confirmation.

Resume is deliberately not offered inside a state. E3, E4 and E5 all established
that repeated inference on one fitted TabPFN state is not bitwise reproducible,
so splicing batches from two processes yields a vector that is no single
realization while looking exactly like one. A partial state is quarantined and
restarted from canonical row 0; completed states are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROWS = 10_137_155
FEATURES = 137
SENTINEL_ROWS = 352
SENTINEL_REPEATS = 8
REQUIRED_N_ESTIMATORS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> str:
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def peak_rss_gb() -> float:
    """VmHWM, the kernel's own high-water mark -- never a sampled value."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    import psutil

    return psutil.Process().memory_info().rss / 1e9


def load_scaler(spec: dict, raw_context: np.ndarray, repo_root: Path) -> Any:
    """The unit's own scaler. The arm is inherited from E4, never re-chosen."""
    import joblib

    src = spec["scaler_source"]
    if spec["scaler_arm"] == "frozen_reference":
        if src["kind"] != "persisted":
            raise SystemExit(f"{spec['unit_id']}: frozen arm is not a persisted scaler")
        path = repo_root / src["path"]
        got = sha256_file(path)
        if got != src["sha256"]:
            raise SystemExit(f"{path.name}: digest {got[:16]} is not the frozen one")
        return joblib.load(path)
    if src["kind"] != "rebuilt_and_verified":
        raise SystemExit(f"{spec['unit_id']}: cell-specific arm is not a rebuild")
    from sklearn.preprocessing import StandardScaler

    # Rebuilt exactly as E4's runner did; `verify_scaler` then rejects it unless
    # it reproduces the scaled X_train stored inside the state.
    return StandardScaler().fit(raw_context)


def assemble(journal: list[dict], parts: Path, run_uuid: str) -> np.ndarray:
    """Rebuild the state vector from its microbatch parts, fail-closed.

    Every check here exists because its absence would let a defect through
    silently: a second process UUID means the vector is a mosaic of two
    inference draws, an overlap or a gap means the fixed partition was not
    honoured, and a drifted part digest means the bytes changed after the
    process that wrote them vouched for them.
    """
    uuids = {j["process_uuid"] for j in journal}
    if uuids != {run_uuid}:
        raise SystemExit(
            f"batches carry {len(uuids)} process UUIDs; a state vector may not "
            "be spliced from more than one process"
        )
    scores = np.empty(ROWS, dtype="float32")
    seen = np.zeros(ROWS, dtype=bool)
    for j in journal:
        path = parts / j["path"]
        if sha256_file(path) != j["sha256"]:
            raise SystemExit(f"{j['path']}: digest drifted after writing")
        part = np.load(path)
        s0, s1 = j["canonical_start"], j["canonical_stop"]
        if part.shape[0] != s1 - s0:
            raise SystemExit(f"{j['path']}: {part.shape[0]} rows, expected {s1 - s0}")
        if seen[s0:s1].any():
            raise SystemExit(f"{j['path']}: overlapping rows")
        scores[s0:s1] = part
        seen[s0:s1] = True
    if not seen.all():
        raise SystemExit(f"{int((~seen).sum()):,} rows were never scored")
    if not np.all(np.isfinite(scores)):
        raise SystemExit("assembled scores contain non-finite values")
    return scores


def quarantine(state_dir: Path, reason: str) -> Path:
    dead = state_dir.with_name(f"{state_dir.name}.QUARANTINED.{time.time_ns()}")
    shutil.move(str(state_dir), str(dead))
    atomic_json(
        dead / "QUARANTINE.json",
        {
            "reason": reason,
            "quarantined_at": time.time(),
            "resume_within_state": "not permitted; restart from canonical row 0",
        },
    )
    return dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--context-cache", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run_uuid = str(uuid.uuid4())
    t_start = time.perf_counter()

    proto = read_json(args.protocol_root / "e6_protocol.json")["protocol"]
    states = read_json(args.protocol_root / "e6_state_manifest.json")["states"]
    mb_manifest = read_json(args.protocol_root / "e6_microbatch_manifest.json")
    sentinel_spec = read_json(args.protocol_root / "e6_sentinel_manifest.json")
    feat = read_json(args.feature_root / "e6_feature_manifest.json")

    spec = next((s for s in states if s["unit_id"] == args.unit_id), None)
    if spec is None:
        raise SystemExit(f"unknown unit {args.unit_id}")

    state_dir = args.out / args.unit_id
    if (state_dir / "STATE_COMPLETE.json").exists():
        print(f"{args.unit_id}: already complete, skipping")
        return 0
    if state_dir.exists():
        dead = quarantine(state_dir, "a partial state was present at launch")
        print(f"{args.unit_id}: quarantined a partial state -> {dead.name}")
    parts = state_dir / "microbatches"
    parts.mkdir(parents=True, exist_ok=True)

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    import torch
    from tabpfn import __version__ as tabpfn_version
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler

    if tabpfn_version != proto["tabpfn"]["version"]:
        raise SystemExit(f"TabPFN {tabpfn_version} != {proto['tabpfn']['version']}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU fallback is prohibited")

    # ---- inputs -----------------------------------------------------------
    matrix_path = args.feature_root / feat["path"]
    if feat["sha256"] != proto["feature_artifact"]["sha256"]:
        raise SystemExit("feature manifest digest differs from the frozen protocol")
    x = np.load(matrix_path, mmap_mode="r")
    if x.shape != (ROWS, FEATURES) or x.dtype != np.float32:
        raise SystemExit(f"feature matrix {x.shape} {x.dtype}")
    raw_index = np.load(args.feature_root / feat["raw_index_path"])

    ctx_npz = args.context_cache / f"seed{spec['context_seed']}__cell{spec['cell']}.npz"
    ctx_meta = read_json(ctx_npz.with_suffix(".json"))
    if sha256_file(ctx_npz) != ctx_meta["npz_sha256"]:
        raise SystemExit(f"{ctx_npz.name}: digest does not match its manifest")
    if ctx_meta["context_manifest_sha256"] != spec["context_manifest_sha256"]:
        raise SystemExit(f"{ctx_npz.name}: built from a different context manifest")
    with np.load(ctx_npz) as z:
        raw_context = np.asarray(z["x"])  # stored dtype; E4 scaled float32
    if raw_context.dtype != np.float32:
        raise SystemExit(f"context dtype {raw_context.dtype}; E4 scaled float32")

    state_path = args.repo_root / spec["state_path"]
    state_sha = sha256_file(state_path)
    if state_sha != spec["state_sha256"]:
        raise SystemExit(f"{args.unit_id}: state digest drifted")

    scaler = load_scaler(spec, raw_context, args.repo_root)
    scaler_check = verify_scaler(scaler, raw_context, state_path)

    t0 = time.perf_counter()
    model = load_fitted_tabpfn_model(state_path, device="cuda:0")
    reload_seconds = time.perf_counter() - t0
    ensemble = verify_ensemble(model)

    atomic_json(
        state_dir / "RUN_STARTED.json",
        {
            "unit_id": args.unit_id,
            "process_uuid": run_uuid,
            "pid": os.getpid(),
            "started": time.time(),
            "state_sha256": state_sha,
            "feature_sha256": feat["sha256"],
            "protocol_sha256": read_json(args.protocol_root / "e6_input_manifest.json")[
                "protocol_sha256"
            ],
            "scaler_verification": scaler_check,
            "ensemble": ensemble,
            "reload_seconds": reload_seconds,
            "no_fit_guard_blocked": blocked,
            "tabpfn": tabpfn_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        },
    )

    # ---- sentinel ---------------------------------------------------------
    # The 352 screening rows are a subset of the holdout, so their features are
    # sliced from the same matrix. That is only sound if the slice reproduces
    # E4's construction, so it is rebuilt once and compared rather than trusted.
    sq = np.load(args.feature_root / "e6_sentinel_query.npz")
    sent_raw = np.asarray(sq["raw_index"], dtype="int64")
    if sent_raw.size != SENTINEL_ROWS:
        raise SystemExit(f"sentinel has {sent_raw.size} rows")
    if (
        hashlib.sha256(sent_raw.tobytes()).hexdigest()
        != sentinel_spec["raw_index_sha256"]
    ):
        raise SystemExit("sentinel raw_index does not match the frozen manifest")
    order = np.argsort(raw_index)
    sent_pos = order[np.searchsorted(raw_index, sent_raw, sorter=order)]
    if not np.array_equal(raw_index[sent_pos], sent_raw):
        raise SystemExit("sentinel rows are not present in the canonical order")
    sent_raw_x = np.asarray(x[sent_pos])
    reference = np.asarray(sq["x"])
    if reference.dtype != np.float32:
        raise SystemExit("sentinel reference dtype is not float32")
    same_nan = np.array_equal(np.isnan(reference), np.isnan(sent_raw_x))
    fin = np.isfinite(reference) & np.isfinite(sent_raw_x)
    sent_diff = (
        float(np.abs(reference[fin] - sent_raw_x[fin]).max()) if fin.any() else 0.0
    )
    if not same_nan or sent_diff != 0.0:
        raise SystemExit(
            f"HARD FAILURE the sentinel slice does not reproduce E4's construction "
            f"(max |difference| = {sent_diff:.3e})"
        )

    sent_q = scaler.transform(sent_raw_x).astype("float32")
    if sent_q.dtype != np.float32:
        raise SystemExit("sentinel query upcast")
    sentinel_runs = []
    for r in range(SENTINEL_REPEATS):
        t0 = time.perf_counter()
        out = model.predict_proba(sent_q)[:, 1].astype("float64")
        sentinel_runs.append(
            {
                "repeat": r,
                "seconds": time.perf_counter() - t0,
                "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
                "mean": float(out.mean()),
                "min": float(out.min()),
                "max": float(out.max()),
                "vector": out.tolist(),
            }
        )
    digests = {r["sha256"] for r in sentinel_runs}
    atomic_json(
        state_dir / "sentinel.json",
        {
            "unit_id": args.unit_id,
            "process_uuid": run_uuid,
            "rows": SENTINEL_ROWS,
            "repeats": SENTINEL_REPEATS,
            "distinct_digests": len(digests),
            "bitwise_identical": len(digests) == 1,
            "enters_any_endpoint": False,
            "is_a_full_holdout_repeat": False,
            "runs": sentinel_runs,
        },
    )
    print(
        f"{args.unit_id}: sentinel {SENTINEL_REPEATS} repeats, "
        f"{len(digests)} distinct digests",
        flush=True,
    )

    # ---- canonical single-process batched pass ----------------------------
    entries = mb_manifest["microbatches"]
    if len(entries) != mb_manifest["census"]["microbatches_per_state"]:
        raise SystemExit("microbatch manifest is inconsistent with its own census")

    journal: list[dict] = []
    covered = 0
    t_pass = time.perf_counter()
    try:
        for i, e in enumerate(entries):
            s0, s1 = e["canonical_start"], e["canonical_stop"]
            block = np.asarray(x[s0:s1])
            if (
                hashlib.sha256(raw_index[s0:s1].tobytes()).hexdigest()
                != e["raw_index_sha256"]
            ):
                raise SystemExit(f"microbatch {i}: row identity does not match")
            q = scaler.transform(block).astype("float32")
            if q.dtype != np.float32:
                raise SystemExit(f"microbatch {i}: float64 upcast")
            t0 = time.perf_counter()
            out = model.predict_proba(q)[:, 1].astype("float32")
            dt = time.perf_counter() - t0
            if out.shape[0] != s1 - s0 or not np.all(np.isfinite(out)):
                raise SystemExit(f"microbatch {i}: invalid output")
            part = parts / f"mb_{e['shard_id']:02d}_{e['microbatch_id']:03d}.npy"
            part_sha = atomic_npy(part, out)
            journal.append(
                {
                    "index": i,
                    "shard_id": e["shard_id"],
                    "microbatch_id": e["microbatch_id"],
                    "canonical_start": s0,
                    "canonical_stop": s1,
                    "rows": s1 - s0,
                    "process_uuid": run_uuid,
                    "path": part.name,
                    "sha256": part_sha,
                    "seconds": dt,
                    "rows_per_second": (s1 - s0) / dt,
                }
            )
            covered += s1 - s0
            del block, q, out
            if i % 25 == 0 or i == len(entries) - 1:
                el = time.perf_counter() - t_pass
                rate = covered / el
                print(
                    f"  {args.unit_id} {i + 1:>4}/{len(entries)}  "
                    f"{covered:>10,}/{ROWS:,} rows  {rate:,.0f} r/s  "
                    f"eta {(ROWS - covered) / rate / 3600:5.2f}h",
                    flush=True,
                )
    except BaseException as exc:  # noqa: BLE001 - fail closed on anything
        atomic_json(
            state_dir / "INTERRUPTED_INCOMPLETE.json",
            {
                "unit_id": args.unit_id,
                "process_uuid": run_uuid,
                "error": f"{type(exc).__name__}: {exc}",
                "microbatches_written": len(journal),
                "rows_covered": covered,
                "resume_within_state": "not permitted",
                "required_action": "quarantine this state and restart it from "
                "canonical row 0; do not splice new-process output onto old",
            },
        )
        raise

    pass_seconds = time.perf_counter() - t_pass

    # ---- assembly and census ----------------------------------------------
    if covered != ROWS:
        raise SystemExit(f"covered {covered:,} rows, expected {ROWS:,}")
    scores = assemble(journal, parts, run_uuid)

    vec = state_dir / "scores.float32.npy"
    vec_sha = atomic_npy(vec, scores)
    atomic_json(state_dir / "journal.json", {"microbatches": journal})

    if sha256_file(state_path) != state_sha:
        raise SystemExit("the state file changed during the pass")
    post_ensemble = verify_ensemble(model)
    if post_ensemble != ensemble:
        raise SystemExit("the ensemble contract changed during the pass")

    shutil.rmtree(parts)

    marker = {
        "unit_id": args.unit_id,
        "process_uuid": run_uuid,
        "single_process": True,
        "estimand": proto["estimand"]["name"],
        "completed": time.time(),
        "rows_scored": int(ROWS),
        "rows_missing": 0,
        "rows_duplicated": 0,
        "microbatches": len(journal),
        "state_sha256": state_sha,
        "state_digest_stable": True,
        "feature_sha256": feat["sha256"],
        "scores_path": vec.name,
        "scores_sha256": vec_sha,
        "scores_mean": float(scores.mean()),
        "scores_min": float(scores.min()),
        "scores_max": float(scores.max()),
        "all_finite": True,
        "scaler_verification": scaler_check,
        "ensemble_before": ensemble,
        "ensemble_after": post_ensemble,
        "sentinel_complete": True,
        "reload_seconds": reload_seconds,
        "pass_seconds": pass_seconds,
        "rows_per_second": ROWS / pass_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "peak_rss_gb": peak_rss_gb(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "no_fit_guard_blocked": blocked,
    }
    atomic_json(state_dir / "STATE_COMPLETE.json", marker)
    print(
        f"{args.unit_id}: COMPLETE {ROWS:,} rows in {pass_seconds / 3600:.2f}h "
        f"({ROWS / pass_seconds:,.0f} r/s)  scores={vec_sha[:16]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
