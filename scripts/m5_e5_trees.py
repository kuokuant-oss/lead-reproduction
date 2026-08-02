"""Score the frozen 192-row query with E4's 24 fixed tree comparators.

The trees are reloaded, never refit. Before any 192-row score is produced, each
reloaded ensemble is required to reproduce E4's frozen 352-row prediction vector
exactly -- that is what proves the reloaded object is the same comparator E4
used, rather than merely a tree ensemble that happens to sit in the right
directory.

One score vector per unit. Trees get no artificial replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

from lead import ROOT
from run_m5_story_ae_probe import load_tree_runner

FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
RECOVERY = FACTORIAL / "recovery" / "states" / "trees"
SCREENING = ROOT / "data" / "processed" / "m5_context_stories" / "queries" / "screening"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def ensemble_predict(runner, saved: dict, x: np.ndarray) -> np.ndarray:
    """Mean probability over the ensemble, in the frozen model order."""
    return np.mean(
        [
            runner.predict_probability(name, saved["models"][name], x)
            for name in saved["model_order"]
        ],
        axis=0,
    ).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    specs = json.loads(args.state_manifest.read_text(encoding="utf-8"))["states"]
    runner = load_tree_runner()

    q_npz = args.cache_root / "query192.npz"
    q_meta = json.loads(q_npz.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(q_npz) != q_meta["npz_sha256"]:
        raise SystemExit("query192 cache digest mismatch")
    with np.load(q_npz) as z:
        raw_q192 = np.asarray(z["q"], dtype="float64")
        q_index = np.asarray(z["raw_index"], dtype="int64")

    with np.load(SCREENING / "queries.npz") as z:
        raw352_index = np.asarray(z["raw_index"], dtype="int64")

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        uid = spec["unit_id"]
        cell_dir = spec["cell_dir"]
        d = RECOVERY / f"seed{spec['context_seed']}" / cell_dir / spec["scaler_arm"]
        saved = joblib.load(d / "tree_ensemble.joblib")
        scaler = joblib.load(d / "scaler.joblib")

        # Prove this is E4's comparator by reproducing its frozen 352-row vector.
        canonical = ROOT / spec["tree_comparator"]
        with np.load(canonical, allow_pickle=True) as z:
            want = np.asarray(z["score"], dtype="float64")
            want_index = np.asarray(z["raw_index"], dtype="int64")
        if not np.array_equal(want_index, raw352_index):
            raise SystemExit(f"{uid}: canonical tree rows are not the screening rows")
        cached352 = (
            args.cache_root / f"seed{spec['context_seed']}__cell{spec['cell']}.npz"
        )
        with np.load(cached352) as z:
            raw_q352 = np.asarray(z["q"], dtype="float64")
        got = ensemble_predict(
            runner, saved, scaler.transform(raw_q352).astype("float32")
        )
        if not np.array_equal(np.asarray(got, dtype="float64"), want):
            raise SystemExit(
                f"HARD FAILURE {uid}: the reloaded tree does not reproduce E4's "
                f"352-row comparator (max |diff| "
                f"{np.abs(np.asarray(got, dtype='float64') - want).max():.3e})"
            )

        score = ensemble_predict(
            runner, saved, scaler.transform(raw_q192).astype("float32")
        )
        if score.size != 192 or not np.all(np.isfinite(score)):
            raise SystemExit(f"{uid}: invalid 192-row tree score")

        out = args.out / f"{uid}.npz"
        tmp = out.with_name(f".{out.name}.tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, score=score, raw_index=q_index)
        os.replace(tmp, out)
        rec = {
            "unit_id": uid,
            "tree_ensemble_sha256": sha256_file(d / "tree_ensemble.joblib"),
            "tree_scaler_sha256": sha256_file(d / "scaler.joblib"),
            "model_order": saved["model_order"],
            "reproduced_e4_352_row_comparator_exactly": True,
            "e4_tree_comparator_sha256": spec["tree_comparator_sha256"],
            "score_192_sha256": hashlib.sha256(
                np.asarray(score, dtype="float64").tobytes()
            ).hexdigest(),
            "npz_sha256": sha256_file(out),
            "replicates": 1,
        }
        records.append(rec)
        print(f"  {uid}: verified against E4 and scored 192 rows", flush=True)

    digest = atomic_json(
        args.out / "e5_tree_manifest.json",
        {
            "schema": "m5_e5_tree_manifest_v1",
            "units": len(records),
            "refit": False,
            "artificial_replicates": False,
            "no_fit_guard_blocked": blocked,
            "records": records,
        },
    )
    print(f"\n{len(records)}/24 tree comparators scored; manifest sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
