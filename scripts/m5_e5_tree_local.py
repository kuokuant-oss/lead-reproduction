"""Score the 24 fixed tree comparators on the laptop, under a 24/24 identity gate.

The trees were fitted here, and only here do they reproduce E4's frozen 352-row
comparator bit for bit. The override authorises scoring them here; it does not
authorise trusting them. So every one of the 24 units must first reproduce E4's
comparator with `max_abs_diff == 0` and 352 of 352 rows exactly equal. No
sampling, no tolerance: one failure stops the whole of E5.

The 192-row feature matrix is not rebuilt here. It is the artifact gpu-host
built, transferred and digest-checked, so both hosts score the same input.

Output goes to a staged root, never straight into the canonical result root.
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

from lead import ROOT, load_m3_frame  # noqa: E402
from lead.m5_context import query_paths  # noqa: E402
from run_m5_story_ae_probe import build_feature_matrix, load_tree_runner  # noqa: E402

FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
RECOVERY = FACTORIAL / "recovery" / "states" / "trees"
QUERY_ROWS = 192
SCREENING_ROWS = 352
FEATURES = 137


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--override", type=Path, required=True)
    ap.add_argument("--query192", type=Path, required=True, help="the shared artifact")
    ap.add_argument("--staged", type=Path, required=True)
    args = ap.parse_args()

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()

    ovr = json.loads(args.override.read_text(encoding="utf-8"))
    if ovr["human_decision"] != "OPTION_A" or ovr["no_refit"] is not True:
        raise SystemExit("the override does not authorise laptop tree scoring")
    want_feature_sha = ovr["shared_input_requirement"]["sha256"]

    # The shared 192-row feature matrix, not a local rebuild.
    if sha256_file(args.query192) != want_feature_sha:
        raise SystemExit(
            "HARD FAILURE the 192-row feature artifact does not match the digest "
            "the override fixed; the laptop must score the same input as gpu-host"
        )
    with np.load(args.query192) as z:
        raw_q192 = np.asarray(z["q"], dtype="float64")
        q_index = np.asarray(z["raw_index"], dtype="int64")
    if raw_q192.shape != (QUERY_ROWS, FEATURES):
        raise SystemExit(f"192-row matrix shape {raw_q192.shape}")

    # The 352-row query, for the identity gate only.
    _, qp = query_paths(ROOT / "data" / "processed" / "m5_context_stories", "screening")
    with np.load(qp) as z:
        raw352_index = np.asarray(z["raw_index"], dtype="int64")
    frame = load_m3_frame(verbose=False)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    x352 = build_feature_matrix(holdout, raw352_index, "F4", full_frame=holdout)
    if x352.shape != (SCREENING_ROWS, FEATURES):
        raise SystemExit(f"352-row matrix shape {x352.shape}")

    specs = json.loads(args.state_manifest.read_text(encoding="utf-8"))["states"]
    runner = load_tree_runner()
    env = environment()
    args.staged.mkdir(parents=True, exist_ok=True)

    print("gate: 24 units must reproduce E4's 352-row comparator exactly\n")
    gate, records = [], []
    for spec in specs:
        uid = spec["unit_id"]
        d = (
            RECOVERY
            / f"seed{spec['context_seed']}"
            / spec["cell_dir"]
            / spec["scaler_arm"]
        )
        ens_path, sc_path = d / "tree_ensemble.joblib", d / "scaler.joblib"
        saved = joblib.load(ens_path)
        scaler = joblib.load(sc_path)

        canonical = ROOT / spec["tree_comparator"]
        with np.load(canonical, allow_pickle=True) as z:
            want = np.asarray(z["score"], dtype="float64")
            want_index = np.asarray(z["raw_index"], dtype="int64")
        if not np.array_equal(want_index, raw352_index):
            raise SystemExit(f"{uid}: comparator rows are not the screening rows")

        got352 = np.asarray(
            ensemble_predict(runner, saved, scaler.transform(x352).astype("float32")),
            dtype="float64",
        )
        diff = np.abs(got352 - want)
        exact_rows = int((diff == 0).sum())
        max_abs = float(diff.max())
        ok = max_abs == 0.0 and exact_rows == SCREENING_ROWS
        gate.append(
            {
                "unit_id": uid,
                "max_abs_diff": max_abs,
                "exact_rows": exact_rows,
                "rows": SCREENING_ROWS,
                "bit_exact": ok,
                "tree_ensemble_sha256": sha256_file(ens_path),
                "tree_scaler_sha256": sha256_file(sc_path),
                "e4_comparator_sha256": spec["tree_comparator_sha256"],
            }
        )
        print(
            f"  {'PASS' if ok else 'FAIL'} {uid:<40} "
            f"max|d|={max_abs:.3e} exact={exact_rows}/{SCREENING_ROWS}"
        )
        if not ok:
            atomic_json(args.staged / "e5_tree_gate_FAILED.json", {"gate": gate})
            raise SystemExit(
                f"HARD FAILURE {uid} is not bit-exact against E4's comparator; "
                "stopping the whole of E5 rather than applying a tolerance"
            )

    passed = sum(1 for g in gate if g["bit_exact"])
    print(f"\ngate: {passed}/24 bit-exact -- scoring the 192-row query\n")

    for spec, g in zip(specs, gate):
        uid = spec["unit_id"]
        d = (
            RECOVERY
            / f"seed{spec['context_seed']}"
            / spec["cell_dir"]
            / spec["scaler_arm"]
        )
        saved = joblib.load(d / "tree_ensemble.joblib")
        scaler = joblib.load(d / "scaler.joblib")
        score = ensemble_predict(
            runner, saved, scaler.transform(raw_q192).astype("float32")
        )
        if score.size != QUERY_ROWS or not np.all(np.isfinite(score)):
            raise SystemExit(f"{uid}: invalid 192-row tree score")
        out = args.staged / f"{uid}.npz"
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, score=score, raw_index=q_index)
        os.replace(tmp, out)
        records.append(
            {
                **g,
                "replicates": 1,
                "score_192_sha256": hashlib.sha256(
                    np.asarray(score, dtype="float64").tobytes()
                ).hexdigest(),
                "npz_sha256": sha256_file(out),
                "finite": True,
            }
        )
        print(f"  scored {uid}")

    digest = atomic_json(
        args.staged / "e5_tree_manifest.json",
        {
            "schema": "m5_e5_tree_manifest_v1",
            "execution_host": "original laptop environment",
            "override_sha256": sha256_file(args.override),
            "base_192_row_feature_sha256": want_feature_sha,
            "units": len(records),
            "refit": False,
            "retuned": False,
            "artificial_replicates": False,
            "no_fit_guard_blocked": blocked,
            "gate_units_bit_exact": passed,
            "environment": env,
            "records": records,
        },
    )
    print(f"\n{len(records)}/24 scored; manifest sha256={digest}")
    print(json.dumps(env, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
