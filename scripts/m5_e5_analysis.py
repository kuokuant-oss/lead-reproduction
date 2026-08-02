"""Analyse the imported E5 results: does E4's negative-support response replicate?

Everything is recomputed from the raw per-repeat score vectors. The estimator is
E4's, called with namespace 5005 so E5's draws are separate from E4's while the
construction stays identical.

The 192-row query has no chilledwater rows, so the chilledwater endpoints are
undefined here. They are reported as unavailable rather than silently dropped or
back-filled from another query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e4_clustered import (  # noqa: E402
    ARMS,
    CELLS,
    CONTEXT_SEEDS,
    DRAWS,
    run_cluster_bootstrap,
    seed_consistency,
)
from m5_e4_endpoints import (  # noqa: E402
    EFFECT_NAMES,
    endpoint_value,
    endpoints,
    factor_effect,
)

NAMESPACE_CODE = 5005
PRIMARY = (
    "steam_positive_vs_hotwater_negative_pairwise_auc",
    "steam_positive_minus_hotwater_negative_score_margin",
)
SECONDARY_RANK = (
    "steam_positive_global_rank",
    "steam_positive_within_meter_rank",
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def half_width(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return float("inf")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def load_results(canonical: Path, proto: dict, root: Path) -> tuple:
    qdir = root / proto["query"]["path"]
    with np.load(qdir / "queries.npz", allow_pickle=True) as z:
        query = pd.DataFrame(
            {
                "raw_index": np.asarray(z["raw_index"], dtype="int64"),
                "meter": np.asarray(z["meter"], dtype="int8"),
                "anomaly": np.asarray(z["anomaly"], dtype="int8"),
                "building_id": np.asarray(z["building_id"], dtype="int64"),
            }
        )
    specs = read_json(canonical / "e5_state_manifest.json")["states"]
    tabpfn: dict[tuple, list[np.ndarray]] = {}
    trees: dict[tuple, np.ndarray] = {}
    meta: dict[tuple, dict] = {}
    for spec in specs:
        key = (spec["context_seed"], spec["cell"], spec["scaler_arm"])
        udir = canonical / spec["unit_id"]
        complete = read_json(udir / "UNIT_COMPLETE.json")
        scores = []
        for p in sorted((udir / "repeats").glob("repeat_*.json")):
            rec = read_json(p)
            s = np.asarray(rec["score"], dtype="float64")
            if hashlib.sha256(s.tobytes()).hexdigest() != rec["score_sha256"]:
                raise SystemExit(f"{spec['unit_id']}/{p.name}: score digest drifted")
            if s.size != 192:
                raise SystemExit(f"{spec['unit_id']}/{p.name}: length {s.size}")
            scores.append(s)
        if len(scores) != 8:
            raise SystemExit(f"{spec['unit_id']}: {len(scores)} repeats")
        tabpfn[key] = scores
        tnpz = canonical / "trees" / f"{spec['unit_id']}.npz"
        with np.load(tnpz) as z:
            if not np.array_equal(
                np.asarray(z["raw_index"], dtype="int64"),
                query["raw_index"].to_numpy(),
            ):
                raise SystemExit(f"{spec['unit_id']}: tree rows are not the query rows")
            trees[key] = np.asarray(z["score"], dtype="float64")
        meta[key] = {
            "unit_id": spec["unit_id"],
            "state_sha256": complete["state_sha256"],
            "process_uuid": complete["process_uuid"],
            "load_seconds": complete.get("load_seconds"),
            "fits_performed": complete.get("fits_performed"),
            "ensemble": complete.get("ensemble", {}),
            "scaler_verification": complete.get("scaler_verification", {}),
        }
    return tabpfn, trees, meta, query


def inference_variation(tabpfn: dict, meta: dict, query: pd.DataFrame) -> dict:
    meter = query["meter"].to_numpy()
    anom = query["anomaly"].to_numpy()
    out = {}
    for key, scores in tabpfn.items():
        per_repeat = [endpoints(s, meter, anom) for s in scores]
        block = {}
        for name in per_repeat[0]:
            vals = np.array([r[name] for r in per_repeat], dtype="float64")
            if not np.all(np.isfinite(vals)):
                block[name] = {
                    "available": False,
                    "reason": "stratum absent from the 192-row query",
                }
                continue
            block[name] = {
                "available": True,
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "half_width": half_width(vals),
            }
        digests = {hashlib.sha256(s.tobytes()).hexdigest() for s in scores}
        out[meta[key]["unit_id"]] = {
            **meta[key],
            "context_seed": key[0],
            "cell": key[1],
            "cell_dir": CELL_DIR[key[1]],
            "scaler_arm": key[2],
            "distinct_score_digests": len(digests),
            "bitwise_identical_repeats": len(digests) == 1,
            "endpoints": block,
        }
    return out


def point_factorial(tabpfn: dict, trees: dict, query: pd.DataFrame, names) -> dict:
    meter = query["meter"].to_numpy()
    anom = query["anomaly"].to_numpy()
    out: dict[str, dict] = {}
    for endpoint in names:
        per_arm: dict[str, dict] = {}
        for arm in ARMS:
            per_model: dict[str, dict] = {}
            for model in ("tabpfn", "tree"):
                by_seed: dict[str, dict[int, float]] = {e: {} for e in EFFECT_NAMES}
                cell_means: dict[int, dict[str, float]] = {}
                for seed in CONTEXT_SEEDS:
                    cells = {}
                    for cell in CELLS:
                        key = (seed, cell, arm)
                        if model == "tabpfn":
                            cells[cell] = float(
                                np.mean(
                                    [
                                        endpoint_value(endpoint, s, meter, anom)
                                        for s in tabpfn[key]
                                    ]
                                )
                            )
                        else:
                            cells[cell] = float(
                                endpoint_value(endpoint, trees[key], meter, anom)
                            )
                    cell_means[seed] = cells
                    eff = factor_effect(cells)
                    for e in EFFECT_NAMES:
                        by_seed[e][seed] = eff[e]
                per_model[model] = {
                    "cell_means_by_seed": {str(s): v for s, v in cell_means.items()},
                    "effects": {e: seed_consistency(by_seed[e]) for e in EFFECT_NAMES},
                }
            per_model["tabpfn_minus_tree"] = {
                "effects": {
                    e: seed_consistency(
                        {
                            s: per_model["tabpfn"]["effects"][e]["per_seed"][str(s)]
                            - per_model["tree"]["effects"][e]["per_seed"][str(s)]
                            for s in CONTEXT_SEEDS
                        }
                    )
                    for e in EFFECT_NAMES
                }
            }
            per_arm[arm] = per_model
        per_arm["scaler_interaction"] = {
            model: {
                "effects": {
                    e: seed_consistency(
                        {
                            s: per_arm["frozen_reference"][model]["effects"][e][
                                "per_seed"
                            ][str(s)]
                            - per_arm["cell_specific"][model]["effects"][e]["per_seed"][
                                str(s)
                            ]
                            for s in CONTEXT_SEEDS
                        }
                    )
                    for e in EFFECT_NAMES
                }
            }
            for model in ("tabpfn", "tree", "tabpfn_minus_tree")
        }
        out[endpoint] = per_arm
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()

    proto = read_json(args.canonical / "e5_protocol.json")["protocol"]
    if proto["clustered_uncertainty"]["namespace_code"] != NAMESPACE_CODE:
        raise SystemExit("protocol and analysis disagree on the clustered namespace")
    tabpfn, trees, meta, query = load_results(args.canonical, proto, args.repo_root)

    print("computing inference variation (8 repeats per reloaded state)", flush=True)
    per_state = inference_variation(tabpfn, meta, query)

    names = (*PRIMARY, *SECONDARY_RANK)
    print("computing factorial point estimates", flush=True)
    factorial = point_factorial(tabpfn, trees, query, names)

    clustered = {}
    for endpoint in PRIMARY:
        for cluster_type in ("building", "segment"):
            t0 = time.perf_counter()
            print(
                f"clustered bootstrap: {endpoint} / {cluster_type} "
                f"({args.draws} draws, namespace {NAMESPACE_CODE})",
                flush=True,
            )
            clustered[f"{endpoint}__{cluster_type}"] = run_cluster_bootstrap(
                cluster_type=cluster_type,
                tabpfn=tabpfn,
                trees=trees,
                query=query,
                endpoint=endpoint,
                draws=args.draws,
                namespace=NAMESPACE_CODE,
            )
            print(f"  done in {time.perf_counter() - t0:,.0f}s", flush=True)

    summary = {
        "schema": "m5_e5_summary_v1",
        "generated": time.time(),
        "protocol_sha256": sha256_file(args.canonical / "e5_protocol.json"),
        "query": proto["query"],
        "coverage": {
            "states_reloaded": len(meta),
            "same_process_repeats": sum(len(v) for v in tabpfn.values()),
            "score_vector_length": 192,
            "tree_score_vectors": len(trees),
            "fits_performed": sum(m["fits_performed"] or 0 for m in meta.values()),
            "distinct_state_identities": len(
                {m["state_sha256"] for m in meta.values()}
            ),
            "distinct_process_uuids": len({m["process_uuid"] for m in meta.values()}),
            "effective_n_estimators_": sorted(
                {m["ensemble"].get("effective_n_estimators_") for m in meta.values()}
            ),
            "scaler_verified_exact": all(
                m["scaler_verification"].get("exact") for m in meta.values()
            ),
        },
        "chilledwater": "absent from the 192-row query; no chilledwater endpoint is "
        "computed and no other query was substituted",
        "per_state": per_state,
    }
    for name, payload in (
        ("e5_summary.json", summary),
        ("e5_factorial.json", {"schema": "m5_e5_factorial_v1", "contrasts": factorial}),
        (
            "e5_clustered.json",
            {
                "schema": "m5_e5_clustered_v1",
                "draws": args.draws,
                "namespace_code": NAMESPACE_CODE,
                "seed_mapping": proto["clustered_uncertainty"],
                "results": clustered,
            },
        ),
    ):
        digest = atomic_json(args.canonical / name, payload)
        print(f"wrote {name}  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
