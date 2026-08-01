"""Analyse the imported E4 results and produce the machine-readable summaries.

Four sources of variation are computed and reported separately, because they
answer different questions and the protocol forbids conflating them:

1. inference variation   -- the 8 same-process repeats of one fit
2. context-seed variation -- across 42 / 123 / 999
3. building-clustered uncertainty
4. segment-clustered uncertainty

The clustered intervals are conditional on the fixed 24 fitted states and the
three specified context seeds. They do not contain model-seed or fresh-fit
variation, which E4 did not execute, and nothing here may be read as if they do.

Everything is recomputed from the raw per-repeat score vectors. No summary the
runner wrote is used as an input.
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

# Primary first: the two endpoints that carry the formal Path A steam claim and
# that the protocol requires to be reported together.
PRIMARY_ENDPOINTS = (
    "steam_positive_vs_hotwater_negative_pairwise_auc",
    "steam_positive_minus_hotwater_negative_score_margin",
)
SECONDARY_ENDPOINTS = (
    "chilledwater_positive_vs_chilledwater_negative_pairwise_auc",
    "chilledwater_positive_minus_chilledwater_negative_score_margin",
)


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
    """Two-sided 95% Student-t half-width on the repeat-level mean."""
    n = values.size
    if n < 2:
        return float("inf")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def load_results(canonical: Path, proto: dict, repo_root: Path) -> tuple:
    """Raw score vectors for 24 fits, the 24 tree comparators, and the query."""
    qpath = repo_root / proto["query"]["manifest"]
    npz = qpath.with_name("queries.npz")
    if sha256_file(npz) != proto["query"]["npz_sha256"]:
        raise SystemExit("query npz does not match the frozen digest")
    with np.load(npz) as z:
        query = pd.DataFrame(
            {
                "raw_index": np.asarray(z["raw_index"], dtype="int64"),
                "meter": np.asarray(z["meter"], dtype="int8"),
                "anomaly": np.asarray(z["anomaly"], dtype="int8"),
                "building_id": np.asarray(z["building_id"], dtype="int64"),
            }
        )

    specs = read_json(canonical / "e4_fit_manifest.json")["fits"]
    tabpfn: dict[tuple, list[np.ndarray]] = {}
    trees: dict[tuple, np.ndarray] = {}
    meta: dict[tuple, dict] = {}

    for spec in specs:
        key = (spec["context_seed"], spec["cell"], spec["scaler_arm"])
        root = canonical / spec["unit_id"]
        complete = read_json(root / "FIT_COMPLETE.json")
        scores = []
        for p in sorted((root / "repeats").glob("repeat_*.json")):
            rec = read_json(p)
            score = np.asarray(rec["score"], dtype="float64")
            if hashlib.sha256(score.tobytes()).hexdigest() != rec["score_sha256"]:
                raise SystemExit(f"{spec['unit_id']}/{p.name}: score digest drifted")
            scores.append(score)
        if len(scores) != 8:
            raise SystemExit(f"{spec['unit_id']}: {len(scores)} repeats, want 8")
        tabpfn[key] = scores

        tpath = repo_root / spec["tree_comparator"]
        if sha256_file(tpath) != spec["tree_comparator_sha256"]:
            raise SystemExit(f"{spec['unit_id']}: tree comparator digest drifted")
        with np.load(tpath, allow_pickle=True) as z:
            traw = np.asarray(z["raw_index"], dtype="int64")
            if not np.array_equal(traw, query["raw_index"].to_numpy()):
                raise SystemExit(
                    f"{spec['unit_id']}: tree comparator rows are not the query rows "
                    "in the query's order"
                )
            trees[key] = np.asarray(z["score"], dtype="float64")
        meta[key] = {
            "unit_id": spec["unit_id"],
            "state_sha256": complete["state_sha256"],
            "process_uuid": complete["process_uuid"],
            "fit_seconds": complete.get("fit_seconds"),
            "peak_gpu_bytes": complete.get("peak_gpu_bytes"),
            "ensemble": complete.get("ensemble", {}),
        }
    return tabpfn, trees, meta, query


def inference_variation(tabpfn: dict, meta: dict, query: pd.DataFrame) -> dict:
    """Per-fit repeat statistics: what the 8 same-process repeats alone show."""
    meter = query["meter"].to_numpy()
    anom = query["anomaly"].to_numpy()
    out = {}
    for key, scores in tabpfn.items():
        per_repeat = [endpoints(s, meter, anom) for s in scores]
        block = {}
        for name in per_repeat[0]:
            vals = np.array([r[name] for r in per_repeat], dtype="float64")
            block[name] = {
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
            "cell_dir": CELLS_DIR[key[1]],
            "scaler_arm": key[2],
            "distinct_score_digests": len(digests),
            "bitwise_identical_repeats": len(digests) == 1,
            "endpoints": block,
        }
    return out


CELLS_DIR = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}


def point_factorial(tabpfn: dict, trees: dict, query: pd.DataFrame) -> dict:
    """Full-query factorial contrasts, no resampling.

    Fit-level values come from averaging each fit's 8 repeat endpoint values --
    never from averaging the row probabilities. These are the point estimates
    the clustered intervals are built around.
    """
    meter = query["meter"].to_numpy()
    anom = query["anomaly"].to_numpy()
    out: dict[str, dict] = {}
    for endpoint in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS):
        per_arm: dict[str, dict] = {}
        for arm in ARMS:
            per_model: dict[str, dict] = {}
            for model in ("tabpfn", "tree"):
                by_seed_effects: dict[str, dict[int, float]] = {
                    e: {} for e in EFFECT_NAMES
                }
                cell_means: dict[int, dict[str, float]] = {}
                for seed in CONTEXT_SEEDS:
                    cells = {}
                    for cell in CELLS:
                        key = (seed, cell, arm)
                        if model == "tabpfn":
                            vals = [
                                endpoint_value(endpoint, s, meter, anom)
                                for s in tabpfn[key]
                            ]
                            cells[cell] = float(np.mean(vals))
                        else:
                            cells[cell] = float(
                                endpoint_value(endpoint, trees[key], meter, anom)
                            )
                    cell_means[seed] = cells
                    eff = factor_effect(cells)
                    for e in EFFECT_NAMES:
                        by_seed_effects[e][seed] = eff[e]
                per_model[model] = {
                    "cell_means_by_seed": {str(s): v for s, v in cell_means.items()},
                    "effects": {
                        e: seed_consistency(by_seed_effects[e]) for e in EFFECT_NAMES
                    },
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
        # scaler interaction, formed from the same fit-level values
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
    ap.add_argument(
        "--endpoints",
        default="primary",
        choices=("primary", "all"),
        help="which endpoints get clustered intervals",
    )
    args = ap.parse_args()

    proto = read_json(args.canonical / "e4_protocol.json")["protocol"]
    tabpfn, trees, meta, query = load_results(args.canonical, proto, args.repo_root)

    print("computing inference variation (8 repeats per fit)", flush=True)
    per_fit = inference_variation(tabpfn, meta, query)

    print("computing full-query factorial point estimates", flush=True)
    factorial = point_factorial(tabpfn, trees, query)

    chosen = PRIMARY_ENDPOINTS
    if args.endpoints == "all":
        chosen = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
    clustered = {}
    for endpoint in chosen:
        for cluster_type in ("building", "segment"):
            t0 = time.perf_counter()
            print(
                f"clustered bootstrap: {endpoint} / {cluster_type} "
                f"({args.draws} draws)",
                flush=True,
            )
            clustered[f"{endpoint}__{cluster_type}"] = run_cluster_bootstrap(
                cluster_type=cluster_type,
                tabpfn=tabpfn,
                trees=trees,
                query=query,
                endpoint=endpoint,
                draws=args.draws,
            )
            print(f"  done in {time.perf_counter() - t0:,.0f}s", flush=True)

    states = {m["state_sha256"] for m in meta.values()}
    summary = {
        "schema": "m5_e4_formal_path_a_summary_v1",
        "generated": time.time(),
        "protocol_sha256": sha256_file(args.canonical / "e4_protocol.json"),
        "realised_order_digest": proto["schedule"]["realised_order_digest"],
        "coverage": {
            "fits": len(meta),
            "same_process_repeats": sum(len(v) for v in tabpfn.values()),
            "distinct_state_identities": len(states),
            "distinct_process_uuids": len({m["process_uuid"] for m in meta.values()}),
            "effective_n_estimators_": sorted(
                {m["ensemble"].get("effective_n_estimators_") for m in meta.values()}
            ),
        },
        "variation_sources_reported_separately": [
            "inference variation: 8 same-process repeats of one fit",
            "context-seed variation: 42 / 123 / 999",
            "building-clustered uncertainty",
            "segment-clustered uncertainty",
        ],
        "clustered_interval_conditioning": proto["uncertainty_interpretation"],
        "per_fit": per_fit,
    }
    for name, payload in (
        ("e4_summary.json", summary),
        ("e4_factorial.json", {"schema": "m5_e4_factorial_v1", "contrasts": factorial}),
        (
            "e4_clustered.json",
            {
                "schema": "m5_e4_clustered_v1",
                "draws": args.draws,
                "seed_mapping": proto["clustered_uncertainty"][
                    "E_addressable_seed_mapping"
                ],
                "results": clustered,
            },
        ),
    ):
        digest = atomic_json(args.canonical / name, payload)
        print(f"wrote {name}  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
