"""Score one estimator-sweep cell and compare it to the official n=1 baseline.

Implements the four alignment proofs the sweep plan requires before any gain
may be claimed (handoff section 7.1): the assembled rows must be exactly the
site's rows inside the canonical 50/50 building holdout, aligned row-for-row on
raw_index, with labels, sites and buildings all matching, and the n=1 baseline
taken from the already-merged official artifact rather than recomputed.

Only then are ROC-AUC and PR-AUC reported, so a gain can never come from
scoring a different row set than the baseline did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from lead import PROC

DEFAULT_BASELINE = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"
SHARDS = ("head", "tail")


def load_shard_scores(shard_root: Path) -> dict[str, np.ndarray]:
    """Concatenate every durable checkpoint of one shard."""
    raw_index: list[np.ndarray] = []
    score: list[np.ndarray] = []
    anomaly: list[np.ndarray] = []
    site_id: list[np.ndarray] = []
    building_id: list[np.ndarray] = []
    for shard in SHARDS:
        chunks = sorted((shard_root / f"{shard}-results" / "chunks").glob("rows_*.npz"))
        if not chunks:
            raise FileNotFoundError(
                f"no checkpoints under {shard_root}/{shard}-results"
            )
        for path in chunks:
            with np.load(path) as payload:
                raw_index.append(np.asarray(payload["raw_index"], dtype="int64"))
                score.append(np.asarray(payload["score"], dtype="float64"))
                anomaly.append(np.asarray(payload["anomaly"], dtype="int8"))
                site_id.append(np.asarray(payload["site_id"], dtype="int8"))
                building_id.append(np.asarray(payload["building_id"], dtype="int16"))
    return {
        "raw_index": np.concatenate(raw_index),
        "score": np.concatenate(score),
        "anomaly": np.concatenate(anomaly),
        "site_id": np.concatenate(site_id),
        "building_id": np.concatenate(building_id),
    }


def prove_alignment(
    observed: dict[str, np.ndarray], baseline_path: Path, site: int
) -> dict[str, Any]:
    with np.load(baseline_path) as baseline:
        mask = np.asarray(baseline["site_id"]) == site
        expected = {
            "raw_index": np.asarray(baseline["raw_index"])[mask].astype("int64"),
            "anomaly": np.asarray(baseline["anomaly"])[mask].astype("int8"),
            "site_id": np.asarray(baseline["site_id"])[mask].astype("int8"),
            "building_id": np.asarray(baseline["building_id"])[mask].astype("int16"),
            "tabpfn": np.asarray(baseline["tabpfn"])[mask].astype("float64"),
        }

    if len(np.unique(observed["raw_index"])) != len(observed["raw_index"]):
        raise AssertionError("assembled rows contain duplicate raw_index values")
    if len(observed["raw_index"]) != len(expected["raw_index"]):
        raise AssertionError(
            f"site {site}: assembled {len(observed['raw_index'])} rows, "
            f"canonical has {len(expected['raw_index'])}"
        )
    if not np.array_equal(
        np.sort(observed["raw_index"]), np.sort(expected["raw_index"])
    ):
        raise AssertionError(f"site {site}: assembled rows are a different row set")

    order = np.argsort(observed["raw_index"])
    baseline_order = np.argsort(expected["raw_index"])
    aligned = {key: value[order] for key, value in observed.items()}
    reference = {key: value[baseline_order] for key, value in expected.items()}
    for column in ("anomaly", "site_id", "building_id"):
        if not np.array_equal(aligned[column], reference[column]):
            raise AssertionError(f"site {site}: {column} disagrees with the canonical")
    if not np.isfinite(aligned["score"]).all():
        raise AssertionError(f"site {site}: assembled scores contain non-finite values")

    return {
        "rows": int(len(aligned["raw_index"])),
        "anomalies": int(aligned["anomaly"].sum()),
        "prevalence": float(aligned["anomaly"].mean()),
        "distinct_scores": int(len(np.unique(aligned["score"]))),
        "aligned_score": aligned["score"],
        "aligned_label": aligned["anomaly"],
        "baseline_score": reference["tabpfn"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=int, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    shard_root = args.shard_root or (
        PROC / f"m5_tabpfn_site{args.site}_context100000_n{args.n_estimators}"
    )
    observed = load_shard_scores(shard_root)
    proof = prove_alignment(observed, args.baseline, args.site)

    label = proof.pop("aligned_label")
    score = proof.pop("aligned_score")
    baseline_score = proof.pop("baseline_score")
    report = {
        "site": args.site,
        "n_estimators": args.n_estimators,
        "shard_root": str(shard_root.resolve()),
        **proof,
        "baseline_n1": {
            "roc_auc": float(roc_auc_score(label, baseline_score)),
            "pr_auc": float(average_precision_score(label, baseline_score)),
        },
        "swept": {
            "roc_auc": float(roc_auc_score(label, score)),
            "pr_auc": float(average_precision_score(label, score)),
        },
    }
    report["delta"] = {
        "roc_auc": report["swept"]["roc_auc"] - report["baseline_n1"]["roc_auc"],
        "pr_auc": report["swept"]["pr_auc"] - report["baseline_n1"]["pr_auc"],
    }

    out = args.out or (
        PROC / f"m5_tabpfn_site{args.site}_n{args.n_estimators}_sweep_metrics.json"
    )
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
