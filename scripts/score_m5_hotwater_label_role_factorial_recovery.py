"""Score a frozen Path-A query from persisted recovery states; never fit models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from lead import ROOT, array_sha256, load_m3_frame
from run_m5_story_ae_probe import (
    build_feature_matrix,
    load_tree_runner,
    validate_feature_matrix,
)


DEFAULT_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tabpfn", "trees"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--query", choices=("independent",), default="independent")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    recovery = args.root / "recovery"
    gate = recovery / "reproduction_gate.json"
    if not gate.is_file() or not json.loads(gate.read_text(encoding="utf-8")).get(
        "passed"
    ):
        raise RuntimeError(
            "refusing independent-query scoring before a passed reproduction gate"
        )
    query_path = args.root / "independent_query" / "queries.npz"
    with np.load(query_path) as payload:
        raw = np.asarray(payload["raw_index"], dtype="int64")
        y = np.asarray(payload["anomaly"], dtype="int8")
    digest = array_sha256(raw)
    frame = load_m3_frame(verbose=True)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    x_query = build_feature_matrix(holdout, raw, "F4", full_frame=holdout)
    validate_feature_matrix(x_query, matrix_name="frozen independent factorial query")
    for cell in sorted((recovery / "states" / args.model).glob("seed*/*/*")):
        if not cell.is_dir():
            continue
        scaler = joblib.load(cell / "scaler.joblib")
        transformed = scaler.transform(x_query).astype("float32", copy=False)
        if args.model == "tabpfn":
            from tabpfn.model_loading import load_fitted_tabpfn_model

            model = load_fitted_tabpfn_model(cell / "model.tabpfn_fit", device="cuda")
            score = np.asarray(model.predict_proba(transformed)[:, 1], dtype="float32")
        else:
            saved = joblib.load(cell / "tree_ensemble.joblib")
            runner = load_tree_runner()
            score = np.mean(
                [
                    runner.predict_probability(name, saved["models"][name], transformed)
                    for name in saved["model_order"]
                ],
                axis=0,
            ).astype("float32")
        if not np.isfinite(score).all():
            raise AssertionError(f"non-finite scores: {cell}")
        target = cell / "independent_predictions.npz"
        with target.with_name(target.name + ".tmp").open("wb") as stream:
            np.savez_compressed(
                stream,
                raw_index=raw,
                anomaly=y,
                score=score,
                query_raw_index_sha256=np.asarray(digest),
            )
        target.with_name(target.name + ".tmp").replace(target)
        (cell / "independent_result.json").write_text(
            json.dumps(
                {
                    "query": "independent",
                    "query_raw_index_sha256": digest,
                    "state_sha256": sha256_file(
                        cell
                        / (
                            "model.tabpfn_fit"
                            if args.model == "tabpfn"
                            else "tree_ensemble.joblib"
                        )
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"scored {cell.relative_to(recovery)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
