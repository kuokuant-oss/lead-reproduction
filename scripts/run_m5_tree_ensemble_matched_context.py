"""Train the frozen four-model Tree Ensemble on TabPFN's exact context rows.

This is the comparison arm the context curve needs. The published blue line was
trained on ~2.7M downsampled rows, so putting TabPFN@5k next to it compares
scarce data against abundant data -- which is not the question. The question is
whether TabPFN is better *at the same amount of labelled data*, so the trees have
to see the same N rows.

"The same N rows" is meant literally. ``nested_balanced_indices`` produces nested
prefixes, so the context TabPFN is fitted on at size N is reused here verbatim:
identical raw row ids, identical order, identical 50/50 balance. The only thing
that differs between the two curves at a given N is the model.

Two deliberate departures from ``run_m3_17_feature_ensemble.py``:

* No ``downsample_indices``. That helper emits ``[negs1, pos, negs2, pos]``, which
  duplicates every positive. At matched N duplication would hand the trees twice
  the positives TabPFN got, at the same nominal N.
* The holdout is scored in batches. Fourteen cells x 10.1M rows x 137 float32 is
  not something to hold in RAM on a 32 GB box.

``--natural-prevalence`` scores the other end of the curve: all training rows at
their real 6.7% anomaly rate. That point answers "trees with literally all the
data", but it changes both N *and* the class balance, so it is written with a
flag in the manifest and must be plotted as a separate reference, not as another
point on the matched curve.

    uv run python scripts/run_m5_tree_ensemble_matched_context.py \
        --context-rows 20000 --features 137
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    load_m3_frame,
    write_json_with_provenance,
)
from run_m3_figure_observations import (
    MODEL_ORDER,
    evaluation_summary,
    fit_frozen_models,
    frozen_model_contract,
    predict_probability,
)

CANONICAL_SCRIPT = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
CANONICAL_ORDER = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
NESTING_PROOF = PROC / "m5_tabpfn_context_nesting_proof.json"
VALUE_CHANGE_REGIME = "timestamp_merge"
# Only 676,817 positives exist in the training half, so a 50/50 context without
# replacement cannot exceed twice that. Past this point the protocol has to
# change (duplicate positives, or drop the balance), which is a second variable.
MAX_BALANCED_ROWS = 1_353_634


def load_canonical_module():
    spec = importlib.util.spec_from_file_location(
        "m5_canonical_trees", CANONICAL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="int64").astype("<i8", copy=False).tobytes()
    ).hexdigest()


def chunk_path(work_dir: Path, start: int, end: int) -> Path:
    return work_dir / f"rows_{start:09d}_{end:09d}.npz"


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write via a temp file so an interrupted write cannot look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=None)
    parser.add_argument(
        "--natural-prevalence",
        action="store_true",
        help="train on every training row at its real class balance",
    )
    parser.add_argument("--features", type=int, choices=(17, 137), default=137)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument("--predict-batch-rows", type=int, default=500_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse cached models and completed scoring chunks under <stem>.work",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.natural_prevalence == (args.context_rows is not None):
        raise SystemExit("pass exactly one of --context-rows or --natural-prevalence")
    if args.context_rows is not None:
        if args.context_rows <= 0 or args.context_rows % 2:
            raise SystemExit("--context-rows must be a positive even integer")
        if args.context_rows > MAX_BALANCED_ROWS:
            raise SystemExit(
                f"--context-rows {args.context_rows:,} exceeds the balanced ceiling "
                f"{MAX_BALANCED_ROWS:,} (only 676,817 positives exist). Use "
                "--natural-prevalence for the all-data point."
            )
    tag = (
        "allrows_natural" if args.natural_prevalence else f"context{args.context_rows}"
    )
    stem = f"m5_tree_ensemble_f{args.features}_{tag}"
    args.out = args.out or PROC / f"{stem}.json"
    args.predictions_out = args.predictions_out or PROC / f"{stem}_predictions.npz"
    args.work_dir = args.work_dir or PROC / f"{stem}.work"
    return args


def feature_columns(features: int, frame_columns: list[str]) -> list[str]:
    if features == 17:
        return list(BASELINE_FEATURE_COLS)
    value_cols = [c for c in frame_columns if c.startswith("lag_value_")]
    if len(value_cols) != 120:
        raise AssertionError(
            f"expected 120 value-change features, got {len(value_cols)}"
        )
    return list(BASELINE_FEATURE_COLS) + value_cols


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    mod = load_canonical_module()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen M3 frame", flush=True)
    frame = load_m3_frame(verbose=True)
    train_mask = (frame["building_id"] % 2 == 0).to_numpy()
    train_index = frame.index[train_mask].to_numpy(dtype="int64")

    # The holdout order is the canonical one every other artifact uses, taken
    # from the artifact itself rather than from an incidental row order here.
    with np.load(CANONICAL_ORDER) as payload:
        holdout_index = np.asarray(payload["validation_raw_index"], dtype="int64")
        holdout_y = np.asarray(payload["anomaly"], dtype="int8")
        holdout_site = np.asarray(payload["site_id"], dtype="int8")
        holdout_building = np.asarray(payload["building_id"], dtype="int16")
    if set(holdout_index.tolist()) & set(train_index.tolist()):
        raise AssertionError("holdout and training rows overlap")

    if args.natural_prevalence:
        fit_index = train_index
        context_sha = None
        print(f"Training rows: {len(fit_index):,} (natural prevalence)", flush=True)
    else:
        validation_index = mod.fixed_score_indices(
            train_index, args.validation_rows, seed=args.seed + 20_000
        )
        candidate_index = train_index[~np.isin(train_index, validation_index)]
        candidate_y = frame.loc[candidate_index, "anomaly"].to_numpy(dtype="int8")
        fit_index = mod.nested_balanced_indices(
            candidate_index, candidate_y, [args.context_rows], seed=args.seed
        )[args.context_rows]
        context_sha = array_sha256(fit_index)
        # The entire matched-N claim is that these are TabPFN's rows. Check it
        # against the digest Gate 1 recorded rather than asserting it in prose:
        # a silent divergence here would leave two curves that look comparable
        # and are not.
        if NESTING_PROOF.is_file():
            proof = json.loads(NESTING_PROOF.read_text(encoding="utf-8"))
            recorded = proof.get("contexts", {}).get(str(args.context_rows))
            if recorded is None:
                raise SystemExit(
                    f"{NESTING_PROOF} has no entry for context {args.context_rows}; "
                    "rerun verify_m5_tabpfn_context_nesting.py with this size"
                )
            if recorded["sha256"] != context_sha:
                raise AssertionError(
                    f"context {args.context_rows} digest {context_sha} does not "
                    f"match the proven TabPFN context {recorded['sha256']}"
                )
            verified = "verified against Gate 1"
        else:
            verified = f"UNVERIFIED -- {NESTING_PROOF.name} missing"
        print(
            f"Context rows: {len(fit_index):,} (sha256 {context_sha[:16]}, {verified})",
            flush=True,
        )

    print("Building timestamp-merge features for the training half", flush=True)
    train_full = add_value_change_features(
        frame.loc[train_mask], list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
    )
    columns = feature_columns(args.features, list(train_full.columns))
    x_fit = train_full.loc[fit_index, columns].to_numpy(dtype="float32")
    y_fit = frame.loc[fit_index, "anomaly"].copy()
    del train_full
    gc.collect()

    # The fitted models are a durable artifact too. Refitting four GBDTs on
    # 1.35M rows to resume a scoring pass that died at 90% is pure waste.
    model_cache = work_dir / "models.joblib"
    if args.resume and model_cache.is_file():
        payload = joblib.load(model_cache)
        scaler, models, fit_seconds = (
            payload["scaler"],
            payload["models"],
            payload["fit_seconds"],
        )
        print(f"Reusing fitted models from {model_cache}", flush=True)
        del x_fit
    else:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
        print(
            f"Fitting the frozen four-model ensemble on {x_fit.shape[0]:,} x "
            f"{x_fit.shape[1]} features",
            flush=True,
        )
        models, fit_seconds = fit_frozen_models(x_fit, y_fit)
        del x_fit
        atomic_dump(
            {"scaler": scaler, "models": models, "fit_seconds": fit_seconds},
            model_cache,
        )
    gc.collect()

    print("Building timestamp-merge features for the holdout", flush=True)
    holdout_full = add_value_change_features(
        frame.loc[~train_mask], list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
    )
    del frame
    gc.collect()

    # Durable chunks, same idiom as run_m5_tabpfn_portable_shard.py: every batch
    # lands on disk before the next one starts, so an interrupted pass resumes at
    # the last completed span instead of rescoring 10.1M rows.
    total = len(holdout_index)
    spans = [
        (start, min(total, start + args.predict_batch_rows))
        for start in range(0, total, args.predict_batch_rows)
    ]
    done = 0
    for start, end in spans:
        path = chunk_path(work_dir, start, end)
        if path.is_file():
            done += end - start
            continue
        block = scaler.transform(
            holdout_full.loc[holdout_index[start:end], columns].to_numpy(
                dtype="float32"
            )
        )
        chunk = {
            name: predict_probability(name, models[name], block).astype("float32")
            for name in MODEL_ORDER
        }
        atomic_write_npz(path, **chunk)
        del block, chunk
        gc.collect()
        done += end - start
        print(f"  scored {done:,}/{total:,} rows -> {path.name}", flush=True)
    del holdout_full
    gc.collect()

    scores = {name: np.empty(total, dtype="float32") for name in MODEL_ORDER}
    for start, end in spans:
        with np.load(chunk_path(work_dir, start, end)) as payload:
            for name in MODEL_ORDER:
                scores[name][start:end] = payload[name]
    scores["ensemble"] = np.mean([scores[name] for name in MODEL_ORDER], axis=0).astype(
        "float32"
    )
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"{name} produced non-finite scores")

    np.savez_compressed(
        args.predictions_out,
        validation_raw_index=holdout_index,
        anomaly=holdout_y,
        site_id=holdout_site,
        building_id=holdout_building,
        **scores,
    )

    metrics = {
        name: evaluation_summary(holdout_y, values) for name, values in scores.items()
    }
    print(
        f"\npooled ROC-AUC {metrics['ensemble']['roc_auc']:.4f}  "
        f"PR-AUC {metrics['ensemble']['pr_auc']:.4f}"
    )
    write_json_with_provenance(
        args.out,
        {
            "schema_version": 1,
            "experiment": "m5_tree_ensemble_matched_context",
            "features": args.features,
            "context_rows": args.context_rows,
            "natural_prevalence": args.natural_prevalence,
            "fit_rows": int(len(fit_index)),
            "fit_positive": int(y_fit.sum()),
            "context_sha256": context_sha,
            "matches_tabpfn_context": context_sha is not None,
            "holdout_rows": int(len(holdout_index)),
            "seed": args.seed,
            "fit_seconds": fit_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "model_contract": frozen_model_contract(args.seed),
            "metrics": metrics,
            "predictions": str(args.predictions_out.relative_to(ROOT).as_posix()),
        },
        root=ROOT,
    )
    print(f"Wrote {args.out} and {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
