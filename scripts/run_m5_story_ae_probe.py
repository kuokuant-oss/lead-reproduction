"""Run the fixed-query Story A/E composition probe.

Use an explicit dry run to resolve manifests without loading the M3 frame:

    uv run python scripts/run_m5_story_ae_probe.py --dry-run

Use ``--preflight`` to build and validate the real F0/F4 matrices and model API
without fitting anything. On the GPU runner, use ``--model tabpfn``;
``--model trees`` runs the matched CPU arm. This command only scores the fixed
screening query artifact. Full-holdout confirmation remains a separate,
resumable GPU job.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import (
    M5_CONTEXT_ROOT,
    ROOT,
    SHIFTS,
    add_value_change_features,
    array_sha256,
    load_m3_frame,
    validate_context_manifest,
)
from lead.m5_context import context_tag_path, feature_names, query_paths


TREE_RUNNER = ROOT / "scripts" / "run_m5_tree_ensemble_matched_context.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("trees", "tabpfn"), default="trees")
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="repeat for explicit manifest cells",
    )
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--out-root", type=Path, default=M5_CONTEXT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="build and validate real feature matrices and model APIs, but never fit",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.preflight:
        parser.error("--dry-run and --preflight are mutually exclusive")
    return args


def load_tree_runner():
    spec = importlib.util.spec_from_file_location(
        "m5_tree_runner_for_probe", TREE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TREE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def validate_model_api(
    model: str, model_path: Path | None, *, n_estimators: int
) -> None:
    if n_estimators < 1:
        raise ValueError("n_estimators must be >= 1")
    if model == "trees":
        runner = load_tree_runner()
        required = {"fit_frozen_models", "predict_probability", "MODEL_ORDER"}
        missing = sorted(name for name in required if not hasattr(runner, name))
        if missing:
            raise RuntimeError("tree runner API missing: " + ", ".join(missing))
        if not runner.MODEL_ORDER:
            raise RuntimeError("tree runner MODEL_ORDER is empty")
        return
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as error:
        raise RuntimeError(
            "TabPFN is not installed in the probe environment"
        ) from error
    parameters = inspect.signature(TabPFNClassifier).parameters
    required = {"device", "random_state", "fit_mode", "memory_saving_mode"}
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError("TabPFNClassifier API missing: " + ", ".join(missing))
    if model_path is None:
        raise RuntimeError(
            "--model-path is required for a reproducible TabPFN preflight"
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {model_path}")


def discover_manifests(root: Path) -> list[Path]:
    manifest_root = root / "manifests"
    paths = sorted(
        [
            *(manifest_root.glob("**/*_f0.json")),
            *(manifest_root.glob("**/*_f4.json")),
        ]
    )
    if not paths:
        raise FileNotFoundError(f"no F0/F4 context manifests under {manifest_root}")
    cells: set[tuple[str, str, int, int]] = set()
    for path in paths:
        manifest = load_json(path)
        cell = (
            str(manifest["feature_tag"]).upper(),
            str(manifest["context_tag"]),
            int(manifest["context_rows"]),
            int(manifest["context_seed"]),
        )
        if cell in cells:
            raise AssertionError(f"duplicate discovered probe cell {cell}: {path}")
        cells.add(cell)
    return paths


def counterfactual_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score movement for a fixed query across context interventions."""
    required = {"raw_index", "score", "context_tag", "model", "feature_tag"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError("sensitivity input missing: " + ", ".join(sorted(missing)))
    output: list[dict[str, Any]] = []
    group_columns = ["model", "feature_tag", "raw_index"]
    for keys, group in rows.groupby(group_columns, sort=True):
        scores = group["score"].to_numpy(dtype="float64")
        order = np.argsort(np.argsort(-scores, kind="stable"), kind="stable") + 1
        output.append(
            {
                "model": keys[0],
                "feature_tag": keys[1],
                "raw_index": int(keys[2]),
                "context_count": int(len(scores)),
                "score_std": float(np.std(scores)),
                "score_range": float(np.max(scores) - np.min(scores)),
                "rank_movement": int(np.max(order) - np.min(order)),
                "threshold_crossing": bool(
                    np.any(scores >= 0.5) and np.any(scores < 0.5)
                ),
            }
        )
    return pd.DataFrame(output)


def validate_feature_matrix(
    values: np.ndarray, *, matrix_name: str
) -> dict[str, float | int]:
    """Accept canonical missing values but reject infinities.

    The frozen M3/M5 contract leaves missing metadata, weather, and
    timestamp-merge boundary values as NaN. StandardScaler ignores them when
    fitting statistics, and both model families handle them natively.
    """
    positive_inf = int(np.isposinf(values).sum())
    negative_inf = int(np.isneginf(values).sum())
    if positive_inf or negative_inf:
        raise AssertionError(
            f"{matrix_name} contains infinities: "
            f"+inf={positive_inf}, -inf={negative_inf}"
        )
    nan_count = int(np.isnan(values).sum())
    return {
        "nan_count": nan_count,
        "nan_fraction": float(nan_count / values.size) if values.size else 0.0,
        "positive_inf": positive_inf,
        "negative_inf": negative_inf,
    }


def metrics_for_scores(
    frame: pd.DataFrame, raw_index: np.ndarray, scores: np.ndarray
) -> list[dict[str, Any]]:
    lookup = frame.iloc[raw_index]
    y = lookup["anomaly"].to_numpy(dtype="int8")
    rows: list[dict[str, Any]] = []
    groups = [("pooled", np.ones(len(lookup), dtype=bool))]
    groups.extend(
        (f"meter_{value}", lookup["meter"].to_numpy() == value)
        for value in sorted(lookup["meter"].unique())
    )
    groups.extend(
        (f"site_{value}", lookup["site_id"].to_numpy() == value)
        for value in sorted(lookup["site_id"].unique())
    )
    for group, mask in groups:
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        rows.append(
            {
                "group": group,
                "rows": int(mask.sum()),
                "positive": int(y[mask].sum()),
                "pr_auc": float(average_precision_score(y[mask], scores[mask])),
                "roc_auc": float(roc_auc_score(y[mask], scores[mask])),
            }
        )
    return rows


def build_feature_matrix(
    frame: pd.DataFrame,
    raw_index: np.ndarray,
    feature_tag: str,
    *,
    full_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    columns = feature_names(feature_tag)
    if feature_tag.upper() in {"F0", "17", "BASELINE"}:
        return frame.iloc[raw_index][columns].to_numpy(dtype="float32", copy=True)
    source = frame if full_frame is None else full_frame
    tagged = source.copy()
    tagged["__raw_index_carrier"] = tagged.index.to_numpy(dtype="int64")
    built = add_value_change_features(
        tagged, list(SHIFTS), value_change_regime="timestamp_merge"
    )
    built.index = built["__raw_index_carrier"].to_numpy(dtype="int64")
    return built.loc[raw_index, columns].to_numpy(dtype="float32", copy=True)


def fit_and_predict_trees(
    x_fit: np.ndarray, y_fit: np.ndarray, x_query: np.ndarray
) -> np.ndarray:
    runner = load_tree_runner()
    scaler = StandardScaler()
    fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
    query = scaler.transform(x_query).astype("float32", copy=False)
    models, _ = runner.fit_frozen_models(fit, pd.Series(y_fit))
    scores = [
        runner.predict_probability(name, models[name], query)
        for name in runner.MODEL_ORDER
    ]
    return np.mean(scores, axis=0).astype("float32")


def fit_and_predict_tabpfn(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_query: np.ndarray,
    *,
    model_path: Path | None,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as error:
        raise RuntimeError(
            "TabPFN is not installed; run with the M5 dependency group on GPU"
        ) from error
    scaler = StandardScaler()
    fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
    query = scaler.transform(x_query).astype("float32", copy=False)
    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "auto_scale_n_estimators": False,
        "device": "cuda",
        "random_state": seed,
        "fit_mode": "low_memory",
        "memory_saving_mode": True,
        "keep_cache_on_device": False,
        "ignore_pretraining_limits": True,
        "n_preprocessing_jobs": 1,
        "inference_config": {"SUBSAMPLE_SAMPLES": None},
        "show_progress_bar": False,
    }
    if model_path is not None:
        kwargs["model_path"] = str(model_path)
    model = TabPFNClassifier(**kwargs)
    model.fit(fit, y_fit)
    return np.asarray(model.predict_proba(query)[:, 1], dtype="float32")


def output_dir(
    root: Path,
    *,
    model: str,
    feature_tag: str,
    context_tag: str,
    context_rows: int,
    seed: int,
) -> Path:
    return (
        root
        / "predictions"
        / "screening"
        / model
        / feature_tag.lower()
        / context_tag_path(context_tag)
        / f"n{context_rows}"
        / f"seed{seed}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = (
        [Path(path) for path in args.manifest]
        if args.manifest
        else discover_manifests(args.out_root)
    )
    resolved_manifests = [(path, load_json(path)) for path in manifests]
    query_manifest_path, query_npz_path = query_paths(args.out_root, "screening")
    if args.query_manifest:
        query_manifest_path = args.query_manifest
        query_npz_path = args.query_manifest.with_name("queries.npz")
    query_manifest = load_json(query_manifest_path)
    with np.load(query_npz_path) as payload:
        query_raw = np.asarray(payload["raw_index"], dtype="int64")
        query_y = np.asarray(payload["anomaly"], dtype="int8")
    if array_sha256(query_raw) != query_manifest["raw_index_sha256"]:
        raise AssertionError("query raw-index digest mismatch")
    print(f"model={args.model}; query={len(query_raw):,} rows; cells={len(manifests)}")
    if args.dry_run:
        for path, manifest in resolved_manifests:
            print(
                f"  {path}: {manifest['feature_tag']} {manifest['context_tag']} N={manifest['context_rows']:,}"
            )
        return 0

    frame = load_m3_frame(verbose=True)
    source = frame.index.to_numpy(dtype="int64")
    if not np.array_equal(source, np.arange(len(frame), dtype="int64")):
        raise AssertionError("probe requires the frozen M3 positional raw-index frame")
    query_frame = frame.iloc[query_raw]
    if query_frame["building_id"].mod(2).eq(0).any():
        raise AssertionError("query artifact contains a fit-building row")
    if not np.array_equal(query_frame["anomaly"].to_numpy(dtype="int8"), query_y):
        raise AssertionError("query labels drifted from the frozen M3 frame")

    f4_manifests = [
        manifest
        for _, manifest in resolved_manifests
        if str(manifest["feature_tag"]).upper() == "F4"
    ]
    f4_fit_raw = np.empty(0, dtype="int64")
    f4_fit_matrix = np.empty((0, 0), dtype="float32")
    f4_query_matrix = np.empty((0, 0), dtype="float32")
    if f4_manifests:
        f4_fit_raw = np.unique(
            np.concatenate(
                [
                    np.asarray(manifest["raw_index"], dtype="int64")
                    for manifest in f4_manifests
                ]
            )
        )
        fit_frame = frame.loc[frame["building_id"] % 2 == 0]
        holdout_frame = frame.loc[frame["building_id"] % 2 == 1]
        print(
            f"building F4 features once for {len(f4_fit_raw):,} unique context "
            f"rows and {len(query_raw):,} query rows",
            flush=True,
        )
        f4_fit_matrix = build_feature_matrix(
            fit_frame,
            f4_fit_raw,
            "F4",
            full_frame=fit_frame,
        )
        f4_query_matrix = build_feature_matrix(
            holdout_frame,
            query_raw,
            "F4",
            full_frame=holdout_frame,
        )
        if f4_fit_matrix.shape != (len(f4_fit_raw), len(feature_names("F4"))):
            raise AssertionError("cached F4 fit matrix has the wrong shape")
        if f4_query_matrix.shape != (len(query_raw), len(feature_names("F4"))):
            raise AssertionError("cached F4 query matrix has the wrong shape")

    if args.preflight:
        validate_model_api(
            args.model,
            args.model_path,
            n_estimators=args.n_estimators,
        )

    all_rows: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, Any]] = []
    for manifest_path_in, manifest in resolved_manifests:
        validate_context_manifest(frame, manifest)
        context_raw = np.asarray(manifest["raw_index"], dtype="int64")
        feature_tag = manifest["feature_tag"]
        started = time.perf_counter()
        if feature_tag.upper() in {"F0", "17", "BASELINE"}:
            x_fit = build_feature_matrix(frame, context_raw, feature_tag)
            x_query = build_feature_matrix(frame, query_raw, feature_tag)
        else:
            positions = np.searchsorted(f4_fit_raw, context_raw)
            if np.any(positions >= len(f4_fit_raw)) or not np.array_equal(
                f4_fit_raw[positions], context_raw
            ):
                raise AssertionError(
                    "F4 context rows are missing from the feature cache"
                )
            x_fit = f4_fit_matrix[positions]
            x_query = f4_query_matrix
        y_fit = frame.iloc[context_raw]["anomaly"].to_numpy(dtype="int8")
        if args.preflight:
            expected_features = len(feature_names(feature_tag))
            if x_fit.shape != (int(manifest["context_rows"]), expected_features):
                raise AssertionError(
                    f"{feature_tag} fit shape {x_fit.shape} does not match manifest"
                )
            if x_query.shape != (len(query_raw), expected_features):
                raise AssertionError(
                    f"{feature_tag} query shape {x_query.shape} does not match query"
                )
            fit_missingness = validate_feature_matrix(
                x_fit, matrix_name=f"{feature_tag} fit"
            )
            query_missingness = validate_feature_matrix(
                x_query, matrix_name=f"{feature_tag} query"
            )
            if int(y_fit.sum()) * 2 != len(y_fit):
                raise AssertionError("preflight fit labels are not exactly 50/50")
            print(
                f"  preflight {manifest['context_tag']} {feature_tag}: "
                f"fit={x_fit.shape}, query={x_query.shape}, "
                f"fit_nan={fit_missingness['nan_fraction']:.2%}, "
                f"query_nan={query_missingness['nan_fraction']:.2%}",
                flush=True,
            )
            continue
        if args.model == "trees":
            scores = fit_and_predict_trees(x_fit, y_fit, x_query)
        else:
            scores = fit_and_predict_tabpfn(
                x_fit,
                y_fit,
                x_query,
                model_path=args.model_path,
                seed=int(manifest["model_seed"]),
                n_estimators=args.n_estimators,
            )
        if len(scores) != len(query_raw) or not np.isfinite(scores).all():
            raise AssertionError("probe produced invalid query scores")
        out = output_dir(
            args.out_root,
            model=args.model,
            feature_tag=feature_tag,
            context_tag=manifest["context_tag"],
            context_rows=int(manifest["context_rows"]),
            seed=int(manifest["context_seed"]),
        )
        out.mkdir(parents=True, exist_ok=True)
        context_digest = str(manifest["raw_index_sha256"])
        query_digest = str(query_manifest["raw_index_sha256"])
        np.savez_compressed(
            out / "predictions.npz",
            raw_index=query_raw,
            anomaly=query_y,
            score=scores,
            context_raw_index_sha256=np.asarray(context_digest),
            query_raw_index_sha256=np.asarray(query_digest),
        )
        rows = pd.DataFrame(
            {
                "raw_index": query_raw,
                "anomaly": query_y,
                "score": scores,
                "model": args.model,
                "feature_tag": feature_tag,
                "context_tag": manifest["context_tag"],
                "context_raw_index_sha256": context_digest,
                "query_raw_index_sha256": query_digest,
            }
        )
        rows.to_csv(out / "predictions.csv", index=False)
        metric_rows = metrics_for_scores(frame, query_raw, scores)
        all_metric_rows.extend(
            {
                "model": args.model,
                "feature_tag": feature_tag,
                "context_tag": manifest["context_tag"],
                **metric,
            }
            for metric in metric_rows
        )
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "feature_tag": feature_tag,
                    "context_tag": manifest["context_tag"],
                    "context_rows": int(manifest["context_rows"]),
                    "context_seed": int(manifest["context_seed"]),
                    "model_seed": int(manifest["model_seed"]),
                    "n_estimators": args.n_estimators,
                    "context_manifest": str(manifest_path_in),
                    "context_raw_index_sha256": context_digest,
                    "query_manifest": str(query_manifest_path),
                    "query_raw_index_sha256": query_digest,
                    "metrics": metric_rows,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        all_rows.append(rows)
        print(
            f"  {manifest['context_tag']} {feature_tag}: {len(scores):,} query scores in {time.perf_counter() - started:.1f}s"
        )
    if args.preflight:
        print(
            f"preflight passed: model={args.model}; cells={len(resolved_manifests)}; "
            "no model was fitted",
            flush=True,
        )
        return 0
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(
        args.out_root / "reports" / f"story_ae_{args.model}_query_scores.csv",
        index=False,
    )
    pd.DataFrame(all_metric_rows).to_csv(
        args.out_root / "reports" / f"story_ae_{args.model}_metrics.csv",
        index=False,
    )
    sensitivity = counterfactual_sensitivity(combined)
    sensitivity.to_csv(
        args.out_root
        / "reports"
        / f"story_ae_{args.model}_counterfactual_sensitivity.csv",
        index=False,
    )
    print(
        f"wrote {len(combined):,} row-level predictions and {len(sensitivity):,} sensitivity rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
