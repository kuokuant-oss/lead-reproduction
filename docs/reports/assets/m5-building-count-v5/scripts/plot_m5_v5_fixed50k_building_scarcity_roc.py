"""Render the provisional V5 fixed-50K building-count ROC-AUC figure.

This is the V5 counterpart of ``plot_m5_v4_building_scarcity.py``.  It keeps
the visual contract of the V4 figure while enforcing the V5 scientific
identity and the frozen non-electricity holdout.

Aggregation for K=50/100/200/400
--------------------------------
The Tree line uses every complete Tree building seed at all four scarcity
budgets, so its K=50/100/200/400 points are stable once the full Tree sweep is
complete.  Row seeds are averaged within building seed before the
cross-building mean and standard error.  The TabPFN line uses only row-seed
cells that are complete for *both* models, preserving paired context identity
for every provisional TabPFN point.  One eligible row seed is sufficient for a
building-seed mean.  A point with only one available building seed is retained,
but its standard error is undefined and no error bar is drawn.

Aggregation for K=725
---------------------
K=725 has no scientific building seed.  The operational building_seed=725 is
only a path sentinel, so the plotted mean and standard error are computed
directly across row seeds 0..4.

The script is read-only with respect to all formal experiment roots.  Its only
writes are the requested figure, an audit summary, and a reporting-only metric
cache under ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


EXPERIMENT_VERSION = "m5_building_count_v5_fixed_50k"
EXPECTED_COMMIT = "4397050376b135ffd1f14d856b6e696767d2588f"
HOLDOUT_SHA256 = "a9f2dd9e49ac57c3b6fe94a12032951f6c1d4d192bcc0505a20bd943ba672f56"
HOLDOUT_ROWS = 4_102_084
CONTEXT_ROWS = 50_000
FEATURES = 137
MODEL_SEED = 42

BUDGETS = (50, 100, 200, 400, 725)
SCARCITY_BUDGETS = (50, 100, 200, 400)
BUILDING_SEEDS = (0, 1, 2, 3, 4)
ROW_SEEDS = (0, 1)
K725_ROW_SEEDS = (0, 1, 2, 3, 4)
MODELS = ("ensemble", "tabpfn")
METERS = ((1, "chilled_water", "Chilled water"), (2, "steam", "Steam"), (3, "hot_water", "Hot water"))

PAIR_IDENTITY_KEYS = (
    "experiment_version",
    "building_budget",
    "building_seed",
    "building_draw_seed",
    "row_seed",
    "features",
    "model_seed",
    "context_rows",
    "context_row_sha256",
    "context_label_sha256",
    "context_feature_matrix_sha256",
    "holdout_row_sha256",
    "source_building_manifest_sha256",
)
HOLDOUT_ARRAY_KEYS = ("validation_raw_index", "anomaly", "meter", "building_id", "site_id")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TABPFN = "#d1498b"
MODEL_STYLE = {
    "ensemble": {
        "label": "Tree Ensemble",
        "color": INK,
        "marker": "p",
        "linewidth": 1.70,
        "markersize": 3.6,
    },
    "tabpfn": {
        "label": "TabPFN",
        "color": TABPFN,
        "marker": "X",
        "linewidth": 1.28,
        "markersize": 3.4,
    },
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "m5_exp_b_building_count_meter_roc_auc_v5_fixed50k_provisional.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=here / "m5_exp_b_building_count_meter_roc_auc_v5_fixed50k_provisional_data.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=here / ".m5_v5_fixed50k_plot_cache_roc",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require all five building seeds and both row seeds at every scarcity K.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate/discover available paired cells without loading predictions.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_directory(model: str, budget: int) -> str:
    if model == "ensemble":
        return f"tree_no_es_k{budget}_f137"
    if model == "tabpfn":
        return f"tabpfn_k{budget}_f137"
    raise ValueError(model)


def cell_dir(root: Path, building_seed: int, row_seed: int, budget: int, model: str) -> Path:
    return (
        root
        / f"building_seed{building_seed}"
        / f"row_seed{row_seed}"
        / model_directory(model, budget)
    )


def is_complete(directory: Path) -> bool:
    cell = directory / "cell.json"
    complete = directory / "COMPLETE.json"
    predictions = directory / "predictions.npz"
    if not (cell.is_file() and complete.is_file() and predictions.is_file()):
        return False
    marker = read_json(complete)
    pointed = Path(marker.get("cell", ""))
    if not pointed.is_absolute():
        pointed = directory / pointed
    return pointed.resolve() == cell.resolve()


def validate_cell(directory: Path, budget: int, building_seed: int, row_seed: int, model: str) -> dict[str, Any]:
    if not is_complete(directory):
        raise RuntimeError(f"incomplete formal cell: {directory}")
    metadata = read_json(directory / "cell.json")
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "building_budget": budget,
        "building_seed": building_seed,
        "row_seed": row_seed,
        "features": FEATURES,
        "model_seed": MODEL_SEED,
        "context_rows": CONTEXT_ROWS,
        "holdout_row_sha256": HOLDOUT_SHA256,
        "holdout_rows": HOLDOUT_ROWS,
        "mode": "FORMAL",
    }
    observed = {
        "experiment_version": metadata.get("experiment_version"),
        "building_budget": int(metadata.get("building_budget", -1)),
        "building_seed": int(metadata.get("building_seed", -1)),
        "row_seed": int(metadata.get("row_seed", -1)),
        "features": int(metadata.get("features", -1)),
        "model_seed": int(metadata.get("model_seed", -1)),
        "context_rows": int(metadata.get("context_rows", -1)),
        "holdout_row_sha256": metadata.get("holdout_row_sha256"),
        "holdout_rows": int(metadata.get("holdout_rows", -1)),
        "mode": str(metadata.get("mode", "")).upper(),
    }
    if observed != expected:
        raise RuntimeError(f"cell identity mismatch in {directory}: {observed} != {expected}")
    if metadata.get("evaluation_meter_ids") != [1, 2, 3]:
        raise RuntimeError(f"evaluation meter drift in {directory}")
    if metadata.get("excluded_evaluation_meter_ids") != [0]:
        raise RuntimeError(f"excluded meter drift in {directory}")
    if model not in metadata.get("score_names", []):
        raise RuntimeError(f"missing score {model} in {directory}")
    provenance = metadata.get("provenance", {})
    if provenance.get("commit") != EXPECTED_COMMIT:
        raise RuntimeError(f"commit drift in {directory}: {provenance.get('commit')}")
    if budget == 725 and metadata.get("building_draw_seed") is not None:
        raise RuntimeError(f"K=725 must have building_draw_seed=null in {directory}")
    if model == "tabpfn":
        if int(metadata.get("n_estimators", -1)) != 8:
            raise RuntimeError(f"TabPFN estimator drift in {directory}")
        verification = metadata.get("context_verification", {})
        if verification and verification.get("sample_subsampling") is not None:
            raise RuntimeError(f"TabPFN sample subsampling enabled in {directory}")
    return metadata


def validate_pair(tree: dict[str, Any], tabpfn: dict[str, Any], label: str) -> None:
    drift = {
        key: {"tree": tree.get(key), "tabpfn": tabpfn.get(key)}
        for key in PAIR_IDENTITY_KEYS
        if tree.get(key) != tabpfn.get(key)
    }
    if drift:
        raise RuntimeError(f"paired train/test identity mismatch for {label}: {drift}")


def discover_scarcity_pairs(model_root: Path) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    selected: dict[int, dict[int, list[int]]] = {budget: {} for budget in SCARCITY_BUDGETS}
    census: dict[str, Any] = {}
    for budget in SCARCITY_BUDGETS:
        budget_census: dict[str, Any] = {}
        for building_seed in BUILDING_SEEDS:
            paired_rows: list[int] = []
            row_state: dict[str, Any] = {}
            for row_seed in ROW_SEEDS:
                tree_dir = cell_dir(model_root, building_seed, row_seed, budget, "ensemble")
                tabpfn_dir = cell_dir(model_root, building_seed, row_seed, budget, "tabpfn")
                tree_complete = is_complete(tree_dir)
                tabpfn_complete = is_complete(tabpfn_dir)
                row_state[str(row_seed)] = {
                    "tree_complete": tree_complete,
                    "tabpfn_complete": tabpfn_complete,
                    "paired_complete": tree_complete and tabpfn_complete,
                }
                if tree_complete and tabpfn_complete:
                    tree = validate_cell(tree_dir, budget, building_seed, row_seed, "ensemble")
                    tabpfn = validate_cell(tabpfn_dir, budget, building_seed, row_seed, "tabpfn")
                    validate_pair(tree, tabpfn, f"K={budget}/b={building_seed}/r={row_seed}")
                    paired_rows.append(row_seed)
            if paired_rows:
                selected[budget][building_seed] = paired_rows
            budget_census[str(building_seed)] = {
                "eligible_paired_row_seeds": paired_rows,
                "rows": row_state,
            }
        census[str(budget)] = budget_census
    return selected, census


def validate_k725(
    repo: Path,
    formal_model_root: Path,
    extension_model_root: Path,
    tabpfn_summary_path: Path,
) -> tuple[dict[str, Any], dict[int, Path]]:
    summary = read_json(tabpfn_summary_path)
    expected = {
        "status": "COMPLETE",
        "experiment_version": EXPERIMENT_VERSION,
        "budget": 725,
        "building_seed": 725,
        "building_draw_seed": None,
        "context_rows": CONTEXT_ROWS,
        "holdout_raw_index_sha256": HOLDOUT_SHA256,
        "holdout_rows": HOLDOUT_ROWS,
        "model_seed": MODEL_SEED,
        "n_estimators": 8,
        "n_features": FEATURES,
        "sample_subsampling": None,
        "row_seeds": list(K725_ROW_SEEDS),
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"K=725 TabPFN summary identity mismatch: {observed} != {expected}")

    tree_dirs: dict[int, Path] = {}
    for row_seed in K725_ROW_SEEDS:
        root = formal_model_root if row_seed < 2 else extension_model_root
        tree_dir = cell_dir(root, 725, row_seed, 725, "ensemble")
        tree = validate_cell(tree_dir, 725, 725, row_seed, "ensemble")
        tree_dirs[row_seed] = tree_dir
        if row_seed < 2:
            tabpfn_dir = cell_dir(formal_model_root, 725, row_seed, 725, "tabpfn")
            tabpfn = validate_cell(tabpfn_dir, 725, 725, row_seed, "tabpfn")
            validate_pair(tree, tabpfn, f"K=725/r={row_seed}")
        else:
            fit_result = read_json(
                repo
                / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
                / f"fitted_states/row_seed{row_seed}/fit_result.json"
            )
            audit = read_json(
                repo
                / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension"
                / f"audit/contexts/k725_full_r{row_seed}.json"
            )
            fit_expected = {
                "status": "PASSED",
                "row_seed": row_seed,
                "building_seed": 725,
                "building_draw_seed": None,
                "context_rows": CONTEXT_ROWS,
                "n_features": FEATURES,
                "model_seed": MODEL_SEED,
                "n_estimators": 8,
                "sample_subsampling": None,
            }
            fit_observed = {key: fit_result.get(key) for key in fit_expected}
            if fit_observed != fit_expected:
                raise RuntimeError(f"K=725 r{row_seed} TabPFN fit identity mismatch")
            if fit_result.get("context_feature_matrix_sha256") != tree.get("context_feature_matrix_sha256"):
                raise RuntimeError(f"K=725 r{row_seed} paired feature matrix mismatch")
            identity = audit.get("identity", {})
            if identity.get("experiment_version") != EXPERIMENT_VERSION or identity.get("row_seed") != row_seed:
                raise RuntimeError(f"K=725 r{row_seed} audit identity mismatch")
            if audit.get("raw_index_sha256") != tree.get("context_row_sha256"):
                raise RuntimeError(f"K=725 r{row_seed} context row mismatch")
            if audit.get("anomaly_sha256") != tree.get("context_label_sha256"):
                raise RuntimeError(f"K=725 r{row_seed} context label mismatch")
    return summary, tree_dirs


def cache_identity(directory: Path) -> dict[str, Any]:
    cell = directory / "cell.json"
    complete = directory / "COMPLETE.json"
    prediction = directory / "predictions.npz"
    stat = prediction.stat()
    return {
        "cell": str(cell.resolve()),
        "cell_sha256": sha256_file(cell),
        "complete_sha256": sha256_file(complete),
        "prediction": str(prediction.resolve()),
        "prediction_size": stat.st_size,
        "prediction_mtime_ns": stat.st_mtime_ns,
    }


def metric_cache_path(cache_dir: Path, metadata: dict[str, Any], model: str) -> Path:
    return cache_dir / (
        f"k{int(metadata['building_budget'])}_"
        f"b{int(metadata['building_seed'])}_"
        f"r{int(metadata['row_seed'])}_{model}.json"
    )


def load_or_compute_cell_metrics(directory: Path, model: str, cache_dir: Path) -> dict[str, Any]:
    metadata = read_json(directory / "cell.json")
    identity = cache_identity(directory)
    cache_path = metric_cache_path(cache_dir, metadata, model)
    if cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("identity") == identity:
            return cached

    prediction_path = directory / "predictions.npz"
    with np.load(prediction_path) as payload:
        missing = [key for key in (*HOLDOUT_ARRAY_KEYS, model) if key not in payload]
        if missing:
            raise RuntimeError(f"missing arrays {missing} in {prediction_path}")
        arrays = {key: np.asarray(payload[key]) for key in HOLDOUT_ARRAY_KEYS}
        scores = np.asarray(payload[model], dtype=np.float64)
    lengths = {key: len(value) for key, value in arrays.items()}
    lengths[model] = len(scores)
    if set(lengths.values()) != {HOLDOUT_ROWS}:
        raise RuntimeError(f"prediction row mismatch in {prediction_path}: {lengths}")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite scores in {prediction_path}")

    labels = arrays["anomaly"].astype(np.int8, copy=False)
    meter = arrays["meter"].astype(np.int8, copy=False)
    meter_roc_auc: dict[str, float] = {}
    for meter_id, meter_name, _ in METERS:
        keep = meter == meter_id
        y = labels[keep]
        if not 0 < int(y.sum()) < len(y):
            raise RuntimeError(f"invalid label support for meter {meter_id} in {prediction_path}")
        meter_roc_auc[meter_name] = float(roc_auc_score(y, scores[keep]))

    result = {
        "schema_version": 1,
        "identity": identity,
        "budget": int(metadata["building_budget"]),
        "building_seed": int(metadata["building_seed"]),
        "row_seed": int(metadata["row_seed"]),
        "model": model,
        "holdout_array_sha256": {key: sha256_array(value) for key, value in arrays.items()},
        "meter_roc_auc": meter_roc_auc,
    }
    atomic_json(cache_path, result)
    return result


def require_same_holdout(tree: dict[str, Any], tabpfn: dict[str, Any], label: str) -> None:
    if tree["holdout_array_sha256"] != tabpfn["holdout_array_sha256"]:
        raise RuntimeError(f"paired prediction holdout arrays mismatch for {label}")


def standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def build_scarcity_aggregates(
    model_root: Path,
    selected: dict[int, dict[int, list[int]]],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for budget in SCARCITY_BUDGETS:
        for building_seed, row_seeds in sorted(selected[budget].items()):
            per_row: dict[int, dict[str, dict[str, Any]]] = {}
            for row_seed in row_seeds:
                tree = load_or_compute_cell_metrics(
                    cell_dir(model_root, building_seed, row_seed, budget, "ensemble"),
                    "ensemble",
                    cache_dir,
                )
                tabpfn = load_or_compute_cell_metrics(
                    cell_dir(model_root, building_seed, row_seed, budget, "tabpfn"),
                    "tabpfn",
                    cache_dir,
                )
                require_same_holdout(tree, tabpfn, f"K={budget}/b={building_seed}/r={row_seed}")
                cell_records.extend((tree, tabpfn))
                per_row[row_seed] = {"ensemble": tree, "tabpfn": tabpfn}
            for _, meter_name, _ in METERS:
                seed_records.append(
                    {
                        "budget": budget,
                        "building_seed": building_seed,
                        "paired_row_seeds": row_seeds,
                        "meter": meter_name,
                        "ensemble": float(
                            np.mean([per_row[row]["ensemble"]["meter_roc_auc"][meter_name] for row in row_seeds])
                        ),
                        "tabpfn": float(
                            np.mean([per_row[row]["tabpfn"]["meter_roc_auc"][meter_name] for row in row_seeds])
                        ),
                    }
                )
    return cell_records, seed_records


def build_full_tree_scarcity(
    model_root: Path,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[int, list[int]]]]:
    """Build stable Tree points from all available scarcity Tree cells."""
    cell_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    selection: dict[int, dict[int, list[int]]] = {
        budget: {} for budget in SCARCITY_BUDGETS
    }
    for budget in SCARCITY_BUDGETS:
        for building_seed in BUILDING_SEEDS:
            row_metrics: dict[int, dict[str, Any]] = {}
            for row_seed in ROW_SEEDS:
                directory = cell_dir(
                    model_root, building_seed, row_seed, budget, "ensemble"
                )
                if not is_complete(directory):
                    continue
                validate_cell(
                    directory, budget, building_seed, row_seed, "ensemble"
                )
                record = load_or_compute_cell_metrics(
                    directory, "ensemble", cache_dir
                )
                cell_records.append(record)
                row_metrics[row_seed] = record
            if not row_metrics:
                continue
            selection[budget][building_seed] = sorted(row_metrics)
            for _, meter_name, _ in METERS:
                seed_records.append(
                    {
                        "budget": budget,
                        "building_seed": building_seed,
                        "row_seeds": sorted(row_metrics),
                        "meter": meter_name,
                        "ensemble": float(
                            np.mean(
                                [
                                    row_metrics[row_seed]["meter_roc_auc"][
                                        meter_name
                                    ]
                                    for row_seed in sorted(row_metrics)
                                ]
                            )
                        ),
                    }
                )
    return cell_records, seed_records, selection


def build_k725_seed_records(
    formal_model_root: Path,
    summary: dict[str, Any],
    tree_dirs: dict[int, Path],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for row_seed in K725_ROW_SEEDS:
        tree = load_or_compute_cell_metrics(tree_dirs[row_seed], "ensemble", cache_dir)
        cell_records.append(tree)
        if row_seed < 2:
            tabpfn = load_or_compute_cell_metrics(
                cell_dir(formal_model_root, 725, row_seed, 725, "tabpfn"),
                "tabpfn",
                cache_dir,
            )
            require_same_holdout(tree, tabpfn, f"K=725/r={row_seed}")
            cell_records.append(tabpfn)
        for meter_id, meter_name, _ in METERS:
            tabpfn_value = float(summary["per_seed_metrics"][str(row_seed)]["meters"][str(meter_id)]["roc_auc"])
            if row_seed < 2:
                computed = float(tabpfn["meter_roc_auc"][meter_name])
                if not math.isclose(tabpfn_value, computed, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"K=725 r{row_seed} TabPFN summary metric mismatch for {meter_name}")
            seed_records.append(
                {
                    "budget": 725,
                    "row_seed": row_seed,
                    "meter": meter_name,
                    "ensemble": float(tree["meter_roc_auc"][meter_name]),
                    "tabpfn": tabpfn_value,
                }
            )
    return cell_records, seed_records


def aggregate_points(
    scarcity_seed_records: list[dict[str, Any]],
    full_tree_scarcity_records: list[dict[str, Any]],
    k725_seed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for budget in SCARCITY_BUDGETS:
        for _, meter_name, _ in METERS:
            rows = [
                row
                for row in full_tree_scarcity_records
                if row["budget"] == budget and row["meter"] == meter_name
            ]
            if rows:
                values = [float(row["ensemble"]) for row in rows]
                points.append(
                    {
                        "budget": budget,
                        "meter": meter_name,
                        "model": "ensemble",
                        "mean_roc_auc": float(np.mean(values)),
                        "standard_error": standard_error(values),
                        "n": len(values),
                        "replicate_type": "full_tree_building_seed_mean",
                        "raw_by_building_seed": {
                            str(row["building_seed"]): {
                                "row_seeds": row["row_seeds"],
                                "roc_auc": float(row["ensemble"]),
                            }
                            for row in rows
                        },
                    }
                )
    for budget in SCARCITY_BUDGETS:
        for _, meter_name, _ in METERS:
            rows = [
                row
                for row in scarcity_seed_records
                if row["budget"] == budget and row["meter"] == meter_name
            ]
            if not rows:
                continue
            for model in ("tabpfn",):
                values = [float(row[model]) for row in rows]
                points.append(
                    {
                        "budget": budget,
                        "meter": meter_name,
                        "model": model,
                        "mean_roc_auc": float(np.mean(values)),
                        "standard_error": standard_error(values),
                        "n": len(values),
                        "replicate_type": "building_seed_mean",
                        "raw_by_building_seed": {
                            str(row["building_seed"]): {
                                "paired_row_seeds": row["paired_row_seeds"],
                                "roc_auc": float(row[model]),
                            }
                            for row in rows
                        },
                    }
                )
    for _, meter_name, _ in METERS:
        rows = [row for row in k725_seed_records if row["meter"] == meter_name]
        for model in MODELS:
            values = [float(row[model]) for row in rows]
            points.append(
                {
                    "budget": 725,
                    "meter": meter_name,
                    "model": model,
                    "mean_roc_auc": float(np.mean(values)),
                    "standard_error": standard_error(values),
                    "n": len(values),
                    "replicate_type": "row_seed",
                    "raw_by_row_seed": {
                        str(row["row_seed"]): float(row[model]) for row in rows
                    },
                }
            )
    return points


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def model_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [],
            [],
            color=MODEL_STYLE[model]["color"],
            marker=MODEL_STYLE[model]["marker"],
            linewidth=MODEL_STYLE[model]["linewidth"],
            markersize=MODEL_STYLE[model]["markersize"],
            label=MODEL_STYLE[model]["label"],
        )
        for model in MODELS
    ]


def selection_caption(
    selected: dict[int, dict[int, list[int]]],
    full_tree_selection: dict[int, dict[int, list[int]]],
) -> str:
    tree_counts = {budget: len(full_tree_selection[budget]) for budget in SCARCITY_BUDGETS}
    if len(set(tree_counts.values())) == 1:
        tree_field = f"Tree K=50–400 n={next(iter(tree_counts.values()))}"
    else:
        tree_field = "Tree " + "/".join(
            f"K={budget} n={tree_counts[budget]}" for budget in SCARCITY_BUDGETS
        )
    parts = []
    for budget in SCARCITY_BUDGETS:
        count = len(selected[budget])
        parts.append(f"K={budget} n={count}" if count else f"K={budget} pending")
    fields = [tree_field, "TabPFN " + parts[0], *parts[1:]]
    fields.append("K=725 both n=5 row seeds")
    return " · ".join(fields)


def render(
    points: list[dict[str, Any]],
    selected: dict[int, dict[int, list[int]]],
    full_tree_selection: dict[int, dict[int, list[int]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.05), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for index, (ax, (_, meter_name, meter_label)) in enumerate(zip(axes, METERS, strict=True)):
        style_axis(ax)
        for model in MODELS:
            rows = sorted(
                [row for row in points if row["meter"] == meter_name and row["model"] == model],
                key=lambda row: row["budget"],
            )
            if not rows:
                continue
            x = [int(row["budget"]) for row in rows]
            y = [float(row["mean_roc_auc"]) for row in rows]
            yerr = np.asarray(
                [np.nan if row["standard_error"] is None else float(row["standard_error"]) for row in rows]
            )
            style = MODEL_STYLE[model]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markeredgewidth=0.8,
                elinewidth=0.9,
                capsize=3.0,
                capthick=0.9,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlim(43, 850)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(value) for value in BUDGETS])
        ax.minorticks_off()
        # Lowest tick is 0.875; the axis extends slightly below it so that tick
        # is not flush with the frame and the lowest error bar (0.8796) clears it.
        ax.set_ylim(0.865, 1.005)
        ax.set_yticks(np.arange(0.875, 1.001, 0.025))
        ax.set_title(meter_label, loc="left", fontsize=11.3, fontweight="bold", color=INK, pad=9)
        ax.set_xlabel("Source buildings (K)", fontsize=8.9, color=SECONDARY, labelpad=7)
        if index == 0:
            ax.set_ylabel("ROC-AUC", fontsize=9.2, color=SECONDARY, labelpad=7)

    fig.suptitle(
        "Experiment B — meter-level ROC-AUC across building count",
        x=0.065,
        y=0.972,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.890,
        "Mean ± standard error · 50,000 training rows (25k anomaly / 25k normal) · 137 features.",
        ha="left",
        fontsize=9.15,
        color=SECONDARY,
    )
    fig.text(
        0.065,
        0.840,
        selection_caption(selected, full_tree_selection),
        ha="left",
        fontsize=8.45,
        color=SECONDARY,
    )
    fig.legend(
        handles=model_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.735, bottom=0.20, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    fig.savefig(temporary, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    formal_root = repo / "data/processed/m5_building_curve/v5_fixed_50k"
    model_root = formal_root / "model_runs"
    extension_root = repo / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension"
    extension_model_root = extension_root / "model_runs"
    tabpfn_summary_path = (
        repo
        / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
        / "model_results/k725_five_seed_summary.json"
    )

    selected, census = discover_scarcity_pairs(model_root)
    tabpfn_summary, tree_dirs = validate_k725(
        repo, model_root, extension_model_root, tabpfn_summary_path
    )
    if args.final:
        incomplete = {
            budget: selected[budget]
            for budget in SCARCITY_BUDGETS
            if set(selected[budget]) != set(BUILDING_SEEDS)
            or any(selected[budget][seed] != list(ROW_SEEDS) for seed in selected[budget])
        }
        if incomplete:
            raise RuntimeError(f"--final requested but scarcity cells remain incomplete: {incomplete}")

    selection_payload = {
        str(budget): {str(seed): rows for seed, rows in sorted(seeds.items())}
        for budget, seeds in selected.items()
    }
    selection_payload["725"] = {"row_seeds": list(K725_ROW_SEEDS)}
    if args.check:
        print(json.dumps({"selection": selection_payload, "census": census}, indent=2, sort_keys=True))
        return 0

    scarcity_cells, scarcity_seed_records = build_scarcity_aggregates(
        model_root, selected, args.cache_dir
    )
    full_tree_cells, full_tree_records, full_tree_selection = (
        build_full_tree_scarcity(model_root, args.cache_dir)
    )
    k725_cells, k725_seed_records = build_k725_seed_records(
        model_root, tabpfn_summary, tree_dirs, args.cache_dir
    )
    points = aggregate_points(
        scarcity_seed_records, full_tree_records, k725_seed_records
    )
    complete_final = all(
        set(selected[budget]) == set(BUILDING_SEEDS)
        and all(selected[budget][seed] == list(ROW_SEEDS) for seed in BUILDING_SEEDS)
        for budget in SCARCITY_BUDGETS
    )
    summary = {
        "schema_version": 1,
        "status": "FINAL" if complete_final else "PROVISIONAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "expected_commit": EXPECTED_COMMIT,
        "holdout_row_sha256": HOLDOUT_SHA256,
        "holdout_rows": HOLDOUT_ROWS,
        "evaluation_meter_ids": [1, 2, 3],
        "excluded_evaluation_meter_ids": [0],
        "context_rows": CONTEXT_ROWS,
        "class_balance": {"anomaly": 25_000, "normal": 25_000},
        "uncertainty": "standard error",
        "aggregation": {
            "Tree_K50_100_200_400": (
                "mean across all complete Tree row seeds within each building seed; "
                "then mean and sample-SD/sqrt(n) standard error across building-seed means"
            ),
            "TabPFN_K100_200_400": (
                "mean across currently available paired row seeds within each building seed; "
                "then mean and sample-SD/sqrt(n) standard error across building-seed means"
            ),
            "K725": "mean and sample-SD/sqrt(5) standard error across row seeds 0..4",
            "n_equals_one": "point retained; standard error undefined and omitted from figure",
        },
        "selection": selection_payload,
        "full_tree_scarcity_selection": {
            str(budget): {
                str(seed): rows for seed, rows in sorted(seeds.items())
            }
            for budget, seeds in full_tree_selection.items()
        },
        "discovery_census": census,
        "cell_metric_records": scarcity_cells + full_tree_cells + k725_cells,
        "scarcity_building_seed_records": scarcity_seed_records,
        "full_tree_scarcity_building_seed_records": full_tree_records,
        "k725_row_seed_records": k725_seed_records,
        "plot_points": points,
        "source_paths": {
            "formal_root": str(formal_root),
            "k725_extension_root": str(extension_root),
            "k725_tabpfn_summary": str(tabpfn_summary_path),
        },
    }
    atomic_json(args.summary, summary)
    render(points, selected, full_tree_selection, args.output)
    print(json.dumps({"output": str(args.output), "summary": str(args.summary), "selection": selection_payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
