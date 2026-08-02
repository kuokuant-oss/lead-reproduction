"""Checkpointed local-CPU executor for M5 E7 Tree units.

Normal use is deliberately split into preparation, bounded validation, formal
OOF/final fitting, label-firewalled scoring, and evaluation commands.  The
formal commands reject a dirty repository and a missing protocol freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import m5_e7_protocol as p

for name, value in p.resource_environment().items():
    os.environ[name] = value

ROOT = p.ROOT
OUT = p.ARTIFACT_ROOT
PHASE0_ARTIFACT_ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
NEUTRAL_SEEDS = {
    "n00": (110, 120),
    "n01": (210, 220),
    "n10": (310, 320),
    "n11": (410, 420),
}

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def write_status(phase: str, completed: int, total: int, detail: str) -> None:
    p.atomic_json(
        OUT / "e7_status.json",
        {
            "phase": phase,
            "completed": completed,
            "total": total,
            "detail": detail,
            "updated_unix": time.time(),
        },
    )


def _save_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)
    return p.sha256_file(path)


def _component_model(name: str):
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**p.component_params(name))
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(**p.component_params(name))
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**p.component_params(name))
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**p.component_params(name))
    raise ValueError(name)


def _predict(name: str, model: Any, matrix: np.ndarray) -> np.ndarray:
    values = (
        np.nan_to_num(matrix, nan=0.0) if name == "hist_gradient_boosting" else matrix
    )
    return model.predict_proba(values)[:, 1].astype("float32", copy=False)


def validate_unit(unit_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    marker = unit_dir / "complete.json"
    if not marker.exists():
        raise ValueError("missing completion marker")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload["spec_digest"] != p.sha256_bytes(p.canonical_json(expected)):
        raise ValueError("unit provenance drift")
    for key in ("model", "scaler", "prediction"):
        record = payload["files"][key]
        path = unit_dir / record["path"]
        if not path.exists() or p.sha256_file(path) != record["sha256"]:
            raise ValueError(f"invalid {key} checkpoint")
    scores = np.load(
        unit_dir / payload["files"]["prediction"]["path"], allow_pickle=False
    )
    if scores.shape != (expected["query_rows"],) or not np.isfinite(scores).all():
        raise ValueError("invalid prediction vector")
    return payload


def quarantine(unit_dir: Path, reason: str) -> None:
    if not unit_dir.exists():
        return
    dead = unit_dir.with_name(unit_dir.name + f".quarantine-{int(time.time())}")
    unit_dir.replace(dead)
    p.atomic_json(
        dead / "QUARANTINE.json",
        {"reason": reason, "resume": "forbidden; rebuild this atomic unit"},
    )


def worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    unit_dir = Path(spec["unit_dir"])
    try:
        validate_unit(unit_dir, spec)
        print("reused valid checkpoint", flush=True)
        return
    except (ValueError, FileNotFoundError):
        if unit_dir.exists():
            quarantine(unit_dir, "partial_or_incompatible_checkpoint")
    unit_dir.mkdir(parents=True)
    started = time.perf_counter()
    x_fit = np.load(spec["x_fit"], mmap_mode="r")
    y_fit = np.load(spec["y_fit"], mmap_mode="r")
    x_query = np.load(spec["x_query"], mmap_mode="r")
    if x_fit.shape[0] != y_fit.shape[0] or x_query.shape[0] != spec["query_rows"]:
        raise SystemExit("component input shape mismatch")
    scaler_ref = Path(spec["scaler_source"]["path"])
    if (
        not scaler_ref.exists()
        or p.sha256_file(scaler_ref) != spec["scaler_source"]["sha256"]
    ):
        raise SystemExit("frozen shared scaler is missing or digest-drifted")
    scaler = joblib.load(scaler_ref)
    fit = scaler.transform(x_fit).astype("float32", copy=False)
    query = scaler.transform(x_query).astype("float32", copy=False)
    name = spec["component"]
    model = _component_model(name)
    model.fit(
        np.nan_to_num(fit, nan=0.0) if name == "hist_gradient_boosting" else fit, y_fit
    )
    scores = _predict(name, model, query)
    files = {
        "model": {
            "path": "model.joblib",
            "sha256": _save_joblib(unit_dir / "model.joblib", model),
        },
        "scaler": {
            "path": "scaler.joblib",
            "sha256": _save_joblib(unit_dir / "scaler.joblib", scaler),
        },
        "prediction": {
            "path": "prediction.npy",
            "sha256": p.atomic_npy(unit_dir / "prediction.npy", scores),
        },
    }
    marker = {
        "unit_id": spec["unit_id"],
        "spec_digest": p.sha256_bytes(p.canonical_json(spec)),
        "files": files,
        "fit_seconds": time.perf_counter() - started,
        "complete": True,
    }
    p.atomic_json(unit_dir / "complete.json", marker)
    validate_unit(unit_dir, spec)
    print(
        json.dumps({"unit_id": spec["unit_id"], "fit_seconds": marker["fit_seconds"]}),
        flush=True,
    )


def run_component(spec: dict[str, Any]) -> dict[str, Any]:
    unit_dir = Path(spec["unit_dir"])
    try:
        return validate_unit(unit_dir, spec)
    except (ValueError, FileNotFoundError):
        pass
    spec_path = unit_dir.parent / f"{spec['unit_id']}.spec.json"
    p.atomic_json(spec_path, spec)
    creationflags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        creationflags=creationflags,
    )
    unit_dir.mkdir(parents=True, exist_ok=True)
    p.atomic_text(unit_dir / "stdout.txt", proc.stdout)
    p.atomic_text(unit_dir / "stderr.txt", proc.stderr)
    if proc.returncode:
        quarantine(unit_dir, f"subprocess_returncode_{proc.returncode}")
        raise RuntimeError(f"{spec['unit_id']} failed: {proc.stderr[-500:]}")
    return validate_unit(unit_dir, spec)


def _historical_frame() -> pd.DataFrame:
    """Load canonical labelled M3 data only for even-building fitting."""
    import lead.data as data

    data.M3 = PHASE0_ARTIFACT_ROOT / "data" / "raw" / "m3"
    frame = data.load_m3_frame(verbose=True)
    even = frame[frame["building_id"].mod(2).eq(0)].copy()
    del frame
    if even.empty or even["building_id"].mod(2).any():
        raise RuntimeError("even-only training firewall failed")
    return even


def deterministic_folds(even: pd.DataFrame) -> dict[str, Any]:
    steam = even[even["meter"].eq(2)]
    census = steam.groupby("building_id", sort=True).agg(
        rows=("anomaly", "size"),
        positives=("anomaly", "sum"),
        site_id=("site_id", "first"),
    )
    if len(census) < 5 or (census["positives"] <= 0).sum() == len(census):
        raise RuntimeError("cannot construct five non-degenerate steam folds")
    fold_rows = [
        {"rows": 0.0, "positives": 0.0, "sites": set(), "buildings": []}
        for _ in range(5)
    ]
    ordered = census.assign(
        priority=census["positives"] * 10 + census["rows"]
    ).sort_values(["priority", "building_id"], ascending=[False, True])
    positive_buildings = [
        (building, row) for building, row in ordered.iterrows() if row.positives > 0
    ]
    if len(positive_buildings) < 5:
        raise RuntimeError("fewer than five positive steam buildings")
    # Building-duration is nearly uniform, so ranked round-robin gives a much
    # stronger positive/site balance than a myopic row-count objective.
    for position, (building, row) in enumerate(ordered.iterrows()):
        chosen = position % 5
        state = fold_rows[chosen]
        state["rows"] += float(row.rows)
        state["positives"] += float(row.positives)
        state["sites"].add(int(row.site_id))
        state["buildings"].append(int(building))
    assigned = {building for state in fold_rows for building in state["buildings"]}
    for position, building in enumerate(
        sorted(set(map(int, even["building_id"].unique())) - assigned)
    ):
        fold_rows[position % 5]["buildings"].append(building)
    result: dict[str, Any] = {
        "schema_version": 1,
        "split": p.protocol()["split"],
        "folds": [],
    }
    for number, state in enumerate(fold_rows):
        buildings = np.array(sorted(state["buildings"]), dtype="int64")
        validation = even[even["building_id"].isin(buildings)]
        train = even[~even["building_id"].isin(buildings)]
        if set(validation["building_id"]) & set(train["building_id"]):
            raise RuntimeError("outer fold building leakage")
        val_steam = validation[validation["meter"].eq(2)]
        if val_steam["anomaly"].nunique() != 2:
            raise RuntimeError(f"fold {number} has degenerate validation steam labels")
        result["folds"].append(
            {
                "fold": number,
                "validation_buildings": buildings.tolist(),
                "validation_building_digest": p.array_digest(buildings),
                "validation_rows": int(len(validation)),
                "validation_steam_rows": int(len(val_steam)),
                "validation_steam_anomalies": int(val_steam["anomaly"].sum()),
                "validation_sites": sorted(map(int, validation["site_id"].unique())),
                "training_buildings": int(train["building_id"].nunique()),
                "training_rows": int(len(train)),
                "raw_index_digest": p.array_digest(
                    validation.index.to_numpy(dtype="int64")
                ),
            }
        )
    return result


def _downsample(pool: pd.DataFrame) -> np.ndarray:
    from lead.sample import downsample_indices

    result = downsample_indices(pool["anomaly"])
    if result.size != 4 * int((pool["anomaly"] == 1).sum()):
        raise RuntimeError("historical downsample cardinality drift")
    return result.astype("int64")


def support_pool(train: pd.DataFrame, cell: str) -> np.ndarray:
    positive_present, negative_present = cell
    forbidden = (
        (train["meter"] == 3) & (train["anomaly"] == 1) & (positive_present == "0")
    ) | ((train["meter"] == 3) & (train["anomaly"] == 0) & (negative_present == "0"))
    pool = train.loc[~forbidden]
    return _downsample(pool)


def neutral_pool(
    train: pd.DataFrame, support_indices: np.ndarray, slot: str
) -> np.ndarray:
    positives = int((train.loc[support_indices, "anomaly"].to_numpy() == 1).sum() // 2)
    source_pos = train.index[train["anomaly"].eq(1)].to_numpy(dtype="int64")
    source_neg = train.index[train["anomaly"].eq(0)].to_numpy(dtype="int64")
    if positives > len(source_pos) or positives > len(source_neg):
        raise RuntimeError(
            "full-support neutral pool cannot exact-match paired class counts"
        )
    seed_a, seed_b = NEUTRAL_SEEDS[slot]
    pos = np.random.RandomState(seed_a).choice(source_pos, positives, replace=False)
    neg_a = np.random.RandomState(seed_a).choice(source_neg, positives, replace=False)
    neg_b = np.random.RandomState(seed_b).choice(source_neg, positives, replace=False)
    result = np.concatenate((neg_a, pos, neg_b, pos)).astype("int64")
    if result.size != support_indices.size:
        raise RuntimeError("neutral total row count mismatch")
    return result


def pool_record(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, Any]:
    values = frame.loc[indices]
    y = values["anomaly"].to_numpy(dtype="int8")
    return {
        "rows": int(len(indices)),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
        "raw_index_digest": p.array_digest(indices),
        "row_order_digest": p.array_digest(indices),
        "meter_rows": {
            str(key): int(value)
            for key, value in values["meter"].value_counts().sort_index().items()
        },
        "meter_anomalies": {
            str(key): int(values.loc[values["meter"].eq(key), "anomaly"].sum())
            for key in sorted(values["meter"].unique())
        },
    }


def prepare_manifests() -> None:
    p.require_local_cpu()
    frame = _historical_frame()
    folds = deterministic_folds(frame)
    p.atomic_json(OUT / "e7_fold_manifest.json", folds)
    records: list[dict[str, Any]] = []
    for fold in folds["folds"]:
        validation = set(fold["validation_buildings"])
        train = frame[~frame["building_id"].isin(validation)]
        for cell in p.SUPPORT_CELLS:
            indices = support_pool(train, cell)
            records.append(
                {
                    "fold": fold["fold"],
                    "family": "support",
                    "slot": f"s{cell}",
                    **pool_record(train, indices),
                }
            )
            neutral = neutral_pool(train, indices, f"n{cell}")
            sr, nr = pool_record(train, indices), pool_record(train, neutral)
            if (sr["rows"], sr["positive_rows"], sr["negative_rows"]) != (
                nr["rows"],
                nr["positive_rows"],
                nr["negative_rows"],
            ):
                raise RuntimeError("paired neutral class-count gate failed")
            records.append(
                {
                    "fold": fold["fold"],
                    "family": "neutral",
                    "slot": f"n{cell}",
                    "paired_support": f"s{cell}",
                    **nr,
                }
            )
    p.atomic_json(
        OUT / "e7_training_pool_manifest.json",
        {
            "schema_version": 1,
            "records": records,
            "all_even_historical_rows": int(len(frame)),
            "odd_labels_used": 0,
        },
    )


def bounded_validation() -> None:
    """Run every expensive type on deterministic synthetic caps only."""
    p.require_local_cpu()
    root = OUT / "validation_only"
    rng = np.random.RandomState(20260802)
    x = rng.normal(size=(80, 137)).astype("float32")
    y = np.array([0, 1] * 40, dtype="int8")
    q = rng.normal(size=(24, 137)).astype("float32")
    for name in p.MODEL_ORDER:
        unit = p.unit_id("validation", "synthetic", "synthetic", "s11", name)
        inputs = root / "inputs"
        p.atomic_npy(inputs / "x.npy", x)
        p.atomic_npy(inputs / "y.npy", y)
        p.atomic_npy(inputs / "q.npy", q)
        scaler = StandardScaler().fit(x)
        scaler_path = inputs / "shared_scaler.joblib"
        scaler_sha = _save_joblib(scaler_path, scaler)
        spec = {
            "unit_id": unit,
            "unit_dir": str(root / "units" / unit),
            "component": name,
            "x_fit": str(inputs / "x.npy"),
            "y_fit": str(inputs / "y.npy"),
            "x_query": str(inputs / "q.npy"),
            "query_rows": len(q),
            "scaler_source": {"path": str(scaler_path), "sha256": scaler_sha},
            "mode": "bounded_non_scientific_validation",
        }
        run_component(spec)
    p.atomic_json(
        root / "complete.json",
        {
            "complete": True,
            "mode": "bounded_non_scientific_validation",
            "components": list(p.MODEL_ORDER),
            "fit_rows": 80,
            "query_rows": 24,
        },
    )


def freeze() -> None:
    p.require_local_cpu()
    required = [
        OUT / "e7_fold_manifest.json",
        OUT / "e7_training_pool_manifest.json",
        OUT / "validation_only" / "complete.json",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise SystemExit("cannot freeze protocol; missing " + ", ".join(missing))
    p.atomic_json(OUT / "e7_protocol.json", p.protocol())
    p.atomic_json(
        OUT / "e7_execution_handoff.json",
        {
            "phase0_commit": p.PHASE0_COMMIT,
            "verified": True,
            "remote_commands": 0,
            "tabpfn_calls": 0,
            "active_e6_files_read": 0,
            "odd_e7_predictions": 0,
            "odd_labels_used_by_pipeline": 0,
        },
    )
    p.atomic_json(OUT / "e7_environment_manifest.json", p.environment_manifest())
    p.atomic_json(
        OUT / "e7_input_manifest.json",
        {
            "source_root": str(PHASE0_ARTIFACT_ROOT),
            "source_map_sha256": p.sha256_file(OUT / "e7_source_map.json"),
            "fold_manifest_sha256": p.sha256_file(OUT / "e7_fold_manifest.json"),
            "training_pool_manifest_sha256": p.sha256_file(
                OUT / "e7_training_pool_manifest.json"
            ),
        },
    )
    p.atomic_json(
        OUT / "e7_support_expert_manifest.json",
        {"slots": [f"s{x}" for x in p.SUPPORT_CELLS], "components_per_expert": 4},
    )
    p.atomic_json(
        OUT / "e7_neutral_expert_manifest.json",
        {
            "slots": list(p.NEUTRAL_SLOTS),
            "sampling_seeds": {
                key: list(value) for key, value in NEUTRAL_SEEDS.items()
            },
        },
    )
    p.atomic_json(
        OUT / "e7_oof_manifest.json",
        {
            "expected_component_fits": p.EXPECTED_OOF_COMPONENTS,
            "expected_ensembles": 40,
            "complete": False,
        },
    )
    p.atomic_json(
        OUT / "e7_meta_model_manifest.json",
        {
            "selected_support_c": None,
            "selected_neutral_c": None,
            "selection_status": "not_run",
            "contract": p.protocol()["meta_model"],
        },
    )
    p.atomic_json(OUT / "e7_bootstrap_manifest.json", p.protocol()["bootstrap"])
    p.atomic_json(OUT / "e7_decision_rules.json", p.protocol()["decision"])


def _f4(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Build frozen timestamp-merge F4 values in requested raw-index order."""
    from lead.data import SHIFTS
    from lead.features import add_value_change_features
    from lead.m5_context import feature_names

    columns = feature_names("F4")
    if len(columns) != 137:
        raise RuntimeError("F4 feature count drift")
    tagged = frame.copy()
    tagged["__raw_index_carrier"] = tagged.index.to_numpy(dtype="int64")
    built = add_value_change_features(
        tagged, list(SHIFTS), value_change_regime="timestamp_merge"
    )
    built.index = built["__raw_index_carrier"].to_numpy(dtype="int64")
    matrix = built.loc[indices, columns].to_numpy(dtype="float32", copy=True)
    if matrix.shape != (len(indices), 137) or matrix.dtype != np.float32:
        raise RuntimeError("F4 matrix contract drift")
    return matrix


def _pool_manifest_record(fold: int, family: str, slot: str) -> dict[str, Any]:
    records = json.loads(
        (OUT / "e7_training_pool_manifest.json").read_text(encoding="utf-8")
    )["records"]
    matches = [
        r
        for r in records
        if r["fold"] == fold and r["family"] == family and r["slot"] == slot
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"missing or duplicate frozen pool record: {fold}/{family}/{slot}"
        )
    return matches[0]


def _frozen_indices(
    train: pd.DataFrame, fold: int, family: str, slot: str
) -> np.ndarray:
    indices = (
        support_pool(train, slot[1:])
        if family == "support"
        else neutral_pool(train, support_pool(train, slot[1:]), slot)
    )
    record = _pool_manifest_record(fold, family, slot)
    observed = pool_record(train, indices)
    for key in (
        "rows",
        "positive_rows",
        "negative_rows",
        "raw_index_digest",
        "row_order_digest",
    ):
        if observed[key] != record[key]:
            raise RuntimeError(f"frozen pool drift for {fold}/{family}/{slot}: {key}")
    return indices


def _unit_spec(
    *,
    phase: str,
    fold_name: str,
    family: str,
    slot: str,
    component: str,
    inputs: Path,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_query: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, Any]:
    identity = p.unit_id(phase, fold_name, family, slot, component)
    slot_root = inputs / f"{phase}__{fold_name}__{family}__{slot}"
    x_path, y_path, q_path = (
        slot_root / "x.npy",
        slot_root / "y.npy",
        slot_root / "query.npy",
    )
    p.atomic_npy(x_path, x_fit)
    p.atomic_npy(y_path, y_fit.astype("int8", copy=False))
    p.atomic_npy(q_path, x_query)
    scaler_path = slot_root / "shared_scaler.joblib"
    scaler_sha = _save_joblib(scaler_path, scaler)
    return {
        "unit_id": identity,
        "phase": phase,
        "fold": fold_name,
        "family": family,
        "slot": slot,
        "component": component,
        "unit_dir": str(OUT / "units" / identity),
        "x_fit": str(x_path),
        "y_fit": str(y_path),
        "x_query": str(q_path),
        "query_rows": int(len(x_query)),
        "training_raw_index_digest": p.array_digest(train_indices),
        "training_row_order_digest": p.array_digest(train_indices),
        "validation_raw_index_digest": p.array_digest(query_indices),
        "feature_column_digest": "eab08c9a1b39ee7a74070c2ded464e3efbfb13d89dc148ef6b3bc3f443b21de3",
        "x_dtype": "float32",
        "environment_sha256": p.sha256_file(OUT / "e7_environment_manifest.json"),
        "protocol_sha256": p.sha256_file(OUT / "e7_protocol.json"),
        "scaler_source": {"path": str(scaler_path), "sha256": scaler_sha},
    }


def execute_oof_fold(fold: int) -> None:
    """Fit the frozen 32 components for one OOF fold; no odd rows are loaded."""
    p.require_local_cpu()
    write_status("oof", 0, 160, f"preparing fold {fold}")
    manifest = json.loads((OUT / "e7_fold_manifest.json").read_text(encoding="utf-8"))[
        "folds"
    ][fold]
    frame = _historical_frame()
    validation_buildings = set(manifest["validation_buildings"])
    train = frame[~frame["building_id"].isin(validation_buildings)].copy()
    validation = frame[frame["building_id"].isin(validation_buildings)].copy()
    if set(train["building_id"]) & set(validation["building_id"]):
        raise RuntimeError("validation-building any-meter leakage")
    if train["building_id"].mod(2).any() or validation["building_id"].mod(2).any():
        raise RuntimeError("odd building entered OOF")
    steam = validation[validation["meter"].eq(2)]
    query_indices = steam.index.to_numpy(dtype="int64")
    query = _f4(validation, query_indices)
    inputs = OUT / "inputs"
    predictions: dict[str, np.ndarray] = {}
    for family, slots in (
        ("support", [f"s{x}" for x in p.SUPPORT_CELLS]),
        ("neutral", list(NEUTRAL_SEEDS)),
    ):
        for slot in slots:
            indices = _frozen_indices(train, fold, family, slot)
            x_fit = _f4(train, indices)
            y_fit = train.loc[indices, "anomaly"].to_numpy(dtype="int8")
            scaler = StandardScaler().fit(x_fit)
            component_scores = []
            for component in p.MODEL_ORDER:
                spec = _unit_spec(
                    phase="oof",
                    fold_name=f"fold{fold}",
                    family=family,
                    slot=slot,
                    component=component,
                    inputs=inputs,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    x_query=query,
                    train_indices=indices,
                    query_indices=query_indices,
                    scaler=scaler,
                )
                result = run_component(spec)
                component_scores.append(
                    np.load(
                        Path(spec["unit_dir"]) / result["files"]["prediction"]["path"],
                        allow_pickle=False,
                    )
                )
            ensemble = np.mean(component_scores, axis=0, dtype="float64").astype(
                "float32"
            )
            if not np.isfinite(ensemble).all():
                raise RuntimeError("non-finite OOF ensemble")
            predictions[slot] = ensemble
            p.atomic_npy(OUT / "oof" / f"fold{fold}" / f"{slot}.npy", ensemble)
    p.atomic_npy(OUT / "oof" / f"fold{fold}" / "raw_index.npy", query_indices)
    p.atomic_npy(
        OUT / "oof" / f"fold{fold}" / "labels.npy",
        steam["anomaly"].to_numpy(dtype="int8"),
    )
    p.atomic_json(
        OUT / "oof" / f"fold{fold}" / "complete.json",
        {
            "fold": fold,
            "query_rows": int(len(query_indices)),
            "query_raw_index_digest": p.array_digest(query_indices),
            "slots": {
                key: p.sha256_file(OUT / "oof" / f"fold{fold}" / f"{key}.npy")
                for key in predictions
            },
            "complete": True,
            "odd_predictions": 0,
        },
    )


def execute_oof() -> None:
    for fold in range(5):
        execute_oof_fold(fold)


def _select_c(
    features: list[np.ndarray], labels: list[np.ndarray]
) -> tuple[float, dict[str, list[float]], list[np.ndarray]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    outcomes: dict[str, list[float]] = {}
    predictions_by_c: dict[float, list[np.ndarray]] = {}
    for value in p.CS:
        blocks: list[np.ndarray] = []
        scores: list[float] = []
        for held in range(5):
            train_x = np.concatenate([features[i] for i in range(5) if i != held])
            train_y = np.concatenate([labels[i] for i in range(5) if i != held])
            scaler = StandardScaler().fit(train_x)
            model = LogisticRegression(
                C=value,
                solver="lbfgs",
                max_iter=5000,
                class_weight=None,
                random_state=42,
            )
            model.fit(scaler.transform(train_x), train_y)
            score = model.predict_proba(scaler.transform(features[held]))[:, 1].astype(
                "float32"
            )
            blocks.append(score)
            scores.append(float(average_precision_score(labels[held], score)))
        outcomes[str(value)] = scores
        predictions_by_c[value] = blocks
    means = {float(key): float(np.mean(value)) for key, value in outcomes.items()}
    best = max(means.values())
    chosen = min(value for value in p.CS if best - means[value] < 1e-4)
    return chosen, outcomes, predictions_by_c[chosen]


def finalise_oof() -> None:
    from sklearn.metrics import average_precision_score

    raw_blocks, labels, support, neutral = [], [], [], []
    for fold in range(5):
        root = OUT / "oof" / f"fold{fold}"
        marker = root / "complete.json"
        if not marker.exists():
            raise SystemExit(f"cannot finalise OOF: missing fold {fold}")
        values = {
            slot: np.load(root / f"{slot}.npy", allow_pickle=False)
            for slot in [f"s{x}" for x in p.SUPPORT_CELLS] + list(NEUTRAL_SEEDS)
        }
        count = len(values["s11"])
        if any(
            len(value) != count or not np.isfinite(value).all()
            for value in values.values()
        ):
            raise SystemExit(f"invalid OOF score field in fold {fold}")
        raw_blocks.append(np.load(root / "raw_index.npy", allow_pickle=False))
        labels.append(np.load(root / "labels.npy", allow_pickle=False))
        support.append(p.factor_features(values, "s"))
        neutral.append(p.factor_features(values, "n"))
    support_c, support_grid, support_stack = _select_c(support, labels)
    neutral_c, neutral_grid, neutral_stack = _select_c(neutral, labels)
    rows = []
    for fold in range(5):
        s11 = support[fold][:, 0]
        rows.append(
            {
                "fold": fold,
                "rows": int(len(labels[fold])),
                "prevalence": float(labels[fold].mean()),
                "s11_ap": float(average_precision_score(labels[fold], s11)),
                "support_stack_ap": float(
                    average_precision_score(labels[fold], support_stack[fold])
                ),
                "neutral_stack_ap": float(
                    average_precision_score(labels[fold], neutral_stack[fold])
                ),
            }
        )
        p.atomic_npy(
            OUT / "oof" / f"fold{fold}" / "support_stack.npy", support_stack[fold]
        )
        p.atomic_npy(
            OUT / "oof" / f"fold{fold}" / "neutral_stack.npy", neutral_stack[fold]
        )
    summary = {
        "complete": True,
        "component_fits": 160,
        "odd_predictions": 0,
        "support_c": support_c,
        "neutral_c": neutral_c,
        "support_grid_fold_aps": support_grid,
        "neutral_grid_fold_aps": neutral_grid,
        "folds": rows,
        "oof_raw_index_digest": p.array_digest(np.concatenate(raw_blocks)),
    }
    p.atomic_json(OUT / "e7_oof_summary.json", summary)
    p.atomic_json(
        OUT / "e7_oof_manifest.json",
        {
            "expected_component_fits": 160,
            "completed_component_fits": 160,
            "expected_ensembles": 40,
            "complete": True,
            "summary_sha256": p.sha256_file(OUT / "e7_oof_summary.json"),
        },
    )
    p.atomic_json(
        OUT / "e7_meta_model_manifest.json",
        {
            "selected_support_c": support_c,
            "selected_neutral_c": neutral_c,
            "selection_status": "complete_even_oof_only",
            "contract": p.protocol()["meta_model"],
            "oof_summary_sha256": p.sha256_file(OUT / "e7_oof_summary.json"),
        },
    )


def freeze_final_all_even_pools() -> None:
    """Create the authorised pre-execution amendment for omitted final pools."""
    p.require_local_cpu()
    if any((OUT / name).exists() for name in ("oof", "final", "scores")):
        raise SystemExit("E7 AMENDMENT PRECONDITION FAILED: execution output exists")
    predecessor = p.sha256_file(OUT / "e7_protocol.json")
    frame = _historical_frame()
    candidate = frame.index.to_numpy(dtype="int64")
    if frame["building_id"].mod(2).any():
        raise RuntimeError("final candidate contains odd buildings")
    candidate_record = pool_record(frame, candidate)
    records: list[dict[str, Any]] = []
    pool_dir = OUT / "final_pools"
    code_identity = p.sha256_file(Path(__file__))
    feature_digest = "eab08c9a1b39ee7a74070c2ded464e3efbfb13d89dc148ef6b3bc3f443b21de3"
    supports: dict[str, np.ndarray] = {}
    for cell in p.SUPPORT_CELLS:
        slot = f"s{cell}"
        first, second = cell
        excluded = (
            (frame["meter"] == 3) & (frame["anomaly"] == 1) & (first == "0")
        ) | ((frame["meter"] == 3) & (frame["anomaly"] == 0) & (second == "0"))
        eligible = frame.loc[~excluded]
        indices = support_pool(frame, cell)
        supports[slot] = indices
        stored = pool_dir / f"final_{slot}_raw_index.npy"
        p.atomic_npy(stored, indices)
        summary = pool_record(frame, indices)
        records.append(
            {
                "schema_version": 1,
                "phase": "final",
                "family": "support",
                "slot": slot,
                "paired_slot": f"n{cell}",
                "building_split_rule": "building_id % 2 == 0",
                "candidate_building_count": int(frame["building_id"].nunique()),
                "candidate_row_count": int(len(frame)),
                "candidate_pool_digest": p.array_digest(candidate),
                "eligible_row_count": int(len(eligible)),
                "eligible_pool_digest": p.array_digest(
                    eligible.index.to_numpy(dtype="int64")
                ),
                "sampled_row_count": summary["rows"],
                "positive_count": summary["positive_rows"],
                "negative_count": summary["negative_rows"],
                "per_meter_row_counts": summary["meter_rows"],
                "per_meter_positive_counts": summary["meter_anomalies"],
                "raw_index_storage_location": str(stored.relative_to(OUT)),
                "raw_index_dtype": "int64",
                "raw_index_length": summary["rows"],
                "raw_index_sha256": summary["raw_index_digest"],
                "row_order_sha256": summary["row_order_digest"],
                "sampling_algorithm_identity": "lead.sample.downsample_indices",
                "sampling_seeds": [10, 20],
                "source_code_sha256": code_identity,
                "feature_contract_digest": feature_digest,
                "protocol_predecessor_commit": p.PHASE0_COMMIT,
                "generated_before_any_fit": True,
                "odd_rows_used": 0,
                "odd_labels_read": 0,
                "historical_s11_identity_evidence": "canonical all-even full-support downsample pipeline; historical source supplies row-count/provenance but no raw-index digest",
            }
        )
    for cell in p.SUPPORT_CELLS:
        slot, paired = f"n{cell}", f"s{cell}"
        indices = neutral_pool(frame, supports[paired], slot)
        stored = pool_dir / f"final_{slot}_raw_index.npy"
        p.atomic_npy(stored, indices)
        summary = pool_record(frame, indices)
        paired_summary = pool_record(frame, supports[paired])
        if (summary["rows"], summary["positive_rows"], summary["negative_rows"]) != (
            paired_summary["rows"],
            paired_summary["positive_rows"],
            paired_summary["negative_rows"],
        ):
            raise RuntimeError("E7 FINAL NEUTRAL MATCH FAILURE")
        records.append(
            {
                "schema_version": 1,
                "phase": "final",
                "family": "neutral",
                "slot": slot,
                "paired_slot": paired,
                "building_split_rule": "building_id % 2 == 0",
                "candidate_building_count": int(frame["building_id"].nunique()),
                "candidate_row_count": int(len(frame)),
                "candidate_pool_digest": p.array_digest(candidate),
                "eligible_row_count": int(len(frame)),
                "eligible_pool_digest": p.array_digest(candidate),
                "sampled_row_count": summary["rows"],
                "positive_count": summary["positive_rows"],
                "negative_count": summary["negative_rows"],
                "per_meter_row_counts": summary["meter_rows"],
                "per_meter_positive_counts": summary["meter_anomalies"],
                "raw_index_storage_location": str(stored.relative_to(OUT)),
                "raw_index_dtype": "int64",
                "raw_index_length": summary["rows"],
                "raw_index_sha256": summary["raw_index_digest"],
                "row_order_sha256": summary["row_order_digest"],
                "sampling_algorithm_identity": "lead.sample.downsample_indices+neutral_pool",
                "sampling_seeds": list(NEUTRAL_SEEDS[slot]),
                "source_code_sha256": code_identity,
                "feature_contract_digest": feature_digest,
                "protocol_predecessor_commit": p.PHASE0_COMMIT,
                "generated_before_any_fit": True,
                "odd_rows_used": 0,
                "odd_labels_read": 0,
                "paired_row_overlap": int(
                    np.intersect1d(indices, supports[paired]).size
                ),
            }
        )
    for record in records:
        record["protocol_predecessor_commit"] = (
            "c4821a0899e334a5b1354598742d47fbf992efd5"
        )
    if len(records) != 8:
        raise RuntimeError("final pool census drift")
    final_manifest = {
        "schema_version": 1,
        "phase": "final",
        "candidate_census": {
            "even_buildings": int(frame["building_id"].nunique()),
            "candidate_rows": int(len(frame)),
            "candidate_raw_index_digest": p.array_digest(candidate),
            "candidate_row_order_digest": p.array_digest(candidate),
            "per_meter_rows": candidate_record["meter_rows"],
            "per_meter_anomalies": candidate_record["meter_anomalies"],
        },
        "records": records,
        "generated_before_any_formal_fit": True,
        "odd_rows_used": 0,
        "odd_labels_read": 0,
    }
    p.atomic_json(OUT / "e7_final_training_pool_manifest.json", final_manifest)
    final_sha = p.sha256_file(OUT / "e7_final_training_pool_manifest.json")
    protocol = json.loads((OUT / "e7_protocol.json").read_text(encoding="utf-8"))
    protocol["pre_execution_amendment_001"] = {
        "predecessor_protocol_commit": "c4821a0899e334a5b1354598742d47fbf992efd5",
        "predecessor_protocol_digest": predecessor,
        "final_training_pool_manifest_sha256": final_sha,
        "reason": "The original protocol freeze specified 32 final all-even component fits but omitted the final support and neutral row-identity manifests required to execute those fits without post-freeze resampling.",
    }
    p.atomic_json(OUT / "e7_protocol.json", protocol)
    authoritative = p.sha256_file(OUT / "e7_protocol.json")
    amendment = {
        "amendment_id": "001",
        "predecessor_protocol_commit": "c4821a0899e334a5b1354598742d47fbf992efd5",
        "predecessor_protocol_digest": predecessor,
        "reason": protocol["pre_execution_amendment_001"]["reason"],
        "authorization": "explicit_human_authorization",
        "timing": "before_any_formal_fit",
        "formal_fits_before_amendment": 0,
        "odd_predictions_before_amendment": 0,
        "odd_label_evaluations_before_amendment": 0,
        "fields_added": "final all-even support/neutral pool identities",
        "scientific_estimand_changed": False,
        "model_architecture_changed": False,
        "feature_contract_changed": False,
        "OOF_design_changed": False,
        "decision_rules_changed": False,
        "final_pool_manifest_digest": final_sha,
        "new_authoritative_protocol_digest": authoritative,
    }
    p.atomic_json(OUT / "e7_protocol_amendment_001.json", amendment)
    inputs = json.loads((OUT / "e7_input_manifest.json").read_text(encoding="utf-8"))
    inputs["final_training_pool_manifest_sha256"] = final_sha
    inputs["authoritative_protocol_digest"] = authoritative
    p.atomic_json(OUT / "e7_input_manifest.json", inputs)
    p.atomic_json(
        OUT / "e7_execution_coverage.json",
        {
            "formal_component_fits": 0,
            "odd_predictions": 0,
            "odd_label_evaluations": 0,
            "amendment_001": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--worker", type=Path)
    group.add_argument("--prepare-manifests", action="store_true")
    group.add_argument("--bounded-validation", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--execute-oof", action="store_true")
    group.add_argument("--execute-oof-fold", type=int, choices=range(5))
    group.add_argument("--finalise-oof", action="store_true")
    group.add_argument("--freeze-final-pools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args.worker)
    elif args.prepare_manifests:
        prepare_manifests()
    elif args.bounded_validation:
        bounded_validation()
    elif args.execute_oof:
        execute_oof()
    elif args.execute_oof_fold is not None:
        execute_oof_fold(args.execute_oof_fold)
    elif args.finalise_oof:
        finalise_oof()
    elif args.freeze_final_pools:
        freeze_final_all_even_pools()
    else:
        freeze()


if __name__ == "__main__":
    main()
