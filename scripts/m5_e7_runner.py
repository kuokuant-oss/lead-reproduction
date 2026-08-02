"""Checkpointed local-CPU executor for M5 E7 Tree units.

Normal use is deliberately split into preparation, bounded validation, formal
OOF/final fitting, label-firewalled scoring, and evaluation commands.  The
formal commands reject a dirty repository and a missing protocol freeze.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    scaled_inputs = "scaled_x_fit" in spec
    x_fit = np.load(
        spec["scaled_x_fit"] if scaled_inputs else spec["x_fit"], mmap_mode="r"
    )
    y_fit = np.load(spec["y_fit"], mmap_mode="r")
    x_query = np.load(
        spec["scaled_x_query"] if scaled_inputs else spec["x_query"], mmap_mode="r"
    )
    if x_fit.shape[0] != y_fit.shape[0] or x_query.shape[0] != spec["query_rows"]:
        raise SystemExit("component input shape mismatch")
    scaler_ref = Path(spec["scaler_source"]["path"])
    if (
        not scaler_ref.exists()
        or p.sha256_file(scaler_ref) != spec["scaler_source"]["sha256"]
    ):
        raise SystemExit("frozen shared scaler is missing or digest-drifted")
    scaler = joblib.load(scaler_ref)
    # New resource-override specs point at shared, scaled, read-only memmaps.
    # The fallback keeps an already-running pre-override worker compatible and
    # is deliberately removed from every newly prepared slot.
    fit = (
        x_fit
        if scaled_inputs
        else scaler.transform(x_fit).astype("float32", copy=False)
    )
    query = (
        x_query
        if scaled_inputs
        else scaler.transform(x_query).astype("float32", copy=False)
    )
    name = spec["component"]
    if name == "hist_gradient_boosting" and "hgb_x_fit" in spec:
        fit = np.load(spec["hgb_x_fit"], mmap_mode="r")
        query = np.load(spec["hgb_x_query"], mmap_mode="r")
    model = _component_model(name)
    model.fit(
        np.nan_to_num(fit, nan=0.0) if name == "hist_gradient_boosting" else fit, y_fit
    )
    scores = (
        np.empty(0, dtype="float32")
        if len(query) == 0
        else _predict(name, model, query)
    )
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


def _available_memory_bytes() -> int:
    """Read physical available memory without importing an evaluation package."""
    if os.name != "nt":
        return 0

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("avail_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.avail_phys)


def scheduler_concurrency(specs: list[dict[str, Any]]) -> int:
    """Return a RAM-gated concurrency level for as-yet-unclaimed units only."""
    policy = json.loads(
        (OUT / "e7_execution_resource_override_001.json").read_text(encoding="utf-8")
    )["new_dynamic_concurrency_policy"]
    available = _available_memory_bytes()
    reserve = 16 * 1024**3
    scale_up_floor = 24 * 1024**3
    # Measured first-unit RSS plus the authorised 25% safety factor.  These
    # values are resource controls only; they never depend on model scores.
    expected_peak = {
        "lightgbm": int(3.2 * 1024**3 * 1.25),
        "xgboost": int(6.0 * 1024**3 * 1.25),
        "catboost": int(3.1 * 1024**3 * 1.25),
        "hist_gradient_boosting": int(7.9 * 1024**3 * 1.25),
    }
    if available < scale_up_floor:
        return 1
    capacity = max(
        1, (available - reserve) // max(expected_peak[s["component"]] for s in specs)
    )
    requested = min(policy["initial_concurrency"], policy["maximum_concurrent_units"])
    return max(1, min(len(specs), requested, int(capacity)))


def run_component_batch(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run independent component subprocesses at the RAM-gated safe limit.

    Every specification refers to the same immutable shared scaler and input
    matrices for its expert slot.  Threading here only dispatches the already
    isolated subprocess workers; model fitting remains CPU-process based and
    each component preserves its one-thread contract.
    """
    if not specs:
        return []
    workers = scheduler_concurrency(specs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_component, specs))


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


def _unlabelled_m3_frame() -> pd.DataFrame:
    """Canonical M3 feature frame that never opens bad_meter_readings.csv."""
    import lead.data as data

    data.M3 = PHASE0_ARTIFACT_ROOT / "data" / "raw" / "m3"
    train = pd.read_csv(
        data.M3 / "train.csv",
        dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"},
    )
    train = data._add_time_features(train)
    meta = data._building_metadata()
    train = train.merge(
        meta[
            [
                "building_id",
                "site_id",
                "primary_use_enc",
                "log_square_feet",
                "year_built",
                "floor_count",
            ]
        ],
        on="building_id",
        how="left",
    )
    weather = data._weather_frame(include_budslab_features=False)
    weather_cols = [
        "site_id",
        "timestamp",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
    ]
    train = train.merge(weather[weather_cols], on=["site_id", "timestamp"], how="left")
    return train[["building_id", "site_id", "timestamp", *data.BASELINE_FEATURE_COLS]]


def build_full_label_free_store() -> None:
    """Build canonical all-meter odd feature chunks without any label artifact."""
    source = (
        PHASE0_ARTIFACT_ROOT
        / "data"
        / "processed"
        / "m5_tabpfn_137_full_test_n8_predictions.npz"
    )
    with np.load(source, allow_pickle=False) as identity:
        raw = identity["raw_index"].astype("int64")
        building = identity["building_id"].astype("int16")
        site = identity["site_id"].astype("int8")
    if (
        len(raw) != 10_137_155
        or len(np.unique(raw)) != len(raw)
        or np.any(building % 2 == 0)
    ):
        raise SystemExit("label-free canonical holdout identity gate failed")
    frame = _unlabelled_m3_frame()
    odd = frame.iloc[raw].copy()
    if not np.array_equal(
        odd["building_id"].to_numpy(dtype="int16"), building
    ) or not np.array_equal(odd["site_id"].to_numpy(dtype="int8"), site):
        raise SystemExit("label-free identity reconstruction drift")
    features = _f4(odd, odd.index.to_numpy(dtype="int64"))
    store = OUT / "full_holdout_store"
    chunks = []
    for start in range(0, len(raw), 100_000):
        stop = min(len(raw), start + 100_000)
        root = store / f"chunk_{start // 100_000:03d}"
        p.atomic_npy(root / "raw_index.npy", raw[start:stop])
        p.atomic_npy(
            root / "meter.npy", odd["meter"].to_numpy(dtype="int8")[start:stop]
        )
        p.atomic_npy(root / "building_id.npy", building[start:stop])
        p.atomic_npy(root / "site_id.npy", site[start:stop])
        p.atomic_npy(root / "features.npy", features[start:stop])
        chunks.append(
            {
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "raw_index_digest": p.array_digest(raw[start:stop]),
                "feature_sha256": p.sha256_file(root / "features.npy"),
                "dtype": "float32",
            }
        )
    p.atomic_json(
        OUT / "e7_full_holdout_feature_manifest.json",
        {
            "schema_version": 1,
            "rows": len(raw),
            "chunks": chunks,
            "chunk_rows": 100000,
            "raw_index_digest": p.array_digest(raw),
            "feature_column_digest": "eab08c9a1b39ee7a74070c2ded464e3efbfb13d89dc148ef6b3bc3f443b21de3",
            "label_fields": [],
            "source_arrays_read": ["raw_index", "building_id", "site_id"],
        },
    )


def extract_label_free_historical_scores() -> None:
    """Extract only score fields from A001; deliberately never touch anomaly."""
    source = (
        PHASE0_ARTIFACT_ROOT / "data" / "processed" / "m3_figure_predictions_50_50.npz"
    )
    fields = ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting", "ensemble")
    output = OUT / "historical_label_free"
    with np.load(source, allow_pickle=False) as artifact:
        for field in fields:
            p.atomic_npy(output / f"{field}.npy", artifact[field].astype("float32"))
    p.atomic_json(
        OUT / "e7_historical_score_extract_manifest.json",
        {
            "rows": 10137155,
            "fields": list(fields),
            "source_arrays_read": list(fields),
            "forbidden_arrays_not_read": ["anomaly", "validation_raw_index"],
            "canonical_identity": "A002 positional order",
        },
    )


def _load_final_component(family: str, slot: str, component: str) -> tuple[Any, Any]:
    unit = OUT / "units" / p.unit_id("final", "all_even", family, slot, component)
    marker = json.loads((unit / "complete.json").read_text(encoding="utf-8"))
    return joblib.load(unit / marker["files"]["model"]["path"]), joblib.load(
        unit / marker["files"]["scaler"]["path"]
    )


def _load_meta(family: str) -> tuple[Any, Any]:
    root = OUT / "final_meta" / family
    return joblib.load(root / "model.joblib"), joblib.load(root / "scaler.joblib")


def score_s11_full_holdout() -> None:
    """Label-free s11 component and ensemble scoring over all canonical chunks."""
    models = {
        name: _load_final_component("support", "s11", name) for name in p.MODEL_ORDER
    }
    feature_manifest = json.loads(
        (OUT / "e7_full_holdout_feature_manifest.json").read_text(encoding="utf-8")
    )
    out = OUT / "scores" / "s11_full"
    for number, chunk in enumerate(feature_manifest["chunks"]):
        features = np.load(
            OUT / "full_holdout_store" / f"chunk_{number:03d}" / "features.npy",
            allow_pickle=False,
        )
        scores = {}
        for component, (model, scaler) in models.items():
            scores[component] = _predict(
                component,
                model,
                scaler.transform(features).astype("float32", copy=False),
            )
            p.atomic_npy(
                out / f"chunk_{number:03d}" / f"{component}.npy", scores[component]
            )
        p.atomic_npy(
            out / f"chunk_{number:03d}" / "ensemble.npy",
            np.mean(list(scores.values()), axis=0, dtype="float64").astype("float32"),
        )


def score_steam_specialists() -> None:
    """Score frozen support/neutral experts only where the label-free router says steam."""
    manifest = json.loads(
        (OUT / "e7_full_holdout_feature_manifest.json").read_text(encoding="utf-8")
    )
    experts = [("support", f"s{cell}") for cell in p.SUPPORT_CELLS] + [
        ("neutral", f"n{cell}") for cell in p.SUPPORT_CELLS
    ]
    loaded = {
        slot: {
            component: _load_final_component(family, slot, component)
            for component in p.MODEL_ORDER
        }
        for family, slot in experts
    }
    out = OUT / "scores" / "steam_specialists"
    for number, _ in enumerate(manifest["chunks"]):
        root = OUT / "full_holdout_store" / f"chunk_{number:03d}"
        meter = np.load(root / "meter.npy", allow_pickle=False)
        rows = meter == 2
        if not rows.any():
            continue
        features = np.load(root / "features.npy", allow_pickle=False)[rows]
        p.atomic_npy(
            out / f"chunk_{number:03d}" / "raw_index.npy",
            np.load(root / "raw_index.npy", allow_pickle=False)[rows],
        )
        for slot, components in loaded.items():
            values = []
            for component, (model, scaler) in components.items():
                score = _predict(
                    component,
                    model,
                    scaler.transform(features).astype("float32", copy=False),
                )
                p.atomic_npy(
                    out / f"chunk_{number:03d}" / f"{slot}_{component}.npy", score
                )
                values.append(score)
            p.atomic_npy(
                out / f"chunk_{number:03d}" / f"{slot}.npy",
                np.mean(values, axis=0, dtype="float64").astype("float32"),
            )


def assemble_hybrid_chunks() -> None:
    """Route label-free final scores exactly by meter; no outcome-dependent branch exists."""
    support_model, support_scaler = _load_meta("support")
    neutral_model, neutral_scaler = _load_meta("neutral")
    manifest = json.loads(
        (OUT / "e7_full_holdout_feature_manifest.json").read_text(encoding="utf-8")
    )
    output = OUT / "scores" / "hybrid"
    for number, _ in enumerate(manifest["chunks"]):
        base = OUT / "full_holdout_store" / f"chunk_{number:03d}"
        meter = np.load(base / "meter.npy", allow_pickle=False)
        s11 = np.load(
            OUT / "scores" / "s11_full" / f"chunk_{number:03d}" / "ensemble.npy",
            allow_pickle=False,
        )
        historical = np.load(
            OUT / "historical_label_free" / "ensemble.npy", mmap_mode="r"
        )[manifest["chunks"][number]["start"] : manifest["chunks"][number]["stop"]]
        deploy, locked, deploy_n, locked_n = (
            s11.copy(),
            historical.copy(),
            s11.copy(),
            historical.copy(),
        )
        rows = meter == 2
        if rows.any():
            specialist = OUT / "scores" / "steam_specialists" / f"chunk_{number:03d}"
            support = {
                f"s{cell}": np.load(specialist / f"s{cell}.npy", allow_pickle=False)
                for cell in p.SUPPORT_CELLS
            }
            neutral = {
                f"n{cell}": np.load(specialist / f"n{cell}.npy", allow_pickle=False)
                for cell in p.SUPPORT_CELLS
            }
            support_score = support_model.predict_proba(
                support_scaler.transform(p.factor_features(support, "s"))
            )[:, 1].astype("float32")
            neutral_score = neutral_model.predict_proba(
                neutral_scaler.transform(p.factor_features(neutral, "n"))
            )[:, 1].astype("float32")
            deploy[rows] = support_score
            locked[rows] = support_score
            deploy_n[rows] = neutral_score
            locked_n[rows] = neutral_score
        for name, value in {
            "deployable_refit_hybrid": deploy,
            "locked_reference_hybrid": locked,
            "deployable_refit_neutral_hybrid": deploy_n,
            "locked_reference_neutral_hybrid": locked_n,
        }.items():
            p.atomic_npy(
                output / f"chunk_{number:03d}" / f"{name}.npy", value.astype("float32")
            )


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
    scaled_x_path, scaled_q_path = (
        slot_root / "x_scaled.npy",
        slot_root / "query_scaled.npy",
    )
    hgb_x_path, hgb_q_path = (
        slot_root / "x_scaled_nan_zero.npy",
        slot_root / "query_scaled_nan_zero.npy",
    )
    scaler_path = slot_root / "shared_scaler.joblib"
    input_manifest_path = slot_root / "shared_input_manifest.json"
    input_identity = {
        "training_raw_index_digest": p.array_digest(train_indices),
        "validation_raw_index_digest": p.array_digest(query_indices),
        "feature_column_digest": "eab08c9a1b39ee7a74070c2ded464e3efbfb13d89dc148ef6b3bc3f443b21de3",
        "protocol_sha256": p.sha256_file(OUT / "e7_protocol.json"),
        "x_shape": list(x_fit.shape),
        "query_shape": list(x_query.shape),
        "dtype": "float32",
    }
    if input_manifest_path.exists():
        shared = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if shared["identity"] != input_identity:
            raise SystemExit(f"shared slot input drift: {slot_root.name}")
    else:
        p.atomic_npy(x_path, x_fit)
        p.atomic_npy(y_path, y_fit.astype("int8", copy=False))
        p.atomic_npy(q_path, x_query)
        scaler_sha = _save_joblib(scaler_path, scaler)
        x_scaled = scaler.transform(x_fit).astype("float32", copy=False)
        q_scaled = scaler.transform(x_query).astype("float32", copy=False)
        p.atomic_npy(scaled_x_path, x_scaled)
        p.atomic_npy(scaled_q_path, q_scaled)
        p.atomic_npy(hgb_x_path, np.nan_to_num(x_scaled, nan=0.0))
        p.atomic_npy(hgb_q_path, np.nan_to_num(q_scaled, nan=0.0))
        p.atomic_json(
            input_manifest_path,
            {
                "identity": input_identity,
                "files": {
                    name: {"path": str(path), "sha256": p.sha256_file(path)}
                    for name, path in {
                        "x": x_path,
                        "y": y_path,
                        "query": q_path,
                        "x_scaled": scaled_x_path,
                        "query_scaled": scaled_q_path,
                        "hgb_x_scaled": hgb_x_path,
                        "hgb_query_scaled": hgb_q_path,
                        "scaler": scaler_path,
                    }.items()
                },
            },
        )
        del x_scaled, q_scaled
    shared = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    scaler_sha = shared["files"]["scaler"]["sha256"]
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
        "scaled_x_fit": str(scaled_x_path),
        "scaled_x_query": str(scaled_q_path),
        "hgb_x_fit": str(hgb_x_path),
        "hgb_x_query": str(hgb_q_path),
        "shared_input_manifest_sha256": p.sha256_file(input_manifest_path),
        "query_rows": int(len(x_query)),
        "training_raw_index_digest": input_identity["training_raw_index_digest"],
        "training_row_order_digest": input_identity["training_raw_index_digest"],
        "validation_raw_index_digest": input_identity["validation_raw_index_digest"],
        "feature_column_digest": "eab08c9a1b39ee7a74070c2ded464e3efbfb13d89dc148ef6b3bc3f443b21de3",
        "x_dtype": "float32",
        "environment_sha256": p.sha256_file(OUT / "e7_environment_manifest.json"),
        "protocol_sha256": input_identity["protocol_sha256"],
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
            specs = [
                _unit_spec(
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
                for component in p.MODEL_ORDER
            ]
            results = run_component_batch(specs)
            component_scores = []
            for spec, result in zip(specs, results, strict=True):
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


def execute_final() -> None:
    """Fit exactly the eight amendment-frozen final pools, with no resampling."""
    p.require_local_cpu()
    manifest_path = OUT / "e7_final_training_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["records"]) != 8:
        raise SystemExit("final manifest must have exactly eight records")
    frame = _historical_frame()
    if frame["building_id"].mod(2).any():
        raise SystemExit("final pool contains odd building")
    empty_query = np.empty((0, 137), dtype="float32")
    inputs = OUT / "inputs"
    for record in manifest["records"]:
        slot, family = record["slot"], record["family"]
        indices = np.load(
            OUT / record["raw_index_storage_location"], allow_pickle=False
        )
        observed = pool_record(frame, indices)
        for expected, actual in (
            (record["raw_index_sha256"], observed["raw_index_digest"]),
            (record["row_order_sha256"], observed["row_order_digest"]),
        ):
            if expected != actual:
                raise SystemExit(f"frozen final pool drift: {slot}")
        if (
            record["sampled_row_count"],
            record["positive_count"],
            record["negative_count"],
        ) != (observed["rows"], observed["positive_rows"], observed["negative_rows"]):
            raise SystemExit(f"frozen final pool count drift: {slot}")
        x_fit = _f4(frame, indices)
        y_fit = frame.loc[indices, "anomaly"].to_numpy(dtype="int8")
        scaler = StandardScaler().fit(x_fit)
        specs = [
            _unit_spec(
                phase="final",
                fold_name="all_even",
                family=family,
                slot=slot,
                component=component,
                inputs=inputs,
                x_fit=x_fit,
                y_fit=y_fit,
                x_query=empty_query,
                train_indices=indices,
                query_indices=np.empty(0, dtype="int64"),
                scaler=scaler,
            )
            for component in p.MODEL_ORDER
        ]
        run_component_batch(specs)
    p.atomic_json(
        OUT / "final" / "complete.json",
        {"expected_component_fits": 32, "complete": True, "odd_predictions": 0},
    )


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


def fit_final_meta_models() -> None:
    """Fit final meta models from OOF blocks only; final expert scores are refused."""
    from sklearn.linear_model import LogisticRegression

    summary = json.loads((OUT / "e7_oof_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete") or summary.get("component_fits") != 160:
        raise SystemExit("final meta requires complete 160-component OOF only")
    meta = json.loads((OUT / "e7_meta_model_manifest.json").read_text(encoding="utf-8"))
    for family, prefix, value in (
        ("support", "s", meta["selected_support_c"]),
        ("neutral", "n", meta["selected_neutral_c"]),
    ):
        features, labels = [], []
        for fold in range(5):
            root = OUT / "oof" / f"fold{fold}"
            scores = {
                f"{prefix}{cell}": np.load(
                    root / f"{prefix}{cell}.npy", allow_pickle=False
                )
                for cell in p.SUPPORT_CELLS
            }
            features.append(p.factor_features(scores, prefix))
            labels.append(np.load(root / "labels.npy", allow_pickle=False))
        x, y = np.concatenate(features), np.concatenate(labels)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=value, solver="lbfgs", max_iter=5000, class_weight=None, random_state=42
        ).fit(scaler.transform(x), y)
        target = OUT / "final_meta" / family
        _save_joblib(target / "scaler.joblib", scaler)
        _save_joblib(target / "model.joblib", model)
        p.atomic_json(
            target / "manifest.json",
            {
                "family": family,
                "selected_c": value,
                "source": "complete_even_building_oof_predictions_only",
                "oof_summary_sha256": p.sha256_file(OUT / "e7_oof_summary.json"),
                "feature_names": p.protocol()["meta_model"]["features"],
            },
        )


def amend_full_holdout_scoring_scope() -> None:
    """Apply the explicitly authorised pre-fit hybrid scoring-scope amendment."""
    if any((OUT / name).exists() for name in ("oof", "final", "scores")):
        raise SystemExit("scoring-scope amendment requires zero prior execution")
    predecessor = p.sha256_file(OUT / "e7_protocol.json")
    protocol = json.loads((OUT / "e7_protocol.json").read_text(encoding="utf-8"))
    protocol["pre_execution_amendment_002"] = {
        "predecessor_amendment_commit": "ec10c50eb5b0bb78e322a5bcdf4a1d47e362df87",
        "predecessor_authoritative_protocol_digest": predecessor,
        "reason": "The frozen E7 protocol correctly specified steam as the primary scientific endpoint, but the score-generation scope was too narrow for the intended hybrid deployment strategy. The hybrid must emit a complete odd-holdout prediction vector: support-aware scores for steam and ordinary full-support Tree scores for all other meters.",
        "training_pool_changed": False,
        "OOF_design_changed": False,
        "feature_contract_changed": False,
        "Tree_architecture_changed": False,
        "meta_model_changed": False,
        "steam_primary_endpoint_changed": False,
        "score_generation_scope_changed": True,
        "previous_scope": "odd steam only",
        "new_scope": "complete odd holdout hybrid",
        "routing": {
            "steam": "support_stack",
            "nonsteam": "final_s11_ensemble",
            "locked_nonsteam": "historical_full_tree_ensemble",
        },
    }
    p.atomic_json(OUT / "e7_protocol.json", protocol)
    authoritative = p.sha256_file(OUT / "e7_protocol.json")
    p.atomic_json(
        OUT / "e7_protocol_amendment_002.json",
        {
            "amendment_id": "E7_AMENDMENT_002",
            "predecessor_amendment_commit": "ec10c50eb5b0bb78e322a5bcdf4a1d47e362df87",
            "predecessor_authoritative_protocol_digest": predecessor,
            "explicit_human_authorization": True,
            "timing": "before_any_formal_fit",
            "formal_fits_before_amendment": 0,
            "odd_predictions_before_amendment": 0,
            "odd_label_reads_before_amendment": 0,
            "training_pool_changed": False,
            "OOF_design_changed": False,
            "feature_contract_changed": False,
            "Tree_architecture_changed": False,
            "meta_model_changed": False,
            "steam_primary_endpoint_changed": False,
            "score_generation_scope_changed": True,
            "previous_scope": "odd steam only",
            "new_scope": "complete odd holdout hybrid",
            "final_hybrid_routing_rule": protocol["pre_execution_amendment_002"][
                "routing"
            ],
            "new_authoritative_protocol_digest": authoritative,
        },
    )
    inputs = json.loads((OUT / "e7_input_manifest.json").read_text(encoding="utf-8"))
    inputs["authoritative_protocol_digest"] = authoritative
    p.atomic_json(OUT / "e7_input_manifest.json", inputs)


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
    group.add_argument("--execute-final", action="store_true")
    group.add_argument("--fit-final-meta", action="store_true")
    group.add_argument("--amend-full-holdout-scoring", action="store_true")
    group.add_argument("--build-full-label-free-store", action="store_true")
    group.add_argument("--extract-label-free-historical-scores", action="store_true")
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
    elif args.execute_final:
        execute_final()
    elif args.fit_final_meta:
        fit_final_meta_models()
    elif args.amend_full_holdout_scoring:
        amend_full_holdout_scoring_scope()
    elif args.build_full_label_free_store:
        build_full_label_free_store()
    elif args.extract_label_free_historical_scores:
        extract_label_free_historical_scores()
    else:
        freeze()


if __name__ == "__main__":
    main()
