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
    scaler = StandardScaler()
    fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
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
        spec = {
            "unit_id": unit,
            "unit_dir": str(root / "units" / unit),
            "component": name,
            "x_fit": str(inputs / "x.npy"),
            "y_fit": str(inputs / "y.npy"),
            "x_query": str(inputs / "q.npy"),
            "query_rows": len(q),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--worker", type=Path)
    group.add_argument("--prepare-manifests", action="store_true")
    group.add_argument("--bounded-validation", action="store_true")
    group.add_argument("--freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args.worker)
    elif args.prepare_manifests:
        prepare_manifests()
    elif args.bounded_validation:
        bounded_validation()
    else:
        freeze()


if __name__ == "__main__":
    main()
