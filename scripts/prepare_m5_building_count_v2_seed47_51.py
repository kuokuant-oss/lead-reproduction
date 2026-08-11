"""Prepare and verify the portable M5 V2 building-seed 47--51 bundle.

This script never launches a model. It can regenerate the frozen ladder audit
from its bundled candidate profiles and atomically install the tracked compact
canonical holdout identity artifact when a fresh clone lacks that processed
prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lead import PROC, ROOT
try:
    from audit_m5_building_candidate_sensitivity import (
        _attach_primary_use,
        build_sensitivity_audit,
    )
    from m5_building_curve_protocol import atomic_write_json, int_array_sha256
except ModuleNotFoundError:  # Package import used by unittest discovery.
    from scripts.audit_m5_building_candidate_sensitivity import (
        _attach_primary_use,
        build_sensitivity_audit,
    )
    from scripts.m5_building_curve_protocol import (
        atomic_write_json,
        int_array_sha256,
    )


EXPECTED_SEEDS = (47, 48, 49, 50, 51)
EXPECTED_BUDGETS = (10, 20, 50, 100)
EXPECTED_CANDIDATE_SHA256 = (
    "4d86c3d328602c0e0d093783011c5d1d7479541fa46e81e51a1f4d3c1716d75c"
)
EXPECTED_HOLDOUT_ROW_SHA256 = (
    "6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893"
)
BUNDLE_ROOT = ROOT / "experiments" / "m5_building_count_v2_seed47_51"
AUDIT_ROOT = BUNDLE_ROOT / "audit"
DEFAULT_PROFILES = AUDIT_ROOT / "candidate_building_profiles.csv"
RAW_ROOT = ROOT / "data" / "raw" / "m3"
CANONICAL_HOLDOUT = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
BUNDLED_CANONICAL_HOLDOUT = BUNDLE_ROOT / "canonical_holdout_identity.npz"
REQUIRED_RAW_FILES = (
    "train.csv",
    "bad_meter_readings.csv",
    "building_metadata.csv",
    "weather_train.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_raw_files(raw_root: Path = RAW_ROOT) -> None:
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_root / name).is_file()]
    if missing:
        raise SystemExit(f"missing required M3 raw files: {missing}")


def validate_audit_bundle(audit_root: Path = AUDIT_ROOT) -> dict[str, object]:
    summary_path = audit_root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"seed47-51 audit summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "audit_passed_ready_for_model_evaluation",
        "sampling_profile": "site_stratified_random",
        "building_seeds": list(EXPECTED_SEEDS),
        "budgets": list(EXPECTED_BUDGETS),
        "row_seed": 42,
        "role_seed": None,
        "model_seed": 42,
        "candidate_building_sha256": EXPECTED_CANDIDATE_SHA256,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"seed47-51 audit summary mismatch: {mismatches}")
    if not summary["meter_feasibility_gate"]["all_passed"]:
        raise SystemExit("seed47-51 meter feasibility gate failed")
    if not summary["distinct_draw_gate"]["passed"]:
        raise SystemExit("seed47-51 distinct-draw gate failed")

    for seed in EXPECTED_SEEDS:
        manifest_path = audit_root / f"building_ladder_seed{seed}.json"
        ladder_path = audit_root / f"building_ladder_seed{seed}.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["building_seed"]) != seed:
            raise SystemExit(f"manifest seed mismatch: {manifest_path}")
        if manifest["budgets"] != list(EXPECTED_BUDGETS):
            raise SystemExit(f"manifest budgets mismatch: {manifest_path}")
        frozen = {
            "sampling_profile": "site_stratified_random",
            "row_seed": 42,
            "row_selection_seed": 42,
            "role_seed": None,
            "model_seed": 42,
            "row_policy": "average_building_cap",
            "average_rows_per_building_limit": 500,
            "max_context_rows": 50_000,
        }
        changed = {
            key: (manifest.get(key), value)
            for key, value in frozen.items()
            if manifest.get(key) != value
        }
        if changed:
            raise SystemExit(f"manifest frozen settings mismatch: {changed}")
        if _sha256_file(ladder_path) != manifest["ladder_csv_sha256"]:
            raise SystemExit(f"ladder CSV digest mismatch: {ladder_path}")
        for budget in EXPECTED_BUDGETS:
            cell = manifest["cells"][str(budget)]
            if len(cell["available_buildings"]) != budget:
                raise SystemExit(f"seed={seed} K={budget} building count mismatch")
            if len(set(cell["available_buildings"])) != budget:
                raise SystemExit(f"seed={seed} K={budget} repeats a building")
            if not cell["constraint_pass"]:
                raise SystemExit(f"seed={seed} K={budget} constraint gate failed")
            expected_rows = min(budget * 500, 50_000)
            if int(cell["allocated_rows"]) != expected_rows:
                raise SystemExit(f"seed={seed} K={budget} row allocation mismatch")
    return summary


def regenerate_audit(
    *,
    profiles_csv: Path = DEFAULT_PROFILES,
    audit_root: Path = AUDIT_ROOT,
    raw_root: Path = RAW_ROOT,
) -> dict[str, object]:
    if not profiles_csv.is_file():
        raise SystemExit(f"candidate profiles are missing: {profiles_csv}")
    profiles = pd.read_csv(profiles_csv)
    profiles = _attach_primary_use(profiles, raw_root / "building_metadata.csv")
    summary = build_sensitivity_audit(
        profiles,
        audit_root,
        building_seeds=EXPECTED_SEEDS,
        budgets=EXPECTED_BUDGETS,
        row_seed=42,
        model_seed=42,
    )
    summary["profile_source"] = "candidate_building_profiles.csv"
    summary["building_metadata"] = "data/raw/m3/building_metadata.csv"
    summary["bundle_root"] = "experiments/m5_building_count_v2_seed47_51"
    atomic_write_json(audit_root / "summary.json", summary)
    return validate_audit_bundle(audit_root)


def validate_canonical_holdout(path: Path = CANONICAL_HOLDOUT) -> None:
    if not path.is_file():
        raise SystemExit(
            f"canonical holdout is missing: {path}; run with --mode prepare-canonical"
        )
    with np.load(path) as payload:
        missing = {
            "validation_raw_index",
            "anomaly",
            "building_id",
            "site_id",
        } - set(payload.files)
        if missing:
            raise SystemExit(f"canonical holdout lacks arrays: {sorted(missing)}")
        raw_index = np.asarray(payload["validation_raw_index"], dtype="int64")
        anomaly = np.asarray(payload["anomaly"], dtype="int8")
        building = np.asarray(payload["building_id"], dtype="int16")
        site = np.asarray(payload["site_id"], dtype="int8")
    if not (len(raw_index) == len(anomaly) == len(building) == len(site)):
        raise SystemExit("canonical holdout arrays differ in length")
    if len(np.unique(raw_index)) != len(raw_index) or np.any(building % 2 == 0):
        raise SystemExit("canonical holdout row/building identity is invalid")
    digest = int_array_sha256(raw_index)
    if digest != EXPECTED_HOLDOUT_ROW_SHA256:
        raise SystemExit(f"canonical holdout digest mismatch: {digest}")


def prepare_canonical_holdout(
    *,
    path: Path = CANONICAL_HOLDOUT,
    bundled_path: Path = BUNDLED_CANONICAL_HOLDOUT,
) -> None:
    if path.is_file():
        validate_canonical_holdout(path)
        return
    validate_canonical_holdout(bundled_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with bundled_path.open("rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    validate_canonical_holdout(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("check", "regenerate-audit", "prepare-canonical", "all"),
        default="check",
    )
    parser.add_argument("--profiles-csv", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--canonical-holdout", type=Path, default=CANONICAL_HOLDOUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode in {"regenerate-audit", "all"}:
        regenerate_audit(
            profiles_csv=args.profiles_csv,
            audit_root=args.audit_root,
            raw_root=args.raw_root,
        )
    validate_audit_bundle(args.audit_root)
    check_raw_files(args.raw_root)
    validate_canonical_holdout(BUNDLED_CANONICAL_HOLDOUT)
    if args.mode in {"prepare-canonical", "all"}:
        prepare_canonical_holdout(path=args.canonical_holdout)
    else:
        validate_canonical_holdout(args.canonical_holdout)
    print(
        json.dumps(
            {
                "status": "ready",
                "building_seeds": list(EXPECTED_SEEDS),
                "budgets": list(EXPECTED_BUDGETS),
                "audit_root": str(args.audit_root),
                "canonical_holdout": str(args.canonical_holdout),
                "holdout_row_sha256": EXPECTED_HOLDOUT_ROW_SHA256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
