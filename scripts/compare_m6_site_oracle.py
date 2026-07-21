"""Create a paired cross-site versus A5 in-site-oracle comparison artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from lead import ROOT, write_json_with_provenance
from m6_site_transfer_protocol import array_fingerprint
from run_m3_full_site_transfer import safe_evaluation_summary


METADATA_KEYS = {
    "validation_raw_index",
    "timestamp_ns",
    "site_id",
    "building_id",
    "meter",
    "anomaly",
    "m3_1_lightgbm",
}


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_path(path: Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def prediction_models(data: dict[str, np.ndarray]) -> list[str]:
    return sorted(key for key in data if key not in METADATA_KEYS)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def paired_oracle_comparison(
    cross: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare both model families on the exact oracle-test building subset."""
    required = {"site_id", "building_id", "meter", "anomaly"}
    for name, data in (("cross", cross), ("oracle", oracle)):
        missing = required - set(data)
        if missing:
            raise ValueError(f"{name} predictions missing keys: {sorted(missing)}")
    oracle_sites = np.unique(oracle["site_id"])
    if len(oracle_sites) != 1:
        raise ValueError("oracle predictions must contain exactly one site")
    site_id = int(oracle_sites[0])
    oracle_buildings = np.unique(oracle["building_id"])
    cross_mask = (cross["site_id"] == site_id) & np.isin(
        cross["building_id"],
        oracle_buildings,
    )
    if not cross_mask.any():
        raise ValueError(f"cross predictions do not cover oracle site {site_id}")

    identity_keys = ["site_id", "building_id", "meter", "anomaly"]
    if "timestamp_ns" in cross and "timestamp_ns" in oracle:
        identity_keys.insert(3, "timestamp_ns")
        identity_basis = "ordered_site_building_meter_timestamp_label"
    else:
        identity_basis = "ordered_site_building_meter_label_legacy_no_timestamp"
    cross_identity = np.column_stack([cross[key][cross_mask] for key in identity_keys])
    oracle_identity = np.column_stack([oracle[key] for key in identity_keys])
    if cross_identity.shape != oracle_identity.shape or not np.array_equal(
        cross_identity,
        oracle_identity,
    ):
        raise AssertionError(
            "cross-site and oracle predictions do not describe the same ordered rows"
        )

    cross_models = set(prediction_models(cross))
    oracle_models = set(prediction_models(oracle))
    common_models = sorted(cross_models & oracle_models)
    if not common_models:
        raise ValueError("cross and oracle artifacts share no prediction models")
    y_true = oracle["anomaly"].astype("int8", copy=False)
    models = {}
    for model in common_models:
        cross_metrics = safe_evaluation_summary(y_true, cross[model][cross_mask])
        oracle_metrics = safe_evaluation_summary(y_true, oracle[model])
        cross_pr = cross_metrics.get("pr_auc")
        oracle_pr = oracle_metrics.get("pr_auc")
        models[model] = {
            "cross_site": cross_metrics,
            "in_site_oracle": oracle_metrics,
            "oracle_minus_cross_pr_auc": (
                None
                if cross_pr is None or oracle_pr is None
                else float(oracle_pr - cross_pr)
            ),
        }
    return {
        "schema_version": 1,
        "experiment": "m6_paired_in_site_oracle",
        "site_id": site_id,
        "oracle_test_buildings": [int(value) for value in oracle_buildings],
        "n_rows": int(len(y_true)),
        "n_anomalies": int(y_true.sum()),
        "identity_basis": identity_basis,
        "paired_eval_key_sha256": array_fingerprint(oracle_identity),
        "common_models": common_models,
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-predictions", type=Path, required=True)
    parser.add_argument("--oracle-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cross_path = resolve_path(args.cross_predictions)
    oracle_path = resolve_path(args.oracle_predictions)
    out_path = resolve_path(args.out)
    payload = paired_oracle_comparison(
        load_npz(cross_path),
        load_npz(oracle_path),
    )
    payload["artifacts"] = {
        "cross_predictions": artifact_path(cross_path),
        "oracle_predictions": artifact_path(oracle_path),
    }
    write_json_with_provenance(
        out_path,
        payload,
        root=ROOT,
        provenance={
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts/compare_m6_site_oracle.py "
                f"--cross-predictions {artifact_path(cross_path)} "
                f"--oracle-predictions {artifact_path(oracle_path)} "
                f"--out {artifact_path(out_path)}"
            ),
            "note": "Paired diagnostic only; no model fitting ran.",
        },
    )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
