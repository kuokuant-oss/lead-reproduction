"""Plan or run the M5 building-count V2 matched-context seed sweep.

V2 pairs the constrained site-stratified ladders with frozen no-early-stopping
trees and TabPFN on byte-identical manifest-allocated context rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC, ROOT

EXPERIMENT_VERSION = "m5_building_count_v2"
DEFAULT_AUDIT_ROOT = (
    PROC / "m5_building_curve" / "sensitivity" / "building_candidate_pilot"
)
DEFAULT_OUT_ROOT = PROC / "m5_building_curve" / "v2"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("tree", "tabpfn"),
        default=["tree", "tabpfn"],
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--validation-context-rows", type=int, default=200)
    parser.add_argument("--validation-holdout-rows", type=int, default=200)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "reports" / "m5-building-count-experiment_V2.md",
    )
    return parser.parse_args(argv)


def build_units(
    audit_root: Path,
    out_root: Path,
    summary: dict[str, Any],
    *,
    families: list[str],
    mode: str,
    model_seed: int,
    validation_context_rows: int,
    validation_holdout_rows: int,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    sweep = "building_seed_sweep_" + "-".join(
        str(seed) for seed in summary["building_seeds"]
    )
    for building_seed in summary["building_seeds"]:
        manifest = audit_root / f"building_ladder_seed{building_seed}.json"
        for budget in summary["budgets"]:
            for family in families:
                family_tag = "tree_no_es" if family == "tree" else "tabpfn"
                output = (
                    out_root
                    / sweep
                    / "model_runs"
                    / f"building_seed{building_seed}"
                    / f"{family_tag}_k{budget}_f137"
                )
                script = (
                    "scripts/run_m5_building_count_v2_tree_cell.py"
                    if family == "tree"
                    else "scripts/run_m5_building_curve_tabpfn_cell.py"
                )
                command = [
                    sys.executable,
                    script,
                    "--building-manifest",
                    str(manifest),
                    "--building-budget",
                    str(budget),
                    "--features",
                    "137",
                    "--model-seed",
                    str(model_seed),
                    "--mode",
                    mode,
                    "--resume",
                    "--out-root",
                    str(output),
                ]
                if family == "tabpfn":
                    command.extend(
                        [
                            "--experiment-version",
                            EXPERIMENT_VERSION,
                            "--n-estimators",
                            "8",
                        ]
                    )
                if mode == "validation":
                    command.extend(
                        [
                            "--max-context-rows",
                            str(validation_context_rows),
                            "--max-holdout-rows",
                            str(validation_holdout_rows),
                        ]
                    )
                units.append(
                    {
                        "identity": {
                            "experiment_version": EXPERIMENT_VERSION,
                            "sampling_profile": summary["sampling_profile"],
                            "building_seed": int(building_seed),
                            "K": int(budget),
                            "features": 137,
                            "model": family,
                            "row_seed": int(summary["row_seed"]),
                            "model_seed": int(model_seed),
                        },
                        "manifest": str(manifest),
                        "output": str(output),
                        "command": command,
                    }
                )
    return units


def _complete(unit: dict[str, Any]) -> bool:
    output = Path(unit["output"])
    marker = output / "COMPLETE.json"
    cell = output / "cell.json"
    predictions = output / "predictions.npz"
    if not marker.is_file() or not cell.is_file() or not predictions.is_file():
        return False
    try:
        metadata = json.loads(cell.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    identity = unit["identity"]
    return (
        metadata.get("experiment_version") == EXPERIMENT_VERSION
        and int(metadata.get("building_seed", -1)) == identity["building_seed"]
        and int(metadata.get("building_budget", -1)) == identity["K"]
        and int(metadata.get("features", -1)) == identity["features"]
        and int(metadata.get("row_seed", -1)) == identity["row_seed"]
        and int(metadata.get("model_seed", -1)) == identity["model_seed"]
    )


def matched_context_gate(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for unit in units:
        identity = unit["identity"]
        grouped.setdefault((identity["building_seed"], identity["K"]), {})[
            identity["model"]
        ] = unit

    records: list[dict[str, Any]] = []
    for (building_seed, budget), family_units in sorted(grouped.items()):
        if set(family_units) != {"tree", "tabpfn"}:
            continue
        metadata = {
            family: json.loads(
                (Path(unit["output"]) / "cell.json").read_text(encoding="utf-8")
            )
            for family, unit in family_units.items()
        }
        tree = metadata["tree"]
        tabpfn = metadata["tabpfn"]
        fields = {
            "context_row_sha256": (
                tree["context_row_sha256"],
                tabpfn["context_row_sha256"],
            ),
            "context_rows": (tree["context_rows"], tabpfn["context_rows"]),
            "holdout_row_sha256": (
                tree["holdout_row_sha256"],
                tabpfn["holdout_row_sha256"],
            ),
            "row_policy": (tree["row_policy"], tabpfn["row_policy"]),
            "row_seed": (tree["row_seed"], tabpfn["row_seed"]),
            "model_seed": (tree["model_seed"], tabpfn["model_seed"]),
            "training_sampling": (
                tree["training_sampling"],
                tabpfn["training_sampling"],
            ),
            "class_ratio_policy": (
                tree["class_ratio_policy"],
                tabpfn["class_ratio_policy"],
            ),
        }
        mismatches = {
            field: values for field, values in fields.items() if values[0] != values[1]
        }
        if mismatches:
            raise AssertionError(
                f"V2 matched-context metadata mismatch for "
                f"building_seed={building_seed} K={budget}: {mismatches}"
            )

        prediction_paths = {
            family: Path(unit["output"]) / "predictions.npz"
            for family, unit in family_units.items()
        }
        with np.load(prediction_paths["tree"]) as tree_predictions:
            with np.load(prediction_paths["tabpfn"]) as tabpfn_predictions:
                for name in (
                    "validation_raw_index",
                    "anomaly",
                    "building_id",
                    "site_id",
                    "meter",
                ):
                    if not np.array_equal(
                        tree_predictions[name], tabpfn_predictions[name]
                    ):
                        raise AssertionError(
                            f"V2 holdout identity mismatch for "
                            f"building_seed={building_seed} K={budget}: {name}"
                        )
        records.append(
            {
                "building_seed": building_seed,
                "K": budget,
                "context_rows": int(tree["context_rows"]),
                "context_row_sha256": tree["context_row_sha256"],
                "holdout_row_sha256": tree["holdout_row_sha256"],
                "passed": True,
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.audit_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "audit_passed_ready_for_model_evaluation":
        raise SystemExit("sampling audit did not pass; V2 model sweep is blocked")
    if summary.get("sampling_profile") != "site_stratified_random":
        raise SystemExit("V2 requires site_stratified_random sampling")
    families = list(dict.fromkeys(args.families))
    units = build_units(
        args.audit_root,
        args.out_root,
        summary,
        families=families,
        mode=args.mode,
        model_seed=args.model_seed,
        validation_context_rows=args.validation_context_rows,
        validation_holdout_rows=args.validation_holdout_rows,
    )
    census = {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": args.mode,
        "sampling_profile": summary["sampling_profile"],
        "building_seeds": summary["building_seeds"],
        "budgets": summary["budgets"],
        "features": [137],
        "row_seed": summary["row_seed"],
        "model_seed": args.model_seed,
        "tree_training": "frozen fixed-iteration contract, no early stopping",
        "matched_context": "tree and TabPFN use byte-identical available rows",
        "units": len(units),
        "completed": sum(_complete(unit) for unit in units),
        "pending": sum(not _complete(unit) for unit in units),
        "unit_identities": [unit["identity"] for unit in units],
    }
    if args.mode == "plan":
        print(json.dumps(census, indent=2))
        return 0
    if args.mode == "formal" and set(families) != {"tree", "tabpfn"}:
        raise SystemExit("formal V2 requires both tree and tabpfn families")

    sweep = "building_seed_sweep_" + "-".join(
        str(seed) for seed in summary["building_seeds"]
    )
    sweep_root = args.out_root / sweep
    status_path = sweep_root / "sweep_status.json"
    for index, unit in enumerate(units, start=1):
        if _complete(unit):
            continue
        _atomic_json(
            status_path,
            {
                **census,
                "status": "running",
                "current_unit": index,
                "current_identity": unit["identity"],
            },
        )
        result = subprocess.run(unit["command"], cwd=ROOT)
        if result.returncode or not _complete(unit):
            _atomic_json(
                status_path,
                {
                    **census,
                    "status": "failed",
                    "failed_identity": unit["identity"],
                    "returncode": result.returncode,
                },
            )
            return result.returncode or 2

    context_records = matched_context_gate(units)
    _atomic_json(
        sweep_root / "matched_context_gate.json",
        {
            "experiment_version": EXPERIMENT_VERSION,
            "expected_cells": len(summary["building_seeds"]) * len(summary["budgets"]),
            "checked_cells": len(context_records),
            "passed": (
                len(context_records)
                == len(summary["building_seeds"]) * len(summary["budgets"])
            ),
            "cells": context_records,
        },
    )
    if len(context_records) != len(summary["building_seeds"]) * len(summary["budgets"]):
        raise AssertionError("V2 matched-context gate did not check every seed/K cell")

    aggregate_root = sweep_root / "aggregate"
    cell_paths = [str(Path(unit["output"]) / "cell.json") for unit in units]
    aggregate_command = [
        sys.executable,
        "scripts/report_m5_building_curve.py",
        *cell_paths,
        "--out-root",
        str(aggregate_root),
    ]
    result = subprocess.run(aggregate_command, cwd=ROOT)
    if result.returncode:
        return result.returncode
    report_command = [
        sys.executable,
        "scripts/update_m5_building_count_v2_report.py",
        "--audit-root",
        str(args.audit_root),
        "--aggregate-root",
        str(aggregate_root),
        "--out",
        str(args.report),
    ]
    result = subprocess.run(report_command, cwd=ROOT)
    if result.returncode:
        return result.returncode
    _atomic_json(
        status_path,
        {
            **census,
            "status": "complete",
            "completed": len(units),
            "pending": 0,
            "matched_context_gate": str(sweep_root / "matched_context_gate.json"),
            "aggregate_root": str(aggregate_root),
            "report": str(args.report),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
