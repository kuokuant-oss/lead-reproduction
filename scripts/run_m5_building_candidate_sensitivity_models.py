"""Run the isolated M5 building-candidate sensitivity model sweep.

The default plan mode only prints the unit census. Formal mode must be requested
explicitly and reuses the checkpointed tree/TabPFN cell runners.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lead import PROC, ROOT

DEFAULT_AUDIT_ROOT = (
    PROC / "m5_building_curve" / "sensitivity" / "building_candidate_pilot"
)


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
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--families", nargs="+", choices=("tree", "tabpfn"), default=["tree", "tabpfn"])
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--validation-fit-rows", type=int, default=200)
    parser.add_argument("--validation-early-stop-rows", type=int, default=100)
    parser.add_argument("--validation-context-rows", type=int, default=200)
    parser.add_argument("--validation-holdout-rows", type=int, default=200)
    return parser.parse_args(argv)


def _units(
    audit_root: Path,
    summary: dict[str, Any],
    *,
    families: list[str],
    mode: str,
    model_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for building_seed in summary["building_seeds"]:
        manifest = audit_root / f"building_ladder_seed{building_seed}.json"
        for budget in summary["budgets"]:
            for family in families:
                output = (
                    audit_root
                    / "model_runs"
                    / f"building_seed{building_seed}"
                    / f"{family}_k{budget}_f137"
                )
                script = (
                    "scripts/run_m5_building_curve_tree_cell.py"
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
                    command.extend(["--n-estimators", "8"])
                if mode == "validation":
                    if family == "tree":
                        command.extend(
                            [
                                "--max-fit-rows",
                                str(args.validation_fit_rows),
                                "--max-early-stop-rows",
                                str(args.validation_early_stop_rows),
                                "--max-holdout-rows",
                                str(args.validation_holdout_rows),
                            ]
                        )
                    else:
                        command.extend(
                            [
                                "--max-context-rows",
                                str(args.validation_context_rows),
                                "--max-holdout-rows",
                                str(args.validation_holdout_rows),
                            ]
                        )
                units.append(
                    {
                        "identity": {
                            "sampling_profile": summary["sampling_profile"],
                            "building_seed": int(building_seed),
                            "K": int(budget),
                            "features": 137,
                            "model": family,
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
        int(metadata.get("building_seed", -1)) == identity["building_seed"]
        and int(metadata.get("building_budget", -1)) == identity["K"]
        and int(metadata.get("features", -1)) == identity["features"]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.audit_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "audit_passed_ready_for_model_evaluation":
        raise SystemExit("selection audit did not pass; model sweep is blocked")
    units = _units(
        args.audit_root,
        summary,
        families=list(dict.fromkeys(args.families)),
        mode=args.mode,
        model_seed=args.model_seed,
        args=args,
    )
    census = {
        "mode": args.mode,
        "units": len(units),
        "completed": sum(_complete(unit) for unit in units),
        "pending": sum(not _complete(unit) for unit in units),
        "unit_identities": [unit["identity"] for unit in units],
    }
    if args.mode == "plan":
        print(json.dumps(census, indent=2))
        return 0

    status_path = args.audit_root / "model_sweep_status.json"
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

    sweep_tag = "building_seed_sweep_" + "-".join(
        str(seed) for seed in summary["building_seeds"]
    )
    aggregate_root = args.audit_root / "model_results" / sweep_tag
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
    _atomic_json(
        status_path,
        {
            **census,
            "status": "complete",
            "completed": len(units),
            "pending": 0,
            "aggregate_root": str(aggregate_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
