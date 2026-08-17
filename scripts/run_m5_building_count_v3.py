"""Plan, validate, or run the dedicated M5 building-count V3 queue.

The default plan mode never launches a model. Validation and formal outputs
are isolated. Formal execution requires the frozen training-context gate, a
successful bounded validation gate, an explicit authorization flag, and a
clean committed implementation.
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

import numpy as np

from lead import PROC, ROOT
from m5_building_count_v3_protocol import (
    BALANCE_SEED,
    BUDGETS,
    BUILDING_SEEDS,
    CANONICAL_HOLDOUT_SHA256,
    CLASS_RATIO_POLICY,
    CONTEXT_ROWS,
    EXPERIMENT_VERSION,
    SAMPLING_PROFILE,
    TRAINING_CONTEXT_POLICY,
    grouped_budget_major_pairs,
    load_balanced_context,
    verify_training_context_gate,
)

DEFAULT_AUDIT_ROOT = ROOT / "experiments/m5_building_count_v3_balanced_context/audit"
DEFAULT_VALIDATION_ROOT = (
    PROC / "m5_building_curve/NON_SCIENTIFIC_VALIDATION_v3_balanced_context"
)
DEFAULT_FORMAL_ROOT = PROC / "m5_building_curve/v3_balanced_context"
DEFAULT_TABPFN_MODEL_PATH = (
    Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".tabpfn-cache")))
    / "tabpfn-v3-classifier-v3_default.ckpt"
)
FAMILIES = ("tree", "tabpfn")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _event(root: Path, event: str, **values: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "events.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, "timestamp": time.time(), **values}))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TABPFN_MODEL_PATH)
    parser.add_argument("--validation-context-rows", type=int, default=200)
    parser.add_argument("--validation-holdout-rows", type=int, default=200)
    parser.add_argument("--authorize-formal", action="store_true")
    args = parser.parse_args(argv)
    if args.model_seed < 0:
        raise ValueError("model seed must be non-negative")
    if args.validation_context_rows <= 0 or args.validation_context_rows % 2:
        raise ValueError("validation context rows must be positive and even")
    if args.validation_holdout_rows <= 0 or args.validation_holdout_rows % 2:
        raise ValueError("validation holdout rows must be positive and even")
    if args.validation_context_rows > CONTEXT_ROWS[10]:
        raise ValueError("validation context rows cannot exceed the K10 context")
    if args.mode != "formal" and args.authorize_formal:
        raise ValueError("--authorize-formal is only valid in formal mode")
    if args.out_root is None:
        args.out_root = (
            DEFAULT_VALIDATION_ROOT
            if args.mode == "validation"
            else DEFAULT_FORMAL_ROOT
        )
    expected_root = (
        DEFAULT_VALIDATION_ROOT if args.mode == "validation" else DEFAULT_FORMAL_ROOT
    )
    if args.mode != "plan" and args.out_root.resolve() != expected_root.resolve():
        marker = "NON_SCIENTIFIC_VALIDATION" in str(args.out_root)
        if args.mode == "validation" and not marker:
            raise ValueError("validation out-root must be visibly non-scientific")
        if args.mode == "formal" and marker:
            raise ValueError("formal out-root cannot be a validation root")
    return args


def build_units(
    audit_root: Path,
    out_root: Path,
    *,
    mode: str,
    model_seed: int,
    model_path: Path,
    validation_context_rows: int,
    validation_holdout_rows: int,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for seed, budget in grouped_budget_major_pairs():
        context_manifest = audit_root / f"balanced_context_seed{seed}.json"
        context = load_balanced_context(context_manifest, budget)
        for family in FAMILIES:
            tag = "tree_no_es" if family == "tree" else "tabpfn"
            output = (
                out_root
                / "model_runs"
                / f"building_seed{seed}"
                / f"{tag}_k{budget}_f137"
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
                str(context.source_manifest_path),
                "--balanced-context-manifest",
                str(context_manifest),
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
                        "--model-path",
                        str(model_path),
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
                        "building_seed": seed,
                        "K": budget,
                        "model": family,
                        "features": 137,
                        "balance_seed": BALANCE_SEED,
                        "model_seed": model_seed,
                        "expected_context_rows": (
                            validation_context_rows
                            if mode == "validation"
                            else CONTEXT_ROWS[budget]
                        ),
                        "expected_holdout_rows": (
                            validation_holdout_rows if mode == "validation" else None
                        ),
                    },
                    "context_manifest": str(context_manifest),
                    "source_building_manifest": str(context.source_manifest_path),
                    "output": str(output),
                    "command": command,
                }
            )
    return units


def _load_cell(unit: dict[str, Any]) -> dict[str, Any] | None:
    output = Path(unit["output"])
    required = (
        output / "COMPLETE.json",
        output / "cell.json",
        output / "predictions.npz",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        return json.loads((output / "cell.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def complete(unit: dict[str, Any], *, mode: str) -> bool:
    metadata = _load_cell(unit)
    if metadata is None:
        return False
    identity = unit["identity"]
    expected_mode = "FORMAL" if mode == "formal" else "NON_SCIENTIFIC_VALIDATION"
    expected_experiment = (
        "m5_building_count_v3_balanced_context_tree_cell"
        if identity["model"] == "tree"
        else "m5_building_count_v3_balanced_context_tabpfn_cell"
    )
    try:
        return (
            metadata["experiment_version"] == EXPERIMENT_VERSION
            and metadata["experiment"] == expected_experiment
            and metadata["mode"] == expected_mode
            and int(metadata["building_seed"]) == identity["building_seed"]
            and int(metadata["building_budget"]) == identity["K"]
            and int(metadata["features"]) == 137
            and int(metadata["row_seed"]) == BALANCE_SEED
            and int(metadata["model_seed"]) == identity["model_seed"]
            and int(metadata["context_rows"]) == identity["expected_context_rows"]
            and int(metadata["holdout_rows"]) > 0
            and metadata["sampling_profile"] == SAMPLING_PROFILE
            and metadata["training_sampling"] == TRAINING_CONTEXT_POLICY
            and metadata["class_ratio_policy"] == CLASS_RATIO_POLICY
            and metadata["validation_feature_scope"]
            == (
                "bounded_context_and_holdout_rows"
                if mode == "validation"
                else "complete_selected_building_histories"
            )
            and (
                mode != "validation"
                or int(metadata["holdout_rows"]) == identity["expected_holdout_rows"]
            )
            and all(
                isinstance(metadata[name], str) and len(metadata[name]) == 64
                for name in (
                    "context_row_sha256",
                    "context_label_sha256",
                    "context_feature_matrix_sha256",
                    "holdout_row_sha256",
                    "manifest_sha256",
                    "source_building_manifest_sha256",
                )
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def matched_context_gate(
    units: list[dict[str, Any]], *, mode: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for unit in units:
        identity = unit["identity"]
        grouped.setdefault((identity["building_seed"], identity["K"]), {})[
            identity["model"]
        ] = unit
    records: list[dict[str, Any]] = []
    for seed, budget in grouped_budget_major_pairs():
        pair = grouped.get((seed, budget), {})
        if set(pair) != set(FAMILIES):
            continue
        metadata = {family: _load_cell(unit) for family, unit in pair.items()}
        if any(value is None for value in metadata.values()):
            continue
        tree = metadata["tree"]
        tabpfn = metadata["tabpfn"]
        assert tree is not None and tabpfn is not None
        fields = (
            "context_row_sha256",
            "context_label_sha256",
            "context_feature_matrix_sha256",
            "context_rows",
            "holdout_row_sha256",
            "holdout_rows",
            "validation_feature_scope",
            "row_policy",
            "row_seed",
            "model_seed",
            "training_sampling",
            "class_ratio_policy",
            "sampling_profile",
            "manifest_sha256",
            "source_building_manifest_sha256",
        )
        mismatches = {
            field: (tree.get(field), tabpfn.get(field))
            for field in fields
            if tree.get(field) != tabpfn.get(field)
        }
        if mismatches:
            raise AssertionError(
                f"V3 matched-context metadata mismatch seed={seed} K={budget}: "
                f"{mismatches}"
            )
        context = load_balanced_context(Path(pair["tree"]["context_manifest"]), budget)
        if mode == "formal":
            if (
                tree["context_row_sha256"]
                != context.manifest["cells"][str(budget)]["raw_index_sha256"]
            ):
                raise AssertionError(
                    f"V3 formal context digest mismatch seed={seed} K={budget}"
                )
            if tree["holdout_row_sha256"] != CANONICAL_HOLDOUT_SHA256:
                raise AssertionError(
                    f"V3 canonical holdout digest mismatch seed={seed} K={budget}"
                )
        prediction_paths = {
            family: Path(unit["output"]) / "predictions.npz"
            for family, unit in pair.items()
        }
        with np.load(prediction_paths["tree"]) as tree_prediction:
            with np.load(prediction_paths["tabpfn"]) as tab_prediction:
                for name in (
                    "validation_raw_index",
                    "anomaly",
                    "building_id",
                    "site_id",
                    "meter",
                ):
                    if not np.array_equal(tree_prediction[name], tab_prediction[name]):
                        raise AssertionError(
                            f"V3 holdout identity mismatch seed={seed} K={budget}: {name}"
                        )
        records.append(
            {
                "building_seed": seed,
                "K": budget,
                "context_rows": int(tree["context_rows"]),
                "context_row_sha256": tree["context_row_sha256"],
                "context_feature_matrix_sha256": tree["context_feature_matrix_sha256"],
                "holdout_row_sha256": tree["holdout_row_sha256"],
                "passed": True,
            }
        )
    return records


def _validation_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"formal V3 requires bounded validation gate: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = len(BUILDING_SEEDS) * len(BUDGETS)
    if not (
        gate.get("experiment_version") == EXPERIMENT_VERSION
        and gate.get("mode") == "validation"
        and gate.get("passed") is True
        and int(gate.get("expected_pairs", -1)) == expected
        and int(gate.get("checked_pairs", -1)) == expected
    ):
        raise SystemExit("formal V3 is blocked by an incomplete validation gate")
    return gate


def _formal_preflight(args: argparse.Namespace) -> None:
    if not args.authorize_formal:
        raise SystemExit("formal V3 requires explicit --authorize-formal")
    _validation_gate(DEFAULT_VALIDATION_ROOT / "matched_context_gate.json")
    if not args.model_path.is_file():
        raise SystemExit(f"TabPFN checkpoint is missing: {args.model_path}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SystemExit("formal V3 requires a clean committed implementation")


def _status_payload(
    units: list[dict[str, Any]],
    *,
    mode: str,
    status: str,
    durations: list[float],
    **values: Any,
) -> dict[str, Any]:
    completed = sum(complete(unit, mode=mode) for unit in units)
    mean = float(np.mean(durations)) if durations else None
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": mode,
        "status": status,
        "units": len(units),
        "completed": completed,
        "pending": len(units) - completed,
        "mean_completed_unit_seconds": mean,
        "eta_seconds": mean * (len(units) - completed) if mean is not None else None,
        "timestamp": time.time(),
        **values,
    }


def _checkpoint_pair(
    *,
    supervisor: Path,
    status_path: Path,
    units: list[dict[str, Any]],
    pair_units: list[dict[str, Any]],
    mode: str,
    durations: list[float],
) -> bool:
    identity = pair_units[0]["identity"]
    seed = int(identity["building_seed"])
    budget = int(identity["K"])
    stage = f"building_seed{seed}_k{budget}"
    try:
        records = matched_context_gate(pair_units, mode=mode)
        if len(records) != 1 or records[0].get("passed") is not True:
            raise AssertionError(f"V3 pair gate did not pass: {stage}")
        payload = {
            "experiment_version": EXPERIMENT_VERSION,
            "mode": mode,
            "stage": stage,
            "passed": True,
            "record": records[0],
            "timestamp": time.time(),
        }
        _atomic_json(supervisor / "pair_gates" / f"{stage}.json", payload)
        _event(supervisor, "pair_gate_passed", **payload)
        return True
    except Exception as error:  # Control boundary records gate diagnostics.
        failure = _status_payload(
            units,
            mode=mode,
            status="failed",
            durations=durations,
            failed_pair=stage,
            reason=f"{type(error).__name__}: {error}",
        )
        _atomic_json(status_path, failure)
        _atomic_json(supervisor / "FAILED.json", failure)
        _event(supervisor, "pair_gate_failed", **failure)
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    training_gate = verify_training_context_gate(args.audit_root)
    units = build_units(
        args.audit_root,
        args.out_root,
        mode=args.mode,
        model_seed=args.model_seed,
        model_path=args.model_path,
        validation_context_rows=args.validation_context_rows,
        validation_holdout_rows=args.validation_holdout_rows,
    )
    pairs = grouped_budget_major_pairs()
    census = {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": args.mode,
        "pair_order_policy": "group42_46_then_group47_51_budget_major_within_group",
        "seed_groups": [[42, 43, 44, 45, 46], [47, 48, 49, 50, 51]],
        "budgets": list(BUDGETS),
        "pairs": len(pairs),
        "units": len(units),
        "families_per_pair": list(FAMILIES),
        "tabpfn_model_path": str(args.model_path.resolve()),
        "training_context_gate": training_gate,
        "unit_identities": [unit["identity"] for unit in units],
    }
    if args.mode == "plan":
        print(json.dumps(census, indent=2))
        return 0
    if args.mode == "formal":
        _formal_preflight(args)

    supervisor = args.out_root / "supervisor"
    status_path = supervisor / "status.json"
    _atomic_json(supervisor / "unit_census.json", census)
    _event(supervisor, "queue_start", mode=args.mode, units=len(units))
    durations: list[float] = []
    for index, unit in enumerate(units, start=1):
        if complete(unit, mode=args.mode):
            _event(supervisor, "unit_reused", index=index, identity=unit["identity"])
            if index % len(FAMILIES) == 0 and not _checkpoint_pair(
                supervisor=supervisor,
                status_path=status_path,
                units=units,
                pair_units=units[index - len(FAMILIES) : index],
                mode=args.mode,
                durations=durations,
            ):
                return 3
            continue
        status = _status_payload(
            units,
            mode=args.mode,
            status="running",
            durations=durations,
            current_unit=index,
            current_identity=unit["identity"],
        )
        _atomic_json(status_path, status)
        _atomic_json(supervisor / "heartbeat.json", status)
        _event(
            supervisor,
            "unit_start",
            index=index,
            identity=unit["identity"],
            command=unit["command"],
        )
        started = time.perf_counter()
        result = subprocess.run(unit["command"], cwd=ROOT)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        if result.returncode or not complete(unit, mode=args.mode):
            failure = _status_payload(
                units,
                mode=args.mode,
                status="failed",
                durations=durations,
                failed_unit=index,
                failed_identity=unit["identity"],
                returncode=result.returncode,
            )
            _atomic_json(status_path, failure)
            _atomic_json(supervisor / "FAILED.json", failure)
            _event(supervisor, "unit_failed", **failure)
            return result.returncode or 2
        _event(
            supervisor,
            "unit_complete",
            index=index,
            identity=unit["identity"],
            elapsed_seconds=elapsed,
        )
        if index % len(FAMILIES) == 0 and not _checkpoint_pair(
            supervisor=supervisor,
            status_path=status_path,
            units=units,
            pair_units=units[index - len(FAMILIES) : index],
            mode=args.mode,
            durations=durations,
        ):
            return 3

    records = matched_context_gate(units, mode=args.mode)
    expected = len(BUILDING_SEEDS) * len(BUDGETS)
    gate = {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": args.mode,
        "expected_pairs": expected,
        "checked_pairs": len(records),
        "passed": len(records) == expected,
        "cells": records,
    }
    _atomic_json(args.out_root / "matched_context_gate.json", gate)
    if not gate["passed"]:
        raise AssertionError("V3 matched-context gate did not check all 40 pairs")

    aggregate_root = args.out_root / "aggregate"
    command = [
        sys.executable,
        "scripts/report_m5_building_curve.py",
        *[str(Path(unit["output"]) / "cell.json") for unit in units],
        "--out-root",
        str(aggregate_root),
    ]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        return result.returncode
    final = _status_payload(
        units,
        mode=args.mode,
        status="completed",
        durations=durations,
        matched_context_gate=str(args.out_root / "matched_context_gate.json"),
        aggregate_root=str(aggregate_root),
    )
    _atomic_json(status_path, final)
    _atomic_json(supervisor / "heartbeat.json", final)
    _atomic_json(supervisor / "COMPLETE.json", final)
    _event(supervisor, "queue_complete", **final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
