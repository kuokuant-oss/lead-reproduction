"""Plan, validate, or run the three-phase fixed-50K M5 V5 queue."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

from lead import PROC, ROOT
import m5_building_count_v5_protocol as protocol


_BASE_SCHEDULER_PATH = Path(__file__).with_name("run_m5_building_count_v4.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_m5_building_count_v5_private_scheduler", _BASE_SCHEDULER_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"cannot load checkpointed scheduler: {_BASE_SCHEDULER_PATH}")
scheduler = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(scheduler)


DEFAULT_VALIDATION_ROOT = (
    PROC / "m5_building_curve/NON_SCIENTIFIC_VALIDATION_v5_fixed_50k"
)
DEFAULT_FORMAL_ROOT = PROC / "m5_building_curve/v5_fixed_50k"
_CONFIGURED = False


def _selected_contexts(mode: str) -> list[tuple[int, int, int]]:
    return (
        list(protocol.VALIDATION_CONTEXTS)
        if mode == "validation"
        else protocol.queued_contexts()
    )


def _build_units(
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
    for building_seed, row_seed, budget in _selected_contexts(mode):
        context_path = protocol.context_manifest_path(
            audit_root, building_seed, row_seed, budget
        )
        context = protocol.load_fixed_context(context_path, budget)
        for family in scheduler.FAMILIES:
            tag = "tree_no_es" if family == "tree" else "tabpfn"
            output = (
                out_root
                / "model_runs"
                / f"building_seed{building_seed}"
                / f"row_seed{row_seed}"
                / f"{tag}_k{budget}_f137"
            )
            script = (
                "scripts/run_m5_building_count_v5_tree_cell.py"
                if family == "tree"
                else "scripts/run_m5_building_count_v5_tabpfn_cell.py"
            )
            command = [
                sys.executable,
                script,
                "--building-manifest",
                str(context.source_manifest_path),
                "--balanced-context-manifest",
                str(context_path),
                "--building-budget",
                str(budget),
                "--features",
                "137",
                "--model-seed",
                str(model_seed),
            ]
            if family == "tabpfn":
                command.extend(
                    [
                        "--experiment-version",
                        protocol.EXPERIMENT_VERSION,
                        "--n-estimators",
                        "8",
                        "--model-path",
                        str(model_path),
                    ]
                )
            command.extend(
                ["--mode", mode, "--resume", "--out-root", str(output)]
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
                        "experiment_version": protocol.EXPERIMENT_VERSION,
                        "building_seed": building_seed,
                        "building_draw_seed": context.building_draw_seed,
                        "support_id": context.support_id,
                        "row_seed": row_seed,
                        "K": budget,
                        "model": family,
                        "features": 137,
                        "model_seed": model_seed,
                        "phase": protocol.phase_for_context(
                            building_seed, row_seed, budget
                        ),
                        "expected_context_rows": (
                            validation_context_rows
                            if mode == "validation"
                            else protocol.CONTEXT_ROWS
                        ),
                        "expected_holdout_rows": (
                            validation_holdout_rows
                            if mode == "validation"
                            else None
                        ),
                    },
                    "context_manifest": str(context_path),
                    "source_building_manifest": str(context.source_manifest_path),
                    "output": str(output),
                    "command": command,
                }
            )
    return units


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    scheduler.BUDGETS = protocol.BUDGETS
    scheduler.BUILDING_DRAW_SEEDS = protocol.BUILDING_DRAW_SEEDS
    scheduler.ROW_DRAW_SEEDS = protocol.ROW_DRAW_SEEDS
    scheduler.SCHEDULED_BUDGETS = protocol.SCHEDULED_BUDGETS
    scheduler.CONTEXT_ROWS = protocol.CONTEXT_ROWS
    scheduler.EXPERIMENT_VERSION = protocol.EXPERIMENT_VERSION
    scheduler.SAMPLING_PROFILE = protocol.SAMPLING_PROFILE
    scheduler.TRAINING_CONTEXT_POLICY = protocol.TRAINING_CONTEXT_POLICY
    scheduler.CLASS_RATIO_POLICY = protocol.CLASS_RATIO_POLICY
    scheduler.PAIR_ORDER_POLICY = protocol.PAIR_ORDER_POLICY
    scheduler.VALIDATION_CONTEXTS = protocol.VALIDATION_CONTEXTS
    scheduler.DEFAULT_AUDIT_ROOT = protocol.DEFAULT_AUDIT_ROOT
    scheduler.DEFAULT_VALIDATION_ROOT = DEFAULT_VALIDATION_ROOT
    scheduler.DEFAULT_FORMAL_ROOT = DEFAULT_FORMAL_ROOT
    scheduler.selected_contexts = _selected_contexts
    scheduler.build_units = _build_units
    scheduler.load_fixed_context = protocol.load_fixed_context
    scheduler.verify_training_context_gate = protocol.verify_training_context_gate

    original_gate = scheduler.matched_context_gate

    def matched_v5(
        units: list[dict[str, Any]], *, mode: str
    ) -> list[dict[str, Any]]:
        scheduler.CANONICAL_HOLDOUT_SHA256 = protocol.canonical_holdout_row_sha256()
        return original_gate(units, mode=mode)

    scheduler.matched_context_gate = matched_v5

    original_checkpoint = scheduler._checkpoint_pair

    def checkpoint_v5(**kwargs: Any) -> bool:
        passed = original_checkpoint(**kwargs)
        if not passed or kwargs["mode"] != "formal":
            return passed
        pair_units = kwargs["pair_units"]
        identity = pair_units[0]["identity"]
        context = (
            int(identity["building_seed"]),
            int(identity["row_seed"]),
            int(identity["K"]),
        )
        phase = int(identity["phase"])
        if protocol.phase_final_contexts().get(phase) == context:
            payload = {
                "experiment_version": protocol.EXPERIMENT_VERSION,
                "status": "PASSED",
                "phase": phase,
                "final_context": {
                    "building_seed": context[0],
                    "row_seed": context[1],
                    "K": context[2],
                },
                "timestamp": time.time(),
            }
            supervisor = kwargs["supervisor"]
            scheduler._atomic_json(
                supervisor / "phase_gates" / f"phase_{phase}_COMPLETE.json",
                payload,
            )
            scheduler._event(supervisor, "phase_complete", **payload)
        return passed

    scheduler._checkpoint_pair = checkpoint_v5

    original_run = scheduler.subprocess.run

    def run_v5(command: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(command, list)
            and len(command) > 1
            and command[1] == "scripts/report_m5_building_curve.py"
        ):
            command = list(command)
            command[1] = "scripts/report_m5_building_count_v5.py"
        return original_run(command, *args, **kwargs)

    scheduler.subprocess.run = run_v5
    _CONFIGURED = True


def main() -> int:
    _configure()
    return scheduler.main()


if __name__ == "__main__":
    raise SystemExit(main())
