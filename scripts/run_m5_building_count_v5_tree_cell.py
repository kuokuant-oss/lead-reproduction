"""V5 adapter for the checkpointed frozen Tree Ensemble cell."""

from __future__ import annotations

from typing import Any

import run_m5_building_count_v2_tree_cell as cell
from m5_building_count_v5_protocol import (
    CLASS_RATIO_POLICY,
    DEFAULT_HOLDOUT_ARTIFACT,
    EXPERIMENT_VERSION,
    TRAINING_CONTEXT_POLICY,
    verify_context_against_frame,
)
from m5_building_count_v5_runtime import load_fixed_context


def _configure() -> None:
    cell.V3_EXPERIMENT_VERSION = EXPERIMENT_VERSION
    cell.CLASS_RATIO_POLICY = CLASS_RATIO_POLICY
    cell.TRAINING_CONTEXT_POLICY = TRAINING_CONTEXT_POLICY
    cell.load_balanced_context = load_fixed_context
    cell.verify_context_against_frame = verify_context_against_frame
    cell.CANONICAL_ORDER = DEFAULT_HOLDOUT_ARTIFACT
    original = cell.write_json_with_provenance

    def write_v5(path: Any, payload: dict[str, Any], **kwargs: Any) -> None:
        updated = dict(payload)
        updated["experiment"] = f"{EXPERIMENT_VERSION}_tree_cell"
        budget = int(updated["building_budget"])
        updated["support_id"] = (
            "full_725" if budget == 725 else f"building_seed{updated['building_seed']}"
        )
        updated["building_draw_seed"] = (
            None if budget == 725 else int(updated["building_seed"])
        )
        updated["evaluation_meter_ids"] = [1, 2, 3]
        updated["excluded_evaluation_meter_ids"] = [0]
        original(path, updated, **kwargs)

    cell.write_json_with_provenance = write_v5


def main() -> int:
    _configure()
    return cell.main()


if __name__ == "__main__":
    raise SystemExit(main())
