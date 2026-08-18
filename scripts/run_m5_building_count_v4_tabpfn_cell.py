"""V4 adapter for the checkpointed TabPFN building-count cell."""

from __future__ import annotations

from typing import Any

import run_m5_building_curve_tabpfn_cell as cell
from m5_building_count_v4_protocol import (
    CLASS_RATIO_POLICY,
    EXPERIMENT_VERSION,
    TRAINING_CONTEXT_POLICY,
    verify_context_against_frame,
)
from m5_building_count_v4_runtime import load_fixed_context


def _configure() -> None:
    cell.V3_EXPERIMENT_VERSION = EXPERIMENT_VERSION
    cell.CLASS_RATIO_POLICY = CLASS_RATIO_POLICY
    cell.TRAINING_CONTEXT_POLICY = TRAINING_CONTEXT_POLICY
    cell.load_balanced_context = load_fixed_context
    cell.verify_context_against_frame = verify_context_against_frame
    original = cell.write_json_with_provenance

    def write_v4(path: Any, payload: dict[str, Any], **kwargs: Any) -> None:
        updated = dict(payload)
        updated["experiment"] = f"{EXPERIMENT_VERSION}_tabpfn_cell"
        original(path, updated, **kwargs)

    cell.write_json_with_provenance = write_v4


def main() -> int:
    _configure()
    return cell.main()


if __name__ == "__main__":
    raise SystemExit(main())
