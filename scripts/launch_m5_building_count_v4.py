"""Launch the V4 scheduler with dedicated cell and reporting adapters."""

from __future__ import annotations

from typing import Any

import run_m5_building_count_v4 as scheduler

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    original_build_units = scheduler.build_units

    def build_v4_units(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        units = original_build_units(*args, **kwargs)
        for unit in units:
            command = list(unit["command"])
            context_flag = command.index("--frozen-context-manifest")
            command[context_flag] = "--balanced-context-manifest"
            family = unit["identity"]["model"]
            command[1] = (
                "scripts/run_m5_building_count_v4_tree_cell.py"
                if family == "tree"
                else "scripts/run_m5_building_count_v4_tabpfn_cell.py"
            )
            if family == "tree":
                version_flag = command.index("--experiment-version")
                del command[version_flag : version_flag + 2]
            unit["command"] = command
        return units

    scheduler.build_units = build_v4_units
    original_run = scheduler.subprocess.run

    def run_v4(command: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(command, list)
            and len(command) > 1
            and command[1] == "scripts/report_m5_building_curve.py"
        ):
            command = list(command)
            command[1] = "scripts/report_m5_building_count_v4.py"
        return original_run(command, *args, **kwargs)

    scheduler.subprocess.run = run_v4
    _CONFIGURED = True


def main() -> int:
    _configure()
    return scheduler.main()


if __name__ == "__main__":
    raise SystemExit(main())
