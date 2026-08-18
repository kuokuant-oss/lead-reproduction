"""Preserve row-seed identity while using the standard M5 curve reporter."""

from __future__ import annotations

import json
from typing import Any

import report_m5_building_curve as report


def main() -> int:
    original = report.aggregate_cell

    def aggregate_with_row_seed(
        metadata: dict[str, Any], payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        metrics, curves = original(metadata, payload)
        for row in (*metrics, *curves):
            row["row_seed"] = metadata.get("row_seed")
        return metrics, curves

    report.aggregate_cell = aggregate_with_row_seed
    result = report.main()
    if result:
        return result
    args = report.parse_args()
    summary_path = args.out_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row_seed_by_path = {
            str(path): json.loads(path.read_text(encoding="utf-8")).get("row_seed")
            for path in args.cells
        }
        for cell_record in summary.get("cells", []):
            cell_record["row_seed"] = row_seed_by_path.get(cell_record["metadata"])
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
