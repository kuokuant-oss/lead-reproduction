"""Aggregate V5 cells while preserving row/support identities."""

from __future__ import annotations

import json
from typing import Any

import report_m5_building_curve as report


def main() -> int:
    original = report.aggregate_cell

    def aggregate_v5(
        metadata: dict[str, Any], payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        meters = set(map(int, payload["meter"]))
        if meters != {1, 2, 3}:
            raise ValueError(f"V5 predictions must contain meters 1/2/3 only: {meters}")
        metrics, curves = original(metadata, payload)
        for row in (*metrics, *curves):
            row["row_seed"] = metadata.get("row_seed")
            row["support_id"] = metadata.get("support_id")
            row["building_draw_seed"] = metadata.get("building_draw_seed")
        return metrics, curves

    report.aggregate_cell = aggregate_v5
    result = report.main()
    if result:
        return result
    args = report.parse_args()
    summary_path = args.out_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity_by_path = {}
        for path in args.cells:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            identity_by_path[str(path)] = {
                "row_seed": metadata.get("row_seed"),
                "support_id": metadata.get("support_id"),
                "building_draw_seed": metadata.get("building_draw_seed"),
            }
        for record in summary.get("cells", []):
            record.update(identity_by_path.get(record["metadata"], {}))
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
