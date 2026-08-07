"""Update the tracked M5 V2 report with resumable overnight progress."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lead import ROOT
from run_m5_building_count_v2 import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUT_ROOT,
    _complete,
    build_units,
    ordered_seed_budget_pairs,
)

BEGIN = "<!-- BEGIN M5 BUILDING COUNT V2 RUN PROGRESS -->"
END = "<!-- END M5 BUILDING COUNT V2 RUN PROGRESS -->"
DEFAULT_REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment_V2.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--supervisor-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--last-pair", default="")
    args = parser.parse_args(argv)
    if args.supervisor_root is None:
        args.supervisor_root = args.out_root / "overnight"
    return args


def _metric(unit: dict[str, Any], score: str, metric: str) -> str:
    if not _complete(unit):
        return ""
    metadata = json.loads(
        (Path(unit["output"]) / "cell.json").read_text(encoding="utf-8")
    )
    value = metadata.get("metrics", {}).get(score, {}).get(metric)
    return "" if value is None else f"{float(value):.6f}"


def _stage(unit: dict[str, Any]) -> str:
    identity = unit["identity"]
    return (
        f"building_seed{identity['building_seed']}_k{identity['K']}_{identity['model']}"
    )


def _failure(supervisor_root: Path, unit: dict[str, Any]) -> bool:
    return (supervisor_root / "failed_stages" / f"{_stage(unit)}.json").is_file()


def render_progress(
    summary: dict[str, Any],
    units: list[dict[str, Any]],
    supervisor_root: Path,
    *,
    last_pair: str,
) -> str:
    by_pair: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for unit in units:
        identity = unit["identity"]
        by_pair.setdefault((identity["building_seed"], identity["K"]), {})[
            identity["model"]
        ] = unit

    rows: list[str] = []
    complete_pairs = 0
    failed_pairs = 0
    for order, pair in enumerate(ordered_seed_budget_pairs(summary), start=1):
        families = by_pair[pair]
        tree = families["tree"]
        tabpfn = families["tabpfn"]
        tree_done = _complete(tree)
        tabpfn_done = _complete(tabpfn)
        failed = _failure(supervisor_root, tree) or _failure(supervisor_root, tabpfn)
        if tree_done and tabpfn_done:
            status = "complete"
            complete_pairs += 1
        elif failed:
            status = "failed/skipped"
            failed_pairs += 1
        elif tree_done or tabpfn_done:
            status = "partial"
        else:
            status = "pending"
        rows.append(
            "| "
            + " | ".join(
                [
                    str(order),
                    str(pair[0]),
                    str(pair[1]),
                    status,
                    "yes" if tree_done else "no",
                    "yes" if tabpfn_done else "no",
                    _metric(tree, "ensemble", "pr_auc"),
                    _metric(tree, "ensemble", "roc_auc"),
                    _metric(tabpfn, "tabpfn", "pr_auc"),
                    _metric(tabpfn, "tabpfn", "roc_auc"),
                ]
            )
            + " |"
        )

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    header = [
        BEGIN,
        "",
        "## Overnight formal-run progress",
        "",
        f"- Last update: {timestamp}.",
        f"- Last checkpointed pair: {last_pair or 'none'}.",
        f"- Completed seed/K pairs: {complete_pairs}/{len(by_pair)}.",
        f"- Failed/skipped seed/K pairs: {failed_pairs}.",
        "- A pair is complete only after both frozen no-ES trees and TabPFN finish.",
        "- Raw model artifacts remain under the ignored V2 data root; this tracked "
        "table is committed and pushed after each completed pair.",
        "",
        "| order | building_seed | K | status | tree | TabPFN | "
        "ensemble PR-AUC | ensemble ROC-AUC | TabPFN PR-AUC | TabPFN ROC-AUC |",
        "|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
        *rows,
        "",
        END,
    ]
    return "\n".join(header)


def replace_progress(report: str, section: str) -> str:
    if BEGIN in report or END in report:
        if report.count(BEGIN) != 1 or report.count(END) != 1:
            raise ValueError("V2 report has malformed overnight progress markers")
        start = report.index(BEGIN)
        end = report.index(END, start) + len(END)
        return report[:start].rstrip() + "\n\n" + section + report[end:]
    return report.rstrip() + "\n\n" + section + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads((args.audit_root / "summary.json").read_text(encoding="utf-8"))
    units = build_units(
        args.audit_root,
        args.out_root,
        summary,
        families=["tree", "tabpfn"],
        mode="formal",
        model_seed=42,
        validation_context_rows=200,
        validation_holdout_rows=200,
    )
    report = args.report.read_text(encoding="utf-8")
    section = render_progress(
        summary,
        units,
        args.supervisor_root,
        last_pair=args.last_pair,
    )
    _atomic_text(args.report, replace_progress(report, section))
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
