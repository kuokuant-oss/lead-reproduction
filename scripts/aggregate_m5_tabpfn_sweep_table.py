"""Build the n=1 -> 4 -> 8 per-site gain table the sweep plan calls for.

Reads the per-cell metric files written by evaluate_m5_tabpfn_site_sweep.py and
emits one table. A cell is only included when its evaluation actually ran, so a
missing or still-running cell shows as absent rather than being quietly filled
in. The n=1 column is the official merged artifact, identical across cells by
construction, and is cross-checked here: if two cells disagree on the n=1
baseline for the same site, the rows were not the same and the table is refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lead import PROC

SWEPT_ESTIMATORS = (4, 8)


def load_cells(sites: list[int]) -> dict[int, dict[int, dict[str, Any]]]:
    cells: dict[int, dict[int, dict[str, Any]]] = {}
    for site in sites:
        for n in SWEPT_ESTIMATORS:
            path = PROC / f"m5_tabpfn_site{site}_n{n}_sweep_metrics.json"
            if not path.is_file():
                continue
            cells.setdefault(site, {})[n] = json.loads(path.read_text(encoding="utf-8"))
    return cells


def baseline_for(site: int, by_estimator: dict[int, dict[str, Any]]) -> dict[str, Any]:
    baselines = {n: cell["baseline_n1"] for n, cell in by_estimator.items()}
    reference = next(iter(baselines.values()))
    for n, value in baselines.items():
        if value != reference:
            raise AssertionError(
                f"site {site}: n={n} reports a different n=1 baseline "
                f"({value} != {reference}); the cells did not score the same rows"
            )
    return reference


def render(cells: dict[int, dict[int, dict[str, Any]]]) -> str:
    lines = [
        "| site | rows | prevalence | metric | n=1 | n=4 | n=8 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for site in sorted(cells):
        by_estimator = cells[site]
        baseline = baseline_for(site, by_estimator)
        any_cell = next(iter(by_estimator.values()))
        rows = any_cell["rows"]
        prevalence = any_cell["prevalence"]
        for metric in ("roc_auc", "pr_auc"):
            values = []
            for n in SWEPT_ESTIMATORS:
                cell = by_estimator.get(n)
                if cell is None:
                    values.append("_pending_")
                else:
                    delta = cell["swept"][metric] - baseline[metric]
                    values.append(f"{cell['swept'][metric]:.4f} ({delta:+.4f})")
            label = "ROC-AUC" if metric == "roc_auc" else "PR-AUC"
            lines.append(
                f"| Site {site} | {rows:,} | {prevalence:.3%} | {label} | "
                f"{baseline[metric]:.4f} | {values[0]} | {values[1]} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out", type=Path, default=PROC / "m5_tabpfn_estimator_sweep_table.json"
    )
    args = parser.parse_args(argv)

    cells = load_cells(args.sites)
    if not cells:
        raise SystemExit("no evaluated cells found; run the per-cell evaluator first")

    table = render(cells)
    print(table)
    summary = {
        "sites": {
            str(site): {
                "rows": next(iter(by_n.values()))["rows"],
                "anomalies": next(iter(by_n.values()))["anomalies"],
                "prevalence": next(iter(by_n.values()))["prevalence"],
                "baseline_n1": baseline_for(site, by_n),
                "swept": {str(n): cell["swept"] for n, cell in by_n.items()},
                "delta": {str(n): cell["delta"] for n, cell in by_n.items()},
            }
            for site, by_n in cells.items()
        },
        "markdown_table": table,
    }
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
