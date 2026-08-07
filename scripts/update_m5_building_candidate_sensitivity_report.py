"""Append or replace the completed building-candidate sensitivity report section."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import PROC, ROOT

BEGIN = "<!-- BEGIN M5 BUILDING CANDIDATE SENSITIVITY PILOT -->"
END = "<!-- END M5 BUILDING CANDIDATE SENSITIVITY PILOT -->"
DEFAULT_AUDIT_ROOT = (
    PROC / "m5_building_curve" / "sensitivity" / "building_candidate_pilot"
)
DEFAULT_REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    rows = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in rows.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return [*lines, ""]


def _aggregate_root(audit_root: Path, seeds: list[int]) -> Path:
    tag = "building_seed_sweep_" + "-".join(str(seed) for seed in seeds)
    return audit_root / "model_results" / tag


def render_section(audit_root: Path) -> str:
    audit = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in audit["building_seeds"]]
    budgets = [int(budget) for budget in audit["budgets"]]
    if audit.get("status") != "audit_passed_ready_for_model_evaluation":
        raise AssertionError("building candidate sensitivity audit did not pass")
    aggregate_root = _aggregate_root(audit_root, seeds)
    aggregate = json.loads(
        (aggregate_root / "summary.json").read_text(encoding="utf-8")
    )
    if len(aggregate["cells"]) != len(seeds) * len(budgets) * 2:
        raise AssertionError("sensitivity aggregate does not contain all model cells")

    overlap = pd.read_csv(audit_root / "building_overlap.csv").sort_values(
        ["K", "seed_a", "seed_b"]
    )
    composition = pd.read_csv(audit_root / "composition_audit.csv").sort_values(
        ["building_seed", "K"]
    )
    summary = pd.read_csv(aggregate_root / "building_seed_summary.csv").sort_values(
        ["building_budget", "model"]
    )
    metrics = pd.read_csv(aggregate_root / "metrics.csv")
    if summary.empty or not summary["n_building_seeds"].eq(len(seeds)).all():
        raise AssertionError("cross-seed summary is incomplete")
    if not composition["quality_gate_pass"].all():
        raise AssertionError("a composition quality gate failed")

    headline_models = summary["model"].isin(["ensemble", "tabpfn"])
    headline = summary.loc[headline_models].rename(columns={"building_budget": "K"})
    raw = metrics.loc[
        metrics["grouping"].eq("overall")
        & metrics["model"].isin(["ensemble", "tabpfn"])
    ].copy()
    raw = raw.rename(columns={"building_budget": "K"}).sort_values(
        ["model", "K", "building_seed"]
    )
    composition_view = composition.rename(
        columns={
            "total_available_rows": "rows",
            "total_anomaly_rows": "anomalies",
            "natural_anomaly_prevalence": "prevalence",
        }
    )

    lines = [
        BEGIN,
        "",
        "## Building-candidate sensitivity pilot",
        "",
        "This secondary pilot varies only the site-stratified random building "
        f"draw across seeds {seeds}. The row seed and model seed remain 42; each "
        "seed uses one strict-nested K=10/20/50/100 ladder, and every model is "
        "evaluated on the identical odd-building canonical natural-prevalence holdout.",
        "",
        f"- Sampling audit: passed for all {len(seeds) * len(budgets)} seed/K prefixes.",
        "- Random draw: PCG64 within-site permutations without replacement, "
        "interleaved by candidate-pool site proportions.",
        "- Feasibility gate: every meter has at least two source buildings at K=10 "
        "and gains at least one source building at each K transition; failed whole "
        "ladders are deterministically redrawn without greedy correction.",
        f"- Raw per-seed metrics: `{aggregate_root / 'metrics.csv'}`.",
        f"- Cross-seed summary: `{aggregate_root / 'building_seed_summary.csv'}`.",
        "",
        "### Cross-seed overall results",
        "",
    ]
    lines.extend(
        _table(
            headline,
            [
                "model",
                "K",
                "n_building_seeds",
                "pr_auc_mean",
                "pr_auc_std",
                "pr_auc_min",
                "pr_auc_max",
                "roc_auc_mean",
                "roc_auc_std",
            ],
        )
    )
    lines.extend(["### Per-seed overall results", ""])
    lines.extend(_table(raw, ["model", "K", "building_seed", "pr_auc", "roc_auc"]))
    lines.extend(["### Building overlap", ""])
    lines.extend(
        _table(
            overlap,
            [
                "K",
                "seed_a",
                "seed_b",
                "intersection_count",
                "jaccard_similarity",
            ],
        )
    )
    lines.extend(["### Composition-quality audit", ""])
    lines.extend(
        _table(
            composition_view,
            [
                "building_seed",
                "K",
                "rows",
                "anomalies",
                "prevalence",
                "prefix_discrepancy",
                "quality_gate_pass",
            ],
        )
    )
    lines.extend([END, ""])
    return "\n".join(lines)


def replace_section(report: str, section: str) -> str:
    if BEGIN in report or END in report:
        if report.count(BEGIN) != 1 or report.count(END) != 1:
            raise AssertionError("sensitivity report markers are unbalanced")
        before, remainder = report.split(BEGIN, maxsplit=1)
        _, after = remainder.split(END, maxsplit=1)
        return before.rstrip() + "\n\n" + section.rstrip() + after
    return report.rstrip() + "\n\n" + section


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    section = render_section(args.audit_root)
    current = args.report.read_text(encoding="utf-8")
    _atomic_text(args.report, replace_section(current, section))
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
