"""Regenerate the M5 building-count V2 report from completed artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from lead import PROC, ROOT

DEFAULT_AUDIT_ROOT = (
    PROC / "m5_building_curve" / "sensitivity" / "building_candidate_pilot"
)
DEFAULT_AGGREGATE_ROOT = (
    PROC
    / "m5_building_curve"
    / "v2"
    / "building_seed_sweep_42-43-44-45-46"
    / "aggregate"
)
DEFAULT_REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment_V2.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _format(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    selected = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return lines


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
    audit = json.loads((args.audit_root / "summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "audit_passed_ready_for_model_evaluation":
        raise SystemExit("V2 sampling audit did not pass")
    seeds = [int(value) for value in audit["building_seeds"]]
    budgets = [int(value) for value in audit["budgets"]]

    metrics = pd.read_csv(args.aggregate_root / "metrics.csv")
    summary = pd.read_csv(args.aggregate_root / "building_seed_summary.csv")
    prefix = pd.read_csv(args.audit_root / "sampling_prefix_audit.csv")
    gate_path = args.aggregate_root.parent / "matched_context_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise SystemExit("V2 tree/TabPFN matched-context gate did not pass")
    expected_cells = len(seeds) * len(budgets)
    if int(gate.get("checked_cells", -1)) != expected_cells:
        raise SystemExit("V2 matched-context gate cell count is incomplete")

    overall = metrics.loc[metrics["grouping"].eq("overall")].sort_values(
        ["model", "building_budget", "building_seed"]
    )
    expected_seed_count = len(seeds)
    if summary.empty or not summary["n_building_seeds"].eq(expected_seed_count).all():
        raise SystemExit("V2 cross-seed summary is incomplete")

    headline_models = {"tabpfn", "ensemble"}
    expected_pairs = {
        (model, budget) for model in headline_models for budget in budgets
    }
    headline = summary.loc[summary["model"].isin(headline_models)].sort_values(
        ["model", "building_budget"]
    )
    summary_pairs = set(
        zip(headline["model"], headline["building_budget"], strict=True)
    )
    if summary_pairs != expected_pairs or len(headline) != len(expected_pairs):
        raise SystemExit("V2 headline cross-seed summary is incomplete")
    raw_headline = overall.loc[overall["model"].isin(headline_models)].rename(
        columns={"building_budget": "K"}
    )
    for model, budget in sorted(expected_pairs):
        observed_seeds = set(
            raw_headline.loc[
                raw_headline["model"].eq(model) & raw_headline["K"].eq(budget),
                "building_seed",
            ].astype(int)
        )
        if observed_seeds != set(seeds):
            raise SystemExit(
                f"V2 raw results are incomplete for model={model} K={budget}"
            )
    prefix_view = prefix.loc[
        :,
        [
            "building_seed",
            "K",
            "sampling_attempt",
            "attempts_used",
            "constraint_pass",
            "reproducibility_digest",
        ],
    ]

    lines = [
        "# M5 building-count experiment V2",
        "",
        "**Status:** complete",
        "",
        "V2 uses constrained site-stratified random source-building ladders and "
        "compares TabPFN with the frozen four-model tree ensemble on byte-identical "
        "manifest-allocated rows. Tree early stopping is disabled.",
        "",
        "## Fixed protocol",
        "",
        f"- Building seeds: {', '.join(map(str, seeds))}.",
        f"- K budgets: {', '.join(map(str, budgets))}; strict nested prefixes.",
        "- Candidate buildings: even IDs only; odd IDs are the canonical holdout.",
        "- Source sampling: PCG64 site-stratified random sampling without replacement, "
        "with whole-ladder meter-feasibility rejection.",
        "- Row policy: fixed row_seed=42 manifest allocation; natural prevalence; "
        "no additional 50:50 redraw and no M3 anomaly duplication.",
        "- Features: 137 timestamp-merge features.",
        "- Tree contract: LightGBM 100, XGBoost 100, CatBoost 1000, HistGBT 100; "
        "fixed model_seed=42; no validation split and no early stopping.",
        "- TabPFN: n_estimators=8; no task-specific weight-update early stopping.",
        "- Evaluation: identical canonical odd-building natural-prevalence holdout.",
        f"- Matched-context gate: passed for all {expected_cells} seed/K pairs.",
        "",
        "## Sampling records",
        "",
    ]
    lines.extend(_table(prefix_view, list(prefix_view.columns)))
    lines.extend(
        [
            "",
            "Full building IDs, site counts, per-meter source-building counts and "
            f"digests: {args.audit_root / 'sampling_prefix_audit.csv'}.",
            "",
            "## Cross-seed overall results",
            "",
        ]
    )
    lines.extend(
        _table(
            headline,
            [
                "model",
                "building_budget",
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
    lines.extend(
        [
            "",
            "## Per-seed overall results",
            "",
        ]
    )
    lines.extend(
        _table(
            raw_headline,
            ["model", "K", "building_seed", "pr_auc", "roc_auc"],
        )
    )
    lines.extend(
        [
            "",
            "## Detailed outputs",
            "",
            f"- Raw metrics: {args.aggregate_root / 'metrics.csv'}.",
            f"- Cross-seed summary: "
            f"{args.aggregate_root / 'building_seed_summary.csv'}.",
            f"- ROC/PR curve points: {args.aggregate_root / 'curves.csv'}.",
            f"- Matched-context gate: {gate_path}.",
            f"- Sampling composition diagnostics: "
            f"{args.audit_root / 'composition_audit.csv'}.",
            "",
            "Per-meter and per-site rows remain in metrics.csv; raw per-seed "
            "results are preserved and are not replaced by the mean.",
            "",
        ]
    )
    _atomic_text(args.out, "\n".join(lines))
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
