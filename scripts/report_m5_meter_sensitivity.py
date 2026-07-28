"""Sensitivity of the pooled TabPFN-vs-trees numbers to meter composition.

The pooled metric is a row-weighted average over four meter types that do not
behave alike, and electricity alone carries 6,035,071 of the 10,137,155 holdout
rows and 356,679 of the 637,397 anomalies. This script recomputes the pooled
figures under nine compositions so the dependence is visible rather than
assumed:

- ``all four``: the pooled number as reported everywhere else.
- ``drop <meter>``: leave-one-out, four rows.
- ``<meter> only``: each meter on its own, four rows.
- ``macro (4 meters)``: the unweighted mean of the four per-meter AUCs, which
  removes row-count weighting entirely.

Both models and both metrics are computed on every composition, at every label
budget N, plus the full-training-set tree for the same feature line.

Row identity is gated the same way as the breakdown report: the tree and TabPFN
artifacts must carry identical ``raw_index`` and labels per cell, and ``meter``
recovered positionally from the frozen M3 frame must reproduce the stored
``building_id`` and ``site_id``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from report_m5_matched_context_breakdown import (
    CONTEXTS,
    LINES,
    METER_NAMES,
    load_cell,
    load_full_tree,
    load_row_keys,
    tabpfn_path,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from lead import PROC, ROOT


def compositions(meter: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Row masks for every composition, in report order."""
    out: list[tuple[str, np.ndarray]] = [("all four", np.ones(len(meter), dtype=bool))]
    for code, name in sorted(METER_NAMES.items()):
        out.append((f"drop {name}", meter != code))
    for code, name in sorted(METER_NAMES.items()):
        out.append((f"{name} only", meter == code))
    return out


def scored(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    y_sub = y[mask]
    if not 0 < int(y_sub.sum()) < len(y_sub):
        return float("nan"), float("nan")
    s_sub = score[mask]
    return (
        float(roc_auc_score(y_sub, s_sub)),
        float(average_precision_score(y_sub, s_sub)),
    )


def macro(y: np.ndarray, score: np.ndarray, meter: np.ndarray) -> tuple[float, float]:
    """Unweighted mean of the four per-meter AUCs.

    Row weighting is what lets one meter decide the pooled figure; the macro
    average is the same numbers with that weighting removed. It is not a better
    metric, only a differently weighted one.
    """
    rocs, prs = [], []
    for code in sorted(METER_NAMES):
        roc, pr = scored(y, score, meter == code)
        rocs.append(roc)
        prs.append(pr)
    return float(np.mean(rocs)), float(np.mean(prs))


def rows_for(
    label: str,
    features: int,
    budget: str,
    model: str,
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    meter: np.ndarray,
) -> dict:
    roc, pr = (
        macro(y, score, meter)
        if label == "macro (4 meters)"
        else scored(y, score, mask)
    )
    return {
        "composition": label,
        "features": features,
        "N": budget,
        "model": model,
        "rows": int(mask.sum()),
        "anomalies": int(y[mask].sum()),
        "roc": roc,
        "pr": pr,
    }


def markdown_table(table: pd.DataFrame, features: int, model: str, metric: str) -> str:
    part = table[(table["features"] == features) & (table["model"] == model)]
    budgets = [b for b in part["N"].unique()]
    header = ["composition", "rows", "anomalies", *budgets]
    align = ["---", "---:", "---:"] + ["---:"] * len(budgets)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for label in part["composition"].unique():
        sub = part[part["composition"] == label]
        first = sub.iloc[0]
        cells = [label, f"{int(first['rows']):,}", f"{int(first['anomalies']):,}"]
        for budget in budgets:
            cell = sub[sub["N"] == budget]
            cells.append("—" if cell.empty else f"{cell.iloc[0][metric]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, table: pd.DataFrame, gates: dict) -> None:
    parts = [
        "# M5 meter-composition sensitivity",
        "",
        "## Setup",
        "",
        "- **Question.** How much of the pooled TabPFN-vs-trees figure depends on "
        "which meter types are in the pool.",
        "- **Compositions.** All four; leave-one-out (four); each meter alone "
        "(four); and a macro average that weights the four meter types equally "
        "instead of by row count.",
        "- **Models.** TabPFN and the tree ensemble, both capped at the same "
        "label budget N. `FULL` is the tree trained on the entire training set "
        "for that feature line.",
        f"- **Holdout.** {gates['holdout_rows']:,} rows, "
        f"{gates['holdout_anomalies']:,} anomalies, before any composition filter.",
        "- **Draw.** One frozen context draw, seed 42.",
        "",
        "Regenerate with `uv run python scripts/report_m5_meter_sensitivity.py`.",
        "",
        "## Gates",
        "",
        "- Tree and TabPFN artifacts must carry identical `raw_index` and labels "
        "per cell.",
        "- `meter` is recovered positionally from the frozen M3 frame; "
        "`building_id` and `site_id` read back at those positions must match the "
        "prediction artifacts.",
        "",
    ]
    for features in sorted(table["features"].unique()):
        for metric in ("pr", "roc"):
            for model, name in (("tabpfn", "TabPFN"), ("trees", "Trees")):
                parts.append(f"## {name}, {features} features, {metric.upper()}-AUC")
                parts.append("")
                parts.append(markdown_table(table, features, model, metric))
                parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", default="m5_meter_sensitivity")
    parser.add_argument(
        "--report",
        default=str(ROOT / "docs" / "reports" / "m5-meter-sensitivity.md"),
    )
    parser.add_argument(
        "--rebuild-report-only",
        action="store_true",
        help="re-emit the markdown from the CSV already on disk",
    )
    args = parser.parse_args()

    if args.rebuild_report_only:
        summary = json.loads(
            (PROC / f"{args.out_prefix}.json").read_text(encoding="utf-8")
        )
        cached = pd.read_csv(PROC / f"{args.out_prefix}.csv")
        cached["N"] = pd.Categorical(
            cached["N"],
            categories=[f"{c:,}" for c in CONTEXTS] + ["FULL"],
            ordered=True,
        )
        cached["composition"] = pd.Categorical(
            cached["composition"], categories=summary["compositions"], ordered=True
        )
        cached = cached.sort_values(["features", "model", "composition", "N"])
        write_markdown(Path(args.report), cached, summary["gates"])
        print(f"rebuilt {args.report}")
        return

    with np.load(tabpfn_path("17", 100_000)) as payload:
        raw_index = np.asarray(payload["raw_index"], dtype="int64")
        y = np.asarray(payload["anomaly"], dtype="int8")
        stored_site = np.asarray(payload["site_id"], dtype="int8")
        stored_building = np.asarray(payload["building_id"], dtype="int16")

    keys = load_row_keys(raw_index)
    if not np.array_equal(keys["building_id"].to_numpy("int16"), stored_building):
        raise SystemExit("building_id gate failed")
    if not np.array_equal(keys["site_id"].to_numpy("int8"), stored_site):
        raise SystemExit("site_id gate failed")
    meter = keys["meter"].to_numpy("int8")
    gates = {
        "holdout_rows": int(len(y)),
        "holdout_anomalies": int(y.sum()),
        "building_id_matches_artifact": True,
        "site_id_matches_artifact": True,
    }
    print(f"gates ok: {gates}", flush=True)

    masks = compositions(meter)
    masks.append(("macro (4 meters)", np.ones(len(meter), dtype=bool)))

    records: list[dict] = []
    for line in LINES:
        full = load_full_tree(line, y, stored_site)
        for label, mask in masks:
            records.append(
                rows_for(label, int(line), "FULL", "trees", y, full, mask, meter)
            )
        print(f"{line}f FULL done", flush=True)
        for context in CONTEXTS:
            cell = load_cell(line, context, tree_model="ensemble")
            if cell is None:
                print(f"{line}f/{context} missing, skipped", flush=True)
                continue
            budget = f"{context:,}"
            for model, score in (("tabpfn", cell.tabpfn), ("trees", cell.trees)):
                for label, mask in masks:
                    records.append(
                        rows_for(label, int(line), budget, model, y, score, mask, meter)
                    )
            print(f"{line}f/{context} done", flush=True)

    table = pd.DataFrame(records)
    # FULL is a tree-only column; it trails the capped budgets in the tables.
    order = [f"{c:,}" for c in CONTEXTS] + ["FULL"]
    table["N"] = pd.Categorical(table["N"], categories=order, ordered=True)
    # Rows read all-four, leave-one-out, single, macro -- not alphabetically.
    table["composition"] = pd.Categorical(
        table["composition"], categories=[label for label, _ in masks], ordered=True
    )
    table = table.sort_values(["features", "model", "composition", "N"])

    csv_path = PROC / f"{args.out_prefix}.csv"
    table.to_csv(csv_path, index=False)
    write_markdown(Path(args.report), table, gates)
    (PROC / f"{args.out_prefix}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "m5_meter_sensitivity",
                "gates": gates,
                "compositions": [label for label, _ in masks],
                "outputs": {"csv": str(csv_path), "report": args.report},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}\nwrote {args.report}")


if __name__ == "__main__":
    main()
