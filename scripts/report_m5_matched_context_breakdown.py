"""Per-site and per-meter breakdown of the matched-context TabPFN-vs-trees grid.

The context curve is reported pooled over all 10,137,155 holdout rows, and §3.0
of the run handoff already showed that the pooled number hides the result: for
trees at 17 features, 5k -> 100k moves the pooled PR-AUC by +0.0083, which is the
net of +0.18 at one site and -0.12 at another. This script produces the
disaggregated view for *both* model families at every context, so that any claim
about data volume, or about TabPFN beating trees, can be checked per site and per
meter type rather than on an aggregate that a few large easy sites dominate.

Both models score byte-identical rows, so the comparison is only meaningful if
the row order really is identical. That is asserted here, not assumed: the tree
artifact's ``validation_raw_index`` and label vector must match the TabPFN
artifact's element-wise before any metric is computed.

``meter`` is not carried in either prediction artifact. It is recovered
positionally from the frozen M3 ``train.csv`` via ``raw_index``, which is the
same positional identity ``load_m3_frame`` assigns. That recovery is gated too:
``building_id`` read back at those positions, and ``site_id`` joined from the
building metadata, must both match what the artifacts stored.

The reference column is the M3 tree ensemble trained on the *full* training set
-- the published 17-feature (pooled ROC 0.9663) and 137-feature (0.9918) lines
behind ``m3_feature_engineering_roc.png``. Every matched-N number is read against
the same feature line's full-data tree, which is what "how much does capping the
labelled set at N cost us" actually means.

Those two M3 artifacts carry a trap, gated against below. Their
``validation_raw_index`` is stored ascending while their ``anomaly`` and score
arrays are in the canonical scoring order, so the index column contradicts its
own file. The published AUCs are unaffected -- score and label are mutually
consistent, and scoring them against the canonical labels reproduces 0.9663 and
0.9918 exactly -- but keying rows by that index instead drops the 137-feature
line to ROC 0.4933, i.e. noise. So the scores are consumed positionally, against
the canonical row order, and never through their own index.

Output is one tidy CSV per grouping plus a JSON of provenance and gate results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from m6_site_names import site_names
from sklearn.metrics import average_precision_score, roc_auc_score

from lead import PROC, ROOT

M3_TRAIN = ROOT / "data" / "raw" / "m3" / "train.csv"
M3_METADATA = ROOT / "data" / "raw" / "m3" / "building_metadata.csv"

CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
LINES = ("17", "137")

# GEPIII meter encoding. Four types, and the frozen frame carries no others --
# checked on load rather than trusted.
METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

# The full-training-set tree line per feature width: the reference every capped-N
# cell is read against.
FULL_TREE = {
    "17": PROC / "m3_17_feature_ensemble_predictions.npz",
    "137": PROC / "m3_figure_predictions_50_50.npz",
}


def tabpfn_path(line: str, context_rows: int) -> Path:
    """Merged TabPFN predictions. 100k is the published artifact, named apart."""
    if context_rows == 100_000:
        return PROC / f"m5_tabpfn_{line}_full_test_n8_predictions.npz"
    return PROC / f"m5_tabpfn_{line}_full_test_context{context_rows}_n8_predictions.npz"


def tree_path(line: str, context_rows: int) -> Path:
    return PROC / f"m5_tree_ensemble_f{line}_context{context_rows}_predictions.npz"


@dataclass
class Cell:
    line: str
    context_rows: int
    tabpfn: np.ndarray
    trees: np.ndarray


def load_cell(line: str, context_rows: int, *, tree_model: str) -> Cell | None:
    """Load one grid cell, or None if either half has not been produced yet.

    The row-identity gate lives here because a cell that fails it must never
    reach the metric code: a silent misalignment would produce a plausible
    per-site table attributing one model's scores to the other's rows.
    """
    tab_file, tree_file = tabpfn_path(line, context_rows), tree_path(line, context_rows)
    if not tab_file.exists() or not tree_file.exists():
        return None

    with np.load(tab_file) as payload:
        raw_index = np.asarray(payload["raw_index"], dtype="int64")
        anomaly = np.asarray(payload["anomaly"], dtype="int8")
        tabpfn = np.asarray(payload["tabpfn"], dtype="float32")
    with np.load(tree_file) as payload:
        tree_index = np.asarray(payload["validation_raw_index"], dtype="int64")
        tree_anomaly = np.asarray(payload["anomaly"], dtype="int8")
        trees = np.asarray(payload[tree_model], dtype="float32")

    if not np.array_equal(raw_index, tree_index):
        raise SystemExit(
            f"{line}f/{context_rows}: tree and TabPFN row order differ; "
            "the matched-N comparison does not hold for this cell"
        )
    if not np.array_equal(anomaly, tree_anomaly):
        raise SystemExit(f"{line}f/{context_rows}: labels differ between artifacts")
    for name, values in (("tabpfn", tabpfn), (tree_model, trees)):
        if not np.isfinite(values).all():
            raise SystemExit(f"{line}f/{context_rows}: non-finite {name} scores")
    return Cell(line=line, context_rows=context_rows, tabpfn=tabpfn, trees=trees)


def load_full_tree(line: str, y: np.ndarray, stored_site: np.ndarray) -> np.ndarray:
    """The full-training-set tree scores for one feature line, canonical order.

    Gated on what the file can prove: its label vector must equal the canonical
    one element-wise, and -- where the artifact carries site_id, which only the
    17-feature one does -- that must match too. Its own ``validation_raw_index``
    is deliberately ignored; see the module docstring for why trusting it turns
    ROC 0.9918 into 0.4933.
    """
    path = FULL_TREE[line]
    with np.load(path) as payload:
        anomaly = np.asarray(payload["anomaly"], dtype="int8")
        scores = np.asarray(payload["ensemble"], dtype="float32")
        site = (
            np.asarray(payload["site_id"], dtype="int8")
            if "site_id" in payload.files
            else None
        )
    if not np.array_equal(anomaly, y):
        raise SystemExit(
            f"{path.name}: labels differ from the canonical holdout, so its rows "
            "are not in canonical order and cannot be used positionally"
        )
    if site is not None and not np.array_equal(site, stored_site):
        raise SystemExit(f"{path.name}: site_id disagrees with the canonical order")
    if not np.isfinite(scores).all():
        raise SystemExit(f"{path.name}: non-finite scores")
    return scores


def load_row_keys(raw_index: np.ndarray) -> pd.DataFrame:
    """Recover meter, building and site for the holdout rows.

    ``raw_index`` is positional into the frozen M3 frame, whose row order is the
    CSV's: the loader only adds columns and left-merges metadata. Reading the
    two needed columns costs ~60 MB against materialising the frame.
    """
    frame = pd.read_csv(
        M3_TRAIN,
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    if raw_index.max() >= len(frame):
        raise SystemExit(
            f"raw_index max {raw_index.max()} exceeds the M3 frame ({len(frame)} rows)"
        )
    rows = frame.iloc[raw_index].reset_index(drop=True)
    del frame

    meta = pd.read_csv(M3_METADATA, usecols=["building_id", "site_id"])
    rows = rows.merge(meta, on="building_id", how="left")
    if rows["site_id"].isna().any():
        raise SystemExit("some holdout buildings are absent from building metadata")
    rows["site_id"] = rows["site_id"].astype("int8")

    unknown = sorted(set(rows["meter"].unique().tolist()) - set(METER_NAMES))
    if unknown:
        raise SystemExit(f"unexpected meter codes in the holdout: {unknown}")
    return rows


def group_metrics(
    y: np.ndarray, scores: dict[str, np.ndarray], keys: np.ndarray, labels: dict
) -> list[dict]:
    """ROC-AUC and PR-AUC for every model within every level of ``keys``.

    A group with one class present has no AUC; it is reported as NaN with its
    support rather than dropped, because "unsolvable here" is itself a finding
    (site 11 sits at PR 0.018 in the pooled table).
    """
    out: list[dict] = []
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    bounds = np.searchsorted(sorted_keys, np.unique(sorted_keys), side="left").tolist()
    bounds.append(len(sorted_keys))

    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[start:end]
        key = int(sorted_keys[start])
        y_grp = y[idx]
        positives = int(y_grp.sum())
        row = {
            "group": key,
            "group_label": labels.get(key, str(key)),
            "rows": int(len(idx)),
            "anomalies": positives,
            "anomaly_rate": float(positives / len(idx)),
        }
        both_classes = 0 < positives < len(idx)
        for name, values in scores.items():
            if both_classes:
                row[f"{name}_roc"] = float(roc_auc_score(y_grp, values[idx]))
                row[f"{name}_pr"] = float(average_precision_score(y_grp, values[idx]))
            else:
                row[f"{name}_roc"] = float("nan")
                row[f"{name}_pr"] = float("nan")
        out.append(row)
    return out


def paired_bootstrap(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """sd of the paired (a - b) AUC differences under row resampling.

    This is the *sampling* noise of the estimate on this holdout: how much the
    difference would move if the same two models were scored on another draw of
    rows from the same population. It is not the context-draw noise the handoff
    reports (that needs refitting under new seeds), and the two do not substitute
    for each other. It is measured because groups like site 4 and site 11 carry
    197 anomalies apiece, where a PR-AUC difference is dominated by which handful
    of positives landed in the group.

    Resampling is paired -- both models see the same resampled rows -- so shared
    row difficulty cancels, which is the same reason the matched-N design pairs
    them in the first place.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    roc = np.empty(draws, dtype="float64")
    pr = np.empty(draws, dtype="float64")
    kept = 0
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        positives = int(y_boot.sum())
        if not 0 < positives < n:
            continue
        a_boot, b_boot = a[idx], b[idx]
        roc[kept] = roc_auc_score(y_boot, a_boot) - roc_auc_score(y_boot, b_boot)
        pr[kept] = average_precision_score(y_boot, a_boot) - average_precision_score(
            y_boot, b_boot
        )
        kept += 1
    if kept < 2:
        return float("nan"), float("nan")
    return float(roc[:kept].std(ddof=1)), float(pr[:kept].std(ddof=1))


def add_bootstrap(
    table: pd.DataFrame,
    cells: list[Cell],
    y: np.ndarray,
    keys: np.ndarray,
    *,
    max_anomalies: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Attach sampling sd of the diffs for the groups thin enough to need it.

    Restricted by anomaly count: on a group with 100k positives the sampling sd
    is far below any effect being discussed, and paying for it on all sixteen
    sites would cost more than the whole rest of this script.
    """
    table["roc_diff_sd"] = float("nan")
    table["pr_diff_sd"] = float("nan")
    thin = sorted(
        {
            int(g)
            for g in table.loc[table["anomalies"] <= max_anomalies, "group"].unique()
        }
    )
    if not thin:
        return table
    print(f"bootstrapping {len(thin)} thin groups x {len(cells)} cells", flush=True)
    for group in thin:
        mask = keys == group
        y_grp = y[mask]
        for cell in cells:
            roc_sd, pr_sd = paired_bootstrap(
                y_grp,
                cell.tabpfn[mask],
                cell.trees[mask],
                draws=draws,
                seed=seed + group,
            )
            row = (
                (table["group"] == group)
                & (table["features"] == int(cell.line))
                & (table["context_rows"] == cell.context_rows)
            )
            table.loc[row, "roc_diff_sd"] = roc_sd
            table.loc[row, "pr_diff_sd"] = pr_sd
        print(f"  group {group} done", flush=True)
    return table


def build_table(
    cells: list[Cell],
    y: np.ndarray,
    keys: np.ndarray,
    labels: dict,
    full_tree: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One tidy frame: a row per (line, context, group).

    The reference is per feature line, not shared: comparing a 17-feature capped
    cell against the 137-feature full-data tree would measure the feature set,
    not the label budget.
    """
    reference = {
        line: pd.DataFrame(group_metrics(y, {"fulltree": scores}, keys, labels))
        for line, scores in full_tree.items()
    }

    frames = []
    for cell in cells:
        scores = {"tabpfn": cell.tabpfn, "trees": cell.trees}
        part = pd.DataFrame(group_metrics(y, scores, keys, labels))
        part.insert(0, "context_rows", cell.context_rows)
        part.insert(0, "features", int(cell.line))
        part["roc_diff"] = part["tabpfn_roc"] - part["trees_roc"]
        part["pr_diff"] = part["tabpfn_pr"] - part["trees_pr"]
        part = part.merge(
            reference[cell.line][["group", "fulltree_roc", "fulltree_pr"]],
            on="group",
            how="left",
        )
        frames.append(part)

    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["features", "context_rows", "group"]).reset_index(
        drop=True
    )


def markdown_block(table: pd.DataFrame, features: int, metric: str, model: str) -> str:
    """One model's table: a row per group, a column per label budget N.

    One model per table. Putting both in a shared cell packs two independent
    curves into one row and neither can be read across; the difference between
    them is a third thing again and is not printed here at all.

    ``FULL`` appears only in the tree tables. It is the tree trained on the whole
    training set, so it extends the tree's own row -- the same model at an
    uncapped N. In a TabPFN table it would be another family's number sitting in
    a column of TabPFN's, which is what it looked like when it was there.
    """
    part = table[table["features"] == features]
    contexts = sorted(part["context_rows"].unique())
    with_full = model == "trees"
    header = ["group", "rows", "anomalies"] + [f"{c:,}" for c in contexts]
    align = ["---", "---:", "---:"] + ["---:"] * len(contexts)
    if with_full:
        # Last column: N rises left to right, and FULL is the end of that axis.
        header.append("FULL")
        align.append("---:")
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]

    for group in sorted(part["group"].unique()):
        rows = part[part["group"] == group]
        first = rows.iloc[0]
        cells = [
            str(first["group_label"]),
            f"{int(first['rows']):,}",
            f"{int(first['anomalies']):,}",
        ]
        for context in contexts:
            cell = rows[rows["context_rows"] == context]
            cells.append(
                "—" if cell.empty else f"{cell.iloc[0][f'{model}_{metric}']:.4f}"
            )
        if with_full:
            cells.append(f"{first[f'fulltree_{metric}']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    by_site: pd.DataFrame,
    by_meter: pd.DataFrame,
    summary: dict,
) -> None:
    """Emit the whole breakdown as one reviewable document."""
    ref = summary["full_tree_reference"]
    pending = ", ".join(summary["pending"]) or "none"
    parts = [
        "# M5 matched-context breakdown: per site and per meter",
        "",
        "TabPFN and tree ensembles trained on byte-identical rows at each label",
        "budget N, scored on the full 10,137,155-row holdout, then disaggregated.",
        "The pooled numbers in the context-curve handoff are an aggregate over",
        "sixteen sites and four meter types whose behaviour differs in sign, so",
        "the pooled figure cannot support a per-site or per-meter claim.",
        "",
        "One table per model. Columns are the label budget N, which both models",
        "are capped at identically.",
        "",
        "The tree tables carry one extra column on the right, `FULL`: the same",
        "tree ensemble",
        "trained on the **entire** training set for that feature line -- pooled",
        f"ROC {ref['17']['pooled_roc']:.4f} at 17 features and",
        f"{ref['137']['pooled_roc']:.4f} at 137, the two lines in",
        "`m3_feature_engineering_roc.png`. It is the tree's own uncapped end point,",
        "which is what makes it the reference for what capping N costs. There is no",
        "TabPFN equivalent -- TabPFN was never run on the full training set, and",
        "could not be: its context is the labelled set it is given.",
        "",
        "Regenerate with `uv run python scripts/report_m5_matched_context_breakdown.py`.",
        "",
        "## Provenance and gates",
        "",
        f"- Holdout: {summary['gates']['holdout_rows']:,} rows, "
        f"{summary['gates']['holdout_anomalies']:,} anomalies.",
        "- Tree and TabPFN artifacts are asserted to carry identical `raw_index`",
        "  and identical labels per cell; a cell failing that never reaches the",
        "  metric code, because the matched-N comparison would not hold.",
        "- `meter` is recovered positionally from the frozen M3 frame and gated:",
        "  `building_id` and `site_id` read back at those positions match what the",
        "  prediction artifacts stored.",
        "- **Trap.** The two full-data M3 artifacts store `validation_raw_index`",
        "  ascending while their score and label arrays are in canonical scoring",
        "  order, so that column contradicts its own file. The published AUCs are",
        "  unaffected -- score and label are mutually consistent, and scoring",
        "  against the canonical labels reproduces",
        f"  {ref['17']['pooled_roc']:.4f} and {ref['137']['pooled_roc']:.4f}",
        "  exactly -- but keying rows by that index instead collapses the",
        "  137-feature line to ROC 0.4933, i.e. noise. They are consumed",
        "  positionally here and never through their own index.",
        f"- Cells not yet produced: {pending}.",
        "- Every number comes from the single frozen context draw (seed 42), so",
        "  the *context* noise is unmeasured per group; the handoff's sd figures",
        "  (ROC 0.0003 at 137 features, 0.0004 at 17) are pooled, and a per-group",
        "  sd is necessarily larger. The *sampling* noise is measured, for the",
        "  groups thin enough for it to bite -- see the last section.",
        "",
    ]

    for kind, table in (("meter", by_meter), ("site", by_site)):
        parts.append(f"## By {kind}")
        parts.append("")
        for features in sorted(table["features"].unique()):
            for metric in ("pr", "roc"):
                for model, name in (("tabpfn", "TabPFN"), ("trees", "Trees")):
                    parts.append(
                        f"### {name} by {kind}: {features} features, "
                        f"{metric.upper()}-AUC"
                    )
                    parts.append("")
                    parts.append(markdown_block(table, features, metric, model))
                    parts.append("")

    thin = by_site.dropna(subset=["pr_diff_sd"]) if "pr_diff_sd" in by_site else None
    if thin is not None and not thin.empty:
        parts.extend(
            [
                "## Sampling noise on the thin sites",
                "",
                "Paired bootstrap over rows, 200 draws, for every site carrying at",
                "most 3,000 anomalies -- the ones where a PR-AUC gap could plausibly",
                "be decided by which handful of positives landed in the group. Both",
                "models are resampled together, so shared row difficulty cancels.",
                "",
                "This measures how much the difference would move on another draw of",
                "*rows*. It does not cover the context draw, which is frozen at seed",
                "42 everywhere in this grid and remains unmeasured per group.",
                "",
                "The measurement mostly refutes the caution it was run to justify:",
                "at 197 anomalies the sd of the PR difference is 0.008-0.037, so gaps",
                "like site 4 at 17f/5k (+0.1869) and site 11 at 137f/10k (-0.3210)",
                "clear it by 7x and 13x. Read them as real, subject to the context",
                "caveat above.",
                "",
                "| features | N | site | anomalies | TabPFN PR | Trees PR | PR diff | sd | ROC diff | sd |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in thin.iterrows():
            parts.append(
                f"| {int(row['features'])} | {int(row['context_rows']):,} | "
                f"{row['group_label']} | {int(row['anomalies']):,} | "
                f"{row['tabpfn_pr']:.4f} | {row['trees_pr']:.4f} | "
                f"{row['pr_diff']:+.4f} | {row['pr_diff_sd']:.4f} | "
                f"{row['roc_diff']:+.4f} | {row['roc_diff_sd']:.4f} |"
            )
        parts.append("")

    path.write_text("\n".join(parts), encoding="utf-8")


def print_pivot(table: pd.DataFrame, metric: str, kind: str) -> None:
    """The context curve per group, one block per feature line."""
    for features in sorted(table["features"].unique()):
        part = table[table["features"] == features]
        wide = part.pivot(index="group_label", columns="context_rows", values=metric)
        support = (
            part.groupby("group_label")[["rows", "anomalies"]].first().astype("int64")
        )
        wide = support.join(wide)
        print(f"\n=== {kind}: {metric}  ({features} features) ===")
        print(wide.to_string(float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", nargs="+", default=list(LINES), choices=list(LINES))
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXTS))
    parser.add_argument(
        "--tree-model",
        default="ensemble",
        help="which tree score to use; 'ensemble' is what the pooled tables report",
    )
    parser.add_argument("--out-prefix", default="m5_matched_context_breakdown")
    parser.add_argument(
        "--bootstrap-max-anomalies",
        type=int,
        default=0,
        help="paired-bootstrap the diffs for groups with at most this many "
        "anomalies (0 disables); these are the groups where a PR-AUC difference "
        "is decided by a handful of positives",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "docs" / "reports" / "m5-matched-context-breakdown.md"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-report-only",
        action="store_true",
        help="re-emit the markdown from the CSVs and JSON already on disk; for "
        "editing the prose without paying for the bootstrap again",
    )
    args = parser.parse_args()

    if args.rebuild_report_only:
        summary = json.loads(
            (PROC / f"{args.out_prefix}.json").read_text(encoding="utf-8")
        )
        write_markdown(
            Path(args.report),
            pd.read_csv(PROC / f"{args.out_prefix}_by_site.csv"),
            pd.read_csv(PROC / f"{args.out_prefix}_by_meter.csv"),
            summary,
        )
        print(f"rebuilt {args.report}")
        return

    cells: list[Cell] = []
    pending: list[str] = []
    for line in args.lines:
        for context_rows in args.contexts:
            cell = load_cell(line, context_rows, tree_model=args.tree_model)
            if cell is None:
                pending.append(f"{line}f/{context_rows}")
                continue
            cells.append(cell)
            print(f"loaded {line}f / {context_rows}", flush=True)
    if not cells:
        raise SystemExit("no complete cells found")
    if pending:
        print(f"pending (skipped): {', '.join(pending)}", flush=True)

    # Every cell shares the canonical holdout order, so row keys are read once
    # off the first cell's artifact and reused.
    with np.load(tabpfn_path(cells[0].line, cells[0].context_rows)) as payload:
        raw_index = np.asarray(payload["raw_index"], dtype="int64")
        y = np.asarray(payload["anomaly"], dtype="int8")
        stored_site = np.asarray(payload["site_id"], dtype="int8")
        stored_building = np.asarray(payload["building_id"], dtype="int16")

    print("recovering meter/site keys from the frozen M3 frame", flush=True)
    rows = load_row_keys(raw_index)

    gates = {
        "building_id_matches_artifact": bool(
            np.array_equal(rows["building_id"].to_numpy("int16"), stored_building)
        ),
        "site_id_matches_artifact": bool(
            np.array_equal(rows["site_id"].to_numpy("int8"), stored_site)
        ),
        "holdout_rows": int(len(raw_index)),
        "holdout_anomalies": int(y.sum()),
    }
    if not (
        gates["building_id_matches_artifact"] and gates["site_id_matches_artifact"]
    ):
        raise SystemExit(
            "positional recovery of meter failed its identity gate: building_id or "
            "site_id read back from the M3 frame disagrees with the artifacts"
        )
    print(f"gates ok: {gates}", flush=True)

    full_tree = {
        line: load_full_tree(line, y, stored_site)
        for line in sorted({c.line for c in cells})
    }
    print(f"full-data tree reference loaded for lines {sorted(full_tree)}", flush=True)

    site_labels = {k: f"site {k} ({v})" for k, v in site_names().items()}

    site_keys = rows["site_id"].to_numpy("int8").astype("int64")
    meter_keys = rows["meter"].to_numpy("int8").astype("int64")
    by_site = build_table(cells, y, site_keys, site_labels, full_tree)
    by_meter = build_table(cells, y, meter_keys, METER_NAMES, full_tree)

    if args.bootstrap_max_anomalies > 0:
        by_site = add_bootstrap(
            by_site,
            cells,
            y,
            site_keys,
            max_anomalies=args.bootstrap_max_anomalies,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
        by_meter = add_bootstrap(
            by_meter,
            cells,
            y,
            meter_keys,
            max_anomalies=args.bootstrap_max_anomalies,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )

    site_csv = PROC / f"{args.out_prefix}_by_site.csv"
    meter_csv = PROC / f"{args.out_prefix}_by_meter.csv"
    by_site.to_csv(site_csv, index=False)
    by_meter.to_csv(meter_csv, index=False)

    summary = {
        "schema_version": 1,
        "experiment": "m5_matched_context_breakdown",
        "tree_model": args.tree_model,
        "full_tree_reference": {
            line: {
                "path": str(FULL_TREE[line]),
                "pooled_roc": float(roc_auc_score(y, scores)),
                "pooled_pr": float(average_precision_score(y, scores)),
                "note": "consumed positionally; its own validation_raw_index is "
                "stored in a different order and must not be used as a row key",
            }
            for line, scores in full_tree.items()
        },
        "cells": [
            {"features": int(c.line), "context_rows": c.context_rows} for c in cells
        ],
        "pending": pending,
        "gates": gates,
        "bootstrap": {
            "max_anomalies": args.bootstrap_max_anomalies,
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "measures": "sampling sd of the paired diff on this holdout, not "
            "context-draw noise",
        },
        "outputs": {"by_site": str(site_csv), "by_meter": str(meter_csv)},
    }
    json_path = PROC / f"{args.out_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    write_markdown(report_path, by_site, by_meter, summary)
    summary["outputs"]["report"] = str(report_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for metric in ("pr_diff", "roc_diff"):
        print_pivot(by_meter, metric, "by meter")
    print_pivot(by_meter, "tabpfn_pr", "by meter")
    print_pivot(by_meter, "trees_pr", "by meter")
    for metric in ("pr_diff", "roc_diff"):
        print_pivot(by_site, metric, "by site")

    print(f"\nwrote {site_csv}\nwrote {meter_csv}\nwrote {json_path}")


if __name__ == "__main__":
    main()
