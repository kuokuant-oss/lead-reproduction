"""Audit fixed-query factorial cells without fitting or rescoring models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_m5_hotwater_label_role_factorial import (
    metrics,
    query_frame,
    segment_clusters,
)
from lead import ROOT


FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FACTORIAL)
    return parser.parse_args()


def cell_values(
    root: Path, query: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str, int, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    score_map: dict[tuple[str, str, int, str], np.ndarray] = {}
    clusters = segment_clusters(query)
    query = query.copy()
    query["segment_cluster"] = clusters
    for result_path in sorted((root / "predictions").glob("*/*/*/*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with np.load(result_path.with_name("predictions.npz")) as payload:
            score = np.asarray(payload["score"], dtype="float64")
        key = (
            result["model"],
            result["scaler_arm"],
            int(result["context_seed"]),
            result["factorial_cell_id"],
        )
        score_map[key] = score
        overall = {
            f"score_q{int(q * 100):02d}": float(np.quantile(score, q))
            for q in (0.01, 0.10, 0.50, 0.90, 0.99)
        }
        normal = score[query["anomaly"].to_numpy() == 0]
        threshold = float(np.quantile(normal, 0.999))
        fp = int((normal >= threshold).sum())
        groups = {
            "hw01_positive": (query["meter"].to_numpy() == 3)
            & (query["meter_reading"].to_numpy() <= 1.0)
            & (query["anomaly"].to_numpy() == 1),
            "hw01_negative": (query["meter"].to_numpy() == 3)
            & (query["meter_reading"].to_numpy() <= 1.0)
            & (query["anomaly"].to_numpy() == 0),
            "steam_positive": (query["meter"].to_numpy() == 2)
            & (query["anomaly"].to_numpy() == 1),
            "global_normal": query["anomaly"].to_numpy() == 0,
            "global_positive": query["anomaly"].to_numpy() == 1,
        }
        for group, mask in groups.items():
            values = score[mask]
            rows.append(
                {
                    "model": key[0],
                    "scaler_arm": key[1],
                    "context_seed": key[2],
                    "factorial_cell_id": key[3],
                    "group": group,
                    "rows": int(mask.sum()),
                    "buildings": int(query.loc[mask, "building_id"].nunique()),
                    "segments": int(query.loc[mask, "segment_cluster"].nunique()),
                    "score_q01": float(np.quantile(values, 0.01)),
                    "score_q10": float(np.quantile(values, 0.10)),
                    "score_q50": float(np.quantile(values, 0.50)),
                    "score_q90": float(np.quantile(values, 0.90)),
                    "score_q99": float(np.quantile(values, 0.99)),
                    "normal_query_rows": len(normal),
                    "min_empirical_fpr_resolution": 1 / len(normal),
                    "threshold_q999_linear": threshold,
                    "false_positive_count": fp,
                    "empirical_fpr": fp / len(normal),
                    **overall,
                    **metrics(score, query),
                }
            )
    return pd.DataFrame(rows), score_map


def interaction_dominance(
    score_map: dict[tuple[str, str, int, str], np.ndarray], query: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cells = {
        False: {
            False: "hw_pos_excluded__hw_neg_excluded",
            True: "hw_pos_excluded__hw_neg_present",
        },
        True: {
            False: "hw_pos_present__hw_neg_excluded",
            True: "hw_pos_present__hw_neg_present",
        },
    }
    for model, arm, seed in sorted({key[:3] for key in score_map}):
        values = {
            (pos, neg): metrics(score_map[(model, arm, seed, cells[pos][neg])], query)
            for pos in (False, True)
            for neg in (False, True)
        }
        for metric in values[(False, False)]:
            cell_values = np.asarray(
                [
                    values[(False, False)][metric],
                    values[(False, True)][metric],
                    values[(True, False)][metric],
                    values[(True, True)][metric],
                ]
            )
            interaction = (
                cell_values[3] - cell_values[2] - cell_values[1] + cell_values[0]
            )
            term = np.asarray(
                [cell_values[3], -cell_values[2], -cell_values[1], cell_values[0]]
            )
            rows.append(
                {
                    "model": model,
                    "scaler_arm": arm,
                    "context_seed": seed,
                    "metric": metric,
                    "cell_00": cell_values[0],
                    "cell_01": cell_values[1],
                    "cell_10": cell_values[2],
                    "cell_11": cell_values[3],
                    "interaction": interaction,
                    "max_abs_term_fraction": float(
                        np.abs(term).max() / np.abs(term).sum()
                    )
                    if np.abs(term).sum()
                    else np.nan,
                    "ceiling_or_floor_cell": bool(
                        np.any(np.isclose(cell_values, 0))
                        or np.any(np.isclose(cell_values, 1))
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    query = query_frame()
    raw, scores = cell_values(args.root, query)
    dominance = interaction_dominance(scores, query)
    reports = args.root / "reports"
    raw.to_csv(reports / "factorial_raw_estimand_audit.csv", index=False)
    dominance.to_csv(reports / "factorial_interaction_dominance_audit.csv", index=False)
    print(
        f"wrote {len(raw)} raw estimand rows and {len(dominance)} interaction-dominance rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
