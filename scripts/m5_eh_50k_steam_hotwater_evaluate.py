"""Evaluate completed Steam/Hotwater-only 50k Tree scores on odd Steam."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CONDITIONS = ("steam_only", "steam_hw_all", "steam_hw_anomaly", "steam_hw_normal")
ROWS, STEAM_ROWS, STEAM_POSITIVES = 10_137_155, 1_350_609, 48_888


def steam_identity(
    m3: Path, canonical: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(canonical, allow_pickle=False) as a002:
        raw = a002["raw_index"].astype("int64", copy=True)
        label = a002["anomaly"].astype("int8", copy=True)
    meter = pd.read_csv(m3 / "train.csv", usecols=["meter"], dtype={"meter": "int8"})[
        "meter"
    ].to_numpy()[raw]
    steam = meter == 2
    if (
        len(raw) != ROWS
        or int(steam.sum()) != STEAM_ROWS
        or int(label[steam].sum()) != STEAM_POSITIVES
    ):
        raise AssertionError("A002 Steam identity gate failed")
    return raw, steam, label


def read_score(
    path: Path, raw: np.ndarray, *, historical: bool, label: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as value:
        if historical:
            if not np.array_equal(value["anomaly"], label):
                raise AssertionError(
                    "Historical A001 anomaly is not positionally equal to A002"
                )
        elif not np.array_equal(value["raw_index"], raw):
            raise AssertionError(f"{path}: A002 raw order mismatch")
        score = value["ensemble"].astype("float32", copy=True)
    if len(score) != ROWS or not np.isfinite(score).all():
        raise AssertionError(f"{path}: invalid scores")
    return score


def metrics(model: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    pred = score >= 0.5
    k = int(y.sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "model": model,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tp": tp,
        "fn": int((~pred & (y == 1)).sum()),
        "fp": fp,
        "tn": int((~pred & (y == 0)).sum()),
        "precision_at_0_5": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_at_0_5": float(tp / y.sum()),
        "top_48888_true_anomalies": int(y[np.argpartition(score, -k)[-k:]].sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--formal-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not (args.formal_root / "FORMAL_COMPLETE.json").is_file():
        raise SystemExit("incomplete formal score root")
    raw, steam, label = steam_identity(args.m3_root, args.canonical)
    rows = [
        metrics(
            "Historical Full Tree",
            label[steam],
            read_score(args.historical, raw, historical=True, label=label)[steam],
        )
    ]
    for condition in CONDITIONS:
        score = read_score(
            args.formal_root / "scores" / condition / "scores.npz",
            raw,
            historical=False,
            label=label,
        )[steam]
        rows.append(metrics(condition, label[steam], score))
    base = next(row["pr_auc"] for row in rows if row["model"] == "steam_only")
    historical = rows[0]["pr_auc"]
    for row in rows:
        row["delta_vs_steam_only"] = float(row["pr_auc"] - base)
        row["delta_vs_historical"] = float(row["pr_auc"] - historical)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "odd_steam_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "steam_rows": STEAM_ROWS,
                "steam_anomalies": STEAM_POSITIVES,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
