"""Evaluate only completed strict-50k Tree score artifacts on odd Steam."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CELLS = ("00", "01", "10", "11")
ROWS, STEAM_ROWS, STEAM_POSITIVE = 10_137_155, 1_350_609, 48_888


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def metric(label: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    predicted = score >= 0.5
    k = int(y.sum())
    return {
        "model": label,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tp": int((predicted & (y == 1)).sum()),
        "fn": int((~predicted & (y == 1)).sum()),
        "fp": int((predicted & (y == 0)).sum()),
        "tn": int((~predicted & (y == 0)).sum()),
        "precision_at_0_5": float((predicted & (y == 1)).sum() / predicted.sum())
        if predicted.any()
        else 0.0,
        "recall_at_0_5": float((predicted & (y == 1)).sum() / y.sum()),
        "top_48888_true_anomalies": int(y[np.argpartition(score, -k)[-k:]].sum()),
    }


def load_odd_steam(
    m3: Path, canonical: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(
        m3 / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    with np.load(canonical, allow_pickle=False) as a002:
        raw, labels = (
            a002["raw_index"].astype("int64", copy=True),
            a002["anomaly"].astype("int8", copy=True),
        )
    steam = train["meter"].to_numpy()[raw] == 2
    if (
        len(raw) != ROWS
        or len(np.unique(raw)) != ROWS
        or int(steam.sum()) != STEAM_ROWS
        or int(labels[steam].sum()) != STEAM_POSITIVE
    ):
        raise AssertionError("canonical odd Steam identity gate failed")
    return raw, steam, labels


def score_from(
    path: Path,
    raw: np.ndarray,
    field: str,
    *,
    positional_historical: bool = False,
    canonical_labels: np.ndarray | None = None,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key = (
            "validation_raw_index"
            if "validation_raw_index" in data.files
            else "raw_index"
        )
        if (
            not positional_historical and not np.array_equal(data[key], raw)
        ) or field not in data.files:
            raise AssertionError(f"{path}: canonical order or score field gate failed")
        if positional_historical and (
            canonical_labels is None
            or not np.array_equal(data["anomaly"], canonical_labels)
        ):
            raise AssertionError(
                "Historical A001 anomaly is not positionally identical to A002"
            )
        score = data[field].astype("float32", copy=True)
    if len(score) != ROWS or not np.isfinite(score).all():
        raise AssertionError(f"{path}: score integrity gate failed")
    return score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--formal-root", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    complete = a.formal_root / "FORMAL_COMPLETE.json"
    if not complete.is_file():
        raise SystemExit("formal score set incomplete")
    raw, steam, labels = load_odd_steam(a.m3_root, a.canonical)
    y = labels[steam]
    rows = []
    historical = score_from(
        a.historical,
        raw,
        "ensemble",
        positional_historical=True,
        canonical_labels=labels,
    )[steam]
    rows.append(metric("Historical Full Tree", y, historical))
    for cell in CELLS:
        score = score_from(
            a.formal_root / "scores" / f"cell{cell}" / "scores.npz", raw, "ensemble"
        )[steam]
        rows.append(metric(f"cell{cell}", y, score))
    baseline = next(row["pr_auc"] for row in rows if row["model"] == "cell11")
    old = rows[0]["pr_auc"]
    for row in rows:
        row["delta_vs_steam_only_cell11"] = float(row["pr_auc"] - baseline)
        row["delta_vs_historical"] = float(row["pr_auc"] - old)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "odd_steam_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        a.out / "summary.json",
        {"rows": rows, "steam_rows": STEAM_ROWS, "steam_anomalies": STEAM_POSITIVE},
    )
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
