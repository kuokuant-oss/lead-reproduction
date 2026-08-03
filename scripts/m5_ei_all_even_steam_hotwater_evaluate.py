"""Evaluate the completed all-even Steam+Hotwater model on canonical odd Steam."""
# ruff: noqa: E701, E702

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--formal-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not (a.formal_root / "FORMAL_COMPLETE.json").is_file():
        raise SystemExit("formal incomplete")
    with np.load(a.canonical, allow_pickle=False) as z:
        raw = z["raw_index"].astype("int64", copy=True)
        y = z["anomaly"].astype("int8", copy=True)
    meter = pd.read_csv(
        a.m3_root / "train.csv", usecols=["meter"], dtype={"meter": "int8"}
    )["meter"].to_numpy()[raw]
    steam = meter == 2
    if len(raw) != 10_137_155 or steam.sum() != 1_350_609 or y[steam].sum() != 48_888:
        raise AssertionError("Steam identity gate failed")
    with np.load(a.historical, allow_pickle=False) as h:
        if not np.array_equal(h["anomaly"], y):
            raise AssertionError("Historical positional label gate failed")
        historical = h["ensemble"].astype("float32", copy=True)[steam]
    with np.load(
        a.formal_root / "scores" / "all_even_steam_hotwater_natural" / "scores.npz",
        allow_pickle=False,
    ) as z:
        if not np.array_equal(z["raw_index"], raw[steam]):
            raise AssertionError("Steam score order gate failed")
        model = z["ensemble"].astype("float32", copy=True)
    if not np.isfinite(model).all():
        raise AssertionError("non-finite scores")

    def result(name, s):
        return {
            "model": name,
            "pr_auc": float(average_precision_score(y[steam], s)),
            "roc_auc": float(roc_auc_score(y[steam], s)),
        }

    rows = [
        result("Historical Full Tree", historical),
        result("all_even_steam_hotwater_natural", model),
    ]
    rows[1]["delta_pr_auc_vs_historical"] = rows[1]["pr_auc"] - rows[0]["pr_auc"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(
        json.dumps(
            {
                "steam_rows": int(steam.sum()),
                "steam_anomalies": int(y[steam].sum()),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
