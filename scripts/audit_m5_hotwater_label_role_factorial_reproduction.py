"""Gate exact-design recovery against the original 352-row factorial scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_m5_hotwater_label_role_factorial import factor_effect, metrics, query_frame
from lead import ROOT


ROOT_OUT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
TOLERANCES = {
    "score_mae_max": 0.005,
    "spearman_min": 0.999,
    "primary_estimand_abs_delta_max": 0.02,
    "factorial_effect_abs_delta_max": 0.04,
    "direction_reference_abs_min": 0.01,
}
PRIMARY = ("hw01_within_rank_gap", "hw01_pair_auc", "steam_pos_vs_hw_neg_auc")


def cell_key(model: str, arm: str, seed: int, cell: str) -> tuple[str, str, int, str]:
    return model, arm, seed, cell


def original_scores(root: Path) -> dict[tuple[str, str, int, str], np.ndarray]:
    result: dict[tuple[str, str, int, str], np.ndarray] = {}
    for meta in root.glob("predictions/*/seed*/*/*/result.json"):
        item = json.loads(meta.read_text(encoding="utf-8"))
        with np.load(meta.with_name("predictions.npz")) as payload:
            result[
                cell_key(
                    item["model"],
                    item["scaler_arm"],
                    int(item["context_seed"]),
                    item["factorial_cell_id"],
                )
            ] = np.asarray(payload["score"], dtype="float64")
    return result


def recovered_scores(recovery: Path) -> dict[tuple[str, str, int, str], np.ndarray]:
    result: dict[tuple[str, str, int, str], np.ndarray] = {}
    for meta in recovery.glob("states/*/seed*/*/*/screening_result.json"):
        item = json.loads(meta.read_text(encoding="utf-8"))
        with np.load(meta.with_name("screening_predictions.npz")) as payload:
            result[
                cell_key(
                    item["model"], item["scaler_arm"], int(item["seed"]), item["cell"]
                )
            ] = np.asarray(payload["score"], dtype="float64")
    return result


def flags_for_effects(original: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    joined = original.merge(
        recovered,
        on=["model", "scaler_arm", "context_seed", "metric", "effect"],
        suffixes=("_original", "_recovered"),
        validate="one_to_one",
    )
    joined["abs_delta"] = (
        joined["estimate_original"] - joined["estimate_recovered"]
    ).abs()
    joined["direction_required"] = (
        joined["estimate_original"].abs() >= TOLERANCES["direction_reference_abs_min"]
    )
    joined["direction_match"] = (~joined["direction_required"]) | (
        np.sign(joined["estimate_original"]) == np.sign(joined["estimate_recovered"])
    )
    joined["within_tolerance"] = (
        joined["abs_delta"] <= TOLERANCES["factorial_effect_abs_delta_max"]
    )
    return joined


def effect_rows(
    score_map: dict[tuple[str, str, int, str], np.ndarray], query: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for model, arm, seed in sorted({key[:3] for key in score_map}):
        for metric in PRIMARY:
            values = {}
            for pos in (False, True):
                for neg in (False, True):
                    cell = f"hw_pos_{'present' if pos else 'excluded'}__hw_neg_{'present' if neg else 'excluded'}"
                    values[pos, neg] = metrics(
                        score_map[cell_key(model, arm, seed, cell)], query
                    )[metric]
            for effect, estimate in factor_effect(values).items():
                rows.append(
                    {
                        "model": model,
                        "scaler_arm": arm,
                        "context_seed": seed,
                        "metric": metric,
                        "effect": effect,
                        "estimate": estimate,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    query = query_frame()
    original, recovered = (
        original_scores(ROOT_OUT),
        recovered_scores(ROOT_OUT / "recovery"),
    )
    if len(original) != 48 or len(recovered) != 48 or set(original) != set(recovered):
        raise RuntimeError(
            f"require matching 48-cell score sets; original={len(original)}, recovered={len(recovered)}"
        )
    cell_rows = []
    for key in sorted(original):
        before, after = original[key], recovered[key]
        rank_corr = pd.Series(before).corr(pd.Series(after), method="spearman")
        before_m, after_m = metrics(before, query), metrics(after, query)
        deltas = {metric: abs(before_m[metric] - after_m[metric]) for metric in PRIMARY}
        cell_rows.append(
            {
                "model": key[0],
                "scaler_arm": key[1],
                "context_seed": key[2],
                "factorial_cell_id": key[3],
                "score_mae": float(np.mean(np.abs(before - after))),
                "score_max_abs": float(np.max(np.abs(before - after))),
                "spearman": float(rank_corr),
                **{f"{metric}_abs_delta": value for metric, value in deltas.items()},
                "score_pass": bool(
                    np.mean(np.abs(before - after)) <= TOLERANCES["score_mae_max"]
                    and rank_corr >= TOLERANCES["spearman_min"]
                ),
                "primary_pass": bool(
                    max(deltas.values()) <= TOLERANCES["primary_estimand_abs_delta_max"]
                ),
            }
        )
    cells = pd.DataFrame(cell_rows)
    effects = flags_for_effects(
        effect_rows(original, query), effect_rows(recovered, query)
    )
    passed = bool(
        cells["score_pass"].all()
        and cells["primary_pass"].all()
        and effects["within_tolerance"].all()
        and effects["direction_match"].all()
    )
    report_dir = ROOT_OUT / "recovery" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(report_dir / "reproduction_cell_comparison.csv", index=False)
    effects.to_csv(report_dir / "reproduction_effect_comparison.csv", index=False)
    (ROOT_OUT / "recovery" / "reproduction_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "tolerances_predeclared": TOLERANCES,
                "cell_rows": len(cells),
                "effect_rows": len(effects),
                "failure_counts": {
                    "cell_score": int((~cells["score_pass"]).sum()),
                    "cell_primary": int((~cells["primary_pass"]).sum()),
                    "effect_magnitude": int((~effects["within_tolerance"]).sum()),
                    "effect_direction": int((~effects["direction_match"]).sum()),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"reproduction gate {'PASSED' if passed else 'FAILED'}: {len(cells)} cells, {len(effects)} effects",
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
