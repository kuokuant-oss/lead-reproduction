"""Endpoint definitions for E4, identical in formula to the E3 runner.

E3 computed its endpoints over the whole 352-row query. The clustered bootstrap
needs the same quantities over an arbitrary resampled row multiset, so the
formulas live here in a row-subsettable form. `tests/test_m5_e4_protocol.py`
asserts these reproduce `m5_e3_runner.endpoints` exactly on the full query; if
that test fails, the E4 margin is no longer the E3 margin and the protocol's
frozen definition source is wrong.

A draw is invalid, never imputed and never substituted, when a stratum a metric
needs is empty or the result is non-finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METER = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}

INVALID = float("nan")

# The two endpoints that carry the formal Path A steam claim.
PRIMARY = (
    "steam_positive_vs_hotwater_negative_pairwise_auc",
    "steam_positive_minus_hotwater_negative_score_margin",
)


def _pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return INVALID
    y = np.concatenate([np.ones(pos.size, "int8"), np.zeros(neg.size, "int8")])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def _margin(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return INVALID
    return float(pos.mean() - neg.mean())


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else INVALID


def endpoints(
    score: np.ndarray, meter: np.ndarray, anomaly: np.ndarray
) -> dict[str, float]:
    """Every readout for one scored pass over one row multiset.

    Score-based and rank-based quantities are computed and reported separately
    and are never pooled.
    """
    score = np.asarray(score, dtype="float64")

    def group(m: str, a: int) -> np.ndarray:
        return score[(meter == METER[m]) & (anomaly == a)]

    steam_pos = group("steam", 1)
    hw_neg = group("hotwater", 0)
    cw_pos = group("chilledwater", 1)
    cw_neg = group("chilledwater", 0)

    global_rank = pd.Series(score).rank(method="average", pct=True).to_numpy()
    cw_mask = meter == METER["chilledwater"]
    st_mask = meter == METER["steam"]
    cw_rank = pd.Series(score[cw_mask]).rank(method="average", pct=True).to_numpy()
    st_rank = pd.Series(score[st_mask]).rank(method="average", pct=True).to_numpy()

    if cw_pos.size and cw_neg.size:
        y_cw = np.concatenate(
            [np.ones(cw_pos.size, "int8"), np.zeros(cw_neg.size, "int8")]
        )
        s_cw = np.concatenate([cw_pos, cw_neg])
        cw_pr = float(average_precision_score(y_cw, s_cw))
        cw_roc = float(roc_auc_score(y_cw, s_cw))
    else:
        cw_pr = cw_roc = INVALID

    return {
        # --- steam principal ---
        "steam_positive_vs_hotwater_negative_pairwise_auc": _pairwise_auc(
            steam_pos, hw_neg
        ),
        "steam_positive_minus_hotwater_negative_score_margin": _margin(
            steam_pos, hw_neg
        ),
        # --- steam rank-based, reported separately ---
        "steam_positive_global_rank": _mean(global_rank[st_mask & (anomaly == 1)]),
        "steam_positive_within_meter_rank": _mean(
            st_rank[anomaly[st_mask] == 1] if st_mask.any() else np.empty(0)
        ),
        # --- chilledwater within-meter secondary ---
        "chilledwater_positive_vs_chilledwater_negative_pairwise_auc": _pairwise_auc(
            cw_pos, cw_neg
        ),
        "chilledwater_positive_minus_chilledwater_negative_score_margin": _margin(
            cw_pos, cw_neg
        ),
        "chilledwater_within_meter_pr_auc": cw_pr,
        "chilledwater_within_meter_roc_auc": cw_roc,
        "chilledwater_positive_within_meter_rank": _mean(
            cw_rank[anomaly[cw_mask] == 1] if cw_mask.any() else np.empty(0)
        ),
        "chilledwater_positive_global_rank": _mean(
            global_rank[cw_mask & (anomaly == 1)]
        ),
        # --- resolution-limited diagnostic only, never mechanism-bearing ---
        "RESOLUTION_LIMITED_DIAGNOSTIC_chilledwater_positive_vs_hotwater_negative_pairwise_auc": _pairwise_auc(
            cw_pos, hw_neg
        ),
    }


ENDPOINT_NAMES = tuple(
    endpoints(
        np.zeros(4),
        np.array(
            [
                METER["steam"],
                METER["hotwater"],
                METER["chilledwater"],
                METER["chilledwater"],
            ]
        ),
        np.array([1, 0, 1, 0], dtype="int8"),
    )
)


def fit_level_estimate(
    repeat_scores: list[np.ndarray],
    meter: np.ndarray,
    anomaly: np.ndarray,
) -> dict[str, float]:
    """Average the endpoint over a fit's repeats -- never the row probabilities.

    Each repeat's score vector is scored on its own; the mean is taken over the
    resulting endpoint values. Averaging the probability vectors first and
    scoring once is a different estimator and is forbidden.
    """
    per_repeat = [endpoints(s, meter, anomaly) for s in repeat_scores]
    out = {}
    for name in per_repeat[0]:
        vals = np.array([r[name] for r in per_repeat], dtype="float64")
        out[name] = float(vals.mean()) if np.all(np.isfinite(vals)) else INVALID
    return out


def factor_effect(values: dict[str, float]) -> dict[str, float]:
    """The repository's exact factorial formulas, sign and coding unchanged.

    Keys are the two-character cell codes: the first character is
    hotwater-positive presence, the second hotwater-negative presence.
    """
    y00, y01, y10, y11 = (
        values["00"],
        values["01"],
        values["10"],
        values["11"],
    )
    return {
        "positive_support_main_effect": (y10 + y11 - y00 - y01) / 2,
        "negative_support_main_effect": (y01 + y11 - y00 - y10) / 2,
        "positive_x_negative_interaction": y11 - y10 - y01 + y00,
    }


EFFECT_NAMES = (
    "positive_support_main_effect",
    "negative_support_main_effect",
    "positive_x_negative_interaction",
)
