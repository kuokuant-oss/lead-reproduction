"""Cluster-weighted estimators that scale to the full holdout.

A cluster bootstrap draw resamples cluster names with replacement. The naive
implementation then materialises the resampled rows and recomputes the metric,
which at 594,318 rows and 1,000 draws is wasteful: the resampled multiset only
differs from the original by per-row integer multiplicities.

Both estimators here take those multiplicities as weights and are *exact* --
equal to the naive estimator by construction, not approximately. The tests prove
it against the naive path on synthetic data and on the real E5 192-row data.

AUC is the cluster-weighted Mann-Whitney statistic. The score order does not
depend on the draw, so the sort happens once per unit and each draw costs one
O(n) sweep. Ties contribute a half, exactly as `roc_auc_score` does.

The margin is a difference of weighted means, computable from per-cluster score
sums and counts alone.
"""

from __future__ import annotations

import numpy as np

MASTER_SEED = 20260730
NAMESPACE_CODE = 6006
CLUSTER_CODE = {"building": 1, "segment": 2}
DRAWS = 1000


def draw_generator(
    cluster_type: str, draw_id: int, namespace: int = NAMESPACE_CODE
) -> np.random.Generator:
    """Addressable per-draw generator, same construction as E4 and E5."""
    if cluster_type not in CLUSTER_CODE:
        raise KeyError(f"unknown cluster type {cluster_type!r}")
    if not 0 <= draw_id < DRAWS:
        raise ValueError(f"draw_id {draw_id} outside the frozen range 0..{DRAWS - 1}")
    seq = np.random.SeedSequence(
        [MASTER_SEED, namespace, CLUSTER_CODE[cluster_type], draw_id]
    )
    return np.random.Generator(np.random.PCG64(seq))


def cluster_multiplicities(
    rng: np.random.Generator, codes: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Per-row weights from one cluster resample.

    `codes` maps each row to a cluster index in 0..n_clusters-1. Drawing
    `n_clusters` cluster indices with replacement and counting them gives each
    cluster its multiplicity; a row's weight is its cluster's multiplicity.
    """
    chosen = rng.integers(0, n_clusters, size=n_clusters)
    mult = np.bincount(chosen, minlength=n_clusters)
    return mult[codes]


class SortedAUC:
    """Pre-sorted state for exact cluster-weighted AUC on one score vector."""

    def __init__(self, score: np.ndarray, positive: np.ndarray) -> None:
        score = np.asarray(score, dtype="float64")
        positive = np.asarray(positive, dtype=bool)
        order = np.argsort(score, kind="stable")
        self.order = order
        self.pos = positive[order]
        # Tie groups: rows sharing a score value contribute half a pair.
        _, self.group = np.unique(score[order], return_inverse=True)
        self.n_groups = int(self.group.max()) + 1 if score.size else 0

    def __call__(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype="float64")[self.order]
        wp = np.where(self.pos, w, 0.0)
        wn = np.where(self.pos, 0.0, w)
        den = wp.sum() * wn.sum()
        if den == 0:
            return float("nan")
        per_group_neg = np.bincount(self.group, weights=wn, minlength=self.n_groups)
        below = np.concatenate(([0.0], np.cumsum(per_group_neg)))[:-1]
        num = float(np.sum(wp * (below[self.group] + 0.5 * per_group_neg[self.group])))
        return num / den


def weighted_margin(
    score: np.ndarray, positive: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted mean(positive) - weighted mean(negative). Exact."""
    s = np.asarray(score, dtype="float64")
    p = np.asarray(positive, dtype=bool)
    w = np.asarray(weights, dtype="float64")
    wp, wn = w[p], w[~p]
    if wp.sum() == 0 or wn.sum() == 0:
        return float("nan")
    return float((s[p] * wp).sum() / wp.sum() - (s[~p] * wn).sum() / wn.sum())


# --------------------------------------------------------------------------
# naive reference implementations, kept so the fast paths can be proved equal
# --------------------------------------------------------------------------


def naive_resampled_rows(codes: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Row indices of the resampled multiset, for the reference path."""
    return np.repeat(np.arange(codes.size), mult)


def naive_auc(score: np.ndarray, positive: np.ndarray, rows: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    s, p = np.asarray(score)[rows], np.asarray(positive)[rows]
    if p.all() or not p.any():
        return float("nan")
    return float(roc_auc_score(p.astype("int8"), s))


def naive_margin(score: np.ndarray, positive: np.ndarray, rows: np.ndarray) -> float:
    s, p = np.asarray(score, dtype="float64")[rows], np.asarray(positive)[rows]
    if not p.any() or p.all():
        return float("nan")
    return float(s[p].mean() - s[~p].mean())
