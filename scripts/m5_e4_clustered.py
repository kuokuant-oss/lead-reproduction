"""Clustered uncertainty for E4 formal Path A, per the human rulings A-E.

One draw is one resampled row multiset, and that same multiset is applied to
every context seed, cell, scaler arm, repeat, and the fixed tree comparator.
That is what makes the scaler-arm interaction and the cross-seed average valid
inside the draw instead of a difference of independent intervals.

Draw seeds are addressable, not sequential: draw `d` of cluster type `t` always
comes from `SeedSequence([20260730, 4004, code[t], d])`, so a draw is
reproducible on its own and no result depends on loop order.

Order of operations inside a draw, which is the part that is easy to get wrong:

    resample rows
      -> score each repeat separately on those rows
        -> average the 8 endpoint values within the fit   (never the probabilities)
          -> factorial contrast from the four fit-level values
            -> scaler interaction as frozen minus cell_specific
              -> equal-weight average over the three context seeds
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m5_e4_endpoints import (
    EFFECT_NAMES,
    endpoint_value,
    factor_effect,
)

MASTER_SEED = 20260730
NAMESPACE_CODE = 4004
CLUSTER_CODE = {"building": 1, "segment": 2}
DRAWS = 1000
CONTEXT_SEEDS = (42, 123, 999)
ARMS = ("cell_specific", "frozen_reference")
CELLS = ("00", "01", "10", "11")


def draw_generator(
    cluster_type: str, draw_id: int, namespace: int = NAMESPACE_CODE
) -> np.random.Generator:
    """Addressable per-draw generator. Never a shared sequential stream.

    `namespace` separates one stage's draws from another's while keeping the
    construction identical: E4 uses 4004, E5 uses 5005. The default preserves
    E4's stream exactly, so an E4 result recomputed today is unchanged.
    """
    if cluster_type not in CLUSTER_CODE:
        raise KeyError(f"unknown cluster type {cluster_type!r}")
    if not 0 <= draw_id < DRAWS:
        raise ValueError(f"draw_id {draw_id} outside the frozen range 0..{DRAWS - 1}")
    seq = np.random.SeedSequence(
        [MASTER_SEED, namespace, CLUSTER_CODE[cluster_type], draw_id]
    )
    return np.random.Generator(np.random.PCG64(seq))


def segment_clusters(raw_index: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """Contiguous anomaly runs; every non-anomaly row is its own cluster.

    Mirrors `analyze_m5_hotwater_label_role_factorial.segment_clusters`: anomaly
    rows whose raw_index is more than one apart start a new segment, and normal
    rows never share a cluster with anything.
    """
    labels = np.empty(raw_index.size, dtype=object)
    order = np.argsort(raw_index, kind="stable")
    seg = -1
    prev = None
    for pos in order:
        if anomaly[pos] == 1:
            if prev is None or raw_index[pos] - prev > 1:
                seg += 1
            labels[pos] = f"segment_{seg}"
            prev = raw_index[pos]
        else:
            labels[pos] = f"normal_{raw_index[pos]}"
    return labels


def cluster_members(labels: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    names = sorted(set(labels.tolist()))
    return names, {n: np.flatnonzero(labels == n) for n in names}


def resample_rows(
    rng: np.random.Generator, names: list[str], members: dict[str, np.ndarray]
) -> np.ndarray:
    """Cluster names with replacement, count preserved, members concatenated."""
    chosen = rng.choice(np.asarray(names, dtype=object), size=len(names), replace=True)
    return np.concatenate([members[n] for n in chosen])


class DrawInvalid(Exception):
    """Raised when a stratum is missing or a value is non-finite in this draw."""


def _fit_level(
    repeat_scores: list[np.ndarray],
    idx: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    endpoint: str,
) -> float:
    """Endpoint averaged over a fit's repeats, on the resampled rows.

    Each repeat is scored independently on the same rows and the mean is taken
    over endpoint values. The row probabilities are never averaged first.
    """
    vals = []
    for score in repeat_scores:
        v = endpoint_value(endpoint, score[idx], meter[idx], anomaly[idx])
        if not np.isfinite(v):
            raise DrawInvalid(endpoint)
        vals.append(v)
    return float(np.mean(vals))


def draw_contrasts(
    *,
    tabpfn: dict[tuple[int, str, str], list[np.ndarray]],
    trees: dict[tuple[int, str, str], np.ndarray],
    idx: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    endpoint: str,
) -> dict[str, float]:
    """All contrasts for one endpoint on one resampled row multiset.

    `tabpfn` maps (context_seed, cell, arm) to that fit's 8 repeat score
    vectors; `trees` maps the same key to the single fixed comparator vector.
    Repeat IDs are never paired across cells -- each cell's repeats are reduced
    to one fit-level value before any cell meets another.
    """
    out: dict[str, float] = {}
    per_seed: dict[str, dict[int, float]] = {}

    for arm in ARMS:
        for model in ("tabpfn", "tree"):
            for effect in EFFECT_NAMES:
                per_seed.setdefault(f"{model}__{arm}__{effect}", {})

    for seed in CONTEXT_SEEDS:
        for arm in ARMS:
            tab_cells, tree_cells = {}, {}
            for cell in CELLS:
                key = (seed, cell, arm)
                tab_cells[cell] = _fit_level(tabpfn[key], idx, meter, anomaly, endpoint)
                tv = endpoint_value(endpoint, trees[key][idx], meter[idx], anomaly[idx])
                if not np.isfinite(tv):
                    raise DrawInvalid(endpoint)
                tree_cells[cell] = float(tv)
            tab_eff = factor_effect(tab_cells)
            tree_eff = factor_effect(tree_cells)
            for effect in EFFECT_NAMES:
                per_seed[f"tabpfn__{arm}__{effect}"][seed] = tab_eff[effect]
                per_seed[f"tree__{arm}__{effect}"][seed] = tree_eff[effect]

    # Equal-weight average over the three pre-specified seeds; nothing is
    # resampled at the seed level and no seed is selected.
    for name, by_seed in per_seed.items():
        vals = [by_seed[s] for s in CONTEXT_SEEDS]
        out[name] = float(np.mean(vals))
        for s in CONTEXT_SEEDS:
            out[f"{name}__seed{s}"] = by_seed[s]

    for effect in EFFECT_NAMES:
        for model in ("tabpfn", "tree"):
            frozen = out[f"{model}__frozen_reference__{effect}"]
            cell_sp = out[f"{model}__cell_specific__{effect}"]
            # Formed inside the draw, never by subtracting two intervals.
            out[f"{model}__scaler_interaction__{effect}"] = frozen - cell_sp
        for arm in (*ARMS, "scaler_interaction"):
            out[f"tabpfn_minus_tree__{arm}__{effect}"] = (
                out[f"tabpfn__{arm}__{effect}"] - out[f"tree__{arm}__{effect}"]
            )
    return out


def run_cluster_bootstrap(
    *,
    cluster_type: str,
    tabpfn: dict[tuple[int, str, str], list[np.ndarray]],
    trees: dict[tuple[int, str, str], np.ndarray],
    query: pd.DataFrame,
    endpoint: str,
    draws: int = DRAWS,
    namespace: int = NAMESPACE_CODE,
) -> dict:
    """Percentile intervals over `draws` addressable clustered draws."""
    raw = query["raw_index"].to_numpy(dtype="int64")
    meter = query["meter"].to_numpy(dtype="int8")
    anomaly = query["anomaly"].to_numpy(dtype="int8")
    labels = (
        query["building_id"].astype(str).to_numpy(dtype=object)
        if cluster_type == "building"
        else segment_clusters(raw, anomaly)
    )
    names, members = cluster_members(labels)

    samples: dict[str, list[float]] = {}
    invalid = 0
    for draw_id in range(draws):
        rng = draw_generator(cluster_type, draw_id, namespace)
        idx = resample_rows(rng, names, members)
        try:
            row = draw_contrasts(
                tabpfn=tabpfn,
                trees=trees,
                idx=idx,
                meter=meter,
                anomaly=anomaly,
                endpoint=endpoint,
            )
        except DrawInvalid:
            invalid += 1
            continue
        for k, v in row.items():
            samples.setdefault(k, []).append(v)

    if not samples:
        raise AssertionError(
            f"clustered bootstrap lost every draw for {endpoint} ({cluster_type})"
        )

    out = {
        "cluster": cluster_type,
        "endpoint": endpoint,
        "namespace_code": namespace,
        "draws_requested": draws,
        "draws_valid": draws - invalid,
        "draws_invalid": invalid,
        "clusters": len(names),
        "contrasts": {},
    }
    for name, values in samples.items():
        arr = np.asarray(values, dtype="float64")
        out["contrasts"][name] = {
            "point_estimate": float(arr.mean()),
            "median": float(np.median(arr)),
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
            "excludes_zero": bool(
                np.quantile(arr, 0.025) > 0 or np.quantile(arr, 0.975) < 0
            ),
            "n": int(arr.size),
        }
    return out


def seed_consistency(per_seed: dict[int, float]) -> dict:
    """Descriptives the ruling requires alongside the equal-weight average.

    Deliberately not an inferential summary: three pre-specified seeds are not
    a sample from a seed population, so no t-test or random-effects model is
    produced here.
    """
    vals = np.array([per_seed[s] for s in CONTEXT_SEEDS], dtype="float64")
    positive = int((vals > 0).sum())
    negative = int((vals < 0).sum())
    return {
        "per_seed": {str(s): float(per_seed[s]) for s in CONTEXT_SEEDS},
        "overall_equal_weight_mean": float(vals.mean()),
        "range": float(vals.max() - vals.min()),
        "sample_sd": float(vals.std(ddof=1)),
        "sign_consistency": f"{max(positive, negative)}/3 "
        f"{'positive' if positive >= negative else 'negative'}",
        "all_same_sign": bool(positive == 3 or negative == 3),
        "interpretation": "equal-weight average of the three pre-specified "
        "seeds 42, 123, 999; not a generalisation beyond them",
    }
