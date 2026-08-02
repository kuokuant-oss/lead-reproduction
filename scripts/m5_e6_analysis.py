"""M5 E6 clustered analysis on the natural-prevalence holdout.

Two co-primary endpoints on the 594,318-row steam-positive / hotwater-negative
subset, two cluster definitions, 1000 addressable draws in namespace 6006. The
factorial formulas, their signs, and their coding are E4's, unchanged.

One draw's cluster resample is shared by everything scored inside it -- all
cells, both arms, all three context seeds, TabPFN and the fixed trees, and both
endpoints -- because the contrasts are differences and giving each arm its own
resample would inflate their variance with noise the contrast does not contain.

The segment definition is E4's and E5's and is not redefined here. It is also
91.8% singletons on this subset, so the segment interval is reported with that
degeneracy stated rather than presented as cluster-level corroboration equal in
independence to the building interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e4_endpoints import EFFECT_NAMES, METER, factor_effect  # noqa: E402
from m5_e6_clustered import (  # noqa: E402
    DRAWS,
    SortedAUC,
    cluster_multiplicities,
    draw_generator,
    weighted_margin,
)

ROWS = 10_137_155
CELLS = ("00", "01", "10", "11")
SEEDS = (42, 123, 999)
ARMS = ("cell_specific", "frozen_reference")
ENDPOINTS = ("auc", "margin")
# Taken from the canonical map rather than restated, so the two cannot drift.
METER_STEAM = METER["steam"]
METER_HOTWATER = METER["hotwater"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_codes(raw_index: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """E4's and E5's segment definition, applied unchanged.

    Contiguous anomaly runs in `raw_index` (a gap greater than 1 breaks a run)
    form one cluster; every non-anomaly row is its own singleton cluster.
    """
    order = np.argsort(raw_index, kind="stable")
    ri, an = raw_index[order], anomaly[order]
    codes = np.empty(ri.size, dtype="int64")
    nxt = 0
    i = 0
    while i < ri.size:
        if an[i] == 1:
            j = i + 1
            while j < ri.size and an[j] == 1 and ri[j] - ri[j - 1] == 1:
                j += 1
            codes[i:j] = nxt
            nxt += 1
            i = j
        else:
            codes[i] = nxt
            nxt += 1
            i += 1
    out = np.empty_like(codes)
    out[order] = codes
    return out


def unit_id(seed: int, cell: str, arm: str) -> str:
    return f"seed{seed}__cell{cell}__{arm}"


def load_vectors(root: Path, name: str, subset: np.ndarray) -> dict[str, np.ndarray]:
    """The 24 score vectors, restricted to the co-primary subset."""
    out = {}
    for seed in SEEDS:
        for cell in CELLS:
            for arm in ARMS:
                uid = unit_id(seed, cell, arm)
                path = root / uid / name
                v = np.load(path, mmap_mode="r")
                if v.shape != (ROWS,):
                    raise SystemExit(f"{uid}: {name} has shape {v.shape}")
                out[uid] = np.asarray(v[subset], dtype="float64")
    return out


def aggregate(per_unit: dict[str, float]) -> dict:
    """Cell values -> factor effects, per seed and arm, then equal-weight means.

    The three context seeds carry equal weight regardless of how the units are
    ordered, and the two scaler arms are averaged only after each arm's own
    factorial contrast has been formed inside the draw.
    """
    per_seed_arm = {}
    for seed in SEEDS:
        for arm in ARMS:
            vals = {c: per_unit[unit_id(seed, c, arm)] for c in CELLS}
            per_seed_arm[f"seed{seed}__{arm}"] = factor_effect(vals)
    overall = {
        name: float(np.mean([v[name] for v in per_seed_arm.values()]))
        for name in EFFECT_NAMES
    }
    per_seed = {
        f"seed{seed}": {
            name: float(np.mean([per_seed_arm[f"seed{seed}__{a}"][name] for a in ARMS]))
            for name in EFFECT_NAMES
        }
        for seed in SEEDS
    }
    per_arm = {
        arm: {
            name: float(np.mean([per_seed_arm[f"seed{s}__{arm}"][name] for s in SEEDS]))
            for name in EFFECT_NAMES
        }
        for arm in ARMS
    }
    return {
        "overall": overall,
        "per_seed": per_seed,
        "per_arm": per_arm,
        "per_seed_arm": per_seed_arm,
    }


def flatten(agg: dict, prefix: str) -> dict[str, float]:
    out = {}
    for name in EFFECT_NAMES:
        out[f"{prefix}|overall|{name}"] = agg["overall"][name]
        for k, v in agg["per_seed"].items():
            out[f"{prefix}|{k}|{name}"] = v[name]
        for k, v in agg["per_arm"].items():
            out[f"{prefix}|{k}|{name}"] = v[name]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--tabpfn-root", type=Path, required=True)
    ap.add_argument("--tree-root", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--dist-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()

    boot = read_json(args.protocol_root / "e6_bootstrap_manifest.json")
    rules = read_json(args.protocol_root / "e6_decision_rules.json")

    # ---- identity and the co-primary subset --------------------------------
    raw, anom, bid, meter = [], [], [], []
    for half in ("head", "tail"):
        md = np.load(args.dist_root / half / "metadata.npz")
        raw.append(np.asarray(md["raw_index"], dtype="int64"))
        anom.append(np.asarray(md["anomaly"], dtype="int8"))
        bid.append(np.asarray(md["building_id"], dtype="int64"))
        f = np.load(args.dist_root / half / "features.float32.npy", mmap_mode="r")
        meter.append(np.asarray(f[:, 0]))
        del f
    raw_index = np.concatenate(raw)
    anomaly = np.concatenate(anom)
    building = np.concatenate(bid)
    m = np.concatenate(meter)
    levels = sorted(np.unique(m).tolist())
    meter_code = np.zeros(m.size, dtype="int8")
    for v, i in {v: i for i, v in enumerate(levels)}.items():
        meter_code[m == v] = i

    steam_pos = (meter_code == METER_STEAM) & (anomaly == 1)
    hw_neg = (meter_code == METER_HOTWATER) & (anomaly == 0)
    subset = np.flatnonzero(steam_pos | hw_neg)
    positive = steam_pos[subset]
    cp = boot["co_primary_subset"]
    if subset.size != cp["total_rows"]:
        raise SystemExit(f"subset has {subset.size} rows, frozen {cp['total_rows']}")
    if int(positive.sum()) != cp["steam_positive_rows"]:
        raise SystemExit("steam positive count differs from the frozen manifest")

    b_codes = np.unique(building[subset], return_inverse=True)[1]
    b_n = int(b_codes.max()) + 1
    s_codes = segment_codes(raw_index[subset], anomaly[subset])
    _, s_codes = np.unique(s_codes, return_inverse=True)
    s_n = int(s_codes.max()) + 1
    if b_n != cp["building_clusters"] or s_n != cp["segment_clusters"]:
        raise SystemExit(
            f"clusters {b_n} building / {s_n} segment differ from the frozen "
            f"{cp['building_clusters']} / {cp['segment_clusters']}"
        )
    print(
        f"co-primary subset: {subset.size:,} rows, {b_n} building clusters, "
        f"{s_n:,} segment clusters"
    )

    # ---- score vectors ------------------------------------------------------
    tab = load_vectors(args.tabpfn_root, "scores.float32.npy", subset)
    tree = load_vectors(args.tree_root, "tree_scores.float32.npy", subset)
    print(f"loaded {len(tab)} TabPFN and {len(tree)} tree vectors")

    auc_state = {
        f"{k}|{uid}": SortedAUC(v, positive)
        for k, src in (("tabpfn", tab), ("tree", tree))
        for uid, v in src.items()
    }

    # ---- point estimates ----------------------------------------------------
    ones = np.ones(subset.size, dtype="float64")
    point = {}
    for family, src in (("tabpfn", tab), ("tree", tree)):
        for ep in ENDPOINTS:
            per_unit = {}
            for uid, v in src.items():
                per_unit[uid] = (
                    auc_state[f"{family}|{uid}"](ones)
                    if ep == "auc"
                    else weighted_margin(v, positive, ones)
                )
            point[f"{family}|{ep}"] = {
                "per_unit": per_unit,
                "effects": aggregate(per_unit),
            }
    for ep in ENDPOINTS:
        gap = {
            uid: point[f"tabpfn|{ep}"]["per_unit"][uid]
            - point[f"tree|{ep}"]["per_unit"][uid]
            for uid in tab
        }
        point[f"gap|{ep}"] = {"per_unit": gap, "effects": aggregate(gap)}

    for family in ("tabpfn", "tree", "gap"):
        for ep in ENDPOINTS:
            e = point[f"{family}|{ep}"]["effects"]["overall"]
            print(
                f"  {family:<7} {ep:<7} negative_support="
                f"{e['negative_support_main_effect']:+.6f}  "
                f"positive_support={e['positive_support_main_effect']:+.6f}  "
                f"interaction={e['positive_x_negative_interaction']:+.6f}"
            )

    # ---- clustered draws ----------------------------------------------------
    keys = sorted(flatten(point["tabpfn|auc"]["effects"], "x"))
    draws: dict[str, dict[str, list[float]]] = {
        f"{ct}|{family}|{ep}": {k: [] for k in keys}
        for ct in ("building", "segment")
        for family in ("tabpfn", "tree", "gap")
        for ep in ENDPOINTS
    }
    t0 = time.perf_counter()
    for ct, codes, n_clusters in (
        ("building", b_codes, b_n),
        ("segment", s_codes, s_n),
    ):
        for d in range(args.draws):
            rng = draw_generator(ct, d)
            w = cluster_multiplicities(rng, codes, n_clusters).astype("float64")
            per = {}
            for family, src in (("tabpfn", tab), ("tree", tree)):
                for ep in ENDPOINTS:
                    per[f"{family}|{ep}"] = {
                        uid: (
                            auc_state[f"{family}|{uid}"](w)
                            if ep == "auc"
                            else weighted_margin(v, positive, w)
                        )
                        for uid, v in src.items()
                    }
            for ep in ENDPOINTS:
                per[f"gap|{ep}"] = {
                    uid: per[f"tabpfn|{ep}"][uid] - per[f"tree|{ep}"][uid]
                    for uid in tab
                }
            for family in ("tabpfn", "tree", "gap"):
                for ep in ENDPOINTS:
                    flat = flatten(aggregate(per[f"{family}|{ep}"]), "x")
                    tgt = draws[f"{ct}|{family}|{ep}"]
                    for k, v in flat.items():
                        tgt[k].append(v)
            if (d + 1) % 100 == 0:
                el = time.perf_counter() - t0
                print(
                    f"  {ct} draw {d + 1}/{args.draws}  ({el / 60:.1f} min)", flush=True
                )

    intervals = {}
    for key, series in draws.items():
        intervals[key] = {
            k: {
                "q025": float(np.quantile(v, 0.025)),
                "q500": float(np.quantile(v, 0.500)),
                "q975": float(np.quantile(v, 0.975)),
                "excludes_zero": bool(
                    np.quantile(v, 0.025) > 0 or np.quantile(v, 0.975) < 0
                ),
                "draws": len(v),
            }
            for k, v in series.items()
        }

    payload = {
        "schema": "m5_e6_analysis_v1",
        "generated": time.time(),
        "draws": args.draws,
        "namespace_code": boot["namespace_code"],
        "master_seed": boot["master_seed"],
        "co_primary_subset": {
            "rows": int(subset.size),
            "steam_positive_rows": int(positive.sum()),
            "hotwater_negative_rows": int((~positive).sum()),
            "building_clusters": b_n,
            "segment_clusters": s_n,
            "segment_singleton_clusters": int(
                (np.bincount(s_codes, minlength=s_n) == 1).sum()
            ),
        },
        "segment_degeneracy_disclosure": boot["segment_degeneracy_disclosure_required"],
        "estimand_note": "each unit contributes one canonical single-process "
        "batched pass; these intervals are conditional on that realized pass "
        "and do not cover same-state full-holdout inference-repeat variation",
        "decision_rules_sha256": sha256_file(
            args.protocol_root / "e6_decision_rules.json"
        ),
        "minimum_practical_effect_threshold": rules[
            "minimum_practical_effect_threshold"
        ],
        "point": point,
        "intervals": intervals,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    digest = atomic_json(args.out / "e6_analysis.json", payload)
    print(f"\nanalysis sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
