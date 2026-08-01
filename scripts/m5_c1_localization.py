"""M5 chilledwater C1 localization.

CPU-only aggregation over existing row-level, segment, and prediction artifacts.
No fit, no refit, no TabPFN inference, no tree refit, no scoring of the frozen
192-row query.

Every unit is an atomic checkpoint, so the run is interruption-safe and resumes
by skipping units already present. There is no wall-clock timeout and no partial
result is interpreted: summaries are assembled only from a complete unit set.

Uncertainty is always clustered by building or by segment. Rows are never
treated as independent.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

CONTEXTS = (5000, 10000, 20000, 50000, 100000)
ROBUSTNESS_CONTEXTS = (10000, 20000, 50000, 100000)
NEG_GROUPS = ("hotwater", "electricity", "chilledwater", "steam")
BOOTSTRAP_DRAWS = 1000
BLOCK = 10  # small blocks: an interruption costs one block, not fifty
SEED = 20260801

_CACHE: dict[str, object] = {}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def movement_path(data_root: Path, learner: str) -> Path:
    name = "tabpfn" if learner == "tabpfn" else "trees"
    return (
        data_root
        / "processed"
        / "m5_context_mechanism_137"
        / f"m5_137_row_score_rank_movement_{name}.parquet"
    )


def load_rows(data_root: Path, meters: tuple[str, ...]) -> pd.DataFrame:
    """Rows for the requested meters, both learners, C1 columns only.

    Only the requested meters are materialised: loading all 10.1M rows for every
    worker would need ~2.6 GB each and exhaust memory at any useful parallelism.
    The two learner tables share a row order, which is asserted rather than
    assumed, so they are concatenated instead of merged.
    """
    key = "rows__" + ",".join(sorted(meters))
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    base_cols = [
        "raw_index",
        "building_id",
        "meter_name",
        "anomaly",
        "meter_reading",
        "reading_regime",
    ]
    score_cols = [f"score_{c}" for c in CONTEXTS]
    rank_cols = [f"global_rank_delta_{c}" for c in CONTEXTS if c != 5000]
    wm_cols = [f"within_meter_rank_delta_{c}" for c in CONTEXTS if c != 5000]
    value_cols = score_cols + rank_cols + wm_cols
    keep = pc.is_in(pc.field("meter_name"), value_set=pa.array(list(meters)))

    tab = pq.read_table(
        movement_path(data_root, "tabpfn"), columns=base_cols + value_cols, filters=keep
    )
    tre = pq.read_table(
        movement_path(data_root, "trees"),
        columns=["raw_index"] + value_cols,
        filters=keep,
    )
    if tab.num_rows != tre.num_rows:
        raise RuntimeError("learner tables disagree on row count")
    frame = tab.to_pandas()
    tree_frame = tre.to_pandas()
    if not np.array_equal(
        frame["raw_index"].to_numpy(), tree_frame["raw_index"].to_numpy()
    ):
        raise RuntimeError("learner tables are not row-aligned on raw_index")
    for col in value_cols:
        frame[f"tree_{col}"] = tree_frame[col].astype("float32")
        frame[col] = frame[col].astype("float32")
    _CACHE[key] = frame
    return frame


def load_segment_map(data_root: Path) -> pd.DataFrame:
    mech = data_root / "processed" / "m5_context_mechanism_137"
    return pq.read_table(mech / "m5_137_anomaly_segments.parquet").to_pandas()


def metric_block(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def quartile_labels(values: pd.Series) -> pd.Series:
    """Quartile labels that survive tied edges.

    `duplicates="drop"` can leave fewer than four bins when a morphology column
    has many repeated values (reading_slope and ratio_1h_mean both do), and a
    fixed four-label list then raises. Label from the realised bin count instead;
    the cutpoints themselves are never adjusted to taste.
    """
    codes = pd.qcut(values, 4, labels=False, duplicates="drop")
    return pd.Series(codes, index=values.index).map(
        lambda c: "unbinned" if pd.isna(c) else f"q{int(c) + 1}"
    )


# --------------------------------------------------------------------------
# A. Movement decomposition
# --------------------------------------------------------------------------
def unit_movement(data_root: Path, context: int) -> dict:
    rows = load_rows(data_root, ("chilledwater",))
    cw = rows
    y = cw["anomaly"].to_numpy()
    out: dict[str, object] = {
        "context_rows": context,
        "chilledwater_rows": int(len(cw)),
        "anomaly_rows": int(y.sum()),
        "normal_rows": int(len(cw) - y.sum()),
    }
    for learner, prefix in (("tabpfn", ""), ("tree", "tree_")):
        s = cw[f"{prefix}score_{context}"].to_numpy()
        base = cw[f"{prefix}score_5000"].to_numpy()
        m = metric_block(y, s)
        out[learner] = {
            # within-chilledwater anomaly-vs-normal ranking
            "within_meter_pr_auc": m["pr_auc"],
            "within_meter_roc_auc": m["roc_auc"],
            # absolute score / calibration level, kept separate from ranking
            "anomaly_mean_score": float(s[y == 1].mean()),
            "normal_mean_score": float(s[y == 0].mean()),
            "score_separation": float(s[y == 1].mean() - s[y == 0].mean()),
            "anomaly_median_score": float(np.median(s[y == 1])),
            "normal_median_score": float(np.median(s[y == 0])),
            # absolute score movement relative to the 5k reference context
            "anomaly_score_movement_from_5k": float((s - base)[y == 1].mean()),
            "normal_score_movement_from_5k": float((s - base)[y == 0].mean()),
        }
        if context != 5000:
            gr = cw[f"{prefix}global_rank_delta_{context}"].to_numpy()
            wr = cw[f"{prefix}within_meter_rank_delta_{context}"].to_numpy()
            out[learner].update(
                {
                    "anomaly_global_rank_movement": float(gr[y == 1].mean()),
                    "normal_global_rank_movement": float(gr[y == 0].mean()),
                    "anomaly_within_meter_rank_movement": float(wr[y == 1].mean()),
                    "normal_within_meter_rank_movement": float(wr[y == 0].mean()),
                }
            )
    tp, tr = out["tabpfn"], out["tree"]  # type: ignore[index]
    out["tabpfn_minus_tree"] = {
        k: float(tp[k] - tr[k]) for k in tp if isinstance(tp[k], float) and k in tr
    }
    return out


# --------------------------------------------------------------------------
# B. Support-source comparison
# --------------------------------------------------------------------------
def unit_support(data_root: Path, negative_meter: str, context: int) -> dict:
    rows = load_rows(data_root, tuple(sorted({"chilledwater", negative_meter})))
    pos = rows[(rows["meter_name"] == "chilledwater") & (rows["anomaly"] == 1)]
    neg = rows[(rows["meter_name"] == negative_meter) & (rows["anomaly"] == 0)]
    y = np.concatenate(
        [np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    )
    out: dict[str, object] = {
        "comparison": f"chilledwater_positive_vs_{negative_meter}_negative",
        "context_rows": context,
        "positive_rows": int(len(pos)),
        "negative_rows": int(len(neg)),
        "positive_buildings": int(pos["building_id"].nunique()),
        "negative_buildings": int(neg["building_id"].nunique()),
    }
    for learner, prefix in (("tabpfn", ""), ("tree", "tree_")):
        sp = pos[f"{prefix}score_{context}"].to_numpy()
        sn = neg[f"{prefix}score_{context}"].to_numpy()
        s = np.concatenate([sp, sn])
        bp = pos[f"{prefix}score_5000"].to_numpy()
        bn = neg[f"{prefix}score_5000"].to_numpy()
        block = {
            "pairwise_auc": float(roc_auc_score(y, s)),
            "continuous_score_margin": float(sp.mean() - sn.mean()),
            "context_movement_margin": float((sp - bp).mean() - (sn - bn).mean()),
        }
        if context != 5000:
            gp = pos[f"{prefix}global_rank_delta_{context}"].to_numpy()
            gn = neg[f"{prefix}global_rank_delta_{context}"].to_numpy()
            block["rank_gap"] = float(gp.mean() - gn.mean())
        out[learner] = block
    tp, tr = out["tabpfn"], out["tree"]  # type: ignore[index]
    out["tabpfn_minus_tree"] = {k: float(tp[k] - tr[k]) for k in tp if k in tr}
    return out


# --------------------------------------------------------------------------
# C. Morphology localization
# --------------------------------------------------------------------------
def unit_morphology(data_root: Path, factor: str) -> dict:
    cw = load_rows(data_root, ("chilledwater",)).copy()
    ctx, base = 100000, 5000
    cw["gap_movement"] = (cw[f"score_{ctx}"] - cw[f"score_{base}"]) - (
        cw[f"tree_score_{ctx}"] - cw[f"tree_score_{base}"]
    )
    out: dict[str, object] = {"factor": factor, "context_rows": ctx}

    if factor in ("raw_reading_quartile", "reading_regime", "building_id"):
        if factor == "raw_reading_quartile":
            cw["stratum"] = quartile_labels(cw["meter_reading"])
        elif factor == "reading_regime":
            cw["stratum"] = cw["reading_regime"].astype(str)
        else:
            cw["stratum"] = cw["building_id"].astype(int)
        grp = cw.groupby("stratum", observed=True)
        agg = grp.agg(
            rows=("gap_movement", "size"),
            positives=("anomaly", "sum"),
            mean_gap_movement=("gap_movement", "mean"),
        )
        agg["abs_total"] = grp["gap_movement"].apply(lambda s: float(np.abs(s).sum()))
        agg["share_of_abs_total"] = agg["abs_total"] / agg["abs_total"].sum()
        agg = agg.sort_values("share_of_abs_total", ascending=False)
        out["strata"] = json.loads(agg.reset_index().to_json(orient="records"))
        out["top_stratum_share"] = float(agg["share_of_abs_total"].iloc[0])
        out["strata_count"] = int(len(agg))
        return out

    seg = load_segment_map(data_root)
    seg = seg[seg["meter_name"] == "chilledwater"].copy()
    seg["gap_movement"] = seg["tabpfn_score_movement"] - seg["tree_score_movement"]
    if factor == "anomaly_phase":
        phases = pq.read_table(
            data_root
            / "processed"
            / "m5_context_mechanism_137"
            / "m5_137_anomaly_segment_phases.parquet"
        ).to_pandas()
        phases = phases[phases["segment_id"].isin(seg["segment_id"])]
        phases["gap_movement"] = (
            phases["tabpfn_score_movement"] - phases["tree_score_movement"]
        )
        grp = phases.groupby("phase", observed=True)
        agg = grp.agg(
            segments=("segment_id", "nunique"),
            rows=("rows", "sum"),
            mean_gap_movement=("gap_movement", "mean"),
        )
        out["strata"] = json.loads(agg.reset_index().to_json(orient="records"))
        out["strata_count"] = int(len(agg))
        return out

    column = {
        "duration": "duration_rows",
        "slope": "reading_slope",
        "deviation_24h": "deviation_24h",
        "deviation_168h": "deviation_168h",
        "diff_morphology": "diff_1h_mean",
        "ratio_morphology": "ratio_1h_mean",
        "segment_id": "segment_id",
    }[factor]
    if factor == "segment_id":
        seg = seg.sort_values("gap_movement", key=lambda s: s.abs(), ascending=False)
        total = float(seg["gap_movement"].abs().sum())
        out["segments"] = int(len(seg))
        out["top1_share"] = float(seg["gap_movement"].abs().iloc[0] / total)
        out["top10_share"] = float(seg["gap_movement"].abs().iloc[:10].sum() / total)
        out["top50_share"] = float(seg["gap_movement"].abs().iloc[:50].sum() / total)
        return out
    seg["stratum"] = quartile_labels(seg[column])
    grp = seg.groupby("stratum", observed=True)
    agg = grp.agg(
        segments=("segment_id", "nunique"),
        rows=("duration_rows", "sum"),
        mean_gap_movement=("gap_movement", "mean"),
    )
    agg["abs_total"] = grp["gap_movement"].apply(lambda s: float(np.abs(s).sum()))
    agg["share_of_abs_total"] = agg["abs_total"] / agg["abs_total"].sum()
    out["strata"] = json.loads(agg.reset_index().to_json(orient="records"))
    out["top_stratum_share"] = float(agg["share_of_abs_total"].max())
    out["strata_count"] = int(len(agg))
    return out


# --------------------------------------------------------------------------
# D. Clustered bootstrap
# --------------------------------------------------------------------------
def _bootstrap_arrays(data_root: Path, cluster: str):
    key = f"boot_{cluster}"
    if key in _CACHE:
        return _CACHE[key]
    cw = load_rows(data_root, ("chilledwater",))
    building = cw["building_id"].to_numpy()
    anomaly = cw["anomaly"].to_numpy()
    if cluster == "segment":
        # Segment clusters apply to anomaly rows; normal rows form their own
        # building-level clusters so the resample stays a partition of the data.
        codes = building.astype("int64") * 2 + (anomaly == 1)
    else:
        codes = building.astype("int64")
    y = anomaly
    scores = {
        c: (cw[f"score_{c}"].to_numpy(), cw[f"tree_score_{c}"].to_numpy())
        for c in ROBUSTNESS_CONTEXTS
    }
    # Integer codes plus one stable sort partition every cluster at once:
    # O(n log n). A string cluster key with a per-cluster `flatnonzero` scan is
    # O(n * clusters) -- 252 clusters x 2.1M rows -- and dominated the runtime.
    dense, levels = pd.factorize(codes, sort=True)
    order = np.argsort(dense, kind="stable")
    counts = np.bincount(dense, minlength=len(levels))
    groups = np.split(order, np.cumsum(counts)[:-1])
    index_by_cluster = dict(enumerate(groups))
    uniq = np.arange(len(levels))
    payload = (y, scores, uniq, index_by_cluster)
    _CACHE[key] = payload
    return payload


def unit_bootstrap(data_root: Path, cluster: str, block_start: int) -> dict:
    y, scores, uniq, index_by_cluster = _bootstrap_arrays(data_root, cluster)
    draws = []
    for draw in range(block_start, min(block_start + BLOCK, BOOTSTRAP_DRAWS)):
        rng = np.random.default_rng([SEED, 1 if cluster == "building" else 2, draw])
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([index_by_cluster[u] for u in pick])
        ys = y[sel]
        if ys.sum() == 0 or ys.sum() == len(ys):
            draws.append({"draw": draw, "invalid": "single_class"})
            continue
        rec: dict[str, object] = {"draw": draw}
        for c, (tab, tre) in scores.items():
            a, b = tab[sel], tre[sel]
            rec[f"pr_gap_{c}"] = float(
                average_precision_score(ys, a) - average_precision_score(ys, b)
            )
            rec[f"roc_gap_{c}"] = float(roc_auc_score(ys, a) - roc_auc_score(ys, b))
        draws.append(rec)
    return {
        "cluster": cluster,
        "block_start": block_start,
        "clusters": int(len(uniq)),
        "draws": draws,
    }


# --------------------------------------------------------------------------
# Unit plan / driver
# --------------------------------------------------------------------------
def unit_plan() -> list[tuple[str, dict]]:
    units: list[tuple[str, dict]] = []
    for c in CONTEXTS:
        units.append((f"movement__ctx{c}", {"kind": "movement", "context": c}))
    for neg in NEG_GROUPS:
        for c in CONTEXTS:
            units.append(
                (
                    f"support__{neg}__ctx{c}",
                    {"kind": "support", "negative": neg, "context": c},
                )
            )
    for f in (
        "raw_reading_quartile",
        "reading_regime",
        "anomaly_phase",
        "duration",
        "slope",
        "deviation_24h",
        "deviation_168h",
        "diff_morphology",
        "ratio_morphology",
        "building_id",
        "segment_id",
    ):
        units.append((f"morphology__{f}", {"kind": "morphology", "factor": f}))
    for cluster in ("building", "segment"):
        for start in range(0, BOOTSTRAP_DRAWS, BLOCK):
            units.append(
                (
                    f"bootstrap__{cluster}__{start:04d}",
                    {"kind": "bootstrap", "cluster": cluster, "start": start},
                )
            )
    return units


def run_unit(data_root: Path, spec: dict) -> dict:
    kind = spec["kind"]
    if kind == "movement":
        return unit_movement(data_root, spec["context"])
    if kind == "support":
        return unit_support(data_root, spec["negative"], spec["context"])
    if kind == "morphology":
        return unit_morphology(data_root, spec["factor"])
    if kind == "bootstrap":
        return unit_bootstrap(data_root, spec["cluster"], spec["start"])
    raise ValueError(kind)


def _worker(args):
    data_root, unit_id, spec = args
    t0 = time.perf_counter()
    payload = run_unit(Path(data_root), spec)
    return unit_id, payload, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--census", action="store_true", help="list units and exit without computing"
    )
    # Independent-shard mode: each process owns a disjoint slice of the unit
    # list and writes its own checkpoints. Preferred over the in-process pool --
    # there is no parent-side result collection to stall, and a dead shard costs
    # only its own in-flight unit.
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    args = ap.parse_args()

    units = unit_plan()
    unit_dir = args.output_root / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    pending = [(u, s) for u, s in units if not (unit_dir / f"{u}.json").exists()]

    if args.census:
        kinds: dict[str, int] = {}
        for _, s in units:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        print(
            json.dumps(
                {
                    "total_units": len(units),
                    "pending": len(pending),
                    "by_kind": kinds,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "block": BLOCK,
                },
                indent=2,
            )
        )
        return 0

    if args.shard_index is not None:
        if args.shard_count is None or not (0 <= args.shard_index < args.shard_count):
            raise SystemExit("--shard-index must be within --shard-count")
        mine = [
            (u, s)
            for i, (u, s) in enumerate(pending)
            if i % args.shard_count == args.shard_index
        ]
        print(
            f"shard {args.shard_index}/{args.shard_count}: {len(mine)} units",
            flush=True,
        )
        started = time.perf_counter()
        for k, (unit_id, spec) in enumerate(mine, start=1):
            path = unit_dir / f"{unit_id}.json"
            if path.exists():
                continue
            t0 = time.perf_counter()
            payload = run_unit(args.data_root, spec)
            secs = time.perf_counter() - t0
            atomic_json(path, {"unit_id": unit_id, "seconds": secs, "payload": payload})
            el = time.perf_counter() - started
            print(
                f"  [{args.shard_index}] {k}/{len(mine)} {unit_id} "
                f"({secs:.1f}s) eta={(len(mine) - k) * el / k / 60:.1f}min",
                flush=True,
            )
        print(
            f"shard {args.shard_index} done in "
            f"{(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )
        return 0

    print(
        f"units total={len(units)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, (str(args.data_root), u, s)): u for u, s in pending
        }
        for fut in as_completed(futures):
            unit_id, payload, secs = fut.result()
            atomic_json(
                unit_dir / f"{unit_id}.json",
                {
                    "unit_id": unit_id,
                    "seconds": secs,
                    "payload": payload,
                },
            )
            done += 1
            if done % 5 == 0 or done == len(pending):
                el = time.perf_counter() - started
                rate = done / el
                print(
                    f"  {done}/{len(pending)} last={unit_id} "
                    f"({secs:.1f}s) eta={(len(pending) - done) / rate / 60:.1f}min",
                    flush=True,
                )
    leftover = [u for u, _ in units if not (unit_dir / f"{u}.json").exists()]
    if leftover:
        print(f"INCOMPLETE: {len(leftover)} units missing")
        return 1
    print(
        f"all {len(units)} units complete in "
        f"{(time.perf_counter() - started) / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
