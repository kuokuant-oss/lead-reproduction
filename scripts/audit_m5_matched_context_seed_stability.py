"""Is the matched-N tree result a property of the method, or of one draw?

The matched-N context is frozen: seed 42, digest e9ffe0cf, fixed three days
before the tree arm produced anything and enforced by a digest check inside the
runner. That establishes the sample was not chosen after seeing tree results. It
does not establish that the number is stable -- a frozen sample can still be a
lucky one, and "this draw happens to be favourable" is not answerable by
provenance. It is answerable by redrawing.

So: same candidate pool, same balanced 50/50 sampler, same four frozen models,
same holdout rows scored every time -- only the seed changes. If the reported
figure is a property of the method, the seeds cluster. If it is a property of the
draw, they scatter, and the frozen number should be read as one sample from that
scatter.

The holdout subsample is drawn once, outside the seed loop, so every seed is
scored on identical rows and the comparison isolates the training draw.

    uv run python scripts/audit_m5_matched_context_seed_stability.py \
        --seeds 42 1 7 2024 --context-rows 100000
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame
from run_m3_figure_observations import (
    MODEL_ORDER,
    fit_frozen_models,
    predict_probability,
)
from run_m5_tree_ensemble_matched_context import (
    build_features_keeping_index,
    feature_columns,
)

CANONICAL_ORDER = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"


def load_canonical_module():
    spec = importlib.util.spec_from_file_location(
        "m5_canonical", ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 1, 7, 2024])
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, choices=(17, 137), default=17)
    parser.add_argument("--holdout-sample", type=int, default=2_000_000)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    args = parser.parse_args()
    mod = load_canonical_module()

    frame = load_m3_frame(verbose=True)
    building = frame["building_id"].to_numpy(dtype="int32")
    row_id = frame.index.to_numpy(dtype="int64")
    order = np.argsort(row_id)
    anomaly = frame["anomaly"].to_numpy(dtype="int8")

    train_mask = building % 2 == 0
    train_index = row_id[train_mask]
    with np.load(CANONICAL_ORDER) as payload:
        holdout_index = np.asarray(payload["validation_raw_index"], dtype="int64")

    # One holdout subsample for every seed: the thing under test is the training
    # draw, so the evaluation rows must not move with it.
    rng = np.random.default_rng(0)
    take = np.sort(
        rng.choice(
            len(holdout_index),
            size=min(args.holdout_sample, len(holdout_index)),
            replace=False,
        )
    )
    hold_ids = holdout_index[take]
    y_hold = anomaly[order[np.searchsorted(row_id, hold_ids, sorter=order)]].astype(int)

    # Every seed's fit rows are resolved before either feature matrix is built,
    # so the two expensive builds happen once instead of once per seed. At 137
    # features a build is ~120 lag columns over 10M rows and dominates both time
    # and memory; doing it per seed would also make this audit heavier than the
    # run it is auditing.
    fits = {}
    for seed in args.seeds:
        validation_index = mod.fixed_score_indices(
            train_index, args.validation_rows, seed=seed + 20_000
        )
        candidate_index = train_index[~np.isin(train_index, validation_index)]
        cand_y = anomaly[order[np.searchsorted(row_id, candidate_index, sorter=order)]]
        fits[seed] = mod.nested_balanced_indices(
            candidate_index, cand_y, [args.context_rows], seed=seed
        )[args.context_rows]

    if args.features == 17:
        cols = list(BASELINE_FEATURE_COLS)
        x_fits = {
            s: frame.loc[i, cols].to_numpy(dtype="float32") for s, i in fits.items()
        }
        y_fits = {s: frame.loc[i, "anomaly"] for s, i in fits.items()}
        x_hold = frame.loc[hold_ids, cols].to_numpy(dtype="float32")
    else:
        # The runner's own builder, imported rather than reimplemented: the
        # comparison is only meaningful if the features are constructed the same
        # way, including the raw_index carrier that keeps .loc meaning what it says.
        print("building 137 features for the training half", flush=True)
        train_full = build_features_keeping_index(frame.loc[train_mask])
        cols = feature_columns(137, list(train_full.columns))
        x_fits, y_fits = {}, {}
        for s, i in fits.items():
            selected = train_full.loc[i]
            if not np.array_equal(selected.index.to_numpy(dtype="int64"), i):
                raise AssertionError(
                    f"seed {s}: feature builder reordered the context rows"
                )
            x_fits[s] = selected[cols].to_numpy(dtype="float32")
            y_fits[s] = frame.loc[i, "anomaly"]
        del train_full, selected
        gc.collect()
        print("building 137 features for the holdout", flush=True)
        holdout_full = build_features_keeping_index(frame.loc[~train_mask])
        x_hold = holdout_full.loc[hold_ids, cols].to_numpy(dtype="float32")
        del holdout_full
        gc.collect()

    del frame
    gc.collect()
    print(f"\n{args.features} features, context {args.context_rows:,}")
    print(
        f"fixed holdout subsample: {len(y_hold):,} rows, "
        f"{y_hold.sum():,} positive ({y_hold.mean():.3%})\n"
    )

    results = []
    for seed in args.seeds:
        started = time.perf_counter()
        x_fit, y_fit = x_fits[seed], y_fits[seed]
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
        models, _ = fit_frozen_models(x_fit, y_fit)

        block = scaler.transform(x_hold)
        scores = np.mean(
            [predict_probability(n, models[n], block) for n in MODEL_ORDER], axis=0
        )
        r, p = roc_auc_score(y_hold, scores), average_precision_score(y_hold, scores)
        results.append((seed, r, p))
        tag = "  <-- the frozen draw" if seed == RANDOM_STATE else ""
        print(
            f"  seed {seed:>5}: ROC {r:.4f}  PR {p:.4f}   ({time.perf_counter() - started:.0f}s){tag}",
            flush=True,
        )

    rs = np.array([r for _, r, _ in results])
    ps = np.array([p for _, _, p in results])
    print(
        f"\n  ROC  mean {rs.mean():.4f}  sd {rs.std(ddof=1):.4f}  range {rs.max() - rs.min():.4f}"
    )
    print(
        f"  PR   mean {ps.mean():.4f}  sd {ps.std(ddof=1):.4f}  range {ps.max() - ps.min():.4f}"
    )
    frozen = [r for s, r, _ in results if s == RANDOM_STATE]
    if frozen:
        z = (frozen[0] - rs.mean()) / rs.std(ddof=1) if rs.std(ddof=1) > 0 else 0.0
        print(f"  the frozen draw sits {z:+.2f} sd from the mean of the redraws")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
