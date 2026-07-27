"""Ask what the matched-N context is a random sample *of*.

`nested_balanced_indices` shuffles uniformly within each class, so it is random
given the label -- but it is not a random sample of the data. It is stratified
50/50 while the holdout runs at 6.3% positive, and because anomaly prevalence
varies by an order of magnitude across sites, conditioning on the label drags the
site and building composition with it.

That matters for how the curve is read. "10,000 labelled rows is enough" is only
true of 10,000 *balanced* rows, which carry 5,000 positives; 10,000 rows drawn at
random would carry roughly 630. This prints the actual distortion instead of
leaving it to be assumed either way.

    uv run python scripts/audit_m5_matched_context_composition.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np

from lead import RANDOM_STATE, ROOT, load_m3_frame


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
    parser.add_argument("--contexts", type=int, nargs="*", default=[10_000, 100_000])
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    args = parser.parse_args()
    mod = load_canonical_module()

    frame = load_m3_frame(verbose=True)
    building = frame["building_id"].to_numpy(dtype="int32")
    site = frame["site_id"].to_numpy(dtype="int16")
    anomaly = frame["anomaly"].to_numpy(dtype="int8")
    row_id = frame.index.to_numpy(dtype="int64")
    order = np.argsort(row_id)
    del frame

    train_mask = building % 2 == 0
    train_index = row_id[train_mask]
    validation_index = mod.fixed_score_indices(
        train_index, args.validation_rows, seed=args.seed + 20_000
    )
    keep = ~np.isin(train_index, validation_index)
    candidate_index = train_index[keep]
    cand_pos = np.searchsorted(row_id, candidate_index, sorter=order)
    cand_pos = order[cand_pos]
    candidate_y = anomaly[cand_pos]
    candidate_site = site[cand_pos]

    print(
        f"\ntraining half: {len(candidate_index):,} rows, "
        f"{int(candidate_y.sum()):,} positive ({candidate_y.mean():.3%})"
    )
    print(
        "A random 100,000-row draw would carry about "
        f"{int(round(candidate_y.mean() * 100_000)):,} positives; the matched "
        "context carries 50,000.\n"
    )

    train_site_share = np.bincount(candidate_site, minlength=16) / len(candidate_site)
    pos_rate = np.array(
        [
            candidate_y[candidate_site == s].mean()
            if (candidate_site == s).any()
            else np.nan
            for s in range(16)
        ]
    )

    for ctx in args.contexts:
        fit = mod.nested_balanced_indices(
            candidate_index, candidate_y, [ctx], seed=args.seed
        )[ctx]
        fpos = order[np.searchsorted(row_id, fit, sorter=order)]
        fsite = site[fpos]
        fbuild = building[fpos]
        share = np.bincount(fsite, minlength=16) / len(fsite)
        print(f"=== context {ctx:,} ===")
        print(
            f"  buildings covered: {len(np.unique(fbuild))} of "
            f"{len(np.unique(building[train_mask]))} in the training half"
        )
        print(
            f"  {'site':>4} {'train share':>12} {'context share':>14} {'ratio':>7} {'site pos-rate':>14}"
        )
        for s in range(16):
            if train_site_share[s] == 0:
                continue
            ratio = (
                share[s] / train_site_share[s] if train_site_share[s] else float("nan")
            )
            flag = (
                "  <-- over" if ratio >= 2 else ("  <-- under" if ratio <= 0.5 else "")
            )
            print(
                f"  {s:>4} {train_site_share[s]:>11.3%} {share[s]:>13.3%} "
                f"{ratio:>7.2f} {pos_rate[s]:>13.3%}{flag}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
